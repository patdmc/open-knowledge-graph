"""
Essentiality Head — pre-training objective for DepMap cell line data.

Predicts per-gene CRISPR knockout essentiality (Chronos scores) from the
backbone's per-node hidden representations. Negative Chronos = essential.

Loss combines:
  1. Masked MSE on genes with known essentiality scores
  2. Pairwise ranking loss: essential genes should score lower than non-essential
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EssentialityHead(nn.Module):
    """Predict per-node essentiality from backbone node representations.

    Takes node_hidden (B, N, hidden) and produces per-node scalar predictions.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_hidden):
        """
        Args:
            node_hidden: (B, N, hidden) from backbone encode()

        Returns:
            predictions: (B, N) essentiality score per node
        """
        return self.head(node_hidden).squeeze(-1)


class EssentialityLoss(nn.Module):
    """Combined masked MSE + pairwise ranking loss for essentiality prediction.

    MSE: regress directly to Chronos scores for nodes with valid targets.
    Ranking: essential genes (Chronos < threshold) should predict lower than
             non-essential genes within the same sample.
    """

    def __init__(self, rank_weight=0.3, essential_threshold=-0.5):
        super().__init__()
        self.rank_weight = rank_weight
        self.essential_threshold = essential_threshold

    def forward(self, predictions, targets, masks):
        """
        Args:
            predictions: (B, N) predicted essentiality
            targets: (B, N) Chronos scores (negative = essential)
            masks: (B, N) 1.0 where target is valid, 0.0 otherwise

        Returns:
            total_loss: scalar
            metrics: dict with component losses
        """
        # Masked MSE
        valid = masks.bool()
        if valid.sum() == 0:
            zero = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            return zero, {"mse": 0.0, "rank": 0.0, "n_valid": 0}

        pred_valid = predictions[valid]
        targ_valid = targets[valid]
        mse_loss = F.mse_loss(pred_valid, targ_valid)

        # Pairwise ranking loss within each sample
        rank_loss = torch.tensor(0.0, device=predictions.device)
        n_pairs = 0

        B, N = predictions.shape
        for b in range(B):
            m = masks[b].bool()
            if m.sum() < 2:
                continue

            p = predictions[b, m]
            t = targets[b, m]

            essential = t < self.essential_threshold
            non_essential = t >= self.essential_threshold

            if essential.sum() == 0 or non_essential.sum() == 0:
                continue

            # Essential genes should predict lower than non-essential
            p_ess = p[essential].unsqueeze(1)      # (n_ess, 1)
            p_non = p[non_essential].unsqueeze(0)   # (1, n_non)

            # Margin ranking: p_ess should be < p_non by margin
            margin = 0.5
            pair_loss = F.relu(p_ess - p_non + margin)
            rank_loss = rank_loss + pair_loss.mean()
            n_pairs += 1

        if n_pairs > 0:
            rank_loss = rank_loss / n_pairs

        total = mse_loss + self.rank_weight * rank_loss

        metrics = {
            "mse": mse_loss.item(),
            "rank": rank_loss.item() if isinstance(rank_loss, torch.Tensor) else rank_loss,
            "n_valid": valid.sum().item(),
        }

        return total, metrics
