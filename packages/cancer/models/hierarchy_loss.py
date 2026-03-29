"""
Functional hierarchy loss — teach the transformer to learn escalation depth.

The graph has a functional hierarchy within each channel: mutations closer
to the pathway root (hub) happened EARLIER in cancer progression. Earlier
mutations have higher VAF (variant allele frequency = clonal fraction).

Training signal: for mutation pairs (i, j) in the same channel, if the
transformer's learned representation places i as "shallower" (closer to root),
then i should have higher VAF than j. This is a pairwise ranking loss.

The transformer doesn't need explicit PPI distance — it learns functional
depth from the combination of graph structure (pairwise edge features) and
the VAF ordering signal. The PPI is one input to attention; the VAF
ordering is the supervision.

This loss is AUXILIARY to Cox loss. It teaches the model to understand
escalation structure, which should improve survival prediction as a
consequence (escalation depth predicts prognosis).

Usage:
    from gnn.models.hierarchy_loss import HierarchyLoss, build_hierarchy_pairs

    # Build pairs once from dataset
    pairs = build_hierarchy_pairs(dataset)

    # In training loop
    h_loss = HierarchyLoss(hidden_dim)
    depth_scores = h_loss.depth_head(transformer_embeddings)  # (B, N, 1)
    loss = h_loss(depth_scores, pairs_batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class HierarchyLoss(nn.Module):
    """Pairwise ranking loss on learned functional depth.

    The transformer produces per-mutation embeddings. A depth head projects
    each embedding to a scalar "depth score". The hierarchy loss encourages:
      - Mutations with higher VAF (earlier/more clonal) → lower depth score
      - Mutations with lower VAF (later/subclonal) → higher depth score

    Within the same channel only — cross-channel pairs are not comparable.

    Loss: margin ranking loss on (depth_i, depth_j) pairs where VAF_i > VAF_j
    implies depth_i < depth_j (i is shallower/earlier).
    """

    def __init__(self, hidden_dim, margin=0.1):
        super().__init__()
        # Depth head: mutation embedding → scalar depth
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.margin = margin

    def forward(self, mutation_embeddings, pair_indices, pair_targets):
        """
        Args:
            mutation_embeddings: (B, N, hidden) — transformer output
            pair_indices: (B, P, 2) long — indices into N for each pair
                          P = max pairs per patient (padded)
            pair_targets: (B, P) float — +1 if first is shallower (higher VAF),
                          -1 if second is shallower, 0 if pad

        Returns:
            loss: scalar
            stats: dict with concordance, n_pairs
        """
        B, N, H = mutation_embeddings.shape

        # Predict depth for all mutations
        depth = self.depth_head(mutation_embeddings).squeeze(-1)  # (B, N)

        # Gather depths for pairs
        # pair_indices: (B, P, 2)
        idx_i = pair_indices[:, :, 0]  # (B, P)
        idx_j = pair_indices[:, :, 1]  # (B, P)

        # Clamp indices to valid range
        idx_i = idx_i.clamp(0, N - 1)
        idx_j = idx_j.clamp(0, N - 1)

        depth_i = torch.gather(depth, 1, idx_i)  # (B, P)
        depth_j = torch.gather(depth, 1, idx_j)  # (B, P)

        # Valid pairs only (target != 0)
        valid = (pair_targets != 0)
        if not valid.any():
            return torch.tensor(0.0, device=mutation_embeddings.device), {
                'concordance': 0.5, 'n_pairs': 0}

        # Margin ranking loss:
        # target = +1 means depth_i should be LESS than depth_j (i is shallower)
        # target = -1 means depth_j should be LESS than depth_i
        # loss = max(0, -target * (depth_j - depth_i) + margin)
        diff = depth_j - depth_i  # positive if j is deeper
        loss_per_pair = F.relu(-pair_targets * diff + self.margin)
        loss = (loss_per_pair * valid.float()).sum() / valid.float().sum().clamp(min=1)

        # Concordance: fraction of pairs correctly ordered
        with torch.no_grad():
            predicted_order = torch.sign(diff)
            concordant = ((predicted_order == pair_targets) & valid).float().sum()
            total_valid = valid.float().sum().clamp(min=1)
            concordance = (concordant / total_valid).item()

        return loss, {
            'concordance': concordance,
            'n_pairs': int(valid.sum().item()),
        }


def build_hierarchy_pairs(gene_names_per_patient, vaf_per_patient,
                          channel_map, max_pairs_per_patient=64,
                          min_vaf_diff=0.05):
    """Build hierarchy training pairs from patient data.

    For each patient, find same-channel mutation pairs where VAFs differ
    meaningfully. The higher-VAF mutation is "shallower" (target = +1 for
    the pair (high_vaf_idx, low_vaf_idx)).

    Args:
        gene_names_per_patient: list of lists — gene names per mutation slot
            Shape conceptually: (N_patients, N_mutations)
        vaf_per_patient: np.array (N_patients, N_mutations) — VAF values
        channel_map: dict gene → channel name
        max_pairs_per_patient: int — cap on pairs (pad to this)
        min_vaf_diff: float — minimum VAF difference to count as ordered

    Returns:
        pair_indices: np.array (N, max_pairs, 2) int
        pair_targets: np.array (N, max_pairs) float — +1, -1, or 0 (pad)
    """
    N = len(gene_names_per_patient)
    pair_indices = np.zeros((N, max_pairs_per_patient, 2), dtype=np.int64)
    pair_targets = np.zeros((N, max_pairs_per_patient), dtype=np.float32)

    stats = {'total_pairs': 0, 'patients_with_pairs': 0}

    for patient_idx in range(N):
        genes = gene_names_per_patient[patient_idx]
        vafs = vaf_per_patient[patient_idx]

        # Group mutation indices by channel
        channel_slots = defaultdict(list)  # channel → [(slot_idx, vaf)]
        for slot_idx, gene in enumerate(genes):
            if gene and gene != 'WT' and gene in channel_map:
                v = vafs[slot_idx]
                if v > 0 and np.isfinite(v):
                    ch = channel_map[gene]
                    channel_slots[ch].append((slot_idx, float(v)))

        # Build pairs within each channel
        pairs = []
        for ch, slots in channel_slots.items():
            if len(slots) < 2:
                continue
            for ii in range(len(slots)):
                for jj in range(ii + 1, len(slots)):
                    idx_a, vaf_a = slots[ii]
                    idx_b, vaf_b = slots[jj]
                    diff = vaf_a - vaf_b

                    if abs(diff) < min_vaf_diff:
                        continue

                    if diff > 0:
                        # a has higher VAF → a is shallower
                        pairs.append((idx_a, idx_b, 1.0))
                    else:
                        # b has higher VAF → b is shallower
                        pairs.append((idx_a, idx_b, -1.0))

        if not pairs:
            continue

        stats['patients_with_pairs'] += 1

        # Subsample if too many
        if len(pairs) > max_pairs_per_patient:
            rng = np.random.RandomState(patient_idx)
            indices = rng.choice(len(pairs), max_pairs_per_patient, replace=False)
            pairs = [pairs[i] for i in indices]

        for p_idx, (idx_a, idx_b, target) in enumerate(pairs):
            pair_indices[patient_idx, p_idx] = [idx_a, idx_b]
            pair_targets[patient_idx, p_idx] = target
            stats['total_pairs'] += 1

    stats['mean_pairs'] = stats['total_pairs'] / max(stats['patients_with_pairs'], 1)
    return pair_indices, pair_targets, stats


class MultiObjectiveLoss(nn.Module):
    """Combines Cox loss + hierarchy loss with learned weighting.

    Uses uncertainty-based weighting (Kendall et al. 2018):
    Each loss gets a learned log-variance parameter. The total loss is
    weighted by 1/(2*sigma^2) * loss + log(sigma), which automatically
    balances losses of different scales.
    """

    def __init__(self, n_losses=2):
        super().__init__()
        # Learned log-variance for each loss (initialized to 0 = equal weight)
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses):
        """
        Args:
            losses: list of scalar tensors [cox_loss, hierarchy_loss, ...]

        Returns:
            total: weighted sum
            weights: list of effective weights (for logging)
        """
        total = torch.tensor(0.0, device=losses[0].device)
        weights = []

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
            weights.append(precision.item())

        return total, weights
