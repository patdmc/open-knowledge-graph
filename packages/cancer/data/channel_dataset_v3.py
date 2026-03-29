"""
Enhanced channel dataset v3 with escalation chain features.

Adds to v2:
  - Escalation chain encoding: 2 biological cascade pathways
    Chain A: PI3K_Growth → DDR → Endocrine (growth → repair → systemic)
    Chain B: CellCycle → TissueArch → Immune (replication → structure → evasion)
  - Per-chain features: progression depth, mutation load gradient, hub involvement
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import CHANNEL_TO_IDX, GNN_CACHE
from .channel_dataset_v2 import build_channel_features_v2, CHANNEL_FEAT_DIM

# Two escalation chains — each is an ordered sequence of channels
# representing the biological cascade from cell-intrinsic to organism-level
ESCALATION_CHAINS = {
    "growth_repair_systemic": ["PI3K_Growth", "DDR", "Endocrine"],
    "replication_structure_evasion": ["CellCycle", "TissueArch", "Immune"],
}
CHAIN_NAMES = list(ESCALATION_CHAINS.keys())
N_CHAINS = len(CHAIN_NAMES)
CHAIN_FEAT_DIM = 10  # features per chain


def build_escalation_features(channel_features):
    """Build escalation chain features from channel-level features.

    Args:
        channel_features: (N, 6, 17) channel feature tensor from v2

    Returns:
        chain_features: (N, 2, CHAIN_FEAT_DIM) escalation chain features
    """
    N = channel_features.shape[0]
    chain_features = torch.zeros(N, N_CHAINS, CHAIN_FEAT_DIM)

    for chain_idx, (chain_name, channels) in enumerate(ESCALATION_CHAINS.items()):
        c_indices = [CHANNEL_TO_IDX[ch] for ch in channels]

        # Extract is_severed for each step in the chain
        severed = torch.stack([channel_features[:, ci, 0] for ci in c_indices])  # (3, N)

        # [0] Progression depth: how many steps are severed (0-3)
        chain_features[:, chain_idx, 0] = severed.sum(dim=0)

        # [1] Normalized progression (0-1)
        chain_features[:, chain_idx, 1] = severed.sum(dim=0) / len(channels)

        # [2] Sequential coherence: are severed channels contiguous from step 0?
        # e.g., [1,1,0] = 1.0 (coherent), [1,0,1] = 0.5 (gap), [0,0,1] = 0.0 (inverted)
        for i in range(len(channels)):
            # Check if steps 0..i are all severed
            prefix_severed = severed[:i+1].min(dim=0).values  # 1 if all steps 0..i severed
            chain_features[:, chain_idx, 2] += prefix_severed
        chain_features[:, chain_idx, 2] /= len(channels)

        # [3] Inversion score: later channels severed without earlier ones
        # High inversion = unusual biology (organism-level severed before cell-intrinsic)
        inversion = torch.zeros(N)
        for i in range(1, len(channels)):
            # Step i severed but step i-1 not
            inversion += severed[i] * (1 - severed[i-1])
        chain_features[:, chain_idx, 3] = inversion / (len(channels) - 1)

        # [4] Mutation load gradient along chain
        # Positive = more mutations at early steps (expected), negative = escalating
        mut_load = torch.stack([channel_features[:, ci, 1] for ci in c_indices])  # log(n_mut)
        if len(channels) >= 2:
            gradient = mut_load[0] - mut_load[-1]  # early - late
            chain_features[:, chain_idx, 4] = gradient

        # [5] Total mutation load across chain
        chain_features[:, chain_idx, 5] = mut_load.sum(dim=0)

        # [6] Hub involvement: any hub gene mutated in chain?
        hub_any = torch.stack([channel_features[:, ci, 3] for ci in c_indices])
        chain_features[:, chain_idx, 6] = hub_any.max(dim=0).values

        # [7] Hub count across chain steps
        chain_features[:, chain_idx, 7] = hub_any.sum(dim=0)

        # [8] Max VAF across chain (clonal dominance signal)
        max_vafs = torch.stack([channel_features[:, ci, 8] for ci in c_indices])
        chain_features[:, chain_idx, 8] = max_vafs.max(dim=0).values

        # [9] VAF gradient (early vs late — clonal evolution direction)
        if len(channels) >= 2:
            chain_features[:, chain_idx, 9] = max_vafs[0] - max_vafs[-1]

    return chain_features


def build_channel_features_v3(study_id="msk_impact_50k"):
    """Build v3 features: v2 + escalation chains.

    Returns dict with all v2 keys plus:
        chain_features: (N, 2, 10) escalation chain features
    """
    cache_path = os.path.join(GNN_CACHE, f"channel_features_v3_{study_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    # Build on top of v2
    data = build_channel_features_v2(study_id)

    print("Building escalation chain features...")
    chain_feats = build_escalation_features(data["channel_features"])

    data["chain_features"] = chain_feats
    torch.save(data, cache_path)
    print(f"  {N_CHAINS} chains, {CHAIN_FEAT_DIM} features each, cached.")
    return data


class ChannelDatasetV3(Dataset):
    """Dataset with channels, tiers, clinical covariates, and escalation chains."""

    def __init__(self, data_dict, indices=None):
        self.indices = indices
        self.data = data_dict

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.data["times"])

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        return {
            "channel_features": self.data["channel_features"][idx],
            "tier_features": self.data["tier_features"][idx],
            "chain_features": self.data["chain_features"][idx],
            "cancer_type_idx": self.data["cancer_type_idx"][idx],
            "age": self.data["age"][idx],
            "sex": self.data["sex"][idx],
            "time": self.data["times"][idx],
            "event": self.data["events"][idx],
        }
