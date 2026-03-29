#!/usr/bin/env python3
"""
Extract cross-channel attention weights from trained ChannelNet V2.
Tests whether the model independently discovers the tier hierarchy.

Hypothesis: channels within the same tier (PI3K↔CellCycle, DDR↔TissueArch,
Endocrine↔Immune) should have higher mutual attention than cross-tier pairs.

Usage:
    python3 -u -m gnn.scripts.extract_attention
"""

import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v2 import build_channel_features_v2, ChannelDatasetV2, CHANNEL_FEAT_DIM
from gnn.models.channel_net_v2 import ChannelNetV2
from gnn.config import GNN_RESULTS, CHANNEL_NAMES

# Tier definitions for comparison
TIER_MAP = {
    "PI3K_Growth": 0, "CellCycle": 0,   # cell-intrinsic
    "DDR": 1, "TissueArch": 1,           # tissue-level
    "Endocrine": 2, "Immune": 2,         # organism-level
}
TIER_NAMES = ["cell_intrinsic", "tissue_level", "organism_level"]


def collate_fn(batch):
    return {
        "channel_features": torch.stack([b["channel_features"] for b in batch]),
        "tier_features": torch.stack([b["tier_features"] for b in batch]),
        "cancer_type_idx": torch.stack([b["cancer_type_idx"] for b in batch]),
        "age": torch.stack([b["age"] for b in batch]),
        "sex": torch.stack([b["sex"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
    }


def extract_attention_weights(model, loader, max_batches=50):
    """Extract attention weights from the cross-channel transformer.

    Hooks into the MultiheadAttention layer to capture attention patterns.

    Returns:
        attn_weights: (6, 6) mean attention matrix across all patients and heads
        per_head: (n_heads, 6, 6) mean attention per head
    """
    model.eval()

    # Find the MHA layer inside the channel transformer
    # CrossChannelAttention -> TransformerEncoder -> TransformerEncoderLayer -> MultiheadAttention
    mha_layers = []
    for name, module in model.ch_transformer.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            mha_layers.append((name, module))

    if not mha_layers:
        print("  No MultiheadAttention layers found!")
        return None, None

    # We'll capture attention from the last MHA layer
    captured_attn = []

    def hook_fn(module, input, output):
        # output is (attn_output, attn_weights) when need_weights=True
        # But TransformerEncoder doesn't pass need_weights by default
        # We need to call forward manually with need_weights=True
        pass

    # Alternative: manually compute attention from Q, K
    # The MHA stores in_proj_weight and in_proj_bias
    all_attn_matrices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            ch_feats = batch["channel_features"]  # (B, 6, 17)
            B = ch_feats.shape[0]

            # Reproduce the channel path up to the transformer
            h_ch = model.ch_encoder(ch_feats)  # (B, 6, H)

            # Add positional encoding
            pos = model.ch_transformer.channel_pos.weight.unsqueeze(0).expand(B, -1, -1)
            x = h_ch + pos  # (B, 6, H)

            # Manually run through each transformer layer and extract attention
            for layer in model.ch_transformer.transformer.layers:
                # Get the self-attention weights
                # TransformerEncoderLayer has self_attn (MultiheadAttention)
                mha = layer.self_attn

                # Call MHA with need_weights=True
                attn_out, attn_w = mha(x, x, x, need_weights=True, average_attn_weights=False)
                # attn_w: (B, n_heads, 6, 6)

                all_attn_matrices.append(attn_w.cpu())

                # Continue through the rest of the layer
                # (residual + norm + feedforward + residual + norm)
                x2 = layer.norm1(x + attn_out)
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
                x = layer.norm2(x2 + layer.dropout2(ff))

    if not all_attn_matrices:
        return None, None

    # Stack all attention matrices: (total_batches * n_layers, B, n_heads, 6, 6)
    # Average across batches and layers
    all_attn = torch.cat(all_attn_matrices, dim=0)  # (total_B, n_heads, 6, 6)
    mean_per_head = all_attn.mean(dim=0).numpy()  # (n_heads, 6, 6)
    mean_overall = mean_per_head.mean(axis=0)  # (6, 6)

    return mean_overall, mean_per_head


def analyze_attention(attn_matrix, channel_names):
    """Analyze whether attention follows tier structure."""

    print("\n  CROSS-CHANNEL ATTENTION MATRIX (mean across patients & heads)")
    print(f"  {'':>14}", end="")
    for name in channel_names:
        print(f"  {name[:8]:>8}", end="")
    print()

    for i, name_i in enumerate(channel_names):
        print(f"  {name_i:>14}", end="")
        for j in range(len(channel_names)):
            val = attn_matrix[i, j]
            print(f"  {val:>8.4f}", end="")
        print()

    # Compare within-tier vs cross-tier attention
    within_tier = []
    cross_tier = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            tier_i = TIER_MAP[channel_names[i]]
            tier_j = TIER_MAP[channel_names[j]]
            if tier_i == tier_j:
                within_tier.append(attn_matrix[i, j])
            else:
                cross_tier.append(attn_matrix[i, j])

    mean_within = np.mean(within_tier)
    mean_cross = np.mean(cross_tier)
    ratio = mean_within / mean_cross if mean_cross > 0 else float('inf')

    print(f"\n  TIER STRUCTURE DISCOVERY")
    print(f"  Within-tier attention:  {mean_within:.4f} (avg of {len(within_tier)} pairs)")
    print(f"  Cross-tier attention:   {mean_cross:.4f} (avg of {len(cross_tier)} pairs)")
    print(f"  Ratio (within/cross):   {ratio:.3f}x")

    if ratio > 1.05:
        print(f"  --> Model assigns {(ratio-1)*100:.1f}% MORE attention to within-tier pairs")
        print(f"  --> The transformer independently discovered the tier hierarchy!")
    elif ratio < 0.95:
        print(f"  --> Model assigns MORE attention to cross-tier pairs")
        print(f"  --> Cross-tier interactions are more informative for survival")
    else:
        print(f"  --> Roughly uniform attention across tiers")

    # Per-tier-pair breakdown
    print(f"\n  PER-TIER-PAIR ATTENTION")
    tier_pair_attn = defaultdict(list)
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            ti = TIER_MAP[channel_names[i]]
            tj = TIER_MAP[channel_names[j]]
            key = (min(ti, tj), max(ti, tj))
            if ti == tj:
                key = (ti, tj)
            tier_pair_attn[key].append(attn_matrix[i, j])

    for (ti, tj), vals in sorted(tier_pair_attn.items()):
        ti_name = TIER_NAMES[ti]
        tj_name = TIER_NAMES[tj]
        label = f"{ti_name} ↔ {tj_name}" if ti != tj else f"{ti_name} (within)"
        print(f"    {label:<40} {np.mean(vals):.4f}")

    # Specific channel pairs of interest
    print(f"\n  KEY CHANNEL PAIRS (biological significance)")
    pairs = [
        (2, 0, "PI3K_Growth ↔ DDR", "Growth deregulation + failed repair"),
        (0, 1, "DDR ↔ CellCycle", "Checkpoint coupling"),
        (3, 2, "Endocrine ↔ PI3K_Growth", "Hormone → PI3K/mTOR axis"),
        (4, 5, "Immune ↔ TissueArch", "Immune evasion + tissue breakdown"),
        (2, 1, "PI3K_Growth ↔ CellCycle", "Same tier: cell-intrinsic"),
        (0, 5, "DDR ↔ TissueArch", "Same tier: tissue-level"),
        (3, 4, "Endocrine ↔ Immune", "Same tier: organism-level"),
    ]
    for i, j, label, bio in pairs:
        val = (attn_matrix[i, j] + attn_matrix[j, i]) / 2
        print(f"    {label:<30} {val:.4f}  ({bio})")

    return {
        "mean_within_tier": float(mean_within),
        "mean_cross_tier": float(mean_cross),
        "ratio": float(ratio),
    }


def main():
    CONFIG = {
        "channel_feat_dim": CHANNEL_FEAT_DIM,
        "hidden_dim": 128,
        "cross_channel_heads": 4,
        "cross_channel_layers": 2,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 15,
        "batch_size": 512,
        "n_folds": 5,
        "random_seed": 42,
        "n_cancer_types": 80,
    }

    model_base = os.path.join(GNN_RESULTS, "channelnet_v2")

    print("=" * 70)
    print("  ATTENTION WEIGHT ANALYSIS — Does the model discover tiers?")
    print("=" * 70)

    data_dict = build_channel_features_v2("msk_impact_50k")
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct

    # Use all data for attention extraction (not just val)
    ds = ChannelDatasetV2(data_dict)
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"],
                        shuffle=False, collate_fn=collate_fn)

    # Extract from each fold's model and average
    all_attn = []
    all_per_head = []

    for fold_idx in range(5):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            print(f"  Fold {fold_idx} model not found, skipping")
            continue

        model = ChannelNetV2(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        print(f"  Extracting attention from fold {fold_idx+1}...")
        attn, per_head = extract_attention_weights(model, loader, max_batches=20)
        if attn is not None:
            all_attn.append(attn)
            all_per_head.append(per_head)

    if not all_attn:
        print("  No attention weights extracted!")
        return

    # Average across folds
    mean_attn = np.mean(all_attn, axis=0)
    mean_per_head = np.mean(all_per_head, axis=0)

    # Analyze
    results = analyze_attention(mean_attn, CHANNEL_NAMES)

    # Per-head analysis
    print(f"\n  PER-HEAD WITHIN/CROSS-TIER RATIO")
    n_heads = mean_per_head.shape[0]
    for h in range(n_heads):
        within, cross = [], []
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                if TIER_MAP[CHANNEL_NAMES[i]] == TIER_MAP[CHANNEL_NAMES[j]]:
                    within.append(mean_per_head[h, i, j])
                else:
                    cross.append(mean_per_head[h, i, j])
        ratio = np.mean(within) / np.mean(cross)
        print(f"    Head {h+1}: {ratio:.3f}x")

    print(f"{'='*70}")

    # Save
    out = {
        "attention_matrix": mean_attn.tolist(),
        "per_head_attention": mean_per_head.tolist(),
        "channel_names": CHANNEL_NAMES,
        "analysis": results,
    }
    out_path = os.path.join(model_base, "attention_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
