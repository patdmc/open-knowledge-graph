#!/usr/bin/env python3
"""
Extract cross-channel attention weights per cancer type.
Produces heatmaps showing how tier importance shifts across cancers.

Usage:
    python3 -u -m gnn.scripts.attention_per_type
"""

import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v2 import build_channel_features_v2, ChannelDatasetV2, CHANNEL_FEAT_DIM
from gnn.models.channel_net_v2 import ChannelNetV2
from gnn.config import GNN_RESULTS, CHANNEL_NAMES

TIER_MAP = {
    "PI3K_Growth": 0, "CellCycle": 0,
    "DDR": 1, "TissueArch": 1,
    "Endocrine": 2, "Immune": 2,
}
TIER_NAMES = ["cell_intrinsic", "tissue_level", "organism_level"]
TIER_COLORS = {"cell_intrinsic": "C0", "tissue_level": "C1", "organism_level": "C2"}

TARGET_TYPES = [
    "Non-Small Cell Lung Cancer",
    "Breast Cancer",
    "Prostate Cancer",
    "Thyroid Cancer",
    "Glioma",
]


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


def extract_attention_for_patients(model, loader):
    """Extract attention from cross-channel transformer for a set of patients."""
    model.eval()
    all_attn = []

    with torch.no_grad():
        for batch in loader:
            ch_feats = batch["channel_features"]
            B = ch_feats.shape[0]

            h_ch = model.ch_encoder(ch_feats)
            pos = model.ch_transformer.channel_pos.weight.unsqueeze(0).expand(B, -1, -1)
            x = h_ch + pos

            # Extract attention from each transformer layer
            layer_attns = []
            for layer in model.ch_transformer.transformer.layers:
                mha = layer.self_attn
                attn_out, attn_w = mha(x, x, x, need_weights=True, average_attn_weights=False)
                layer_attns.append(attn_w.cpu())  # (B, n_heads, 6, 6)

                x2 = layer.norm1(x + attn_out)
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
                x = layer.norm2(x2 + layer.dropout2(ff))

            # Average across layers, keep per-head
            stacked = torch.stack(layer_attns, dim=0)  # (n_layers, B, n_heads, 6, 6)
            mean_layers = stacked.mean(dim=0)  # (B, n_heads, 6, 6)
            all_attn.append(mean_layers)

    all_attn = torch.cat(all_attn, dim=0)  # (N_patients, n_heads, 6, 6)
    return all_attn


def render_heatmap_ascii(matrix, row_labels, col_labels, title=""):
    """Render a 6x6 attention matrix as formatted text."""
    lines = []
    if title:
        lines.append(f"\n  {title}")
        lines.append(f"  {'':>14}" + "".join(f"  {c[:8]:>8}" for c in col_labels))

    for i, rl in enumerate(row_labels):
        row = f"  {rl:>14}"
        for j in range(len(col_labels)):
            val = matrix[i, j]
            # Highlight high values
            if val >= 0.20:
                row += f"  {val:>7.4f}*"
            else:
                row += f"  {val:>8.4f}"
        lines.append(row)
    return "\n".join(lines)


def compute_tier_summary(attn_matrix):
    """Compute tier-level summary from 6x6 attention matrix."""
    tier_attn = np.zeros((3, 3))
    tier_count = np.zeros((3, 3))

    for i in range(6):
        for j in range(6):
            ti = TIER_MAP[CHANNEL_NAMES[i]]
            tj = TIER_MAP[CHANNEL_NAMES[j]]
            tier_attn[ti, tj] += attn_matrix[i, j]
            tier_count[ti, tj] += 1

    tier_count = np.maximum(tier_count, 1)
    tier_avg = tier_attn / tier_count

    # Dominant channel per cancer type
    row_means = attn_matrix.mean(axis=1)  # how much each channel receives attention
    col_means = attn_matrix.mean(axis=0)  # how much each channel sends attention

    return tier_avg, row_means, col_means


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

    print("=" * 75)
    print("  PER-CANCER-TYPE ATTENTION HEATMAPS")
    print("  Do tier importance patterns shift between cancer types?")
    print("=" * 75)

    data_dict = build_channel_features_v2("msk_impact_50k")
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    vocab = data_dict["cancer_type_vocab"]
    ct_indices = data_dict["cancer_type_idx"]

    # Map target types to vocab indices
    target_ct_map = {}
    for ct_name in TARGET_TYPES:
        if ct_name in vocab:
            target_ct_map[ct_name] = vocab.index(ct_name)
        else:
            print(f"  WARNING: {ct_name} not in vocab")

    # Load models from all folds and average attention
    all_results = {}

    for fold_idx in range(5):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue

        model = ChannelNetV2(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        print(f"\n  Extracting from fold {fold_idx + 1}...")

        for ct_name, ct_idx in target_ct_map.items():
            # Get patients of this type
            mask = (ct_indices == ct_idx)
            patient_indices = torch.where(mask)[0].tolist()

            if len(patient_indices) < 50:
                continue

            ds = ChannelDatasetV2(data_dict, indices=patient_indices)
            loader = DataLoader(ds, batch_size=CONFIG["batch_size"],
                                shuffle=False, collate_fn=collate_fn)

            attn = extract_attention_for_patients(model, loader)
            # Average across patients and heads -> (6, 6)
            mean_attn = attn.mean(dim=(0, 1)).numpy()

            if ct_name not in all_results:
                all_results[ct_name] = []
            all_results[ct_name].append(mean_attn)

    # Average across folds
    print("\n" + "=" * 75)

    type_summaries = {}

    for ct_name in TARGET_TYPES:
        if ct_name not in all_results:
            continue

        mean_attn = np.mean(all_results[ct_name], axis=0)

        # Print heatmap
        print(render_heatmap_ascii(mean_attn, CHANNEL_NAMES, CHANNEL_NAMES,
                                    title=f"{ct_name} — Cross-Channel Attention"))

        # Tier summary
        tier_avg, row_means, col_means = compute_tier_summary(mean_attn)

        print(f"\n  Tier-level attention (averaged):")
        print(f"  {'':>20}" + "".join(f"  {t[:12]:>12}" for t in TIER_NAMES))
        for i, tn in enumerate(TIER_NAMES):
            row = f"  {tn:>20}"
            for j in range(3):
                row += f"  {tier_avg[i, j]:>12.4f}"
            print(row)

        # Which channels are most attended TO (column means)?
        ranked = sorted(range(6), key=lambda i: -col_means[i])
        print(f"\n  Channel importance (attention received = column means):")
        for rank, i in enumerate(ranked):
            tier = TIER_NAMES[TIER_MAP[CHANNEL_NAMES[i]]]
            bar = "█" * int(col_means[i] * 100)
            print(f"    {rank+1}. {CHANNEL_NAMES[i]:<14} {col_means[i]:.4f}  [{tier}]  {bar}")

        # Within-tier vs cross-tier
        within, cross = [], []
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                if TIER_MAP[CHANNEL_NAMES[i]] == TIER_MAP[CHANNEL_NAMES[j]]:
                    within.append(mean_attn[i, j])
                else:
                    cross.append(mean_attn[i, j])

        ratio = np.mean(within) / np.mean(cross) if np.mean(cross) > 0 else 0
        print(f"\n  Within-tier / cross-tier ratio: {ratio:.3f}x")

        type_summaries[ct_name] = {
            "attention_matrix": mean_attn.tolist(),
            "tier_summary": tier_avg.tolist(),
            "channel_importance": {CHANNEL_NAMES[i]: float(col_means[i]) for i in range(6)},
            "within_cross_ratio": float(ratio),
        }

        print(f"  {'-' * 70}")

    # Cross-type comparison: which channels shift most?
    print(f"\n{'=' * 75}")
    print(f"  CROSS-TYPE COMPARISON: Channel Importance Ranking")
    print(f"{'=' * 75}")
    print(f"  {'Channel':<14}" + "".join(f"  {t[:12]:>12}" for t in TARGET_TYPES))
    print(f"  {'-'*14}" + "  ".join(["-" * 12] * len(TARGET_TYPES)))

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        row = f"  {ch_name:<14}"
        for ct_name in TARGET_TYPES:
            if ct_name in type_summaries:
                imp = type_summaries[ct_name]["channel_importance"][ch_name]
                row += f"  {imp:>12.4f}"
            else:
                row += f"  {'—':>12}"
        tier = TIER_NAMES[TIER_MAP[ch_name]]
        row += f"  [{tier}]"
        print(row)

    # Variance across types — which channels shift most?
    print(f"\n  Channel variance across cancer types (higher = more type-specific):")
    variances = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        vals = [type_summaries[ct]["channel_importance"][ch_name]
                for ct in TARGET_TYPES if ct in type_summaries]
        v = np.var(vals)
        variances.append((ch_name, v, np.mean(vals)))

    variances.sort(key=lambda x: -x[1])
    for ch_name, var, mean in variances:
        tier = TIER_NAMES[TIER_MAP[ch_name]]
        bar = "█" * int(var * 10000)
        print(f"    {ch_name:<14} var={var:.6f}  mean={mean:.4f}  [{tier}]  {bar}")

    print(f"{'=' * 75}")

    # Save
    out_path = os.path.join(model_base, "attention_per_type.json")
    with open(out_path, "w") as f:
        json.dump(type_summaries, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
