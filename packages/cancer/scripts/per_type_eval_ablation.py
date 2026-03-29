#!/usr/bin/env python3
"""
Per-type eval for ablation (V2 + MSI/TMB) vs V6c (8ch + MSI/TMB).
Shows whether 8 channels add value per cancer type.

Usage:
    python3 -u -m gnn.scripts.per_type_eval_ablation
"""

import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v2_msi import (
    build_channel_features_v2_msi, ChannelDatasetV2MSI, CHANNEL_FEAT_DIM,
)
from gnn.models.channel_net_v2_msi import ChannelNetV2MSI
from gnn.training.metrics import concordance_index
from gnn.config import GNN_RESULTS


def collate_fn(batch):
    return {
        "channel_features": torch.stack([b["channel_features"] for b in batch]),
        "tier_features": torch.stack([b["tier_features"] for b in batch]),
        "cancer_type_idx": torch.stack([b["cancer_type_idx"] for b in batch]),
        "age": torch.stack([b["age"] for b in batch]),
        "sex": torch.stack([b["sex"] for b in batch]),
        "msi_score": torch.stack([b["msi_score"] for b in batch]),
        "msi_high": torch.stack([b["msi_high"] for b in batch]),
        "tmb": torch.stack([b["tmb"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
    }


CONFIG = {
    "channel_feat_dim": CHANNEL_FEAT_DIM,
    "hidden_dim": 128,
    "cross_channel_heads": 4,
    "cross_channel_layers": 2,
    "dropout": 0.3,
    "batch_size": 512,
    "n_folds": 5,
    "random_seed": 42,
    "n_cancer_types": 80,
}

COX_SAGE = {
    "Breast Cancer": 0.669,
    "Bladder Cancer": 0.627,
    "Colorectal Cancer": 0.646,
    "Glioblastoma": 0.675,
    "Non-Small Cell Lung Cancer": 0.630,
    "Stomach Cancer": 0.632,
    "Endometrial Cancer": 0.716,
}

# Load V6c per-type results for comparison
V6C_RESULTS = None
_v6c_path = os.path.join(GNN_RESULTS, "channelnet_v6c_msi", "per_type_results.json")
if os.path.exists(_v6c_path):
    with open(_v6c_path) as f:
        _v6c_list = json.load(f)
    V6C_RESULTS = {r["cancer_type"]: r["v6c_c_index"] for r in _v6c_list}

# Load V2 per-type results
V2_RESULTS = None
_v2_path = os.path.join(GNN_RESULTS, "channelnet_v2", "per_type_results.json")
if os.path.exists(_v2_path):
    with open(_v2_path) as f:
        _v2_list = json.load(f)
    V2_RESULTS = {r["cancer_type"]: r["c_index"] for r in _v2_list}


def main():
    model_base = os.path.join(GNN_RESULTS, "ablation_v2_msi")

    print("Loading V2+MSI data...")
    data_dict = build_channel_features_v2_msi("msk_impact_50k")
    N = len(data_dict["times"])
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    vocab = data_dict["cancer_type_vocab"]

    print(f"  {N} patients, {n_ct} cancer types")

    event_arr = data_dict["events"].numpy().astype(int)
    ch_count = (data_dict["channel_features"][:, :, 0] > 0).sum(dim=1).numpy().clip(max=3)
    strat = event_arr * 4 + ch_count

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["random_seed"])

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), strat)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            print(f"  Fold {fold_idx} model not found, skipping")
            continue

        model = ChannelNetV2MSI(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        val_ds = ChannelDatasetV2MSI(data_dict, indices=val_idx)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                                shuffle=False, collate_fn=collate_fn)

        hazards = []
        with torch.no_grad():
            for batch in val_loader:
                h = model(batch)
                hazards.append(h)
        hazards = torch.cat(hazards)

        for i, vi in enumerate(val_idx):
            all_hazards[vi] = hazards[i]
            all_in_val[vi] = True

        print(f"  Fold {fold_idx+1}: {len(val_idx)} patients evaluated")

    ct_indices = data_dict["cancer_type_idx"].numpy()
    times = data_dict["times"]
    events = data_dict["events"]

    print(f"\n{'='*100}")
    print(f"  ABLATION: V2+MSI/TMB (6ch) vs V6c (8ch+MSI/TMB) vs V2 (6ch)")
    print(f"{'='*100}")
    print(f"  {'Cancer Type':<30} {'N':>6} {'V2+MSI':>8} {'V6c(8ch)':>8} {'V2(6ch)':>8} {'8ch adds':>8} {'CoxSage':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    results = []

    for ct_idx, ct_name in enumerate(vocab):
        mask = (ct_indices == ct_idx) & all_in_val.numpy()
        n_patients = mask.sum()
        if n_patients < 50:
            continue

        h = all_hazards[mask]
        t = times[mask]
        e = events[mask]
        n_events = int(e.sum())

        if n_events < 10:
            continue

        ci = concordance_index(h, t, e)

        v6c_ci = V6C_RESULTS.get(ct_name) if V6C_RESULTS else None
        v2_ci = V2_RESULTS.get(ct_name) if V2_RESULTS else None
        cox_sage = COX_SAGE.get(ct_name)

        v6c_str = f"{v6c_ci:.4f}" if v6c_ci else "   ---"
        v2_str = f"{v2_ci:.4f}" if v2_ci else "   ---"
        cox_str = f"{cox_sage:.3f}" if cox_sage else "   ---"

        # 8ch contribution = V6c - V2+MSI
        ch8_delta = v6c_ci - ci if v6c_ci else None
        ch8_str = f"{ch8_delta:>+8.4f}" if ch8_delta else "     ---"

        print(f"  {ct_name:<30} {n_patients:>6} {ci:>8.4f} {v6c_str:>8} {v2_str:>8} {ch8_str} {cox_str:>8}")
        results.append({
            "cancer_type": ct_name,
            "n_patients": int(n_patients),
            "n_events": n_events,
            "v2_msi_c_index": ci,
            "v6c_c_index": v6c_ci,
            "v2_c_index": v2_ci,
            "ch8_delta": ch8_delta,
            "cox_sage": cox_sage,
        })

    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Overall
    mask = all_in_val.numpy()
    overall_ci = concordance_index(all_hazards[mask], times[mask], events[mask])
    msi_gain = overall_ci - 0.6760
    ch8_gain = 0.686 - overall_ci  # approximate
    print(f"  {'OVERALL':<30} {mask.sum():>6} {overall_ci:>8.4f} {'~0.686':>8} {'0.6760':>8} {ch8_gain:>+8.4f}")

    print(f"\n  Signal decomposition:")
    print(f"    V2 baseline:          0.6760")
    print(f"    + MSI/TMB:            {overall_ci:.4f} ({msi_gain:+.4f})")
    print(f"    + 8 channels:         ~0.686  ({ch8_gain:+.4f})")
    print(f"{'='*100}")

    out_path = os.path.join(model_base, "per_type_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
