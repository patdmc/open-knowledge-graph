#!/usr/bin/env python3
"""
Evaluate ChannelNet v2 per cancer type.
Loads best models from each fold, runs inference, computes C-index per type.

Usage:
    python3 -u -m gnn.scripts.per_type_eval
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
from gnn.training.metrics import concordance_index
from gnn.config import GNN_RESULTS


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

# Cox-Sage reported results for comparison
COX_SAGE = {
    "Breast Cancer": 0.669,
    "Bladder Cancer": 0.627,
    "Colorectal Cancer": 0.646,
    "Glioblastoma": 0.675,
    "Non-Small Cell Lung Cancer": 0.630,  # LUAD equivalent
    "Stomach Cancer": 0.632,  # STAD
    "Endometrial Cancer": 0.716,  # UCEC
}

# Map our cancer type names to Cox-Sage names (approximate)
NAME_MAP = {
    "Breast Cancer": "BRCA",
    "Non-Small Cell Lung Cancer": "LUAD",
    "Colorectal Cancer": "COADREAD",
    "Bladder Cancer": "BLCA",
    "Glioblastoma": "GBM",
    "Endometrial Cancer": "UCEC",
    "Stomach Cancer": "STAD",
    "Esophagogastric Cancer": "STAD",
}


def main():
    model_base = os.path.join(GNN_RESULTS, "channelnet_v2")

    print("Loading data...")
    data_dict = build_channel_features_v2("msk_impact_50k")
    N = len(data_dict["times"])
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    vocab = data_dict["cancer_type_vocab"]

    print(f"  {N} patients, {n_ct} cancer types")

    # Recreate the same folds
    event_arr = data_dict["events"].numpy().astype(int)
    ch_count = (data_dict["channel_features"][:, :, 0] > 0).sum(dim=1).numpy().clip(max=3)
    strat = event_arr * 4 + ch_count

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["random_seed"])

    # Collect per-patient predictions across folds (each patient is in val once)
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), strat)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            print(f"  Fold {fold_idx} model not found, skipping")
            continue

        model = ChannelNetV2(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        val_ds = ChannelDatasetV2(data_dict, indices=val_idx)
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

    # Now compute per-type C-index
    ct_indices = data_dict["cancer_type_idx"].numpy()
    times = data_dict["times"]
    events = data_dict["events"]

    print(f"\n{'='*75}")
    print(f"  PER-CANCER-TYPE C-INDEX (ChannelNet V2, pan-cancer model)")
    print(f"{'='*75}")
    print(f"  {'Cancer Type':<35} {'N':>6} {'Events':>7} {'C-index':>8}  {'Cox-Sage':>9}")
    print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*8}  {'-'*9}")

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

        cox_sage = COX_SAGE.get(ct_name, None)
        cox_str = f"{cox_sage:.3f}" if cox_sage else "  —"
        flag = ""
        if cox_sage:
            if ci > cox_sage + 0.01:
                flag = " <<"
            elif ci < cox_sage - 0.01:
                flag = ""

        print(f"  {ct_name:<35} {n_patients:>6} {n_events:>7} {ci:>8.4f}  {cox_str:>9}{flag}")
        results.append({
            "cancer_type": ct_name,
            "n_patients": int(n_patients),
            "n_events": n_events,
            "c_index": ci,
            "cox_sage": cox_sage,
        })

    print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*8}  {'-'*9}")

    # Overall
    mask = all_in_val.numpy()
    overall_ci = concordance_index(all_hazards[mask], times[mask], events[mask])
    print(f"  {'OVERALL (pan-cancer)':<35} {mask.sum():>6} {int(events[mask].sum()):>7} {overall_ci:>8.4f}")

    print(f"\n  << = beats Cox-Sage (which uses RNA-seq, per-type training)")
    print(f"{'='*75}")

    # Save
    out_path = os.path.join(model_base, "per_type_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
