#!/usr/bin/env python3
"""
Evaluate ChannelNet V6 (8 channels) per cancer type.
Compares against V2 (6 channels) and Cox-Sage to isolate the
contribution of the epigenetic meta-channels.

Usage:
    python3 -u -m gnn.scripts.per_type_eval_v6
"""

import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v6 import (
    build_channel_features_v6, ChannelDatasetV6, CHANNEL_FEAT_DIM_V6, N_CHANNELS_V6,
)
from gnn.models.channel_net_v6 import ChannelNetV6
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
    "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
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

# Cox-Sage reported results
COX_SAGE = {
    "Breast Cancer": 0.669,
    "Bladder Cancer": 0.627,
    "Colorectal Cancer": 0.646,
    "Glioblastoma": 0.675,
    "Non-Small Cell Lung Cancer": 0.630,
    "Stomach Cancer": 0.632,
    "Endometrial Cancer": 0.716,
}

# V2 per-type results (from per_type_eval.py)
V2_RESULTS = None
_v2_path = os.path.join(GNN_RESULTS, "channelnet_v2", "per_type_results.json")
if os.path.exists(_v2_path):
    with open(_v2_path) as f:
        _v2_list = json.load(f)
    V2_RESULTS = {r["cancer_type"]: r["c_index"] for r in _v2_list}


def main():
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6_7ch")

    print("Loading V6 data...")
    data_dict = build_channel_features_v6("msk_impact_50k")
    N = len(data_dict["times"])
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    vocab = data_dict["cancer_type_vocab"]

    print(f"  {N} patients, {n_ct} cancer types, {N_CHANNELS_V6} channels")

    # Recreate folds
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

        model = ChannelNetV6(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        val_ds = ChannelDatasetV6(data_dict, indices=val_idx)
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

    # Per-type C-index
    ct_indices = data_dict["cancer_type_idx"].numpy()
    times = data_dict["times"]
    events = data_dict["events"]

    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE C-INDEX: V6 (8ch) vs V2 (6ch) vs Cox-Sage")
    print(f"{'='*90}")
    print(f"  {'Cancer Type':<30} {'N':>6} {'Events':>7} {'V6(8ch)':>8} {'V2(6ch)':>8} {'Delta':>7} {'CoxSage':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

    results = []
    total_improved = 0
    total_compared = 0

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

        v2_ci = V2_RESULTS.get(ct_name) if V2_RESULTS else None
        cox_sage = COX_SAGE.get(ct_name)

        v2_str = f"{v2_ci:.4f}" if v2_ci else "   —"
        delta = ci - v2_ci if v2_ci else None
        delta_str = f"{delta:>+7.4f}" if delta else "    —"
        cox_str = f"{cox_sage:.3f}" if cox_sage else "   —"

        flag = ""
        if v2_ci and delta > 0.005:
            flag = " ↑"
            total_improved += 1
        elif v2_ci and delta < -0.005:
            flag = " ↓"
        if v2_ci:
            total_compared += 1

        print(f"  {ct_name:<30} {n_patients:>6} {n_events:>7} {ci:>8.4f} {v2_str:>8} {delta_str} {cox_str:>8}{flag}")
        results.append({
            "cancer_type": ct_name,
            "n_patients": int(n_patients),
            "n_events": n_events,
            "v6_c_index": ci,
            "v2_c_index": v2_ci,
            "delta": delta,
            "cox_sage": cox_sage,
        })

    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

    # Overall
    mask = all_in_val.numpy()
    overall_ci = concordance_index(all_hazards[mask], times[mask], events[mask])
    v2_overall = None
    if V2_RESULTS:
        # Recompute from V2 per_type_eval would need V2 model — use saved overall
        v2_overall = 0.6760  # Known from training
    v2_o_str = f"{v2_overall:.4f}" if v2_overall else "   —"
    delta_o = overall_ci - v2_overall if v2_overall else None
    delta_o_str = f"{delta_o:>+7.4f}" if delta_o else "    —"
    print(f"  {'OVERALL':<30} {mask.sum():>6} {int(events[mask].sum()):>7} {overall_ci:>8.4f} {v2_o_str:>8} {delta_o_str}")

    if total_compared > 0:
        print(f"\n  {total_improved}/{total_compared} cancer types improved by >0.005")

    # Which types improved most? (epigenetic channel impact)
    if any(r["delta"] for r in results if r["delta"] is not None):
        print(f"\n  Top improvers (V6 vs V2):")
        sorted_by_delta = sorted(
            [r for r in results if r["delta"] is not None],
            key=lambda r: -r["delta"]
        )
        for r in sorted_by_delta[:10]:
            print(f"    {r['cancer_type']:<30} {r['delta']:>+.4f}  (V6={r['v6_c_index']:.4f})")

        print(f"\n  Largest drops:")
        for r in sorted_by_delta[-5:]:
            if r["delta"] < 0:
                print(f"    {r['cancer_type']:<30} {r['delta']:>+.4f}  (V6={r['v6_c_index']:.4f})")

    print(f"\n{'='*90}")

    # Save
    out_path = os.path.join(model_base, "per_type_results_v6.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
