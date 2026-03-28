#!/usr/bin/env python3
"""
Unified ablation: all variants on the SAME 5 folds for clean comparison.

Variants:
  V2      — 6 channels, no MSI/TMB
  V2+MSI  — 6 channels + MSI/TMB clinical
  V6      — 8 channels, no MSI/TMB
  V6c     — 8 channels + MSI/TMB clinical

All use: hidden=128, layers=2, batch=512, lr=5e-4, patience=15, seed=42.
Stratification: event-only (same for all variants).

Usage:
    python3 -u -m gnn.scripts.unified_ablation
    python3 -u -m gnn.scripts.unified_ablation --resume
"""

import sys
import os
import time
import json
import copy
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v2 import (
    build_channel_features_v2, ChannelDatasetV2, CHANNEL_FEAT_DIM,
)
from gnn.data.channel_dataset_v2_msi import (
    build_channel_features_v2_msi, ChannelDatasetV2MSI,
    CHANNEL_FEAT_DIM as CHANNEL_FEAT_DIM_V2MSI,
)
from gnn.data.channel_dataset_v6 import (
    build_channel_features_v6, ChannelDatasetV6,
    CHANNEL_FEAT_DIM_V6, N_CHANNELS_V6, N_TIERS_V6,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c, ChannelDatasetV6c

from gnn.models.channel_net_v2 import ChannelNetV2
from gnn.models.channel_net_v2_msi import ChannelNetV2MSI
from gnn.models.channel_net_v6 import ChannelNetV6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index
from gnn.config import GNN_RESULTS

SAVE_BASE = os.path.join(GNN_RESULTS, "unified_ablation")
CHECKPOINT_PATH = os.path.join(SAVE_BASE, "checkpoint.json")

SHARED_CONFIG = {
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

# Variant definitions
VARIANTS = {
    "V2": {
        "config_extra": {"channel_feat_dim": CHANNEL_FEAT_DIM},
        "model_cls": ChannelNetV2,
        "dataset_cls": ChannelDatasetV2,
        "build_fn": build_channel_features_v2,
        "collate_keys": [
            "channel_features", "tier_features", "cancer_type_idx",
            "age", "sex", "time", "event",
        ],
    },
    "V2_MSI": {
        "config_extra": {"channel_feat_dim": CHANNEL_FEAT_DIM_V2MSI},
        "model_cls": ChannelNetV2MSI,
        "dataset_cls": ChannelDatasetV2MSI,
        "build_fn": build_channel_features_v2_msi,
        "collate_keys": [
            "channel_features", "tier_features", "cancer_type_idx",
            "age", "sex", "msi_score", "msi_high", "tmb", "time", "event",
        ],
    },
    "V6": {
        "config_extra": {"channel_feat_dim": CHANNEL_FEAT_DIM_V6},
        "model_cls": ChannelNetV6,
        "dataset_cls": ChannelDatasetV6,
        "build_fn": build_channel_features_v6,
        "collate_keys": [
            "channel_features", "tier_features", "cancer_type_idx",
            "age", "sex", "time", "event",
        ],
    },
    "V6c": {
        "config_extra": {"channel_feat_dim": CHANNEL_FEAT_DIM_V6},
        "model_cls": ChannelNetV6c,
        "dataset_cls": ChannelDatasetV6c,
        "build_fn": build_channel_features_v6c,
        "collate_keys": [
            "channel_features", "tier_features", "cancer_type_idx",
            "age", "sex", "msi_score", "msi_high", "tmb", "time", "event",
        ],
    },
}

VARIANT_ORDER = ["V2", "V2_MSI", "V6", "V6c"]


def make_collate_fn(keys):
    def collate_fn(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in keys}
    return collate_fn


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        optimizer.zero_grad()
        hazard = model(batch)
        loss = loss_fn(hazard, batch["time"], batch["event"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    all_h, all_t, all_e = [], [], []
    total_loss, n = 0, 0
    for batch in loader:
        hazard = model(batch)
        loss = loss_fn(hazard, batch["time"], batch["event"])
        total_loss += loss.item()
        n += 1
        all_h.append(hazard)
        all_t.append(batch["time"])
        all_e.append(batch["event"])

    all_h = torch.cat(all_h)
    all_t = torch.cat(all_t)
    all_e = torch.cat(all_e)
    return {
        "loss": total_loss / max(n, 1),
        "c_index": concordance_index(all_h, all_t, all_e),
    }


def run_variant_fold(variant_name, fold_idx, train_idx, val_idx, data_dict, n_ct):
    """Train one variant on one fold, return C-index."""
    vdef = VARIANTS[variant_name]
    config = {**SHARED_CONFIG, **vdef["config_extra"], "n_cancer_types": n_ct}
    collate_fn = make_collate_fn(vdef["collate_keys"])

    train_ds = vdef["dataset_cls"](data_dict, indices=train_idx)
    val_ds = vdef["dataset_cls"](data_dict, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = vdef["model_cls"](config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )
    loss_fn = CoxPartialLikelihoodLoss()

    best_ci, best_state, patience_ctr = 0.0, None, 0
    fold_dir = os.path.join(SAVE_BASE, variant_name, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_m = evaluate(model, val_loader, loss_fn)
        scheduler.step(val_m["c_index"])

        if val_m["c_index"] > best_ci:
            best_ci = val_m["c_index"]
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"      Ep {epoch+1:3d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_m['loss']:.4f} | "
                  f"C-idx: {val_m['c_index']:.4f} | "
                  f"LR: {lr:.1e}")

        if patience_ctr >= config["patience"]:
            print(f"      Early stop ep {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

    final = evaluate(model, val_loader, loss_fn)
    return {
        "variant": variant_name,
        "fold": fold_idx,
        "c_index": final["c_index"],
        "best_c_index": best_ci,
        "epochs": epoch + 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  UNIFIED ABLATION — SAME FOLDS FOR ALL VARIANTS")
    print(f"  Variants: {', '.join(VARIANT_ORDER)}")
    print(f"  Config: hidden=128, layers=2, batch=512, lr=5e-4, patience=15")
    print(f"  Stratification: event-only (identical across all variants)")
    print("=" * 90)

    # Build all datasets
    print("\nBuilding datasets...")
    datasets = {}
    for vname in VARIANT_ORDER:
        vdef = VARIANTS[vname]
        print(f"  {vname}...", end=" ", flush=True)
        datasets[vname] = vdef["build_fn"]("msk_impact_50k")
        N = len(datasets[vname]["times"])
        print(f"{N} patients")

    n_ct = len(datasets["V2"]["cancer_type_vocab"])

    # Unified stratification: event-only (same for all variants)
    events = datasets["V2"]["events"].numpy().astype(int)
    N = len(events)

    skf = StratifiedKFold(n_splits=SHARED_CONFIG["n_folds"], shuffle=True,
                          random_state=SHARED_CONFIG["random_seed"])
    folds = list(skf.split(np.arange(N), events))

    # Load checkpoint if resuming
    completed = {}
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        completed = ckpt.get("completed", {})
        n_done = sum(len(v) for v in completed.values())
        print(f"\n  Resuming: {n_done} variant-folds already completed")

    t0 = time.time()

    # Train all variants on all folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*90}")
        print(f"  FOLD {fold_idx+1}/{SHARED_CONFIG['n_folds']} "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*90}")

        for vname in VARIANT_ORDER:
            key = f"{vname}_fold{fold_idx}"
            if key in completed:
                ci = completed[key]["c_index"]
                print(f"\n    {vname}: already done (C-index={ci:.4f})")
                continue

            print(f"\n    {vname}:")
            result = run_variant_fold(
                vname, fold_idx, train_idx, val_idx,
                datasets[vname], n_ct,
            )
            print(f"    → C-index = {result['c_index']:.4f} "
                  f"(best={result['best_c_index']:.4f}, "
                  f"epochs={result['epochs']})")

            completed[key] = result

            # Checkpoint after every variant-fold
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump({"completed": completed, "config": SHARED_CONFIG},
                          f, indent=2, default=str)

    elapsed = time.time() - t0

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  UNIFIED ABLATION RESULTS ({elapsed/60:.1f} min)")
    print(f"{'='*90}")

    header = f"  {'Fold':>6}"
    for vname in VARIANT_ORDER:
        header += f"  {vname:>10}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*10}" * len(VARIANT_ORDER))

    means = {v: [] for v in VARIANT_ORDER}

    for fold_idx in range(SHARED_CONFIG["n_folds"]):
        row = f"  {fold_idx+1:>6}"
        for vname in VARIANT_ORDER:
            key = f"{vname}_fold{fold_idx}"
            ci = completed[key]["c_index"]
            means[vname].append(ci)
            row += f"  {ci:>10.4f}"
        print(row)

    print(f"  {'-'*6}" + f"  {'-'*10}" * len(VARIANT_ORDER))

    # Mean row
    row = f"  {'Mean':>6}"
    mean_vals = {}
    for vname in VARIANT_ORDER:
        m = np.mean(means[vname])
        s = np.std(means[vname])
        mean_vals[vname] = m
        row += f"  {m:>.4f}±{s:.4f}"[:11].rjust(10)
    print(row)

    # Std row
    row = f"  {'Std':>6}"
    for vname in VARIANT_ORDER:
        s = np.std(means[vname])
        row += f"  {s:>10.4f}"
    print(row)

    # Delta rows
    print(f"\n  Signal decomposition (mean C-index):")
    v2 = mean_vals["V2"]
    v2m = mean_vals["V2_MSI"]
    v6 = mean_vals["V6"]
    v6c = mean_vals["V6c"]

    print(f"    V2 (6ch baseline):              {v2:.4f}")
    print(f"    + MSI/TMB only (V2_MSI):        {v2m:.4f}  ({v2m - v2:+.4f})")
    print(f"    + 8ch only (V6):                {v6:.4f}  ({v6 - v2:+.4f})")
    print(f"    + 8ch + MSI/TMB (V6c):          {v6c:.4f}  ({v6c - v2:+.4f})")
    print(f"")
    print(f"    MSI/TMB contribution:           {v2m - v2:+.4f}")
    print(f"    8-channel contribution:         {v6 - v2:+.4f}")
    print(f"    Combined:                       {v6c - v2:+.4f}")
    additive = (v2m - v2) + (v6 - v2)
    print(f"    If perfectly additive:          {additive:+.4f}")
    print(f"    Interaction (combined - sum):   {(v6c - v2) - additive:+.4f}")
    print(f"{'='*90}")

    # Paired fold-level comparison
    print(f"\n  Paired fold-level deltas (V6c vs V2):")
    for fold_idx in range(SHARED_CONFIG["n_folds"]):
        v2_ci = completed[f"V2_fold{fold_idx}"]["c_index"]
        v6c_ci = completed[f"V6c_fold{fold_idx}"]["c_index"]
        print(f"    Fold {fold_idx+1}: {v6c_ci - v2_ci:+.4f}")

    # Save results
    summary = {
        "config": SHARED_CONFIG,
        "variant_order": VARIANT_ORDER,
        "completed": completed,
        "means": {v: float(np.mean(means[v])) for v in VARIANT_ORDER},
        "stds": {v: float(np.std(means[v])) for v in VARIANT_ORDER},
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
