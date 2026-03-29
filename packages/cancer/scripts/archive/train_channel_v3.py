#!/usr/bin/env python3
"""
Train ChannelNetV3 — V2 + escalation chain features.
Checkpoints after every fold for resume capability.

Usage:
    python3 -u -m gnn.scripts.train_channel_v3
    python3 -u -m gnn.scripts.train_channel_v3 --resume
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
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

from gnn.data.channel_dataset_v3 import (
    build_channel_features_v3, ChannelDatasetV3, CHANNEL_FEAT_DIM,
    CHAIN_FEAT_DIM, N_CHAINS,
)
from gnn.models.channel_net_v3 import ChannelNetV3
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS

CONFIG = {
    "channel_feat_dim": CHANNEL_FEAT_DIM,
    "chain_feat_dim": CHAIN_FEAT_DIM,
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

SAVE_BASE = os.path.join(GNN_RESULTS, "channelnet_v3")
CHECKPOINT_PATH = os.path.join(SAVE_BASE, "checkpoint.json")


def collate_fn(batch):
    return {
        "channel_features": torch.stack([b["channel_features"] for b in batch]),
        "tier_features": torch.stack([b["tier_features"] for b in batch]),
        "chain_features": torch.stack([b["chain_features"] for b in batch]),
        "cancer_type_idx": torch.stack([b["cancer_type_idx"] for b in batch]),
        "age": torch.stack([b["age"] for b in batch]),
        "sex": torch.stack([b["sex"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
    }


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
        "td_auc": time_dependent_auc(all_h, all_t, all_e),
    }


def run_fold(fold_idx, train_idx, val_idx, data_dict, config):
    train_ds = ChannelDatasetV3(data_dict, indices=train_idx)
    val_ds = ChannelDatasetV3(data_dict, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = ChannelNetV3(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )
    loss_fn = CoxPartialLikelihoodLoss()

    best_ci, best_state, patience_ctr = 0.0, None, 0
    fold_dir = os.path.join(SAVE_BASE, f"fold_{fold_idx}")
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
            print(f"    Epoch {epoch+1:3d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_m['loss']:.4f} | "
                  f"C-idx: {val_m['c_index']:.4f} | "
                  f"LR: {lr:.1e}")

        if patience_ctr >= config["patience"]:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

    final = evaluate(model, val_loader, loss_fn)
    return {
        "fold": fold_idx,
        "c_index": final["c_index"],
        "td_auc": final["td_auc"],
        "epochs": epoch + 1,
        "best_c_index": best_ci,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 60)
    print("  CHANNELNET V3 TRAINING (with escalation chains)")
    print(f"  hidden={CONFIG['hidden_dim']}, layers={CONFIG['cross_channel_layers']}, "
          f"batch={CONFIG['batch_size']}")
    print(f"  Features: 6 channels + 3 tiers + 2 chains + cancer type + clinical")
    print("=" * 60)

    data_dict = build_channel_features_v3("msk_impact_50k")
    N = len(data_dict["times"])
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    print(f"  {N} patients, {n_ct} cancer types")

    # Stratification
    event_arr = data_dict["events"].numpy().astype(int)
    ch_count = (data_dict["channel_features"][:, :, 0] > 0).sum(dim=1).numpy().clip(max=3)
    strat = event_arr * 4 + ch_count

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["random_seed"])

    # Load checkpoint if resuming
    fold_results = []
    start_fold = 0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        fold_results = ckpt.get("fold_results", [])
        start_fold = len(fold_results)
        if start_fold > 0:
            print(f"  Resuming from fold {start_fold + 1} "
                  f"(folds 1-{start_fold} loaded)")
            for r in fold_results:
                print(f"    Fold {r['fold']+1}: C-index = {r['c_index']:.4f}")

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), strat)):
        if fold_idx < start_fold:
            continue

        print(f"\n  Fold {fold_idx+1}/{CONFIG['n_folds']} "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        result = run_fold(fold_idx, train_idx, val_idx, data_dict, CONFIG)
        fold_results.append(result)
        print(f"  Fold {fold_idx+1}: C-index = {result['c_index']:.4f}")

        # Checkpoint
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({"fold_results": fold_results, "config": CONFIG},
                      f, indent=2, default=str)
        print(f"  [checkpoint saved]")

    elapsed = time.time() - t0

    c_indices = [r["c_index"] for r in fold_results]
    mean_ci = np.mean(c_indices)
    std_ci = np.std(c_indices)

    print(f"\n{'='*60}")
    print(f"  RESULTS ({elapsed/60:.1f} min)")
    print(f"  C-index: {mean_ci:.4f} +/- {std_ci:.4f}")
    for t in [12, 36, 60]:
        aucs = [r["td_auc"].get(t, float("nan")) for r in fold_results]
        aucs = [a for a in aucs if not np.isnan(a)]
        if aucs:
            print(f"  AUC@{t}mo: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"\n  vs Cox PH baseline: 0.587")
    print(f"  vs ChannelNet v1:   ~0.62")
    print(f"  vs ChannelNet v2:   (running)")
    print(f"  vs Cox-Sage (TCGA): 0.63-0.72 (per-type, gene expression)")
    print(f"{'='*60}")

    summary = {
        "model": "ChannelNetV3",
        "config": CONFIG,
        "mean_c_index": mean_ci,
        "std_c_index": std_ci,
        "fold_results": fold_results,
        "n_patients": N,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
