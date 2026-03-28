#!/usr/bin/env python3
"""
Train ChannelNet — channel-level GNN with skip connections.

6 nodes instead of 99. Pre-aggregated features. Should be 10-100x faster
than gene-level GNN and produce better results due to skip connection
giving direct access to linear-model-equivalent features.

Usage:
    python3 -u -m gnn.scripts.train_channel
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import sys
import os
import time
import json
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset import build_channel_features, ChannelDataset, CHANNEL_FEAT_DIM
from gnn.models.channel_net import ChannelNet
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS


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
}


def collate_fn(batch):
    """Custom collate for channel dataset."""
    return {
        "channel_features": torch.stack([b["channel_features"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
    }


class ChannelBatch:
    """Simple batch wrapper to match model interface."""
    def __init__(self, d):
        self.channel_features = d["channel_features"]
        self.time = d["time"]
        self.event = d["event"]


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    n = 0
    for batch_dict in loader:
        batch = ChannelBatch(batch_dict)
        optimizer.zero_grad()
        hazard = model(batch)
        loss = loss_fn(hazard, batch.time, batch.event)
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
    total_loss = 0
    n = 0
    for batch_dict in loader:
        batch = ChannelBatch(batch_dict)
        hazard = model(batch)
        loss = loss_fn(hazard, batch.time, batch.event)
        total_loss += loss.item()
        n += 1
        all_h.append(hazard)
        all_t.append(batch.time)
        all_e.append(batch.event)

    all_h = torch.cat(all_h)
    all_t = torch.cat(all_t)
    all_e = torch.cat(all_e)
    ci = concordance_index(all_h, all_t, all_e)
    td_auc = time_dependent_auc(all_h, all_t, all_e)
    return {
        "loss": total_loss / max(n, 1),
        "c_index": ci,
        "td_auc": td_auc,
    }


def run_fold(fold_idx, train_idx, val_idx, features, times, events, config, save_dir):
    train_ds = ChannelDataset(features, times, events, train_idx)
    val_ds = ChannelDataset(features, times, events, val_idx)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = ChannelNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )
    loss_fn = CoxPartialLikelihoodLoss()

    best_ci = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics = evaluate(model, val_loader, loss_fn)
        scheduler.step(val_metrics["c_index"])

        if val_metrics["c_index"] > best_ci:
            best_ci = val_metrics["c_index"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch+1:3d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} | "
                  f"C-idx: {val_metrics['c_index']:.4f} | "
                  f"LR: {lr:.1e}")

        if patience_counter >= config["patience"]:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # Restore best and final eval
    if best_state:
        model.load_state_dict(best_state)
    final = evaluate(model, val_loader, loss_fn)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, "best_model.pt"))

    return {
        "fold": fold_idx,
        "c_index": final["c_index"],
        "td_auc": final["td_auc"],
        "epochs": epoch + 1,
        "best_c_index": best_ci,
    }


def main():
    print("=" * 60)
    print("  CHANNELNET TRAINING")
    print(f"  hidden={CONFIG['hidden_dim']}, layers={CONFIG['cross_channel_layers']}, "
          f"batch={CONFIG['batch_size']}, lr={CONFIG['lr']}")
    print("=" * 60)

    features, times, events = build_channel_features("msk_impact_50k")
    N = len(times)
    print(f"  {N} patients, {CHANNEL_FEAT_DIM} features per channel")

    # Stratification
    event_arr = events.numpy().astype(int)
    ch_count = (features[:, :, 0] > 0).sum(dim=1).numpy().clip(max=3)
    strat = event_arr * 4 + ch_count

    save_base = os.path.join(GNN_RESULTS, "channelnet")
    os.makedirs(save_base, exist_ok=True)

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["random_seed"])

    # Resume from checkpoint if available
    checkpoint_path = os.path.join(save_base, "checkpoint.json")
    fold_results = []
    start_fold = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        fold_results = ckpt.get("fold_results", [])
        start_fold = len(fold_results)
        if start_fold > 0:
            print(f"  Resuming from fold {start_fold + 1} "
                  f"(folds 1-{start_fold} loaded from checkpoint)")

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), strat)):
        if fold_idx < start_fold:
            continue

        print(f"\n  Fold {fold_idx+1}/{CONFIG['n_folds']} "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        result = run_fold(
            fold_idx, train_idx, val_idx, features, times, events,
            CONFIG, os.path.join(save_base, f"fold_{fold_idx}")
        )
        fold_results.append(result)
        print(f"  Fold {fold_idx+1}: C-index = {result['c_index']:.4f}")

        # Checkpoint after each fold
        with open(checkpoint_path, "w") as f:
            json.dump({"fold_results": fold_results}, f, indent=2, default=str)
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
    print(f"{'='*60}")

    summary = {
        "model": "ChannelNet",
        "config": CONFIG,
        "mean_c_index": mean_ci,
        "std_c_index": std_ci,
        "fold_results": fold_results,
        "n_patients": N,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(save_base, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to {save_base}")


if __name__ == "__main__":
    main()
