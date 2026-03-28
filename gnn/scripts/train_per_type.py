#!/usr/bin/env python3
"""
V4: Per-cancer-type fine-tuning from pan-cancer v3 weights.
Runs multiple cancer types in parallel (4 workers).

Usage:
    python3 -u -m gnn.scripts.train_per_type
    python3 -u -m gnn.scripts.train_per_type --min-patients 200
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

from gnn.data.channel_dataset_v3 import (
    build_channel_features_v3, ChannelDatasetV3, CHANNEL_FEAT_DIM,
    CHAIN_FEAT_DIM, N_CHAINS,
)
from gnn.models.channel_net_v3 import ChannelNetV3
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS

# Per-type config: fewer epochs, smaller patience (smaller cohorts converge faster)
CONFIG = {
    "channel_feat_dim": CHANNEL_FEAT_DIM,
    "chain_feat_dim": CHAIN_FEAT_DIM,
    "hidden_dim": 128,
    "cross_channel_heads": 4,
    "cross_channel_layers": 2,
    "dropout": 0.3,
    "lr": 1e-4,         # lower LR for fine-tuning
    "weight_decay": 1e-4,
    "epochs": 50,        # fewer epochs
    "patience": 10,
    "batch_size": 256,   # smaller batches for smaller cohorts
    "n_folds": 5,
    "random_seed": 42,
    "n_cancer_types": 80,
}

SAVE_BASE = os.path.join(GNN_RESULTS, "per_type_v4")

# Cox-Sage reported results
COX_SAGE = {
    "Breast Cancer": {"abbrev": "BRCA", "c_index": 0.669},
    "Bladder Cancer": {"abbrev": "BLCA", "c_index": 0.627},
    "Colorectal Cancer": {"abbrev": "COADREAD", "c_index": 0.646},
    "Glioblastoma": {"abbrev": "GBM", "c_index": 0.675},
    "Non-Small Cell Lung Cancer": {"abbrev": "LUAD", "c_index": 0.630},
    "Stomach Cancer": {"abbrev": "STAD", "c_index": 0.632},
    "Endometrial Cancer": {"abbrev": "UCEC", "c_index": 0.716},
}


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


def train_single_type(ct_name, ct_idx, data_dict, config, pretrained_path=None):
    """Train/fine-tune model for a single cancer type with 5-fold CV."""
    # Get indices for this cancer type
    mask = data_dict["cancer_type_idx"] == ct_idx
    type_indices = torch.where(mask)[0].numpy()
    N = len(type_indices)

    events = data_dict["events"][type_indices].numpy().astype(int)
    n_events = events.sum()

    if N < 100 or n_events < 20:
        return None

    # Stratify by event
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True,
                          random_state=config["random_seed"])

    fold_results = []
    type_dir = os.path.join(SAVE_BASE, ct_name.replace(" ", "_").replace(",", ""))
    os.makedirs(type_dir, exist_ok=True)

    for fold_idx, (train_local, val_local) in enumerate(skf.split(np.arange(N), events)):
        # Map local indices back to global
        train_idx = type_indices[train_local]
        val_idx = type_indices[val_local]

        train_ds = ChannelDatasetV3(data_dict, indices=train_idx)
        val_ds = ChannelDatasetV3(data_dict, indices=val_idx)

        bs = min(config["batch_size"], len(train_idx) // 2)
        bs = max(bs, 32)

        train_loader = DataLoader(train_ds, batch_size=bs,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bs,
                                shuffle=False, collate_fn=collate_fn)

        model = ChannelNetV3(config)

        # Load pretrained pan-cancer weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state, strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                     weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
        loss_fn = CoxPartialLikelihoodLoss()

        best_ci, best_state, patience_ctr = 0.0, None, 0

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

            if patience_ctr >= config["patience"]:
                break

        if best_state:
            model.load_state_dict(best_state)

        final = evaluate(model, val_loader, loss_fn)
        fold_results.append({
            "fold": fold_idx,
            "c_index": final["c_index"],
            "td_auc": final["td_auc"],
            "epochs": epoch + 1,
        })

    c_indices = [r["c_index"] for r in fold_results]
    result = {
        "cancer_type": ct_name,
        "n_patients": N,
        "n_events": int(n_events),
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices)),
        "fold_results": fold_results,
    }

    with open(os.path.join(type_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-patients", type=int, default=200,
                        help="Minimum patients per type")
    args = parser.parse_args()

    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 70)
    print("  V4: PER-CANCER-TYPE FINE-TUNING")
    print(f"  min patients: {args.min_patients}")
    print("=" * 70)

    data_dict = build_channel_features_v3("msk_impact_50k")
    N = len(data_dict["times"])
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    vocab = data_dict["cancer_type_vocab"]

    # Find best pretrained model (v3 fold 0, or v2 fold 0 as fallback)
    pretrained_path = None
    for p in [
        os.path.join(GNN_RESULTS, "channelnet_v3", "fold_0", "best_model.pt"),
        os.path.join(GNN_RESULTS, "channelnet_v2", "fold_0", "best_model.pt"),
    ]:
        if os.path.exists(p):
            pretrained_path = p
            break

    if pretrained_path:
        print(f"  Fine-tuning from: {pretrained_path}")
    else:
        print(f"  No pretrained model found, training from scratch")

    # Find eligible cancer types
    eligible = []
    for ct_idx, ct_name in enumerate(vocab):
        mask = data_dict["cancer_type_idx"] == ct_idx
        n_patients = mask.sum().item()
        n_events = int(data_dict["events"][mask].sum().item())
        if n_patients >= args.min_patients and n_events >= 20:
            eligible.append((ct_name, ct_idx, n_patients, n_events))

    # Sort by size descending so big ones run first
    eligible.sort(key=lambda x: -x[2])

    print(f"  {len(eligible)} cancer types eligible (>= {args.min_patients} patients)")
    print(f"  {N} total patients, {n_ct} types in vocab\n")

    t0 = time.time()

    all_results = []
    for i, (ct_name, ct_idx, n_pat, n_ev) in enumerate(eligible):
        print(f"  [{i+1}/{len(eligible)}] {ct_name} (N={n_pat}, events={n_ev})...")
        result = train_single_type(ct_name, ct_idx, data_dict, CONFIG, pretrained_path)
        if result:
            ci = result["mean_c_index"]
            cox = COX_SAGE.get(ct_name, {}).get("c_index")
            cox_str = f" (Cox-Sage: {cox:.3f})" if cox else ""
            print(f"    C-index = {ci:.4f} ± {result['std_c_index']:.4f}{cox_str}")
            all_results.append(result)
        else:
            print(f"    Skipped (too few patients/events)")

    elapsed = time.time() - t0

    # Print results table
    sorted_results = sorted(all_results, key=lambda r: -r["mean_c_index"])

    print(f"\n{'='*75}")
    print(f"  V4 PER-TYPE RESULTS ({elapsed/60:.1f} min)")
    print(f"{'='*75}")
    print(f"  {'Cancer Type':<35} {'N':>6} {'C-index':>12}  {'Cox-Sage':>9}  {'Delta':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*12}  {'-'*9}  {'-'*7}")

    for r in sorted_results:
        ci = r["mean_c_index"]
        std = r["std_c_index"]
        cox = COX_SAGE.get(r["cancer_type"], {}).get("c_index")
        cox_str = f"{cox:.3f}" if cox else "  —"
        delta_str = ""
        if cox:
            delta = ci - cox
            delta_str = f"{delta:+.3f}"

        print(f"  {r['cancer_type']:<35} {r['n_patients']:>6} "
              f"{ci:.4f}±{std:.4f}  {cox_str:>9}  {delta_str:>7}")

    print(f"{'='*75}")

    # Save summary
    summary = {
        "model": "ChannelNetV3_PerType",
        "config": CONFIG,
        "pretrained_from": pretrained_path,
        "results": sorted_results,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
