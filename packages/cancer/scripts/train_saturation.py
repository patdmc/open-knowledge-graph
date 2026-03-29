#!/usr/bin/env python3
"""
Saturation experiment: does n_channels_severed add predictive signal?

Tests whether the channel count (saturation/redundancy effect) improves
survival prediction when added as an explicit clinical feature.

Variants run on identical folds as unified_ablation:
  V6c           — baseline (8ch + MSI/TMB, 5 clinical features)
  V6c+nch       — adds n_channels_severed as 6th clinical feature
  clinical_only — ONLY n_channels_severed + age + sex + msi + tmb (no channel features)

If clinical_only matches V6c, the 72 channel features were proxies
for what a single integer captures.

Usage:
    python3 -u -m gnn.scripts.train_saturation
"""

import sys
import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import CHANNEL_FEAT_DIM_V6, N_CHANNELS_V6, N_TIERS_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index
from gnn.config import GNN_RESULTS

SAVE_BASE = os.path.join(GNN_RESULTS, "saturation_experiment")

CONFIG = {
    "hidden_dim": 128,
    "cross_channel_heads": 4,
    "cross_channel_layers": 2,
    "dropout": 0.3,
    "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 15,
    "batch_size": 512,
    "n_folds": 5,
    "random_seed": 42,
}

COLLATE_KEYS = [
    "channel_features", "tier_features", "cancer_type_idx",
    "age", "sex", "msi_score", "msi_high", "tmb",
    "n_channels", "time", "event",
]


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in COLLATE_KEYS}


class SaturationDataset(Dataset):
    def __init__(self, data_dict, indices=None):
        self.indices = indices
        self.data = data_dict

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data["times"])

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        return {
            "channel_features": self.data["channel_features"][idx],
            "tier_features": self.data["tier_features"][idx],
            "cancer_type_idx": self.data["cancer_type_idx"][idx],
            "age": self.data["age"][idx],
            "sex": self.data["sex"][idx],
            "msi_score": self.data["msi_score"][idx],
            "msi_high": self.data["msi_high"][idx],
            "tmb": self.data["tmb"][idx],
            "n_channels": self.data["n_channels"][idx],
            "time": self.data["times"][idx],
            "event": self.data["events"][idx],
        }


class ChannelNetV6cSat(nn.Module):
    """V6c + n_channels_severed as 6th clinical feature."""

    def __init__(self, config):
        super().__init__()
        feat_dim = config.get("channel_feat_dim", CHANNEL_FEAT_DIM_V6)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        self.ch_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout),
        )

        from gnn.models.channel_net_v6c import CrossChannelTransformerV6c, TierTransformerV6c
        self.ch_transformer = CrossChannelTransformerV6c(
            hidden, config["cross_channel_heads"],
            config.get("cross_channel_layers", 2), dropout,
        )
        self.tier_encoder = nn.Sequential(
            nn.Linear(5, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout),
        )
        self.tier_transformer = TierTransformerV6c(hidden, 2, dropout)
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # 6 clinical inputs: age, sex, msi_score, msi_high, tmb, n_channels
        self.clinical_encoder = nn.Sequential(
            nn.Linear(6, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )

        skip_raw = N_CHANNELS_V6 * feat_dim + N_TIERS_V6 * 5
        total_dim = hidden + hidden + hidden // 2 + hidden // 4 + skip_raw

        self.readout = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        ch_feats = batch["channel_features"]
        tier_feats = batch["tier_features"]
        ct_idx = batch["cancer_type_idx"]
        B = ch_feats.shape[0]

        h_ch = self.ch_encoder(ch_feats)
        ch_pooled = self.ch_transformer(h_ch)
        h_tier = self.tier_encoder(tier_feats)
        tier_pooled = self.tier_transformer(h_tier)
        cancer = self.cancer_embed(ct_idx)

        clinical_input = torch.stack([
            batch["age"], batch["sex"].float(),
            batch["msi_score"], batch["msi_high"].float(),
            batch["tmb"], batch["n_channels"],
        ], dim=-1)
        clinical = self.clinical_encoder(clinical_input)

        raw = torch.cat([ch_feats.reshape(B, -1), tier_feats.reshape(B, -1)], dim=-1)
        combined = torch.cat([ch_pooled, tier_pooled, cancer, clinical, raw], dim=-1)
        return self.readout(combined).squeeze(-1)


class ClinicalOnlyNet(nn.Module):
    """Only clinical features + n_channels — no channel/tier features."""

    def __init__(self, config):
        super().__init__()
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # 6 clinical: age, sex, msi_score, msi_high, tmb, n_channels
        self.encoder = nn.Sequential(
            nn.Linear(6 + hidden // 2, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, batch):
        cancer = self.cancer_embed(batch["cancer_type_idx"])
        clinical = torch.stack([
            batch["age"], batch["sex"].float(),
            batch["msi_score"], batch["msi_high"].float(),
            batch["tmb"], batch["n_channels"],
        ], dim=-1)
        x = torch.cat([clinical, cancer], dim=-1)
        return self.encoder(x).squeeze(-1)


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
    return {
        "loss": total_loss / max(n, 1),
        "c_index": concordance_index(torch.cat(all_h), torch.cat(all_t), torch.cat(all_e)),
    }


def run_variant(name, model_cls, data, folds, n_ct):
    config = {**CONFIG, "n_cancer_types": n_ct}
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ds = SaturationDataset(data, indices=train_idx)
        val_ds = SaturationDataset(data, indices=val_idx)
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                                shuffle=False, collate_fn=collate_fn)

        model = model_cls(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                                     weight_decay=CONFIG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=7, factor=0.5)
        loss_fn = CoxPartialLikelihoodLoss()

        best_ci, best_state, patience_ctr = 0.0, None, 0
        fold_dir = os.path.join(SAVE_BASE, name, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(CONFIG["epochs"]):
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
                print(f"      Ep {epoch+1:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_m['loss']:.4f} | C-idx: {val_m['c_index']:.4f} | LR: {lr:.1e}")

            if patience_ctr >= CONFIG["patience"]:
                print(f"      Early stop ep {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
            torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

        final = evaluate(model, val_loader, loss_fn)
        fold_results.append({
            "fold": fold_idx,
            "c_index": final["c_index"],
            "best_c_index": best_ci,
            "epochs": epoch + 1,
        })
        print(f"    → Fold {fold_idx+1}: C-index = {final['c_index']:.4f} "
              f"(best={best_ci:.4f}, epochs={epoch+1})")

    return fold_results


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  SATURATION EXPERIMENT")
    print("  Does n_channels_severed add signal beyond per-channel features?")
    print("=" * 90)

    # Build data
    print("\nBuilding features...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

    # Compute n_channels_severed from existing channel features
    # channel_features[:, :, 0] is the is_severed flag per channel
    n_channels = data["channel_features"][:, :, 0].sum(dim=1)  # (N,)
    # Normalize to ~[0,1]
    n_channels_norm = n_channels / N_CHANNELS_V6
    data["n_channels"] = n_channels_norm

    print(f"  N_channels distribution:")
    for k in range(N_CHANNELS_V6 + 1):
        n = (n_channels == k).sum().item()
        print(f"    {k} channels: {n:>6} ({100*n/N:.1f}%)")

    # Same folds as unified ablation
    events = data["events"].numpy().astype(int)
    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["random_seed"])
    folds = list(skf.split(np.arange(N), events))

    t0 = time.time()
    all_results = {}

    # Variant 1: V6c baseline (reuse ChannelNetV6c but pass n_channels through)
    # We need a wrapper since V6c doesn't use n_channels
    class V6cWrapper(ChannelNetV6c):
        def forward(self, batch):
            # Ignore n_channels, use standard V6c forward
            return super().forward(batch)

    variants = [
        ("V6c_baseline", V6cWrapper),
        ("V6c+nch", ChannelNetV6cSat),
        ("clinical_only", ClinicalOnlyNet),
    ]

    for name, model_cls in variants:
        print(f"\n{'='*90}")
        print(f"  {name}")
        print(f"{'='*90}")
        results = run_variant(name, model_cls, data, folds, n_ct)
        all_results[name] = results

    elapsed = time.time() - t0

    # Summary
    print(f"\n\n{'='*90}")
    print(f"  SATURATION EXPERIMENT RESULTS ({elapsed/60:.1f} min)")
    print(f"{'='*90}")

    header = f"  {'Fold':>6}"
    for name, _ in variants:
        header += f"  {name:>15}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*15}" * len(variants))

    means = {name: [] for name, _ in variants}
    for fold_idx in range(CONFIG["n_folds"]):
        row = f"  {fold_idx+1:>6}"
        for name, _ in variants:
            ci = all_results[name][fold_idx]["best_c_index"]
            means[name].append(ci)
            row += f"  {ci:>15.4f}"
        print(row)

    print(f"  {'-'*6}" + f"  {'-'*15}" * len(variants))
    row = f"  {'Mean':>6}"
    for name, _ in variants:
        m = np.mean(means[name])
        s = np.std(means[name])
        row += f"  {m:.4f}±{s:.4f}".rjust(15)
    print(row)

    # Deltas
    v6c_mean = np.mean(means["V6c_baseline"])
    print(f"\n  Deltas vs V6c baseline ({v6c_mean:.4f}):")
    for name, _ in variants:
        if name == "V6c_baseline":
            continue
        m = np.mean(means[name])
        print(f"    {name}: {m:.4f} ({m - v6c_mean:+.4f})")

    # Compare with unified ablation V6c
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    if os.path.exists(ua_path):
        with open(ua_path) as f:
            ua = json.load(f)
        ua_v6c = ua["means"].get("V6c", 0)
        print(f"\n  Unified ablation V6c: {ua_v6c:.4f}")
        print(f"  This experiment V6c:  {v6c_mean:.4f}")

    print(f"\n{'='*90}")

    # Save
    summary = {
        "variants": [name for name, _ in variants],
        "results": {name: all_results[name] for name, _ in variants},
        "means": {name: float(np.mean(means[name])) for name, _ in variants},
        "stds": {name: float(np.std(means[name])) for name, _ in variants},
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
