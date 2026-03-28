#!/usr/bin/env python3
"""
Ablation: ChannelNet with tiers + chains but NO cancer type embedding.
This is the apples-to-apples test of the coupling-channel grouping theory.

The question: do tiers and escalation chains add predictive value
purely from mutation structure, without knowing what cancer it is?

Usage:
    python3 -u -m gnn.scripts.ablation_no_cancer_type
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
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v3 import (
    build_channel_features_v3, ChannelDatasetV3, CHANNEL_FEAT_DIM,
    CHAIN_FEAT_DIM, N_CHAINS,
)
from gnn.models.layers import CrossChannelAttention
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS


class ChannelNetNoCancerType(nn.Module):
    """V3 architecture minus cancer type embedding and clinical covariates.
    Pure mutation-derived features: channels + tiers + escalation chains."""

    def __init__(self, config):
        super().__init__()
        ch_feat_dim = config.get("channel_feat_dim", 17)
        chain_feat_dim = config.get("chain_feat_dim", CHAIN_FEAT_DIM)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]

        # --- Channel path (6 channels) ---
        self.ch_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.ch_transformer = CrossChannelAttention(
            hidden, num_heads=config["cross_channel_heads"],
            num_layers=config.get("cross_channel_layers", 2),
            dropout=dropout,
        )
        self.ch_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Tier path (3 tiers) ---
        self.tier_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.tier_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=config.get("cross_channel_layers", 2),
        )
        self.tier_pos = nn.Embedding(3, hidden)
        self.tier_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Escalation chain path (2 chains) ---
        self.chain_encoder = nn.Sequential(
            nn.Linear(chain_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.chain_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=1,
        )
        self.chain_pos = nn.Embedding(N_CHAINS, hidden)
        self.chain_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Skip connection + readout (NO cancer type, NO clinical) ---
        skip_raw = 6 * ch_feat_dim + 3 * ch_feat_dim + N_CHAINS * chain_feat_dim
        total_dim = hidden + hidden + hidden + skip_raw

        self.readout = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        ch_feats = batch["channel_features"]
        tier_feats = batch["tier_features"]
        chain_feats = batch["chain_features"]
        B = ch_feats.shape[0]

        # Channel path
        h_ch = self.ch_encoder(ch_feats)
        h_ch = self.ch_transformer(h_ch)
        ch_scores = self.ch_attn(h_ch).squeeze(-1)
        ch_weights = F.softmax(ch_scores, dim=-1)
        ch_pooled = (ch_weights.unsqueeze(-1) * h_ch).sum(1)

        # Tier path
        h_tier = self.tier_encoder(tier_feats)
        pos_t = self.tier_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h_tier = self.tier_transformer(h_tier + pos_t)
        tier_scores = self.tier_attn(h_tier).squeeze(-1)
        tier_weights = F.softmax(tier_scores, dim=-1)
        tier_pooled = (tier_weights.unsqueeze(-1) * h_tier).sum(1)

        # Chain path
        h_chain = self.chain_encoder(chain_feats)
        pos_c = self.chain_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h_chain = self.chain_transformer(h_chain + pos_c)
        chain_scores = self.chain_attn(h_chain).squeeze(-1)
        chain_weights = F.softmax(chain_scores, dim=-1)
        chain_pooled = (chain_weights.unsqueeze(-1) * h_chain).sum(1)

        # Skip: raw features only
        raw_ch = ch_feats.reshape(B, -1)
        raw_tier = tier_feats.reshape(B, -1)
        raw_chain = chain_feats.reshape(B, -1)

        combined = torch.cat([
            ch_pooled, tier_pooled, chain_pooled,
            raw_ch, raw_tier, raw_chain,
        ], dim=-1)

        return self.readout(combined).squeeze(-1)


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
}

SAVE_BASE = os.path.join(GNN_RESULTS, "ablation_no_cancer_type")
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

    model = ChannelNetNoCancerType(config)
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
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 60)
    print("  ABLATION: NO CANCER TYPE (pure channel grouping test)")
    print(f"  Features: 6 channels + 3 tiers + 2 chains (NO cancer type, NO clinical)")
    print("=" * 60)

    data_dict = build_channel_features_v3("msk_impact_50k")
    N = len(data_dict["times"])
    print(f"  {N} patients")

    event_arr = data_dict["events"].numpy().astype(int)
    ch_count = (data_dict["channel_features"][:, :, 0] > 0).sum(dim=1).numpy().clip(max=3)
    strat = event_arr * 4 + ch_count

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["random_seed"])

    fold_results = []
    start_fold = 0
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        fold_results = ckpt.get("fold_results", [])
        start_fold = len(fold_results)
        if start_fold > 0:
            print(f"  Resuming from fold {start_fold + 1}")

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), strat)):
        if fold_idx < start_fold:
            continue

        print(f"\n  Fold {fold_idx+1}/{CONFIG['n_folds']} "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        result = run_fold(fold_idx, train_idx, val_idx, data_dict, CONFIG)
        fold_results.append(result)
        print(f"  Fold {fold_idx+1}: C-index = {result['c_index']:.4f}")

        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({"fold_results": fold_results, "config": CONFIG},
                      f, indent=2, default=str)
        print(f"  [checkpoint saved]")

    elapsed = time.time() - t0

    c_indices = [r["c_index"] for r in fold_results]
    mean_ci = np.mean(c_indices)
    std_ci = np.std(c_indices)

    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS: NO CANCER TYPE ({elapsed/60:.1f} min)")
    print(f"  C-index: {mean_ci:.4f} +/- {std_ci:.4f}")
    for t in [12, 36, 60]:
        aucs = [r["td_auc"].get(t, float("nan")) for r in fold_results]
        aucs = [a for a in aucs if not np.isnan(a)]
        if aucs:
            print(f"  AUC@{t}mo: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"\n  COMPARISON (apples-to-apples, mutation features only):")
    print(f"  Cox PH (channels only):         0.587")
    print(f"  ChannelNet v1 (channels):        0.620")
    print(f"  THIS (channels+tiers+chains):    {mean_ci:.4f}")
    print(f"  Cox-Sage (per-type, RNA-seq):    0.63-0.72")
    print(f"{'='*60}")

    summary = {
        "model": "ChannelNetNoCancerType",
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
