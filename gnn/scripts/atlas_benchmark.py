#!/usr/bin/env python3
"""
Benchmark: expanded atlas (tier 1-4 from Neo4j) vs old atlas (tier 1-3 only).

Measures C-index improvement from graph-imputed tier 4 entries.
Uses the same TreatmentDataset pipeline with Ridge regression on atlas features.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sksurv.metrics import concordance_index_censored

from gnn.config import CHANNEL_MAP, GNN_RESULTS


def run_benchmark():
    from gnn.data.treatment_dataset import TreatmentDataset

    print("Loading dataset...", flush=True)
    ds = TreatmentDataset()
    t0 = time.time()

    print("\nBuilding features...", flush=True)
    data = ds.build_features()

    N = data["node_features"].shape[0]
    times = data["times"].numpy()
    events = data["events"].numpy()

    # Build simple per-patient features from node features
    nf = data["node_features"].numpy()   # (N, max_nodes, 26)
    nm = data["node_masks"].numpy()      # (N, max_nodes)
    atlas_sums = data["atlas_sums"].numpy()  # (N, 1)

    print(f"Patients: {N}, build time: {time.time()-t0:.1f}s", flush=True)

    # Feature extraction: aggregate per-patient node features
    # For each patient, compute statistics over their mutation nodes
    n_node_feat = nf.shape[2]
    X_list = []

    for i in range(N):
        real = int(nm[i].sum())
        if real == 0:
            feats = np.zeros(n_node_feat * 4 + 3)
        else:
            nodes = nf[i, :real, :]
            feat_mean = nodes.mean(axis=0)
            feat_max = nodes.max(axis=0)
            feat_min = nodes.min(axis=0)
            feat_std = nodes.std(axis=0) if real > 1 else np.zeros(n_node_feat)

            feats = np.concatenate([
                feat_mean, feat_max, feat_min, feat_std,
                [real, atlas_sums[i, 0], float(events[i])],
            ])
            # Remove event from features (target leak!)
            feats[-1] = 0.0
        X_list.append(feats)

    X = np.array(X_list, dtype=np.float32)

    # Also extract per-patient tier breakdown
    # Tier is stored as tier/3.0 in dim 2, so recover raw: round(val * 3)
    tier_counts = np.zeros((N, 5), dtype=np.float32)  # tiers 0-4
    for i in range(N):
        real = int(nm[i].sum())
        for j in range(real):
            tier_val = int(round(nf[i, j, 2] * 3.0))  # tier is dim 2, normalized /3.0
            if 0 <= tier_val <= 4:
                tier_counts[i, tier_val] += 1

    X_full = np.hstack([X, tier_counts])

    # Valid patients only
    valid = times > 0
    X_v = X_full[valid]
    t_v = times[valid]
    e_v = events[valid].astype(bool)

    print(f"\nValid patients: {valid.sum()}")
    print(f"Feature dim: {X_v.shape[1]}")
    print(f"Events: {e_v.sum()} ({e_v.mean()*100:.1f}%)")
    print(f"\nTier coverage breakdown:")
    print(f"  Tier 0 (no atlas): {(tier_counts[:, 0] > 0).sum()} patients with >=1 unmatched")
    print(f"  Tier 1 (variant):  {(tier_counts[:, 1] > 0).sum()} patients")
    print(f"  Tier 2 (gene):     {(tier_counts[:, 2] > 0).sum()} patients")
    print(f"  Tier 3 (channel):  {(tier_counts[:, 3] > 0).sum()} patients")
    print(f"  Tier 4 (imputed):  {(tier_counts[:, 4] > 0).sum()} patients")

    # Also build "no tier 4" version: re-run build_features with t4 disabled
    print("\nBuilding features WITHOUT tier 4 (baseline)...", flush=True)
    saved_t4 = ds.t4
    ds.t4 = {}  # disable tier 4
    data_no_t4 = ds.build_features()
    ds.t4 = saved_t4  # restore

    nf_no_t4 = data_no_t4["node_features"].numpy()
    nm_no_t4 = data_no_t4["node_masks"].numpy()
    atlas_sums_no_t4 = data_no_t4["atlas_sums"].numpy()

    X_no_t4_list = []
    tier_counts_no_t4 = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        real = int(nm_no_t4[i].sum())
        if real == 0:
            feats = np.zeros(n_node_feat * 4 + 3)
        else:
            nodes = nf_no_t4[i, :real, :]
            feat_mean = nodes.mean(axis=0)
            feat_max = nodes.max(axis=0)
            feat_min = nodes.min(axis=0)
            feat_std = nodes.std(axis=0) if real > 1 else np.zeros(n_node_feat)
            feats = np.concatenate([
                feat_mean, feat_max, feat_min, feat_std,
                [real, atlas_sums_no_t4[i, 0], 0.0],
            ])
        X_no_t4_list.append(feats)
        for j in range(real):
            tv = int(round(nf_no_t4[i, j, 2] * 3.0))
            if 0 <= tv <= 4:
                tier_counts_no_t4[i, tv] += 1

    X_no_t4_arr = np.hstack([np.array(X_no_t4_list, dtype=np.float32), tier_counts_no_t4])
    X_no_t4_v = X_no_t4_arr[valid]

    # Check tier 4 disappeared
    print(f"  No-T4 tier 4 nodes: {int(tier_counts_no_t4[:, 4].sum())}")
    print(f"  No-T4 tier 0 nodes: {int(tier_counts_no_t4[:, 0].sum())}")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_scores_full = []
    c_scores_no_t4 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_v, e_v)):
        # Full model (all tiers including tier 4)
        ridge_full = Ridge(alpha=1.0)
        ridge_full.fit(X_v[train_idx], t_v[train_idx])
        pred_full = ridge_full.predict(X_v[val_idx])
        try:
            c_full = concordance_index_censored(e_v[val_idx], t_v[val_idx], -pred_full)[0]
        except Exception:
            c_full = 0.5
        c_scores_full.append(c_full)

        # Without tier 4 (clean re-build)
        ridge_base = Ridge(alpha=1.0)
        ridge_base.fit(X_no_t4_v[train_idx], t_v[train_idx])
        pred_base = ridge_base.predict(X_no_t4_v[val_idx])
        try:
            c_base = concordance_index_censored(e_v[val_idx], t_v[val_idx], -pred_base)[0]
        except Exception:
            c_base = 0.5
        c_scores_no_t4.append(c_base)

        print(f"  Fold {fold}: full={c_full:.4f}, no_t4={c_base:.4f}, "
              f"delta={c_full - c_base:+.4f}")

    mean_full = np.mean(c_scores_full)
    mean_no_t4 = np.mean(c_scores_no_t4)
    std_full = np.std(c_scores_full)

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Full atlas (T1-T4):  C = {mean_full:.4f} ± {std_full:.4f}")
    print(f"  Old atlas (T1-T3):   C = {mean_no_t4:.4f} ± {np.std(c_scores_no_t4):.4f}")
    print(f"  Delta:               {mean_full - mean_no_t4:+.4f}")
    print(f"  Tier 4 entries used: {len(ds.t4)}")
    print(f"  Total atlas:         {len(ds.t1) + len(ds.t2) + len(ds.t3) + len(ds.t4)}")


if __name__ == "__main__":
    run_benchmark()
