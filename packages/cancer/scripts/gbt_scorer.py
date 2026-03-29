#!/usr/bin/env python3
"""
GBT (Gradient-Boosted Tree) scorer — LightGBM with Cox objective.

Reuses the 75-dim pure graph feature extraction from graph_walk_scorer.py.
Replaces ridge with LightGBM to capture nonlinear feature interactions
(hub_damage × channel_shift, frac_isolated when GOF/LOF ratio is high, etc.).

Also runs per-CT GBT for cancer types with ≥200 patients.

No model dependency — pure graph features + survival target.

Usage:
    python3 -u -m gnn.scripts.gbt_scorer
"""

import sys, os, json, time
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from gnn.config import GNN_RESULTS, CHANNEL_NAMES
from gnn.training.metrics import concordance_index

SAVE_BASE = os.path.join(GNN_RESULTS, "gbt_scorer")

FEAT_NAMES = (
    ["ct_baseline", "weighted_ch_shift", "ct_gene_shift", "global_gene_shift"]
    + [f"ch_dmg_{ch}" for ch in CHANNEL_NAMES]
    + ["entropy", "hub_damage", "tier_conn", "n_mutated", "hhi"]
    + ["mean_ppi_dist", "min_ppi_dist", "frac_connected", "frac_dist1",
       "frac_dist2_3", "frac_dist4plus",
       "mean_overlap", "min_overlap", "max_overlap",
       "frac_same_ch", "frac_cross_ch", "mean_cooccur_wt", "max_cooccur_wt",
       "frac_with_cooccur", "mean_combined_ent", "n_hub_hub", "n_hub_nonhub",
       "frac_close_cross", "frac_far_same", "mean_tier_dist", "frac_cross_tier",
       "n_pairs"]
    + ["n_components", "largest_comp", "frac_in_largest", "n_isolated",
       "frac_isolated", "comp_entropy", "max_comp_damage", "comp_spread"]
    + ["n_ct_cooccur", "ct_cooccur_wt", "max_ct_cooccur", "cross_ch_cooccur"]
    + [f"ch_max_hr_{ch}" for ch in CHANNEL_NAMES]
    + ["branching_ratio", "fragmentation", "linearity", "connectivity",
       "depth", "frontier_width", "ppi_coverage"]
    + ["tissue_delta", "sex", "msi", "msi_high", "tmb_proxy"]
    + ["frac_gof", "frac_lof", "gof_lof_ratio", "frac_actionable"]
    # Escalation regime features [75-89]
    + ["dfs_score", "bfs_score",
       "hub_x_dfs", "hub_x_bfs",
       "nmut_x_tier", "nmut_x_single_tier",
       "nmut_x_entropy", "nmut_x_hhi",
       "secondary_hub_hits", "nhub_x_dfs",
       "isolated_x_actionable", "isolated_x_unknown",
       "hubhub_x_bfs",
       "connected_x_dfs", "connected_x_bfs"]
)


def load_features_and_labels():
    """Load graph features using graph_walk_scorer's data pipeline."""
    import networkx as nx
    from gnn.config import GNN_CACHE, HUB_GENES, CHANNEL_MAP
    from gnn.scripts.graph_walk_scorer import (
        load_graph_data, compute_shifts, graph_walk_features,
        precompute_ppi_distances, TIER_MAP,
    )
    from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles

    patients, gene_log_hr, global_gene_hr, cooccur_ct, global_cooccur = load_graph_data()

    global_baseline, ct_baseline, global_gene_shift, ct_gene_shift, ct_ch_shift = \
        compute_shifts(patients, gene_log_hr, global_gene_hr)

    expanded_cm = dict(CHANNEL_MAP)
    all_genes = set()
    for p in patients.values():
        all_genes |= set(p['genes'].keys())
    all_genes &= set(expanded_cm.keys())

    G_ppi = nx.Graph()
    for gene, ch in expanded_cm.items():
        G_ppi.add_node(gene, channel=ch, tier=TIER_MAP.get(ch, -1))

    ppi_cache = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    if os.path.exists(ppi_cache):
        with open(ppi_cache) as f:
            ppi_data = json.load(f)
        for edge in ppi_data:
            g1, g2 = edge[0], edge[1]
            score = float(edge[2]) if len(edge) > 2 else 0.5
            if g1 in expanded_cm and g2 in expanded_cm:
                G_ppi.add_edge(g1, g2, weight=score)

    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)
    ppi_dists = precompute_ppi_distances(G_ppi, all_genes)

    hub_gene_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_gene_set |= ch_hubs

    print(f"  Computing 75-dim features for {len(patients)} patients...")
    X, pid_list = graph_walk_features(
        patients, channel_profiles, ppi_dists, G_ppi,
        ct_baseline, global_baseline, ct_gene_shift, global_gene_shift, ct_ch_shift,
        cooccur_ct, global_cooccur, hub_gene_set, expanded_cm,
    )

    N = len(pid_list)
    times = np.array([patients[pid]['time'] for pid in pid_list])
    events = np.array([patients[pid]['event'] for pid in pid_list])
    cts = np.array([patients[pid]['ct'] for pid in pid_list])

    return X, times, events, cts, pid_list


def eval_c_index(scores, times, events):
    return concordance_index(
        torch.tensor(scores.astype(np.float32)),
        torch.tensor(times.astype(np.float32)),
        torch.tensor(events.astype(np.float32)),
    )


def train_gbt_survival(X_train, y_time, y_event, params,
                        X_val=None, y_time_val=None, y_event_val=None):
    """Train sklearn HistGradientBoostingRegressor for survival prediction.

    Target: log(time) * (1 - 0.3 * event) — censored patients get partial credit.
    Model predicts log-survival (higher = longer life).

    Uses HistGradientBoostingRegressor (fast, handles 41K rows well).
    """
    label = np.log1p(y_time) * (1 - y_event * 0.3)

    kwargs = dict(
        max_depth=params.get("max_depth", 5),
        max_iter=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        min_samples_leaf=params.get("min_child_samples", 30),
        max_leaf_nodes=params.get("num_leaves", 31),
        random_state=42,
        verbose=0,
    )
    if X_val is not None:
        kwargs["early_stopping"] = True
        kwargs["validation_fraction"] = 0.1
        kwargs["n_iter_no_change"] = 30
    else:
        kwargs["early_stopping"] = False

    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(X_train, label)
    return model


def hyperparameter_sweep(X, times, events, folds):
    """Grid search over LightGBM Cox hyperparameters."""
    param_grid = []
    for max_depth in [3, 5, 7]:
        for n_est in [200, 500]:
            for lr in [0.01, 0.05]:
                    param_grid.append({
                        "objective": "cox",
                        "metric": "",
                        "max_depth": max_depth,
                        "n_estimators": n_est,
                        "learning_rate": lr,
                        "min_child_samples": 30,
                        "num_leaves": 2 ** max_depth - 1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "verbose": -1,
                        "seed": 42,
                    })

    print(f"  Sweeping {len(param_grid)} hyperparameter combos...")
    best_c = 0.0
    best_params = None
    N = len(times)

    for pi, params in enumerate(param_grid):
        scores = np.zeros(N)
        for train_idx, val_idx in folds:
            model = train_gbt_survival(
                X[train_idx], times[train_idx], events[train_idx], params,
                X[val_idx], times[val_idx], events[val_idx],
            )
            scores[val_idx] = -model.predict(X[val_idx])

        ci = eval_c_index(scores, times, events)
        if (pi + 1) % 8 == 0 or ci > best_c:
            tag = " ***" if ci > best_c else ""
            print(f"    [{pi+1}/{len(param_grid)}] depth={params['max_depth']} "
                  f"n={params['n_estimators']} lr={params['learning_rate']} "
                  f"mc={params['min_child_samples']} → C={ci:.4f}{tag}")
        if ci > best_c:
            best_c = ci
            best_params = params.copy()

    return best_params, best_c


def run_global_gbt(X, times, events, folds, params):
    N = len(times)
    scores = np.zeros(N)
    for train_idx, val_idx in folds:
        model = train_gbt_survival(
            X[train_idx], times[train_idx], events[train_idx], params,
            X[val_idx], times[val_idx], events[val_idx],
        )
        scores[val_idx] = -model.predict(X[val_idx])
    return scores


def run_per_ct_gbt(X, times, events, cts, folds, params):
    N = len(times)
    scores = np.zeros(N)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        ct_train = defaultdict(list)
        for idx in train_idx:
            ct_train[cts[idx]].append(idx)

        model_global = train_gbt_survival(
            X[train_idx], times[train_idx], events[train_idx], params,
        )

        ct_models = {}
        for ct_name, ct_indices in ct_train.items():
            ct_arr = np.array(ct_indices)
            n_events = events[ct_arr].sum()
            if len(ct_arr) >= 200 and n_events >= 20:
                ct_params = params.copy()
                ct_params["max_depth"] = min(params["max_depth"], 5)
                ct_params["n_estimators"] = min(params["n_estimators"], 200)
                ct_params["min_child_samples"] = max(params["min_child_samples"], 30)
                ct_models[ct_name] = train_gbt_survival(
                    X[ct_arr], times[ct_arr], events[ct_arr], ct_params,
                )

        for idx in val_idx:
            ct_name = cts[idx]
            if ct_name in ct_models:
                scores[idx] = -ct_models[ct_name].predict(X[idx:idx+1])[0]
            else:
                scores[idx] = -model_global.predict(X[idx:idx+1])[0]

    return scores


def run_ridge_baseline(X, times, events, cts, folds, alpha=100.0):
    """Ridge baselines for comparison.
    Ridge predicts survival time (higher = longer life), so we negate
    to get risk scores (higher = higher risk) for concordance_index.
    """
    N = len(times)

    # Global
    global_scores = np.zeros(N)
    for train_idx, val_idx in folds:
        reg = Ridge(alpha=alpha)
        y_train = times[train_idx] * (1 - events[train_idx] * 0.5)
        reg.fit(X[train_idx], y_train)
        global_scores[val_idx] = -reg.predict(X[val_idx])  # negate: higher risk = shorter survival

    # Per-CT
    perct_scores = np.zeros(N)
    for train_idx, val_idx in folds:
        ct_train = defaultdict(list)
        for idx in train_idx:
            ct_train[cts[idx]].append(idx)

        reg_global = Ridge(alpha=alpha)
        y_train = times[train_idx] * (1 - events[train_idx] * 0.5)
        reg_global.fit(X[train_idx], y_train)

        ct_models = {}
        for ct_name, ct_indices in ct_train.items():
            if len(ct_indices) >= 200:
                ct_arr = np.array(ct_indices)
                reg_ct = Ridge(alpha=alpha)
                y_ct = times[ct_arr] * (1 - events[ct_arr] * 0.5)
                reg_ct.fit(X[ct_arr], y_ct)
                ct_models[ct_name] = reg_ct

        for idx in val_idx:
            ct_name = cts[idx]
            if ct_name in ct_models:
                perct_scores[idx] = -ct_models[ct_name].predict(X[idx:idx+1])[0]
            else:
                perct_scores[idx] = -reg_global.predict(X[idx:idx+1])[0]

    return global_scores, perct_scores


def ct_breakdown(scores, times, events, cts, label=""):
    """Per-CT C-index breakdown. Returns dict of results."""
    ct_data = defaultdict(lambda: {'s': [], 't': [], 'e': []})
    for i in range(len(scores)):
        ct_data[cts[i]]['s'].append(scores[i])
        ct_data[cts[i]]['t'].append(times[i])
        ct_data[cts[i]]['e'].append(events[i])

    results = {}
    for ct_name in sorted(ct_data.keys()):
        d = ct_data[ct_name]
        n = len(d['s'])
        n_events = sum(d['e'])
        if n >= 50 and n_events >= 5:
            ci = eval_c_index(
                np.array(d['s'], dtype=np.float32),
                np.array(d['t'], dtype=np.float32),
                np.array(d['e'], dtype=np.float32),
            )
            results[ct_name] = {'n': n, 'events': n_events, 'c': float(ci)}

    return results


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("  GBT SCORER (LightGBM Cox — pure graph features)")
    print("=" * 80)

    # Load features (cache to avoid recomputing)
    cache_path = os.path.join(SAVE_BASE, "features_cache.npz")
    if os.path.exists(cache_path):
        print("\n  Loading cached features...")
        cached = np.load(cache_path, allow_pickle=True)
        X = cached['X']
        times = cached['times']
        events = cached['events']
        cts = cached['cts']
        pid_list = cached['pid_list'].tolist()
    else:
        print("\n  Loading features from Neo4j graph walk...")
        X, times, events, cts, pid_list = load_features_and_labels()
        np.savez_compressed(cache_path, X=X, times=times, events=events,
                            cts=cts, pid_list=np.array(pid_list))
        print(f"    Cached to {cache_path}")
    N = X.shape[0]
    print(f"    {N} patients, {X.shape[1]} features")

    # Validate features — clip extremes, report issues
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {n_nan} NaN, {n_inf} Inf in features — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    for col in range(X.shape[1]):
        col_max = np.abs(X[:, col]).max()
        if col_max > 1e6:
            name = FEAT_NAMES[col] if col < len(FEAT_NAMES) else f"[{col}]"
            print(f"  WARNING: {name} has max |value| = {col_max:.0f} — clipping to ±1e4")
            X[:, col] = np.clip(X[:, col], -1e4, 1e4)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))

    # =========================================================================
    # Ridge baseline
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  RIDGE BASELINE")
    print(f"{'='*80}")

    ridge_global_scores, ridge_perct_scores = run_ridge_baseline(X, times, events, cts, folds)
    ridge_global_c = eval_c_index(ridge_global_scores, times, events)
    ridge_perct_c = eval_c_index(ridge_perct_scores, times, events)
    print(f"  Global Ridge: {ridge_global_c:.4f}")
    print(f"  Per-CT Ridge: {ridge_perct_c:.4f}")

    # =========================================================================
    # GBT — quick sweep (3 configs) then full eval with best
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  GBT QUICK SWEEP")
    print(f"{'='*80}")

    quick_configs = [
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05,
         "min_child_samples": 30, "num_leaves": 7,
         "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1, "seed": 42},
        {"max_depth": 5, "n_estimators": 300, "learning_rate": 0.03,
         "min_child_samples": 30, "num_leaves": 31,
         "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1, "seed": 42},
        {"max_depth": 7, "n_estimators": 500, "learning_rate": 0.01,
         "min_child_samples": 50, "num_leaves": 63,
         "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1, "seed": 42},
    ]

    best_params = None
    sweep_c = 0.0
    for i, params in enumerate(quick_configs):
        t_sweep = time.time()
        scores_tmp = run_global_gbt(X, times, events, folds, params)
        ci = eval_c_index(scores_tmp, times, events)
        print(f"  Config {i+1}: depth={params['max_depth']} n={params['n_estimators']} "
              f"lr={params['learning_rate']} → C={ci:.4f} [{time.time()-t_sweep:.0f}s]")
        if ci > sweep_c:
            sweep_c = ci
            best_params = params

    print(f"\n  Best: depth={best_params['max_depth']}, n={best_params['n_estimators']}, "
          f"lr={best_params['learning_rate']}")
    print(f"  Best global GBT: {sweep_c:.4f}")

    # =========================================================================
    # Global GBT final
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  GLOBAL GBT (best params)")
    print(f"{'='*80}")

    gbt_global_scores = run_global_gbt(X, times, events, folds, best_params)
    gbt_global_c = eval_c_index(gbt_global_scores, times, events)

    fold_cis = []
    for _, val_idx in folds:
        ci = eval_c_index(gbt_global_scores[val_idx], times[val_idx], events[val_idx])
        fold_cis.append(ci)
    print(f"  Global GBT: {gbt_global_c:.4f}  [{' '.join(f'{c:.4f}' for c in fold_cis)}]")

    # =========================================================================
    # Per-CT GBT
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PER-CT GBT")
    print(f"{'='*80}")

    gbt_perct_scores = run_per_ct_gbt(X, times, events, cts, folds, best_params)
    gbt_perct_c = eval_c_index(gbt_perct_scores, times, events)

    fold_cis_perct = []
    for _, val_idx in folds:
        ci = eval_c_index(gbt_perct_scores[val_idx], times[val_idx], events[val_idx])
        fold_cis_perct.append(ci)
    print(f"  Per-CT GBT: {gbt_perct_c:.4f}  [{' '.join(f'{c:.4f}' for c in fold_cis_perct)}]")

    # =========================================================================
    # Per-CT comparison: GBT vs Ridge
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PER-CT COMPARISON (GBT vs Ridge)")
    print(f"{'='*80}")

    gbt_ct = ct_breakdown(gbt_perct_scores, times, events, cts)
    ridge_ct = ct_breakdown(ridge_perct_scores, times, events, cts)

    print(f"\n  {'Cancer Type':<35} {'N':>6} {'Ridge':>8} {'GBT':>8} {'Δ':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    gbt_wins = 0
    ridge_wins = 0
    for ct_name in sorted(gbt_ct.keys(), key=lambda x: -gbt_ct[x]['n']):
        r = ridge_ct.get(ct_name, {}).get('c', 0.5)
        g = gbt_ct[ct_name]['c']
        n = gbt_ct[ct_name]['n']
        delta = g - r
        marker = " +" if delta > 0.005 else " -" if delta < -0.005 else "  "
        print(f"  {ct_name:<35} {n:>6} {r:>8.4f} {g:>8.4f} {delta:>+8.4f}{marker}")
        if g > r:
            gbt_wins += 1
        else:
            ridge_wins += 1

    print(f"\n  GBT wins: {gbt_wins}, Ridge wins: {ridge_wins}")

    # =========================================================================
    # Feature importance
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  FEATURE IMPORTANCE (LightGBM gain)")
    print(f"{'='*80}")

    final_model = train_gbt_survival(X, times, events, best_params)

    importance = final_model.feature_importances_
    # Get gain-based importance
    final_model_gain = lgb.LGBMRegressor(
        importance_type='gain',
        max_depth=best_params.get("max_depth", 5),
        n_estimators=best_params.get("n_estimators", 300),
        learning_rate=best_params.get("learning_rate", 0.05),
        min_child_samples=best_params.get("min_child_samples", 30),
        num_leaves=best_params.get("num_leaves", 31),
        verbose=-1, random_state=42,
    )
    label_full = np.log1p(times) * (1 - events * 0.3)
    final_model_gain.fit(X, label_full)
    importance = final_model_gain.feature_importances_
    split_imp = final_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print(f"\n  {'Feature':<30} {'Gain':>10} {'Splits':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")
    for j in sorted_idx[:25]:
        name = FEAT_NAMES[j] if j < len(FEAT_NAMES) else f"[{j}]"
        print(f"  {name:<30} {importance[j]:>10.1f} {split_imp[j]:>8}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<25} {'Global C':>10} {'Per-CT C':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'Ridge':<25} {ridge_global_c:>10.4f} {ridge_perct_c:>10.4f}")
    print(f"  {'GBT (global)':<25} {gbt_global_c:>10.4f} {'':>10}")
    print(f"  {'GBT (per-CT)':<25} {'':>10} {gbt_perct_c:>10.4f}")
    print(f"  {'Lift (GBT - Ridge)':<25} {gbt_global_c - ridge_global_c:>+10.4f} {gbt_perct_c - ridge_perct_c:>+10.4f}")

    # Save
    results = {
        'ridge_global_c': float(ridge_global_c),
        'ridge_perct_c': float(ridge_perct_c),
        'gbt_global_c': float(gbt_global_c),
        'gbt_perct_c': float(gbt_perct_c),
        'best_params': {k: v for k, v in best_params.items() if k != 'verbose'},
        'n_patients': N,
        'n_features': int(X.shape[1]),
        'ct_results_gbt': {ct: d for ct, d in gbt_ct.items()},
        'ct_results_ridge': {ct: d for ct, d in ridge_ct.items()},
        'feature_importance': {
            FEAT_NAMES[j] if j < len(FEAT_NAMES) else f"[{j}]": float(importance[j])
            for j in sorted_idx[:30]
        },
    }
    with open(os.path.join(SAVE_BASE, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Saved to {SAVE_BASE}/results.json")


if __name__ == "__main__":
    main()
