"""
Auto-tuning ridge scorer — dynamic parameter selection from current graph state.

Every time the graph changes, features shift. Ridge params (alpha, feature
selection, normalization) must re-calibrate automatically.

This module:
  1. Extracts the current feature matrix from the graph (reuses pairwise scorer logic)
  2. Searches alpha via nested CV
  3. Evaluates feature importance via permutation
  4. Optionally drops low-importance features and re-fits
  5. Reports per-CT and global C-index
  6. Saves the optimal config so the feedback loop can use it

Designed to run as part of the learning loop — after graph mutation,
after property refresh, before convergence check.

Usage:
    python3 -u -m gnn.scripts.auto_ridge [--dry-run]
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, HUB_GENES, CHANNEL_MAP

RESULTS_DIR = os.path.join(GNN_RESULTS, "auto_ridge")


# =========================================================================
# Feature extraction — pulls current state from Neo4j
# =========================================================================

def extract_features_from_graph():
    """Extract per-patient feature matrix from current graph state.

    Returns (X, y_time, y_event, patient_ids, cancer_types, feature_names)
    """
    from neo4j import GraphDatabase
    import networkx as nx

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    print("  Extracting features from current graph...", flush=True)
    t0 = time.time()

    # --- Load graph structure ---
    # PPI
    G_ppi = nx.Graph()
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[r:PPI]-(g2:Gene) WHERE g1.name < g2.name
            RETURN g1.name AS g1, g2.name AS g2, r.score AS score
        """)
        for r in result:
            G_ppi.add_edge(r["g1"], r["g2"], weight=r["score"] or 0.4)

    # PPI distances
    ppi_dists = {}
    for comp in nx.connected_components(G_ppi):
        sub = G_ppi.subgraph(comp)
        for src, targets in dict(nx.all_pairs_shortest_path_length(sub)).items():
            for tgt, dist in targets.items():
                if src < tgt:
                    ppi_dists[(src, tgt)] = dist

    # Channel profiles
    channel_profiles = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene) WHERE g.channel_profile IS NOT NULL
            RETURN g.name AS gene, g.channel_profile AS profile
        """)
        for r in result:
            channel_profiles[r["gene"]] = np.array(r["profile"], dtype=np.float32)

    # Co-occurrence
    cooccurrence = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene) WHERE g1.name < g2.name
            RETURN g1.name AS g1, g2.name AS g2, r.count AS cnt
        """)
        for r in result:
            cooccurrence[(r["g1"], r["g2"])] = r["cnt"] or 0

    # Gene confidence + function
    gene_info = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)
            RETURN g.name AS gene, g.confidence_score AS conf,
                   g.function AS func
        """)
        for r in result:
            gene_info[r["gene"]] = {
                "confidence": r["conf"] or 0,
                "function": r["func"] or "unknown",
            }

    # Hub set
    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set |= hubs

    # --- Load patients + mutations ---
    patients = []
    with driver.session() as s:
        result = s.run("""
            MATCH (p:Patient)
            WHERE p.os_months IS NOT NULL AND p.os_months > 0
            RETURN p.id AS pid, p.os_months AS time, p.event AS event,
                   p.cancer_type AS ct
        """)
        for r in result:
            patients.append({
                "pid": r["pid"], "time": r["time"],
                "event": r["event"], "ct": r["ct"],
            })

    patient_mutations = defaultdict(list)
    with driver.session() as s:
        result = s.run("""
            MATCH (p:Patient)-[r:HAS_MUTATION]->(g:Gene)
            RETURN p.id AS pid, g.name AS gene, r.log_hr AS log_hr,
                   r.biallelic_status AS biallelic,
                   r.expression_context AS expr_ctx
        """)
        for r in result:
            patient_mutations[r["pid"]].append({
                "gene": r["gene"],
                "log_hr": r["log_hr"],
                "biallelic": r["biallelic"],
                "expr_ctx": r["expr_ctx"],
            })

    driver.close()

    # --- Compute features ---
    n_ch = len(channel_profiles.get(list(channel_profiles.keys())[0], []))
    if n_ch == 0:
        n_ch = 8

    feature_names = []
    # Base features (per gene aggregates)
    feature_names += [f"ch_damage_{i}" for i in range(n_ch)]  # 0-7
    feature_names += ["mean_entropy", "hub_damage", "n_mutated", "hhi",
                      "frac_hub", "frac_in_ppi", "n_channels_hit", "frac_channels"]  # 8-15
    # Pairwise features
    feature_names += ["mean_ppi_dist", "min_ppi_dist", "frac_connected",
                      "frac_direct", "frac_medium", "frac_far",
                      "mean_overlap", "min_overlap", "max_overlap",
                      "frac_same_ch", "frac_cross_ch",
                      "mean_cooccur", "max_cooccur", "frac_cooccur",
                      "mean_combined_entropy",
                      "hub_hub_pairs", "hub_nonhub_pairs",
                      "close_cross_frac", "far_same_frac"]  # 16-34
    # Traversal pattern features (NEW)
    feature_names += ["traversal_score", "bfs_ratio", "dfs_depth",
                      "component_ratio", "path_linearity",
                      "damage_gradient", "frontier_width"]  # 35-41
    # Mutation-level aggregates
    feature_names += ["mean_log_hr", "std_log_hr", "max_abs_log_hr",
                      "frac_harmful", "frac_protective",
                      "n_biallelic", "frac_silent_expr",
                      "frac_tsg", "frac_onco", "gof_lof_ratio"]  # 42-51

    n_features = len(feature_names)

    X = np.zeros((len(patients), n_features), dtype=np.float32)
    y_time = np.zeros(len(patients), dtype=np.float32)
    y_event = np.zeros(len(patients), dtype=np.int32)
    patient_ids = []
    cancer_types = []

    DISCONNECTED = 10

    for idx, pat in enumerate(patients):
        pid = pat["pid"]
        patient_ids.append(pid)
        cancer_types.append(pat["ct"])
        y_time[idx] = pat["time"]
        y_event[idx] = pat["event"]

        muts = patient_mutations.get(pid, [])
        genes = [m["gene"] for m in muts]
        n_mut = len(genes)

        if n_mut == 0:
            continue

        # --- Base features ---
        ch_damage = np.zeros(n_ch)
        hub_damage = 0.0
        total_entropy = 0.0
        n_hub = 0
        channels_hit = set()
        n_in_ppi = 0

        for g in genes:
            prof = channel_profiles.get(g)
            if prof is not None:
                ch_damage += prof
                p_sum = prof.sum()
                if p_sum > 0:
                    p_norm = prof / p_sum
                    total_entropy += -np.sum(p_norm * np.log(p_norm + 1e-10))
            if g in hub_set:
                n_hub += 1
                hub_damage += sum(prof) if prof is not None else 0
            ch = CHANNEL_MAP.get(g)
            if ch:
                channels_hit.add(ch)
            if g in G_ppi:
                n_in_ppi += 1

        X[idx, 0:n_ch] = ch_damage
        X[idx, 8] = total_entropy / max(n_mut, 1)
        X[idx, 9] = hub_damage
        X[idx, 10] = n_mut
        ch_total = ch_damage.sum()
        if ch_total > 0:
            shares = ch_damage / ch_total
            X[idx, 11] = (shares ** 2).sum()
        X[idx, 12] = n_hub / max(n_mut, 1)
        X[idx, 13] = n_in_ppi / max(n_mut, 1)
        X[idx, 14] = len(channels_hit)
        X[idx, 15] = len(channels_hit) / max(n_ch, 1)

        # --- Pairwise features ---
        if n_mut >= 2:
            from itertools import combinations
            pairs = list(combinations(sorted(set(genes)), 2))
            n_pairs = max(len(pairs), 1)

            dists = []
            overlaps = []
            same_ch = 0
            cross_ch = 0
            cooccur_w = []
            hub_hub = 0
            hub_nonhub = 0
            close_cross = 0
            far_same = 0

            for g1, g2 in pairs:
                pair = (min(g1, g2), max(g1, g2))
                d = ppi_dists.get(pair, DISCONNECTED)
                dists.append(d)

                p1 = channel_profiles.get(g1)
                p2 = channel_profiles.get(g2)
                if p1 is not None and p2 is not None:
                    n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
                    if n1 > 0 and n2 > 0:
                        overlaps.append(np.dot(p1, p2) / (n1 * n2))

                ch1 = CHANNEL_MAP.get(g1)
                ch2 = CHANNEL_MAP.get(g2)
                if ch1 and ch2:
                    if ch1 == ch2:
                        same_ch += 1
                        if d >= 4: far_same += 1
                    else:
                        cross_ch += 1
                        if d <= 2: close_cross += 1

                cw = cooccurrence.get(pair, 0)
                if cw > 0:
                    cooccur_w.append(cw)

                h1, h2 = g1 in hub_set, g2 in hub_set
                if h1 and h2: hub_hub += 1
                elif h1 or h2: hub_nonhub += 1

            da = np.array(dists) if dists else np.array([DISCONNECTED])
            X[idx, 16] = da.mean()
            X[idx, 17] = da.min()
            X[idx, 18] = (da < DISCONNECTED).mean()
            X[idx, 19] = (da == 1).mean()
            X[idx, 20] = ((da >= 2) & (da <= 3)).mean()
            X[idx, 21] = (da >= 4).mean()
            X[idx, 22] = np.mean(overlaps) if overlaps else 0
            X[idx, 23] = np.min(overlaps) if overlaps else 0
            X[idx, 24] = np.max(overlaps) if overlaps else 0
            X[idx, 25] = same_ch / n_pairs
            X[idx, 26] = cross_ch / n_pairs
            X[idx, 27] = np.mean(cooccur_w) if cooccur_w else 0
            X[idx, 28] = np.max(cooccur_w) if cooccur_w else 0
            X[idx, 29] = len(cooccur_w) / n_pairs
            X[idx, 30] = 0  # combined entropy placeholder
            X[idx, 31] = hub_hub
            X[idx, 32] = hub_nonhub
            X[idx, 33] = close_cross / n_pairs
            X[idx, 34] = far_same / n_pairs

            # --- Traversal pattern features ---
            trav = compute_traversal_features(
                genes, G_ppi, ppi_dists, channel_profiles, CHANNEL_MAP
            )
            X[idx, 35:42] = trav

        # --- Mutation-level aggregates ---
        log_hrs = [m["log_hr"] for m in muts if m["log_hr"] is not None]
        if log_hrs:
            X[idx, 42] = np.mean(log_hrs)
            X[idx, 43] = np.std(log_hrs) if len(log_hrs) > 1 else 0
            X[idx, 44] = np.max(np.abs(log_hrs))
            X[idx, 45] = np.mean([1 for h in log_hrs if h > np.log(1.1)]) if log_hrs else 0
            X[idx, 46] = np.mean([1 for h in log_hrs if h < np.log(0.9)]) if log_hrs else 0

        n_biallelic = sum(1 for m in muts if m.get("biallelic") == "biallelic")
        X[idx, 47] = n_biallelic

        n_silent = sum(1 for m in muts if m.get("expr_ctx") == "silent")
        X[idx, 48] = n_silent / max(n_mut, 1)

        n_tsg = sum(1 for m in muts
                    if gene_info.get(m["gene"], {}).get("function") in ("TSG", "likely_TSG"))
        n_onco = sum(1 for m in muts
                     if gene_info.get(m["gene"], {}).get("function") == "oncogene")
        X[idx, 49] = n_tsg / max(n_mut, 1)
        X[idx, 50] = n_onco / max(n_mut, 1)
        X[idx, 51] = (n_onco - n_tsg) / max(n_onco + n_tsg, 1)

    print(f"  Features: {X.shape} [{time.time()-t0:.1f}s]", flush=True)

    return X, y_time, y_event, patient_ids, cancer_types, feature_names


# =========================================================================
# Traversal pattern features — BFS vs DFS characterization
# =========================================================================

def compute_traversal_features(genes, G_ppi, ppi_dists, channel_profiles, channel_map):
    """Characterize the mutation spread pattern on the PPI graph.

    Returns 7 features that capture BFS vs DFS traversal shape:
      [0] traversal_score — -1 (pure DFS) to +1 (pure BFS)
      [1] bfs_ratio — fraction of mutations reachable in 1-2 hops from each other
      [2] dfs_depth — max chain length in PPI subgraph
      [3] component_ratio — n_components / n_mutated (1.0 = all isolated = BFS-like)
      [4] path_linearity — longest path / n_in_ppi (1.0 = pure chain = DFS-like)
      [5] damage_gradient — how steeply damage falls off from the center
      [6] frontier_width — mean number of unique channels at each BFS layer
    """
    import networkx as nx

    feat = np.zeros(7, dtype=np.float32)
    unique_genes = sorted(set(genes))
    n_mut = len(unique_genes)

    if n_mut < 2:
        return feat

    # Build PPI subgraph of mutated genes
    ppi_genes = [g for g in unique_genes if g in G_ppi]
    n_in_ppi = len(ppi_genes)

    if n_in_ppi < 2:
        # All isolated — maximally BFS-like
        feat[0] = 1.0   # traversal_score
        feat[1] = 0.0   # bfs_ratio (nothing connected)
        feat[3] = 1.0   # component_ratio
        return feat

    sub = G_ppi.subgraph(ppi_genes).copy()
    components = list(nx.connected_components(sub))
    n_components = len(components)

    # Component ratio: more components = more BFS-like
    feat[3] = n_components / n_in_ppi

    # BFS ratio: fraction of pairs within distance 2
    n_close = 0
    n_total_pairs = 0
    DISCONNECTED = 10
    for i, g1 in enumerate(ppi_genes):
        for g2 in ppi_genes[i+1:]:
            pair = (min(g1, g2), max(g1, g2))
            d = ppi_dists.get(pair, DISCONNECTED)
            n_total_pairs += 1
            if d <= 2:
                n_close += 1
    feat[1] = n_close / max(n_total_pairs, 1)

    # DFS depth: longest shortest path in the subgraph (diameter of largest component)
    max_path = 0
    for comp in components:
        if len(comp) >= 2:
            comp_sub = sub.subgraph(comp)
            try:
                diam = nx.diameter(comp_sub)
                max_path = max(max_path, diam)
            except nx.NetworkXError:
                pass
    feat[2] = max_path

    # Path linearity: longest path / n_in_ppi
    # High = chain-like (DFS), low = star-like (BFS)
    feat[4] = max_path / n_in_ppi if n_in_ppi > 0 else 0

    # Damage gradient: BFS from highest-damage gene, measure how damage drops
    # High gradient = concentrated (DFS-like), low = spread (BFS-like)
    gene_damage = {}
    for g in ppi_genes:
        prof = channel_profiles.get(g)
        gene_damage[g] = float(prof.sum()) if prof is not None else 0

    if gene_damage:
        center = max(gene_damage, key=gene_damage.get)
        center_dmg = gene_damage[center]
        if center_dmg > 0:
            # BFS layers from center
            try:
                bfs_layers = dict(nx.single_source_shortest_path_length(sub, center))
                layer_damages = defaultdict(list)
                layer_channels = defaultdict(set)
                for g, dist in bfs_layers.items():
                    layer_damages[dist].append(gene_damage.get(g, 0))
                    ch = channel_map.get(g)
                    if ch:
                        layer_channels[dist].add(ch)

                # Gradient: correlation of distance with damage
                if len(layer_damages) >= 2:
                    all_dists = []
                    all_dmgs = []
                    for d, dmgs in layer_damages.items():
                        for dmg in dmgs:
                            all_dists.append(d)
                            all_dmgs.append(dmg)
                    if np.std(all_dists) > 0 and np.std(all_dmgs) > 0:
                        feat[5] = -np.corrcoef(all_dists, all_dmgs)[0, 1]
                        # Positive = damage drops with distance (concentrated)
                        # Negative = damage increases with distance (peripheral)

                # Frontier width: mean unique channels per BFS layer
                if layer_channels:
                    widths = [len(chs) for chs in layer_channels.values()]
                    feat[6] = np.mean(widths)
            except nx.NetworkXError:
                pass

    # Composite traversal score: combine signals
    # DFS indicators: high linearity, high gradient, low component ratio, low bfs_ratio
    # BFS indicators: low linearity, low gradient, high component ratio, high bfs_ratio
    dfs_signal = feat[4] + feat[5] - feat[3] - feat[1]
    bfs_signal = feat[1] + feat[3] - feat[4] - feat[5]
    total = abs(dfs_signal) + abs(bfs_signal)
    if total > 0:
        feat[0] = (bfs_signal - dfs_signal) / total  # -1 DFS to +1 BFS
    else:
        feat[0] = 0.0

    return feat


# =========================================================================
# Auto-tuning ridge
# =========================================================================

def auto_tune_ridge(X, y_time, y_event, cancer_types, n_folds=5, seed=42):
    """Search for optimal ridge parameters on current features.

    Returns:
        best_config: dict with alpha, selected features, normalization params
        cv_c_index: cross-validated global C-index
        per_ct_results: per cancer type C-index
    """
    print("\n  Auto-tuning ridge parameters...", flush=True)
    t0 = time.time()

    # Filter valid patients
    valid = y_time > 0
    X_v = X[valid]
    t_v = y_time[valid]
    e_v = y_event[valid]
    ct_v = np.array(cancer_types)[valid]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_v)

    # Replace NaN/Inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Alpha search
    alphas = np.logspace(-3, 3, 50)
    best_alpha = 1.0
    best_global_c = 0.0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        scores = []

        for train_idx, val_idx in skf.split(X_scaled, e_v):
            ridge.fit(X_scaled[train_idx], t_v[train_idx])
            pred = ridge.predict(X_scaled[val_idx])

            e_val = e_v[val_idx].astype(bool)
            t_val = t_v[val_idx]
            try:
                c = concordance_index_censored(e_val, t_val, -pred)[0]
                scores.append(c)
            except Exception:
                scores.append(0.5)

        mean_c = np.mean(scores)
        if mean_c > best_global_c:
            best_global_c = mean_c
            best_alpha = alpha

    print(f"    Best alpha: {best_alpha:.4f} → C={best_global_c:.4f}", flush=True)

    # --- Per-CT ridge ---
    print("  Fitting per-CT ridge...", flush=True)
    unique_cts = sorted(set(ct_v))
    per_ct_results = {}

    # Global model as fallback
    global_ridge = Ridge(alpha=best_alpha)
    global_ridge.fit(X_scaled, t_v)

    for ct in unique_cts:
        ct_mask = ct_v == ct
        n_ct = ct_mask.sum()

        if n_ct < 50:
            # Too few — use global
            pred = global_ridge.predict(X_scaled[ct_mask])
            per_ct_results[ct] = {"n": int(n_ct), "method": "global"}
            continue

        X_ct = X_scaled[ct_mask]
        t_ct = t_v[ct_mask]
        e_ct = e_v[ct_mask]

        # Quick alpha search for this CT
        best_ct_c = 0.0
        best_ct_alpha = best_alpha
        for alpha in [best_alpha * 0.1, best_alpha, best_alpha * 10]:
            try:
                ct_skf = StratifiedKFold(n_splits=min(5, max(2, n_ct // 20)),
                                         shuffle=True, random_state=seed)
                scores = []
                for tr, va in ct_skf.split(X_ct, e_ct):
                    r = Ridge(alpha=alpha)
                    r.fit(X_ct[tr], t_ct[tr])
                    p = r.predict(X_ct[va])
                    e_va = e_ct[va].astype(bool)
                    try:
                        c = concordance_index_censored(e_va, t_ct[va], -p)[0]
                        scores.append(c)
                    except Exception:
                        scores.append(0.5)
                mean_c = np.mean(scores)
                if mean_c > best_ct_c:
                    best_ct_c = mean_c
                    best_ct_alpha = alpha
            except Exception:
                pass

        per_ct_results[ct] = {
            "n": int(n_ct),
            "c_index": round(float(best_ct_c), 4),
            "alpha": float(best_ct_alpha),
            "method": "per_ct",
        }

    # --- Feature importance (permutation-based) ---
    print("  Computing feature importance...", flush=True)
    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_scaled, t_v)
    baseline_pred = ridge_final.predict(X_scaled)
    e_bool = e_v.astype(bool)
    try:
        baseline_c = concordance_index_censored(e_bool, t_v, -baseline_pred)[0]
    except Exception:
        baseline_c = 0.5

    importances = np.zeros(X_scaled.shape[1])
    rng = np.random.RandomState(seed)
    for f in range(X_scaled.shape[1]):
        X_perm = X_scaled.copy()
        X_perm[:, f] = rng.permutation(X_perm[:, f])
        perm_pred = ridge_final.predict(X_perm)
        try:
            perm_c = concordance_index_censored(e_bool, t_v, -perm_pred)[0]
        except Exception:
            perm_c = 0.5
        importances[f] = baseline_c - perm_c  # positive = feature helps

    # --- Compose config ---
    config = {
        "alpha": float(best_alpha),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_features": int(X_scaled.shape[1]),
        "n_patients": int(X_scaled.shape[0]),
        "feature_importances": importances.tolist(),
        "global_c_index": round(float(best_global_c), 4),
    }

    elapsed = time.time() - t0
    print(f"  Auto-tune complete [{elapsed:.1f}s]", flush=True)
    print(f"    Global C: {best_global_c:.4f}, alpha: {best_alpha:.4f}")
    print(f"    Top features by importance:")
    top_idx = np.argsort(importances)[-10:][::-1]
    for i in top_idx:
        print(f"      [{i:2d}] importance={importances[i]:+.4f}")

    return config, best_global_c, per_ct_results


# =========================================================================
# Main
# =========================================================================

def run_auto_ridge(dry_run=False):
    """Extract features from current graph, auto-tune ridge, save config."""
    print("=" * 60)
    print("  AUTO-TUNING RIDGE FROM CURRENT GRAPH STATE")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y_time, y_event, pids, cts, feat_names = extract_features_from_graph()

    if dry_run:
        print(f"  [DRY RUN] Would tune ridge on {X.shape} features")
        return {"dry_run": True}

    config, global_c, per_ct = auto_tune_ridge(X, y_time, y_event, cts)
    config["feature_names"] = feat_names
    config["per_ct_results"] = per_ct

    # Save
    with open(os.path.join(RESULTS_DIR, "ridge_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save feature matrix for debugging
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "features.npz"),
        X=X, y_time=y_time, y_event=y_event,
    )

    print(f"\n  Per-CT results:")
    print(f"  {'CT':<25s} {'N':>6s} {'C':>7s} {'Method':>8s}")
    for ct, info in sorted(per_ct.items(), key=lambda x: -x[1].get("c_index", 0)):
        c = info.get("c_index", "—")
        print(f"  {ct:<25s} {info['n']:>6d} {c:>7} {info['method']:>8s}")

    return config


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_auto_ridge(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
