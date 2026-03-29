#!/usr/bin/env python3
"""
Graph traversal scoring: simulate cascade propagation through the PPI graph.

Instead of weighting mutations linearly, TRAVERSE the graph from each mutated
node and compute:
  1. How many paths are blocked by this mutation
  2. How many alternative paths remain
  3. What fraction of the channel is still reachable from each hub
  4. Whether the damage propagates across channel/tier boundaries

Then: per-cancer-type breakdown of the traversal score.

Usage:
    python3 -u -m gnn.scripts.graph_traversal_scorer
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES, GENE_POSITION,
    GENE_FUNCTION, TRUNCATING,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "graph_traversal")


def build_full_graph():
    """Build the PPI graph with channel/tier attributes."""
    with open(os.path.join(GNN_CACHE, "string_ppi_edges.json")) as f:
        ppi = json.load(f)

    G = nx.Graph()
    for gene in V6_CHANNEL_MAP:
        ch = V6_CHANNEL_MAP[gene]
        G.add_node(gene, channel=ch, tier=V6_TIER_MAP.get(ch, -1),
                   position=GENE_POSITION.get(gene, "unk"),
                   function=GENE_FUNCTION.get(gene, "unk"))

    for ch_name in ppi:
        for src, tgt, score in ppi[ch_name]:
            if src in V6_CHANNEL_MAP and tgt in V6_CHANNEL_MAP:
                G.add_edge(src, tgt, weight=score / 1000.0)
    return G


def cascade_damage(G, mutated_genes, gene_shifts):
    """
    Simulate cascade propagation from mutated nodes.

    For each channel subgraph:
      1. Remove mutated nodes (they're "severed")
      2. Check what fraction of the channel is still reachable from each hub
      3. Compute the "isolation score" — how fragmented the channel becomes

    For cross-channel:
      4. Check if cross-channel hub-to-hub paths are still intact
      5. Compute tier reachability — can lower tiers still escalate to upper?

    Returns a feature vector per patient.
    """
    mutated = set(mutated_genes)

    features = {}

    # --- Per-channel damage ---
    total_isolation = 0.0
    total_hub_damage = 0.0
    channels_severed = 0
    hub_shift_sum = 0.0

    for ch_name in V6_CHANNEL_NAMES:
        ch_genes = [g for g, c in V6_CHANNEL_MAP.items() if c == ch_name]
        ch_hubs = HUB_GENES.get(ch_name, set())
        ch_mutated = mutated & set(ch_genes)

        if not ch_mutated:
            continue

        # Build channel subgraph
        G_ch = G.subgraph(ch_genes).copy()
        n_before = G_ch.number_of_nodes()

        # Remove mutated nodes
        G_damaged = G_ch.copy()
        G_damaged.remove_nodes_from(ch_mutated)

        # Fragmentation: how many components now vs before?
        comp_before = nx.number_connected_components(G_ch) if n_before > 0 else 1
        comp_after = nx.number_connected_components(G_damaged) if G_damaged.number_of_nodes() > 0 else n_before

        # What fraction of non-mutated genes are reachable from surviving hubs?
        surviving_hubs = ch_hubs - ch_mutated
        reachable_from_hubs = set()
        for hub in surviving_hubs:
            if hub in G_damaged:
                reachable_from_hubs |= set(nx.descendants(G_damaged, hub)) | {hub}

        surviving_genes = set(ch_genes) - ch_mutated
        if surviving_genes:
            reachability = len(reachable_from_hubs & surviving_genes) / len(surviving_genes)
        else:
            reachability = 0.0

        # Hub damage: are hubs themselves mutated?
        hub_hit = len(ch_mutated & ch_hubs)
        hub_total = len(ch_hubs) if ch_hubs else 1

        # Isolation: genes cut off from all hubs
        isolated = surviving_genes - reachable_from_hubs
        isolation_frac = len(isolated) / max(len(surviving_genes), 1)

        is_severed = len(ch_mutated) >= 2 or hub_hit > 0

        total_isolation += isolation_frac
        total_hub_damage += hub_hit / hub_total
        if is_severed:
            channels_severed += 1

        # Sum shifts for mutated genes in this channel
        for g in ch_mutated:
            if g in gene_shifts:
                hub_shift_sum += gene_shifts[g]

    # --- Cross-channel / tier escalation ---
    # Can signals still escalate from tier 0 to tier 3?
    G_damaged_full = G.copy()
    G_damaged_full.remove_nodes_from(mutated)

    # Tier connectivity: for each pair of adjacent tiers,
    # are there still cross-channel paths?
    tier_genes = defaultdict(set)
    for g, ch in V6_CHANNEL_MAP.items():
        if g not in mutated:
            tier_genes[V6_TIER_MAP.get(ch, -1)].add(g)

    tier_connected = 0
    tier_pairs_tested = 0
    for t1 in range(4):
        for t2 in range(t1 + 1, 4):
            # Is there any path from a tier t1 gene to a tier t2 gene?
            found_path = False
            for g1 in list(tier_genes.get(t1, set()))[:5]:  # sample for speed
                for g2 in list(tier_genes.get(t2, set()))[:5]:
                    if g1 in G_damaged_full and g2 in G_damaged_full:
                        if nx.has_path(G_damaged_full, g1, g2):
                            found_path = True
                            break
                if found_path:
                    break
            tier_pairs_tested += 1
            if found_path:
                tier_connected += 1

    tier_connectivity = tier_connected / max(tier_pairs_tested, 1)

    # Largest connected component after damage
    if G_damaged_full.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G_damaged_full), key=len)
        lcc_frac = len(largest_cc) / G.number_of_nodes()
    else:
        lcc_frac = 0.0

    features = {
        "total_isolation": total_isolation,
        "total_hub_damage": total_hub_damage,
        "channels_severed": channels_severed,
        "hub_shift_sum": hub_shift_sum,
        "tier_connectivity": tier_connectivity,
        "lcc_fraction": lcc_frac,
        "n_mutated": len(mutated & set(V6_CHANNEL_MAP.keys())),
    }
    return features


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  GRAPH TRAVERSAL SCORER")
    print("  Simulate cascade propagation through PPI graph")
    print("=" * 90)

    G = build_full_graph()
    print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load data
    print("  Loading data + predictions...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(data["cancer_type_vocab"]),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {k: data[k][idx] for k in [
                "channel_features", "tier_features", "cancer_type_idx",
                "age", "sex", "msi_score", "msi_high", "tmb"
            ]}
            with torch.no_grad():
                h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline = hazards[valid_mask].mean()
    print(f"  {all_in_val.sum().item()} patients")

    # Load mutations
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange", "mutationType"],
    )
    clin = pd.read_csv(os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"), low_memory=False)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["time"] > 0]
    patient_ids = clin["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mut[mut["gene.hugoGeneSymbol"].isin(channel_genes)].copy()
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)

    # Per-patient gene sets
    patient_genes = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes[idx].add(row["gene.hugoGeneSymbol"])

    # Gene-level shifts
    gene_patients_map = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)

    gene_shift = {}
    for gene, pts in gene_patients_map.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            gene_shift[gene] = float(hazards[pts_valid].mean() - baseline)

    # Cancer type info
    cancer_type_idx = data["cancer_type_idx"].numpy()
    ct_vocab = data["cancer_type_vocab"]

    # =========================================================================
    # Compute traversal features for each patient
    # =========================================================================
    print("\n  Computing traversal features for each patient...")

    feature_names = [
        "total_isolation", "total_hub_damage", "channels_severed",
        "hub_shift_sum", "tier_connectivity", "lcc_fraction", "n_mutated",
    ]
    patient_features = np.zeros((N, len(feature_names)))

    # Process in batches for speed reporting
    for idx in range(N):
        genes = patient_genes.get(idx, set())
        if not genes:
            patient_features[idx, 5] = 1.0  # lcc_fraction = 1 (no damage)
            patient_features[idx, 4] = 1.0  # tier_connectivity = 1
            continue
        feats = cascade_damage(G, genes, gene_shift)
        for fi, fname in enumerate(feature_names):
            patient_features[idx, fi] = feats.get(fname, 0)

        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

    print(f"    Done. {N} patients processed.")

    # =========================================================================
    # Correlate traversal features with hazard
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TRAVERSAL FEATURES vs ACTUAL HAZARD")
    print("=" * 90)

    valid_idx = np.where(valid_mask)[0]

    print(f"\n  {'Feature':<25} {'r':>8} {'p':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*12}")
    for fi, fname in enumerate(feature_names):
        feat_vals = patient_features[valid_idx, fi]
        haz_vals = hazards[valid_idx]
        if np.std(feat_vals) > 0:
            r, p = stats.pearsonr(feat_vals, haz_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {fname:<25} {r:>+8.4f} {p:>12.2e} {sig}")

    # =========================================================================
    # Build scoring functions
    # =========================================================================
    print("\n" + "=" * 90)
    print("  SCORING FUNCTIONS")
    print("=" * 90)

    folds = list(skf.split(np.arange(N), events))
    scores = {}

    # A: gene shift sum (baseline)
    score_gene = np.zeros(N)
    for idx, genes in patient_genes.items():
        for g in genes:
            if g in gene_shift:
                score_gene[idx] += gene_shift[g]
    scores["A_gene_shift"] = score_gene

    # B: traversal isolation score
    # Higher isolation = more damage = higher hazard
    scores["B_isolation"] = patient_features[:, 0]  # total_isolation

    # C: traversal composite
    # Combine: isolation (bad) + hub_damage (bad) - tier_connectivity (good)
    scores["C_traversal_composite"] = (
        patient_features[:, 0] * 1.0  # isolation
        + patient_features[:, 1] * 1.0  # hub_damage
        - patient_features[:, 4] * 0.5  # tier_connectivity (protective)
        + patient_features[:, 2] * 0.2  # channels_severed
    )

    # D: gene shift + traversal correction
    score_d = score_gene.copy()
    score_d += patient_features[:, 0] * 0.3  # isolation adds harm
    score_d -= patient_features[:, 4] * 0.2  # tier connectivity reduces harm
    score_d += patient_features[:, 1] * 0.2  # hub damage adds harm
    scores["D_gene+traversal"] = score_d

    # E: learned weights via per-fold regression
    from sklearn.linear_model import Ridge

    score_e = np.zeros(N)
    all_X = np.column_stack([
        score_gene,
        patient_features,
    ])
    feature_labels = ["gene_shift"] + feature_names

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train = all_X[train_idx]
        y_train = hazards[train_idx]
        w_train = valid_mask[train_idx].astype(float)

        # Only train on valid patients
        train_valid = w_train > 0
        reg = Ridge(alpha=1.0)
        reg.fit(X_train[train_valid], y_train[train_valid])
        score_e[val_idx] = reg.predict(all_X[val_idx])

    scores["E_ridge_traversal"] = score_e

    # Evaluate
    print(f"\n  {'Scorer':<30} {'Mean CI':>8} {'Std':>7}  Fold CIs")
    print(f"  {'-'*30} {'-'*8} {'-'*7}  {'-'*40}")

    results = {}
    for name in ["A_gene_shift", "B_isolation", "C_traversal_composite",
                  "D_gene+traversal", "E_ridge_traversal"]:
        s = scores[name]
        fold_cis = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ci = concordance_index(
                torch.tensor(s[val_idx].astype(np.float32)),
                torch.tensor(times[val_idx].astype(np.float32)),
                torch.tensor(events[val_idx].astype(np.float32)),
            )
            fold_cis.append(ci)
        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        results[name] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis}
        folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
        print(f"  {name:<30} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # Print ridge coefficients from last fold
    print(f"\n  Ridge coefficients (last fold):")
    for fi, fname in enumerate(feature_labels):
        print(f"    {fname:<25} {reg.coef_[fi]:>+8.4f}")

    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    if os.path.exists(ua_path):
        with open(ua_path) as f:
            ua = json.load(f)
        v6c_mean = ua["means"].get("V6c", 0)
        best = max(r["mean"] for r in results.values())
        best_name = max(results, key=lambda k: results[k]["mean"])
        print(f"\n  V6c transformer:     {v6c_mean:.4f}")
        print(f"  Best graph scorer:   {best:.4f} ({best_name})")
        print(f"  Gap:                 {v6c_mean - best:+.4f}")

    # =========================================================================
    # PER-CANCER-TYPE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 90)
    print("  PER-CANCER-TYPE GRAPH TRAVERSAL ANALYSIS")
    print("  How does graph damage pattern differ by cancer type?")
    print("=" * 90)

    # For each cancer type: mean traversal features + mean hazard + scoring performance
    ct_counts = defaultdict(int)
    ct_features = defaultdict(lambda: np.zeros(len(feature_names)))
    ct_hazards = defaultdict(list)
    ct_gene_shift = defaultdict(list)
    ct_valid = defaultdict(int)

    for idx in range(N):
        ct_idx = int(cancer_type_idx[idx])
        ct_name = ct_vocab[ct_idx] if ct_idx < len(ct_vocab) else "Other"
        ct_counts[ct_name] += 1
        ct_features[ct_name] += patient_features[idx]
        ct_gene_shift[ct_name].append(score_gene[idx])
        if valid_mask[idx]:
            ct_hazards[ct_name].append(hazards[idx])
            ct_valid[ct_name] += 1

    # Per-cancer-type C-index for the best scorer
    best_scorer = scores[max(results, key=lambda k: results[k]["mean"])]

    print(f"\n  {'Cancer Type':<30} {'N':>6} {'MeanHaz':>8} {'Isolation':>9} {'HubDmg':>7} "
          f"{'ChSev':>5} {'TierConn':>8} {'LCC':>5} {'GeneShift':>9}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*9} {'-'*7} "
          f"{'-'*5} {'-'*8} {'-'*5} {'-'*9}")

    ct_rows = []
    for ct_name in sorted(ct_counts, key=lambda x: -ct_counts[x]):
        n = ct_counts[ct_name]
        if n < 100:
            continue
        mean_feats = ct_features[ct_name] / n
        mean_haz = np.mean(ct_hazards[ct_name]) if ct_hazards[ct_name] else 0
        mean_gs = np.mean(ct_gene_shift[ct_name])

        ct_rows.append({
            "name": ct_name, "n": n, "mean_haz": mean_haz,
            "isolation": mean_feats[0], "hub_damage": mean_feats[1],
            "channels_severed": mean_feats[2], "tier_connectivity": mean_feats[4],
            "lcc_fraction": mean_feats[5], "gene_shift": mean_gs,
        })

        print(f"  {ct_name:<30} {n:>6} {mean_haz:>+8.3f} {mean_feats[0]:>9.3f} "
              f"{mean_feats[1]:>7.3f} {mean_feats[2]:>5.1f} {mean_feats[4]:>8.3f} "
              f"{mean_feats[5]:>5.3f} {mean_gs:>+9.3f}")

    # =========================================================================
    # Per-cancer-type C-index
    # =========================================================================
    print(f"\n  Per-cancer-type C-index (ridge traversal scorer vs V6c model):")
    print(f"  {'Cancer Type':<30} {'N_valid':>7} {'Ridge':>7} {'V6c':>7} {'Delta':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for ct_name in sorted(ct_counts, key=lambda x: -ct_valid.get(x, 0)):
        nv = ct_valid.get(ct_name, 0)
        if nv < 100:
            continue

        ct_mask = np.array([
            (int(cancer_type_idx[i]) < len(ct_vocab) and ct_vocab[int(cancer_type_idx[i])] == ct_name)
            for i in range(N)
        ])
        ct_valid_mask = ct_mask & valid_mask
        ct_indices = np.where(ct_valid_mask)[0]

        if len(ct_indices) < 50:
            continue

        ct_events = events[ct_indices]
        if ct_events.sum() < 10:
            continue

        # Ridge traversal C-index for this cancer type
        ci_ridge = concordance_index(
            torch.tensor(score_e[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(ct_events.astype(np.float32)),
        )

        # V6c C-index for this cancer type
        ci_v6c = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(ct_events.astype(np.float32)),
        )

        delta = ci_ridge - ci_v6c
        marker = " <-- BEATS V6c" if delta > 0.01 else ""
        print(f"  {ct_name:<30} {len(ct_indices):>7} {ci_ridge:>7.4f} {ci_v6c:>7.4f} "
              f"{delta:>+7.4f}{marker}")

    # =========================================================================
    # Save
    # =========================================================================
    with open(os.path.join(SAVE_BASE, "scoring_results.json"), "w") as f:
        json.dump({k: {"mean": v["mean"], "std": v["std"]} for k, v in results.items()},
                  f, indent=2)

    with open(os.path.join(SAVE_BASE, "cancer_type_profiles.json"), "w") as f:
        json.dump(ct_rows, f, indent=2)

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
