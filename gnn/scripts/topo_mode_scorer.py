#!/usr/bin/env python3
"""
Topology-only mode classification: use PPI graph distance alone,
no channel labels.

Mode rules (from graph topology only):
  - distance 1:  RESCUE (direct neighbors, compensatory)
  - distance 2-3: MULTIPLICATIVE (close enough to interact, different function)
  - distance 4+:  INDEPENDENT (far apart, additive)
  - disconnected: INDEPENDENT

Within-channel vs cross-channel is derived from topology, not labels.
Genes in the same connected component share a subnetwork.

Usage:
    python3 -u -m gnn.scripts.topo_mode_scorer
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_TIER_MAP, CHANNEL_FEAT_DIM_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence, fit_per_ct_ridge, precompute_channel_gene_sets,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "topo_mode_scorer")


def topo_mode(dist):
    """Classify interaction mode from PPI distance alone."""
    if dist == 1:
        return 'rescue'
    elif 2 <= dist <= 3:
        return 'multiplicative'
    else:  # dist >= 4 or disconnected (-1)
        return 'independent'


def topo_walk_batch(G_ppi, patient_genes_map, channel_features_all,
                    tier_features_all, ct_per_patient,
                    ct_ch_shift, ct_gene_shift, global_gene_shift,
                    age_all, sex_all, msi_all, msi_high_all, tmb_all,
                    ct_baseline_map, baseline,
                    expanded_channel_map, N, dist_matrix):
    """
    Topology-mode graph walk. No channel labels for mode classification.

    For each patient with mutations:
      1. Get all gene shifts
      2. For each pair, classify mode from PPI distance
      3. Combine:
         - rescue pairs: dampen (average the two shifts)
         - multiplicative pairs: compound ((1+a)(1+b)-1)
         - independent pairs: sum
      4. Aggregate using iterative pairwise combination

    Features (119):
      0: ct_baseline
      1: topo_combined_shift (mode-classified)
      2: linear_shift_sum (for comparison)
      3: n_mutated
      4: n_rescue_pairs
      5: n_mult_pairs
      6: n_indep_pairs
      7: rescue_fraction
      8: mult_fraction
      9: mean_pair_distance
      10: n_components_hit (connected components with mutations)
      11: max_component_muts (most mutations in a single component)
      12: total_isolation
      13: total_hub_damage
      14: tier_conn
      15-17: reserved
      18-89: channel features (72)
      90-109: tier features (20)
      110-114: clinical (5)
    = 115
    """
    all_expanded = set(expanded_channel_map.keys())
    gene_tier = {}
    for g, ch in expanded_channel_map.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    ch_gene_sets = precompute_channel_gene_sets(expanded_channel_map)

    # Precompute connected components of the full PPI graph
    full_components = list(nx.connected_components(G_ppi))
    gene_to_comp = {}
    for ci, comp in enumerate(full_components):
        for g in comp:
            gene_to_comp[g] = ci

    X = np.zeros((N, 115))

    for idx in range(N):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

        patient_genes = patient_genes_map.get(idx, set())
        mutated = sorted(patient_genes & all_expanded)
        ct_name = ct_per_patient[idx]
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        # Clinical + channel/tier always filled
        X[idx, 18:90] = channel_features_all[idx].flatten()
        X[idx, 90:110] = tier_features_all[idx].flatten()
        X[idx, 110] = age_all[idx]
        X[idx, 111] = sex_all[idx]
        X[idx, 112] = msi_all[idx]
        X[idx, 113] = msi_high_all[idx]
        X[idx, 114] = tmb_all[idx]
        X[idx, 0] = ct_bl

        if not mutated:
            continue

        # Per-gene shifts
        gene_shifts = {}
        for g in mutated:
            s = ct_gene_shift.get((ct_name, g), None)
            if s is None:
                s = global_gene_shift.get(g, 0.0)
            gene_shifts[g] = s

        linear_sum = sum(gene_shifts.values())

        # Pairwise mode classification
        n_rescue = 0
        n_mult = 0
        n_indep = 0
        pair_dists = []

        if len(mutated) >= 2:
            for i in range(len(mutated)):
                for j in range(i + 1, len(mutated)):
                    g1, g2 = mutated[i], mutated[j]
                    key = (min(g1, g2), max(g1, g2))
                    dist = dist_matrix.get(key, -1)
                    mode = topo_mode(dist)

                    if mode == 'rescue':
                        n_rescue += 1
                    elif mode == 'multiplicative':
                        n_mult += 1
                    else:
                        n_indep += 1

                    if dist > 0:
                        pair_dists.append(dist)

        total_pairs = n_rescue + n_mult + n_indep

        # ---- Combine shifts using topology modes ----
        # Group genes by connected component
        comp_genes = defaultdict(list)
        for g in mutated:
            c = gene_to_comp.get(g, -1)
            comp_genes[c].append(g)

        # Within each component: combine based on pairwise distances
        comp_shifts = []
        for comp_id, genes in comp_genes.items():
            if len(genes) == 1:
                comp_shifts.append(gene_shifts[genes[0]])
                continue

            # For genes in the same component, combine pairwise
            # Use iterative approach: start with first gene's shift,
            # combine with each subsequent based on distance
            shifts = [gene_shifts[g] for g in genes]

            # Count rescue vs multiplicative within this component
            comp_rescue = 0
            comp_mult = 0
            for i in range(len(genes)):
                for j in range(i + 1, len(genes)):
                    key = (min(genes[i], genes[j]), max(genes[i], genes[j]))
                    dist = dist_matrix.get(key, -1)
                    mode = topo_mode(dist)
                    if mode == 'rescue':
                        comp_rescue += 1
                    elif mode == 'multiplicative':
                        comp_mult += 1

            if comp_rescue > comp_mult:
                # Rescue-dominant component: dampen. Mean shift.
                comp_shifts.append(np.mean(shifts))
            elif comp_mult > 0:
                # Multiplicative-dominant: compound
                product = 1.0
                for s in shifts:
                    product *= (1.0 + s)
                comp_shifts.append(product - 1.0)
            else:
                # All independent within component (far apart)
                comp_shifts.append(sum(shifts))

        # Across components: independent (additive)
        if len(comp_shifts) == 0:
            combined = 0.0
        else:
            combined = sum(comp_shifts)

        # Components hit
        comps_hit = set()
        max_comp_muts = 0
        for comp_id, genes in comp_genes.items():
            if comp_id >= 0:
                comps_hit.add(comp_id)
            if len(genes) > max_comp_muts:
                max_comp_muts = len(genes)

        # Isolation + hub damage (reuse channel-based approach)
        total_isolation = 0.0
        total_hub_damage = 0.0
        mutated_set = set(mutated)
        for ch_name, ch_genes in ch_gene_sets.items():
            ch_mutated = mutated_set & ch_genes
            if not ch_mutated:
                continue
            ch_hubs = HUB_GENES.get(ch_name, set())
            ch_in_ppi = ch_genes & set(G_ppi.nodes())
            if ch_in_ppi:
                G_ch = G_ppi.subgraph(ch_in_ppi - ch_mutated)
                components = list(nx.connected_components(G_ch))
                surviving_hubs = ch_hubs - ch_mutated
                hub_comps = set()
                for ci, comp in enumerate(components):
                    if comp & surviving_hubs:
                        hub_comps.add(ci)
                reachable = set()
                for ci in hub_comps:
                    reachable |= components[ci]
                surviving = ch_in_ppi - ch_mutated
                total_isolation += len(surviving - reachable) / max(len(surviving), 1)
            hub_hit = len(ch_mutated & ch_hubs)
            total_hub_damage += hub_hit / max(len(ch_hubs), 1)

        # Tier connectivity
        ppi_nodes = set(G_ppi.nodes()) - mutated_set
        if ppi_nodes:
            G_d = G_ppi.subgraph(ppi_nodes)
            components_d = list(nx.connected_components(G_d))
            node_comp = {}
            for ci, comp in enumerate(components_d):
                for g in comp:
                    node_comp[g] = ci
            tier_comps = defaultdict(set)
            for g in ppi_nodes:
                t = gene_tier.get(g, -1)
                if t >= 0 and g in node_comp:
                    tier_comps[t].add(node_comp[g])
            tier_connected = 0
            for t1 in range(4):
                for t2 in range(t1 + 1, 4):
                    if tier_comps[t1] & tier_comps[t2]:
                        tier_connected += 1
            tier_conn = tier_connected / 6.0
        else:
            tier_conn = 0.0

        # Assemble features
        X[idx, 0] = ct_bl
        X[idx, 1] = combined  # topo-mode combined shift
        X[idx, 2] = linear_sum  # linear comparison
        X[idx, 3] = len(mutated)
        X[idx, 4] = n_rescue
        X[idx, 5] = n_mult
        X[idx, 6] = n_indep
        X[idx, 7] = n_rescue / max(total_pairs, 1)
        X[idx, 8] = n_mult / max(total_pairs, 1)
        X[idx, 9] = np.mean(pair_dists) if pair_dists else 0
        X[idx, 10] = len(comps_hit)
        X[idx, 11] = max_comp_muts
        X[idx, 12] = total_isolation
        X[idx, 13] = total_hub_damage
        X[idx, 14] = tier_conn

    return X


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  TOPOLOGY-ONLY MODE SCORER")
    print("  PPI distance classifies modes — no channel labels")
    print("=" * 90)

    expanded_cm = load_expanded_channel_map()

    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))

    channel_features = data["channel_features"].numpy()
    tier_features = data["tier_features"].numpy()
    age = data["age"].numpy()
    sex = data["sex"].numpy()
    msi = data["msi_score"].numpy()
    msi_high = data["msi_high"].numpy()
    tmb = data["tmb"].numpy()

    # V6c
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path): continue
        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {k: data[k][idx] for k in [
                "channel_features", "tier_features", "cancer_type_idx",
                "age", "sex", "msi_score", "msi_high", "tmb"]}
            with torch.no_grad(): h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline = hazards[valid_mask].mean()
    print(f"\n  {all_in_val.sum().item()} patients")

    # Mutations
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange", "mutationType"])
    clin = pd.read_csv(os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"), low_memory=False)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["time"] > 0]
    patient_ids = clin["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    expanded_genes = set(expanded_cm.keys())
    mut_exp = mut[mut["gene.hugoGeneSymbol"].isin(expanded_genes)].copy()
    mut_exp["patient_idx"] = mut_exp["patientId"].map(pid_to_idx)
    mut_exp = mut_exp[mut_exp["patient_idx"].notna()]
    mut_exp["patient_idx"] = mut_exp["patient_idx"].astype(int)

    patient_genes = defaultdict(set)
    patient_channels = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            g = row["gene.hugoGeneSymbol"]
            patient_genes[idx].add(g)
            ch = expanded_cm.get(g)
            if ch: patient_channels[idx].add(ch)

    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    # Graph
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    ppi_edges = fetch_string_expanded(expanded_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    print(f"  PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # Shifts
    gene_patients_map = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N: gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)

    global_gene_shift = {}
    for gene, pts in gene_patients_map.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            global_gene_shift[gene] = float(hazards[pts_valid].mean() - baseline)

    ct_patients = defaultdict(set)
    ct_ch_patients = defaultdict(lambda: defaultdict(set))
    ct_gene_patients_map = defaultdict(lambda: defaultdict(set))
    for idx in range(N):
        ct_name = ct_per_patient[idx]
        ct_patients[ct_name].add(idx)
        for ch in patient_channels.get(idx, set()): ct_ch_patients[ct_name][ch].add(idx)
        for g in patient_genes.get(idx, set()): ct_gene_patients_map[ct_name][g].add(idx)

    ct_baseline_map = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 50: ct_baseline_map[ct_name] = float(hazards[pts_valid].mean())

    ct_ch_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for ch_name in set(expanded_cm.values()):
            pts = ct_ch_patients[ct_name].get(ch_name, set())
            pts_arr = np.array(sorted(pts)) if pts else np.array([], dtype=int)
            if len(pts_arr) == 0: continue
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 20:
                ct_ch_shift[(ct_name, ch_name)] = float(hazards[pts_valid].mean() - bl)

    ct_gene_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for gene, pts in ct_gene_patients_map[ct_name].items():
            pts_arr = np.array(sorted(pts))
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 15:
                ct_gene_shift[(ct_name, gene)] = float(hazards[pts_valid].mean() - bl)

    # Distance cache
    print("\n  Precomputing distances...")
    dist_matrix = {}
    all_mutated = set()
    for genes in patient_genes.values():
        all_mutated |= genes
    all_mutated &= set(G_ppi.nodes())
    for source, lengths in nx.all_pairs_shortest_path_length(G_ppi):
        if source not in all_mutated: continue
        for target, dist in lengths.items():
            if target not in all_mutated: continue
            key = (min(source, target), max(source, target))
            if key not in dist_matrix:
                dist_matrix[key] = dist
    print(f"  {len(dist_matrix)} distances cached")

    # Walk
    print("\n  Running topology-mode walk...")
    X_all = topo_walk_batch(
        G_ppi, patient_genes, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, N, dist_matrix)
    print("  Done.")

    # Mode distribution
    valid_idx = np.where(valid_mask)[0]
    print(f"\n{'='*90}")
    print(f"  MODE DISTRIBUTION (topology-only)")
    print(f"{'='*90}")
    rescue_total = X_all[valid_idx, 4].sum()
    mult_total = X_all[valid_idx, 5].sum()
    indep_total = X_all[valid_idx, 6].sum()
    all_total = rescue_total + mult_total + indep_total
    print(f"\n  {'Mode':<20} {'Pairs':>12} {'%':>8} {'Mean/Pt':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
    print(f"  {'Rescue (dist=1)':<20} {rescue_total:>12,.0f} {rescue_total/max(all_total,1)*100:>7.1f}% {X_all[valid_idx,4].mean():>8.2f}")
    print(f"  {'Multiplicative(2-3)':<20} {mult_total:>12,.0f} {mult_total/max(all_total,1)*100:>7.1f}% {X_all[valid_idx,5].mean():>8.2f}")
    print(f"  {'Independent (4+)':<20} {indep_total:>12,.0f} {indep_total/max(all_total,1)*100:>7.1f}% {X_all[valid_idx,6].mean():>8.2f}")

    # Ablation
    graph_idx = list(range(0, 15))
    ch_idx = list(range(18, 90))
    tier_idx = list(range(90, 110))
    clin_idx = list(range(110, 115))

    configs = {
        "A topo-mode graph only":       graph_idx,
        "B topo + clin":                graph_idx + clin_idx,
        "C topo + ch + tier":           graph_idx + ch_idx + tier_idx,
        "D full (topo+ch+tier+clin)":   graph_idx + ch_idx + tier_idx + clin_idx,
        "E ch+tier+clin (no graph)":    ch_idx + tier_idx + clin_idx,
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION")
    print(f"{'='*90}")
    print(f"\n  {'Scorer':<35} {'Mean CI':>8} {'Std':>7}  Folds")
    print(f"  {'-'*35} {'-'*8} {'-'*7}  {'-'*35}")

    results = {}
    for name, feat_indices in configs.items():
        X = X_all[:, feat_indices]
        score = np.zeros(N)
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            vt = valid_mask[train_idx]
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx][vt], hazards[train_idx][vt])
            score[val_idx] = reg.predict(X[val_idx])

        fold_cis = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ci = concordance_index(
                torch.tensor(score[val_idx].astype(np.float32)),
                torch.tensor(times[val_idx].astype(np.float32)),
                torch.tensor(events[val_idx].astype(np.float32)))
            fold_cis.append(ci)

        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        results[name] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis, "score": score}
        folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
        print(f"  {name:<35} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # Per-CT ridge
    print(f"\n  Per-CT ridge...")
    best_feat = configs["D full (topo+ch+tier+clin)"]
    X_best = X_all[:, best_feat]
    score_perct = fit_per_ct_ridge(X_best, hazards, valid_mask, events, times,
                                    ct_per_patient, folds, ct_min_patients=200)
    fold_cis_perct = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        ci = concordance_index(
            torch.tensor(score_perct[val_idx].astype(np.float32)),
            torch.tensor(times[val_idx].astype(np.float32)),
            torch.tensor(events[val_idx].astype(np.float32)))
        fold_cis_perct.append(ci)
    mean_perct = np.mean(fold_cis_perct)
    folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis_perct)
    print(f"  F per-CT ridge (full):         {mean_perct:.4f}  {folds_str}")
    results["F per-CT ridge"] = {"mean": mean_perct, "folds": fold_cis_perct, "score": score_perct}

    # Summary
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    best = max(r["mean"] for r in results.values())
    best_name = max(results, key=lambda k: results[k]["mean"])

    print(f"\n{'='*90}")
    print(f"  COMPARISON")
    print(f"{'='*90}")
    print(f"  V6c transformer:             {v6c:.4f}")
    print(f"  Channel-mode per-CT:         0.6883")
    print(f"  Topo-mode best:              {best:.4f} ({best_name})")
    print(f"  Delta vs channel-mode:       {best - 0.6883:+.4f}")

    # Ridge coefficients for topo features
    print(f"\n  Ridge coefficients (topo-mode features):")
    feat_names = ["ct_baseline", "topo_combined", "linear_sum", "n_mutated",
                  "n_rescue", "n_mult", "n_indep", "rescue_frac", "mult_frac",
                  "mean_dist", "n_comps_hit", "max_comp_muts",
                  "isolation", "hub_damage", "tier_conn"]
    # Fit one global ridge to get coefficients
    X_graph = X_all[:, graph_idx]
    vt = valid_mask
    reg = Ridge(alpha=1.0)
    reg.fit(X_graph[vt], hazards[vt])
    for fi, fname in enumerate(feat_names):
        print(f"    {fname:<20} {reg.coef_[fi]:>+8.4f}")

    # Save
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump({k: {"mean": float(v["mean"])} for k, v in results.items()}, f, indent=2)
    np.save(os.path.join(SAVE_BASE, "X_all.npy"), X_all)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
