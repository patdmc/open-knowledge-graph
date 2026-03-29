#!/usr/bin/env python3
"""
Mode-classified graph scorer: replace linear edge weights with
discrete interaction modes.

Four modes for each pair of mutated genes:
  1. RESCUE     — distance-1 neighbors, same channel. Dampen.
  2. REDUNDANT  — same channel, distance > 1. Max, don't sum.
  3. MULTIPLICATIVE — cross-channel, connected. Compound.
  4. INDEPENDENT — disconnected. Additive.

The mode determines how gene shifts combine, not a learned weight.

Usage:
    python3 -u -m gnn.scripts.mode_graph_scorer
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence, fit_per_ct_ridge, precompute_channel_gene_sets,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "mode_graph_scorer")


def classify_pair(g1, g2, channel_map, G_ppi, dist_cache):
    """Classify interaction mode for a gene pair.

    Returns: 'rescue', 'redundant', 'multiplicative', 'independent'
    """
    ch1 = channel_map.get(g1)
    ch2 = channel_map.get(g2)
    same_channel = ch1 == ch2

    # Get distance
    key = (min(g1, g2), max(g1, g2))
    if key in dist_cache:
        dist = dist_cache[key]
    else:
        if g1 in G_ppi and g2 in G_ppi:
            try:
                dist = nx.shortest_path_length(G_ppi, g1, g2)
            except nx.NetworkXNoPath:
                dist = -1  # disconnected
        else:
            dist = -1
        dist_cache[key] = dist

    if dist == -1:
        return 'independent'
    elif same_channel and dist == 1:
        return 'rescue'
    elif same_channel and dist > 1:
        return 'redundant'
    else:  # cross-channel, connected
        return 'multiplicative'


def combine_shifts(shifts, modes):
    """Combine a list of gene shifts using mode-classified rules.

    For each pair, the mode determines how to combine:
      - rescue: dampen (take weighted average, not sum)
      - redundant: take max absolute, don't sum
      - multiplicative: compound (shift_a + shift_b + shift_a * shift_b)
      - independent: additive

    Strategy: process all pairs, accumulate combined score.
    For a patient with N mutations, we process the N shifts using pair modes.

    Approach: channel-wise aggregation first, then cross-channel combination.
    """
    if len(shifts) == 0:
        return 0.0
    if len(shifts) == 1:
        return shifts[0]

    # Group shifts by mode with their partner
    # For multi-mutation patients, we need a principled aggregation.
    #
    # Within-channel (rescue/redundant):
    #   - Rescue pairs: take the WEAKER shift (biology compensates)
    #   - Redundant pairs: take the STRONGER shift (parallel, don't stack)
    # Cross-channel (multiplicative):
    #   - Compound: product of (1 + shift) - 1
    # Independent:
    #   - Sum
    #
    # Simple approach: for each pair, compute the PAIR combination,
    # then return the aggregated result.
    #
    # But this gets O(n^2) complex for many mutations. Instead:
    # Aggregate by channel using rescue/redundant rules,
    # then combine channels using multiplicative/independent rules.

    return None  # placeholder — we do this in the walk function


def mode_graph_walk_batch(G_ppi, patient_genes_map, channel_features_all,
                          tier_features_all, ct_per_patient,
                          ct_ch_shift, ct_gene_shift, global_gene_shift,
                          age_all, sex_all, msi_all, msi_high_all, tmb_all,
                          ct_baseline_map, baseline,
                          expanded_channel_map, cooccurrence, N,
                          dist_cache):
    """
    Mode-classified graph walk.

    Per patient:
      1. For each channel with mutations:
         a. Classify pairs within channel (rescue/redundant)
         b. Aggregate channel shift: rescue=dampen, redundant=max
      2. Across channels:
         a. Classify channel-to-channel mode (multiplicative/independent)
         b. Combine: multiplicative=compound, independent=sum
      3. Mode counts as features (how many of each type)
      4. Standard graph features (isolation, hub damage, tier conn)
      5. Channel/tier/clinical features

    Feature vector: 10 graph + 8 mode + 72 channel + 20 tier + 5 clinical = 115
    """
    ch_gene_sets = precompute_channel_gene_sets(expanded_channel_map)
    EXP_CHANNELS = sorted(ch_gene_sets.keys())
    all_expanded = set(expanded_channel_map.keys())

    gene_tier = {}
    for g, ch in expanded_channel_map.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    # Precompute per-channel representative genes for cross-channel distance
    ch_representatives = {}
    for ch, genes in ch_gene_sets.items():
        # Pick genes with highest PPI degree as channel reps
        degs = [(g, G_ppi.degree(g) if g in G_ppi else 0) for g in genes]
        degs.sort(key=lambda x: -x[1])
        ch_representatives[ch] = [g for g, d in degs[:3] if d > 0]

    X = np.zeros((N, 115))

    for idx in range(N):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

        patient_genes = patient_genes_map.get(idx, set())
        mutated = patient_genes & all_expanded
        ct_name = ct_per_patient[idx]
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        if not mutated:
            # No mutations: only clinical + ct_baseline
            X[idx, 0] = ct_bl
            X[idx, 18:90] = channel_features_all[idx].flatten()
            X[idx, 90:110] = tier_features_all[idx].flatten()
            X[idx, 110] = age_all[idx]
            X[idx, 111] = sex_all[idx]
            X[idx, 112] = msi_all[idx]
            X[idx, 113] = msi_high_all[idx]
            X[idx, 114] = tmb_all[idx]
            continue

        # ---- Per-channel aggregation ----
        channel_agg_shifts = {}  # channel -> aggregated shift
        n_rescue = 0
        n_redundant = 0
        n_multiplicative = 0
        n_independent = 0
        total_isolation = 0.0
        total_hub_damage = 0.0
        channels_severed = 0
        ct_ch_defense_sum = 0.0
        n_channels_hit = 0

        for ch_name in EXP_CHANNELS:
            ch_genes = ch_gene_sets[ch_name]
            ch_mutated = sorted(mutated & ch_genes)

            if not ch_mutated:
                continue

            n_channels_hit += 1
            ct_ch_defense_sum += ct_ch_shift.get((ct_name, ch_name), 0.0)

            # Get per-gene shifts
            gene_shifts = []
            for g in ch_mutated:
                s = ct_gene_shift.get((ct_name, g), None)
                if s is None:
                    s = global_gene_shift.get(g, 0.0)
                gene_shifts.append(s)

            # Classify within-channel pairs
            if len(ch_mutated) == 1:
                channel_agg_shifts[ch_name] = gene_shifts[0]
            else:
                # For each pair, classify mode
                pair_modes = []
                for i in range(len(ch_mutated)):
                    for j in range(i + 1, len(ch_mutated)):
                        mode = classify_pair(ch_mutated[i], ch_mutated[j],
                                            expanded_channel_map, G_ppi, dist_cache)
                        pair_modes.append(mode)
                        if mode == 'rescue':
                            n_rescue += 1
                        elif mode == 'redundant':
                            n_redundant += 1

                # Aggregate within channel:
                # If any pair is rescue -> dampen (mean of shifts)
                # If all pairs redundant -> max shift
                # Mixed -> weighted
                rescue_count = sum(1 for m in pair_modes if m == 'rescue')
                redundant_count = sum(1 for m in pair_modes if m == 'redundant')

                if rescue_count > redundant_count:
                    # Rescue dominant: biology compensates. Take mean (dampened).
                    channel_agg_shifts[ch_name] = np.mean(gene_shifts)
                elif redundant_count > 0:
                    # Redundant dominant: parallel hits. Take strongest.
                    max_idx = np.argmax(np.abs(gene_shifts))
                    channel_agg_shifts[ch_name] = gene_shifts[max_idx]
                else:
                    # Default: sum (shouldn't happen within-channel often)
                    channel_agg_shifts[ch_name] = sum(gene_shifts)

            # Isolation (same as before — connected components on PPI)
            ch_hubs = HUB_GENES.get(ch_name, set())
            ch_in_ppi = ch_genes & set(G_ppi.nodes())
            ch_mut_set = set(ch_mutated)
            if ch_in_ppi:
                G_ch = G_ppi.subgraph(ch_in_ppi - ch_mut_set)
                components = list(nx.connected_components(G_ch))
                surviving_hubs = ch_hubs - ch_mut_set
                hub_comps = set()
                for ci, comp in enumerate(components):
                    if comp & surviving_hubs:
                        hub_comps.add(ci)
                reachable = set()
                for ci in hub_comps:
                    reachable |= components[ci]
                surviving = ch_in_ppi - ch_mut_set
                total_isolation += len(surviving - reachable) / max(len(surviving), 1)

            hub_hit = len(ch_mut_set & ch_hubs)
            total_hub_damage += hub_hit / max(len(ch_hubs), 1)
            if len(ch_mutated) >= 2 or hub_hit > 0:
                channels_severed += 1

        # ---- Cross-channel combination ----
        hit_channels = sorted(channel_agg_shifts.keys())

        if len(hit_channels) == 0:
            combined_shift = 0.0
        elif len(hit_channels) == 1:
            combined_shift = list(channel_agg_shifts.values())[0]
        else:
            # Classify cross-channel pairs
            for i in range(len(hit_channels)):
                for j in range(i + 1, len(hit_channels)):
                    ch_a, ch_b = hit_channels[i], hit_channels[j]
                    # Cross-channel mode: check if any gene in ch_a connects to any in ch_b
                    connected = False
                    mutated_a = sorted(mutated & ch_gene_sets[ch_a])
                    mutated_b = sorted(mutated & ch_gene_sets[ch_b])
                    for ga in mutated_a[:3]:
                        for gb in mutated_b[:3]:
                            key = (min(ga, gb), max(ga, gb))
                            if key in dist_cache:
                                d = dist_cache[key]
                            else:
                                if ga in G_ppi and gb in G_ppi:
                                    try:
                                        d = nx.shortest_path_length(G_ppi, ga, gb)
                                    except nx.NetworkXNoPath:
                                        d = -1
                                else:
                                    d = -1
                                dist_cache[key] = d
                            if d > 0:
                                connected = True
                                break
                        if connected:
                            break

                    if connected:
                        n_multiplicative += 1
                    else:
                        n_independent += 1

            # Combine channel shifts using modes:
            # Multiplicative channels: compound product
            # Independent channels: additive
            # Simple approach: compound all shifts as (1 + s_a)(1 + s_b) - 1
            # This naturally handles multiplicative interaction
            # For truly independent channels, additive is fine
            # The ratio of multiplicative to independent pairs tells us which to use

            total_pairs = n_multiplicative + n_independent
            mult_frac = n_multiplicative / max(total_pairs, 1)

            if mult_frac > 0.5:
                # Predominantly multiplicative: compound
                product = 1.0
                for ch in hit_channels:
                    product *= (1.0 + channel_agg_shifts[ch])
                combined_shift = product - 1.0
            else:
                # Predominantly independent: additive
                combined_shift = sum(channel_agg_shifts.values())

        # ---- Tier connectivity ----
        ppi_nodes = set(G_ppi.nodes()) - mutated
        if ppi_nodes:
            G_d = G_ppi.subgraph(ppi_nodes)
            components = list(nx.connected_components(G_d))
            node_comp = {}
            for ci, comp in enumerate(components):
                for g in comp:
                    node_comp[g] = ci
            tier_comps = defaultdict(set)
            for g in ppi_nodes:
                t = gene_tier.get(g, -1)
                if t >= 0 and g in node_comp:
                    tier_comps[t].add(node_comp[g])
            tier_connected = 0
            tier_tested = 0
            for t1 in range(4):
                for t2 in range(t1 + 1, 4):
                    tier_tested += 1
                    if tier_comps[t1] & tier_comps[t2]:
                        tier_connected += 1
            tier_conn = tier_connected / max(tier_tested, 1)
        else:
            tier_conn = 0.0

        # ---- Assemble ----
        # 10 graph features
        X[idx, 0] = ct_bl
        X[idx, 1] = ct_ch_defense_sum
        X[idx, 2] = combined_shift  # MODE-CLASSIFIED combined shift
        X[idx, 3] = sum(global_gene_shift.get(g, 0) for g in mutated)  # raw sum for comparison
        X[idx, 4] = n_channels_hit
        X[idx, 5] = total_isolation
        X[idx, 6] = total_hub_damage
        X[idx, 7] = channels_severed
        X[idx, 8] = tier_conn
        X[idx, 9] = len(mutated)

        # 8 mode features
        X[idx, 10] = n_rescue
        X[idx, 11] = n_redundant
        X[idx, 12] = n_multiplicative
        X[idx, 13] = n_independent
        total_mode = n_rescue + n_redundant + n_multiplicative + n_independent
        X[idx, 14] = n_rescue / max(total_mode, 1)       # rescue fraction
        X[idx, 15] = n_multiplicative / max(total_mode, 1) # multiplicative fraction
        X[idx, 16] = combined_shift  # duplicate for ablation clarity
        X[idx, 17] = sum(channel_agg_shifts.values()) if channel_agg_shifts else 0  # additive for comparison

        # 72 channel + 20 tier + 5 clinical
        X[idx, 18:90] = channel_features_all[idx].flatten()
        X[idx, 90:110] = tier_features_all[idx].flatten()
        X[idx, 110] = age_all[idx]
        X[idx, 111] = sex_all[idx]
        X[idx, 112] = msi_all[idx]
        X[idx, 113] = msi_high_all[idx]
        X[idx, 114] = tmb_all[idx]

    return X


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  MODE-CLASSIFIED GRAPH SCORER")
    print("  Four interaction modes: rescue, redundant, multiplicative, independent")
    print("=" * 90)

    expanded_cm = load_expanded_channel_map()
    print(f"\n  Expanded channel map: {len(expanded_cm)} genes")

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
                "age", "sex", "msi_score", "msi_high", "tmb",
            ]}
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
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange", "mutationType"],
    )
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
    print(f"  PPI graph: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

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

    # Precompute distance cache
    print("\n  Precomputing all-pairs shortest paths...")
    dist_cache = {}
    # Only compute for genes that are actually mutated in patients
    all_mutated_genes = set()
    for genes in patient_genes.values():
        all_mutated_genes |= genes
    all_mutated_genes &= set(G_ppi.nodes())

    # Use NetworkX all_pairs_shortest_path_length for efficiency
    for source, lengths in nx.all_pairs_shortest_path_length(G_ppi):
        if source not in all_mutated_genes:
            continue
        for target, dist in lengths.items():
            if target not in all_mutated_genes:
                continue
            key = (min(source, target), max(source, target))
            if key not in dist_cache:
                dist_cache[key] = dist
    print(f"  Cached {len(dist_cache)} pairwise distances")

    # Graph walk
    print("\n  Running mode-classified graph walk...")
    X_all = mode_graph_walk_batch(
        G_ppi, patient_genes, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, N, dist_cache,
    )
    print("  Done.")

    # =========================================================================
    # Mode distribution
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  INTERACTION MODE DISTRIBUTION")
    print(f"{'='*90}")

    valid_idx = np.where(valid_mask)[0]
    rescue_counts = X_all[valid_idx, 10]
    redundant_counts = X_all[valid_idx, 11]
    mult_counts = X_all[valid_idx, 12]
    indep_counts = X_all[valid_idx, 13]

    total_rescue = rescue_counts.sum()
    total_redundant = redundant_counts.sum()
    total_mult = mult_counts.sum()
    total_indep = indep_counts.sum()
    total_all = total_rescue + total_redundant + total_mult + total_indep

    print(f"\n  {'Mode':<20} {'Total Pairs':>12} {'%':>8} {'Mean/Patient':>13}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*13}")
    print(f"  {'Rescue':<20} {total_rescue:>12,.0f} {total_rescue/max(total_all,1)*100:>7.1f}% {rescue_counts.mean():>13.2f}")
    print(f"  {'Redundant':<20} {total_redundant:>12,.0f} {total_redundant/max(total_all,1)*100:>7.1f}% {redundant_counts.mean():>13.2f}")
    print(f"  {'Multiplicative':<20} {total_mult:>12,.0f} {total_mult/max(total_all,1)*100:>7.1f}% {mult_counts.mean():>13.2f}")
    print(f"  {'Independent':<20} {total_indep:>12,.0f} {total_indep/max(total_all,1)*100:>7.1f}% {indep_counts.mean():>13.2f}")

    # =========================================================================
    # Ablation
    # =========================================================================
    # Feature indices:
    # 0-9: graph features (with mode-combined shift at [2])
    # 10-17: mode features
    # 18-89: channel features (72)
    # 90-109: tier features (20)
    # 110-114: clinical (5)
    graph_idx = list(range(0, 10))
    mode_idx = list(range(10, 18))
    ch_idx = list(range(18, 90))
    tier_idx = list(range(90, 110))
    clin_idx = list(range(110, 115))

    configs = {
        "A mode graph only":            graph_idx + mode_idx,
        "B mode graph + clin":          graph_idx + mode_idx + clin_idx,
        "C mode graph + ch + tier":     graph_idx + mode_idx + ch_idx + tier_idx,
        "D full (mode+ch+tier+clin)":   graph_idx + mode_idx + ch_idx + tier_idx + clin_idx,
        "E ch+tier+clin (no graph)":    ch_idx + tier_idx + clin_idx,
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION: MODE-CLASSIFIED GRAPH")
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
                torch.tensor(events[val_idx].astype(np.float32)),
            )
            fold_cis.append(ci)

        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        results[name] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis, "score": score}
        folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
        print(f"  {name:<35} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # Per-CT ridge on best config
    print(f"\n  Fitting per-CT ridge on full features...")
    best_feat = configs["D full (mode+ch+tier+clin)"]
    X_best = X_all[:, best_feat]
    score_perct = fit_per_ct_ridge(X_best, hazards, valid_mask, events, times,
                                    ct_per_patient, folds, ct_min_patients=200)

    fold_cis_perct = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        ci = concordance_index(
            torch.tensor(score_perct[val_idx].astype(np.float32)),
            torch.tensor(times[val_idx].astype(np.float32)),
            torch.tensor(events[val_idx].astype(np.float32)),
        )
        fold_cis_perct.append(ci)

    mean_perct = np.mean(fold_cis_perct)
    folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis_perct)
    print(f"  Per-CT ridge (mode+all):       {mean_perct:.4f}  {folds_str}")
    results["F per-CT ridge (full)"] = {"mean": mean_perct, "folds": fold_cis_perct, "score": score_perct}

    # =========================================================================
    # Comparison
    # =========================================================================
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    prev_linear = 0.6681  # expanded_graph_scorer E config
    prev_perct = 0.6881   # expanded_graph_scorer per-CT ridge

    best = max(r["mean"] for r in results.values())
    best_name = max(results, key=lambda k: results[k]["mean"])

    print(f"\n{'='*90}")
    print(f"  COMPARISON")
    print(f"{'='*90}")
    print(f"\n  V6c transformer:                   {v6c:.4f}")
    print(f"  Previous linear (expanded):        {prev_linear:.4f}")
    print(f"  Previous per-CT (expanded):        {prev_perct:.4f}")
    print(f"  Mode-classified best:              {best:.4f} ({best_name})")
    print(f"  Delta vs previous per-CT:          {best - prev_perct:+.4f}")
    print(f"  Gap to V6c:                        {v6c - best:+.4f}")

    # Per-cancer-type for best
    print(f"\n  Per-cancer-type (top 15):")
    print(f"  {'Cancer Type':<35} {'N':>5} {'Mode':>6} {'V6c':>6} {'Delta':>6}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")

    best_score = results[best_name]["score"]
    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x]))[:15]:
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 100 or events[ct_indices].sum() < 10: continue

        ci_g = concordance_index(
            torch.tensor(best_score[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)))
        ci_v = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)))
        delta = ci_g - ci_v
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_g:>6.4f} {ci_v:>6.4f} {delta:>+6.4f}")

    # Save
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump({k: {"mean": float(v["mean"])} for k, v in results.items()}, f, indent=2)
    np.save(os.path.join(SAVE_BASE, "X_all.npy"), X_all)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
