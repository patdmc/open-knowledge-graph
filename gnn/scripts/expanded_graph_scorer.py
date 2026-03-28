#!/usr/bin/env python3
"""
Expanded graph scorer: full 503-gene MSK-IMPACT atlas with co-occurrence edges.

Expansion over full_graph_scorer.py:
  1. All 503 MSK-IMPACT genes (vs 243) — uses expanded_channel_map.json
  2. STRING PPI edges for all 503 genes (fresh query, cached)
  3. Co-occurrence edge weights per cancer type
  4. Per-cancer-type ridge fitting (not just global)

Usage:
    python3 -u -m gnn.scripts.expanded_graph_scorer
"""

import sys, os, json, time, numpy as np, networkx as nx, torch, pandas as pd, requests
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_POSITION, GENE_FUNCTION, TRUNCATING,
    STRING_SPECIES, STRING_SCORE_THRESHOLD,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

SAVE_BASE = os.path.join(GNN_RESULTS, "expanded_graph_scorer")
EXPANDED_PPI_CACHE = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")


# =========================================================================
# 1. Build expanded channel map (503 genes)
# =========================================================================
def load_expanded_channel_map():
    """Load the 509-gene expanded channel map, filter to MSK-IMPACT genes."""
    with open(os.path.join(GNN_RESULTS, "expanded_channel_map.json")) as f:
        ecm = json.load(f)
    # channel map: gene -> channel_name
    channel_map = {}
    for gene, info in ecm.items():
        if gene and gene != "nan":
            channel_map[gene] = info["channel"]
    return channel_map


# =========================================================================
# 2. Query STRING for all 503 genes
# =========================================================================
def fetch_string_expanded(genes):
    """Fetch STRING PPI for all MSK-IMPACT genes. Cache result."""
    if os.path.exists(EXPANDED_PPI_CACHE):
        print(f"  Loading cached expanded STRING PPI from {EXPANDED_PPI_CACHE}")
        with open(EXPANDED_PPI_CACHE) as f:
            return json.load(f)

    print(f"  Querying STRING API for {len(genes)} genes...")
    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "%0d".join(sorted(genes)),
        "species": STRING_SPECIES,
        "required_score": STRING_SCORE_THRESHOLD,
        "network_type": "functional",
        "caller_identity": "coupling_channel_gnn",
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    gene_set = set(genes)
    edges = []
    for interaction in data:
        g1 = interaction["preferredName_A"]
        g2 = interaction["preferredName_B"]
        score = interaction["score"]
        if g1 in gene_set and g2 in gene_set:
            edges.append([g1, g2, score])

    print(f"  Retrieved {len(edges)} edges")

    with open(EXPANDED_PPI_CACHE, "w") as f:
        json.dump(edges, f)
    print(f"  Cached to {EXPANDED_PPI_CACHE}")
    return edges


# =========================================================================
# 3. Build expanded PPI graph with co-occurrence edges
# =========================================================================
def build_expanded_graph(channel_map, ppi_edges, cooccurrence):
    """Build NetworkX graph with PPI + co-occurrence edge attributes.
    Returns (G_full, G_ppi) — full graph and PPI-only graph for topology."""
    G = nx.Graph()
    G_ppi = nx.Graph()

    # Add all genes as nodes
    for gene, channel in channel_map.items():
        tier = V6_TIER_MAP.get(channel, -1)
        G.add_node(gene, channel=channel, tier=tier)
        G_ppi.add_node(gene, channel=channel, tier=tier)

    # Add PPI edges
    for g1, g2, score in ppi_edges:
        if g1 in channel_map and g2 in channel_map:
            G.add_edge(g1, g2, ppi_weight=score, cooccurrence={})
            G_ppi.add_edge(g1, g2, weight=score)

    # Add co-occurrence as edge attributes (full graph only)
    for (g1, g2), ct_counts in cooccurrence.items():
        if g1 in channel_map and g2 in channel_map:
            if G.has_edge(g1, g2):
                G[g1][g2]["cooccurrence"] = ct_counts
            else:
                G.add_edge(g1, g2, ppi_weight=0.0, cooccurrence=ct_counts)

    return G, G_ppi


def compute_cooccurrence(patient_genes_map, ct_per_patient, min_count=10):
    """Compute per-cancer-type co-occurrence counts for gene pairs.

    Returns: dict of (g1, g2) -> {ct_name: count}
    Only includes pairs that co-occur >= min_count times in at least one CT.
    """
    # Per cancer type, count co-occurring pairs
    ct_pair_counts = defaultdict(Counter)
    for idx, genes in patient_genes_map.items():
        if len(genes) < 2:
            continue
        ct = ct_per_patient[idx]
        for g1, g2 in combinations(sorted(genes), 2):
            ct_pair_counts[ct][(g1, g2)] += 1

    # Aggregate: for each pair, which CTs have significant co-occurrence
    pair_ct = defaultdict(dict)
    for ct, counts in ct_pair_counts.items():
        for (g1, g2), count in counts.items():
            if count >= min_count:
                pair_ct[(g1, g2)][ct] = count

    return dict(pair_ct)


# =========================================================================
# 4. Expanded graph walk
# =========================================================================
def precompute_channel_gene_sets(expanded_channel_map):
    """Precompute channel -> set of genes (called once, not per-patient)."""
    ch_genes = defaultdict(set)
    for g, ch in expanded_channel_map.items():
        ch_genes[ch].add(g)
    return dict(ch_genes)


def expanded_graph_walk_batch(G_ppi, patient_genes_map, channel_features_all,
                              tier_features_all, ct_per_patient,
                              ct_ch_shift, ct_gene_shift, global_gene_shift,
                              age_all, sex_all, msi_all, msi_high_all, tmb_all,
                              ct_baseline_map, baseline,
                              expanded_channel_map, cooccurrence, N):
    """
    Optimized batch graph walk for all patients.
    Uses connected components instead of per-query path-finding.

    Returns: (N, 111) feature matrix
    """
    ch_gene_sets = precompute_channel_gene_sets(expanded_channel_map)
    EXP_CHANNELS = sorted(ch_gene_sets.keys())
    all_expanded_genes = set(expanded_channel_map.keys())

    # Precompute tier membership
    gene_tier = {}
    for g, ch in expanded_channel_map.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    # Feature dim: 10 graph + 4 cooccur + 72 channel + 20 tier + 5 clinical = 111
    X = np.zeros((N, 111))

    for idx in range(N):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

        patient_genes = patient_genes_map.get(idx, set())
        mutated = patient_genes & all_expanded_genes
        ct_name = ct_per_patient[idx]
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        # --- 1. Graph features ---
        ct_ch_defense_sum = 0.0
        ct_gene_shift_sum = 0.0
        global_gene_shift_sum = 0.0
        n_channels_hit = 0
        total_isolation = 0.0
        total_hub_damage = 0.0
        channels_severed = 0

        for ch_name in EXP_CHANNELS:
            ch_genes = ch_gene_sets[ch_name]
            ch_mutated = mutated & ch_genes
            if not ch_mutated:
                continue

            n_channels_hit += 1
            ct_ch_defense_sum += ct_ch_shift.get((ct_name, ch_name), 0.0)

            for g in ch_mutated:
                ct_s = ct_gene_shift.get((ct_name, g), None)
                if ct_s is not None:
                    ct_gene_shift_sum += ct_s
                gl_s = global_gene_shift.get(g, None)
                if gl_s is not None:
                    global_gene_shift_sum += gl_s

            # Isolation via connected components on PPI subgraph
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
                iso_frac = len(surviving - reachable) / max(len(surviving), 1)
                total_isolation += iso_frac

            hub_hit = len(ch_mutated & ch_hubs)
            total_hub_damage += hub_hit / max(len(ch_hubs), 1)
            if len(ch_mutated) >= 2 or hub_hit > 0:
                channels_severed += 1

        # --- 2. Tier connectivity via connected components ---
        ppi_nodes = set(G_ppi.nodes()) - mutated
        if ppi_nodes:
            G_d = G_ppi.subgraph(ppi_nodes)
            components = list(nx.connected_components(G_d))
            node_comp = {}
            for ci, comp in enumerate(components):
                for g in comp:
                    node_comp[g] = ci

            # Precompute tier -> set of component IDs
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

        # --- 3. Co-occurrence features ---
        n_cooccur_pairs = 0
        total_cooccur_weight = 0.0
        max_cooccur_weight = 0.0
        cross_channel_cooccur = 0

        if len(mutated) >= 2:
            sorted_mutated = sorted(mutated)
            for i in range(len(sorted_mutated)):
                for j in range(i + 1, len(sorted_mutated)):
                    key = (sorted_mutated[i], sorted_mutated[j])
                    ct_counts = cooccurrence.get(key, {})
                    w = ct_counts.get(ct_name, 0)
                    if w > 0:
                        n_cooccur_pairs += 1
                        total_cooccur_weight += w
                        if w > max_cooccur_weight:
                            max_cooccur_weight = w
                        if expanded_channel_map.get(key[0]) != expanded_channel_map.get(key[1]):
                            cross_channel_cooccur += 1

        # --- 4. Assemble ---
        X[idx, 0] = ct_bl
        X[idx, 1] = ct_ch_defense_sum
        X[idx, 2] = ct_gene_shift_sum
        X[idx, 3] = global_gene_shift_sum
        X[idx, 4] = n_channels_hit
        X[idx, 5] = total_isolation
        X[idx, 6] = total_hub_damage
        X[idx, 7] = channels_severed
        X[idx, 8] = tier_conn
        X[idx, 9] = len(mutated)
        X[idx, 10] = n_cooccur_pairs
        X[idx, 11] = total_cooccur_weight
        X[idx, 12] = max_cooccur_weight
        X[idx, 13] = cross_channel_cooccur
        X[idx, 14:86] = channel_features_all[idx].flatten()
        X[idx, 86:106] = tier_features_all[idx].flatten()
        X[idx, 106] = age_all[idx]
        X[idx, 107] = sex_all[idx]
        X[idx, 108] = msi_all[idx]
        X[idx, 109] = msi_high_all[idx]
        X[idx, 110] = tmb_all[idx]

    return X


# =========================================================================
# 5. Per-cancer-type ridge fitting
# =========================================================================
def fit_per_ct_ridge(X, hazards, valid_mask, events, times, ct_per_patient,
                     folds, ct_min_patients=200):
    """Fit separate ridge models per cancer type (with global fallback)."""
    N = len(hazards)
    score = np.zeros(N)

    # Identify CTs with enough patients
    ct_counts = Counter(ct_per_patient[i] for i in range(N) if valid_mask[i])
    large_cts = {ct for ct, n in ct_counts.items() if n >= ct_min_patients}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        vt = valid_mask[train_idx]
        train_valid = train_idx[vt]

        # Global model (fallback)
        global_reg = Ridge(alpha=1.0)
        global_reg.fit(X[train_valid], hazards[train_valid])

        # Per-CT models
        ct_models = {}
        for ct in large_cts:
            ct_train = np.array([i for i in train_valid if ct_per_patient[i] == ct])
            if len(ct_train) >= 50:
                reg = Ridge(alpha=1.0)
                reg.fit(X[ct_train], hazards[ct_train])
                ct_models[ct] = reg

        # Predict
        for i in val_idx:
            ct = ct_per_patient[i]
            if ct in ct_models:
                score[i] = ct_models[ct].predict(X[i:i+1])[0]
            else:
                score[i] = global_reg.predict(X[i:i+1])[0]

    return score


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  EXPANDED GRAPH SCORER")
    print("  503-gene atlas + co-occurrence edges + per-CT ridge")
    print("=" * 90)

    # --- Load expanded channel map ---
    expanded_cm = load_expanded_channel_map()
    print(f"\n  Expanded channel map: {len(expanded_cm)} genes")
    ch_counts = Counter(expanded_cm.values())
    for ch in sorted(ch_counts):
        print(f"    {ch}: {ch_counts[ch]} genes")

    # --- Load data ---
    print("\n  Loading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    channel_features = data["channel_features"].numpy()  # N x 8 x 9
    tier_features = data["tier_features"].numpy()  # N x 4 x 5
    age = data["age"].numpy()
    sex = data["sex"].numpy()
    msi = data["msi_score"].numpy()
    msi_high = data["msi_high"].numpy()
    tmb = data["tmb"].numpy()

    # V6c predictions
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
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
                "age", "sex", "msi_score", "msi_high", "tmb",
            ]}
            with torch.no_grad():
                h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline = hazards[valid_mask].mean()
    print(f"  {all_in_val.sum().item()} patients")

    # --- Load mutations (ALL genes, not just V6) ---
    print("\n  Loading mutations (all 503 genes)...")
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
            if ch:
                patient_channels[idx].add(ch)

    n_with_nodes = sum(1 for i in range(N) if len(patient_genes.get(i, set())) > 0)
    n_v6_only = sum(1 for i in range(N)
                     if len(patient_genes.get(i, set()) & set(V6_CHANNEL_MAP.keys())) > 0)
    print(f"  Patients with expanded graph nodes: {n_with_nodes} ({n_with_nodes/N*100:.1f}%)")
    print(f"  Patients with V6-only nodes:        {n_v6_only} ({n_v6_only/N*100:.1f}%)")
    print(f"  New patients gained:                {n_with_nodes - n_v6_only}")

    # Cancer type per patient
    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    # --- Compute co-occurrence ---
    print("\n  Computing co-occurrence...")
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    total_pairs = len(cooccurrence)
    total_ct_entries = sum(len(v) for v in cooccurrence.values())
    print(f"  Co-occurring pairs: {total_pairs}")
    print(f"  Total CT-specific entries: {total_ct_entries}")

    # Top co-occurring pairs
    top_pairs = sorted(cooccurrence.items(),
                       key=lambda x: max(x[1].values()), reverse=True)[:15]
    print(f"\n  Top 15 co-occurring pairs:")
    print(f"  {'Pair':<30} {'Max CT':>20} {'Count':>6}")
    print(f"  {'-'*30} {'-'*20} {'-'*6}")
    for (g1, g2), ct_counts in top_pairs:
        best_ct = max(ct_counts, key=ct_counts.get)
        print(f"  {g1+'×'+g2:<30} {best_ct:>20} {ct_counts[best_ct]:>6}")

    # --- Query STRING ---
    msk_genes_in_data = set()
    for genes in patient_genes.values():
        msk_genes_in_data |= genes
    msk_genes_in_data &= expanded_genes
    print(f"\n  Genes with mutations in expanded map: {len(msk_genes_in_data)}")

    ppi_edges = fetch_string_expanded(msk_genes_in_data)
    print(f"  PPI edges: {len(ppi_edges)}")

    # --- Build graph ---
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    print(f"  Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  PPI-only graph: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # --- Compute shifts (expanded) ---
    print("\n  Computing gene/channel/CT shifts...")
    gene_patients_map = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)

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
        for ch in patient_channels.get(idx, set()):
            ct_ch_patients[ct_name][ch].add(idx)
        for g in patient_genes.get(idx, set()):
            ct_gene_patients_map[ct_name][g].add(idx)

    ct_baseline_map = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 50:
            ct_baseline_map[ct_name] = float(hazards[pts_valid].mean())

    ct_ch_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for ch_name in set(expanded_cm.values()):
            pts = ct_ch_patients[ct_name].get(ch_name, set())
            pts_arr = np.array(sorted(pts)) if pts else np.array([], dtype=int)
            if len(pts_arr) == 0:
                continue
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

    print(f"  Global gene shifts: {len(global_gene_shift)}")
    print(f"  CT gene shifts: {len(ct_gene_shift)}")
    print(f"  CT channel shifts: {len(ct_ch_shift)}")

    # =========================================================================
    # Graph walk for all patients (optimized batch)
    # =========================================================================
    print("\n  Running expanded graph walk for all patients...")
    print(f"  Feature vector dimension: 111")

    X_all = expanded_graph_walk_batch(
        G_ppi, patient_genes, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, N,
    )
    print(f"    Done.")

    # =========================================================================
    # Feature groups for ablation
    # =========================================================================
    folds = list(skf.split(np.arange(N), events))

    # Feature indices
    # graph_feats: 0-9 (10)
    # cooccur_feats: 10-13 (4)
    # channel_feats: 14-85 (72)
    # tier_feats: 86-105 (20)
    # clinical: 106-110 (5)
    graph_idx = list(range(0, 10))
    cooccur_idx = list(range(10, 14))
    ch_idx = list(range(14, 86))
    tier_idx = list(range(86, 106))
    clin_idx = list(range(106, 111))

    configs = {
        "A graph only (expanded)":       graph_idx,
        "B graph + cooccur":             graph_idx + cooccur_idx,
        "C graph + cooccur + clin":      graph_idx + cooccur_idx + clin_idx,
        "D graph + cooccur + ch + tier": graph_idx + cooccur_idx + ch_idx + tier_idx,
        "E full (graph+co+ch+ti+clin)":  graph_idx + cooccur_idx + ch_idx + tier_idx + clin_idx,
        "F ch + tier + clin (no graph)": ch_idx + tier_idx + clin_idx,
        "G prev 243-gene equivalent":    graph_idx + ch_idx + tier_idx + clin_idx,  # no cooccur
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION: EXPANDED GRAPH (503 genes + co-occurrence)")
    print(f"{'='*90}")

    print(f"\n  {'Scorer':<40} {'Mean CI':>8} {'Std':>7}  Folds")
    print(f"  {'-'*40} {'-'*8} {'-'*7}  {'-'*35}")

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
        print(f"  {name:<40} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # =========================================================================
    # Per-CT ridge fitting (best config)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE RIDGE FITTING")
    print(f"{'='*90}")

    best_config = "E full (graph+co+ch+ti+clin)"
    best_indices = configs[best_config]
    X_best = X_all[:, best_indices]

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
    std_perct = np.std(fold_cis_perct)
    folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis_perct)
    print(f"\n  Per-CT ridge (full features):          {mean_perct:.4f} +/- {std_perct:.4f}")
    print(f"  Folds: {folds_str}")
    results["H per-CT ridge (full)"] = {
        "mean": mean_perct, "std": std_perct, "folds": fold_cis_perct, "score": score_perct
    }

    # =========================================================================
    # Summary
    # =========================================================================
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    best_overall = max(r["mean"] for r in results.values())
    best_name = max(results, key=lambda k: results[k]["mean"])
    prev_best = 0.6673  # from full_graph_scorer.py (243 genes)

    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"\n  V6c transformer:                       {v6c:.4f}")
    print(f"  Previous best (243-gene):              {prev_best:.4f}")
    print(f"  New best ({best_name}):  {best_overall:.4f}")
    print(f"  Improvement from expansion:            {best_overall - prev_best:+.4f}")
    print(f"  Gap to V6c:                            {v6c - best_overall:+.4f}")
    print(f"  Gap closed (from 0.538 baseline):      {(best_overall - 0.5381) / (v6c - 0.5381) * 100:.1f}%")

    # =========================================================================
    # Per-cancer-type comparison (best scorer)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE: {best_name}")
    print(f"{'='*90}")

    best_score = results[best_name]["score"]

    print(f"\n  {'Cancer Type':<35} {'N':>5} {'Graph':>7} {'V6c':>7} {'Delta':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    ct_results = {}
    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 100 or events[ct_indices].sum() < 10:
            continue

        ci_graph = concordance_index(
            torch.tensor(best_score[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v6c = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        delta = ci_graph - ci_v6c
        marker = " <-- BEATS" if delta > 0.005 else ""
        ct_results[ct_name] = {"n": len(ct_indices), "graph": float(ci_graph),
                                "v6c": float(ci_v6c), "delta": float(delta)}
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_graph:>7.4f} {ci_v6c:>7.4f} "
              f"{delta:>+7.4f}{marker}")

    # Co-occurrence impact on high co-occurrence CTs
    print(f"\n  HIGH CO-OCCURRENCE CANCER TYPES:")
    high_cooccur_cts = ["Colorectal Cancer", "Endometrial Cancer", "Bladder Cancer",
                        "Small Cell Lung Cancer"]
    for ct in high_cooccur_cts:
        if ct in ct_results:
            r = ct_results[ct]
            print(f"    {ct}: graph={r['graph']:.4f} v6c={r['v6c']:.4f} delta={r['delta']:+.4f}")

    # Save
    save_results = {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                    for k, v in results.items()}
    save_results["per_cancer_type"] = ct_results
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
