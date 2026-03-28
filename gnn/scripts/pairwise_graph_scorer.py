#!/usr/bin/env python3
"""
Pairwise graph scorer: features computed from gene PAIRS, not just per-gene sums.

The key insight: aggregating per-gene profiles into patient sums throws away
interaction structure. Two patients with identical channel damage sums can
have completely different mutation topologies on the PPI graph.

For each patient, we compute:
  1. Per-gene features (curated-anchor profiles, hub status)
  2. Pairwise features for all co-mutated gene pairs:
     - PPI distance
     - Channel profile overlap (cosine similarity)
     - Same/cross channel primary
     - Co-occurrence weight
     - Combined profile entropy
  3. N-wise (component) features:
     - Connected components among mutated genes
     - Largest component size, number of components
     - Intra-component vs inter-component damage distribution

These pairwise/n-wise statistics give a linear readout access to interaction
structure that was previously invisible.

Usage:
    python3 -u -m gnn.scripts.pairwise_graph_scorer
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_FUNCTION, TRUNCATING,
)
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
    compute_cooccurrence, fit_per_ct_ridge,
)
from gnn.scripts.focused_multichannel_scorer import (
    compute_curated_profiles, profile_entropy,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "pairwise_graph_scorer")
CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}


# =========================================================================
# Precompute pairwise distances on PPI graph
# =========================================================================
def precompute_ppi_distances(G_ppi, gene_set):
    """All-pairs shortest paths among genes in gene_set that are in PPI.
    Returns dict of (g1, g2) -> distance. Missing = disconnected."""
    ppi_genes = gene_set & set(G_ppi.nodes())
    dists = {}
    # Use BFS from each gene (efficient for sparse graph)
    for g in ppi_genes:
        lengths = nx.single_source_shortest_path_length(G_ppi, g)
        for g2, d in lengths.items():
            if g2 in ppi_genes and g != g2:
                key = tuple(sorted([g, g2]))
                if key not in dists:
                    dists[key] = d
    return dists


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


# =========================================================================
# Pairwise + N-wise graph walk
# =========================================================================
def pairwise_graph_walk_batch(
    G_ppi, patient_genes_map, channel_profiles, ppi_dists,
    channel_features_all, tier_features_all, ct_per_patient,
    ct_ch_shift, ct_gene_shift, global_gene_shift,
    age_all, sex_all, msi_all, msi_high_all, tmb_all,
    ct_baseline_map, baseline, expanded_cm, cooccurrence,
    hub_gene_set, N,
):
    """
    Feature vector per patient:

    BASE FEATURES (from focused scorer, 17):
      [0]     ct_baseline
      [1]     weighted_ch_shift
      [2]     ct_gene_shift_sum
      [3]     global_gene_shift_sum
      [4-11]  channel_damage (8-dim curated-anchor)
      [12]    total_entropy
      [13]    hub_damage
      [14]    tier_conn
      [15]    n_mutated
      [16]    hhi (damage concentration)

    PAIRWISE FEATURES (22):
      [17]    mean_ppi_dist (mean distance among co-mutated pairs, disconnected=10)
      [18]    min_ppi_dist
      [19]    frac_connected (fraction of pairs connected in PPI)
      [20]    frac_dist1 (fraction of pairs at distance 1 = direct PPI)
      [21]    frac_dist2_3 (distance 2-3 = same neighborhood)
      [22]    frac_dist4plus (distance 4+ = distant)
      [23]    mean_profile_overlap (mean cosine similarity of profile vectors)
      [24]    min_profile_overlap (most dissimilar pair)
      [25]    max_profile_overlap (most similar pair)
      [26]    frac_same_primary_ch (fraction of pairs sharing primary channel)
      [27]    frac_cross_ch (fraction with different primary channel)
      [28]    mean_cooccur_weight
      [29]    max_cooccur_weight
      [30]    frac_with_cooccur (fraction of pairs with any co-occurrence)
      [31]    mean_combined_entropy (mean entropy of (prof1+prof2)/2 for each pair)
      [32]    n_hub_hub_pairs (pairs where both are hub genes)
      [33]    n_hub_nonhub_pairs
      [34]    frac_close_cross_ch (dist<=2 AND different primary channel — key interaction)
      [35]    frac_far_same_ch (dist>=4 AND same primary — redundant hits)
      [36]    mean_tier_distance (mean |tier(g1) - tier(g2)|)
      [37]    frac_cross_tier (pairs spanning different tiers)
      [38]    n_pairs (total gene pairs)

    N-WISE (COMPONENT) FEATURES (8):
      [39]    n_components (connected components among mutated genes in PPI)
      [40]    largest_component_size
      [41]    frac_in_largest_component
      [42]    n_isolated (mutated genes not in PPI or singleton components)
      [43]    frac_isolated
      [44]    component_entropy (entropy of component size distribution)
      [45]    max_component_damage (max channel damage from single component)
      [46]    inter_component_spread (how spread is damage across components)

    ORIGINAL FEATURES (97):
      [47-50]  co-occurrence raw (4)
      [51-122] channel_features (72)
      [123-142] tier_features (20)
      [143]    age
      [144]    sex
      [145]    msi
      [146]    msi_high
      [147]    tmb

    Total: 148 features
    """
    all_expanded_genes = set(expanded_cm.keys())
    FEAT_DIM = 148
    X = np.zeros((N, FEAT_DIM))

    # Gene tier lookup
    gene_tier = {}
    for g, ch in expanded_cm.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)
    for g, ch in V6_CHANNEL_MAP.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    # Channel gene sets for isolation
    ch_gene_sets = defaultdict(set)
    for g, ch in V6_CHANNEL_MAP.items():
        ch_gene_sets[ch].add(g)
    for g, ch in expanded_cm.items():
        if ch in CH_TO_IDX:
            ch_gene_sets[ch].add(g)

    ppi_node_set = set(G_ppi.nodes())
    DISCONNECTED_DIST = 10  # sentinel for disconnected pairs

    for idx in range(N):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

        patient_genes = patient_genes_map.get(idx, set())
        mutated = patient_genes & all_expanded_genes
        mutated_list = sorted(mutated)
        n_mut = len(mutated_list)
        ct_name = ct_per_patient[idx]
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        # ===== BASE FEATURES =====
        channel_damage = np.zeros(N_CH)
        ct_gene_shift_sum = 0.0
        global_gene_shift_sum = 0.0

        for g in mutated:
            profile = channel_profiles.get(g)
            if profile is not None:
                channel_damage += profile
            ct_s = ct_gene_shift.get((ct_name, g), None)
            if ct_s is not None:
                ct_gene_shift_sum += ct_s
            gl_s = global_gene_shift.get(g, None)
            if gl_s is not None:
                global_gene_shift_sum += gl_s

        total_damage = channel_damage.sum()
        if total_damage > 0:
            damage_frac = channel_damage / total_damage
            entropy = profile_entropy(damage_frac)
            hhi = float(np.sum(damage_frac ** 2))
        else:
            damage_frac = np.zeros(N_CH)
            entropy = 0.0
            hhi = 0.0

        weighted_ch_shift = 0.0
        if total_damage > 0:
            for ci, ch_name in enumerate(CHANNELS):
                if damage_frac[ci] > 0:
                    weighted_ch_shift += damage_frac[ci] * ct_ch_shift.get((ct_name, ch_name), 0.0)

        # Hub damage
        total_hub_damage = 0.0
        for ci, ch_name in enumerate(CHANNELS):
            if channel_damage[ci] < 0.05:
                continue
            ch_mutated = set()
            for g in mutated:
                p = channel_profiles.get(g)
                if p is not None and p[ci] > 0.05:
                    ch_mutated.add(g)
            ch_hubs = HUB_GENES.get(ch_name, set())
            hub_hit = len(ch_mutated & ch_hubs)
            total_hub_damage += (hub_hit / max(len(ch_hubs), 1)) * channel_damage[ci]
        if total_damage > 0:
            total_hub_damage /= total_damage

        # Tier connectivity
        ppi_nodes = ppi_node_set - mutated
        if ppi_nodes:
            G_d = G_ppi.subgraph(ppi_nodes)
            components = list(nx.connected_components(G_d))
            node_comp = {}
            for comp_i, comp in enumerate(components):
                for g in comp:
                    node_comp[g] = comp_i
            tier_comp_sets = defaultdict(set)
            for g in ppi_nodes:
                t = gene_tier.get(g, -1)
                if t >= 0 and g in node_comp:
                    tier_comp_sets[t].add(node_comp[g])
            tier_connected = 0
            tier_tested = 0
            for t1 in range(4):
                for t2 in range(t1 + 1, 4):
                    tier_tested += 1
                    if tier_comp_sets[t1] & tier_comp_sets[t2]:
                        tier_connected += 1
            tier_conn = tier_connected / max(tier_tested, 1)
        else:
            tier_conn = 0.0

        X[idx, 0] = ct_bl
        X[idx, 1] = weighted_ch_shift
        X[idx, 2] = ct_gene_shift_sum
        X[idx, 3] = global_gene_shift_sum
        X[idx, 4:12] = channel_damage
        X[idx, 12] = entropy
        X[idx, 13] = total_hub_damage
        X[idx, 14] = tier_conn
        X[idx, 15] = n_mut
        X[idx, 16] = hhi

        # ===== PAIRWISE FEATURES =====
        if n_mut >= 2:
            dists = []
            overlaps = []
            cooccur_weights = []
            same_ch = 0
            cross_ch = 0
            hub_hub = 0
            hub_nonhub = 0
            close_cross = 0
            far_same = 0
            tier_dists_list = []
            cross_tier = 0
            combined_ents = []
            n_pairs = 0

            for i in range(n_mut):
                g1 = mutated_list[i]
                p1 = channel_profiles.get(g1)
                t1 = gene_tier.get(g1, -1)
                g1_hub = g1 in hub_gene_set
                g1_primary = np.argmax(p1) if p1 is not None else -1

                for j in range(i + 1, n_mut):
                    g2 = mutated_list[j]
                    p2 = channel_profiles.get(g2)
                    t2 = gene_tier.get(g2, -1)
                    g2_hub = g2 in hub_gene_set
                    g2_primary = np.argmax(p2) if p2 is not None else -1
                    n_pairs += 1

                    # PPI distance
                    key = tuple(sorted([g1, g2]))
                    d = ppi_dists.get(key, DISCONNECTED_DIST)
                    dists.append(d)

                    # Profile overlap
                    if p1 is not None and p2 is not None:
                        ov = cosine_sim(p1, p2)
                        overlaps.append(ov)
                        # Combined entropy
                        combined = (p1 + p2)
                        cs = combined.sum()
                        if cs > 0:
                            combined_ents.append(profile_entropy(combined / cs))

                    # Same/cross channel
                    if g1_primary >= 0 and g2_primary >= 0:
                        if g1_primary == g2_primary:
                            same_ch += 1
                        else:
                            cross_ch += 1
                            if d <= 2:
                                close_cross += 1
                        if g1_primary == g2_primary and d >= 4:
                            far_same += 1

                    # Co-occurrence
                    cooccur_key = (g1, g2) if g1 < g2 else (g2, g1)
                    ct_counts = cooccurrence.get(cooccur_key, {})
                    w = ct_counts.get(ct_name, 0)
                    cooccur_weights.append(w)

                    # Hub pairs
                    if g1_hub and g2_hub:
                        hub_hub += 1
                    elif g1_hub or g2_hub:
                        hub_nonhub += 1

                    # Tier distance
                    if t1 >= 0 and t2 >= 0:
                        td = abs(t1 - t2)
                        tier_dists_list.append(td)
                        if td > 0:
                            cross_tier += 1

            dists_arr = np.array(dists)
            connected_mask = dists_arr < DISCONNECTED_DIST

            X[idx, 17] = np.mean(dists_arr)
            X[idx, 18] = np.min(dists_arr)
            X[idx, 19] = np.mean(connected_mask)  # frac_connected
            X[idx, 20] = np.mean(dists_arr == 1)   # frac_dist1
            X[idx, 21] = np.mean((dists_arr >= 2) & (dists_arr <= 3))  # frac_dist2_3
            X[idx, 22] = np.mean(dists_arr >= 4)    # frac_dist4plus (includes disconnected)
            X[idx, 23] = np.mean(overlaps) if overlaps else 0.0
            X[idx, 24] = min(overlaps) if overlaps else 0.0
            X[idx, 25] = max(overlaps) if overlaps else 0.0
            X[idx, 26] = same_ch / n_pairs if n_pairs > 0 else 0.0
            X[idx, 27] = cross_ch / n_pairs if n_pairs > 0 else 0.0
            X[idx, 28] = np.mean(cooccur_weights) if cooccur_weights else 0.0
            X[idx, 29] = max(cooccur_weights) if cooccur_weights else 0.0
            X[idx, 30] = np.mean(np.array(cooccur_weights) > 0) if cooccur_weights else 0.0
            X[idx, 31] = np.mean(combined_ents) if combined_ents else 0.0
            X[idx, 32] = hub_hub
            X[idx, 33] = hub_nonhub
            X[idx, 34] = close_cross / n_pairs if n_pairs > 0 else 0.0
            X[idx, 35] = far_same / n_pairs if n_pairs > 0 else 0.0
            X[idx, 36] = np.mean(tier_dists_list) if tier_dists_list else 0.0
            X[idx, 37] = cross_tier / n_pairs if n_pairs > 0 else 0.0
            X[idx, 38] = n_pairs

        # ===== N-WISE (COMPONENT) FEATURES =====
        if n_mut >= 1:
            mutated_in_ppi = mutated & ppi_node_set
            if mutated_in_ppi:
                G_mut = G_ppi.subgraph(mutated_in_ppi)
                mut_comps = list(nx.connected_components(G_mut))
                comp_sizes = [len(c) for c in mut_comps]
                n_not_in_ppi = n_mut - len(mutated_in_ppi)

                # Each gene not in PPI is its own "component"
                all_comp_sizes = comp_sizes + [1] * n_not_in_ppi
                n_components = len(all_comp_sizes)
                largest = max(all_comp_sizes)
                n_isolated = sum(1 for s in all_comp_sizes if s == 1)

                # Component entropy
                if n_components > 1:
                    size_frac = np.array(all_comp_sizes) / sum(all_comp_sizes)
                    comp_ent = profile_entropy(size_frac)
                else:
                    comp_ent = 0.0

                # Per-component channel damage
                comp_damages = []
                for comp in mut_comps:
                    cd = np.zeros(N_CH)
                    for g in comp:
                        p = channel_profiles.get(g)
                        if p is not None:
                            cd += p
                    comp_damages.append(cd.sum())
                # Add isolated gene damages
                for g in mutated - mutated_in_ppi:
                    p = channel_profiles.get(g)
                    comp_damages.append(p.sum() if p is not None else 0.0)

                max_comp_damage = max(comp_damages) if comp_damages else 0.0
                total_comp_damage = sum(comp_damages)
                if total_comp_damage > 0 and n_components > 1:
                    comp_frac = np.array(comp_damages) / total_comp_damage
                    inter_spread = profile_entropy(comp_frac)
                else:
                    inter_spread = 0.0

                X[idx, 39] = n_components
                X[idx, 40] = largest
                X[idx, 41] = largest / n_mut
                X[idx, 42] = n_isolated
                X[idx, 43] = n_isolated / n_mut
                X[idx, 44] = comp_ent
                X[idx, 45] = max_comp_damage
                X[idx, 46] = inter_spread
            else:
                # All mutations outside PPI
                X[idx, 39] = n_mut
                X[idx, 42] = n_mut
                X[idx, 43] = 1.0

        # ===== CO-OCCURRENCE RAW =====
        n_cooccur_pairs = 0
        total_cooccur_weight = 0.0
        max_cooccur_wt = 0.0
        cross_ch_cooccur = 0
        if n_mut >= 2:
            for i in range(n_mut):
                for j in range(i + 1, n_mut):
                    key = (mutated_list[i], mutated_list[j])
                    ct_counts = cooccurrence.get(key, {})
                    w = ct_counts.get(ct_name, 0)
                    if w > 0:
                        n_cooccur_pairs += 1
                        total_cooccur_weight += w
                        if w > max_cooccur_wt:
                            max_cooccur_wt = w
                        p1 = channel_profiles.get(key[0])
                        p2 = channel_profiles.get(key[1])
                        if p1 is not None and p2 is not None:
                            if np.argmax(p1) != np.argmax(p2):
                                cross_ch_cooccur += 1

        X[idx, 47] = n_cooccur_pairs
        X[idx, 48] = total_cooccur_weight
        X[idx, 49] = max_cooccur_wt
        X[idx, 50] = cross_ch_cooccur

        # ===== ORIGINAL V6 FEATURES =====
        X[idx, 51:123] = channel_features_all[idx].flatten()
        X[idx, 123:143] = tier_features_all[idx].flatten()
        X[idx, 143] = age_all[idx]
        X[idx, 144] = sex_all[idx]
        X[idx, 145] = msi_all[idx]
        X[idx, 146] = msi_high_all[idx]
        X[idx, 147] = tmb_all[idx]

    return X


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  PAIRWISE GRAPH SCORER")
    print("  Gene-pair interaction features + component structure")
    print("=" * 90)

    # --- Load ---
    expanded_cm = load_expanded_channel_map()
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    channel_features = data["channel_features"].numpy()
    tier_features = data["tier_features"].numpy()
    age = data["age"].numpy()
    sex = data["sex"].numpy()
    msi = data["msi_score"].numpy()
    msi_high = data["msi_high"].numpy()
    tmb = data["tmb"].numpy()
    print(f"  {N} patients")

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
    print(f"  {all_in_val.sum().item()} valid patients")

    # --- Mutations ---
    print("  Loading mutations...")
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
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes[idx].add(row["gene.hugoGeneSymbol"])

    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    # --- Co-occurrence + PPI ---
    print("  Computing co-occurrence...")
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    msk_genes_in_data = set()
    for genes in patient_genes.values():
        msk_genes_in_data |= genes
    msk_genes_in_data &= expanded_genes
    ppi_edges = fetch_string_expanded(msk_genes_in_data)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    print(f"  PPI graph: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # --- Curated-anchor profiles ---
    print("  Computing curated-anchor channel profiles...")
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)

    # --- Precompute PPI distances ---
    print("  Precomputing all-pairs PPI distances...")
    all_mutated_genes = set()
    for genes in patient_genes.values():
        all_mutated_genes |= genes
    all_mutated_genes &= expanded_genes
    ppi_dists = precompute_ppi_distances(G_ppi, all_mutated_genes)
    print(f"  Computed {len(ppi_dists)} pairwise distances")

    # Distance distribution
    dist_vals = list(ppi_dists.values())
    if dist_vals:
        dist_arr = np.array(dist_vals)
        print(f"  Distance distribution: d=1: {np.sum(dist_arr==1)}, d=2: {np.sum(dist_arr==2)}, "
              f"d=3: {np.sum(dist_arr==3)}, d=4+: {np.sum(dist_arr>=4)}")

    # --- Hub gene set ---
    hub_gene_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_gene_set |= ch_hubs

    # --- Shifts ---
    print("  Computing shifts...")
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
    for idx in range(N):
        ct_name = ct_per_patient[idx]
        ct_patients[ct_name].add(idx)
        for g in patient_genes.get(idx, set()):
            prof = channel_profiles.get(g)
            if prof is not None:
                for ci, ch_name in enumerate(CHANNELS):
                    if prof[ci] > 0.05:
                        ct_ch_patients[ct_name][ch_name].add(idx)

    ct_baseline_map = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 50:
            ct_baseline_map[ct_name] = float(hazards[pts_valid].mean())

    ct_ch_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for ch_name in CHANNELS:
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
        ct_gene_map = defaultdict(set)
        for idx in ct_patients[ct_name]:
            for g in patient_genes.get(idx, set()):
                ct_gene_map[g].add(idx)
        for gene, pts in ct_gene_map.items():
            pts_arr = np.array(sorted(pts))
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 15:
                ct_gene_shift[(ct_name, gene)] = float(hazards[pts_valid].mean() - bl)

    # =========================================================================
    # Graph walk
    # =========================================================================
    print(f"\n  Running pairwise graph walk (148 features)...")

    X_all = pairwise_graph_walk_batch(
        G_ppi, patient_genes, channel_profiles, ppi_dists,
        channel_features, tier_features, ct_per_patient,
        ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, hub_gene_set, N,
    )
    print(f"  Done.")

    # =========================================================================
    # Ablation
    # =========================================================================
    folds = list(skf.split(np.arange(N), events))

    base_idx = list(range(0, 17))
    pair_idx = list(range(17, 39))
    comp_idx = list(range(39, 47))
    cooccur_idx = list(range(47, 51))
    v6ch_idx = list(range(51, 123))
    v6tier_idx = list(range(123, 143))
    clin_idx = list(range(143, 148))

    graph_all = base_idx + pair_idx + comp_idx + cooccur_idx

    configs = {
        "A base only (no pairwise)":          base_idx,
        "B base + pairwise":                  base_idx + pair_idx,
        "C base + pairwise + components":     base_idx + pair_idx + comp_idx,
        "D graph_all (base+pair+comp+cooc)":  graph_all,
        "E pairwise only":                    pair_idx,
        "F components only":                  comp_idx,
        "G graph_all + v6ch+tier+clin":       graph_all + v6ch_idx + v6tier_idx + clin_idx,
        "H base + v6ch+tier+clin (prev best)": base_idx + cooccur_idx + v6ch_idx + v6tier_idx + clin_idx,
        "I v6ch+tier+clin (no graph)":        v6ch_idx + v6tier_idx + clin_idx,
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION: PAIRWISE GRAPH SCORER")
    print(f"{'='*90}")

    print(f"\n  {'Scorer':<45} {'Mean CI':>8} {'Std':>7}  Folds")
    print(f"  {'-'*45} {'-'*8} {'-'*7}  {'-'*35}")

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
        print(f"  {name:<45} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # =========================================================================
    # Per-CT ridge
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE RIDGE")
    print(f"{'='*90}")

    for label, feat_idx in [
        ("G (full with pairwise)", graph_all + v6ch_idx + v6tier_idx + clin_idx),
        ("H (base, no pairwise)", base_idx + cooccur_idx + v6ch_idx + v6tier_idx + clin_idx),
    ]:
        X_best = X_all[:, feat_idx]
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
        print(f"\n  Per-CT ridge {label}:")
        print(f"    {mean_perct:.4f} +/- {std_perct:.4f}  Folds: {folds_str}")
        results[f"perCT_{label}"] = {
            "mean": mean_perct, "std": std_perct, "folds": fold_cis_perct, "score": score_perct,
        }

    # =========================================================================
    # Ridge coefficients (pairwise features)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  RIDGE COEFFICIENTS (graph features)")
    print(f"{'='*90}")

    full_feat_idx = graph_all + v6ch_idx + v6tier_idx + clin_idx
    valid_idx = np.where(valid_mask)[0]
    reg_full = Ridge(alpha=1.0)
    reg_full.fit(X_all[valid_idx][:, full_feat_idx], hazards[valid_idx])

    graph_feat_names = (
        ["ct_baseline", "weighted_ch_shift", "ct_gene_shift", "global_gene_shift"]
        + [f"ch_dmg_{ch}" for ch in CHANNELS]
        + ["entropy", "hub_damage", "tier_conn", "n_mutated", "hhi"]
        + ["mean_ppi_dist", "min_ppi_dist", "frac_connected", "frac_dist1",
           "frac_dist2_3", "frac_dist4plus",
           "mean_overlap", "min_overlap", "max_overlap",
           "frac_same_ch", "frac_cross_ch",
           "mean_cooccur_wt", "max_cooccur_wt", "frac_with_cooccur",
           "mean_combined_ent", "n_hub_hub", "n_hub_nonhub",
           "frac_close_cross", "frac_far_same",
           "mean_tier_dist", "frac_cross_tier", "n_pairs"]
        + ["n_components", "largest_comp", "frac_in_largest", "n_isolated",
           "frac_isolated", "comp_entropy", "max_comp_damage", "inter_comp_spread"]
        + ["n_cooccur", "cooccur_wt", "max_cooccur", "cross_ch_cooccur"]
    )
    n_graph = len(graph_feat_names)
    coefs = reg_full.coef_[:n_graph]

    sorted_coef_idx = np.argsort(np.abs(coefs))[::-1]
    print(f"\n  {'Feature':<25} {'Coeff':>10} {'|Coeff|':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for i in sorted_coef_idx:
        if abs(coefs[i]) > 0.001:
            print(f"  {graph_feat_names[i]:<25} {coefs[i]:>10.4f} {abs(coefs[i]):>10.4f}")

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  COMPARISON")
    print(f"{'='*90}")

    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    best_name = max(results, key=lambda k: results[k]["mean"])
    best_ci = results[best_name]["mean"]

    print(f"\n  V6c transformer:                       {v6c:.4f}")
    print(f"  Previous best (single-ch per-CT):      0.6882")
    print(f"  Pairwise best ({best_name}):")
    print(f"                                         {best_ci:.4f}")
    print(f"  Delta vs V6c:                          {best_ci - v6c:+.4f}")

    # Pairwise contribution
    base_ci = results.get("A base only (no pairwise)", {}).get("mean", 0)
    pair_ci = results.get("B base + pairwise", {}).get("mean", 0)
    comp_ci = results.get("C base + pairwise + components", {}).get("mean", 0)
    print(f"\n  Pairwise feature contribution:")
    print(f"    Base only:                           {base_ci:.4f}")
    print(f"    + pairwise:                          {pair_ci:.4f} ({pair_ci - base_ci:+.4f})")
    print(f"    + components:                        {comp_ci:.4f} ({comp_ci - pair_ci:+.4f})")

    # Per-CT breakdown
    best_key = max((k for k in results if k.startswith("perCT_")),
                   key=lambda k: results[k]["mean"], default=None)
    if best_key:
        best_score = results[best_key]["score"]
        print(f"\n  Per-cancer-type ({best_key}):")
        print(f"  {'Cancer Type':<35} {'N':>5} {'Graph':>7} {'V6c':>7} {'Delta':>7}")
        print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

        ct_results = {}
        g_wins = 0
        v_wins = 0
        for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
            ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
            ct_valid = ct_mask & valid_mask
            ct_indices = np.where(ct_valid)[0]
            if len(ct_indices) < 100 or events[ct_indices].sum() < 10:
                continue
            ci_g = concordance_index(
                torch.tensor(best_score[ct_indices].astype(np.float32)),
                torch.tensor(times[ct_indices].astype(np.float32)),
                torch.tensor(events[ct_indices].astype(np.float32)),
            )
            ci_v = concordance_index(
                torch.tensor(hazards[ct_indices].astype(np.float32)),
                torch.tensor(times[ct_indices].astype(np.float32)),
                torch.tensor(events[ct_indices].astype(np.float32)),
            )
            delta = ci_g - ci_v
            marker = " ▲" if delta > 0.005 else ""
            ct_results[ct_name] = {"n": len(ct_indices), "graph": float(ci_g),
                                    "v6c": float(ci_v), "delta": float(delta)}
            if delta > 0.001: g_wins += 1
            elif delta < -0.001: v_wins += 1
            print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_g:>7.4f} {ci_v:>7.4f} "
                  f"{delta:>+7.4f}{marker}")

        print(f"\n  Graph wins: {g_wins}  V6c wins: {v_wins}")

    # Save
    save_results = {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                    for k, v in results.items()}
    if best_key:
        save_results["per_cancer_type"] = ct_results
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
