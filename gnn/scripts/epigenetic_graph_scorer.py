#!/usr/bin/env python3
"""
Epigenetic graph modifier scorer.

Key insight: chromatin remodeling and DNA methylation mutations don't just damage
one node — they change the active graph topology itself.

- Chromatin remodeling = write head. Creates/removes nodes from the active subgraph.
  When a chromatin remodeler is mutated, its PPI neighborhood becomes "destabilized" —
  those nodes may or may not be accessible. We model this as amplified damage scores
  in the destabilization zone.

- DNA methylation = edge permissions. Controls which TF-target interactions can happen.
  When a methylation gene is mutated, edges in its neighborhood lose gating.
  We model this as reweighted edges in the PPI graph.

Architecture:
  1. Group patients by unique epigenetic mutation set (~3K groups vs 43K patients)
  2. For each unique set, create a modified PPI graph
  3. Walk the modified graph once per group
  4. Map features back to all patients

Usage:
    python3 -u -m gnn.scripts.epigenetic_graph_scorer
"""

import sys, os, json, copy, time
import numpy as np, networkx as nx, torch, pandas as pd
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
from gnn.scripts.pairwise_graph_scorer import (
    precompute_ppi_distances, cosine_sim, pairwise_graph_walk_batch,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "epigenetic_graph_scorer")
CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

# Epigenetic gene classification
CHROMATIN_GENES = {
    'ANKRD11', 'ARID1B', 'ARID2', 'BCOR', 'CREBBP', 'EP300', 'H3C7',
    'KDM6A', 'KMT2A', 'KMT2B', 'KMT2C', 'KMT2D', 'NSD1', 'SETD2', 'SMARCA4',
}
METHYLATION_GENES = {
    'ATRX', 'DAXX', 'DNMT3A', 'DNMT3B', 'IDH1', 'IDH2', 'TET1', 'TET2',
}
EPIGENETIC_GENES = CHROMATIN_GENES | METHYLATION_GENES


# =========================================================================
# Graph modification functions
# =========================================================================

def get_regulatory_neighborhood(G_ppi, gene, max_hops=2):
    """Get the neighborhood of a gene up to max_hops in PPI.
    These are the genes potentially regulated by an epigenetic modifier."""
    if gene not in G_ppi:
        return set()
    neighborhood = set()
    for hop in range(1, max_hops + 1):
        for node, dist in nx.single_source_shortest_path_length(G_ppi, gene, cutoff=hop).items():
            if dist > 0:  # exclude self
                neighborhood.add(node)
    return neighborhood


def modify_graph_for_epigenetics(G_ppi, epi_mutations, channel_profiles,
                                  chromatin_amplify=1.5, methyl_edge_decay=0.5):
    """
    Create a modified PPI graph based on a patient's epigenetic mutations.

    Chromatin remodeling mutations (node writes):
      - Destabilize the remodeler's neighborhood (k=2 hops)
      - Nodes in the destabilization zone get amplified damage profiles
        (uncertainty = risk, the cell can't control these nodes)
      - Returns amplification factors per gene

    DNA methylation mutations (edge permissions):
      - Reweight edges in the methylation gene's neighborhood
      - Edges touching regulated genes get decayed (gating is lost,
        normal regulatory connections are unreliable)
      - Returns a modified graph with reweighted edges

    Returns: (G_modified, node_amplification_dict)
    """
    chromatin_muts = epi_mutations & CHROMATIN_GENES
    methyl_muts = epi_mutations & METHYLATION_GENES

    # Node amplification from chromatin remodeling mutations
    node_amplification = defaultdict(lambda: 1.0)
    destabilized_nodes = set()

    for gene in chromatin_muts:
        neighborhood = get_regulatory_neighborhood(G_ppi, gene, max_hops=2)
        destabilized_nodes |= neighborhood
        for node in neighborhood:
            # Each chromatin remodeler mutation compounds the instability
            node_amplification[node] *= chromatin_amplify

    # Edge modification from methylation mutations
    G_mod = G_ppi  # start with original
    affected_edges = set()

    if methyl_muts:
        G_mod = G_ppi.copy()
        for gene in methyl_muts:
            neighborhood = get_regulatory_neighborhood(G_ppi, gene, max_hops=2)
            for node in neighborhood:
                for nb in G_mod.neighbors(node):
                    edge_key = tuple(sorted([node, nb]))
                    if edge_key not in affected_edges:
                        affected_edges.add(edge_key)
                        # Decay edge weight — regulatory gating is lost
                        old_w = G_mod[node][nb].get('weight', 0.7)
                        G_mod[node][nb]['weight'] = old_w * methyl_edge_decay

    return G_mod, dict(node_amplification), len(destabilized_nodes), len(affected_edges)


def compute_modified_profiles(G_modified, expanded_cm):
    """Recompute curated-anchor profiles on a modified graph."""
    return compute_curated_profiles(G_modified, expanded_cm)


# =========================================================================
# Epigenetic-aware feature computation
# =========================================================================

def epigenetic_walk_features(
    G_ppi, G_modified, patient_genes_set, channel_profiles_base,
    channel_profiles_mod, node_amplification, ppi_dists_base,
    expanded_cm, hub_gene_set, ct_name,
    ct_ch_shift, ct_gene_shift, global_gene_shift,
    ct_baseline_map, baseline, cooccurrence,
):
    """
    Compute epigenetic-specific features for a patient.

    Returns a feature vector with:
    - Amplified channel damage (using node amplification from chromatin mutations)
    - Profile shift (how much the modified graph changes channel profiles)
    - Destabilization zone features
    - Cross-layer interaction features
    """
    all_expanded = set(expanded_cm.keys())
    mutated = patient_genes_set & all_expanded
    n_mut = len(mutated)

    # Epigenetic features (21 dims)
    feats = np.zeros(21)

    if n_mut == 0:
        return feats

    # --- Feature 0-7: Amplified channel damage ---
    # Channel damage with chromatin destabilization amplification
    amp_channel_damage = np.zeros(N_CH)
    base_channel_damage = np.zeros(N_CH)
    for g in mutated:
        profile = channel_profiles_base.get(g)
        if profile is not None:
            amp = node_amplification.get(g, 1.0)
            amp_channel_damage += profile * amp
            base_channel_damage += profile

    feats[0:8] = amp_channel_damage

    # --- Feature 8: Total amplification factor ---
    base_total = base_channel_damage.sum()
    amp_total = amp_channel_damage.sum()
    feats[8] = (amp_total / base_total - 1.0) if base_total > 0 else 0.0

    # --- Feature 9-10: Profile shift from graph modification ---
    # How much do channel profiles change when we modify the graph?
    profile_shifts = []
    for g in mutated:
        p_base = channel_profiles_base.get(g)
        p_mod = channel_profiles_mod.get(g)
        if p_base is not None and p_mod is not None:
            shift = np.linalg.norm(p_mod - p_base)
            profile_shifts.append(shift)
    feats[9] = np.mean(profile_shifts) if profile_shifts else 0.0
    feats[10] = np.max(profile_shifts) if profile_shifts else 0.0

    # --- Feature 11-12: Destabilization zone overlap ---
    # How many of this patient's mutations are in destabilized zones?
    n_in_destab = sum(1 for g in mutated if node_amplification.get(g, 1.0) > 1.0)
    feats[11] = n_in_destab
    feats[12] = n_in_destab / n_mut if n_mut > 0 else 0.0

    # --- Feature 13-14: Epigenetic mutation counts ---
    n_chromatin = len(mutated & CHROMATIN_GENES)
    n_methyl = len(mutated & METHYLATION_GENES)
    feats[13] = n_chromatin
    feats[14] = n_methyl

    # --- Feature 15: Chromatin × non-epi interaction ---
    # Fraction of non-epigenetic mutations in destabilized zone
    non_epi = mutated - EPIGENETIC_GENES
    if non_epi and n_in_destab > 0:
        n_non_epi_destab = sum(1 for g in non_epi if node_amplification.get(g, 1.0) > 1.0)
        feats[15] = n_non_epi_destab / len(non_epi)
    else:
        feats[15] = 0.0

    # --- Feature 16: Amplified hub damage ---
    amp_hub_damage = 0.0
    for g in mutated & hub_gene_set:
        profile = channel_profiles_base.get(g)
        if profile is not None:
            amp = node_amplification.get(g, 1.0)
            amp_hub_damage += profile.sum() * amp
    feats[16] = amp_hub_damage

    # --- Feature 17: Channel damage entropy on modified graph ---
    if amp_total > 0:
        amp_frac = amp_channel_damage / amp_total
        feats[17] = profile_entropy(amp_frac)

    # --- Feature 18: Concentrated epigenetic damage ---
    # Is the epigenetic damage focused on one channel or spread?
    epi_damage = np.zeros(N_CH)
    for g in mutated & EPIGENETIC_GENES:
        profile = channel_profiles_base.get(g)
        if profile is not None:
            epi_damage += profile
    epi_total = epi_damage.sum()
    if epi_total > 0:
        epi_frac = epi_damage / epi_total
        feats[18] = float(np.max(epi_frac))  # concentration of epi damage

    # --- Feature 19: Regulatory reach ---
    # Total number of genes in destabilization zones / total PPI nodes
    n_destab_total = sum(1 for g, a in node_amplification.items() if a > 1.0)
    feats[19] = n_destab_total / max(G_ppi.number_of_nodes(), 1)

    # --- Feature 20: Methylation-chromatin co-occurrence ---
    # Having both types simultaneously = loss of both node control and edge control
    feats[20] = 1.0 if (n_chromatin > 0 and n_methyl > 0) else 0.0

    return feats


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t0 = time.time()

    print("=" * 90)
    print("  EPIGENETIC GRAPH MODIFIER SCORER")
    print("  Chromatin = node writes, Methylation = edge permissions")
    print("=" * 90)

    # --- Load data (same as pairwise scorer) ---
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

    # --- PPI + co-occurrence ---
    print("  Computing co-occurrence + PPI...")
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    msk_genes = set()
    for genes in patient_genes.values():
        msk_genes |= genes
    msk_genes &= expanded_genes
    ppi_edges = fetch_string_expanded(msk_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    print(f"  PPI graph: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # Base profiles + distances
    channel_profiles_base = compute_curated_profiles(G_ppi, expanded_cm)
    ppi_dists = precompute_ppi_distances(G_ppi, msk_genes)

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
            prof = channel_profiles_base.get(g)
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
    # Step 1: Group patients by epigenetic mutation set
    # =========================================================================
    print("\n  Grouping patients by epigenetic mutation set...")
    epi_set_to_patients = defaultdict(list)
    for idx in range(N):
        epi_muts = patient_genes.get(idx, set()) & EPIGENETIC_GENES
        key = frozenset(epi_muts)
        epi_set_to_patients[key].append(idx)

    n_groups = len(epi_set_to_patients)
    n_with_epi = sum(len(v) for k, v in epi_set_to_patients.items() if k)
    print(f"  {n_groups} unique epigenetic mutation sets")
    print(f"  {n_with_epi} patients with ≥1 epigenetic mutation ({100*n_with_epi/N:.1f}%)")
    print(f"  {N - n_with_epi} patients with no epigenetic mutations ({100*(N-n_with_epi)/N:.1f}%)")

    # =========================================================================
    # Step 2: Compute base pairwise features (cached from gap analysis)
    # =========================================================================
    cache_path = os.path.join(ANALYSIS_CACHE, "gap_analysis_features.npz")
    if os.path.exists(cache_path):
        print("\n  Loading cached base pairwise features (148-dim)...")
        X_base = np.load(cache_path)["X_all"]
    else:
        print("\n  Computing base pairwise features (148-dim)...")
        X_base = pairwise_graph_walk_batch(
            G_ppi, patient_genes, channel_profiles_base, ppi_dists,
            channel_features, tier_features, ct_per_patient,
            ct_ch_shift, ct_gene_shift, global_gene_shift,
            age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
            expanded_cm, cooccurrence, hub_gene_set, N,
        )
        np.savez_compressed(cache_path, X_all=X_base)

    # =========================================================================
    # Step 3: Per-group epigenetic graph modification
    # =========================================================================
    EPI_FEAT_DIM = 21
    X_epi = np.zeros((N, EPI_FEAT_DIM))

    print(f"\n  Computing epigenetic features for {n_groups} groups...")
    t1 = time.time()

    # Precompute regulatory neighborhoods (shared across patients)
    epi_neighborhoods = {}
    for gene in EPIGENETIC_GENES:
        epi_neighborhoods[gene] = get_regulatory_neighborhood(G_ppi, gene, max_hops=2)

    n_done = 0
    for epi_set, patient_indices in epi_set_to_patients.items():
        if not epi_set:
            # No epigenetic mutations — features stay zero
            n_done += 1
            continue

        # Modify graph for this epigenetic mutation set
        G_mod, node_amp, n_destab, n_affected_edges = modify_graph_for_epigenetics(
            G_ppi, epi_set, channel_profiles_base,
            chromatin_amplify=1.5, methyl_edge_decay=0.5,
        )

        # Recompute profiles on modified graph
        channel_profiles_mod = compute_modified_profiles(G_mod, expanded_cm)

        # Compute epigenetic features for each patient in this group
        for idx in patient_indices:
            patient_genes_set = patient_genes.get(idx, set())
            ct_name = ct_per_patient[idx]

            feats = epigenetic_walk_features(
                G_ppi, G_mod, patient_genes_set,
                channel_profiles_base, channel_profiles_mod,
                node_amp, ppi_dists,
                expanded_cm, hub_gene_set, ct_name,
                ct_ch_shift, ct_gene_shift, global_gene_shift,
                ct_baseline_map, baseline, cooccurrence,
            )
            X_epi[idx] = feats

        n_done += 1
        if n_done % 500 == 0:
            elapsed = time.time() - t1
            rate = n_done / elapsed
            remaining = (n_groups - n_done) / rate
            print(f"    {n_done}/{n_groups} groups ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t1
    print(f"  Epigenetic features computed in {elapsed:.1f}s")

    # =========================================================================
    # Step 4: Combine base + epigenetic features and evaluate
    # =========================================================================
    TOTAL_DIM = 148 + EPI_FEAT_DIM  # 169
    X_combined = np.hstack([X_base, X_epi])
    print(f"\n  Combined feature vector: {TOTAL_DIM} dims (148 base + {EPI_FEAT_DIM} epigenetic)")

    folds = list(skf.split(np.arange(N), events))

    # --- Evaluate: base only (sanity check) ---
    print("\n  Evaluating base features only (148-dim)...")
    score_base = fit_per_ct_ridge(X_base, hazards, valid_mask, events, times,
                                   ct_per_patient, folds, ct_min_patients=200)

    # --- Evaluate: combined ---
    print("  Evaluating combined features (169-dim)...")
    score_combined = fit_per_ct_ridge(X_combined, hazards, valid_mask, events, times,
                                       ct_per_patient, folds, ct_min_patients=200)

    # --- Evaluate: epigenetic features only (for patients with epi mutations) ---
    # Check if epigenetic features alone have signal
    epi_mask = np.any(X_epi != 0, axis=1)
    print(f"\n  Patients with non-zero epigenetic features: {epi_mask.sum()}")

    # --- Per-CT comparison ---
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE RESULTS")
    print(f"{'='*90}")

    print(f"\n  {'Cancer Type':<35} {'N':>5} {'Base':>7} {'Combined':>9} {'Delta':>7} {'V6c':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")

    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 50:
            continue

        ci_base = concordance_index(
            torch.tensor(score_base[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_comb = concordance_index(
            torch.tensor(score_combined[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v6c = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        delta = ci_comb - ci_base
        marker = " +" if delta > 0.002 else (" -" if delta < -0.002 else "")
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_base:>7.4f} {ci_comb:>9.4f} {delta:>+7.4f} {ci_v6c:>7.4f}{marker}")

    # --- Epigenetic feature importance ---
    print(f"\n{'='*90}")
    print(f"  EPIGENETIC FEATURE ANALYSIS")
    print(f"{'='*90}")

    epi_feature_names = [
        "amp_ch_DDR", "amp_ch_CellCycle", "amp_ch_PI3K", "amp_ch_Endocrine",
        "amp_ch_Immune", "amp_ch_TissueArch", "amp_ch_ChromRemodel", "amp_ch_DNAMethyl",
        "amplification_excess", "mean_profile_shift", "max_profile_shift",
        "n_in_destab_zone", "frac_in_destab_zone",
        "n_chromatin_muts", "n_methyl_muts",
        "frac_non_epi_destabilized", "amplified_hub_damage",
        "amp_damage_entropy", "epi_damage_concentration",
        "regulatory_reach", "methyl_chromatin_cooccur",
    ]

    # Fit a global ridge on combined features to see coefficients
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_epi_scaled = scaler.fit_transform(X_epi[valid_mask])
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_epi_scaled, hazards[valid_mask])

    coef_order = np.argsort(np.abs(ridge.coef_))[::-1]
    print(f"\n  {'Feature':<35} {'Coef':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for i in coef_order:
        if abs(ridge.coef_[i]) < 0.001:
            continue
        name = epi_feature_names[i] if i < len(epi_feature_names) else f"feat_{i}"
        print(f"  {name:<35} {ridge.coef_[i]:>8.4f} {X_epi[valid_mask, i].mean():>8.4f} {X_epi[valid_mask, i].std():>8.4f}")

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.0f}s")

    # Save features
    np.savez_compressed(
        os.path.join(SAVE_BASE, "features.npz"),
        X_base=X_base, X_epi=X_epi, X_combined=X_combined,
    )
    print(f"  Features saved to {SAVE_BASE}/features.npz")

    print("\n  Done.")


if __name__ == "__main__":
    main()
