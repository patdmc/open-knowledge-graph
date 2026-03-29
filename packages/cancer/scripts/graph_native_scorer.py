#!/usr/bin/env python3
"""
Graph-native survival scorer — zero learned parameters.

The prediction is a graph traversal. Period.

For each patient:
  1. Start with their mutated genes as "damaged" nodes (damage = 1.0)
  2. Edge weights encode the biology:
     - PPI weight = physical interaction strength
     - Chromatin mutation → amplify neighborhood node damage (nodes going offline)
     - Methylation mutation → decay neighborhood edge weights (connections ungated)
  3. Propagate damage through weighted edges (iterative message passing to convergence)
  4. Sum converged damage across channels = hazard score

No Ridge. No GBT. No transformer. No training.
The weighting is baked into the edge weights.
The topology is the prediction.

Optimization: group by unique mutation set (~14K unique from 43K patients).

Usage:
    python3 -u -m gnn.scripts.graph_native_scorer
"""

import sys, os, time, json
import numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_FUNCTION,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.training.metrics import concordance_index

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence,
)
from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles

SAVE_BASE = os.path.join(GNN_RESULTS, "graph_native_scorer")
CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

# Epigenetic genes
CHROMATIN_GENES = {
    'ANKRD11', 'ARID1B', 'ARID2', 'BCOR', 'CREBBP', 'EP300', 'H3C7',
    'KDM6A', 'KMT2A', 'KMT2B', 'KMT2C', 'KMT2D', 'NSD1', 'SETD2', 'SMARCA4',
}
METHYLATION_GENES = {
    'ATRX', 'DAXX', 'DNMT3A', 'DNMT3B', 'IDH1', 'IDH2', 'TET1', 'TET2',
}
EPIGENETIC_GENES = CHROMATIN_GENES | METHYLATION_GENES

# Damping factor for propagation (like PageRank)
DAMPING = 0.85
N_ITERATIONS = 5  # converges fast on small subgraphs


# =========================================================================
# Pure graph walk — no learned parameters
# =========================================================================

def graph_walk_score(mutated_genes, G_ppi, channel_profiles, hub_gene_set):
    """
    Compute a survival hazard score from pure graph traversal.

    1. Initialize damage at mutated nodes
    2. Modify graph based on epigenetic mutations
    3. Propagate damage through PPI edges
    4. Aggregate converged damage by channel

    Returns: (total_score, channel_scores[8], walk_stats)
    """
    if not mutated_genes:
        return 0.0, np.zeros(N_CH), {}

    ppi_nodes = set(G_ppi.nodes())
    expanded_mutated = mutated_genes & ppi_nodes
    chromatin_muts = mutated_genes & CHROMATIN_GENES
    methyl_muts = mutated_genes & METHYLATION_GENES

    # --- Build patient-specific edge weights ---
    # Start with PPI weights, then modify for epigenetic state

    # Identify chromatin destabilization zones (nodes going offline)
    destab_nodes = set()
    destab_amp = {}  # node → amplification factor
    for cg in chromatin_muts:
        if cg not in G_ppi:
            continue
        for nb, dist in nx.single_source_shortest_path_length(G_ppi, cg, cutoff=2).items():
            if dist > 0:
                destab_nodes.add(nb)
                amp = 1.5 if dist == 1 else 1.25  # closer = more destabilized
                destab_amp[nb] = destab_amp.get(nb, 1.0) * amp  # compounds

    # Identify methylation zones (edges losing gating)
    methyl_zone = set()
    for mg in methyl_muts:
        if mg not in G_ppi:
            continue
        for nb, dist in nx.single_source_shortest_path_length(G_ppi, mg, cutoff=2).items():
            if dist > 0:
                methyl_zone.add(nb)

    # --- Initialize damage at mutated nodes ---
    # Each mutated gene starts with damage = its channel profile weighted by function
    all_genes = sorted(ppi_nodes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n = len(all_genes)

    # Damage vector per channel
    damage = np.zeros((n, N_CH))

    for g in mutated_genes:
        if g not in gene_to_idx:
            continue
        i = gene_to_idx[g]
        prof = channel_profiles.get(g)
        if prof is None:
            continue

        # Base damage = channel profile
        d = prof.copy()

        # Hub genes do more initial damage (they're more connected)
        if g in hub_gene_set:
            d *= 1.3

        # GOF mutations activate pathways (positive damage)
        # LOF mutations break pathways (also positive damage — either way, dysfunction)
        func = GENE_FUNCTION.get(g, "context")
        if func == "GOF":
            d *= 1.1  # slight boost — activating mutations are often drivers
        elif func == "LOF":
            d *= 1.0  # baseline

        # Chromatin destabilization amplification
        if g in destab_amp:
            d *= destab_amp[g]

        damage[i] = d

    # --- Build adjacency matrix with modified weights ---
    # Sparse representation for efficiency
    adj = np.zeros((n, n))
    for g in all_genes:
        if g not in G_ppi:
            continue
        i = gene_to_idx[g]
        for nb in G_ppi.neighbors(g):
            if nb not in gene_to_idx:
                continue
            j = gene_to_idx[nb]
            w = G_ppi[g][nb].get('weight', 0.7)

            # Methylation decay: edges in methylation zone lose gating
            if g in methyl_zone or nb in methyl_zone:
                w *= 0.5

            adj[i, j] = w

    # Row-normalize adjacency (stochastic matrix)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums

    # --- Propagate damage (PageRank-like) ---
    # damage_t+1 = (1-d) * initial_damage + d * A^T @ damage_t
    initial_damage = damage.copy()

    for _ in range(N_ITERATIONS):
        propagated = adj_norm.T @ damage
        damage = (1 - DAMPING) * initial_damage + DAMPING * propagated

    # --- Aggregate: sum damage across all nodes, per channel ---
    channel_scores = damage.sum(axis=0)  # (N_CH,)
    total_score = channel_scores.sum()

    # Walk statistics
    n_reached = (damage.sum(axis=1) > 1e-6).sum()
    stats = {
        "n_mutated": len(mutated_genes),
        "n_in_ppi": len(expanded_mutated),
        "n_destabilized": len(destab_nodes),
        "n_methyl_zone": len(methyl_zone),
        "n_reached": n_reached,
        "reach_fraction": n_reached / max(n, 1),
    }

    return total_score, channel_scores, stats


def graph_walk_score_fast(mutated_genes, G_ppi, channel_profiles, hub_gene_set,
                          precomputed_adj, gene_to_idx, all_genes,
                          ct_adj_overlay=None):
    """
    Fast version using precomputed base adjacency.
    Only recomputes the epigenetic modifications per patient.

    If ct_adj_overlay is provided, it's added to the PPI adjacency before
    normalization — this encodes cancer-type-specific co-occurrence strength.
    """
    if not mutated_genes:
        return 0.0, np.zeros(N_CH)

    n = len(all_genes)
    chromatin_muts = mutated_genes & CHROMATIN_GENES
    methyl_muts = mutated_genes & METHYLATION_GENES

    # --- Start with PPI + CT-specific co-occurrence overlay ---
    adj = precomputed_adj.copy()
    if ct_adj_overlay is not None:
        adj += ct_adj_overlay

    # Methylation: decay edges in methylation zone
    if methyl_muts:
        methyl_zone_idx = set()
        for mg in methyl_muts:
            if mg not in G_ppi:
                continue
            for nb, dist in nx.single_source_shortest_path_length(G_ppi, mg, cutoff=2).items():
                if dist > 0 and nb in gene_to_idx:
                    methyl_zone_idx.add(gene_to_idx[nb])
        for i in methyl_zone_idx:
            adj[i, :] *= 0.5
            adj[:, i] *= 0.5

    # Row-normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums

    # --- Chromatin destabilization ---
    destab_amp = {}
    for cg in chromatin_muts:
        if cg not in G_ppi:
            continue
        for nb, dist in nx.single_source_shortest_path_length(G_ppi, cg, cutoff=2).items():
            if dist > 0 and nb in gene_to_idx:
                amp = 1.5 if dist == 1 else 1.25
                idx = gene_to_idx[nb]
                destab_amp[idx] = destab_amp.get(idx, 1.0) * amp

    # --- Initialize damage ---
    damage = np.zeros((n, N_CH))
    for g in mutated_genes:
        if g not in gene_to_idx:
            continue
        i = gene_to_idx[g]
        prof = channel_profiles.get(g)
        if prof is None:
            continue

        d = prof.copy()
        if g in hub_gene_set:
            d *= 1.3
        func = GENE_FUNCTION.get(g, "context")
        if func == "GOF":
            d *= 1.1
        if i in destab_amp:
            d *= destab_amp[i]
        damage[i] = d

    # --- Propagate ---
    initial_damage = damage.copy()
    for _ in range(N_ITERATIONS):
        propagated = adj_norm.T @ damage
        damage = (1 - DAMPING) * initial_damage + DAMPING * propagated

    channel_scores = damage.sum(axis=0)
    return channel_scores.sum(), channel_scores


def build_ct_adj_overlays(cooccurrence, gene_to_idx, n_ppi):
    """Precompute per-cancer-type adjacency overlays from co-occurrence data.

    Co-occurrence counts are normalized: for each CT, the raw count is divided
    by the max count across all CTs for that pair, giving a 0-1 scale. This is
    then scaled by a COOCCUR_WEIGHT factor to blend with PPI weights.

    Returns: dict ct_name → (n_ppi, n_ppi) sparse overlay matrix
    """
    COOCCUR_WEIGHT = 0.3  # blend factor — co-occurrence adds up to 0.3 to edge weight

    # First pass: find max count per pair (for normalization)
    pair_max = {}
    for (g1, g2), ct_counts in cooccurrence.items():
        max_count = max(ct_counts.values())
        pair_max[(g1, g2)] = max_count

    # Second pass: build per-CT overlays
    ct_entries = defaultdict(list)  # ct → [(i, j, weight)]
    for (g1, g2), ct_counts in cooccurrence.items():
        if g1 not in gene_to_idx or g2 not in gene_to_idx:
            continue
        i, j = gene_to_idx[g1], gene_to_idx[g2]
        max_c = pair_max[(g1, g2)]
        if max_c == 0:
            continue
        for ct_name, count in ct_counts.items():
            w = COOCCUR_WEIGHT * (count / max_c)
            ct_entries[ct_name].append((i, j, w))

    ct_overlays = {}
    for ct_name, entries in ct_entries.items():
        overlay = np.zeros((n_ppi, n_ppi))
        for i, j, w in entries:
            overlay[i, j] = w
            overlay[j, i] = w  # symmetric
        ct_overlays[ct_name] = overlay

    return ct_overlays


def derive_ct_channel_weights(all_channel_scores, events, times, ct_per_patient,
                              train_idx_set, channel_names):
    """Derive per-CT channel importance weights from train set.

    For each cancer type, compute Spearman rank correlation between each
    channel's damage score and survival time (among events only, to avoid
    censoring noise). Negative correlation = more damage → shorter survival
    = channel is prognostic.

    Weights are clipped to [0, inf] (only use channels that predict worse
    outcomes with more damage) and normalized to sum to 1.

    Returns: dict ct_name → np.array(N_CH,) weights
    """
    n_ch = len(channel_names)
    train_idx = np.array(sorted(train_idx_set))

    # Group train patients by CT
    ct_train = defaultdict(list)
    for idx in train_idx:
        ct_train[ct_per_patient[idx]].append(idx)

    ct_weights = {}
    for ct_name, indices in ct_train.items():
        idx_arr = np.array(indices)
        # Use events only for cleaner signal
        mask = (events[idx_arr] == 1) & (times[idx_arr] > 0)
        idx_events = idx_arr[mask]

        if len(idx_events) < 20:
            # Too few events — use uniform weights
            ct_weights[ct_name] = np.ones(n_ch) / n_ch
            continue

        t = times[idx_events]
        ch = all_channel_scores[idx_events]  # (n_events, n_ch)

        weights = np.zeros(n_ch)
        for ci in range(n_ch):
            if ch[:, ci].std() < 1e-8:
                continue
            rho, p = spearmanr(ch[:, ci], t)
            # Negative rho = more damage → shorter survival = prognostic
            # Clip to non-negative (only keep prognostic channels)
            weights[ci] = max(-rho, 0.0)

        # Normalize
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(n_ch) / n_ch

        ct_weights[ct_name] = weights

    # Global fallback for unseen CTs
    ct_weights["_global"] = np.ones(n_ch) / n_ch

    return ct_weights


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t0 = time.time()

    print("=" * 90)
    print("  GRAPH-NATIVE SURVIVAL SCORER — CT-CONDITIONED CHANNEL WEIGHTS")
    print("  Zero learned parameters. Channel importance derived from train set.")
    print("=" * 90)

    # --- Load data ---
    expanded_cm = load_expanded_channel_map()
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    print(f"  {N} patients, {len(ct_vocab)} cancer types")

    # --- Holdback partition (matches V6 training script: seed=42, 15%) ---
    np.random.seed(42)
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_holdback = int(N * 0.15)
    holdback_idx = set(all_idx[:n_holdback].tolist())
    train_idx_set = set(all_idx[n_holdback:].tolist())
    print(f"  Holdback: {n_holdback}, Train: {len(train_idx_set)}")

    # --- Mutations + PPI ---
    print("  Loading mutations + PPI...")
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol"],
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

    # --- Co-occurrence from TRAIN patients only (no holdback leakage) ---
    train_patient_genes = {k: v for k, v in patient_genes.items() if k in train_idx_set}
    train_ct = {k: v for k, v in ct_per_patient.items() if k in train_idx_set}
    cooccurrence = compute_cooccurrence(train_patient_genes, train_ct, min_count=10)
    print(f"  Co-occurrence pairs (train-only): {len(cooccurrence)}")

    msk_genes = set()
    for genes in patient_genes.values():
        msk_genes |= genes
    msk_genes &= expanded_genes
    ppi_edges = fetch_string_expanded(msk_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)
    print(f"  PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    hub_gene_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_gene_set |= ch_hubs

    # --- Precompute base adjacency matrix ---
    all_genes = sorted(G_ppi.nodes())
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n_ppi = len(all_genes)
    print(f"  Building {n_ppi}x{n_ppi} base adjacency matrix...")

    base_adj = np.zeros((n_ppi, n_ppi))
    for g in all_genes:
        i = gene_to_idx[g]
        for nb in G_ppi.neighbors(g):
            if nb in gene_to_idx:
                j = gene_to_idx[nb]
                base_adj[i, j] = G_ppi[g][nb].get('weight', 0.7)

    # =========================================================================
    # Score all patients — grouped by mutation set
    # =========================================================================
    print("\n  Scoring patients (grouped by mutation set)...")
    t1 = time.time()

    mutset_to_patients = defaultdict(list)
    for idx in range(N):
        key_genes = frozenset(patient_genes.get(idx, set()))
        mutset_to_patients[key_genes].append(idx)

    n_unique = len(mutset_to_patients)
    print(f"  {n_unique} unique mutation sets")

    all_channel_scores = np.zeros((N, N_CH))
    n_done = 0
    for mutset, indices in mutset_to_patients.items():
        _, ch_scores = graph_walk_score_fast(
            set(mutset), G_ppi, channel_profiles, hub_gene_set,
            base_adj, gene_to_idx, all_genes,
        )
        for idx in indices:
            all_channel_scores[idx] = ch_scores
        n_done += 1
        if n_done % 5000 == 0:
            elapsed = time.time() - t1
            print(f"    {n_done}/{n_unique} ({elapsed:.0f}s)")
    elapsed = time.time() - t1
    print(f"  Walk completed in {elapsed:.1f}s")

    # Unconditioned score = equal-weight sum of channels
    all_scores_raw = all_channel_scores.sum(axis=1)

    # =========================================================================
    # Derive CT-specific channel weights from TRAIN set
    # =========================================================================
    print("\n  Deriving CT-specific channel weights from train set...")
    ct_channel_weights = derive_ct_channel_weights(
        all_channel_scores, events, times, ct_per_patient,
        train_idx_set, CHANNELS,
    )

    print(f"\n  {'Cancer Type':<35} ", end="")
    for ch in CHANNELS:
        print(f"{ch[:6]:>7}", end="")
    print()
    print(f"  {'-'*35} " + " -------" * N_CH)
    for ct_name in sorted(ct_channel_weights,
                          key=lambda x: -len([i for i in range(N) if ct_per_patient.get(i) == x])):
        if ct_name == "_global":
            continue
        w = ct_channel_weights[ct_name]
        n_ct = sum(1 for i in range(N) if ct_per_patient.get(i) == ct_name)
        if n_ct < 50:
            continue
        print(f"  {ct_name:<35} ", end="")
        for wi in w:
            print(f"{wi:>7.3f}", end="")
        print()

    # CT-conditioned score = weighted sum of channels per patient's CT
    all_scores_ct = np.zeros(N)
    for idx in range(N):
        ct = ct_per_patient[idx]
        w = ct_channel_weights.get(ct, ct_channel_weights["_global"])
        all_scores_ct[idx] = np.dot(all_channel_scores[idx], w)

    # =========================================================================
    # Temporal conditioning — era-adjusted walk scores
    # =========================================================================
    print("\n  Building temporal estimates...")
    from gnn.data.temporal import TemporalEstimator, build_era_baselines

    te = TemporalEstimator(clin)

    # Get estimated enrollment year for each patient (by dataset index)
    clin_indexed = clin.set_index('patientId')
    pid_nums = np.zeros(N)
    for idx in range(N):
        pid = patient_ids[idx] if idx < len(patient_ids) else None
        if pid and pid in clin_indexed.index:
            m = pd.Series([pid]).str.extract(r'P-(\d+)')
            pid_nums[idx] = float(m.iloc[0, 0]) if pd.notna(m.iloc[0, 0]) else 50000
        else:
            pid_nums[idx] = 50000  # middle of range

    est_years, year_confs = te.estimate_by_index(pid_nums)

    # Build era baselines from TRAIN patients only
    train_clin = clin[clin['patientId'].isin(
        {patient_ids[i] for i in train_idx_set if i < len(patient_ids)}
    )]
    baselines, era_baseline_fn = build_era_baselines(train_clin, te, min_patients=30)

    # Show temporal coverage
    print(f"  Enrollment year range: {est_years.min():.1f} - {est_years.max():.1f}")
    print(f"  Year confidence range: {year_confs.min():.1f} - {year_confs.max():.1f} years")
    print(f"  Era baselines available for {len(baselines)} cancer types")

    # Era-adjusted score: walk_ct * era_death_rate
    # Higher era death rate = worse treatment era = walk score matters more
    # Lower era death rate = better treatment = walk score dampened
    all_scores_era = np.zeros(N)
    era_adjustments = np.zeros(N)
    for idx in range(N):
        ct = ct_per_patient[idx]
        yr = est_years[idx]
        era_dr = era_baseline_fn(ct, yr)
        era_adjustments[idx] = era_dr
        all_scores_era[idx] = all_scores_ct[idx] * era_dr

    # Show era adjustment distribution
    print(f"\n  Era adjustment distribution:")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    {pct}th percentile: {np.percentile(era_adjustments, pct):.3f}")

    # Also try: walk + era as additive features (let the ranking sort it out)
    # Normalize both to [0,1] range, then combine
    raw_norm = all_scores_ct.copy()
    raw_max = np.percentile(raw_norm[raw_norm > 0], 99) if (raw_norm > 0).any() else 1.0
    raw_norm = np.clip(raw_norm / raw_max, 0, 1)

    # Era mortality as standalone signal (higher = worse era)
    era_norm = era_adjustments.copy()

    # Combined: walk contributes "how bad are your mutations" (higher = worse)
    # era contributes "how bad is your treatment era" (higher death rate = worse)
    # Simple product captures the interaction
    all_scores_combined = raw_norm * era_norm

    # Also test: additive combination (walks score + era effect)
    all_scores_additive = raw_norm + era_norm

    # =========================================================================
    # Evaluate on HOLDBACK only
    # =========================================================================
    hold_idx = np.array(sorted(holdback_idx))
    hold_events = events[hold_idx]
    hold_times = times[hold_idx]
    hold_valid = hold_times > 0
    hold_eval = hold_idx[hold_valid]

    print(f"\n{'='*90}")
    print(f"  HOLDBACK EVALUATION ({len(hold_eval)} patients)")
    print(f"{'='*90}")

    def ci_on(scores, idx_arr):
        return concordance_index(
            torch.tensor(scores[idx_arr].astype(np.float32)),
            torch.tensor(times[idx_arr].astype(np.float32)),
            torch.tensor(events[idx_arr].astype(np.float32)),
        )

    ci_raw_hold = ci_on(all_scores_raw, hold_eval)
    ci_ct_hold = ci_on(all_scores_ct, hold_eval)
    ci_era_hold = ci_on(all_scores_era, hold_eval)
    ci_combined_hold = ci_on(all_scores_combined, hold_eval)
    ci_additive_hold = ci_on(all_scores_additive, hold_eval)
    ci_era_only = ci_on(era_adjustments, hold_eval)

    print(f"\n  {'Model':<40} {'C-index':>8} {'vs Raw':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8}")
    print(f"  {'Unconditioned walk':<40} {ci_raw_hold:>8.4f} {'':>8}")
    print(f"  {'CT-channel walk':<40} {ci_ct_hold:>8.4f} {ci_ct_hold-ci_raw_hold:>+8.4f}")
    print(f"  {'Era death rate only':<40} {ci_era_only:>8.4f} {ci_era_only-ci_raw_hold:>+8.4f}")
    print(f"  {'CT-channel x era (multiplicative)':<40} {ci_era_hold:>8.4f} {ci_era_hold-ci_raw_hold:>+8.4f}")
    print(f"  {'Normalized walk x era':<40} {ci_combined_hold:>8.4f} {ci_combined_hold-ci_raw_hold:>+8.4f}")
    print(f"  {'Normalized walk + era (additive)':<40} {ci_additive_hold:>8.4f} {ci_additive_hold-ci_raw_hold:>+8.4f}")

    # =========================================================================
    # Per-CT holdback results
    # =========================================================================
    ct_hold_dict = defaultdict(list)
    for idx in hold_eval:
        ct_hold_dict[ct_per_patient[idx]].append(idx)

    # Find best method globally
    methods = {
        'raw': all_scores_raw,
        'ct_channel': all_scores_ct,
        'era_only': era_adjustments,
        'ct_x_era': all_scores_era,
        'norm_x_era': all_scores_combined,
        'norm+era': all_scores_additive,
    }
    best_global = max(methods.keys(),
                      key=lambda m: ci_on(methods[m], hold_eval))

    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE HOLDBACK RESULTS")
    print(f"{'='*90}")
    print(f"\n  {'Cancer Type':<30} {'N':>5} {'Raw':>6} {'CT':>6} {'Era':>6} {'CTxEra':>6} {'N+Era':>6} {'Best':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

    ct_results = {}
    for ct_name in sorted(ct_hold_dict, key=lambda x: -len(ct_hold_dict[x])):
        ct_idx = np.array(ct_hold_dict[ct_name])
        if len(ct_idx) < 30:
            continue

        scores = {}
        for mname, marr in methods.items():
            scores[mname] = ci_on(marr, ct_idx)

        best = max(scores, key=scores.get)
        ct_results[ct_name] = {**scores, "n": len(ct_idx), "best": best}

        print(f"  {ct_name:<30} {len(ct_idx):>5} "
              f"{scores['raw']:>6.3f} {scores['ct_channel']:>6.3f} "
              f"{scores['era_only']:>6.3f} {scores['ct_x_era']:>6.3f} "
              f"{scores['norm+era']:>6.3f} {best:>8}")

    # Summary
    best_counts = defaultdict(int)
    for v in ct_results.values():
        best_counts[v["best"]] += 1
    print(f"\n  Best method distribution:")
    for m, c in sorted(best_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:<20} {c} cancer types")

    # =========================================================================
    # Score distribution (holdback)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  SCORE DISTRIBUTION (holdback)")
    print(f"{'='*90}")

    for label, scores in [("Unconditioned", all_scores_raw), ("CT-conditioned", all_scores_ct)]:
        h_scores = scores[hold_eval]
        print(f"\n  {label}:")
        print(f"    mean={h_scores.mean():.3f}, std={h_scores.std():.3f}")
        print(f"    min={h_scores.min():.3f}, max={h_scores.max():.3f}")
        print(f"    zero: {(h_scores == 0).sum()} ({100*(h_scores==0).sum()/len(h_scores):.1f}%)")

    # =========================================================================
    # Save results
    # =========================================================================
    save_data = {
        "n_holdback": len(hold_eval),
        "n_train": len(train_idx_set),
        "global_results": {
            "raw_walk": float(ci_raw_hold),
            "ct_channel": float(ci_ct_hold),
            "era_only": float(ci_era_only),
            "ct_x_era": float(ci_era_hold),
            "norm_x_era": float(ci_combined_hold),
            "norm_plus_era": float(ci_additive_hold),
        },
        "ct_results": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                           for kk, vv in v.items()}
                       for k, v in ct_results.items()},
        "channel_names": CHANNELS,
        "ct_channel_weights": {k: v.tolist() for k, v in ct_channel_weights.items()},
        "enrollment_year_range": [float(est_years.min()), float(est_years.max())],
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {SAVE_BASE}/results.json")

    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.0f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
