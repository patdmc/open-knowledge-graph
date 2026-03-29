#!/usr/bin/env python3
"""
Pure graph walk scorer — no model dependency.

Computes the same pairwise interaction features as pairwise_graph_scorer.py
but sources ALL data from the Neo4j graph. No transformer checkpoint needed.

Shifts computed from log_hr on HAS_MUTATION edges (survival atlas).
Channel features computed from BELONGS_TO weights + gene channel_profile.

Feature vector per patient (107 features):

  SHIFT FEATURES (4):
    [0]  ct_baseline_log_hr — mean log_hr for this cancer type
    [1]  weighted_ch_shift — CT-specific channel damage shift
    [2]  ct_gene_shift_sum — sum of CT-specific per-gene log_hr shifts
    [3]  global_gene_shift_sum — sum of global per-gene log_hr shifts

  CHANNEL DAMAGE (13):
    [4-11]  channel_damage (8-dim from curated-anchor profiles)
    [12]    total_entropy
    [13]    hub_damage
    [14]    tier_conn
    [15]    n_mutated
    [16]    hhi (damage concentration)

  PAIRWISE FEATURES (22):
    [17-38] PPI distance stats, profile overlap, co-occurrence, hub pairs, etc.

  COMPONENT FEATURES (8):
    [39-46] connected components, isolation, component entropy, damage spread

  CO-OCCURRENCE (4):
    [47-50] n_cooccur, cooccur_wt, max_cooccur, cross_ch_cooccur

  CHANNEL PROFILE FEATURES (8):
    [51-58] per-channel max log_hr across mutated genes

  TRAVERSAL PATTERN (7):
    [59-65] BFS/DFS ratio, depth, linearity, gradient, frontier width

  CLINICAL (5):
    [66-70] age, sex, msi, msi_high, tmb

  GOF/LOF FEATURES (4):
    [71-74] frac_gof, frac_lof, gof_lof_ratio, n_actionable

  Total: 75 features

Usage:
    python3 -u -m gnn.scripts.graph_walk_scorer
"""

import sys, os, json, time
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_FUNCTION, CHANNEL_MAP, CHANNEL_NAMES,
    GENE_TO_CHANNEL_IDX,
)
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from gnn.scripts.focused_multichannel_scorer import (
    compute_curated_profiles, profile_entropy,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "graph_walk_scorer")
N_CH = len(CHANNEL_NAMES)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}

# Tier map: channel → tier index
TIER_MAP = {
    'CellCycle': 0, 'PI3K_Growth': 0,
    'DDR': 1, 'TissueArch': 1,
    'Endocrine': 2, 'Immune': 2,
    'ChromatinRemodel': 3, 'DNAMethylation': 3,
}


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def precompute_ppi_distances(G_ppi, gene_set):
    ppi_genes = gene_set & set(G_ppi.nodes())
    dists = {}
    for g in ppi_genes:
        lengths = nx.single_source_shortest_path_length(G_ppi, g)
        for g2, d in lengths.items():
            if g2 in ppi_genes and g != g2:
                key = tuple(sorted([g, g2]))
                if key not in dists:
                    dists[key] = d
    return dists


def load_graph_data(dataset=None):
    """Load all data from Neo4j for the scorer.

    Args:
        dataset: Filter by dataset property. None = MSK only (no dataset property),
                 'TCGA' = TCGA only, 'all' = everything.
    """
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    patients = {}      # pid → {ct, time, event, age, sex, genes: {gene: {log_hr, direction, protein_change}}}
    gene_log_hr = {}   # (ct, gene) → [log_hr values]
    global_gene_hr = defaultdict(list)

    # Build dataset filter clause
    if dataset == 'all':
        ds_filter = ""
    elif dataset == 'TCGA':
        ds_filter = "AND p.dataset = 'TCGA'"
    else:
        # MSK only — no dataset property
        ds_filter = "AND p.dataset IS NULL"

    print(f"  Loading patients + mutations from Neo4j (dataset={dataset or 'MSK'})...")
    t0 = time.time()

    with driver.session() as s:
        # Patients with mutations
        result = s.run(f"""
            MATCH (p:Patient)-[r:HAS_MUTATION]->(g:Gene)
            WHERE p.os_months IS NOT NULL AND p.event IS NOT NULL AND p.os_months > 0
            {ds_filter}
            RETURN p.id AS pid, p.cancer_type AS ct, p.os_months AS time,
                   p.event AS event, p.tissue_delta AS td,
                   g.name AS gene, r.log_hr AS log_hr, r.direction AS dir,
                   r.protein_change AS pc
        """)
        for r in result:
            pid = r['pid']
            if pid not in patients:
                patients[pid] = {
                    'ct': r['ct'], 'time': float(r['time']),
                    'event': int(r['event']),
                    'td': float(r['td']) if r['td'] is not None else 0.0,
                    'genes': {},
                }
            gene = r['gene']
            log_hr = float(r['log_hr']) if r['log_hr'] is not None else 0.0
            patients[pid]['genes'][gene] = {
                'log_hr': log_hr,
                'direction': r['dir'],
                'protein_change': r['pc'],
            }
            ct = r['ct']
            gene_log_hr.setdefault((ct, gene), []).append(log_hr)
            global_gene_hr[gene].append(log_hr)

        # Clinical features (approximate — MSI, TMB from patient properties if available)
        result = s.run("""
            MATCH (p:Patient)
            WHERE p.os_months IS NOT NULL AND p.event IS NOT NULL AND p.os_months > 0
            RETURN p.id AS pid, p.n_scored_mutations AS nsm
        """)
        for r in result:
            pid = r['pid']
            if pid in patients:
                patients[pid]['n_scored'] = int(r['nsm']) if r['nsm'] is not None else 0

        # Co-occurrence per CT
        cooccur = {}  # (ct, g1, g2) → count
        result = s.run("""
            MATCH (a:Gene)-[r:COOCCURS]->(b:Gene)
            WHERE r.cancer_type IS NOT NULL
            RETURN a.name AS g1, b.name AS g2, r.count AS cnt, r.cancer_type AS ct
        """)
        for r in result:
            key = (r['ct'], *sorted([r['g1'], r['g2']]))
            cooccur[key] = int(r['cnt'])

        # Global co-occurrence (sum across CTs)
        global_cooccur = defaultdict(int)
        for (ct, g1, g2), cnt in cooccur.items():
            global_cooccur[tuple(sorted([g1, g2]))] += cnt

    driver.close()
    print(f"    {len(patients)} patients, {len(cooccur)} CT-cooccur pairs [{time.time()-t0:.1f}s]")

    return patients, gene_log_hr, global_gene_hr, cooccur, global_cooccur


def compute_shifts(patients, gene_log_hr, global_gene_hr):
    """Compute log_hr-based shifts from graph data."""
    # Global baseline
    all_hrs = []
    for p in patients.values():
        for g_info in p['genes'].values():
            all_hrs.append(g_info['log_hr'])
    global_baseline = np.mean(all_hrs) if all_hrs else 0.0

    # Per-CT baseline
    ct_hrs = defaultdict(list)
    for p in patients.values():
        for g_info in p['genes'].values():
            ct_hrs[p['ct']].append(g_info['log_hr'])
    ct_baseline = {ct: np.mean(hrs) for ct, hrs in ct_hrs.items() if len(hrs) >= 50}

    # Per-gene global shift
    global_gene_shift = {}
    for gene, hrs in global_gene_hr.items():
        if len(hrs) >= 20:
            global_gene_shift[gene] = np.mean(hrs) - global_baseline

    # Per-CT per-gene shift
    ct_gene_shift = {}
    for (ct, gene), hrs in gene_log_hr.items():
        bl = ct_baseline.get(ct, global_baseline)
        if len(hrs) >= 15:
            ct_gene_shift[(ct, gene)] = np.mean(hrs) - bl

    # Per-CT per-channel shift
    ct_ch_shift = {}
    ct_ch_hrs = defaultdict(lambda: defaultdict(list))
    for p in patients.values():
        ct = p['ct']
        for gene, g_info in p['genes'].items():
            ch = CHANNEL_MAP.get(gene)
            if ch:
                ct_ch_hrs[ct][ch].append(g_info['log_hr'])
    for ct, ch_map in ct_ch_hrs.items():
        bl = ct_baseline.get(ct, global_baseline)
        for ch, hrs in ch_map.items():
            if len(hrs) >= 20:
                ct_ch_shift[(ct, ch)] = np.mean(hrs) - bl

    return global_baseline, ct_baseline, global_gene_shift, ct_gene_shift, ct_ch_shift


def graph_walk_features(
    patients, channel_profiles, ppi_dists, G_ppi,
    ct_baseline, global_baseline, ct_gene_shift, global_gene_shift, ct_ch_shift,
    cooccur_ct, global_cooccur, hub_gene_set, expanded_cm,
):
    """Compute feature matrix from pure graph data."""
    pid_list = sorted(patients.keys())
    N = len(pid_list)
    FEAT_DIM = 90
    X = np.zeros((N, FEAT_DIM))
    DISCONNECTED_DIST = 10
    ppi_node_set = set(G_ppi.nodes()) if G_ppi else set()
    all_expanded_genes = set(expanded_cm.keys())

    for i, pid in enumerate(pid_list):
        if i % 5000 == 0 and i > 0:
            print(f"    {i}/{N}...")

        p = patients[pid]
        ct = p['ct']
        mutated = set(p['genes'].keys()) & all_expanded_genes
        mutated_list = sorted(mutated)
        n_mut = len(mutated_list)
        ct_bl = ct_baseline.get(ct, global_baseline)

        # === SHIFT FEATURES [0-3] ===
        X[i, 0] = ct_bl
        weighted_ch_shift = 0.0
        ct_gene_sum = 0.0
        global_gene_sum = 0.0
        channel_damage = np.zeros(N_CH)

        for g in mutated:
            profile = channel_profiles.get(g)
            if profile is not None:
                channel_damage += profile
            s = ct_gene_shift.get((ct, g))
            if s is not None:
                ct_gene_sum += s
            gs = global_gene_shift.get(g)
            if gs is not None:
                global_gene_sum += gs

        for ci, ch in enumerate(CHANNEL_NAMES):
            s = ct_ch_shift.get((ct, ch), 0.0)
            weighted_ch_shift += s * channel_damage[ci]

        X[i, 1] = weighted_ch_shift
        X[i, 2] = ct_gene_sum
        X[i, 3] = global_gene_sum

        # === CHANNEL DAMAGE [4-16] ===
        X[i, 4:4+N_CH] = channel_damage
        total = channel_damage.sum()
        X[i, 12] = profile_entropy(channel_damage) if total > 0 else 0.0
        hub_dmg = sum(channel_damage[CH_TO_IDX.get(CHANNEL_MAP.get(g, ''), 0)]
                      for g in mutated if g in hub_gene_set)
        X[i, 13] = hub_dmg
        # Tier connectivity
        tiers_hit = set()
        for g in mutated:
            ch = CHANNEL_MAP.get(g)
            if ch:
                tiers_hit.add(TIER_MAP.get(ch, -1))
        tiers_hit.discard(-1)
        X[i, 14] = len(tiers_hit) / 4.0 if tiers_hit else 0.0
        X[i, 15] = n_mut
        if total > 0:
            fracs = channel_damage / total
            X[i, 16] = float(np.sum(fracs ** 2))  # HHI

        if n_mut < 2:
            # Pairwise/component features stay 0
            # Channel max log_hr
            for g in mutated:
                ch = CHANNEL_MAP.get(g)
                if ch and ch in CH_TO_IDX:
                    ci = CH_TO_IDX[ch]
                    hr = p['genes'][g]['log_hr']
                    X[i, 51 + ci] = max(X[i, 51 + ci], abs(hr))

            # GOF/LOF
            n_gof = sum(1 for g in mutated if GENE_FUNCTION.get(g) == 'GOF')
            n_lof = sum(1 for g in mutated if GENE_FUNCTION.get(g) == 'LOF')
            X[i, 71] = n_gof / max(n_mut, 1)
            X[i, 72] = n_lof / max(n_mut, 1)
            X[i, 73] = (n_gof - n_lof) / max(n_gof + n_lof, 1)

            # Clinical
            X[i, 66] = p.get('td', 0.0)
            X[i, 70] = p.get('n_scored', 0) / 20.0  # TMB proxy
            continue

        # === PAIRWISE FEATURES [17-38] ===
        pairs = list(combinations(mutated_list, 2))
        n_pairs = len(pairs)
        dists = []
        overlaps = []
        same_ch_count = 0
        cross_ch_count = 0
        cooccur_wts = []
        hub_hub = 0
        hub_nonhub = 0
        close_cross = 0
        far_same = 0
        tier_dists = []

        for g1, g2 in pairs:
            key = tuple(sorted([g1, g2]))
            d = ppi_dists.get(key, DISCONNECTED_DIST)
            dists.append(d)

            p1 = channel_profiles.get(g1, np.zeros(N_CH))
            p2 = channel_profiles.get(g2, np.zeros(N_CH))
            overlaps.append(cosine_sim(p1, p2))

            ch1 = CHANNEL_MAP.get(g1, '')
            ch2 = CHANNEL_MAP.get(g2, '')
            if ch1 == ch2 and ch1:
                same_ch_count += 1
            else:
                cross_ch_count += 1

            # CT-specific co-occurrence
            ct_key = (ct, *sorted([g1, g2]))
            cwt = cooccur_ct.get(ct_key, 0)
            if cwt == 0:
                cwt = global_cooccur.get(key, 0)
            cooccur_wts.append(cwt)

            h1 = g1 in hub_gene_set
            h2 = g2 in hub_gene_set
            if h1 and h2:
                hub_hub += 1
            elif h1 or h2:
                hub_nonhub += 1

            if d <= 2 and ch1 != ch2:
                close_cross += 1
            if d >= 4 and ch1 == ch2 and ch1:
                far_same += 1

            t1 = TIER_MAP.get(ch1, -1)
            t2 = TIER_MAP.get(ch2, -1)
            if t1 >= 0 and t2 >= 0:
                tier_dists.append(abs(t1 - t2))

        dists_arr = np.array(dists)
        X[i, 17] = dists_arr.mean()
        X[i, 18] = dists_arr.min()
        connected = dists_arr < DISCONNECTED_DIST
        X[i, 19] = connected.mean()
        X[i, 20] = (dists_arr == 1).mean()
        X[i, 21] = ((dists_arr >= 2) & (dists_arr <= 3)).mean()
        X[i, 22] = (dists_arr >= 4).mean() if connected.any() else 0.0

        overlaps_arr = np.array(overlaps)
        X[i, 23] = overlaps_arr.mean()
        X[i, 24] = overlaps_arr.min()
        X[i, 25] = overlaps_arr.max()

        X[i, 26] = same_ch_count / n_pairs
        X[i, 27] = cross_ch_count / n_pairs

        cwt_arr = np.array(cooccur_wts, dtype=float)
        X[i, 28] = cwt_arr.mean()
        X[i, 29] = cwt_arr.max()
        X[i, 30] = (cwt_arr > 0).mean()

        # Combined entropy
        ents = []
        for g1, g2 in pairs:
            p1 = channel_profiles.get(g1, np.zeros(N_CH))
            p2 = channel_profiles.get(g2, np.zeros(N_CH))
            ents.append(profile_entropy((p1 + p2) / 2))
        X[i, 31] = np.mean(ents) if ents else 0.0

        X[i, 32] = hub_hub
        X[i, 33] = hub_nonhub
        X[i, 34] = close_cross / n_pairs if n_pairs else 0.0
        X[i, 35] = far_same / n_pairs if n_pairs else 0.0
        if tier_dists:
            X[i, 36] = np.mean(tier_dists)
            X[i, 37] = sum(1 for d in tier_dists if d > 0) / len(tier_dists)
        X[i, 38] = n_pairs

        # === COMPONENT FEATURES [39-46] ===
        ppi_mutated = mutated & ppi_node_set
        if ppi_mutated:
            subG = G_ppi.subgraph(ppi_mutated)
            comps = list(nx.connected_components(subG))
            comp_sizes = [len(c) for c in comps]
            n_comps = len(comps)
            largest = max(comp_sizes)
            isolated = sum(1 for s in comp_sizes if s == 1)
            non_ppi = len(mutated - ppi_node_set)
            total_isolated = isolated + non_ppi

            X[i, 39] = n_comps
            X[i, 40] = largest
            X[i, 41] = largest / n_mut
            X[i, 42] = total_isolated
            X[i, 43] = total_isolated / n_mut
            if n_comps > 1:
                p_arr = np.array(comp_sizes) / sum(comp_sizes)
                X[i, 44] = -np.sum(p_arr * np.log(p_arr + 1e-10))
            # Component damage
            comp_damages = []
            for comp in comps:
                cd = sum(channel_damage[CH_TO_IDX.get(CHANNEL_MAP.get(g, ''), 0)]
                         for g in comp if CHANNEL_MAP.get(g))
                comp_damages.append(cd)
            if comp_damages:
                X[i, 45] = max(comp_damages)
                if len(comp_damages) > 1:
                    X[i, 46] = np.std(comp_damages)

        # === CO-OCCURRENCE [47-50] ===
        ct_cooccur_hits = 0
        ct_cooccur_total = 0.0
        ct_cooccur_max = 0.0
        cross_ch_cooccur = 0
        for g1, g2 in pairs:
            key = (ct, *sorted([g1, g2]))
            cnt = cooccur_ct.get(key, 0)
            if cnt > 0:
                ct_cooccur_hits += 1
                ct_cooccur_total += cnt
                ct_cooccur_max = max(ct_cooccur_max, cnt)
                ch1 = CHANNEL_MAP.get(g1, '')
                ch2 = CHANNEL_MAP.get(g2, '')
                if ch1 != ch2:
                    cross_ch_cooccur += 1
        X[i, 47] = ct_cooccur_hits
        X[i, 48] = ct_cooccur_total
        X[i, 49] = ct_cooccur_max
        X[i, 50] = cross_ch_cooccur

        # === CHANNEL MAX LOG_HR [51-58] ===
        for g in mutated:
            ch = CHANNEL_MAP.get(g)
            if ch and ch in CH_TO_IDX:
                ci = CH_TO_IDX[ch]
                hr = abs(p['genes'][g]['log_hr'])
                X[i, 51 + ci] = max(X[i, 51 + ci], hr)

        # === TRAVERSAL PATTERN [59-65] ===
        if ppi_mutated and len(ppi_mutated) >= 2:
            subG = G_ppi.subgraph(ppi_mutated)
            degrees = [subG.degree(n) for n in subG.nodes()]
            max_deg = max(degrees) if degrees else 0
            mean_deg = np.mean(degrees) if degrees else 0

            # BFS-like = broad (high mean degree, multiple components)
            # DFS-like = deep (low branching, long chains)
            comps = list(nx.connected_components(subG))
            n_comps = len(comps)
            largest_comp = max(comps, key=len)
            largest_sub = subG.subgraph(largest_comp)

            # Longest path proxy (diameter of largest component)
            try:
                diameter = nx.diameter(largest_sub)
            except Exception:
                diameter = 0

            X[i, 59] = mean_deg / max(max_deg, 1)  # branching ratio
            X[i, 60] = n_comps / n_mut  # fragmentation
            X[i, 61] = diameter / max(len(largest_comp) - 1, 1)  # linearity
            X[i, 62] = mean_deg  # connectivity density
            # Damage gradient along longest path
            if diameter >= 2:
                try:
                    path = nx.diameter(largest_sub)  # just use diameter
                except Exception:
                    path = 0
                X[i, 63] = diameter  # depth
            X[i, 64] = max_deg  # frontier width
            X[i, 65] = len(ppi_mutated) / n_mut  # PPI coverage

        # === CLINICAL [66-70] ===
        X[i, 66] = p.get('td', 0.0)  # tissue_delta as age proxy
        # sex, msi not in patient node — leave as 0 for now
        X[i, 70] = p.get('n_scored', 0) / 20.0  # TMB proxy

        # === GOF/LOF [71-74] ===
        n_gof = sum(1 for g in mutated if GENE_FUNCTION.get(g) == 'GOF')
        n_lof = sum(1 for g in mutated if GENE_FUNCTION.get(g) == 'LOF')
        n_actionable = sum(1 for g in mutated
                          if GENE_FUNCTION.get(g) in ('GOF', 'LOF'))
        X[i, 71] = n_gof / max(n_mut, 1)
        X[i, 72] = n_lof / max(n_mut, 1)
        X[i, 73] = (n_gof - n_lof) / max(n_gof + n_lof, 1)
        X[i, 74] = n_actionable / max(n_mut, 1)

        # === ESCALATION REGIME FEATURES [75-89] ===
        # These capture DFS vs BFS escalation patterns and the sign-flip
        # interactions discovered in gap analysis.
        #
        # Key insight: features like hub_damage, n_mutated, entropy have
        # OPPOSITE survival correlations depending on the escalation pattern.
        # DFS (deep, focused) = single pathway driven, concentrated lethality
        # BFS (broad, diffuse) = multi-pathway, immune-visible, different regime

        entropy_val = X[i, 12]      # channel damage entropy
        hub_dmg_val = X[i, 13]      # hub damage
        tier_val = X[i, 14]         # tier connectivity (0-1)
        hhi_val = X[i, 16]          # damage concentration
        frac_conn = X[i, 19]        # fraction connected pairs
        frac_same = X[i, 26]        # fraction same-channel pairs
        frac_act = X[i, 74]         # fraction actionable

        # Escalation type: DFS-like (focused, deep) vs BFS-like (broad, shallow)
        # DFS signal: high HHI, low entropy, high same-channel, low tier span
        # BFS signal: low HHI, high entropy, high cross-channel, high tier span
        dfs_score = hhi_val - entropy_val / 3.0 + frac_same - tier_val
        bfs_score = entropy_val / 3.0 - hhi_val + (1.0 - frac_same) + tier_val
        X[i, 75] = dfs_score
        X[i, 76] = bfs_score

        # Sign-flip interaction: hub_damage × escalation type
        # Hub hit in DFS = devastating (single pathway collapse)
        # Hub hit in BFS = targetable (immune-visible, spread thin)
        X[i, 77] = hub_dmg_val * dfs_score   # hub damage in focused escalation
        X[i, 78] = hub_dmg_val * bfs_score   # hub damage in broad escalation

        # Sign-flip interaction: n_mutated × tier span
        # Many mutations single-tier = concentrated lethal hit (BFS within tier)
        # Many mutations multi-tier = systemic, better prognosis
        X[i, 79] = n_mut * tier_val          # mutations × tier span
        X[i, 80] = n_mut * (1.0 - tier_val)  # mutations × single-tier

        # Sign-flip interaction: n_mutated × entropy
        # More mutations + focused = DFS escalation (worse)
        # More mutations + diffuse = BFS escalation (better)
        X[i, 81] = n_mut * entropy_val       # mutations × diffuse
        X[i, 82] = n_mut * hhi_val           # mutations × concentrated

        # Hub escalation depth: hub damage relative to total
        # Secondary hub hit (hub damage > 1 hub) = escalation beyond first hit
        n_hub_mut = sum(1 for g in mutated if g in hub_gene_set)
        X[i, 83] = max(0, n_hub_mut - 1)    # secondary hub hits (escalation)
        X[i, 84] = n_hub_mut * dfs_score     # hub count × DFS

        # Isolation × actionability flip
        # Isolated mutations in known genes = clear signal
        # Isolated mutations in unknown genes = noise
        frac_iso = X[i, 43]
        X[i, 85] = frac_iso * frac_act       # isolated + actionable
        X[i, 86] = frac_iso * (1.0 - frac_act)  # isolated + unknown

        # Connected hub-hub in BFS = multi-pathway bridge (escalation signal)
        hub_hub_val = X[i, 32]
        X[i, 87] = hub_hub_val * bfs_score   # hub-hub in broad escalation

        # PPI connectivity × escalation type
        X[i, 88] = frac_conn * dfs_score     # connected + focused = pathway chain
        X[i, 89] = frac_conn * bfs_score     # connected + broad = network damage

    return X, pid_list


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("  PURE GRAPH WALK SCORER (no model dependency)")
    print("=" * 80)

    # Load everything from Neo4j
    patients, gene_log_hr, global_gene_hr, cooccur_ct, global_cooccur = load_graph_data()

    # Compute shifts from log_hr
    print("  Computing shifts from log_hr...")
    global_baseline, ct_baseline, global_gene_shift, ct_gene_shift, ct_ch_shift = \
        compute_shifts(patients, gene_log_hr, global_gene_hr)
    print(f"    CT baselines: {len(ct_baseline)}")
    print(f"    CT-gene shifts: {len(ct_gene_shift)}")
    print(f"    Global gene shifts: {len(global_gene_shift)}")
    print(f"    CT-channel shifts: {len(ct_ch_shift)}")

    # Build PPI graph directly from Neo4j + cached STRING edges
    print("  Building PPI graph...")
    expanded_cm = dict(CHANNEL_MAP)  # use current config, not stale file
    all_genes = set()
    for p in patients.values():
        all_genes |= set(p['genes'].keys())
    all_genes &= set(expanded_cm.keys())

    G_ppi = nx.Graph()
    for gene, ch in expanded_cm.items():
        G_ppi.add_node(gene, channel=ch, tier=TIER_MAP.get(ch, -1))

    # Load STRING PPI edges from cache
    ppi_cache = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    if os.path.exists(ppi_cache):
        with open(ppi_cache) as f:
            ppi_data = json.load(f)
        for edge in ppi_data:
            g1, g2 = edge[0], edge[1]
            score = float(edge[2]) if len(edge) > 2 else 0.5
            if g1 in expanded_cm and g2 in expanded_cm:
                G_ppi.add_edge(g1, g2, weight=score)
    print(f"    PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # Curated profiles
    print("  Computing curated-anchor profiles...")
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)

    # PPI distances
    print("  Precomputing PPI distances...")
    ppi_dists = precompute_ppi_distances(G_ppi, all_genes)
    print(f"    {len(ppi_dists)} pairwise distances")

    # Hub gene set
    hub_gene_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_gene_set |= ch_hubs

    # Compute features
    print(f"\n  Computing 90-dim graph walk features for {len(patients)} patients...")
    X, pid_list = graph_walk_features(
        patients, channel_profiles, ppi_dists, G_ppi,
        ct_baseline, global_baseline, ct_gene_shift, global_gene_shift, ct_ch_shift,
        cooccur_ct, global_cooccur, hub_gene_set, expanded_cm,
    )
    print(f"    Features: {X.shape}")

    # Prepare for CV
    N = len(pid_list)
    times = np.array([patients[pid]['time'] for pid in pid_list])
    events = np.array([patients[pid]['event'] for pid in pid_list])
    ct_per_patient = {i: patients[pid]['ct'] for i, pid in enumerate(pid_list)}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))

    # =========================================================================
    # Global ridge with alpha sweep
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  GLOBAL RIDGE (alpha sweep)")
    print(f"{'='*80}")

    best_alpha = 1.0
    best_c = 0.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        scores = np.zeros(N)
        for train_idx, val_idx in folds:
            reg = Ridge(alpha=alpha)
            reg.fit(X[train_idx], times[train_idx] * (1 - events[train_idx] * 0.5))
            scores[val_idx] = reg.predict(X[val_idx])
        # Concordance (predict negative risk = longer survival)
        fold_cis = []
        for train_idx, val_idx in folds:
            ci = concordance_index(
                torch.tensor(-scores[val_idx].astype(np.float32)),
                torch.tensor(times[val_idx].astype(np.float32)),
                torch.tensor(events[val_idx].astype(np.float32)),
            )
            fold_cis.append(ci)
        mean_ci = np.mean(fold_cis)
        print(f"    alpha={alpha:>10.2f}  C={mean_ci:.4f}")
        if mean_ci > best_c:
            best_c = mean_ci
            best_alpha = alpha

    print(f"\n    Best: alpha={best_alpha}, C={best_c:.4f}")

    # =========================================================================
    # Per-CT ridge
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PER-CT RIDGE (alpha={best_alpha})")
    print(f"{'='*80}")

    scores_perct = np.zeros(N)
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Group train by CT
        ct_train = defaultdict(list)
        for idx in train_idx:
            ct_train[ct_per_patient[idx]].append(idx)

        # Global fallback
        reg_global = Ridge(alpha=best_alpha)
        y_train = times[train_idx] * (1 - events[train_idx] * 0.5)
        reg_global.fit(X[train_idx], y_train)

        # Per-CT models
        ct_models = {}
        for ct_name, ct_indices in ct_train.items():
            if len(ct_indices) >= 200:
                ct_arr = np.array(ct_indices)
                reg_ct = Ridge(alpha=best_alpha)
                y_ct = times[ct_arr] * (1 - events[ct_arr] * 0.5)
                reg_ct.fit(X[ct_arr], y_ct)
                ct_models[ct_name] = reg_ct

        # Predict
        for idx in val_idx:
            ct_name = ct_per_patient[idx]
            if ct_name in ct_models:
                scores_perct[idx] = ct_models[ct_name].predict(X[idx:idx+1])[0]
            else:
                scores_perct[idx] = reg_global.predict(X[idx:idx+1])[0]

    # Evaluate
    fold_cis = []
    ct_cis = defaultdict(lambda: {'scores': [], 'times': [], 'events': []})
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        ci = concordance_index(
            torch.tensor(-scores_perct[val_idx].astype(np.float32)),
            torch.tensor(times[val_idx].astype(np.float32)),
            torch.tensor(events[val_idx].astype(np.float32)),
        )
        fold_cis.append(ci)
        for idx in val_idx:
            ct = ct_per_patient[idx]
            ct_cis[ct]['scores'].append(-scores_perct[idx])
            ct_cis[ct]['times'].append(times[idx])
            ct_cis[ct]['events'].append(events[idx])

    mean_ci = np.mean(fold_cis)
    std_ci = np.std(fold_cis)
    folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
    print(f"\n  Per-CT Ridge: {mean_ci:.4f} +/- {std_ci:.4f}  [{folds_str}]")

    # Per-CT breakdown
    print(f"\n  {'Cancer Type':<40} {'N':>6} {'C':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*8}")
    ct_results = []
    for ct_name in sorted(ct_cis.keys()):
        d = ct_cis[ct_name]
        if len(d['scores']) >= 50 and sum(d['events']) >= 5:
            ci = concordance_index(
                torch.tensor(np.array(d['scores'], dtype=np.float32)),
                torch.tensor(np.array(d['times'], dtype=np.float32)),
                torch.tensor(np.array(d['events'], dtype=np.float32)),
            )
            ct_results.append((ct_name, len(d['scores']), ci))

    for ct_name, n, ci in sorted(ct_results, key=lambda x: -x[2]):
        print(f"  {ct_name:<40} {n:>6} {ci:>8.4f}")

    # Feature importance
    print(f"\n{'='*80}")
    print(f"  FEATURE IMPORTANCE")
    print(f"{'='*80}")

    reg_full = Ridge(alpha=best_alpha)
    y_full = times * (1 - events * 0.5)
    reg_full.fit(X, y_full)

    feat_names = (
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

    coefs = reg_full.coef_
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    print(f"\n  {'Feature':<25} {'Coeff':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for j in sorted_idx[:20]:
        name = feat_names[j] if j < len(feat_names) else f"[{j}]"
        print(f"  {name:<25} {coefs[j]:>+10.4f}")

    # Save results
    results = {
        'global_c': best_c,
        'global_alpha': best_alpha,
        'perct_c': float(mean_ci),
        'perct_std': float(std_ci),
        'perct_folds': [float(c) for c in fold_cis],
        'n_patients': N,
        'n_features': X.shape[1],
        'ct_results': {ct: {'n': n, 'c': float(ci)} for ct, n, ci in ct_results},
    }
    with open(os.path.join(SAVE_BASE, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Results saved to {SAVE_BASE}/results.json")


if __name__ == "__main__":
    import torch
    main()
