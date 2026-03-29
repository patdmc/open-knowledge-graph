#!/usr/bin/env python3
"""
TCGA graph walk scorer — richer features, smaller cohort.

Same graph topology (PPI, channels, blocks) as MSK scorer, but adds
TCGA-specific clinical features: stage, MSI, FGA, aneuploidy, TMB,
neoadjuvant treatment, radiation, hypoxia scores.

Also uses expression-derived and CNA-derived edges from the graph
to enrich per-patient features.

Usage:
    python3 -u -m gnn.scripts.tcga_graph_scorer
"""

import sys, os, json, time
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE,
    HUB_GENES, GENE_FUNCTION, CHANNEL_MAP, CHANNEL_NAMES,
)
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from gnn.scripts.focused_multichannel_scorer import (
    compute_curated_profiles, profile_entropy,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "tcga_graph_scorer")
N_CH = len(CHANNEL_NAMES)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
TIER_MAP = {
    'CellCycle': 0, 'PI3K_Growth': 0,
    'DDR': 1, 'TissueArch': 1,
    'Endocrine': 2, 'Immune': 2,
    'ChromatinRemodel': 3, 'DNAMethylation': 3,
}

TCGA_CACHE = os.path.join(GNN_CACHE, "tcga")


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


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


def load_tcga_data():
    """Load TCGA patients from CSV + graph edges from Neo4j."""
    t0 = time.time()

    # Clinical
    clin = pd.read_csv(os.path.join(TCGA_CACHE, "tcga_clinical_detail.csv"))
    clin = clin[clin['OS_MONTHS'].notna() & clin['OS_STATUS'].notna()].copy()
    clin['time'] = clin['OS_MONTHS'].astype(float)
    clin['event'] = clin['OS_STATUS'].apply(
        lambda x: 1 if 'DECEASED' in str(x).upper() else 0)
    clin = clin[clin['time'] > 0].copy()

    # Mutations
    mut = pd.read_csv(os.path.join(TCGA_CACHE, "tcga_mutations.csv"))
    atlas_genes = set(CHANNEL_MAP.keys())

    # Build patient dict
    patients = {}
    for _, row in clin.iterrows():
        pid = row['patient_id']
        # Stage encoding
        stage_str = str(row.get('stage', '')).upper()
        stage_num = 0
        if 'IV' in stage_str:
            stage_num = 4
        elif 'III' in stage_str:
            stage_num = 3
        elif 'II' in stage_str:
            stage_num = 2
        elif 'I' in stage_str:
            stage_num = 1

        patients[pid] = {
            'ct': row.get('CANCER_TYPE_ACRONYM', row.get('cancer_type', 'UNK')),
            'time': float(row['time']),
            'event': int(row['event']),
            'genes': {},
            # Rich clinical features
            'age': float(row['age']) / 100.0 if pd.notna(row.get('age')) else 0.5,
            'sex': 1.0 if str(row.get('sex', '')).upper() == 'MALE' else 0.0,
            'stage': stage_num / 4.0,
            'msi_score': float(row['MSI_SCORE_MANTIS']) if pd.notna(row.get('MSI_SCORE_MANTIS')) else 0.0,
            'msi_high': 1.0 if pd.notna(row.get('MSI_SCORE_MANTIS')) and float(row['MSI_SCORE_MANTIS']) > 0.4 else 0.0,
            'fga': float(row['FRACTION_GENOME_ALTERED']) if pd.notna(row.get('FRACTION_GENOME_ALTERED')) else 0.0,
            'aneuploidy': float(row['ANEUPLOIDY_SCORE']) / 40.0 if pd.notna(row.get('ANEUPLOIDY_SCORE')) else 0.0,
            'tmb': float(row['TMB_NONSYNONYMOUS']) / 50.0 if pd.notna(row.get('TMB_NONSYNONYMOUS')) else 0.0,
            'mutation_count': float(row['MUTATION_COUNT']) / 100.0 if pd.notna(row.get('MUTATION_COUNT')) else 0.0,
            'neoadjuvant': 1.0 if str(row.get('HISTORY_NEOADJUVANT_TRTYN', '')).upper() == 'YES' else 0.0,
            'radiation': 1.0 if str(row.get('RADIATION_THERAPY', '')).upper() == 'YES' else 0.0,
            'hypoxia_buffa': float(row['BUFFA_HYPOXIA_SCORE']) if pd.notna(row.get('BUFFA_HYPOXIA_SCORE')) else 0.0,
            'hypoxia_ragnum': float(row['RAGNUM_HYPOXIA_SCORE']) if pd.notna(row.get('RAGNUM_HYPOXIA_SCORE')) else 0.0,
            # DFS/PFS outcomes (for multi-endpoint analysis)
            'dfs_months': float(row['DFS_MONTHS']) if pd.notna(row.get('DFS_MONTHS')) else None,
            'dfs_event': int(row['DFS_STATUS'].startswith('1')) if pd.notna(row.get('DFS_STATUS')) else None,
            'pfs_months': float(row['PFS_MONTHS']) if pd.notna(row.get('PFS_MONTHS')) else None,
            'pfs_event': int(row['PFS_STATUS'].startswith('1')) if pd.notna(row.get('PFS_STATUS')) else None,
        }

    # Add mutations
    for _, row in mut.iterrows():
        pid = row['patient_id']
        gene = row['gene']
        if pid in patients and gene in atlas_genes:
            patients[pid]['genes'][gene] = {
                'protein_change': row.get('protein_change', ''),
                'mutation_type': row.get('mutation_type', ''),
                'direction': GENE_FUNCTION.get(gene, 'unknown'),
            }

    # Load PROGNOSTIC_IN edges from Neo4j for log_hr
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    gene_ct_hr = {}  # (gene, ct) → hr
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)-[r:PROGNOSTIC_IN]->(c:CancerType)
            RETURN g.name AS gene, c.name AS ct, r.hr AS hr
        """)
        for r in result:
            gene_ct_hr[(r['gene'], r['ct'])] = float(r['hr'])

        # Co-occurrence per CT
        cooccur = {}
        result = s.run("""
            MATCH (a:Gene)-[r:COOCCURS]->(b:Gene)
            WHERE r.cancer_type IS NOT NULL
            RETURN a.name AS g1, b.name AS g2, r.count AS cnt,
                   r.cancer_type AS ct
        """)
        for r in result:
            key = (r['ct'], *sorted([r['g1'], r['g2']]))
            cooccur[key] = int(r['cnt'])

        # Expression edges (TCGA-specific)
        expr_edges = {}
        result = s.run("""
            MATCH (g:Gene)-[r:EXPRESSION_IN]->(c:CancerType)
            RETURN g.name AS gene, c.name AS ct, r.direction AS dir,
                   r.z_score AS z
        """)
        for r in result:
            expr_edges[(r['gene'], r['ct'])] = {
                'direction': r['dir'],
                'z_score': float(r['z']) if r['z'] else 0.0,
            }

        # CNA edges
        cna_edges = {}
        result = s.run("""
            MATCH (g:Gene)-[r:CNA_IN]->(c:CancerType)
            RETURN g.name AS gene, c.name AS ct, r.direction AS dir,
                   r.amp_freq AS amp, r.del_freq AS del
        """)
        for r in result:
            cna_edges[(r['gene'], r['ct'])] = {
                'direction': r['dir'],
                'amp_freq': float(r['amp']) if r['amp'] else 0.0,
                'del_freq': float(r['del']) if r['del'] else 0.0,
            }

    driver.close()

    # Assign log_hr to patient mutations (CT-specific where available)
    # Map TCGA CT acronyms to graph CancerType names
    ct_name_map = {}
    for p in patients.values():
        ct_name_map[p['ct']] = p['ct']

    for pid, p in patients.items():
        for gene in p['genes']:
            # Try CT-specific HR first
            hr = gene_ct_hr.get((gene, p['ct']))
            if hr and hr > 0:
                p['genes'][gene]['log_hr'] = np.log(hr)
            else:
                # Try by full CT name
                p['genes'][gene]['log_hr'] = 0.0

    # Global co-occurrence
    global_cooccur = defaultdict(int)
    for (ct, g1, g2), cnt in cooccur.items():
        global_cooccur[tuple(sorted([g1, g2]))] += cnt

    print(f"    {len(patients)} patients, {sum(len(p['genes']) for p in patients.values())} mutations")
    print(f"    PROGNOSTIC_IN: {len(gene_ct_hr)} gene-CT pairs")
    print(f"    EXPRESSION_IN: {len(expr_edges)} gene-CT pairs")
    print(f"    CNA_IN: {len(cna_edges)} gene-CT pairs")
    print(f"    COOCCURS: {len(cooccur)} CT-specific pairs")
    print(f"    [{time.time()-t0:.1f}s]")

    return patients, cooccur, global_cooccur, expr_edges, cna_edges


def tcga_features(patients, channel_profiles, ppi_dists, G_ppi,
                  cooccur_ct, global_cooccur, expr_edges, cna_edges,
                  hub_gene_set):
    """Feature matrix for TCGA patients.

    Features (95):
      [0-3]    shifts (CT baseline, channel, gene-level)
      [4-16]   channel damage + topology
      [17-38]  pairwise graph features
      [39-46]  component features
      [47-50]  co-occurrence
      [51-58]  channel max log_hr
      [59-65]  traversal pattern
      [66-78]  TCGA-rich clinical (13): age, sex, stage, msi_score, msi_high,
               fga, aneuploidy, tmb, mutation_count, neoadjuvant, radiation,
               hypoxia_buffa, hypoxia_ragnum
      [79-82]  GOF/LOF features
      [83-90]  expression context per channel (TCGA-specific)
      [91-94]  CNA context: amp_burden, del_burden, amp_channels, del_channels
    """
    pid_list = sorted(patients.keys())
    N = len(pid_list)
    FEAT_DIM = 95
    X = np.zeros((N, FEAT_DIM))
    DISCONNECTED_DIST = 10
    ppi_node_set = set(G_ppi.nodes()) if G_ppi else set()
    atlas_genes = set(CHANNEL_MAP.keys())

    # Compute shifts from PROGNOSTIC_IN log_hr
    ct_hrs = defaultdict(list)
    global_gene_hr = defaultdict(list)
    for p in patients.values():
        for g, info in p['genes'].items():
            ct_hrs[p['ct']].append(info.get('log_hr', 0.0))
            global_gene_hr[g].append(info.get('log_hr', 0.0))

    global_baseline = np.mean([h for hrs in ct_hrs.values() for h in hrs]) if ct_hrs else 0.0
    ct_baseline = {ct: np.mean(hrs) for ct, hrs in ct_hrs.items() if len(hrs) >= 30}

    global_gene_shift = {g: np.mean(hrs) - global_baseline
                         for g, hrs in global_gene_hr.items() if len(hrs) >= 10}

    ct_gene_hr = defaultdict(lambda: defaultdict(list))
    for p in patients.values():
        for g, info in p['genes'].items():
            ct_gene_hr[p['ct']][g].append(info.get('log_hr', 0.0))

    ct_gene_shift = {}
    for ct, gene_hrs in ct_gene_hr.items():
        bl = ct_baseline.get(ct, global_baseline)
        for g, hrs in gene_hrs.items():
            if len(hrs) >= 5:
                ct_gene_shift[(ct, g)] = np.mean(hrs) - bl

    for i, pid in enumerate(pid_list):
        if i % 2000 == 0 and i > 0:
            print(f"    {i}/{N}...")

        p = patients[pid]
        ct = p['ct']
        mutated = set(p['genes'].keys()) & atlas_genes
        mutated_list = sorted(mutated)
        n_mut = len(mutated_list)
        ct_bl = ct_baseline.get(ct, global_baseline)

        # === SHIFTS [0-3] ===
        X[i, 0] = ct_bl
        channel_damage = np.zeros(N_CH)
        ct_gene_sum = 0.0
        global_gene_sum = 0.0

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

        X[i, 1] = 0.0  # weighted_ch_shift (simplified)
        X[i, 2] = ct_gene_sum
        X[i, 3] = global_gene_sum

        # === CHANNEL DAMAGE [4-16] ===
        X[i, 4:4+N_CH] = channel_damage
        total = channel_damage.sum()
        X[i, 12] = profile_entropy(channel_damage) if total > 0 else 0.0
        hub_dmg = sum(1 for g in mutated if g in hub_gene_set)
        X[i, 13] = hub_dmg
        tiers_hit = set()
        for g in mutated:
            ch = CHANNEL_MAP.get(g)
            if ch:
                tiers_hit.add(TIER_MAP.get(ch, -1))
        tiers_hit.discard(-1)
        X[i, 14] = len(tiers_hit) / 4.0
        X[i, 15] = n_mut
        if total > 0:
            fracs = channel_damage / total
            X[i, 16] = float(np.sum(fracs ** 2))

        # === PAIRWISE [17-38] ===
        if n_mut >= 2:
            pairs = list(combinations(mutated_list, 2))
            n_pairs = len(pairs)
            dists, overlaps, cooccur_wts = [], [], []
            same_ch, cross_ch, hub_hub, hub_nonhub = 0, 0, 0, 0
            close_cross, far_same = 0, 0
            tier_dists = []

            for g1, g2 in pairs:
                key = tuple(sorted([g1, g2]))
                d = ppi_dists.get(key, DISCONNECTED_DIST)
                dists.append(d)
                p1 = channel_profiles.get(g1, np.zeros(N_CH))
                p2 = channel_profiles.get(g2, np.zeros(N_CH))
                overlaps.append(cosine_sim(p1, p2))

                ch1, ch2 = CHANNEL_MAP.get(g1, ''), CHANNEL_MAP.get(g2, '')
                if ch1 == ch2 and ch1:
                    same_ch += 1
                else:
                    cross_ch += 1

                ct_key = (ct, *sorted([g1, g2]))
                cwt = cooccur_ct.get(ct_key, global_cooccur.get(key, 0))
                cooccur_wts.append(cwt)

                h1, h2 = g1 in hub_gene_set, g2 in hub_gene_set
                if h1 and h2: hub_hub += 1
                elif h1 or h2: hub_nonhub += 1
                if d <= 2 and ch1 != ch2: close_cross += 1
                if d >= 4 and ch1 == ch2 and ch1: far_same += 1

                t1 = TIER_MAP.get(ch1, -1)
                t2 = TIER_MAP.get(ch2, -1)
                if t1 >= 0 and t2 >= 0:
                    tier_dists.append(abs(t1 - t2))

            da = np.array(dists)
            X[i, 17] = da.mean()
            X[i, 18] = da.min()
            connected = da < DISCONNECTED_DIST
            X[i, 19] = connected.mean()
            X[i, 20] = (da == 1).mean()
            X[i, 21] = ((da >= 2) & (da <= 3)).mean()
            X[i, 22] = (da >= 4).mean() if connected.any() else 0.0
            oa = np.array(overlaps)
            X[i, 23] = oa.mean()
            X[i, 24] = oa.min()
            X[i, 25] = oa.max()
            X[i, 26] = same_ch / n_pairs
            X[i, 27] = cross_ch / n_pairs
            ca = np.array(cooccur_wts, dtype=float)
            X[i, 28] = ca.mean()
            X[i, 29] = ca.max()
            X[i, 30] = (ca > 0).mean()
            ents = [profile_entropy((channel_profiles.get(g1, np.zeros(N_CH)) +
                                     channel_profiles.get(g2, np.zeros(N_CH))) / 2)
                    for g1, g2 in pairs]
            X[i, 31] = np.mean(ents) if ents else 0.0
            X[i, 32] = hub_hub
            X[i, 33] = hub_nonhub
            X[i, 34] = close_cross / n_pairs if n_pairs else 0.0
            X[i, 35] = far_same / n_pairs if n_pairs else 0.0
            if tier_dists:
                X[i, 36] = np.mean(tier_dists)
                X[i, 37] = sum(1 for d in tier_dists if d > 0) / len(tier_dists)
            X[i, 38] = n_pairs

            # === COMPONENT [39-46] ===
            ppi_mut = mutated & ppi_node_set
            if ppi_mut:
                subG = G_ppi.subgraph(ppi_mut)
                comps = list(nx.connected_components(subG))
                sizes = [len(c) for c in comps]
                largest = max(sizes)
                isolated = sum(1 for s in sizes if s == 1) + len(mutated - ppi_node_set)
                X[i, 39] = len(comps)
                X[i, 40] = largest
                X[i, 41] = largest / n_mut
                X[i, 42] = isolated
                X[i, 43] = isolated / n_mut
                if len(comps) > 1:
                    pa = np.array(sizes) / sum(sizes)
                    X[i, 44] = -np.sum(pa * np.log(pa + 1e-10))
                comp_dmg = [sum(channel_damage[CH_TO_IDX.get(CHANNEL_MAP.get(g, ''), 0)]
                               for g in c if CHANNEL_MAP.get(g)) for c in comps]
                if comp_dmg:
                    X[i, 45] = max(comp_dmg)
                    if len(comp_dmg) > 1:
                        X[i, 46] = np.std(comp_dmg)

        # === CO-OCCURRENCE [47-50] ===
        if n_mut >= 2:
            hits, total_w, max_w, cross_cooccur = 0, 0.0, 0.0, 0
            for g1, g2 in combinations(mutated_list, 2):
                key = (ct, *sorted([g1, g2]))
                cnt = cooccur_ct.get(key, 0)
                if cnt > 0:
                    hits += 1
                    total_w += cnt
                    max_w = max(max_w, cnt)
                    if CHANNEL_MAP.get(g1) != CHANNEL_MAP.get(g2):
                        cross_cooccur += 1
            X[i, 47] = hits
            X[i, 48] = total_w
            X[i, 49] = max_w
            X[i, 50] = cross_cooccur

        # === CHANNEL MAX LOG_HR [51-58] ===
        for g in mutated:
            ch = CHANNEL_MAP.get(g)
            if ch and ch in CH_TO_IDX:
                ci = CH_TO_IDX[ch]
                hr = abs(p['genes'][g].get('log_hr', 0.0))
                X[i, 51 + ci] = max(X[i, 51 + ci], hr)

        # === TRAVERSAL [59-65] (simplified) ===
        ppi_mut = mutated & ppi_node_set
        if len(ppi_mut) >= 2:
            subG = G_ppi.subgraph(ppi_mut)
            degs = [subG.degree(n) for n in subG.nodes()]
            max_d = max(degs) if degs else 0
            mean_d = np.mean(degs) if degs else 0
            X[i, 59] = mean_d / max(max_d, 1)
            comps = list(nx.connected_components(subG))
            X[i, 60] = len(comps) / n_mut
            X[i, 62] = mean_d
            X[i, 64] = max_d
            X[i, 65] = len(ppi_mut) / n_mut

        # === TCGA-RICH CLINICAL [66-78] ===
        X[i, 66] = p['age']
        X[i, 67] = p['sex']
        X[i, 68] = p['stage']
        X[i, 69] = p['msi_score']
        X[i, 70] = p['msi_high']
        X[i, 71] = p['fga']
        X[i, 72] = p['aneuploidy']
        X[i, 73] = p['tmb']
        X[i, 74] = p['mutation_count']
        X[i, 75] = p['neoadjuvant']
        X[i, 76] = p['radiation']
        X[i, 77] = p['hypoxia_buffa']
        X[i, 78] = p['hypoxia_ragnum']

        # === GOF/LOF [79-82] ===
        n_gof = sum(1 for g in mutated if GENE_FUNCTION.get(g) == 'GOF')
        n_lof = sum(1 for g in mutated if GENE_FUNCTION.get(g) == 'LOF')
        X[i, 79] = n_gof / max(n_mut, 1)
        X[i, 80] = n_lof / max(n_mut, 1)
        X[i, 81] = (n_gof - n_lof) / max(n_gof + n_lof, 1)
        X[i, 82] = (n_gof + n_lof) / max(n_mut, 1)

        # === EXPRESSION CONTEXT [83-90] ===
        for g in mutated:
            einfo = expr_edges.get((g, ct))
            if einfo:
                ch = CHANNEL_MAP.get(g)
                if ch and ch in CH_TO_IDX:
                    ci = CH_TO_IDX[ch]
                    z = einfo['z_score']
                    X[i, 83 + ci] = max(X[i, 83 + ci], abs(z))

        # === CNA CONTEXT [91-94] ===
        amp_burden, del_burden = 0.0, 0.0
        amp_chs, del_chs = set(), set()
        for g in mutated:
            cinfo = cna_edges.get((g, ct))
            if cinfo:
                amp_burden += cinfo['amp_freq']
                del_burden += cinfo['del_freq']
                ch = CHANNEL_MAP.get(g)
                if cinfo['amp_freq'] > 0.05:
                    amp_chs.add(ch)
                if cinfo['del_freq'] > 0.05:
                    del_chs.add(ch)
        X[i, 91] = amp_burden
        X[i, 92] = del_burden
        X[i, 93] = len(amp_chs)
        X[i, 94] = len(del_chs)

    return X, pid_list


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t_start = time.time()
    import torch

    print("=" * 80)
    print("  TCGA GRAPH WALK SCORER (rich clinical features)")
    print("=" * 80)

    # Load data
    print("  Loading TCGA data + graph edges...")
    patients, cooccur_ct, global_cooccur, expr_edges, cna_edges = load_tcga_data()

    # PPI graph
    print("  Building PPI graph...")
    G_ppi = nx.Graph()
    for gene, ch in CHANNEL_MAP.items():
        G_ppi.add_node(gene, channel=ch)
    ppi_cache = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    with open(ppi_cache) as f:
        for edge in json.load(f):
            g1, g2 = edge[0], edge[1]
            score = float(edge[2]) if len(edge) > 2 else 0.5
            if g1 in CHANNEL_MAP and g2 in CHANNEL_MAP:
                G_ppi.add_edge(g1, g2, weight=score)
    print(f"    PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # Profiles + distances
    print("  Computing profiles + distances...")
    channel_profiles = compute_curated_profiles(G_ppi, dict(CHANNEL_MAP))
    all_genes = set()
    for p in patients.values():
        all_genes |= set(p['genes'].keys())
    all_genes &= set(CHANNEL_MAP.keys())
    ppi_dists = precompute_ppi_distances(G_ppi, all_genes)
    print(f"    {len(ppi_dists)} distances")

    hub_gene_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_gene_set |= ch_hubs

    # Features
    print(f"\n  Computing 95-dim features for {len(patients)} TCGA patients...")
    X, pid_list = tcga_features(
        patients, channel_profiles, ppi_dists, G_ppi,
        cooccur_ct, global_cooccur, expr_edges, cna_edges, hub_gene_set,
    )
    print(f"    Features: {X.shape}")

    N = len(pid_list)
    times = np.array([patients[pid]['time'] for pid in pid_list])
    events = np.array([patients[pid]['event'] for pid in pid_list])
    ct_per = {i: patients[pid]['ct'] for i, pid in enumerate(pid_list)}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))

    # =========================================================================
    # Ablation: graph-only vs graph+clinical vs full
    # =========================================================================
    graph_idx = list(range(0, 66))  # shifts + channel + pairwise + component + cooccur + traversal
    clinical_idx = list(range(66, 79))  # TCGA-rich clinical
    gof_idx = list(range(79, 83))
    expr_idx = list(range(83, 91))
    cna_idx = list(range(91, 95))

    configs = {
        "graph only (same as MSK)": graph_idx,
        "graph + GOF/LOF": graph_idx + gof_idx,
        "graph + clinical": graph_idx + clinical_idx,
        "graph + clinical + GOF": graph_idx + clinical_idx + gof_idx,
        "graph + clinical + expr + CNA": graph_idx + clinical_idx + expr_idx + cna_idx,
        "FULL (all 95)": list(range(95)),
        "clinical ONLY (no graph)": clinical_idx,
        "clinical + GOF + expr + CNA": clinical_idx + gof_idx + expr_idx + cna_idx,
    }

    print(f"\n{'='*80}")
    print(f"  ABLATION: TCGA GRAPH SCORER")
    print(f"{'='*80}")
    print(f"\n  {'Config':<40} {'Global C':>8} {'Per-CT C':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8}")

    results = {}
    for name, feat_idx in configs.items():
        Xf = X[:, feat_idx]

        # Global ridge
        best_alpha, best_gc = 1.0, 0.0
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            scores = np.zeros(N)
            for train_i, val_i in folds:
                reg = Ridge(alpha=alpha)
                y_tr = times[train_i] * (1 - events[train_i] * 0.5)
                reg.fit(Xf[train_i], y_tr)
                scores[val_i] = reg.predict(Xf[val_i])
            fc = []
            for _, val_i in folds:
                ci = concordance_index(
                    torch.tensor(-scores[val_i].astype(np.float32)),
                    torch.tensor(times[val_i].astype(np.float32)),
                    torch.tensor(events[val_i].astype(np.float32)),
                )
                fc.append(ci)
            gc = np.mean(fc)
            if gc > best_gc:
                best_gc = gc
                best_alpha = alpha

        # Per-CT ridge
        scores_perct = np.zeros(N)
        for train_i, val_i in folds:
            ct_train = defaultdict(list)
            for idx in train_i:
                ct_train[ct_per[idx]].append(idx)

            reg_glob = Ridge(alpha=best_alpha)
            y_tr = times[train_i] * (1 - events[train_i] * 0.5)
            reg_glob.fit(Xf[train_i], y_tr)

            ct_models = {}
            for ct_name, indices in ct_train.items():
                if len(indices) >= 100:
                    arr = np.array(indices)
                    reg_ct = Ridge(alpha=best_alpha)
                    reg_ct.fit(Xf[arr], times[arr].astype(float) * (1 - events[arr] * 0.5))
                    ct_models[ct_name] = reg_ct

            for idx in val_i:
                ct_name = ct_per[idx]
                if ct_name in ct_models:
                    scores_perct[idx] = ct_models[ct_name].predict(Xf[idx:idx+1])[0]
                else:
                    scores_perct[idx] = reg_glob.predict(Xf[idx:idx+1])[0]

        fc_perct = []
        for _, val_i in folds:
            ci = concordance_index(
                torch.tensor(-scores_perct[val_i].astype(np.float32)),
                torch.tensor(times[val_i].astype(np.float32)),
                torch.tensor(events[val_i].astype(np.float32)),
            )
            fc_perct.append(ci)
        perct_c = np.mean(fc_perct)

        results[name] = {'global': float(best_gc), 'perct': float(perct_c),
                         'alpha': best_alpha}
        print(f"  {name:<40} {best_gc:>8.4f} {perct_c:>8.4f}")

    # Feature importance for full model
    print(f"\n{'='*80}")
    print(f"  TOP FEATURES (full model)")
    print(f"{'='*80}")
    reg = Ridge(alpha=results.get("FULL (all 95)", {}).get('alpha', 100.0))
    reg.fit(X, times * (1 - events * 0.5))

    feat_names = (
        ["ct_baseline", "weighted_ch_shift", "ct_gene_shift", "global_gene_shift"]
        + [f"ch_dmg_{ch}" for ch in CHANNEL_NAMES]
        + ["entropy", "hub_damage", "tier_conn", "n_mutated", "hhi"]
        + ["mean_ppi_dist", "min_ppi_dist", "frac_connected", "frac_dist1",
           "frac_dist2_3", "frac_dist4plus", "mean_overlap", "min_overlap",
           "max_overlap", "frac_same_ch", "frac_cross_ch", "mean_cooccur",
           "max_cooccur", "frac_cooccur", "combined_ent", "hub_hub", "hub_nonhub",
           "close_cross", "far_same", "tier_dist", "cross_tier", "n_pairs"]
        + ["n_comps", "largest", "frac_largest", "isolated", "frac_isolated",
           "comp_entropy", "max_comp_dmg", "comp_spread"]
        + ["ct_cooccur_n", "ct_cooccur_wt", "ct_cooccur_max", "cross_ch_cooccur"]
        + [f"ch_max_hr_{ch}" for ch in CHANNEL_NAMES]
        + ["branch_ratio", "fragment", "linearity_placeholder", "connectivity",
           "depth_placeholder", "frontier", "ppi_coverage"]
        + ["age", "sex", "stage", "msi_score", "msi_high", "fga", "aneuploidy",
           "tmb", "mutation_count", "neoadjuvant", "radiation", "hypoxia_buffa",
           "hypoxia_ragnum"]
        + ["frac_gof", "frac_lof", "gof_lof_ratio", "frac_actionable"]
        + [f"expr_{ch}" for ch in CHANNEL_NAMES]
        + ["amp_burden", "del_burden", "amp_channels", "del_channels"]
    )

    coefs = reg.coef_
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    print(f"\n  {'Feature':<25} {'Coeff':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for j in sorted_idx[:25]:
        name = feat_names[j] if j < len(feat_names) else f"[{j}]"
        print(f"  {name:<25} {coefs[j]:>+10.4f}")

    # Save
    with open(os.path.join(SAVE_BASE, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Total time: {time.time()-t_start:.0f}s")
    print(f"  Saved to {SAVE_BASE}/results.json")


if __name__ == "__main__":
    main()
