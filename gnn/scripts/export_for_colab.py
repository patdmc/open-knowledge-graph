"""
Export all training data to .pt files for GPU training without Neo4j.

Produces two files:
  - colab_patient_data.pt   (~2GB) — patient survival training data
  - colab_depmap_data.pt    (~50MB) — DepMap pre-training data

Run locally:
    python3 -u -m gnn.scripts.export_for_colab
"""

import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, NODE_FEAT_DIM, MAX_NODES
from gnn.data.cell_line_dataset import CellLineDataset
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, GNN_CACHE, GNN_RESULTS, ANALYSIS_CACHE
from gnn.data.block_assignments import load_block_assignments
from gnn.data.temporal import TemporalEstimator, year_features

EXPORT_DIR = os.path.join(GNN_RESULTS, "colab_export")


def build_gene_vocab(gene_names_list):
    gene_vocab = {}
    for patient_genes in gene_names_list:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    return gene_vocab


def build_gene_indices(gene_names_list, gene_vocab):
    N = len(gene_names_list)
    gene_indices = np.zeros((N, MAX_NODES), dtype=np.int64)
    for b, genes in enumerate(gene_names_list):
        for s, g in enumerate(genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    return torch.tensor(gene_indices, dtype=torch.long)


def build_block_channel_ids(gene_names_list, gene_block, n_blocks, n_channels):
    channel_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
    N = len(gene_names_list)
    block_ids = np.full((N, MAX_NODES), n_blocks, dtype=np.int64)
    channel_ids = np.full((N, MAX_NODES), n_channels, dtype=np.int64)
    for b, genes in enumerate(gene_names_list):
        for s, g in enumerate(genes):
            if g and g != '' and g != 'WT':
                info = gene_block.get(g)
                if info:
                    block_ids[b, s] = info['block_id']
                    channel_ids[b, s] = info['channel_id']
                else:
                    ch = CHANNEL_MAP.get(g)
                    if ch and ch in channel_idx:
                        channel_ids[b, s] = channel_idx[ch]
    return (torch.tensor(block_ids, dtype=torch.long),
            torch.tensor(channel_ids, dtype=torch.long))


def load_graph_data():
    from neo4j import GraphDatabase
    import networkx as nx
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    print("  Loading graph data from Neo4j...", flush=True)

    # PPI distances via shortest path
    ppi_cache = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    if os.path.exists(ppi_cache):
        with open(ppi_cache) as f:
            ppi_raw = json.load(f)
        G_ppi = nx.Graph()
        for e in ppi_raw:
            if isinstance(e, list):
                G_ppi.add_edge(e[0], e[1], weight=e[2] if len(e) > 2 else 0.4)
            else:
                G_ppi.add_edge(e['gene1'], e['gene2'], weight=e.get('score', 0.4))
    else:
        G_ppi = nx.Graph()
        with driver.session() as s:
            for r in s.run("MATCH (g1:Gene)-[r:PPI]-(g2:Gene) WHERE g1.name < g2.name "
                           "RETURN g1.name AS g1, g2.name AS g2, r.score AS score"):
                G_ppi.add_edge(r["g1"], r["g2"], weight=r["score"] or 0.4)

    ppi_dists = {}
    for comp in nx.connected_components(G_ppi):
        sub = G_ppi.subgraph(comp)
        for src, targets in dict(nx.all_pairs_shortest_path_length(sub)).items():
            for tgt, dist in targets.items():
                if src <= tgt:
                    ppi_dists[(src, tgt)] = dist

    channel_profiles = {}
    with driver.session() as s:
        for r in s.run("MATCH (g:Gene) WHERE g.channel_profile IS NOT NULL "
                       "RETURN g.name AS gene, g.channel_profile AS profile"):
            channel_profiles[r["gene"]] = np.array(r["profile"], dtype=np.float32)

    # All edge types: binary existence per gene pair
    # 10 mechanistic types + UNKNOWN (unexplained co-occurrence residual)
    #
    # Classification of Neo4j relationships:
    #   PPI           ← PPI (STRING physical binding)
    #   TRANSPOSES    ← TRANSPOSES (paralogs)
    #   BELONGS_TO    ← derived from CHANNEL_MAP (same channel)
    #   ANALOGOUS     ← ANALOGOUS (same edit mechanism)
    #   COUPLES       ← COUPLES + CO_EXPRESSED + CO_CNA (functional coupling, any evidence)
    #   SL_PARTNER    ← SL_PARTNER + SYNTHETIC_LETHAL (same concept, two rel names)
    #   CONVERGES     ← CONVERGES (mutually exclusive)
    #   ENABLES       ← ENABLES + ATTENDS_TO (directional, asymmetric)
    #   UNIQUE        ← UNIQUE (brand new mutation type)
    #   CO_ESSENTIAL  ← CO_ESSENTIAL (CRISPR co-dependency, distinct from expression coupling)
    #
    # Not gene-gene pairs (excluded):
    #   HAS_SENSITIVITY_EVIDENCE, HAS_RESISTANCE_EVIDENCE — self-loops (gene→gene drug evidence)

    # Primary neo4j rels per mechanistic type
    EDGE_TYPE_RELS = [
        # (canonical_name, [neo4j_rels], sign, is_gene_gene)
        ("PPI",          ["PPI"],                              +1, True),
        ("TRANSPOSES",   ["TRANSPOSES"],                       +1, True),
        ("BELONGS_TO",   [],                                   +1, False),  # derived from CHANNEL_MAP
        ("ANALOGOUS",    ["ANALOGOUS"],                        +1, True),
        ("COUPLES",      ["COUPLES", "CO_EXPRESSED", "CO_CNA"],+1, True),
        ("SL_PARTNER",   ["SL_PARTNER", "SYNTHETIC_LETHAL"],   -1, True),
        ("CONVERGES",    ["CONVERGES"],                        -1, True),
        ("ENABLES",      ["ENABLES", "ATTENDS_TO"],            +1, True),
        ("UNIQUE",       ["UNIQUE"],                           +1, True),
        ("CO_ESSENTIAL", ["CO_ESSENTIAL"],                     +1, True),
    ]

    edge_type_pairs = {}
    for name, rels, sign, is_gene_gene in EDGE_TYPE_RELS:
        pairs = set()
        if is_gene_gene:
            for rel in rels:
                with driver.session() as s:
                    result = s.run(f"""
                        MATCH (g1:Gene)-[:{rel}]-(g2:Gene)
                        WHERE g1.name < g2.name
                        RETURN g1.name AS g1, g2.name AS g2
                    """)
                    for r in result:
                        pairs.add((r["g1"], r["g2"]))
        elif name == "BELONGS_TO":
            # Pairs = genes in the same channel
            from collections import defaultdict
            ch_genes = defaultdict(list)
            for gene, ch in CHANNEL_MAP.items():
                ch_genes[ch].append(gene)
            for ch, genes in ch_genes.items():
                for a in range(len(genes)):
                    for b in range(a + 1, len(genes)):
                        pair = (min(genes[a], genes[b]), max(genes[a], genes[b]))
                        pairs.add(pair)
        edge_type_pairs[name] = pairs
        rels_str = f" ← {'+'.join(rels)}" if rels else " ← CHANNEL_MAP"
        print(f"    {name:>15s}: {len(pairs):>6,} pairs{rels_str}", flush=True)

    # Co-occurrence for UNKNOWN residual detection
    cooccurrence = {}
    with driver.session() as s:
        for r in s.run("MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene) WHERE g1.name < g2.name "
                       "RETURN g1.name AS g1, g2.name AS g2, r.count AS cnt"):
            cooccurrence[(r["g1"], r["g2"])] = r["cnt"] or 0
    max_cooccur = max(cooccurrence.values()) if cooccurrence else 1

    # UNKNOWN = co-occurring but not explained by any of the 10 types
    all_cooccur = set(cooccurrence.keys())
    explained = set()
    for pairs in edge_type_pairs.values():
        explained |= pairs
    unknown_pairs = all_cooccur - explained
    edge_type_pairs["UNKNOWN"] = unknown_pairs
    print(f"    {'UNKNOWN':>15s}: {len(unknown_pairs):>6,} unexplained co-occurrence pairs", flush=True)

    pct_explained = 100 * len(explained & all_cooccur) / max(len(all_cooccur), 1)
    print(f"    Decomposition: {pct_explained:.1f}% of co-occurrences explained by 10 edge types", flush=True)

    # Per-edge signed correlations — the sign is on the instance, not the type
    pair_correlations = {}  # (g1, g2) → float in [-1, +1]
    for rel, prop in [("CO_ESSENTIAL", "correlation"), ("CO_CNA", "correlation"),
                      ("CO_EXPRESSED", "correlation")]:
        with driver.session() as s:
            for r in s.run(f"""
                MATCH (g1:Gene)-[r:{rel}]-(g2:Gene)
                WHERE g1.name < g2.name AND r.{prop} IS NOT NULL
                RETURN g1.name AS g1, g2.name AS g2, r.{prop} AS corr
            """):
                pair = (r["g1"], r["g2"])
                corr = float(r["corr"])
                # Average if multiple sources give different correlations
                if pair in pair_correlations:
                    pair_correlations[pair] = (pair_correlations[pair] + corr) / 2
                else:
                    pair_correlations[pair] = corr
    n_pos = sum(1 for v in pair_correlations.values() if v > 0)
    n_neg = sum(1 for v in pair_correlations.values() if v < 0)
    print(f"    Per-edge correlations: {len(pair_correlations):,} pairs  +:{n_pos:,}  -:{n_neg:,}", flush=True)

    # COUPLES pair_type sign: GOF×GOF=+1, LOF×LOF=+1, GOF×LOF=-1
    pair_type_signs = {}
    with driver.session() as s:
        for r in s.run("""
            MATCH (g1:Gene)-[r:COUPLES]-(g2:Gene)
            WHERE g1.name < g2.name AND r.pair_type IS NOT NULL
            RETURN g1.name AS g1, g2.name AS g2, r.pair_type AS pt
        """):
            pair = (r["g1"], r["g2"])
            pt = r["pt"]
            # Same direction = reinforcing (+1), opposite = opposing (-1)
            if pt in ("gof_gof", "lof_lof"):
                pair_type_signs[pair] = 1.0
            elif pt in ("gof_lof", "lof_gof"):
                pair_type_signs[pair] = -1.0
            # lof_lol and others → 0 (ambiguous)
    print(f"    COUPLES pair_type signs: {len(pair_type_signs):,} pairs", flush=True)

    driver.close()
    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set |= hubs

    return {
        'ppi_dists': ppi_dists, 'channel_profiles': channel_profiles,
        'cooccurrence': cooccurrence, 'max_cooccur': max_cooccur,
        'edge_type_pairs': edge_type_pairs, 'hub_set': hub_set,
        'pair_correlations': pair_correlations,
        'pair_type_signs': pair_type_signs,
    }


def compute_gene_pair_matrix(graph_data, gene_vocab):
    """Build (G, G, D) gene-pair feature matrix.

    Layout (26 dims):
      [0]  PPI distance (normalized)
      [1]  PPI direct neighbor
      [2]  PPI within 3 hops
      [3]  Same channel
      [4]  Cross channel
      [5]  Co-occurrence (log-normalized)
      [6]  Channel profile cosine similarity
      [7]  Both hub genes
      [8-18]  Edge type exists: PPI, TRANSPOSES, BELONGS_TO, ANALOGOUS,
              COUPLES, SL_PARTNER, CONVERGES, ENABLES, UNIQUE,
              CO_ESSENTIAL, UNKNOWN
      [19] Signed correlation (from CO_ESSENTIAL, CO_CNA, CO_EXPRESSED)
           — per-edge sign, not per-type. +1 = co-dependent, -1 = compensatory
      [20] Has any positive-sign edge type
      [21] Has any negative-sign edge type
      [22] Has any unknown-sign edge type
      [23] n_edge_types (how many types explain this pair)
      [24] COUPLES pair_type signal (GOF×GOF=+1, LOF×LOF=+1, GOF×LOF=-1)
      [25] Self-loop
    """
    EDGE_TYPE_ORDER = [
        "PPI", "TRANSPOSES", "BELONGS_TO", "ANALOGOUS", "COUPLES",
        "SL_PARTNER", "CONVERGES", "ENABLES", "UNIQUE",
        "CO_ESSENTIAL", "UNKNOWN",
    ]

    G = len(gene_vocab)
    GRAPH_EDGE_DIM = 26
    matrix = np.zeros((G, G, GRAPH_EDGE_DIM), dtype=np.float32)
    ppi_dists = graph_data['ppi_dists']
    profiles = graph_data['channel_profiles']
    cooccurrence = graph_data['cooccurrence']
    max_cooccur = graph_data['max_cooccur']
    edge_type_pairs = graph_data['edge_type_pairs']
    hub_set = graph_data['hub_set']
    pair_correlations = graph_data['pair_correlations']
    pair_type_signs = graph_data['pair_type_signs']
    DISCONNECTED = 10
    idx_to_gene = {i: g for g, i in gene_vocab.items()}

    # Pre-index edge types by gene pair for fast lookup
    pair_edges = {}  # (gi, gj) → set of edge type names
    for etype, pairs in edge_type_pairs.items():
        for pair in pairs:
            pair_edges.setdefault(pair, set()).add(etype)

    # Per-type default signs (used when no per-edge sign is available)
    # These are structural defaults — per-edge data overrides when present
    TYPE_DEFAULT_SIGN = {
        "PPI": +1, "TRANSPOSES": 0, "BELONGS_TO": +1, "ANALOGOUS": 0,
        "COUPLES": 0, "SL_PARTNER": -1, "CONVERGES": -1, "ENABLES": 0,
        "UNIQUE": 0, "CO_ESSENTIAL": 0, "UNKNOWN": 0,
    }

    for i in range(G):
        gi = idx_to_gene[i]
        matrix[i, i, 25] = 1.0  # self-loop
        for j in range(i + 1, G):
            gj = idx_to_gene[j]
            pair = (min(gi, gj), max(gi, gj))

            # [0-2] PPI distance features
            d = ppi_dists.get(pair, DISCONNECTED)
            matrix[i, j, 0] = matrix[j, i, 0] = d / DISCONNECTED
            matrix[i, j, 1] = matrix[j, i, 1] = 1.0 if d == 1 else 0.0
            matrix[i, j, 2] = matrix[j, i, 2] = 1.0 if d <= 3 else 0.0

            # [3-4] Channel relationship
            ch_i, ch_j = CHANNEL_MAP.get(gi), CHANNEL_MAP.get(gj)
            if ch_i and ch_j:
                matrix[i, j, 3] = matrix[j, i, 3] = 1.0 if ch_i == ch_j else 0.0
                matrix[i, j, 4] = matrix[j, i, 4] = 1.0 if ch_i != ch_j else 0.0

            # [5] Co-occurrence
            cooccur = cooccurrence.get(pair, 0)
            if cooccur > 0:
                matrix[i, j, 5] = matrix[j, i, 5] = np.log1p(cooccur) / np.log1p(max_cooccur)

            # [6] Channel profile cosine similarity
            pi, pj = profiles.get(gi), profiles.get(gj)
            if pi is not None and pj is not None:
                ni, nj = np.linalg.norm(pi), np.linalg.norm(pj)
                if ni > 0 and nj > 0:
                    matrix[i, j, 6] = matrix[j, i, 6] = np.dot(pi, pj) / (ni * nj)

            # [7] Both hub genes
            matrix[i, j, 7] = matrix[j, i, 7] = 1.0 if (gi in hub_set and gj in hub_set) else 0.0

            # [8-18] Edge type binary flags (11 types)
            pair_etypes = pair_edges.get(pair, set())
            for k, etype in enumerate(EDGE_TYPE_ORDER):
                if etype in pair_etypes:
                    matrix[i, j, 8 + k] = matrix[j, i, 8 + k] = 1.0

            # [19] Per-edge signed correlation (from actual data, not type assumption)
            corr = pair_correlations.get(pair)
            if corr is not None:
                matrix[i, j, 19] = matrix[j, i, 19] = np.clip(corr, -1.0, 1.0)

            # [20-22] Aggregate sign signals from all evidence
            if pair_etypes:
                # Collect per-edge sign from data, fall back to type default
                edge_sign = corr if corr is not None else 0.0
                pt_sign = pair_type_signs.get(pair, 0.0)

                # Combine: any evidence of positive, negative, or unknown
                has_pos = (edge_sign > 0.1) or (pt_sign > 0)
                has_neg = (edge_sign < -0.1) or (pt_sign < 0)
                # Also check structural defaults for types without per-edge data
                for et in pair_etypes:
                    ds = TYPE_DEFAULT_SIGN.get(et, 0)
                    if ds > 0 and corr is None:
                        has_pos = True
                    elif ds < 0 and corr is None:
                        has_neg = True

                has_unknown = (not has_pos and not has_neg) or any(
                    TYPE_DEFAULT_SIGN.get(et, 0) == 0 for et in pair_etypes
                    if et != "UNKNOWN")

                matrix[i, j, 20] = matrix[j, i, 20] = float(has_pos)
                matrix[i, j, 21] = matrix[j, i, 21] = float(has_neg)
                matrix[i, j, 22] = matrix[j, i, 22] = float(has_unknown)

            # [23] Number of edge types explaining this pair
            n_types = len(pair_etypes - {"UNKNOWN"})
            if n_types > 0:
                matrix[i, j, 23] = matrix[j, i, 23] = min(n_types / 5.0, 1.0)

            # [24] COUPLES pair_type sign (GOF×GOF=+1, LOF×LOF=+1, GOF×LOF=-1)
            pt_sign = pair_type_signs.get(pair)
            if pt_sign is not None:
                matrix[i, j, 24] = matrix[j, i, 24] = pt_sign

    return matrix, GRAPH_EDGE_DIM


def compute_patient_edge_features(node_features_np):
    PATIENT_EDGE_DIM = 4
    nf = node_features_np if isinstance(node_features_np, np.ndarray) else node_features_np.numpy()
    B, N, _ = nf.shape
    feats = np.zeros((B, N, N, PATIENT_EDGE_DIM), dtype=np.float32)
    log_hr = np.abs(nf[:, :, 0])
    harmful = nf[:, :, 10] > 0.5
    protective = nf[:, :, 11] > 0.5
    feats[:, :, :, 0] = log_hr[:, :, None] * log_hr[:, None, :]
    feats[:, :, :, 1] = (harmful[:, :, None] & harmful[:, None, :]).astype(np.float32)
    feats[:, :, :, 2] = (protective[:, :, None] & protective[:, None, :]).astype(np.float32)
    feats[:, :, :, 3] = ((harmful[:, :, None] & protective[:, None, :]) |
                          (protective[:, :, None] & harmful[:, None, :])).astype(np.float32)
    return feats, PATIENT_EDGE_DIM


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    t_start = time.time()

    # === Block/channel assignments (shared) ===
    print("Loading block assignments...", flush=True)
    gene_block, n_blocks, n_channels = load_block_assignments()

    # === Graph data (shared) ===
    graph_data = load_graph_data()

    # =================================================================
    # PART 1: Patient survival data
    # =================================================================
    print("\n=== PATIENT DATA ===", flush=True)
    ds = AtlasDataset()
    data = ds.build_features()

    patient_gene_vocab = build_gene_vocab(data['gene_names'])
    G_patient = len(patient_gene_vocab)
    print(f"  Gene vocabulary: {G_patient}")

    gene_indices_t = build_gene_indices(data['gene_names'], patient_gene_vocab)
    block_ids_t, channel_ids_t = build_block_channel_ids(
        data['gene_names'], gene_block, n_blocks, n_channels)

    print("  Computing gene-pair matrix...", flush=True)
    gene_pair_matrix, graph_edge_dim = compute_gene_pair_matrix(graph_data, patient_gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32)

    print("  Computing patient edge features...", flush=True)
    patient_edge_feats, patient_edge_dim = compute_patient_edge_features(data['node_features'])
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32)

    # Temporal
    patient_ids = ds.clinical['patientId'].tolist()
    te = TemporalEstimator(ds.clinical)
    years, confs = te.estimate_all(patient_ids)
    temporal_feats = year_features(years, confs)
    temporal_t = torch.tensor(temporal_feats, dtype=torch.float32)

    clinical = torch.cat([
        data['ages'].unsqueeze(-1),
        data['sexes'].unsqueeze(-1),
        temporal_t,
    ], dim=-1)

    patient_export = {
        'node_features': data['node_features'],
        'node_masks': data['node_masks'],
        'cancer_types': data['cancer_types'],
        'clinical': clinical,
        'atlas_sums': data['atlas_sums'],
        'times': data['times'],
        'events': data['events'],
        'n_cancer_types': data['n_cancer_types'],
        'gene_pair_matrix': gene_pair_matrix_t,
        'patient_edge_feats': patient_edge_feats_t,
        'gene_indices': gene_indices_t,
        'block_ids': block_ids_t,
        'channel_ids': channel_ids_t,
        'n_blocks': n_blocks,
        'n_channels': n_channels,
        'gene_vocab': patient_gene_vocab,
        'cancer_type_map': data['cancer_type_map'],
        'graph_edge_dim': graph_edge_dim,
        'patient_edge_dim': patient_edge_dim,
        'gene_names': data['gene_names'],
    }

    patient_path = os.path.join(EXPORT_DIR, "colab_patient_data.pt")
    torch.save(patient_export, patient_path)
    size_mb = os.path.getsize(patient_path) / 1e6
    print(f"  Saved: {patient_path} ({size_mb:.0f}MB)")

    # =================================================================
    # PART 2: DepMap pre-training data
    # =================================================================
    print("\n=== DEPMAP DATA ===", flush=True)
    cl_ds = CellLineDataset()
    cl_data = cl_ds.build_features()

    depmap_gene_vocab = build_gene_vocab(cl_data['gene_names'])
    G_depmap = len(depmap_gene_vocab)
    print(f"  Gene vocabulary: {G_depmap}")

    cl_gene_indices_t = build_gene_indices(cl_data['gene_names'], depmap_gene_vocab)
    cl_block_ids_t, cl_channel_ids_t = build_block_channel_ids(
        cl_data['gene_names'], gene_block, n_blocks, n_channels)

    print("  Computing gene-pair matrix...", flush=True)
    cl_gene_pair_matrix, cl_graph_edge_dim = compute_gene_pair_matrix(graph_data, depmap_gene_vocab)
    cl_gene_pair_matrix_t = torch.tensor(cl_gene_pair_matrix, dtype=torch.float32)

    print("  Computing cell line edge features...", flush=True)
    cl_patient_edge_feats, cl_patient_edge_dim = compute_patient_edge_features(
        cl_data['node_features'])
    cl_patient_edge_feats_t = torch.tensor(cl_patient_edge_feats, dtype=torch.float32)

    N_cl = len(cl_data['cancer_types'])
    cl_clinical = torch.zeros(N_cl, 5, dtype=torch.float32)
    cl_clinical[:, 0] = cl_data['ages']
    cl_clinical[:, 1] = cl_data['sexes']

    depmap_export = {
        'node_features': cl_data['node_features'],
        'node_masks': cl_data['node_masks'],
        'cancer_types': cl_data['cancer_types'],
        'clinical': cl_clinical,
        'essentiality': cl_data['essentiality'],
        'essentiality_masks': cl_data['essentiality_masks'],
        'n_cancer_types': cl_data['n_cancer_types'],
        'gene_pair_matrix': cl_gene_pair_matrix_t,
        'patient_edge_feats': cl_patient_edge_feats_t,
        'gene_indices': cl_gene_indices_t,
        'block_ids': cl_block_ids_t,
        'channel_ids': cl_channel_ids_t,
        'n_blocks': n_blocks,
        'n_channels': n_channels,
        'gene_vocab': depmap_gene_vocab,
        'graph_edge_dim': cl_graph_edge_dim,
        'patient_edge_dim': cl_patient_edge_dim,
        'gene_names': cl_data['gene_names'],
    }

    depmap_path = os.path.join(EXPORT_DIR, "colab_depmap_data.pt")
    torch.save(depmap_export, depmap_path)
    size_mb = os.path.getsize(depmap_path) / 1e6
    print(f"  Saved: {depmap_path} ({size_mb:.0f}MB)")

    # =================================================================
    # PART 3: Model code (self-contained .py files)
    # =================================================================
    # Copy the model file so Colab can import it without the full gnn package
    import shutil
    model_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "atlas_transformer_v6.py")
    model_dst = os.path.join(EXPORT_DIR, "atlas_transformer_v6.py")
    shutil.copy2(model_src, model_dst)

    head_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models", "essentiality_head.py")
    head_dst = os.path.join(EXPORT_DIR, "essentiality_head.py")
    shutil.copy2(head_src, head_dst)

    cox_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "cox_sage.py")
    cox_dst = os.path.join(EXPORT_DIR, "cox_sage.py")
    shutil.copy2(cox_src, cox_dst)

    print(f"\n  Copied model files to {EXPORT_DIR}")

    # =================================================================
    # PART 4: Graph metadata (everything Colab needs for analysis)
    # =================================================================
    from gnn.config import GENE_FUNCTION

    # Channel-organ affinity from results
    affinity_path = os.path.join(GNN_RESULTS, "channel_organ_affinity.json")
    organ_affinity = {}
    if os.path.exists(affinity_path):
        with open(affinity_path) as f:
            organ_affinity = json.load(f)

    # Reverse maps for analysis
    channel_to_genes = {}
    for ch in CHANNEL_NAMES:
        channel_to_genes[ch] = [g for g, c in CHANNEL_MAP.items() if c == ch]

    metadata = {
        'channel_map': CHANNEL_MAP,           # gene → channel
        'channel_names': CHANNEL_NAMES,       # ordered channel list
        'channel_to_genes': channel_to_genes, # channel → [genes]
        'hub_genes': {k: list(v) for k, v in HUB_GENES.items()},
        'gene_function': GENE_FUNCTION,       # gene → oncogene/TSG
        'cancer_type_map': data['cancer_type_map'],  # name → index
        'organ_affinity': organ_affinity,     # channel-organ edges + organ nodes
        'node_feat_layout': {
            0: 'log_hr', 1: 'ci_width', 2: 'tier',
            3: 'is_hub (zeroed)', 4: 'channel_onehot[0] (zeroed)',
            10: 'is_harmful', 11: 'is_protective',
            12: 'log_n_patients', 13: 'protein_position',
            14: 'biallelic_status', 15: 'expression_z',
            16: 'gof_lof', 17: 'is_truncating',
        },
        'gene_pair_layout': {
            0: 'ppi_distance', 1: 'ppi_neighbor', 2: 'ppi_proximity',
            3: 'same_channel', 4: 'cross_channel', 5: 'co_occurrence',
            6: 'channel_similarity', 7: 'both_hubs',
            8: 'edge_PPI', 9: 'edge_TRANSPOSES', 10: 'edge_BELONGS_TO',
            11: 'edge_ANALOGOUS', 12: 'edge_COUPLES', 13: 'edge_SL_PARTNER',
            14: 'edge_CONVERGES', 15: 'edge_ENABLES', 16: 'edge_UNIQUE',
            17: 'edge_CO_ESSENTIAL', 18: 'edge_UNKNOWN',
            19: 'signed_correlation', 20: 'has_positive_sign',
            21: 'has_negative_sign', 22: 'has_unknown_sign',
            23: 'n_edge_types', 24: 'couples_pair_type_sign', 25: 'self_loop',
        },
        'edge_type_classification': {
            'PPI': ['PPI'],
            'TRANSPOSES': ['TRANSPOSES'],
            'BELONGS_TO': ['CHANNEL_MAP derived'],
            'ANALOGOUS': ['ANALOGOUS'],
            'COUPLES': ['COUPLES', 'CO_EXPRESSED', 'CO_CNA'],
            'SL_PARTNER': ['SL_PARTNER', 'SYNTHETIC_LETHAL'],
            'CONVERGES': ['CONVERGES'],
            'ENABLES': ['ENABLES', 'ATTENDS_TO'],
            'UNIQUE': ['UNIQUE'],
            'CO_ESSENTIAL': ['CO_ESSENTIAL'],
            'UNKNOWN': ['residual — co-occurring but unexplained'],
        },
    }

    meta_path = os.path.join(EXPORT_DIR, "graph_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path}")

    print(f"\nDone in {time.time() - t_start:.1f}s")
    print(f"\nUpload to Colab:")
    print(f"  1. {depmap_path}")
    print(f"  2. {patient_path}")
    print(f"  3. {meta_path}")
    print(f"  4. {EXPORT_DIR}/*.py")
    print(f"  5. The Colab notebook: gnn/scripts/colab_train.ipynb")


if __name__ == "__main__":
    main()
