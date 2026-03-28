"""
Train AtlasTransformerV6 — mutation-level with graph-structured attention.

The key difference from V3: pairwise graph structure flows directly into
the attention matrix, not as a patient-level summary vector.

For each patient's mutation set, we compute an (N, N, edge_feat_dim) tensor
where entry (i, j) encodes the graph relationship between mutation i and
mutation j. This includes:
  - PPI distance (normalized)
  - PPI direct neighbor flag
  - Same channel flag
  - Cross-channel flag
  - Co-occurrence frequency (from population)
  - Profile cosine similarity
  - Synthetic lethal flag
  - Co-expression flag
  - Combined damage (product of |log_hr|)

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v6 [--epochs 200]
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, NODE_FEAT_DIM, MAX_NODES
from gnn.models.atlas_transformer_v6 import AtlasTransformerV6
from gnn.models.cox_sage import cox_ph_loss
from gnn.models.hierarchy_loss import HierarchyLoss, build_hierarchy_pairs, MultiObjectiveLoss
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, GNN_CACHE, ANALYSIS_CACHE
from gnn.data.block_assignments import load_block_assignments
from gnn.data.temporal import TemporalEstimator, year_features
from gnn.models.graph_predictions import (
    CoMutationPredictor, CancerTypePredictor,
    SyntheticLethalityValidator, load_patient_data,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "atlas_transformer_v6",
)

# =========================================================================
# Pairwise edge feature computation
# =========================================================================

# Edge feature layout per mutation pair:
# [0] ppi_distance_norm     — PPI shortest path / 10 (1.0 = disconnected)
# [1] ppi_direct_neighbor   — 1.0 if PPI distance == 1
# [2] ppi_close             — 1.0 if PPI distance <= 3
# [3] same_channel          — 1.0 if both mutations in same channel
# [4] cross_channel         — 1.0 if different channels
# [5] cooccurrence_norm     — co-occurrence count / max, log-scaled
# [6] profile_cosine        — cosine similarity of channel profiles
# [7] combined_damage       — product of |log_hr| values (interaction magnitude)
# [8] both_harmful          — 1.0 if both hr > 1.1
# [9] both_protective       — 1.0 if both hr < 0.9
# [10] opposing_direction   — 1.0 if one harmful and one protective
# [11] hub_pair             — 1.0 if both are hub genes
# [12] sl_partner           — 1.0 if synthetic lethal partners
# Graph-level (computed once):
#   [0-9] gene pair features (PPI, channel, cooccur, profile, hub, SL, same_gene)
# Patient-level (per-patient):
#   [10-13] combined_damage, both_harmful, both_protective, opposing_direction
# Total: 10 + 4 = 14 (same dim, but factored computation)
EDGE_FEAT_DIM = 14  # kept for results JSON compatibility


def load_graph_data():
    """Load graph structure needed for pairwise features."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    print("  Loading graph data for pairwise features...", flush=True)
    t0 = time.time()

    # PPI distances
    ppi_cache = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    import networkx as nx
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
            result = s.run("""
                MATCH (g1:Gene)-[r:PPI]-(g2:Gene) WHERE g1.name < g2.name
                RETURN g1.name AS g1, g2.name AS g2, r.score AS score
            """)
            for r in result:
                G_ppi.add_edge(r["g1"], r["g2"], weight=r["score"] or 0.4)

    ppi_dists = {}
    for comp in nx.connected_components(G_ppi):
        sub = G_ppi.subgraph(comp)
        lengths = dict(nx.all_pairs_shortest_path_length(sub))
        for src, targets in lengths.items():
            for tgt, dist in targets.items():
                if src <= tgt:
                    ppi_dists[(src, tgt)] = dist

    print(f"    PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges, "
          f"{len(ppi_dists)} distances", flush=True)

    # Channel profiles
    channel_profiles = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene) WHERE g.channel_profile IS NOT NULL
            RETURN g.name AS gene, g.channel_profile AS profile
        """)
        for r in result:
            channel_profiles[r["gene"]] = np.array(r["profile"], dtype=np.float32)
    print(f"    Channel profiles: {len(channel_profiles)}", flush=True)

    # Co-occurrence
    cooccurrence = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene) WHERE g1.name < g2.name
            RETURN g1.name AS g1, g2.name AS g2, r.count AS cnt
        """)
        for r in result:
            cooccurrence[(r["g1"], r["g2"])] = r["cnt"] or 0
    max_cooccur = max(cooccurrence.values()) if cooccurrence else 1
    print(f"    Co-occurrence: {len(cooccurrence)} pairs", flush=True)

    # Synthetic lethal partners
    sl_pairs = set()
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[:SL_PARTNER]-(g2:Gene) WHERE g1.name < g2.name
            RETURN g1.name AS g1, g2.name AS g2
        """)
        for r in result:
            sl_pairs.add((r["g1"], r["g2"]))
    print(f"    SL pairs: {len(sl_pairs)}", flush=True)

    driver.close()

    # Hub set
    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set |= hubs

    print(f"    Graph data loaded [{time.time()-t0:.1f}s]", flush=True)

    return {
        'ppi_dists': ppi_dists,
        'channel_profiles': channel_profiles,
        'cooccurrence': cooccurrence,
        'max_cooccur': max_cooccur,
        'sl_pairs': sl_pairs,
        'hub_set': hub_set,
    }


def compute_gene_pair_matrix(graph_data, gene_vocab):
    """Compute (G, G, GRAPH_EDGE_DIM) matrix — properties of gene pairs.

    This is computed ONCE for all genes. Many patients share the same gene pairs.
    The expensive graph lookups happen here, not per-patient.

    Gene-pair features (graph-level, patient-independent):
    [0] ppi_distance_norm      [1] ppi_direct_neighbor  [2] ppi_close
    [3] same_channel           [4] cross_channel
    [5] cooccurrence_norm      [6] profile_cosine
    [7] hub_pair               [8] sl_partner           [9] same_gene
    """
    G = len(gene_vocab)
    GRAPH_EDGE_DIM = 10

    matrix = np.zeros((G, G, GRAPH_EDGE_DIM), dtype=np.float32)

    ppi_dists = graph_data['ppi_dists']
    profiles = graph_data['channel_profiles']
    cooccurrence = graph_data['cooccurrence']
    max_cooccur = graph_data['max_cooccur']
    sl_pairs = graph_data['sl_pairs']
    hub_set = graph_data['hub_set']
    DISCONNECTED = 10

    idx_to_gene = {i: g for g, i in gene_vocab.items()}

    for i in range(G):
        gi = idx_to_gene[i]
        # Self-loop
        matrix[i, i, 9] = 1.0

        for j in range(i + 1, G):
            gj = idx_to_gene[j]

            pair = (min(gi, gj), max(gi, gj))
            d = ppi_dists.get(pair, DISCONNECTED)
            matrix[i, j, 0] = matrix[j, i, 0] = d / DISCONNECTED
            matrix[i, j, 1] = matrix[j, i, 1] = 1.0 if d == 1 else 0.0
            matrix[i, j, 2] = matrix[j, i, 2] = 1.0 if d <= 3 else 0.0

            ch_i = CHANNEL_MAP.get(gi)
            ch_j = CHANNEL_MAP.get(gj)
            if ch_i and ch_j:
                matrix[i, j, 3] = matrix[j, i, 3] = 1.0 if ch_i == ch_j else 0.0
                matrix[i, j, 4] = matrix[j, i, 4] = 1.0 if ch_i != ch_j else 0.0

            cooccur = cooccurrence.get(pair, 0)
            if cooccur > 0:
                matrix[i, j, 5] = matrix[j, i, 5] = np.log1p(cooccur) / np.log1p(max_cooccur)

            pi = profiles.get(gi)
            pj = profiles.get(gj)
            if pi is not None and pj is not None:
                ni = np.linalg.norm(pi)
                nj = np.linalg.norm(pj)
                if ni > 0 and nj > 0:
                    matrix[i, j, 6] = matrix[j, i, 6] = np.dot(pi, pj) / (ni * nj)

            matrix[i, j, 7] = matrix[j, i, 7] = 1.0 if (gi in hub_set and gj in hub_set) else 0.0
            matrix[i, j, 8] = matrix[j, i, 8] = 1.0 if pair in sl_pairs else 0.0

    return matrix, GRAPH_EDGE_DIM


def compute_patient_edge_features(node_features):
    """Compute per-patient pairwise features from mutation-level node features.

    These depend on the specific patient's mutation properties (HR values,
    harmful/protective status) — they can't be precomputed at the gene level.

    Patient-pair features:
    [0] combined_damage       — product of |log_hr|
    [1] both_harmful          [2] both_protective     [3] opposing_direction
    """
    PATIENT_EDGE_DIM = 4
    nf = node_features if isinstance(node_features, np.ndarray) else node_features.numpy()
    B, N, _ = nf.shape

    feats = np.zeros((B, N, N, PATIENT_EDGE_DIM), dtype=np.float32)

    # Vectorized: compute for all pairs at once per patient
    # log_hr is feature [0], is_harmful is [10], is_protective is [11]
    log_hr = np.abs(nf[:, :, 0])          # (B, N)
    harmful = nf[:, :, 10] > 0.5           # (B, N) bool
    protective = nf[:, :, 11] > 0.5        # (B, N) bool

    # Combined damage: outer product of |log_hr|
    feats[:, :, :, 0] = log_hr[:, :, None] * log_hr[:, None, :]

    # Both harmful
    feats[:, :, :, 1] = (harmful[:, :, None] & harmful[:, None, :]).astype(np.float32)

    # Both protective
    feats[:, :, :, 2] = (protective[:, :, None] & protective[:, None, :]).astype(np.float32)

    # Opposing direction
    feats[:, :, :, 3] = ((harmful[:, :, None] & protective[:, None, :]) |
                          (protective[:, :, None] & harmful[:, None, :])).astype(np.float32)

    return feats, PATIENT_EDGE_DIM


def gather_edge_features(gene_pair_matrix, patient_edge_feats,
                         gene_indices, gene_masks):
    """Gather precomputed gene-pair features + patient features per batch.

    Instead of storing (B, N, N, 14) for all patients, we:
    1. Index into (G, G, 10) gene pair matrix using patient's gene indices
    2. Concatenate with (B, N, N, 4) patient-specific features

    Args:
        gene_pair_matrix: torch.Tensor (G, G, graph_edge_dim)
        patient_edge_feats: torch.Tensor (B, N, N, patient_edge_dim)
        gene_indices: torch.LongTensor (B, N) — index into gene vocab
        gene_masks: torch.Tensor (B, N) — 1 for real, 0 for pad

    Returns:
        edge_features: (B, N, N, graph_edge_dim + patient_edge_dim)
    """
    B, N = gene_indices.shape
    safe_idx = gene_indices.clamp(0, gene_pair_matrix.shape[0] - 1)

    # Gather: for each patient, for each pair (i,j), get gene_pair_matrix[gi, gj]
    # Expand indices for gathering from (G, G, D) matrix
    idx_i = safe_idx.unsqueeze(2).expand(B, N, N)  # (B, N, N)
    idx_j = safe_idx.unsqueeze(1).expand(B, N, N)  # (B, N, N)

    D_graph = gene_pair_matrix.shape[-1]
    # gene_pair_matrix[idx_i, idx_j] → (B, N, N, D_graph)
    graph_feats = gene_pair_matrix[idx_i, idx_j]  # advanced indexing

    # Mask: zero out pad positions
    pair_mask = (gene_masks.unsqueeze(1) * gene_masks.unsqueeze(2)).unsqueeze(-1)  # (B, N, N, 1)
    graph_feats = graph_feats * pair_mask

    # Concatenate graph-level and patient-level features
    edge_features = torch.cat([graph_feats, patient_edge_feats * pair_mask], dim=-1)
    return edge_features


# =========================================================================
# Hierarchy pairs (VAF-based functional depth supervision)
# =========================================================================

def load_patient_vafs(gene_names_per_patient, patient_ids):
    """Load VAF (variant allele frequency) per mutation slot per patient.

    VAF = tumorAltCount / (tumorAltCount + tumorRefCount).
    Higher VAF = earlier/more clonal mutation = shallower in hierarchy.

    Returns: np.array (N_patients, MAX_NODES) of VAF values.
    """
    import pandas as pd
    t0 = time.time()

    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange",
                 "tumorAltCount", "tumorRefCount"],
    )
    mut['vaf'] = mut['tumorAltCount'] / (
        mut['tumorAltCount'] + mut['tumorRefCount']).clip(lower=1)

    # Build lookup: (patientId, gene) → max VAF (take max across protein changes)
    # Using max because the same gene might appear multiple times with different
    # protein changes, and the atlas dataset picks the most impactful one
    vaf_lookup = {}
    for _, row in mut.iterrows():
        key = (row['patientId'], row['gene.hugoGeneSymbol'])
        v = row['vaf']
        if np.isfinite(v) and v > 0:
            vaf_lookup[key] = max(vaf_lookup.get(key, 0), v)

    # Map to patient × slot
    N = len(patient_ids)
    vafs = np.zeros((N, MAX_NODES), dtype=np.float32)

    n_filled = 0
    for p_idx, pid in enumerate(patient_ids):
        genes = gene_names_per_patient[p_idx]
        for s_idx, gene in enumerate(genes):
            if gene and gene != 'WT':
                v = vaf_lookup.get((pid, gene), 0.0)
                vafs[p_idx, s_idx] = v
                if v > 0:
                    n_filled += 1

    print(f"  VAFs loaded: {n_filled:,} filled slots "
          f"({n_filled / max(N * MAX_NODES, 1) * 100:.1f}%) [{time.time()-t0:.1f}s]")
    return vafs


# =========================================================================
# Training
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fresh', action='store_true',
                        help='Delete patient-derived caches before loading')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.fresh:
        import glob as glob_mod
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "cache")
        patient_caches = [
            "channel_v6c_features_msk_impact_50k.pt",
            "precomputed_pairwise_edges.json",
            "precomputed_gene_properties.json",
            "precomputed_gene_centrality.json",
            "precomputed_communities.json",
            "precomputed_cross_channel_matrix.json",
        ]
        for name in patient_caches:
            path = os.path.join(cache_dir, name)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Purged cache: {name}")
        for fold_pt in glob_mod.glob(os.path.join(RESULTS_DIR, "fold_*", "*.pt")):
            os.remove(fold_pt)
            print(f"  Purged model: {fold_pt}")
        print()

    # === Load data ===
    ds = AtlasDataset()
    data = ds.build_features()

    # === Build gene vocabulary ===
    gene_vocab = {}  # gene_name → index
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    G = len(gene_vocab)
    print(f"\n  Gene vocabulary: {G} unique genes")

    # Map per-patient gene names → vocabulary indices
    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)

    # === Load block/channel assignments ===
    print("  Loading block assignments...", flush=True)
    gene_block, n_blocks, n_channels = load_block_assignments()
    channel_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}

    # Build per-patient block_ids and channel_ids tensors
    # Default: block_id=n_blocks (unassigned block), channel_id=n_channels (unassigned)
    block_ids = np.full((len(data['gene_names']), MAX_NODES), n_blocks, dtype=np.int64)
    channel_ids = np.full((len(data['gene_names']), MAX_NODES), n_channels, dtype=np.int64)

    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT':
                info = gene_block.get(g)
                if info:
                    block_ids[b, s] = info['block_id']
                    channel_ids[b, s] = info['channel_id']
                else:
                    # Gene not in any block — try to assign channel from CHANNEL_MAP
                    ch = CHANNEL_MAP.get(g)
                    if ch and ch in channel_idx:
                        channel_ids[b, s] = channel_idx[ch]

    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    channel_ids_t = torch.tensor(channel_ids, dtype=torch.long)
    filled_mask = block_ids < n_blocks
    total_filled = filled_mask.sum()
    total_slots = (block_ids >= 0).sum()
    print(f"  Blocks: {n_blocks}, Channels: {n_channels}")
    print(f"  Mutation slots filled: {total_filled:,}/{total_slots:,} ({total_filled/total_slots:.1%})")
    # Of filled slots, what % got a real block vs unassigned?
    has_gene_mask = np.array([[bool(g and g != '' and g != 'WT') for g in patient]
                               for patient in data['gene_names']])
    n_with_gene = has_gene_mask.sum()
    n_with_block = filled_mask.sum()
    print(f"  Block coverage of actual mutations: {n_with_block}/{n_with_gene} ({n_with_block/max(n_with_gene,1):.1%})")

    # === Compute gene-pair matrix ONCE (graph-level features) ===
    print("  Computing gene-pair matrix (runs once)...", flush=True)
    t0 = time.time()
    graph_data = load_graph_data()
    gene_pair_matrix, graph_edge_dim = compute_gene_pair_matrix(graph_data, gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32)
    mem_mb = gene_pair_matrix.nbytes / 1e6
    print(f"  Gene-pair matrix: ({G}, {G}, {graph_edge_dim}) = {mem_mb:.1f}MB [{time.time()-t0:.1f}s]")

    # === Compute patient-level edge features (vectorized) ===
    print("  Computing patient-level edge features...", flush=True)
    t0 = time.time()
    patient_edge_feats, patient_edge_dim = compute_patient_edge_features(data['node_features'])
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32)
    mem_mb2 = patient_edge_feats.nbytes / 1e6
    print(f"  Patient edge features: {patient_edge_feats_t.shape} = {mem_mb2:.1f}MB [{time.time()-t0:.1f}s]")

    total_edge_dim = graph_edge_dim + patient_edge_dim
    print(f"  Total edge feature dim: {graph_edge_dim} (graph) + {patient_edge_dim} (patient) = {total_edge_dim}")

    # === Build hierarchy pairs (VAF-based depth supervision) ===
    print("\n  Building hierarchy pairs from VAF data...", flush=True)
    patient_ids = ds.clinical['patientId'].tolist()
    vafs = load_patient_vafs(data['gene_names'], patient_ids)

    pair_indices_np, pair_targets_np, pair_stats = build_hierarchy_pairs(
        data['gene_names'], vafs, CHANNEL_MAP,
        max_pairs_per_patient=64, min_vaf_diff=0.05,
    )
    print(f"  Hierarchy pairs: {pair_stats['total_pairs']:,} total, "
          f"{pair_stats['patients_with_pairs']:,} patients, "
          f"{pair_stats['mean_pairs']:.1f} mean/patient", flush=True)

    pair_indices_t = torch.tensor(pair_indices_np, dtype=torch.long)
    pair_targets_t = torch.tensor(pair_targets_np, dtype=torch.float32)

    # === Temporal features ===
    print("  Building temporal features...", flush=True)
    te = TemporalEstimator(ds.clinical)
    years, confs = te.estimate_all(patient_ids)
    temporal_feats = year_features(years, confs)  # (N, 3)
    temporal_t = torch.tensor(temporal_feats, dtype=torch.float32).to(device)
    print(f"  Temporal: year range [{years.min():.1f}, {years.max():.1f}], "
          f"mean conf={confs.mean():.2f}")

    # === Unpack data ===
    nf = data['node_features'].to(device)
    nm = data['node_masks'].to(device)
    ct = data['cancer_types'].to(device)
    ages = data['ages'].to(device)
    sexes = data['sexes'].to(device)
    atlas_sums = data['atlas_sums'].to(device)
    times = data['times']
    events = data['events']
    n_cancer_types = data['n_cancer_types']

    clinical = torch.cat([
        ages.unsqueeze(-1),
        sexes.unsqueeze(-1),
        temporal_t,
    ], dim=-1)  # (B, 5)
    gene_pair_matrix_t = gene_pair_matrix_t.to(device)
    patient_edge_feats_t = patient_edge_feats_t.to(device)
    gene_indices_t = gene_indices_t.to(device)

    block_ids_t = block_ids_t.to(device)
    channel_ids_t = channel_ids_t.to(device)

    # === Config ===
    config = {
        'node_feat_dim': NODE_FEAT_DIM,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_intra_layers': args.n_layers,
        'dropout': args.dropout,
        'n_cancer_types': n_cancer_types,
        'edge_feat_dim': total_edge_dim,
        'n_channels': n_channels,
        'n_blocks': n_blocks,
    }

    # === Holdback ===
    n_total = len(events)
    np.random.seed(args.seed)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    n_holdback = int(n_total * 0.15)
    holdback_idx = all_idx[:n_holdback]
    cv_idx = all_idx[n_holdback:]
    print(f"\nHoldback: {n_holdback}, CV pool: {len(cv_idx)}")

    events_cv = events[cv_idx].numpy()
    times_cv = times[cv_idx].numpy()

    # === Cross-validation ===
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_rel, val_rel) in enumerate(skf.split(cv_idx, events_cv)):
        print(f"\n=== Fold {fold} ===")

        train_idx = cv_idx[train_rel]
        val_idx = cv_idx[val_rel]

        train_t = torch.tensor(train_idx, dtype=torch.long)
        val_t = torch.tensor(val_idx, dtype=torch.long)

        model = AtlasTransformerV6(config).to(device)
        h_loss_mod = HierarchyLoss(args.hidden_dim, margin=0.1).to(device)
        mo_loss = MultiObjectiveLoss(n_losses=2).to(device)

        all_params = (list(model.parameters()) +
                      list(h_loss_mod.parameters()) +
                      list(mo_loss.parameters()))
        n_params = sum(p.numel() for p in all_params if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_c = 0.0
        best_state = None
        no_improve = 0
        batch_size = args.batch_size

        for epoch in range(args.epochs):
            model.train()
            perm = np.random.permutation(len(train_idx))
            epoch_loss = 0.0
            n_batches = 0

            epoch_h_conc = 0.0  # hierarchy concordance tracker

            total_batches = (len(perm) + batch_size - 1) // batch_size
            if epoch == 0:
                print(f"    {total_batches} batches/epoch, batch_size={batch_size}", flush=True)

            for b_start in range(0, len(perm), batch_size):
                b_rel = perm[b_start:b_start + batch_size]
                b_abs = train_t[b_rel]

                if epoch == 0 and n_batches % 25 == 0:
                    print(f"      batch {n_batches}/{total_batches}...", flush=True)

                optimizer.zero_grad()

                # Gather edge features for this batch: gene-pair matrix + patient features
                batch_edge_feats = gather_edge_features(
                    gene_pair_matrix_t, patient_edge_feats_t[b_abs],
                    gene_indices_t[b_abs], nm[b_abs],
                )

                hazard = model(
                    nf[b_abs], nm[b_abs], ct[b_abs],
                    clinical[b_abs], atlas_sums[b_abs],
                    batch_edge_feats,
                    block_ids_t[b_abs], channel_ids_t[b_abs],
                )

                # Cox survival loss
                loss_cox = cox_ph_loss(
                    hazard, times[b_abs].to(device), events[b_abs].to(device)
                )

                # Hierarchy loss (functional depth from VAF ordering)
                mut_embeds = model._last_mutation_embeddings  # (B, N, hidden)
                h_loss, h_stats = h_loss_mod(
                    mut_embeds,
                    pair_indices_t[b_abs].to(device),
                    pair_targets_t[b_abs].to(device),
                )

                # Multi-objective: learned weighting of Cox + hierarchy
                total_loss, weights = mo_loss([loss_cox, h_loss])

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                epoch_loss += loss_cox.item()
                epoch_h_conc += h_stats['concordance']
                n_batches += 1

            scheduler.step()

            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Gather edge features for validation set (in batches to save memory)
                    val_preds = []
                    for v_start in range(0, len(val_idx), batch_size):
                        v_end = min(v_start + batch_size, len(val_idx))
                        v_abs = val_t[v_start:v_end]
                        v_edge = gather_edge_features(
                            gene_pair_matrix_t, patient_edge_feats_t[v_abs],
                            gene_indices_t[v_abs], nm[v_abs],
                        )
                        h = model(
                            nf[v_abs], nm[v_abs], ct[v_abs],
                            clinical[v_abs], atlas_sums[v_abs],
                            v_edge,
                            block_ids_t[v_abs], channel_ids_t[v_abs],
                        )
                        val_preds.append(h.cpu())
                    h_val = torch.cat(val_preds).numpy().flatten()

                e_val = events[val_idx].numpy().astype(bool)
                t_val = times[val_idx].numpy()
                valid = t_val > 0
                try:
                    c = concordance_index_censored(
                        e_val[valid], t_val[valid], h_val[valid]
                    )[0]
                except Exception:
                    c = 0.5

                if c > best_c:
                    best_c = c
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if (epoch + 1) % 10 == 0:
                    h_conc_avg = epoch_h_conc / max(n_batches, 1)
                    w_str = f"w=[{weights[0]:.2f},{weights[1]:.2f}]" if weights else ""
                    print(f"    Epoch {epoch+1:3d}: cox={epoch_loss/n_batches:.4f} "
                          f"C={c:.4f} best={best_c:.4f} "
                          f"h_conc={h_conc_avg:.3f} {w_str}", flush=True)

                if no_improve >= args.patience // 5:
                    print(f"    Early stop epoch {epoch+1}, C-index: {best_c:.4f}")
                    break

        # Save fold
        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        if best_state:
            torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))
            torch.save(h_loss_mod.state_dict(), os.path.join(fold_dir, "hierarchy_head.pt"))

        print(f"  Fold {fold}: C-index = {best_c:.4f}")
        fold_results.append(best_c)

    # === Summary ===
    mean_c = np.mean(fold_results)
    std_c = np.std(fold_results)
    print(f"\n{'='*60}")
    print(f"  V6 MUTATION-LEVEL RESULTS")
    print(f"  Mean C-index: {mean_c:.4f} +/- {std_c:.4f}")
    print(f"  Per-fold: {[f'{c:.4f}' for c in fold_results]}")
    print(f"{'='*60}")

    # === Holdback evaluation ===
    if best_state:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            h_hold = torch.tensor(holdback_idx, dtype=torch.long)
            hold_preds = []
            for h_start in range(0, len(holdback_idx), batch_size):
                h_end = min(h_start + batch_size, len(holdback_idx))
                h_abs = h_hold[h_start:h_end]
                h_edge = gather_edge_features(
                    gene_pair_matrix_t, patient_edge_feats_t[h_abs],
                    gene_indices_t[h_abs], nm[h_abs],
                )
                h = model(
                    nf[h_abs], nm[h_abs], ct[h_abs],
                    clinical[h_abs], atlas_sums[h_abs],
                    h_edge,
                    block_ids_t[h_abs], channel_ids_t[h_abs],
                )
                hold_preds.append(h.cpu())
            h_pred = torch.cat(hold_preds).numpy().flatten()

        e_hold = events[holdback_idx].numpy().astype(bool)
        t_hold = times[holdback_idx].numpy()
        valid = t_hold > 0
        try:
            c_hold = concordance_index_censored(
                e_hold[valid], t_hold[valid], h_pred[valid]
            )[0]
        except Exception:
            c_hold = 0.5
        print(f"  Holdback C-index: {c_hold:.4f}")

        # === Atlas ablation: zero atlas features and re-evaluate ===
        with torch.no_grad():
            nf_ablated = nf.clone()
            nf_ablated[:, :, 0:2] = 0.0  # zero log_hr and CI width
            atlas_ablated = torch.zeros_like(atlas_sums)

            hold_preds_abl = []
            for h_start in range(0, len(holdback_idx), batch_size):
                h_end = min(h_start + batch_size, len(holdback_idx))
                h_abs = h_hold[h_start:h_end]
                h_edge = gather_edge_features(
                    gene_pair_matrix_t, patient_edge_feats_t[h_abs],
                    gene_indices_t[h_abs], nm[h_abs],
                )
                h = model(
                    nf_ablated[h_abs], nm[h_abs], ct[h_abs],
                    clinical[h_abs], atlas_ablated[h_abs],
                    h_edge, block_ids_t[h_abs], channel_ids_t[h_abs],
                )
                hold_preds_abl.append(h.cpu())
            h_pred_abl = torch.cat(hold_preds_abl).numpy().flatten()

        try:
            c_hold_ablated = concordance_index_censored(
                e_hold[valid], t_hold[valid], h_pred_abl[valid]
            )[0]
        except Exception:
            c_hold_ablated = 0.5

        atlas_delta = c_hold - c_hold_ablated
        print(f"  Holdback C-index (no atlas): {c_hold_ablated:.4f}")
        print(f"  Atlas contribution: {atlas_delta:+.4f}")

        # === Per-CT C-index on holdback ===
        ct_names_holdback = [ds.clinical.iloc[i]['CANCER_TYPE'] for i in holdback_idx]
        per_ct_c = {}
        for ct_name in set(ct_names_holdback):
            ct_mask = np.array([n == ct_name for n in ct_names_holdback])
            if ct_mask.sum() < 20:
                continue
            e_ct, t_ct, p_ct = e_hold[ct_mask], t_hold[ct_mask], h_pred[ct_mask]
            valid_ct = t_ct > 0
            if e_ct[valid_ct].sum() < 5:
                continue
            try:
                per_ct_c[ct_name] = float(concordance_index_censored(
                    e_ct[valid_ct], t_ct[valid_ct], p_ct[valid_ct]
                )[0])
            except Exception:
                pass
        # === Learned CT sign coefficients ===
        ct_sign_weights = model.ct_sign.weight.detach().cpu().numpy().flatten()
        ct_map_inv = {v: k for k, v in data['cancer_type_map'].items()}
        negative_cts = [(ct_map_inv.get(i, f'CT_{i}'), float(ct_sign_weights[i]))
                        for i in range(len(ct_sign_weights))
                        if ct_sign_weights[i] < 0.5 and i in ct_map_inv]
        if negative_cts:
            print(f"\n  Learned CT sign coefficients (< 0.5, potential inversions):")
            for ct_name, w in sorted(negative_cts, key=lambda x: x[1]):
                print(f"    {ct_name}: {w:.3f}")

        # Classify CTs by discriminative power and direction
        concordant_cts = {}   # C > 0.5: model agrees with atlas sign
        anticoncordant_cts = {}  # C < 0.5: real signal, wrong sign
        noise_cts = {}        # |C - 0.5| < 0.02: no signal

        for ct_name, c_val in per_ct_c.items():
            disc = abs(c_val - 0.5)
            entry = {'c_index': c_val, 'disc_power': disc}
            if disc < 0.02:
                noise_cts[ct_name] = entry
            elif c_val >= 0.5:
                concordant_cts[ct_name] = entry
            else:
                anticoncordant_cts[ct_name] = entry

        print(f"\n  Per-CT holdback C-index ({len(per_ct_c)} types):")
        print(f"    CONCORDANT (C > 0.5, atlas sign correct):")
        for ct_name, e in sorted(concordant_cts.items(), key=lambda x: -x[1]['disc_power']):
            print(f"      {ct_name}: C={e['c_index']:.4f}  |C-0.5|={e['disc_power']:.4f}")

        if anticoncordant_cts:
            print(f"\n    ANTI-CONCORDANT (C < 0.5, inverted signal — sign candidates):")
            for ct_name, e in sorted(anticoncordant_cts.items(), key=lambda x: -x[1]['disc_power']):
                # Verify: flip hazard sign and check if C improves
                ct_mask = np.array([n == ct_name for n in ct_names_holdback])
                e_ct = e_hold[ct_mask]
                t_ct = t_hold[ct_mask]
                p_ct = h_pred[ct_mask]
                valid_ct = t_ct > 0
                try:
                    c_flipped = concordance_index_censored(
                        e_ct[valid_ct], t_ct[valid_ct], -p_ct[valid_ct]
                    )[0]
                except Exception:
                    c_flipped = 0.5
                print(f"      {ct_name}: C={e['c_index']:.4f}  "
                      f"|C-0.5|={e['disc_power']:.4f}  "
                      f"C(flipped)={c_flipped:.4f}")

        if noise_cts:
            print(f"\n    NOISE (|C-0.5| < 0.02, no signal):")
            for ct_name in sorted(noise_cts):
                print(f"      {ct_name}: C={noise_cts[ct_name]['c_index']:.4f}")

        # Store enriched per-CT data for results.json
        per_ct_enriched = {}
        for ct_name, c_val in per_ct_c.items():
            disc = abs(c_val - 0.5)
            if c_val < 0.5 and disc >= 0.02:
                category = 'anti_concordant'
            elif disc < 0.02:
                category = 'noise'
            else:
                category = 'concordant'
            per_ct_enriched[ct_name] = {
                'c_index': c_val, 'disc_power': disc, 'category': category,
            }

        # === Multi-target graph predictions ===
        print("\n  Running multi-target graph predictions on holdback...")
        try:
            patient_genes_dict, cancer_types_dict, _ = load_patient_data()
            holdback_pids = [patient_ids[i] for i in holdback_idx]
            train_pids = [patient_ids[i] for i in cv_idx]

            # Co-mutation (clean edges only, then with train-only COOCCURS)
            cmp = CoMutationPredictor()
            cmp.load_clean_edges()
            comut_clean = cmp.evaluate(patient_genes_dict, cancer_types_dict,
                                       holdback_pids, train_pids, use_cooccurs=False)
            comut_with = cmp.evaluate(patient_genes_dict, cancer_types_dict,
                                      holdback_pids, train_pids, use_cooccurs=True)

            # Cancer type prediction
            ctp = CancerTypePredictor()
            ct_pred_result = ctp.evaluate(patient_genes_dict, cancer_types_dict,
                                          holdback_pids, train_pids)

            # Synthetic lethality (external, no holdback needed)
            slv = SyntheticLethalityValidator()
            sl_result = slv.validate()

            graph_pred_results = {
                'co_mutation_clean': comut_clean,
                'co_mutation_with_cooccurs': comut_with,
                'cancer_type': ct_pred_result,
                'synthetic_lethality': sl_result,
            }
        except Exception as e:
            print(f"  WARNING: Graph predictions failed ({e}), continuing...")
            graph_pred_results = {'error': str(e)}
    else:
        c_hold = 0.5
        c_hold_ablated = 0.5
        atlas_delta = 0.0
        per_ct_c = {}
        graph_pred_results = {'error': 'no model state'}

    # =====================================================================
    # STAGE 2: Per-CT ridge on transformer embeddings
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 2: PER-CT RIDGE ON EMBEDDINGS")
    print(f"{'='*60}")

    from sklearn.linear_model import Ridge

    perct_ridge_c = 0.5
    perct_fold_results = []

    if best_state:
        model.load_state_dict(best_state)
        model.eval()

        # Extract embeddings for ALL patients
        all_channel_embeds = []
        with torch.no_grad():
            for s in range(0, n_total, batch_size):
                e = min(s + batch_size, n_total)
                b_idx = torch.arange(s, e, dtype=torch.long)
                b_edge = gather_edge_features(
                    gene_pair_matrix_t, patient_edge_feats_t[b_idx],
                    gene_indices_t[b_idx], nm[b_idx],
                )
                _ = model(
                    nf[b_idx], nm[b_idx], ct[b_idx],
                    clinical[b_idx], atlas_sums[b_idx],
                    b_edge, block_ids_t[b_idx], channel_ids_t[b_idx],
                )
                # Channel tokens: (B, n_channels, hidden)
                ch_tok = model._last_channel_tokens.cpu()
                ch_active = model._last_channel_active.cpu()
                # Flatten: concat channel tokens → (B, n_channels * hidden)
                ch_flat = ch_tok.reshape(ch_tok.shape[0], -1)
                all_channel_embeds.append(ch_flat)

        embeddings = torch.cat(all_channel_embeds, dim=0).numpy()  # (N, n_ch * hidden)
        print(f"  Embeddings: {embeddings.shape}")

        # Per-CT ridge on CV folds
        ct_names_all = []
        if hasattr(ds, 'clinical'):
            ct_names_all = [ds.clinical.iloc[i].get('CANCER_TYPE', 'Unknown')
                           for i in range(n_total)]
        else:
            ct_arr = ct.numpy()
            ct_names_all = [str(ct_arr[i]) for i in range(n_total)]

        for fold, (train_rel, val_rel) in enumerate(skf.split(cv_idx, events_cv)):
            train_abs = cv_idx[train_rel]
            val_abs = cv_idx[val_rel]

            X_train = embeddings[train_abs]
            X_val = embeddings[val_abs]
            t_train = times[train_abs].numpy()
            e_train = events[train_abs].numpy()
            t_val = times[val_abs].numpy()
            e_val = events[val_abs].numpy().astype(bool)

            # Per-CT models
            ct_models = {}
            ct_train_groups = defaultdict(list)
            for idx in train_abs:
                ct_train_groups[ct_names_all[idx]].append(idx)

            # Global fallback
            y_train = t_train * (1 - e_train * 0.5)
            reg_global = Ridge(alpha=100.0)
            reg_global.fit(X_train, y_train)

            for ct_name, ct_indices in ct_train_groups.items():
                if len(ct_indices) >= 200:
                    ct_arr_local = np.array(ct_indices)
                    y_ct = times[ct_arr_local].numpy() * (1 - events[ct_arr_local].numpy() * 0.5)
                    reg_ct = Ridge(alpha=100.0)
                    reg_ct.fit(embeddings[ct_arr_local], y_ct)
                    ct_models[ct_name] = reg_ct

            # Predict
            scores = np.zeros(len(val_abs))
            for j, idx in enumerate(val_abs):
                ct_name = ct_names_all[idx]
                if ct_name in ct_models:
                    scores[j] = ct_models[ct_name].predict(embeddings[idx:idx+1])[0]
                else:
                    scores[j] = reg_global.predict(embeddings[idx:idx+1])[0]

            valid = t_val > 0
            try:
                c = concordance_index_censored(
                    e_val[valid], t_val[valid], -scores[valid]
                )[0]
            except Exception:
                c = 0.5
            perct_fold_results.append(c)

        if perct_fold_results:
            perct_ridge_c = np.mean(perct_fold_results)
            print(f"  Per-CT Ridge on embeddings: {perct_ridge_c:.4f} "
                  f"+/- {np.std(perct_fold_results):.4f}")
            print(f"  Folds: {[f'{c:.4f}' for c in perct_fold_results]}")
        else:
            print("  Per-CT Ridge: no results")

    # =====================================================================
    # STAGE 3: Per-CT fine-tuning (large CTs + low-outcome cluster)
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 3: PER-CT FINE-TUNING")
    print(f"{'='*60}")

    finetune_results = {}

    if best_state:
        # Identify CTs with enough patients for fine-tuning
        ct_counts = defaultdict(int)
        ct_event_counts = defaultdict(int)
        ct_median_time = defaultdict(list)
        for i in range(n_total):
            ct_name = ct_names_all[i]
            ct_counts[ct_name] += 1
            if events[i].item() > 0:
                ct_event_counts[ct_name] += 1
            ct_median_time[ct_name].append(times[i].item())

        # Large CTs (≥500 patients): individual fine-tuning
        large_cts = {ct for ct, n in ct_counts.items()
                     if n >= 500 and ct_event_counts[ct] >= 50}

        # Low-outcome cluster: CTs where median survival < 24 months
        # AND graph walker C < 0.55
        low_outcome_cts = set()
        for ct_name, time_list in ct_median_time.items():
            median_t = np.median(time_list)
            if (median_t < 24 and ct_counts[ct_name] >= 100
                    and ct_event_counts[ct_name] >= 20
                    and ct_name not in large_cts):
                low_outcome_cts.add(ct_name)

        print(f"  Large CTs for individual fine-tuning ({len(large_cts)}):")
        for ct_name in sorted(large_cts):
            print(f"    {ct_name}: {ct_counts[ct_name]} patients, "
                  f"{ct_event_counts[ct_name]} events")

        print(f"\n  Low-outcome cluster ({len(low_outcome_cts)} CTs):")
        for ct_name in sorted(low_outcome_cts):
            med = np.median(ct_median_time[ct_name])
            print(f"    {ct_name}: {ct_counts[ct_name]} patients, "
                  f"median OS={med:.1f}mo")

        ft_epochs = 30
        ft_lr = 1e-4  # lower LR for fine-tuning

        # Fine-tune per large CT
        for ct_name in sorted(large_cts):
            ct_patient_idx = np.array([i for i in range(n_total)
                                       if ct_names_all[i] == ct_name])
            ct_events = events[ct_patient_idx].numpy()
            ct_n = len(ct_patient_idx)

            # Simple train/val split (80/20)
            np.random.seed(42)
            perm = np.random.permutation(ct_n)
            split = int(ct_n * 0.8)
            ct_train = ct_patient_idx[perm[:split]]
            ct_val = ct_patient_idx[perm[split:]]

            # Clone model from global best
            ft_model = AtlasTransformerV6(config).to(device)
            ft_model.load_state_dict(best_state)
            ft_optimizer = torch.optim.Adam(ft_model.parameters(),
                                            lr=ft_lr, weight_decay=1e-4)

            best_ft_c = 0.0
            ct_train_t = torch.tensor(ct_train, dtype=torch.long)

            for ep in range(ft_epochs):
                ft_model.train()
                perm_ep = np.random.permutation(len(ct_train))
                for b_start in range(0, len(perm_ep), batch_size):
                    b_rel = perm_ep[b_start:b_start + batch_size]
                    b_abs = ct_train_t[b_rel]
                    ft_optimizer.zero_grad()
                    b_edge = gather_edge_features(
                        gene_pair_matrix_t, patient_edge_feats_t[b_abs],
                        gene_indices_t[b_abs], nm[b_abs],
                    )
                    hazard = ft_model(
                        nf[b_abs], nm[b_abs], ct[b_abs],
                        clinical[b_abs], atlas_sums[b_abs],
                        b_edge, block_ids_t[b_abs], channel_ids_t[b_abs],
                    )
                    loss = cox_ph_loss(
                        hazard, times[b_abs].to(device), events[b_abs].to(device)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
                    ft_optimizer.step()

                # Validate every 5 epochs
                if (ep + 1) % 5 == 0:
                    ft_model.eval()
                    with torch.no_grad():
                        v_abs = torch.tensor(ct_val, dtype=torch.long)
                        v_edge = gather_edge_features(
                            gene_pair_matrix_t, patient_edge_feats_t[v_abs],
                            gene_indices_t[v_abs], nm[v_abs],
                        )
                        h = ft_model(
                            nf[v_abs], nm[v_abs], ct[v_abs],
                            clinical[v_abs], atlas_sums[v_abs],
                            v_edge, block_ids_t[v_abs], channel_ids_t[v_abs],
                        ).cpu().numpy().flatten()

                    e_v = events[ct_val].numpy().astype(bool)
                    t_v = times[ct_val].numpy()
                    valid = t_v > 0
                    try:
                        c = concordance_index_censored(e_v[valid], t_v[valid], h[valid])[0]
                    except Exception:
                        c = 0.5
                    best_ft_c = max(best_ft_c, c)

            finetune_results[ct_name] = {
                'c_index': float(best_ft_c),
                'n_patients': int(ct_n),
                'method': 'individual',
            }
            print(f"    {ct_name}: {best_ft_c:.4f} (n={ct_n})")

        # Fine-tune on low-outcome cluster (pooled)
        if low_outcome_cts:
            cluster_idx = np.array([i for i in range(n_total)
                                    if ct_names_all[i] in low_outcome_cts])
            cluster_n = len(cluster_idx)
            print(f"\n  Low-outcome cluster fine-tuning: {cluster_n} patients "
                  f"across {len(low_outcome_cts)} CTs")

            np.random.seed(42)
            perm = np.random.permutation(cluster_n)
            split = int(cluster_n * 0.8)
            cl_train = cluster_idx[perm[:split]]
            cl_val = cluster_idx[perm[split:]]

            ft_model = AtlasTransformerV6(config).to(device)
            ft_model.load_state_dict(best_state)
            ft_optimizer = torch.optim.Adam(ft_model.parameters(),
                                            lr=ft_lr, weight_decay=1e-4)

            best_cluster_c = 0.0
            cl_train_t = torch.tensor(cl_train, dtype=torch.long)

            for ep in range(ft_epochs):
                ft_model.train()
                perm_ep = np.random.permutation(len(cl_train))
                for b_start in range(0, len(perm_ep), batch_size):
                    b_rel = perm_ep[b_start:b_start + batch_size]
                    b_abs = cl_train_t[b_rel]
                    ft_optimizer.zero_grad()
                    b_edge = gather_edge_features(
                        gene_pair_matrix_t, patient_edge_feats_t[b_abs],
                        gene_indices_t[b_abs], nm[b_abs],
                    )
                    hazard = ft_model(
                        nf[b_abs], nm[b_abs], ct[b_abs],
                        clinical[b_abs], atlas_sums[b_abs],
                        b_edge, block_ids_t[b_abs], channel_ids_t[b_abs],
                    )
                    loss = cox_ph_loss(
                        hazard, times[b_abs].to(device), events[b_abs].to(device)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
                    ft_optimizer.step()

                if (ep + 1) % 5 == 0:
                    ft_model.eval()
                    with torch.no_grad():
                        v_abs = torch.tensor(cl_val, dtype=torch.long)
                        v_edge = gather_edge_features(
                            gene_pair_matrix_t, patient_edge_feats_t[v_abs],
                            gene_indices_t[v_abs], nm[v_abs],
                        )
                        h = ft_model(
                            nf[v_abs], nm[v_abs], ct[v_abs],
                            clinical[v_abs], atlas_sums[v_abs],
                            v_edge, block_ids_t[v_abs], channel_ids_t[v_abs],
                        ).cpu().numpy().flatten()

                    e_v = events[cl_val].numpy().astype(bool)
                    t_v = times[cl_val].numpy()
                    valid = t_v > 0
                    try:
                        c = concordance_index_censored(e_v[valid], t_v[valid], h[valid])[0]
                    except Exception:
                        c = 0.5
                    best_cluster_c = max(best_cluster_c, c)

            # Per-CT breakdown within cluster
            print(f"    Cluster overall: {best_cluster_c:.4f}")
            for ct_name in sorted(low_outcome_cts):
                ct_val_mask = np.array([ct_names_all[i] == ct_name for i in cl_val])
                if ct_val_mask.sum() >= 10:
                    ct_indices = cl_val[ct_val_mask]
                    e_ct = events[ct_indices].numpy().astype(bool)
                    t_ct = times[ct_indices].numpy()
                    # Use last ft_model predictions
                    try:
                        ct_in_val = np.where(ct_val_mask)[0]
                        h_ct = h[ct_in_val]
                        valid_ct = t_ct > 0
                        c_ct = concordance_index_censored(
                            e_ct[valid_ct], t_ct[valid_ct], h_ct[valid_ct]
                        )[0]
                        finetune_results[ct_name] = {
                            'c_index': float(c_ct),
                            'n_patients': int(ct_val_mask.sum()),
                            'method': 'low_outcome_cluster',
                        }
                        print(f"      {ct_name}: {c_ct:.4f} (n={ct_val_mask.sum()})")
                    except Exception:
                        pass

            finetune_results['_low_outcome_cluster'] = {
                'c_index': float(best_cluster_c),
                'n_patients': int(cluster_n),
                'n_cancer_types': len(low_outcome_cts),
                'cancer_types': sorted(low_outcome_cts),
            }

    # === Save results ===
    results = {
        'model': 'AtlasTransformerV6',
        'version': 'v6_temporal_clean',
        'mean_c_index': float(mean_c),
        'std_c_index': float(std_c),
        'holdback_c_index': float(c_hold),
        'holdback_c_index_no_atlas': float(c_hold_ablated),
        'atlas_contribution': float(atlas_delta),
        'fold_results': [float(c) for c in fold_results],
        'per_ct_holdback': per_ct_enriched,
        'config': config,
        'n_patients': n_total,
        'n_holdback': n_holdback,
        'temporal': {
            'year_range': [float(years.min()), float(years.max())],
            'mean_confidence': float(confs.mean()),
        },
        'graph_predictions': graph_pred_results,
        'hierarchy_pairs': pair_stats,
        'cancer_type_map': data['cancer_type_map'],
        'ct_sign_weights': {ct_map_inv.get(i, f'CT_{i}'): float(ct_sign_weights[i])
                            for i in range(len(ct_sign_weights))
                            if i in ct_map_inv},
        'fresh_cache': args.fresh,
        'perct_ridge_on_embeddings': {
            'mean_c': float(perct_ridge_c),
            'fold_results': [float(c) for c in perct_fold_results],
        },
        'finetune_results': finetune_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
