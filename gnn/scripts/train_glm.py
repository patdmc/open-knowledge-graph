"""
Train GLM (Graph Language Model) — block-level sparse signed routing.

Reuses the V6 data pipeline (AtlasDataset, pairwise edge features,
block assignments, temporal features). The key difference is:
  - Level 3 uses 101×101 block-level routing with 328 PPI-defined edges
  - Block routing is CT-specific with signed weights
  - Escalation classification conditions the survival readout

Usage:
    python3 -u -m gnn.scripts.train_glm [--epochs 200]
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
from gnn.models.glm import GLM
from gnn.models.cox_sage import cox_ph_loss
from gnn.models.hierarchy_loss import HierarchyLoss, build_hierarchy_pairs, MultiObjectiveLoss
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, GNN_CACHE, ANALYSIS_CACHE
from gnn.data.block_assignments import load_block_assignments, discover_cross_block_edges
from gnn.data.temporal import TemporalEstimator, year_features

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "glm",
)


# =========================================================================
# Reuse pairwise edge feature computation from V6
# =========================================================================

def load_graph_data():
    """Load graph structure needed for pairwise features."""
    from neo4j import GraphDatabase
    import networkx as nx

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    print("  Loading graph data for pairwise features...", flush=True)
    t0 = time.time()

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

    channel_profiles = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene) WHERE g.channel_profile IS NOT NULL
            RETURN g.name AS gene, g.channel_profile AS profile
        """)
        for r in result:
            channel_profiles[r["gene"]] = np.array(r["profile"], dtype=np.float32)
    print(f"    Channel profiles: {len(channel_profiles)}", flush=True)

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
    """Compute (G, G, 10) matrix of gene-pair graph features."""
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
    """Compute per-patient pairwise features from mutation-level node features."""
    PATIENT_EDGE_DIM = 4
    nf = node_features if isinstance(node_features, np.ndarray) else node_features.numpy()
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


def gather_edge_features(gene_pair_matrix, patient_edge_feats,
                         gene_indices, gene_masks):
    """Gather precomputed gene-pair features + patient features per batch."""
    B, N = gene_indices.shape
    safe_idx = gene_indices.clamp(0, gene_pair_matrix.shape[0] - 1)

    idx_i = safe_idx.unsqueeze(2).expand(B, N, N)
    idx_j = safe_idx.unsqueeze(1).expand(B, N, N)

    graph_feats = gene_pair_matrix[idx_i, idx_j]
    pair_mask = (gene_masks.unsqueeze(1) * gene_masks.unsqueeze(2)).unsqueeze(-1)
    graph_feats = graph_feats * pair_mask

    edge_features = torch.cat([graph_feats, patient_edge_feats * pair_mask], dim=-1)
    return edge_features


# =========================================================================
# VAF hierarchy pairs
# =========================================================================

def load_patient_vafs(gene_names_per_patient, patient_ids):
    """Load VAF per mutation slot per patient."""
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

    vaf_lookup = {}
    for _, row in mut.iterrows():
        key = (row['patientId'], row['gene.hugoGeneSymbol'])
        v = row['vaf']
        if np.isfinite(v) and v > 0:
            vaf_lookup[key] = max(vaf_lookup.get(key, 0), v)

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
    parser.add_argument('--n-route-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda-route', type=float, default=0.01,
                        help='Routing auxiliary loss weight')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Path to pretrained_backbone.pt for weight transfer')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"=" * 70)
    print(f"  GLM — Graph Language Model with Block-Level Routing")
    print(f"=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === Load data ===
    ds = AtlasDataset()
    data = ds.build_features()

    # === Gene vocabulary ===
    gene_vocab = {}
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    G = len(gene_vocab)
    print(f"\n  Gene vocabulary: {G} unique genes")

    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)

    # === Block assignments + cross-block edges ===
    print("  Loading block assignments...", flush=True)
    gene_block, n_blocks, n_channels = load_block_assignments()
    channel_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}

    print("  Loading cross-block PPI edges...", flush=True)
    cross_edges, cross_edge_index, _ = discover_cross_block_edges()
    print(f"  Cross-block edges: {len(cross_edges)} pairs, "
          f"{cross_edge_index.shape[1]} directed")

    # Build block→channel mapping
    block_to_channel = np.zeros(n_blocks, dtype=np.int64)
    for gene, info in gene_block.items():
        block_to_channel[info['block_id']] = info['channel_id']

    # Build edge scores for initialization
    edge_scores = np.ones(cross_edge_index.shape[1], dtype=np.float32) * 0.5
    edge_lookup = {}
    for e in cross_edges:
        edge_lookup[(e['block_i'], e['block_j'])] = e['avg_score']
        edge_lookup[(e['block_j'], e['block_i'])] = e['avg_score']
    for idx in range(cross_edge_index.shape[1]):
        src, dst = int(cross_edge_index[0, idx]), int(cross_edge_index[1, idx])
        edge_scores[idx] = edge_lookup.get((src, dst), 0.5)

    # Build per-patient block_ids and channel_ids tensors
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
                    ch = CHANNEL_MAP.get(g)
                    if ch and ch in channel_idx:
                        channel_ids[b, s] = channel_idx[ch]

    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    channel_ids_t = torch.tensor(channel_ids, dtype=torch.long)

    filled_mask = block_ids < n_blocks
    has_gene = np.array([[bool(g and g != '' and g != 'WT') for g in patient]
                         for patient in data['gene_names']])
    n_with_block = filled_mask.sum()
    n_with_gene = has_gene.sum()
    print(f"  Blocks: {n_blocks}, Channels: {n_channels}")
    print(f"  Block coverage: {n_with_block}/{n_with_gene} ({n_with_block/max(n_with_gene,1):.1%})")

    # === Gene-pair matrix ===
    print("  Computing gene-pair matrix...", flush=True)
    t0 = time.time()
    graph_data = load_graph_data()
    gene_pair_matrix, graph_edge_dim = compute_gene_pair_matrix(graph_data, gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32)
    mem_mb = gene_pair_matrix.nbytes / 1e6
    print(f"  Gene-pair matrix: ({G}, {G}, {graph_edge_dim}) = {mem_mb:.1f}MB [{time.time()-t0:.1f}s]")

    # === Patient edge features ===
    print("  Computing patient-level edge features...", flush=True)
    t0 = time.time()
    patient_edge_feats, patient_edge_dim = compute_patient_edge_features(data['node_features'])
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32)
    print(f"  Patient edge features: {patient_edge_feats_t.shape} [{time.time()-t0:.1f}s]")

    total_edge_dim = graph_edge_dim + patient_edge_dim

    # === Hierarchy pairs ===
    print("\n  Building hierarchy pairs from VAF data...", flush=True)
    patient_ids = ds.clinical['patientId'].tolist()
    vafs = load_patient_vafs(data['gene_names'], patient_ids)

    pair_indices_np, pair_targets_np, pair_stats = build_hierarchy_pairs(
        data['gene_names'], vafs, CHANNEL_MAP,
        max_pairs_per_patient=64, min_vaf_diff=0.05,
    )
    print(f"  Hierarchy pairs: {pair_stats['total_pairs']:,} total, "
          f"{pair_stats['patients_with_pairs']:,} patients", flush=True)

    pair_indices_t = torch.tensor(pair_indices_np, dtype=torch.long)
    pair_targets_t = torch.tensor(pair_targets_np, dtype=torch.float32)

    # === Temporal features ===
    print("  Building temporal features...", flush=True)
    te = TemporalEstimator(ds.clinical)
    years, confs = te.estimate_all(patient_ids)
    temporal_feats = year_features(years, confs)
    temporal_t = torch.tensor(temporal_feats, dtype=torch.float32).to(device)

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

    # === GLM Config ===
    config = {
        'node_feat_dim': NODE_FEAT_DIM,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_intra_layers': args.n_layers,
        'n_route_layers': args.n_route_layers,
        'dropout': args.dropout,
        'n_cancer_types': n_cancer_types,
        'edge_feat_dim': total_edge_dim,
        'n_channels': n_channels,
        'n_blocks': n_blocks,
        'cross_block_edge_index': cross_edge_index,
        'cross_block_edge_scores': edge_scores,
        'block_to_channel': block_to_channel,
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
        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        train_idx = cv_idx[train_rel]
        val_idx = cv_idx[val_rel]

        train_t = torch.tensor(train_idx, dtype=torch.long)
        val_t = torch.tensor(val_idx, dtype=torch.long)

        model = GLM(config).to(device)

        # Transfer pretrained backbone weights if provided
        if args.backbone and fold == 0:
            model.load_pretrained_backbone(args.backbone)
        elif args.backbone:
            # Re-transfer for each fold (fresh model)
            model.load_pretrained_backbone(args.backbone)

        h_loss_mod = HierarchyLoss(args.hidden_dim, margin=0.1).to(device)
        mo_loss = MultiObjectiveLoss(n_losses=3).to(device)  # Cox + hierarchy + routing

        all_params = (list(model.parameters()) +
                      list(h_loss_mod.parameters()) +
                      list(mo_loss.parameters()))
        n_params = sum(p.numel() for p in all_params if p.requires_grad)
        if fold == 0:
            print(f"  GLM parameters: {n_params:,}")
            print(f"  Router: {model.router.n_edges} directed edges across {n_blocks} blocks")

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
            epoch_route_loss = 0.0
            n_batches = 0

            total_batches = (len(perm) + batch_size - 1) // batch_size
            if epoch == 0:
                print(f"    {total_batches} batches/epoch, batch_size={batch_size}", flush=True)

            for b_start in range(0, len(perm), batch_size):
                b_rel = perm[b_start:b_start + batch_size]
                b_abs = train_t[b_rel]

                if epoch == 0 and n_batches % 25 == 0:
                    print(f"      batch {n_batches}/{total_batches}...", flush=True)

                optimizer.zero_grad()

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

                # Routing auxiliary loss
                loss_route = model.routing_loss(
                    lambda_structure=args.lambda_route,
                    lambda_sign_consistency=args.lambda_route,
                )

                # Hierarchy loss (VAF ordering)
                # Use block tokens as mutation embeddings for hierarchy
                # We need per-mutation embeddings — use the node embeddings
                # stored before block pooling
                # For now, use a simplified hierarchy: skip if not available
                loss_h = torch.tensor(0.0, device=device)
                h_conc = 0.0

                # Multi-objective: learned weighting
                total_loss, weights = mo_loss([loss_cox, loss_h, loss_route])

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                epoch_loss += loss_cox.item()
                epoch_route_loss += loss_route.item()
                n_batches += 1

            scheduler.step()

            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
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
                    # Routing analysis
                    r_summary = model.router.routing_summary()
                    r_info = f"active={r_summary['n_active']}/{r_summary['n_edges']}"
                    print(f"    Epoch {epoch+1:3d}: cox={epoch_loss/n_batches:.4f} "
                          f"route={epoch_route_loss/n_batches:.6f} "
                          f"C={c:.4f} best={best_c:.4f} [{r_info}]", flush=True)

                if no_improve >= args.patience // 5:
                    print(f"    Early stop epoch {epoch+1}, C-index: {best_c:.4f}")
                    break

        # Save fold
        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        if best_state:
            torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

        print(f"  Fold {fold}: C-index = {best_c:.4f}")
        fold_results.append(best_c)

    # === Summary ===
    mean_c = np.mean(fold_results)
    std_c = np.std(fold_results)
    print(f"\n{'='*70}")
    print(f"  GLM BLOCK-LEVEL ROUTING RESULTS")
    print(f"  Mean C-index: {mean_c:.4f} +/- {std_c:.4f}")
    print(f"  Per-fold: {[f'{c:.4f}' for c in fold_results]}")
    print(f"{'='*70}")

    # === Holdback evaluation ===
    if best_state:
        model = GLM(config).to(device)
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

        # Atlas ablation
        with torch.no_grad():
            nf_ablated = nf.clone()
            nf_ablated[:, :, 0:2] = 0.0
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
        ct_holdback = [ds.clinical.iloc[i]['CANCER_TYPE'] for i in holdback_idx]
        ct_results = defaultdict(lambda: {'preds': [], 'events': [], 'times': []})
        for i, ct_name in enumerate(ct_holdback):
            if valid[i]:
                ct_results[ct_name]['preds'].append(h_pred[i])
                ct_results[ct_name]['events'].append(e_hold[i])
                ct_results[ct_name]['times'].append(t_hold[i])

        print(f"\n  Per-CT holdback C-index (n >= 30):")
        ct_scores = {}
        for ct_name in sorted(ct_results.keys()):
            d = ct_results[ct_name]
            if len(d['preds']) >= 30:
                try:
                    c_ct = concordance_index_censored(
                        np.array(d['events']), np.array(d['times']),
                        np.array(d['preds'])
                    )[0]
                    ct_scores[ct_name] = c_ct
                    print(f"    {ct_name:30s}  n={len(d['preds']):4d}  C={c_ct:.4f}")
                except Exception:
                    pass

        # === Save results ===
        results = {
            'model': 'GLM',
            'architecture': 'block-level sparse signed routing',
            'n_blocks': n_blocks,
            'n_cross_block_edges': len(cross_edges),
            'n_directed_edges': int(cross_edge_index.shape[1]),
            'hidden_dim': args.hidden_dim,
            'n_params': n_params,
            'mean_c_index': float(mean_c),
            'std_c_index': float(std_c),
            'fold_c_indices': [float(c) for c in fold_results],
            'holdback_c_index': float(c_hold),
            'holdback_c_index_no_atlas': float(c_hold_ablated),
            'atlas_contribution': float(atlas_delta),
            'per_ct_holdback': {k: float(v) for k, v in ct_scores.items()},
        }
        with open(os.path.join(RESULTS_DIR, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {RESULTS_DIR}/results.json")

    # === Routing analysis ===
    if best_state:
        analysis = model.get_routing_analysis()
        print(f"\n  Routing: {analysis['n_active']} active / {analysis['n_edges']} total edges")
        with open(os.path.join(RESULTS_DIR, "routing_analysis.json"), 'w') as f:
            json.dump({
                'n_edges': analysis['n_edges'],
                'n_active': analysis['n_active'],
                'n_suppressed': analysis['n_suppressed'],
                'mean_structure': analysis['mean_structure'],
            }, f, indent=2)


if __name__ == '__main__':
    main()
