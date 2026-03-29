#!/usr/bin/env python3
"""
Pre-train AtlasTransformer V6 backbone on DepMap cell line essentiality.

The backbone learns biologically meaningful mutation interaction patterns
from CRISPR knockout data before fine-tuning on patient survival.

Pre-training signal: predict per-gene Chronos essentiality scores from
cell line mutation profiles. 489 cell lines, ~5K valid targets.

Augmentation: mutation dropout (randomly mask 10-30% of nodes per sample)
to learn robust representations that don't depend on any single mutation.

Uses the same edge feature pipeline as train_atlas_transformer_v6.py:
gene-pair matrix (G, G, 10) + patient-level features (B, N, N, 4).

Usage:
    python3 -u -m gnn.scripts.pretrain_depmap
    python3 -u -m gnn.scripts.pretrain_depmap --epochs 200 --lr 1e-3
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.cell_line_dataset import CellLineDataset
from gnn.data.atlas_dataset import MAX_NODES, NODE_FEAT_DIM
from gnn.models.atlas_transformer_v6 import AtlasTransformerV6
from gnn.models.essentiality_head import EssentialityHead, EssentialityLoss
from gnn.config import GNN_RESULTS, CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mut-dropout-min", type=float, default=0.1)
    parser.add_argument("--mut-dropout-max", type=float, default=0.3)
    parser.add_argument("--rank-weight", type=float, default=0.3)
    return parser.parse_args()


# =========================================================================
# Edge feature pipeline (matches train_atlas_transformer_v6.py)
# =========================================================================

def load_graph_data():
    """Load graph structure for pairwise features."""
    from gnn.config import GNN_CACHE
    from neo4j import GraphDatabase
    import networkx as nx

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    print("  Loading graph data for pairwise features...", flush=True)

    # PPI distances
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

    # Channel profiles
    channel_profiles = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene) WHERE g.channel_profile IS NOT NULL
            RETURN g.name AS gene, g.channel_profile AS profile
        """)
        for r in result:
            channel_profiles[r["gene"]] = np.array(r["profile"], dtype=np.float32)

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

    # Synthetic lethal partners
    sl_pairs = set()
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[:SL_PARTNER]-(g2:Gene) WHERE g1.name < g2.name
            RETURN g1.name AS g1, g2.name AS g2
        """)
        for r in result:
            sl_pairs.add((r["g1"], r["g2"]))

    driver.close()

    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set |= hubs

    return {
        'ppi_dists': ppi_dists,
        'channel_profiles': channel_profiles,
        'cooccurrence': cooccurrence,
        'max_cooccur': max_cooccur,
        'sl_pairs': sl_pairs,
        'hub_set': hub_set,
    }


def compute_gene_pair_matrix(graph_data, gene_vocab):
    """Compute (G, G, 10) gene-pair feature matrix."""
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
        matrix[i, i, 9] = 1.0  # self-loop
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
                ni, nj = np.linalg.norm(pi), np.linalg.norm(pj)
                if ni > 0 and nj > 0:
                    matrix[i, j, 6] = matrix[j, i, 6] = np.dot(pi, pj) / (ni * nj)

            matrix[i, j, 7] = matrix[j, i, 7] = 1.0 if (gi in hub_set and gj in hub_set) else 0.0
            matrix[i, j, 8] = matrix[j, i, 8] = 1.0 if pair in sl_pairs else 0.0

    return matrix, GRAPH_EDGE_DIM


def compute_patient_edge_features(node_features_np):
    """Compute per-patient pairwise features from mutation HR values."""
    PATIENT_EDGE_DIM = 4
    nf = node_features_np
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
    """Gather gene-pair + patient features into (B, N, N, D) tensor."""
    B, N = gene_indices.shape
    safe_idx = gene_indices.clamp(0, gene_pair_matrix.shape[0] - 1)

    idx_i = safe_idx.unsqueeze(2).expand(B, N, N)
    idx_j = safe_idx.unsqueeze(1).expand(B, N, N)
    graph_feats = gene_pair_matrix[idx_i, idx_j]

    pair_mask = (gene_masks.unsqueeze(1) * gene_masks.unsqueeze(2)).unsqueeze(-1)
    graph_feats = graph_feats * pair_mask

    return torch.cat([graph_feats, patient_edge_feats * pair_mask], dim=-1)


# =========================================================================
# Training
# =========================================================================

def apply_mutation_dropout(node_features, node_masks, edge_features,
                           essentiality, essentiality_masks,
                           drop_min=0.1, drop_max=0.3):
    """Randomly mask out mutations during training."""
    B, N, _ = node_features.shape
    nf = node_features.clone()
    nm = node_masks.clone()
    ef = edge_features.clone()
    ess = essentiality.clone()
    em = essentiality_masks.clone()

    for b in range(B):
        real = nm[b].bool()
        n_real = real.sum().item()
        if n_real <= 1:
            continue
        drop_rate = np.random.uniform(drop_min, drop_max)
        n_drop = max(1, int(n_real * drop_rate))
        n_drop = min(n_drop, n_real - 1)
        real_indices = real.nonzero(as_tuple=True)[0]
        drop_indices = real_indices[torch.randperm(len(real_indices))[:n_drop]]
        nm[b, drop_indices] = 0.0
        nf[b, drop_indices] = 0.0
        ef[b, drop_indices, :] = 0.0
        ef[b, :, drop_indices] = 0.0
        em[b, drop_indices] = 0.0

    return nf, nm, ef, ess, em


def train_epoch(model, head, loss_fn, optimizer, train_loader,
                gene_pair_matrix_t, device, args):
    model.train()
    head.train()
    total_loss = 0
    total_mse = 0
    total_rank = 0
    n_batches = 0

    for batch in train_loader:
        nf, nm, ct, clin, gi, pef, bi, ci, ess, em = [
            b.to(device) for b in batch
        ]

        # Gather edge features at batch time
        ef = gather_edge_features(
            gene_pair_matrix_t.to(device), pef, gi, nm
        )

        # Mutation dropout
        nf, nm, ef, ess, em = apply_mutation_dropout(
            nf, nm, ef, ess, em,
            drop_min=args.mut_dropout_min,
            drop_max=args.mut_dropout_max,
        )

        optimizer.zero_grad()
        node_hidden, _ = model.encode(nf, nm, ct, clin, ef, bi, ci)
        predictions = head(node_hidden)
        loss, metrics = loss_fn(predictions, ess, em)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), 1.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_mse += metrics["mse"]
        total_rank += metrics["rank"]
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mse": total_mse / max(n_batches, 1),
        "rank": total_rank / max(n_batches, 1),
    }


@torch.no_grad()
def eval_epoch(model, head, loss_fn, val_tensors, gene_pair_matrix_t, device):
    model.eval()
    head.eval()

    nf, nm, ct, clin, gi, pef, bi, ci, ess, em = [
        v.to(device) for v in val_tensors
    ]

    ef = gather_edge_features(gene_pair_matrix_t.to(device), pef, gi, nm)
    node_hidden, _ = model.encode(nf, nm, ct, clin, ef, bi, ci)
    predictions = head(node_hidden)
    loss, metrics = loss_fn(predictions, ess, em)

    valid = em.bool()
    if valid.sum() > 10:
        p = predictions[valid].cpu().numpy()
        t = ess[valid].cpu().numpy()
        corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0
    else:
        corr = 0.0

    return {
        "loss": loss.item(),
        "mse": metrics["mse"],
        "rank": metrics["rank"],
        "corr": corr,
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build cell line dataset (base features only — V6 uses its own edge pipeline)
    ds = CellLineDataset()
    data = ds.build_features()

    node_features = data["node_features"]
    node_masks = data["node_masks"]
    cancer_types = data["cancer_types"]
    ages = data["ages"]
    sexes = data["sexes"]
    essentiality = data["essentiality"]
    essentiality_masks = data["essentiality_masks"]
    n_cancer_types = data["n_cancer_types"]
    gene_names = data["gene_names"]

    # Clinical: V6 uses (B, 5) = [age_z, sex, norm_year, year_confidence, is_mature]
    # Cell lines have no temporal data, so zero those out
    N = len(cancer_types)
    clinical = torch.zeros(N, 5, dtype=torch.float32)
    clinical[:, 0] = ages  # age_z (already 0 for cell lines)
    clinical[:, 1] = sexes

    # Build gene vocabulary (same as V6 training script)
    gene_vocab = {}
    for patient_genes in gene_names:
        for g in patient_genes:
            if g and g != "" and g != "WT" and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    G = len(gene_vocab)
    print(f"\n  Gene vocabulary: {G} unique genes")

    # Map gene names → vocab indices
    gene_indices = np.zeros((N, MAX_NODES), dtype=np.int64)
    for b, genes in enumerate(gene_names):
        for s, g in enumerate(genes):
            if g and g != "" and g != "WT" and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)

    # Block/channel assignments
    from gnn.data.block_assignments import load_block_assignments
    gene_block, n_blocks, n_channels = load_block_assignments()
    channel_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}

    block_ids = np.full((N, MAX_NODES), n_blocks, dtype=np.int64)
    channel_ids = np.full((N, MAX_NODES), n_channels, dtype=np.int64)

    for b, genes in enumerate(gene_names):
        for s, g in enumerate(genes):
            if g and g != "" and g != "WT":
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

    # Compute gene-pair matrix (graph-level features, computed once)
    print("  Computing gene-pair matrix...", flush=True)
    graph_data = load_graph_data()
    gene_pair_matrix, graph_edge_dim = compute_gene_pair_matrix(graph_data, gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32)
    print(f"  Gene-pair matrix: ({G}, {G}, {graph_edge_dim})")

    # Compute patient-level edge features (vectorized)
    patient_edge_feats, patient_edge_dim = compute_patient_edge_features(
        node_features.numpy()
    )
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32)
    total_edge_dim = graph_edge_dim + patient_edge_dim
    print(f"  Edge feature dim: {graph_edge_dim} (graph) + {patient_edge_dim} (patient) = {total_edge_dim}")

    print(f"\nDataset: {N} cell lines")
    print(f"  Node features: {node_features.shape}")
    print(f"  Valid essentiality targets: {int(essentiality_masks.sum().item())}")

    # Model config
    config = {
        "node_feat_dim": NODE_FEAT_DIM,
        "hidden_dim": args.hidden_dim,
        "edge_feat_dim": total_edge_dim,
        "n_channels": n_channels,
        "n_blocks": n_blocks,
        "n_cancer_types": n_cancer_types,
        "n_heads": args.n_heads,
        "n_intra_layers": args.n_layers,
        "dropout": args.dropout,
    }

    # Results dir
    results_dir = os.path.join(GNN_RESULTS, "depmap_pretrain")
    os.makedirs(results_dir, exist_ok=True)

    # Train/val split
    indices = np.arange(N)
    ct_np = cancer_types.numpy()
    ct_counts = Counter(ct_np)
    strat_labels = np.array([ct if ct_counts[ct] >= 5 else -1 for ct in ct_np])
    rare_count = (strat_labels == -1).sum()

    from sklearn.model_selection import train_test_split
    use_stratify = strat_labels if rare_count >= 2 else None
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed,
        stratify=use_stratify,
    )
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # TensorDatasets: nf, nm, ct, clin, gene_indices, patient_edge_feats, bi, ci, ess, em
    train_ds = TensorDataset(
        node_features[train_idx], node_masks[train_idx],
        cancer_types[train_idx], clinical[train_idx],
        gene_indices_t[train_idx], patient_edge_feats_t[train_idx],
        block_ids_t[train_idx], channel_ids_t[train_idx],
        essentiality[train_idx], essentiality_masks[train_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    val_tensors = (
        node_features[val_idx], node_masks[val_idx],
        cancer_types[val_idx], clinical[val_idx],
        gene_indices_t[val_idx], patient_edge_feats_t[val_idx],
        block_ids_t[val_idx], channel_ids_t[val_idx],
        essentiality[val_idx], essentiality_masks[val_idx],
    )

    # Initialize model + head
    model = AtlasTransformerV6(config).to(device)
    head = EssentialityHead(config["hidden_dim"], dropout=args.dropout).to(device)
    loss_fn = EssentialityLoss(rank_weight=args.rank_weight)

    n_params_backbone = sum(p.numel() for p in model.parameters())
    n_params_head = sum(p.numel() for p in head.parameters())
    print(f"\n  Backbone params: {n_params_backbone:,}")
    print(f"  Head params: {n_params_head:,}")
    print(f"  Total: {n_params_backbone + n_params_head:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Training loop
    best_val_loss = float("inf")
    best_corr = 0.0
    best_state = None
    patience_counter = 0

    print(f"\nPre-training on DepMap essentiality...\n", flush=True)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, head, loss_fn, optimizer, train_loader,
            gene_pair_matrix_t, device, args,
        )
        val_metrics = eval_epoch(
            model, head, loss_fn, val_tensors,
            gene_pair_matrix_t, device,
        )
        scheduler.step()

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_corr = val_metrics["corr"]
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                "config": config,
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "val_corr": best_corr,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or improved or patience_counter >= args.patience:
            marker = " *" if improved else ""
            print(
                f"  Epoch {epoch+1:3d}: "
                f"train_mse={train_metrics['mse']:.4f} "
                f"val_mse={val_metrics['mse']:.4f} "
                f"val_corr={val_metrics['corr']:.3f} "
                f"best_corr={best_corr:.3f}{marker}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(f"\n  Early stop at epoch {epoch+1}", flush=True)
            break

    # Save
    checkpoint_path = os.path.join(results_dir, "pretrained_backbone.pt")
    torch.save(best_state, checkpoint_path)
    print(f"\nSaved pre-trained backbone: {checkpoint_path}", flush=True)

    backbone_path = os.path.join(results_dir, "backbone_weights.pt")
    torch.save(best_state["backbone"], backbone_path)
    print(f"Saved backbone weights: {backbone_path}", flush=True)

    results = {
        "model": "AtlasTransformerV6 + EssentialityHead",
        "task": "DepMap CRISPR essentiality pre-training",
        "n_cell_lines": N,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_epoch": best_state["epoch"],
        "best_val_loss": best_val_loss,
        "best_val_corr": best_corr,
        "config": {k: v for k, v in config.items()
                   if not isinstance(v, (np.ndarray, torch.Tensor))},
        "args": vars(args),
    }
    with open(os.path.join(results_dir, "pretrain_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # === Precipitate attention edges into graph ===
    print(f"\nExtracting attention edges from pre-trained model...", flush=True)
    precipitate_attention_edges(
        model, best_state["backbone"], config, device,
        node_features, node_masks, cancer_types, clinical,
        gene_indices_t, patient_edge_feats_t,
        block_ids_t, channel_ids_t, gene_pair_matrix_t,
        gene_names, gene_vocab, results_dir,
    )

    print(f"\n{'='*60}")
    print(f"  DEPMAP PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val correlation: {best_corr:.3f}")
    print(f"  Epochs: {best_state['epoch']}")
    print(f"  Checkpoint: {checkpoint_path}")


@torch.no_grad()
def precipitate_attention_edges(model, backbone_state, config, device,
                                 node_features, node_masks, cancer_types, clinical,
                                 gene_indices_t, patient_edge_feats_t,
                                 block_ids_t, channel_ids_t, gene_pair_matrix_t,
                                 gene_names, gene_vocab, results_dir):
    """Extract high-attention gene pairs and push to Neo4j as ATTENDS_TO edges.

    Runs the trained backbone over all cell lines, accumulates attention
    weights per gene pair, and commits the top edges to the graph with
    provenance attribution.
    """
    model.load_state_dict(backbone_state)
    model = model.to(device)
    model.eval()

    idx_to_gene = {i: g for g, i in gene_vocab.items()}
    N = len(node_features)

    # Accumulate attention per gene pair
    pair_attn = {}   # (gene_a, gene_b) → list of attention weights
    pair_count = {}  # (gene_a, gene_b) → number of co-occurrences

    BATCH = 64
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        sl = slice(start, end)

        nf = node_features[sl].to(device)
        nm = node_masks[sl].to(device)
        ct = cancer_types[sl].to(device)
        cl = clinical[sl].to(device)
        gi = gene_indices_t[sl].to(device)
        pef = patient_edge_feats_t[sl].to(device)
        bi = block_ids_t[sl].to(device)
        ci = channel_ids_t[sl].to(device)

        ef = gather_edge_features(gene_pair_matrix_t.to(device), pef, gi, nm)

        # Use encode_with_attention to get attention from the last intra-block layer
        _, _, attn_w = model.encode_with_attention(nf, nm, ct, cl, ef, bi, ci)

        # attn_w is (B, N, N) — average attention from last intra-block layer
        B_batch = nf.shape[0]
        attn_np = attn_w.cpu().numpy()

        for b_offset in range(B_batch):
            b_idx = start + b_offset
            genes = gene_names[b_idx]
            mask = node_masks[b_idx].numpy()

            for i in range(MAX_NODES):
                if mask[i] == 0 or not genes[i] or genes[i] == "":
                    continue
                for j in range(i + 1, MAX_NODES):
                    if mask[j] == 0 or not genes[j] or genes[j] == "":
                        continue

                    ga, gb = sorted([genes[i], genes[j]])
                    w = (attn_np[b_offset, i, j] + attn_np[b_offset, j, i]) / 2

                    key = (ga, gb)
                    if key not in pair_attn:
                        pair_attn[key] = []
                        pair_count[key] = 0
                    pair_attn[key].append(w)
                    pair_count[key] += 1

    # Compute mean attention per pair
    pair_mean = {}
    for key, weights in pair_attn.items():
        pair_mean[key] = float(np.mean(weights))

    # Threshold: top edges with sufficient observations
    MIN_OBS = 5
    MIN_ATTN = 0.1  # mean attention threshold

    candidates = []
    for (ga, gb), mean_w in sorted(pair_mean.items(), key=lambda x: -x[1]):
        if ga == gb:
            continue  # skip self-loops
        n_obs = pair_count[(ga, gb)]
        if n_obs >= MIN_OBS and mean_w >= MIN_ATTN:
            ch_a = CHANNEL_MAP.get(ga, "")
            ch_b = CHANNEL_MAP.get(gb, "")
            candidates.append({
                "from": ga,
                "to": gb,
                "weight": round(mean_w, 4),
                "n_obs": n_obs,
                "cross_channel": ch_a != ch_b,
            })

    print(f"  Gene pairs with attention data: {len(pair_mean)}")
    print(f"  Candidates (obs>={MIN_OBS}, attn>={MIN_ATTN}): {len(candidates)}")

    if candidates:
        # Show top 10
        print(f"\n  Top attention edges:")
        for c in candidates[:10]:
            cross = " [cross-channel]" if c["cross_channel"] else ""
            print(f"    {c['from']:>8s} — {c['to']:<8s}  "
                  f"attn={c['weight']:.3f}  n={c['n_obs']}{cross}")

        # Save to results
        edges_path = os.path.join(results_dir, "attention_edges.json")
        with open(edges_path, "w") as f:
            json.dump(candidates, f, indent=2)
        print(f"\n  Saved {len(candidates)} edges to {edges_path}")

        # Commit to Neo4j
        try:
            from gnn.data.graph_changelog import GraphGateway
            gw = GraphGateway()
            try:
                before = gw.count_edges("ATTENDS_TO")
                edges = [{
                    "from": c["from"],
                    "to": c["to"],
                    "weight": c["weight"],
                    "n_obs": c["n_obs"],
                    "cross_channel": c["cross_channel"],
                    "cycle": "depmap_pretrain",
                } for c in candidates]

                n = gw.merge_edges(
                    "ATTENDS_TO", edges,
                    source="depmap_pretrain",
                    source_detail=f"{len(edges)} edges from DepMap essentiality pre-training",
                )
                after = gw.count_edges("ATTENDS_TO")
                print(f"  ATTENDS_TO edges: {before} → {after} (delta: {after - before:+d})")

                # Invalidate caches
                from gnn.data.graph_schema import GraphSchema
                schema = GraphSchema()
                schema.invalidate_cache()
                print(f"  Schema cache invalidated (graph changed)")
            finally:
                gw.close()
        except Exception as e:
            print(f"  WARNING: Could not commit edges to Neo4j: {e}")
            print(f"  Edges saved to {edges_path} for manual import")
    else:
        print("  No edges met threshold for precipitation")


if __name__ == "__main__":
    main()
