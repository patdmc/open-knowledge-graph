#!/usr/bin/env python3
"""
Train AtlasTransformer V5b — factored attention.

Gene-gene attention is computed ONCE on the full gene set.
Patient scoring gathers the relevant gene embeddings and interactions.

This is orders of magnitude faster than V5 because:
  - Gene attention: (509, 509) computed once per forward pass
  - Patient scoring: just index gathering, no per-patient attention
  - Memory: no (B, 32, 32, 24) edge feature tensors

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v5b
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES, NODE_FEAT_DIM
from gnn.data.graph_schema import get_schema
from gnn.models.atlas_transformer_v5b import AtlasTransformerV5b
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS, ALL_GENES, GENE_TO_IDX, CHANNEL_MAP, CHANNEL_NAMES


def build_gene_level_features(schema):
    """Build the global gene feature matrix and edge feature matrix.

    Returns:
        gene_node_features: (G, node_feat_dim + extra_dim) — all genes
        gene_edge_features: (G, G, edge_feature_dim) — all pairwise
        gene_block_ids: (G,) long — sub-pathway block per gene
    """
    G = len(ALL_GENES)
    node_feat_dim = NODE_FEAT_DIM + schema.node_extra_dim

    # Gene node features: base atlas features (zeros for now, filled per-gene)
    # + dynamic graph node properties
    gene_node_features = np.zeros((G, node_feat_dim), dtype=np.float32)

    for i, gene in enumerate(ALL_GENES):
        # Channel one-hot (features 4-11 in the 14-dim base features)
        ch = CHANNEL_MAP.get(gene)
        if ch and ch in CHANNEL_NAMES:
            gene_node_features[i, 4 + CHANNEL_NAMES.index(ch)] = 1.0

        # Hub status (feature 3)
        from gnn.config import HUB_GENES
        for hubs in HUB_GENES.values():
            if gene in hubs:
                gene_node_features[i, 3] = 1.0
                break

        # Extra graph node properties
        extra = schema.get_node_extra_features(gene)
        gene_node_features[i, NODE_FEAT_DIM:] = extra

    # Gene pairwise edge features
    gene_edge_features = np.zeros((G, G, schema.edge_feature_dim), dtype=np.float32)
    for i, gi in enumerate(ALL_GENES):
        for j, gj in enumerate(ALL_GENES):
            if i != j:
                feat = schema.get_edge_feature_vector(gi, gj)
                gene_edge_features[i, j] = feat

    # NaN guard
    gene_node_features = np.nan_to_num(gene_node_features, nan=0.0)
    gene_edge_features = np.nan_to_num(gene_edge_features, nan=0.0)

    # Block IDs
    gene_block_ids = np.zeros(G, dtype=np.int64)
    for i, gene in enumerate(ALL_GENES):
        bid = schema.get_block_id(gene)
        gene_block_ids[i] = bid if bid >= 0 else schema.n_blocks

    print(f"Gene-level features: {G} genes, {node_feat_dim} node dims, "
          f"{schema.edge_feature_dim} edge dims")
    n_edges = (gene_edge_features.sum(axis=-1) != 0).sum()
    print(f"  Non-zero gene pairs: {n_edges:,} / {G*G:,}")

    return (torch.tensor(gene_node_features),
            torch.tensor(gene_edge_features),
            torch.tensor(gene_block_ids))


def build_patient_indices(data):
    """Convert per-patient gene names to gene vocabulary indices.

    Returns:
        gene_indices: (N, MAX_NODES) long — index into ALL_GENES
        gene_masks: same as data['node_masks']
        channel_ids: (N, MAX_NODES) long
    """
    gene_names = data['gene_names']
    N = len(gene_names)

    gene_indices = np.zeros((N, MAX_NODES), dtype=np.int64)
    channel_ids = np.zeros((N, MAX_NODES), dtype=np.int64)
    n_channels = len(CHANNEL_NAMES)

    for i in range(N):
        for j in range(MAX_NODES):
            gene = gene_names[i][j]
            if gene and gene != '' and gene != 'WT':
                idx = GENE_TO_IDX.get(gene, 0)
                gene_indices[i, j] = idx

                ch = CHANNEL_MAP.get(gene)
                if ch and ch in CHANNEL_NAMES:
                    channel_ids[i, j] = CHANNEL_NAMES.index(ch)
                else:
                    channel_ids[i, j] = n_channels  # unassigned
            else:
                channel_ids[i, j] = n_channels

    return torch.tensor(gene_indices), torch.tensor(channel_ids)


def train_fold(model, gene_data, train_loader, val_data, config, device):
    """Train one fold. Gene encoding runs once per forward pass."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])
    cox_loss = CoxPartialLikelihoodLoss()

    gene_nf, gene_ef, gene_bi = [g.to(device) for g in gene_data]

    best_c = 0.0
    best_state = None
    patience_counter = 0

    (val_gi, val_gm, val_ci, val_ct, val_clin, val_atlas, val_nf, val_t, val_e) = val_data

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            (gi, gm, ci, ct, clin, atlas, nf, t, e) = [b.to(device) for b in batch]
            optimizer.zero_grad()

            # Encode genes each batch so gradients flow to gene encoder
            model.encode_genes(gene_nf, gene_ef, gene_bi)
            hazard = model(gi, gm, ci, ct, clin, atlas, node_features=nf).squeeze()
            loss = cox_loss(hazard, t, e)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            model.encode_genes(gene_nf, gene_ef, gene_bi)
            val_hazard = model(
                val_gi.to(device), val_gm.to(device), val_ci.to(device),
                val_ct.to(device), val_clin.to(device), val_atlas.to(device),
                node_features=val_nf.to(device),
            ).squeeze().cpu().numpy()

        val_hazard = np.nan_to_num(val_hazard, nan=0.0)
        c_idx = concordance_index(val_t.numpy(), val_e.numpy(), val_hazard)

        if c_idx > best_c:
            best_c = c_idx
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss/max(n_batches,1):.4f} "
                  f"C={c_idx:.4f} best={best_c:.4f}", flush=True)

        if patience_counter >= config['patience']:
            print(f"    Early stop epoch {epoch+1}, best C={best_c:.4f}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_c, best_state, epoch + 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-block-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--atlas-weight', type=float, default=0.1,
                        help='Weight for atlas skip (0=disabled, 1=full)')
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # === Schema ===
    schema = get_schema(force_refresh=True)
    print(f"Schema: {len(schema.edge_types)} edge types, "
          f"{schema.edge_feature_dim} edge features, "
          f"{schema.node_extra_dim} node extras")

    # === Gene-level features (computed ONCE) ===
    gene_nf, gene_ef, gene_bi = build_gene_level_features(schema)
    gene_data = (gene_nf, gene_ef, gene_bi)

    # === Patient data ===
    ds = AtlasDataset()
    data = ds.build_features()  # base features only — no V5 edge tensors needed

    gene_indices, channel_ids = build_patient_indices(data)
    gene_masks = data['node_masks']
    node_features = data['node_features']  # (N, MAX_NODES, 14) — per-patient atlas features
    cancer_types = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    clinical = torch.stack([ages, sexes], dim=-1)
    atlas_sums = data['atlas_sums']
    times = data['times']
    events = data['events']
    n_cancer_types = data['n_cancer_types']
    N = len(events)

    print(f"\nPatients: {N}")
    print(f"Gene vocabulary: {len(ALL_GENES)} genes")
    print(f"Patient tensors: gene_indices {gene_indices.shape}, "
          f"masks {gene_masks.shape} — NO per-patient edge features")

    # Memory comparison
    v5_edge_mem = N * MAX_NODES * MAX_NODES * schema.edge_feature_dim * 4 / 1e9
    v5b_edge_mem = len(ALL_GENES) ** 2 * schema.edge_feature_dim * 4 / 1e9
    print(f"\nMemory: V5 edge features = {v5_edge_mem:.1f} GB, "
          f"V5b gene edges = {v5b_edge_mem:.3f} GB "
          f"({v5_edge_mem/v5b_edge_mem:.0f}x reduction)")

    # === Config ===
    config = AtlasTransformerV5b.config_from_schema(schema, n_cancer_types)
    config.update({
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_block_layers': args.n_block_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'random_seed': args.seed,
        'n_folds': args.n_folds,
        'max_nodes': MAX_NODES,
        'patient_node_dim': NODE_FEAT_DIM,  # 14 — atlas per-mutation features
    })

    tag = "atlas_transformer_v5"  # same output dir — replaces V5
    results_dir = os.path.join(GNN_RESULTS, tag)
    os.makedirs(results_dir, exist_ok=True)

    # === Cross-validation ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_folds = max(args.n_folds, 2)  # StratifiedKFold requires >= 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    max_folds = args.n_folds  # may run fewer than n_splits
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events.numpy())):
        if fold >= max_folds:
            break
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} — Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*60}", flush=True)

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)

        train_ds = TensorDataset(
            gene_indices[train_idx], gene_masks[train_idx],
            channel_ids[train_idx], cancer_types[train_idx],
            clinical[train_idx], atlas_sums[train_idx],
            node_features[train_idx],
            times[train_idx], events[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

        model = AtlasTransformerV5b(config).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}", flush=True)

        val_data = (
            gene_indices[val_idx], gene_masks[val_idx],
            channel_ids[val_idx], cancer_types[val_idx],
            clinical[val_idx], atlas_sums[val_idx],
            node_features[val_idx],
            times[val_idx], events[val_idx],
        )

        best_c, best_state, n_epochs = train_fold(
            model, gene_data, train_loader, val_data, config, device)

        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

        # TD-AUC
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            model.encode_genes(*[g.to(device) for g in gene_data])
            val_hazard = model(
                gene_indices[val_idx].to(device), gene_masks[val_idx].to(device),
                channel_ids[val_idx].to(device), cancer_types[val_idx].to(device),
                clinical[val_idx].to(device), atlas_sums[val_idx].to(device),
                node_features=node_features[val_idx].to(device),
            ).squeeze().cpu().numpy()

        val_hazard = np.nan_to_num(val_hazard, nan=0.0)
        td_auc = time_dependent_auc(val_hazard, times[val_idx].numpy(),
                                     events[val_idx].numpy(), [12, 36, 60])

        fold_results.append({
            'fold': fold, 'c_index': best_c, 'n_epochs': n_epochs, 'td_auc': td_auc,
        })
        print(f"  Fold {fold}: C={best_c:.4f}, TD-AUC={td_auc}, epochs={n_epochs}")

    # === Summary ===
    cs = [r['c_index'] for r in fold_results]
    mean_c = float(np.mean(cs))
    std_c = float(np.std(cs))

    print(f"\n{'='*60}")
    print(f"  ATLAS TRANSFORMER V5b RESULTS")
    print(f"{'='*60}")
    print(f"  Mean C-index: {mean_c:.4f} ± {std_c:.4f}")
    for r in fold_results:
        print(f"    Fold {r['fold']}: C={r['c_index']:.4f} ({r['n_epochs']} epochs)")
    print(f"\n  BASELINES:")
    print(f"    Graph scorer (hierarchical):  C ≈ 0.598")
    print(f"    AtlasTransformer V3:          C ≈ 0.666")
    print(f"    ChannelNetV5 (hub/leaf):      C ≈ 0.677")

    results = {
        'model': 'AtlasTransformerV5b',
        'config': {k: v for k, v in config.items()
                   if not isinstance(v, (np.ndarray, torch.Tensor))},
        'mean_c_index': mean_c,
        'std_c_index': std_c,
        'fold_results': fold_results,
        'n_patients': N,
        'n_cancer_types': n_cancer_types,
        'n_genes': len(ALL_GENES),
        'schema': {
            'edge_types': schema.edge_types,
            'edge_feature_dim': schema.edge_feature_dim,
            'node_extra_dim': schema.node_extra_dim,
            'n_channels': schema.n_channels,
            'n_blocks': schema.n_blocks,
        },
    }

    with open(os.path.join(results_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
