#!/usr/bin/env python3
"""
Train AtlasTransformer V5 — hierarchical block-sparse, edge-informed attention.

The graph IS the model:
  - Block structure from PPI community detection
  - Edge features from ALL edge types bias attention directly
  - Node features = atlas + dynamic graph node properties
  - Channel pooling → cross-channel attention → readout

Uses current graph topology (8 channels, 509 genes, dynamic edge types).

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v5
    python3 -u -m gnn.scripts.train_atlas_transformer_v5 --epochs 50 --batch-size 128
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES
from gnn.data.graph_schema import get_schema
from gnn.models.atlas_transformer_v5 import AtlasTransformerV5
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-block-layers', type=int, default=2)
    parser.add_argument('--n-cross-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n-folds', type=int, default=5)
    return parser.parse_args()


def train_fold(model, train_loader, val_data, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    cox_loss = CoxPartialLikelihoodLoss()

    best_c = 0.0
    best_state = None
    patience_counter = 0

    (val_nf, val_nm, val_cp, val_ct, val_clin, val_atlas,
     val_ef, val_bi, val_ci, val_t, val_e) = val_data

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            (nf, nm, cp, ct, clin, atlas, ef, bi, ci, t, e) = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            hazard = model(nf, nm, cp, ct, clin, atlas, ef, bi, ci).squeeze()

            loss = cox_loss(hazard, t, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_hazard = model(
                val_nf.to(device), val_nm.to(device), val_cp.to(device),
                val_ct.to(device), val_clin.to(device), val_atlas.to(device),
                val_ef.to(device), val_bi.to(device), val_ci.to(device),
            ).squeeze().cpu().numpy()

        # Guard against NaN in output
        if np.isnan(val_hazard).any():
            n_nan = np.isnan(val_hazard).sum()
            print(f"    WARNING: {n_nan} NaN values in val output, replacing with 0", flush=True)
            val_hazard = np.nan_to_num(val_hazard, nan=0.0)

        c_idx = concordance_index(val_t.numpy(), val_e.numpy(), val_hazard)

        if c_idx > best_c:
            best_c = c_idx
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss/max(n_batches,1):.4f} "
                  f"C={c_idx:.4f} best={best_c:.4f}", flush=True)

        if patience_counter >= config['patience']:
            print(f"    Early stop epoch {epoch+1}, best C={best_c:.4f}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_c, best_state, epoch + 1


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # === Discover schema from graph ===
    print("\nDiscovering graph schema...", flush=True)
    schema = get_schema(force_refresh=True)
    print(f"  Edge types: {len(schema.edge_types)}, Edge feature dim: {schema.edge_feature_dim}")
    print(f"  Node extra dim: {schema.node_extra_dim}")
    print(f"  Channels: {schema.n_channels}, Blocks: {schema.n_blocks}")

    # === Build dataset with V5 features ===
    ds = AtlasDataset()
    data = ds.build_v5_features(schema=schema)

    node_features = data['node_features']
    node_masks = data['node_masks']
    channel_pos_ids = data['channel_pos_ids']
    atlas_sums = data['atlas_sums']
    times = data['times']
    events = data['events']
    cancer_types = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    edge_features = data['edge_features']
    block_ids = data['block_ids']
    channel_ids = data['channel_ids']
    n_cancer_types = data['n_cancer_types']

    clinical = torch.stack([ages, sexes], dim=-1)
    N = len(events)

    print(f"\nDataset: {N} patients")
    print(f"  Node features: {node_features.shape}")
    print(f"  Edge features: {edge_features.shape}")
    print(f"  Block IDs range: [{block_ids.min()}, {block_ids.max()}]")
    print(f"  Channel IDs range: [{channel_ids.min()}, {channel_ids.max()}]")
    print(f"  Cancer types: {n_cancer_types}")

    # === Config from schema ===
    config = AtlasTransformerV5.config_from_schema(schema, n_cancer_types)
    config.update({
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_block_layers': args.n_block_layers,
        'n_cross_layers': args.n_cross_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'random_seed': args.seed,
        'n_folds': args.n_folds,
        'max_nodes': MAX_NODES,
    })

    print(f"\nConfig: {json.dumps({k: v for k, v in config.items()}, indent=2)}")

    tag = "atlas_transformer_v5"
    results_dir = os.path.join(GNN_RESULTS, tag)
    os.makedirs(results_dir, exist_ok=True)

    # === 5-fold CV ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events.numpy())):
        print(f"\n{'='*60}", flush=True)
        print(f"  FOLD {fold} — Train: {len(train_idx)}, Val: {len(val_idx)}", flush=True)
        print(f"{'='*60}", flush=True)

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)

        train_ds = TensorDataset(
            node_features[train_idx], node_masks[train_idx],
            channel_pos_ids[train_idx], cancer_types[train_idx],
            clinical[train_idx], atlas_sums[train_idx],
            edge_features[train_idx], block_ids[train_idx],
            channel_ids[train_idx],
            times[train_idx], events[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

        model = AtlasTransformerV5(config).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}", flush=True)

        val_data = (
            node_features[val_idx], node_masks[val_idx],
            channel_pos_ids[val_idx], cancer_types[val_idx],
            clinical[val_idx], atlas_sums[val_idx],
            edge_features[val_idx], block_ids[val_idx],
            channel_ids[val_idx],
            times[val_idx], events[val_idx],
        )

        best_c, best_state, n_epochs = train_fold(
            model, train_loader, val_data, config, device,
        )

        # Save fold model
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

        # Compute TD-AUC on val
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            val_hazard = model(
                node_features[val_idx].to(device), node_masks[val_idx].to(device),
                channel_pos_ids[val_idx].to(device), cancer_types[val_idx].to(device),
                clinical[val_idx].to(device), atlas_sums[val_idx].to(device),
                edge_features[val_idx].to(device), block_ids[val_idx].to(device),
                channel_ids[val_idx].to(device),
            ).squeeze().cpu().numpy()

        td_auc = time_dependent_auc(val_hazard, times[val_idx].numpy(),
                                     events[val_idx].numpy(), [12, 36, 60])

        fold_results.append({
            'fold': fold,
            'c_index': best_c,
            'n_epochs': n_epochs,
            'td_auc': td_auc,
        })

        print(f"  Fold {fold}: C={best_c:.4f}, TD-AUC={td_auc}, epochs={n_epochs}")

    # === Summary ===
    cs = [r['c_index'] for r in fold_results]
    mean_c = float(np.mean(cs))
    std_c = float(np.std(cs))

    print(f"\n{'='*60}", flush=True)
    print(f"  ATLAS TRANSFORMER V5 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Mean C-index: {mean_c:.4f} ± {std_c:.4f}", flush=True)
    for r in fold_results:
        print(f"    Fold {r['fold']}: C={r['c_index']:.4f} ({r['n_epochs']} epochs)", flush=True)
    print(f"", flush=True)
    print(f"  BASELINES:", flush=True)
    print(f"    Atlas lookup (zero-param):    C ≈ 0.577", flush=True)
    print(f"    Graph scorer (hierarchical):  C ≈ 0.598", flush=True)
    print(f"    AtlasTransformer V3:          C ≈ 0.666", flush=True)
    print(f"    ChannelNetV5 (hub/leaf):      C ≈ 0.677", flush=True)

    results = {
        'model': 'AtlasTransformerV5',
        'config': {k: v for k, v in config.items()
                   if not isinstance(v, (np.ndarray, torch.Tensor))},
        'mean_c_index': mean_c,
        'std_c_index': std_c,
        'fold_results': fold_results,
        'n_patients': N,
        'n_cancer_types': n_cancer_types,
        'schema': {
            'edge_types': schema.edge_types,
            'edge_feature_dim': schema.edge_feature_dim,
            'node_extra_dim': schema.node_extra_dim,
            'n_channels': schema.n_channels,
            'n_blocks': schema.n_blocks,
        },
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
