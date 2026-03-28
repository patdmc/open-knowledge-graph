#!/usr/bin/env python3
"""
Train AtlasTransformer V2 — context-conditioned attention.

Cancer type conditions edges (attention), age conditions nodes.
Same dataset as V1, different model.

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v2
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, NODE_FEAT_DIM, MAX_NODES
from gnn.models.atlas_transformer_v2 import AtlasTransformerV2
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS

from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdback', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.3)
    return parser.parse_args()


def train_fold(model, train_loader, val_data, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    criterion = CoxPartialLikelihoodLoss()

    best_c = 0.0
    best_state = None
    patience_counter = 0

    val_nf, val_nm, val_cp, val_ct, val_clin, val_atlas, val_t, val_e = val_data

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            nf, nm, cp, ct, clin, atlas, t, e = [b.to(device) for b in batch]
            optimizer.zero_grad()
            hazard = model(nf, nm, cp, ct, clin, atlas)
            loss = criterion(hazard.squeeze(), t, e)
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
            ).squeeze().cpu().numpy()

        c_idx = concordance_index(
            val_t.numpy(), val_e.numpy(), val_hazard
        )

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
            print(f"    Early stop epoch {epoch+1}, C-index: {best_c:.4f}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_c, best_state, epoch + 1


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    ds = AtlasDataset()
    data = ds.build_features()

    node_features = data['node_features']
    node_masks = data['node_masks']
    channel_pos_ids = data['channel_pos_ids']
    atlas_sums = data['atlas_sums']
    times = data['times']
    events = data['events']
    cancer_types = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    n_cancer_types = data['n_cancer_types']

    clinical = torch.stack([ages, sexes], dim=1)

    config = {
        'node_feat_dim': NODE_FEAT_DIM,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'n_folds': args.n_folds,
        'random_seed': args.seed,
        'n_cancer_types': n_cancer_types,
        'max_nodes': MAX_NODES,
    }

    # Holdback
    n_total = len(events)
    np.random.seed(args.seed)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)

    n_holdback = int(n_total * args.holdback)
    holdback_idx = all_idx[:n_holdback]
    train_pool_idx = all_idx[n_holdback:]

    hb_nf = node_features[holdback_idx]
    hb_nm = node_masks[holdback_idx]
    hb_cp = channel_pos_ids[holdback_idx]
    hb_ct = cancer_types[holdback_idx]
    hb_clin = clinical[holdback_idx]
    hb_atlas = atlas_sums[holdback_idx]
    hb_times = times[holdback_idx]
    hb_events = events[holdback_idx]

    node_features = node_features[train_pool_idx]
    node_masks = node_masks[train_pool_idx]
    channel_pos_ids = channel_pos_ids[train_pool_idx]
    atlas_sums = atlas_sums[train_pool_idx]
    times = times[train_pool_idx]
    events = events[train_pool_idx]
    cancer_types = cancer_types[train_pool_idx]
    clinical = clinical[train_pool_idx]

    print(f"\nHoldback: {n_holdback}, CV pool: {len(events)}", flush=True)

    tag = "atlas_transformer_v2"
    results_dir = os.path.join(GNN_RESULTS, tag)
    os.makedirs(results_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                           random_state=args.seed)

    fold_results = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.arange(len(events)), events.numpy())):

        print(f"\n=== Fold {fold} ===", flush=True)

        train_ds = TensorDataset(
            node_features[train_idx], node_masks[train_idx],
            channel_pos_ids[train_idx], cancer_types[train_idx],
            clinical[train_idx], atlas_sums[train_idx],
            times[train_idx], events[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

        model = AtlasTransformerV2(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if fold == 0:
            print(f"  Parameters: {n_params:,}", flush=True)

        val_data = (
            node_features[val_idx], node_masks[val_idx],
            channel_pos_ids[val_idx], cancer_types[val_idx],
            clinical[val_idx], atlas_sums[val_idx],
            times[val_idx], events[val_idx],
        )

        best_c, best_state, n_epochs = train_fold(
            model, train_loader, val_data, config, device,
        )

        fold_result = {'fold': fold, 'c_index': best_c, 'epochs': n_epochs}
        fold_results.append(fold_result)
        print(f"  Fold {fold}: C-index = {best_c:.4f}", flush=True)

        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

    elapsed = time.time() - t0
    c_indices = [r['c_index'] for r in fold_results]

    # Holdback evaluation
    print(f"\n{'='*50}", flush=True)
    print("Holdback evaluation...", flush=True)

    holdback_hazards = []
    for fr in fold_results:
        fold_dir = os.path.join(results_dir, f"fold_{fr['fold']}")
        state = torch.load(os.path.join(fold_dir, "best_model.pt"),
                           map_location=device, weights_only=True)
        model = AtlasTransformerV2(config).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            h = model(
                hb_nf.to(device), hb_nm.to(device), hb_cp.to(device),
                hb_ct.to(device), hb_clin.to(device), hb_atlas.to(device),
            ).squeeze().cpu().numpy()
        holdback_hazards.append(h)

    ensemble_hazard = np.mean(holdback_hazards, axis=0)
    holdback_c = concordance_index(
        hb_times.numpy(), hb_events.numpy(), ensemble_hazard
    )

    per_fold_hb = [concordance_index(hb_times.numpy(), hb_events.numpy(), h)
                   for h in holdback_hazards]

    cv_mean = np.mean(c_indices)
    overfit_gap = cv_mean - holdback_c

    # Time-dependent AUC on holdback
    td_auc = time_dependent_auc(ensemble_hazard, hb_times.numpy(),
                                 hb_events.numpy(), [12, 36, 60])

    print(f"\n{'='*50}", flush=True)
    print(f"ATLAS TRANSFORMER V2 RESULTS", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"CV Mean: {cv_mean:.4f} ± {np.std(c_indices):.4f}", flush=True)
    print(f"Per-fold: {[f'{c:.4f}' for c in c_indices]}", flush=True)
    print(f"Holdback ensemble: {holdback_c:.4f}", flush=True)
    print(f"Holdback per-fold: {[f'{c:.4f}' for c in per_fold_hb]}", flush=True)
    print(f"Overfit gap: {overfit_gap:+.4f} ({'OK' if abs(overfit_gap) < 0.02 else 'WARNING'})", flush=True)
    print(f"TD-AUC: {td_auc}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)
    print(f"", flush=True)
    print(f"BASELINES:", flush=True)
    print(f"  Atlas lookup (zero-param):    C = 0.5772", flush=True)
    print(f"  AtlasTransformer V1:          C = 0.6733 / holdback 0.6657", flush=True)
    print(f"  ChannelNetV5 (12-node):       C = 0.6770", flush=True)
    print(f"  ChannelNetV7 (hotspot):       C = 0.6733 / holdback 0.6657", flush=True)

    results = {
        'model': 'AtlasTransformerV2',
        'config': config,
        'mean_c_index': cv_mean,
        'std_c_index': float(np.std(c_indices)),
        'fold_results': fold_results,
        'holdback': {
            'n_patients': n_holdback,
            'ensemble_c_index': holdback_c,
            'per_fold_c_indices': per_fold_hb,
            'overfit_gap': overfit_gap,
            'td_auc': td_auc,
        },
        'n_patients_cv': len(events),
        'n_patients_total': n_total,
        'elapsed_seconds': elapsed,
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
