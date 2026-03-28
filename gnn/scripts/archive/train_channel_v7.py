#!/usr/bin/env python3
"""
Train ChannelNetV7 — hotspot mutation encoding.

Usage:
    python -m gnn.scripts.train_channel_v7
    python -m gnn.scripts.train_channel_v7 --with-tmb    # add TMB covariate
    python -m gnn.scripts.train_channel_v7 --channels 8  # 8 channels (with epigenetic)
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

from gnn.data.channel_dataset_v7 import ChannelDatasetV7, BASE_FEAT_DIM
from gnn.models.channel_net_v7 import ChannelNetV7
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS

from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-tmb', action='store_true',
                        help='Include TMB as clinical covariate')
    parser.add_argument('--channels', type=int, default=6,
                        help='Number of channels (6 or 8)')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-hotspot', type=int, default=50)
    parser.add_argument('--max-hotspots-per-node', type=int, default=30)
    parser.add_argument('--holdback', type=float, default=0.15,
                        help='Fraction of data to hold back as test set')
    return parser.parse_args()


def train_fold(model, train_loader, val_features, val_ct, val_clinical,
               val_times, val_events, config, device):
    """Train one fold, return best val C-index and model state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    criterion = CoxPartialLikelihoodLoss()

    best_c = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            feat, ct, clin, t, e = [b.to(device) for b in batch]

            optimizer.zero_grad()
            hazard = model(feat, ct, clin)
            loss = criterion(hazard.squeeze(), t, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_hazard = model(
                val_features.to(device),
                val_ct.to(device),
                val_clinical.to(device),
            ).squeeze().cpu().numpy()

        c_index = concordance_index(
            val_times.numpy(), val_events.numpy(), val_hazard
        )

        if c_index > best_c:
            best_c = c_index
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"    Early stop at epoch {epoch+1}, best C-index: {best_c:.4f}")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_c, best_state, epoch + 1


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build dataset
    ds = ChannelDatasetV7(
        min_hotspot_count=args.min_hotspot,
        max_hotspots_per_node=args.max_hotspots_per_node,
    )
    data = ds.build_features()
    ds.save_hotspot_map()

    features = data['features']
    times = data['times']
    events = data['events']
    cancer_types = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    n_cancer_types = data['n_cancer_types']
    feat_dim = data['feat_dim']

    # Clinical covariates
    if args.with_tmb:
        # TMB = total non-silent mutations per patient
        # Already encoded as feature[node, 0] summed across nodes
        tmb = features[:, :, 0].sum(dim=1)  # total mutations across all nodes
        tmb = (tmb - tmb.mean()) / (tmb.std() + 1e-8)
        clinical = torch.stack([ages, sexes, tmb], dim=1)
        clinical_dim = 3
    else:
        clinical = torch.stack([ages, sexes], dim=1)
        clinical_dim = 2

    config = {
        'feat_dim': feat_dim,
        'base_feat_dim': BASE_FEAT_DIM,
        'hidden_dim': args.hidden_dim,
        'cross_channel_heads': 4,
        'cross_channel_layers': 2,
        'dropout': 0.3,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'n_folds': args.n_folds,
        'random_seed': args.seed,
        'n_cancer_types': n_cancer_types,
        'clinical_dim': clinical_dim,
        'with_tmb': args.with_tmb,
        'n_channels': args.channels,
        'min_hotspot_count': args.min_hotspot,
        'max_hotspots_per_node': args.max_hotspots_per_node,
        'holdback_fraction': args.holdback,
    }

    # --- Holdback test set (never seen during CV) ---
    n_total = len(events)
    np.random.seed(args.seed)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)

    n_holdback = int(n_total * args.holdback)
    holdback_idx = all_idx[:n_holdback]
    train_pool_idx = all_idx[n_holdback:]

    holdback_feat = features[holdback_idx]
    holdback_ct = cancer_types[holdback_idx]
    holdback_clin = clinical[holdback_idx]
    holdback_times = times[holdback_idx]
    holdback_events = events[holdback_idx]

    # Restrict CV pool
    features = features[train_pool_idx]
    times = times[train_pool_idx]
    events = events[train_pool_idx]
    cancer_types = cancer_types[train_pool_idx]
    clinical = clinical[train_pool_idx]

    print(f"\nHoldback test set: {n_holdback} patients ({args.holdback:.0%})")
    print(f"CV training pool: {len(events)} patients")

    # Results directory
    tag = f"channelnet_v7{'_tmb' if args.with_tmb else ''}"
    results_dir = os.path.join(GNN_RESULTS, tag)
    os.makedirs(results_dir, exist_ok=True)

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                           random_state=args.seed)

    fold_results = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.arange(len(events)), events.numpy())):

        print(f"\n=== Fold {fold} ===")

        train_feat = features[train_idx]
        train_ct = cancer_types[train_idx]
        train_clin = clinical[train_idx]
        train_times = times[train_idx]
        train_events = events[train_idx]

        val_feat = features[val_idx]
        val_ct = cancer_types[val_idx]
        val_clin = clinical[val_idx]
        val_times = times[val_idx]
        val_events = events[val_idx]

        train_ds = TensorDataset(train_feat, train_ct, train_clin,
                                  train_times, train_events)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

        model = ChannelNetV7(config).to(device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        best_c, best_state, n_epochs = train_fold(
            model, train_loader, val_feat, val_ct, val_clin,
            val_times, val_events, config, device
        )

        # Time-dependent AUC
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_hazard = model(
                val_feat.to(device), val_ct.to(device), val_clin.to(device)
            ).squeeze().cpu().numpy()

        td_auc = {}
        for t_point in [12, 36, 60]:
            try:
                auc = time_dependent_auc(
                    val_times.numpy(), val_events.numpy(),
                    val_hazard, t_point
                )
                td_auc[str(t_point)] = auc
            except Exception:
                td_auc[str(t_point)] = None

        fold_result = {
            'fold': fold,
            'c_index': best_c,
            'td_auc': td_auc,
            'epochs': n_epochs,
            'best_c_index': best_c,
        }
        fold_results.append(fold_result)
        print(f"  Fold {fold}: C-index = {best_c:.4f}, "
              f"AUC@36m = {td_auc.get('36', 'N/A')}")

        # Save model
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

    elapsed = time.time() - t0
    c_indices = [r['c_index'] for r in fold_results]

    # --- Holdback evaluation (ensemble of all fold models) ---
    print(f"\n{'='*50}")
    print("Evaluating on holdback test set...")

    holdback_hazards = []
    for fold_result in fold_results:
        fold_dir = os.path.join(results_dir, f"fold_{fold_result['fold']}")
        state = torch.load(os.path.join(fold_dir, "best_model.pt"),
                           map_location=device, weights_only=True)
        model = ChannelNetV7(config).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            h = model(
                holdback_feat.to(device),
                holdback_ct.to(device),
                holdback_clin.to(device),
            ).squeeze().cpu().numpy()
        holdback_hazards.append(h)

    # Ensemble: average hazard across folds
    ensemble_hazard = np.mean(holdback_hazards, axis=0)
    holdback_c = concordance_index(
        holdback_times.numpy(), holdback_events.numpy(), ensemble_hazard
    )

    # Also per-fold holdback C-index
    per_fold_holdback = []
    for h in holdback_hazards:
        c = concordance_index(
            holdback_times.numpy(), holdback_events.numpy(), h
        )
        per_fold_holdback.append(c)

    holdback_td_auc = {}
    for t_point in [12, 36, 60]:
        try:
            auc = time_dependent_auc(
                holdback_times.numpy(), holdback_events.numpy(),
                ensemble_hazard, t_point
            )
            holdback_td_auc[str(t_point)] = auc
        except Exception:
            holdback_td_auc[str(t_point)] = None

    print(f"Holdback ensemble C-index: {holdback_c:.4f}")
    print(f"Holdback per-fold C-indices: {[f'{c:.4f}' for c in per_fold_holdback]}")
    print(f"Holdback AUC@36m: {holdback_td_auc.get('36', 'N/A')}")

    # Overfit check
    cv_mean = np.mean(c_indices)
    overfit_gap = cv_mean - holdback_c
    print(f"\nCV mean: {cv_mean:.4f}")
    print(f"Holdback: {holdback_c:.4f}")
    print(f"Overfit gap: {overfit_gap:+.4f} ({'OK' if abs(overfit_gap) < 0.01 else 'WARNING'})")

    results = {
        'model': f'ChannelNetV7{"_TMB" if args.with_tmb else ""}',
        'config': config,
        'mean_c_index': np.mean(c_indices),
        'std_c_index': np.std(c_indices),
        'fold_results': fold_results,
        'holdback': {
            'n_patients': n_holdback,
            'fraction': args.holdback,
            'ensemble_c_index': holdback_c,
            'per_fold_c_indices': per_fold_holdback,
            'td_auc': holdback_td_auc,
            'overfit_gap': overfit_gap,
        },
        'n_patients_cv': len(events),
        'n_patients_total': n_total,
        'elapsed_seconds': elapsed,
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"CV Mean C-index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
    print(f"Holdback C-index: {holdback_c:.4f}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
