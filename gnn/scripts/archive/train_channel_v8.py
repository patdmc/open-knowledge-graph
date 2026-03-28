#!/usr/bin/env python3
"""
Train ChannelNetV8 — interaction-aware model.

Usage:
    python -u -m gnn.scripts.train_channel_v8
    python -u -m gnn.scripts.train_channel_v8 --with-tmb
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

from gnn.data.channel_dataset_v8 import ChannelDatasetV8, BASE_FEAT_DIM, N_INTERACTIONS
from gnn.models.channel_net_v8 import ChannelNetV8
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS

from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-tmb', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdback', type=float, default=0.15)
    return parser.parse_args()


def train_fold(model, train_loader, val_feat, val_ct, val_clin,
               val_interact, val_iw, val_times, val_events, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    criterion = CoxPartialLikelihoodLoss()
    iw = config['interaction_weights'].to(device)

    best_c = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            feat, ct, clin, interact, t, e = [b.to(device) for b in batch]
            optimizer.zero_grad()
            hazard = model(feat, ct, clin, interact, iw)
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
                val_feat.to(device), val_ct.to(device), val_clin.to(device),
                val_interact.to(device), iw,
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

    ds = ChannelDatasetV8()
    data = ds.build_features()

    features = data['features']
    interactions = data['interactions']
    interaction_weights = data['interaction_weights']
    times = data['times']
    events = data['events']
    cancer_types = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    n_cancer_types = data['n_cancer_types']
    feat_dim = data['feat_dim']

    if args.with_tmb:
        tmb = features[:, :, 0].sum(dim=1)
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
        'n_interactions': N_INTERACTIONS,
        'interaction_weights': interaction_weights,
        'with_tmb': args.with_tmb,
    }

    # Holdback
    n_total = len(events)
    np.random.seed(args.seed)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)

    n_holdback = int(n_total * args.holdback)
    holdback_idx = all_idx[:n_holdback]
    train_pool_idx = all_idx[n_holdback:]

    hb_feat = features[holdback_idx]
    hb_ct = cancer_types[holdback_idx]
    hb_clin = clinical[holdback_idx]
    hb_interact = interactions[holdback_idx]
    hb_times = times[holdback_idx]
    hb_events = events[holdback_idx]

    features = features[train_pool_idx]
    times = times[train_pool_idx]
    events = events[train_pool_idx]
    cancer_types = cancer_types[train_pool_idx]
    clinical = clinical[train_pool_idx]
    interactions = interactions[train_pool_idx]

    print(f"\nHoldback: {n_holdback}, CV pool: {len(events)}", flush=True)

    tag = f"channelnet_v8{'_tmb' if args.with_tmb else ''}"
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
            features[train_idx], cancer_types[train_idx], clinical[train_idx],
            interactions[train_idx], times[train_idx], events[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

        model = ChannelNetV8(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}", flush=True)

        best_c, best_state, n_epochs = train_fold(
            model, train_loader,
            features[val_idx], cancer_types[val_idx], clinical[val_idx],
            interactions[val_idx], interaction_weights,
            times[val_idx], events[val_idx],
            config, device,
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

    iw_device = interaction_weights.to(device)
    holdback_hazards = []
    for fr in fold_results:
        fold_dir = os.path.join(results_dir, f"fold_{fr['fold']}")
        state = torch.load(os.path.join(fold_dir, "best_model.pt"),
                           map_location=device, weights_only=True)
        model = ChannelNetV8(config).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            h = model(
                hb_feat.to(device), hb_ct.to(device), hb_clin.to(device),
                hb_interact.to(device), iw_device,
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

    print(f"\nCV Mean: {cv_mean:.4f} ± {np.std(c_indices):.4f}", flush=True)
    print(f"Holdback ensemble: {holdback_c:.4f}", flush=True)
    print(f"Holdback per-fold: {[f'{c:.4f}' for c in per_fold_hb]}", flush=True)
    print(f"Overfit gap: {overfit_gap:+.4f} ({'OK' if abs(overfit_gap) < 0.01 else 'WARNING'})", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)

    results = {
        'model': f'ChannelNetV8{"_TMB" if args.with_tmb else ""}',
        'config': {k: v for k, v in config.items()
                   if k != 'interaction_weights'},
        'mean_c_index': cv_mean,
        'std_c_index': np.std(c_indices),
        'fold_results': fold_results,
        'holdback': {
            'n_patients': n_holdback,
            'ensemble_c_index': holdback_c,
            'per_fold_c_indices': per_fold_hb,
            'overfit_gap': overfit_gap,
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
