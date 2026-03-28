#!/usr/bin/env python3
"""
Train AtlasTransformer V4 — treatment-conditional survival prediction.

Uses enriched 22-dim node features (expression, CNA, DepMap, SL, CIViC)
and 11-dim treatment vectors. Stratified Cox loss within treatment arms.

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v4
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.treatment_dataset import TreatmentDataset, NODE_FEAT_DIM, MAX_NODES
from gnn.models.atlas_transformer_v4 import AtlasTransformerV4
from gnn.models.cox_loss import CoxPartialLikelihoodLoss, StratifiedCoxLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS

# Graph features from V3 — reuse the same computation
from gnn.scripts.train_atlas_transformer_v3 import (
    compute_graph_features, GRAPH_FEAT_DIM,
)
from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence,
)
from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles
from gnn.data.channel_dataset_v6 import V6_CHANNEL_MAP, V6_TIER_MAP
from collections import defaultdict
from itertools import combinations
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--stratified-weight', type=float, default=0.5,
                        help='Weight for stratified Cox loss (0=global only, 1=stratified only)')
    return parser.parse_args()


def compute_treatment_arm(treatment_vecs):
    """Assign each patient to a treatment arm for stratified Cox loss.

    Surgery [0] and radiation [1] are zeroed out (confounders),
    so arms are based on systemic therapies only.
    """
    N = treatment_vecs.shape[0]
    arms = torch.zeros(N, dtype=torch.long)
    # Priority: immuno > targeted > chemo > endocrine > none
    for i in range(N):
        t = treatment_vecs[i]
        if t[5] > 0:   arms[i] = 4  # immunotherapy
        elif t[4] > 0: arms[i] = 3  # targeted
        elif t[2] > 0: arms[i] = 2  # chemo
        elif t[3] > 0: arms[i] = 1  # endocrine
        # else 0 = no systemic treatment
    return arms


def train_fold(model, train_loader, val_data, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    cox_loss = CoxPartialLikelihoodLoss()
    strat_loss = StratifiedCoxLoss()
    sw = config['stratified_weight']

    best_c = 0.0
    best_state = None
    patience_counter = 0

    (val_nf, val_nm, val_cp, val_ct, val_clin, val_atlas, val_gf,
     val_tv, val_tm, val_t, val_e, val_arms) = val_data

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            (nf, nm, cp, ct, clin, atlas, gf, tv, tm, t, e, arms) = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            hazard = model(nf, nm, cp, ct, clin, atlas, gf, tv, tm).squeeze()

            # Combined loss: global + stratified
            loss_global = cox_loss(hazard, t, e)
            loss_strat = strat_loss(hazard, t, e, arms)
            loss = (1 - sw) * loss_global + sw * loss_strat

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
                val_gf.to(device), val_tv.to(device), val_tm.to(device),
            ).squeeze().cpu().numpy()

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

    # === Load treatment dataset ===
    ds = TreatmentDataset()
    data = ds.build_features()

    node_features = data['node_features']
    node_masks = data['node_masks']
    channel_pos_ids = data['channel_pos_ids']
    atlas_sums = data['atlas_sums']
    times = data['times']
    events = data['events']
    cancer_types = data['cancer_types']
    clinical = data['clinical']
    treatment_vecs = data['treatment_vec']
    treatment_masks = data['treatment_known_mask']
    n_cancer_types = data['n_cancer_types']
    splits = data['split_indices']

    N = len(events)
    treatment_arms = compute_treatment_arm(treatment_vecs)

    # === Compute graph structural features ===
    print("\nComputing graph structural features...", flush=True)

    expanded_cm = load_expanded_channel_map()
    expanded_genes = set(expanded_cm.keys())

    # Build patient gene map from our mutation data
    import pandas as pd
    mut_df = pd.read_csv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'cache', 'tcga', 'tcga_mutations.csv'
    ))

    # We need patient_ids in the same order as the dataset
    patient_ids = data.get('gene_names', [])  # This might not be right
    # Actually we need patient ordering from the dataset
    # The dataset processes patients in a specific order.
    # For graph features, we compute per-patient based on their mutations.
    # Since the dataset already has enriched features including SL,
    # the graph features are complementary structural features.

    # Simple approach: compute graph features per patient using the node_features
    # to identify which genes are mutated (node_mask > 0 means real mutation).
    # But we don't have gene names in the tensors.

    # Better: build graph features from the mutation data, matching patient order.
    # The dataset returns patients in a specific order. We can rebuild by
    # creating a zero graph_features tensor and filling where we can match.

    # Simplest: use zero graph features for V4 first run, since V3 showed they
    # don't help. Focus on the NEW signal (treatment + enriched features).
    print("  Using zero graph features (V3 showed no benefit from graph structure)", flush=True)
    graph_features = torch.zeros(N, GRAPH_FEAT_DIM, dtype=torch.float32)

    # === Config ===
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
        'random_seed': args.seed,
        'n_cancer_types': n_cancer_types,
        'max_nodes': MAX_NODES,
        'graph_feat_dim': GRAPH_FEAT_DIM,
        'stratified_weight': args.stratified_weight,
    }

    # === Use pre-built splits ===
    train_idx = splits['train']
    val_idx = splits['val']
    holdback_idx = splits['holdback']

    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}, Holdback: {len(holdback_idx)}", flush=True)

    tag = "atlas_transformer_v4"
    results_dir = os.path.join(GNN_RESULTS, tag)
    os.makedirs(results_dir, exist_ok=True)

    # === Train ===
    print(f"\n{'='*50}", flush=True)
    print("Training AtlasTransformer V4", flush=True)
    print(f"{'='*50}", flush=True)

    train_ds = TensorDataset(
        node_features[train_idx], node_masks[train_idx],
        channel_pos_ids[train_idx], cancer_types[train_idx],
        clinical[train_idx], atlas_sums[train_idx],
        graph_features[train_idx],
        treatment_vecs[train_idx], treatment_masks[train_idx],
        times[train_idx], events[train_idx],
        treatment_arms[train_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, drop_last=True)

    model = AtlasTransformerV4(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    val_data = (
        node_features[val_idx], node_masks[val_idx],
        channel_pos_ids[val_idx], cancer_types[val_idx],
        clinical[val_idx], atlas_sums[val_idx],
        graph_features[val_idx],
        treatment_vecs[val_idx], treatment_masks[val_idx],
        times[val_idx], events[val_idx],
        treatment_arms[val_idx],
    )

    best_c, best_state, n_epochs = train_fold(
        model, train_loader, val_data, config, device,
    )
    print(f"\nVal C-index: {best_c:.4f} (epochs: {n_epochs})", flush=True)

    # Save best model
    torch.save(best_state, os.path.join(results_dir, "best_model.pt"))

    # === Holdback evaluation ===
    print(f"\n{'='*50}", flush=True)
    print("Holdback evaluation (PURE UNSEEN TEST SET)", flush=True)
    print(f"{'='*50}", flush=True)

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        hb_hazard = model(
            node_features[holdback_idx].to(device),
            node_masks[holdback_idx].to(device),
            channel_pos_ids[holdback_idx].to(device),
            cancer_types[holdback_idx].to(device),
            clinical[holdback_idx].to(device),
            atlas_sums[holdback_idx].to(device),
            graph_features[holdback_idx].to(device),
            treatment_vecs[holdback_idx].to(device),
            treatment_masks[holdback_idx].to(device),
        ).squeeze().cpu().numpy()

    hb_times = times[holdback_idx].numpy()
    hb_events = events[holdback_idx].numpy()

    holdback_c = concordance_index(hb_times, hb_events, hb_hazard)
    overfit_gap = best_c - holdback_c

    td_auc = time_dependent_auc(hb_hazard, hb_times, hb_events, [12, 36, 60])

    # Per-treatment-arm C-index on holdback
    hb_arms = treatment_arms[holdback_idx].numpy()
    arm_names = ['none', 'endocrine', 'chemo', 'targeted', 'immuno']
    arm_results = {}
    for arm_id, arm_name in enumerate(arm_names):
        mask = hb_arms == arm_id
        n_arm = mask.sum()
        n_events = hb_events[mask].sum()
        if n_events >= 10:
            arm_c = concordance_index(hb_times[mask], hb_events[mask], hb_hazard[mask])
            arm_results[arm_name] = {'c_index': arm_c, 'n': int(n_arm), 'events': int(n_events)}

    print(f"\n{'='*50}", flush=True)
    print(f"ATLAS TRANSFORMER V4 RESULTS", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Validation C-index:  {best_c:.4f}", flush=True)
    print(f"Holdback C-index:    {holdback_c:.4f}", flush=True)
    print(f"Overfit gap:         {overfit_gap:+.4f} ({'OK' if abs(overfit_gap) < 0.03 else 'WARNING'})", flush=True)
    print(f"TD-AUC:              {td_auc}", flush=True)
    print(f"Epochs:              {n_epochs}", flush=True)
    print(f"", flush=True)
    print(f"Per-treatment-arm holdback:", flush=True)
    for arm_name, ar in sorted(arm_results.items(), key=lambda x: -x[1]['events']):
        print(f"  {arm_name:12s}: C={ar['c_index']:.4f} (n={ar['n']}, events={ar['events']})", flush=True)
    print(f"", flush=True)
    print(f"BASELINES:", flush=True)
    print(f"  Atlas lookup (zero-param):    C = 0.5772", flush=True)
    print(f"  AtlasTransformer V1:          C = 0.6733 (MSK-IMPACT)", flush=True)
    print(f"  AtlasTransformer V3:          C = 0.6657 (MSK-IMPACT)", flush=True)
    print(f"  ChannelNetV6c:                C = 0.6991 (MSK-IMPACT)", flush=True)
    print(f"  Pairwise graph + ridge:       C = 0.6882 (MSK-IMPACT)", flush=True)
    print(f"  NOTE: V4 is on TCGA (~10K) not MSK-IMPACT (~44K)", flush=True)

    results = {
        'model': 'AtlasTransformerV4',
        'config': {k: v for k, v in config.items() if not isinstance(v, np.ndarray)},
        'val_c_index': best_c,
        'holdback': {
            'n_patients': len(holdback_idx),
            'c_index': holdback_c,
            'overfit_gap': overfit_gap,
            'td_auc': td_auc,
            'per_arm': arm_results,
        },
        'n_patients_train': len(train_idx),
        'n_patients_val': len(val_idx),
        'n_patients_total': N,
        'n_epochs': n_epochs,
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
