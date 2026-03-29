#!/usr/bin/env python3
"""
Train AtlasTransformer V3 — graph-informed context-conditioned attention.

Same per-mutation node features as V2, but adds patient-level graph structural
features computed from PPI/pathway topology:
  - Pairwise distances between mutated genes
  - Connected components in patient's mutation subgraph
  - Cross-channel edge density
  - Hub centrality of mutations

The graph generates dimensions. The transformer learns what they mean.

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v3
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, NODE_FEAT_DIM, MAX_NODES, MUTATION_ONLY
from gnn.models.atlas_transformer_v3 import AtlasTransformerV3
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS, ANALYSIS_CACHE, HUB_GENES

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence,
)
from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles, profile_entropy
from gnn.data.channel_dataset_v6 import V6_CHANNEL_NAMES, V6_CHANNEL_MAP, V6_TIER_MAP

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import networkx as nx
from itertools import combinations

N_CH = len(V6_CHANNEL_NAMES)
CH_TO_IDX = {ch: i for i, ch in enumerate(V6_CHANNEL_NAMES)}

# Graph feature dimension
# 17 base + 22 pairwise + 8 component = 47 graph-structural features
GRAPH_FEAT_DIM = 47


def compute_graph_features(patient_genes_map, G_ppi, expanded_cm,
                            channel_profiles, ppi_dists, cooccurrence,
                            ct_per_patient, N):
    """
    Compute patient-level graph structural features.

    These are the features the graph precipitates — they only exist because
    of the PPI/pathway topology. A flat mutation vector can't compute these.
    """
    all_expanded_genes = set(expanded_cm.keys())
    hub_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_set |= ch_hubs

    ppi_node_set = set(G_ppi.nodes())
    DISCONNECTED_DIST = 10

    gene_tier = {}
    for g, ch in expanded_cm.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)
    for g, ch in V6_CHANNEL_MAP.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    X = np.zeros((N, GRAPH_FEAT_DIM), dtype=np.float32)

    for idx in range(N):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Graph features: {idx}/{N}...", flush=True)

        patient_genes = patient_genes_map.get(idx, set())
        mutated = patient_genes & all_expanded_genes
        mutated_list = sorted(mutated)
        n_mut = len(mutated_list)

        if n_mut == 0:
            continue

        # === BASE FEATURES (17) ===
        # Channel damage profile
        ch_damage = np.zeros(N_CH)
        hub_damage = 0.0
        total_entropy = 0.0
        n_hub = 0

        for g in mutated_list:
            prof = channel_profiles.get(g)
            if prof is not None:
                ch_damage += prof
                total_entropy += profile_entropy(prof)
            if g in hub_set:
                n_hub += 1
                hub_damage += sum(prof) if prof is not None else 0.0

        X[idx, 0:8] = ch_damage                              # channel damage (8)
        X[idx, 8] = total_entropy / max(n_mut, 1)            # mean entropy
        X[idx, 9] = hub_damage                                # hub damage
        X[idx, 10] = n_mut                                    # n_mutated
        # HHI (damage concentration)
        ch_total = ch_damage.sum()
        if ch_total > 0:
            shares = ch_damage / ch_total
            X[idx, 11] = (shares ** 2).sum()
        # Fraction hub
        X[idx, 12] = n_hub / max(n_mut, 1)
        # Fraction in PPI
        n_in_ppi = sum(1 for g in mutated_list if g in ppi_node_set)
        X[idx, 13] = n_in_ppi / max(n_mut, 1)
        # Number of unique channels hit
        channels_hit = set()
        for g in mutated_list:
            ch = expanded_cm.get(g)
            if ch:
                channels_hit.add(ch)
        X[idx, 14] = len(channels_hit)
        X[idx, 15] = len(channels_hit) / N_CH  # frac channels hit
        X[idx, 16] = 1.0 if n_mut == 1 else 0.0  # single mutation flag

        if n_mut < 2:
            continue

        # === PAIRWISE FEATURES (22) ===
        pairs = list(combinations(mutated_list, 2))
        n_pairs = len(pairs)

        dists = []
        overlaps = []
        same_ch = 0
        cross_ch = 0
        cooccur_weights = []
        combined_entropies = []
        hub_hub = 0
        hub_nonhub = 0
        close_cross = 0
        far_same = 0
        tier_dists = []
        cross_tier = 0

        for g1, g2 in pairs:
            # PPI distance
            d = ppi_dists.get((g1, g2), ppi_dists.get((g2, g1), DISCONNECTED_DIST))
            dists.append(d)

            # Profile overlap
            p1 = channel_profiles.get(g1)
            p2 = channel_profiles.get(g2)
            if p1 is not None and p2 is not None:
                norm1 = np.linalg.norm(p1)
                norm2 = np.linalg.norm(p2)
                if norm1 > 0 and norm2 > 0:
                    overlaps.append(np.dot(p1, p2) / (norm1 * norm2))
                # Combined entropy
                combined = (p1 + p2) / 2
                combined_entropies.append(profile_entropy(combined))

            # Same/cross channel
            ch1 = expanded_cm.get(g1)
            ch2 = expanded_cm.get(g2)
            if ch1 and ch2:
                if ch1 == ch2:
                    same_ch += 1
                    if d >= 4:
                        far_same += 1
                else:
                    cross_ch += 1
                    if d <= 2:
                        close_cross += 1

            # Co-occurrence (cooccurrence values are {ct: count} dicts)
            cw_dict = cooccurrence.get((g1, g2), cooccurrence.get((g2, g1), {}))
            if isinstance(cw_dict, dict):
                cw_total = sum(cw_dict.values()) if cw_dict else 0
            else:
                cw_total = cw_dict
            if cw_total > 0:
                cooccur_weights.append(cw_total)

            # Hub pairs
            h1 = g1 in hub_set
            h2 = g2 in hub_set
            if h1 and h2:
                hub_hub += 1
            elif h1 or h2:
                hub_nonhub += 1

            # Tier distance
            t1 = gene_tier.get(g1, -1)
            t2 = gene_tier.get(g2, -1)
            if t1 >= 0 and t2 >= 0:
                tier_dists.append(abs(t1 - t2))
                if t1 != t2:
                    cross_tier += 1

        dists_arr = np.array(dists)
        connected = dists_arr < DISCONNECTED_DIST

        X[idx, 17] = dists_arr.mean()
        X[idx, 18] = dists_arr.min() if len(dists) > 0 else DISCONNECTED_DIST
        X[idx, 19] = connected.mean()
        X[idx, 20] = (dists_arr == 1).mean()
        X[idx, 21] = ((dists_arr >= 2) & (dists_arr <= 3)).mean()
        X[idx, 22] = (dists_arr >= 4).mean()
        X[idx, 23] = np.mean(overlaps) if overlaps else 0
        X[idx, 24] = np.min(overlaps) if overlaps else 0
        X[idx, 25] = np.max(overlaps) if overlaps else 0
        X[idx, 26] = same_ch / max(n_pairs, 1)
        X[idx, 27] = cross_ch / max(n_pairs, 1)
        X[idx, 28] = np.mean(cooccur_weights) if cooccur_weights else 0
        X[idx, 29] = np.max(cooccur_weights) if cooccur_weights else 0
        X[idx, 30] = len(cooccur_weights) / max(n_pairs, 1)
        X[idx, 31] = np.mean(combined_entropies) if combined_entropies else 0
        X[idx, 32] = hub_hub
        X[idx, 33] = hub_nonhub
        X[idx, 34] = close_cross / max(n_pairs, 1)
        X[idx, 35] = far_same / max(n_pairs, 1)
        X[idx, 36] = np.mean(tier_dists) if tier_dists else 0
        X[idx, 37] = cross_tier / max(n_pairs, 1)
        X[idx, 38] = n_pairs

        # === COMPONENT FEATURES (8) ===
        ppi_mutated = [g for g in mutated_list if g in ppi_node_set]
        if len(ppi_mutated) >= 2:
            sub = G_ppi.subgraph(ppi_mutated)
            components = list(nx.connected_components(sub))
            comp_sizes = [len(c) for c in components]

            X[idx, 39] = len(components)
            X[idx, 40] = max(comp_sizes)
            X[idx, 41] = max(comp_sizes) / len(ppi_mutated)

            isolated = sum(1 for g in mutated_list if g not in ppi_node_set)
            X[idx, 42] = isolated
            X[idx, 43] = isolated / max(n_mut, 1)

            # Component entropy
            if len(comp_sizes) > 1:
                total = sum(comp_sizes)
                probs = np.array(comp_sizes) / total
                X[idx, 44] = -np.sum(probs * np.log(probs + 1e-10))

            # Max component damage
            max_comp_dmg = 0
            for comp in components:
                comp_dmg = 0
                for g in comp:
                    prof = channel_profiles.get(g)
                    if prof is not None:
                        comp_dmg += prof.sum()
                max_comp_dmg = max(max_comp_dmg, comp_dmg)
            X[idx, 45] = max_comp_dmg

            # Inter-component spread
            if len(components) > 1:
                comp_damages = []
                for comp in components:
                    cd = sum(channel_profiles.get(g, np.zeros(N_CH)).sum() for g in comp)
                    comp_damages.append(cd)
                total_cd = sum(comp_damages)
                if total_cd > 0:
                    shares = np.array(comp_damages) / total_cd
                    X[idx, 46] = -np.sum(shares * np.log(shares + 1e-10))
        elif len(ppi_mutated) == 1:
            X[idx, 39] = 1
            X[idx, 40] = 1
            X[idx, 41] = 1.0
            isolated = sum(1 for g in mutated_list if g not in ppi_node_set)
            X[idx, 42] = isolated
            X[idx, 43] = isolated / max(n_mut, 1)

    return X


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

    val_nf, val_nm, val_cp, val_ct, val_clin, val_atlas, val_gf, val_t, val_e = val_data

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            nf, nm, cp, ct, clin, atlas, gf, t, e = [b.to(device) for b in batch]
            optimizer.zero_grad()
            hazard = model(nf, nm, cp, ct, clin, atlas, gf)
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
                val_gf.to(device),
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

    # === Load atlas dataset (per-mutation features) ===
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
    N = len(events)

    # === Compute graph structural features ===
    print("\n  Computing graph structural features...", flush=True)

    expanded_cm = load_expanded_channel_map()
    expanded_genes = set(expanded_cm.keys())

    # Use AtlasDataset's patient ordering (ds.clinical iterrows order)
    # This ensures graph features align with the tensor indices
    atlas_patient_ids = ds.clinical['patientId'].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(atlas_patient_ids)}

    # Load mutations and map to atlas patient indices
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False, usecols=["patientId", "gene.hugoGeneSymbol"],
    )
    mut_exp = mut[mut["gene.hugoGeneSymbol"].isin(expanded_genes)].copy()
    mut_exp["patient_idx"] = mut_exp["patientId"].map(pid_to_idx)
    mut_exp = mut_exp[mut_exp["patient_idx"].notna()]
    mut_exp["patient_idx"] = mut_exp["patient_idx"].astype(int)

    patient_genes_map = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes_map[idx].add(row["gene.hugoGeneSymbol"])

    # Build cancer type per patient from atlas dataset's cancer_type_map (name→idx)
    ct_map = data['cancer_type_map']  # {cancer_type_name: idx}
    idx_to_ct = {v: k for k, v in ct_map.items()}  # {idx: cancer_type_name}
    ct_idx_arr = cancer_types.numpy()
    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = idx_to_ct.get(int(ct_idx_arr[idx]), "Other")

    # Build PPI graph
    msk_genes = set()
    for genes in patient_genes_map.values():
        msk_genes |= genes
    msk_genes &= expanded_genes

    cooccurrence = compute_cooccurrence(patient_genes_map, ct_per_patient, min_count=10)
    ppi_edges = fetch_string_expanded(msk_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)

    # Precompute PPI distances
    print("  Computing PPI distances...", flush=True)
    ppi_dists = {}
    for comp in nx.connected_components(G_ppi):
        sub = G_ppi.subgraph(comp)
        lengths = dict(nx.all_pairs_shortest_path_length(sub))
        for src, targets in lengths.items():
            for tgt, dist in targets.items():
                if src < tgt:
                    ppi_dists[(src, tgt)] = dist

    print(f"  PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges, "
          f"{len(ppi_dists)} distances", flush=True)

    # Compute graph features
    graph_features_np = compute_graph_features(
        patient_genes_map, G_ppi, expanded_cm, channel_profiles,
        ppi_dists, cooccurrence, ct_per_patient, N,
    )
    graph_features = torch.tensor(graph_features_np, dtype=torch.float32)
    print(f"  Graph features: shape={graph_features.shape}, "
          f"non-zero patients: {(graph_features.abs().sum(1) > 0).sum()}/{N}")

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
        'n_folds': args.n_folds,
        'random_seed': args.seed,
        'n_cancer_types': n_cancer_types,
        'max_nodes': MAX_NODES,
        'graph_feat_dim': GRAPH_FEAT_DIM,
        'mutation_only': MUTATION_ONLY,
    }

    # Holdback split
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
    hb_gf = graph_features[holdback_idx]
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
    graph_features = graph_features[train_pool_idx]

    print(f"\nHoldback: {n_holdback}, CV pool: {len(events)}", flush=True)

    tag = "atlas_transformer_v3"
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
            graph_features[train_idx],
            times[train_idx], events[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

        model = AtlasTransformerV3(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if fold == 0:
            print(f"  Parameters: {n_params:,}", flush=True)

        val_data = (
            node_features[val_idx], node_masks[val_idx],
            channel_pos_ids[val_idx], cancer_types[val_idx],
            clinical[val_idx], atlas_sums[val_idx],
            graph_features[val_idx],
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
        model = AtlasTransformerV3(config).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            h = model(
                hb_nf.to(device), hb_nm.to(device), hb_cp.to(device),
                hb_ct.to(device), hb_clin.to(device), hb_atlas.to(device),
                hb_gf.to(device),
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

    td_auc = time_dependent_auc(ensemble_hazard, hb_times.numpy(),
                                 hb_events.numpy(), [12, 36, 60])

    print(f"\n{'='*50}", flush=True)
    print(f"ATLAS TRANSFORMER V3 RESULTS", flush=True)
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
    print(f"  AtlasTransformer V1:          C = 0.6733", flush=True)
    print(f"  AtlasTransformer V2:          C = (partial)", flush=True)
    print(f"  ChannelNetV6c:                C = 0.6991", flush=True)
    print(f"  Pairwise graph + ridge:       C = 0.6882", flush=True)

    results = {
        'model': 'AtlasTransformerV3',
        'config': {k: v for k, v in config.items() if not isinstance(v, np.ndarray)},
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
        'n_patients_cv': int(len(events)),
        'n_patients_total': n_total,
        'elapsed_seconds': elapsed,
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
