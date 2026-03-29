"""
Walk the channels — train per-channel bilinear models.

For each of the 6 coupling channels, train a bilinear model on patients
who have at least one mutation in that channel. The per-channel W matrix
reveals which edge-type interactions matter WITHIN that channel, and
the residuals point to missing sub-channel structure.

Usage:
    python3 -u -m gnn.scripts.walk_channels
"""

import os
import sys
import json
import time
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES
from gnn.models.cox_sage import cox_ph_loss
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES
from gnn.scripts.train_bilinear_edge import (
    BilinearEdgeModel, EDGE_TYPES, N_EDGE_TYPES,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "channel_walk",
)


def train_channel_model(channel_name, patient_edges, nm, ct, times, events,
                        n_cancer_types, channel_patient_idx,
                        epochs=100, lr=1e-3, batch_size=1024, seed=42):
    """Train bilinear model on patients with mutations in a specific channel."""

    idx = channel_patient_idx
    n = len(idx)

    if n < 200:
        print(f"  Skipping {channel_name}: only {n} patients")
        return None

    # Holdback
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_holdback = int(n * 0.15)
    holdback_rel = perm[:n_holdback]
    cv_rel = perm[n_holdback:]

    holdback_abs = idx[holdback_rel]
    cv_abs = idx[cv_rel]

    events_cv = events[cv_abs].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    best_global_state = None
    best_global_c = 0.0

    for fold, (train_rel, val_rel) in enumerate(skf.split(cv_abs, events_cv)):
        train_abs = cv_abs[train_rel]
        val_abs = cv_abs[val_rel]

        model = BilinearEdgeModel(N_EDGE_TYPES, n_cancer_types).to('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_c = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            ep_perm = np.random.permutation(len(train_abs))
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, len(ep_perm), batch_size):
                b_rel = ep_perm[b_start:b_start + batch_size]
                b_abs = torch.tensor(train_abs[b_rel], dtype=torch.long)

                optimizer.zero_grad()
                hazard = model(patient_edges[b_abs], nm[b_abs], ct[b_abs])
                loss = cox_ph_loss(hazard, times[b_abs], events[b_abs].float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_t = torch.tensor(val_abs, dtype=torch.long)
                    h_val = model(patient_edges[val_t], nm[val_t], ct[val_t]
                                  ).numpy().flatten()

                e_val = events[val_abs].numpy().astype(bool)
                t_val = times[val_abs].numpy()
                valid = t_val > 0
                try:
                    c = concordance_index_censored(e_val[valid], t_val[valid], h_val[valid])[0]
                except Exception:
                    c = 0.5

                if c > best_c:
                    best_c = c
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= 10:
                    break

        fold_results.append(best_c)
        if best_c > best_global_c:
            best_global_c = best_c
            best_global_state = best_state

    # Holdback
    model.load_state_dict(best_global_state)
    model.eval()
    with torch.no_grad():
        h_abs = torch.tensor(holdback_abs, dtype=torch.long)
        h_pred = model(patient_edges[h_abs], nm[h_abs], ct[h_abs]).numpy().flatten()

    e_hb = events[holdback_abs].numpy().astype(bool)
    t_hb = times[holdback_abs].numpy()
    valid_hb = t_hb > 0
    c_hb = concordance_index_censored(e_hb[valid_hb], t_hb[valid_hb], h_pred[valid_hb])[0]

    # Extract W matrix (average across top CTs for this channel)
    ct_counts = defaultdict(int)
    for i in idx:
        ct_counts[ct[i].item()] += 1
    top_cts = sorted(ct_counts, key=ct_counts.get, reverse=True)[:5]

    W_matrices = {}
    for ct_idx in top_cts:
        W_matrices[ct_idx] = model.get_W_matrix(ct_idx)

    return {
        'channel': channel_name,
        'n_patients': n,
        'fold_results': fold_results,
        'mean_c': np.mean(fold_results),
        'holdback_c': c_hb,
        'W_matrices': W_matrices,
        'top_cts': top_cts,
        'model_state': best_global_state,
    }


def main():
    print("=" * 70)
    print("CHANNEL WALK: Per-channel bilinear models")
    print("=" * 70)

    # === Load data ===
    print("\nLoading data...", flush=True)
    ds = AtlasDataset()
    data = ds.build_features()
    ct_map = data['cancer_type_map']
    ct_reverse = {v: k for k, v in ct_map.items()}

    gene_vocab = {}
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    G = len(gene_vocab)

    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]

    # Load cached edge matrix
    bilinear_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "bilinear_edge",
    )
    edge_matrix = np.load(os.path.join(bilinear_dir, "raw_edge_matrix.npy"))
    edge_matrix_t = torch.tensor(edge_matrix, dtype=torch.float32)

    nm = data['node_masks']
    ct = data['cancer_types']
    times = data['times']
    events = data['events']
    n_cancer_types = data['n_cancer_types']
    N_patients = len(events)

    # Precompute patient edges
    print("  Precomputing patient edge features...", flush=True)
    t0 = time.time()
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)
    patient_edges = torch.zeros(N_patients, MAX_NODES, MAX_NODES, N_EDGE_TYPES,
                                dtype=torch.float32)
    batch_gather = 4096
    for start in range(0, N_patients, batch_gather):
        end = min(start + batch_gather, N_patients)
        idx = gene_indices_t[start:end].clamp(0, G - 1)
        idx_i = idx.unsqueeze(2).expand(-1, MAX_NODES, MAX_NODES)
        idx_j = idx.unsqueeze(1).expand(-1, MAX_NODES, MAX_NODES)
        patient_edges[start:end] = edge_matrix_t[idx_i, idx_j]
        m = nm[start:end]
        pair_mask = (m.unsqueeze(1) * m.unsqueeze(2)).unsqueeze(-1)
        patient_edges[start:end] *= pair_mask
    print(f"  Done [{time.time()-t0:.1f}s]")

    # === Identify patients per channel ===
    # Build reverse map: channel → set of genes
    channel_genes = defaultdict(set)
    for gene, channel in CHANNEL_MAP.items():
        channel_genes[channel].add(gene)

    # For each patient, which channels do they have mutations in?
    patient_channels = defaultdict(set)
    for b, patient_genes in enumerate(data['gene_names']):
        for g in patient_genes:
            if g and g != '' and g != 'WT':
                ch = CHANNEL_MAP.get(g)
                if ch:
                    patient_channels[b].add(ch)

    # === Train per-channel models ===
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: load any previously completed results
    results_path = os.path.join(RESULTS_DIR, "channel_walk_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        completed = {r['channel'] for r in all_results}
        print(f"\n  Resuming: {len(completed)} channels already done: {completed}")
    else:
        all_results = []
        completed = set()

    for channel_name in CHANNEL_NAMES:
        if channel_name in completed:
            print(f"\n  Skipping {channel_name} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"  CHANNEL: {channel_name}")
        print(f"{'='*60}")

        # Patients with at least one mutation in this channel
        channel_idx = np.array([b for b in range(N_patients)
                                if channel_name in patient_channels.get(b, set())])
        n_events = events[channel_idx].sum().item()
        print(f"  Patients: {len(channel_idx)}, Events: {n_events}")

        result = train_channel_model(
            channel_name, patient_edges, nm, ct, times, events,
            n_cancer_types, channel_idx,
        )

        if result is None:
            continue

        print(f"  Mean CV C-index: {result['mean_c']:.4f}")
        print(f"  Per-fold: {[f'{c:.4f}' for c in result['fold_results']]}")
        print(f"  Holdback C-index: {result['holdback_c']:.4f}")

        # Print W matrix for largest CT
        print(f"\n  W matrix (largest CT in this channel):")
        top_ct = result['top_cts'][0]
        ct_name = ct_reverse.get(top_ct, f"CT_{top_ct}")
        W = result['W_matrices'][top_ct]
        print(f"  Cancer type: {ct_name}")
        print(f"  {'':12s} " + " ".join(f"{et[:6]:>7s}" for et in EDGE_TYPES))
        for i, et_i in enumerate(EDGE_TYPES):
            vals = " ".join(f"{W[i,j]:7.3f}" for j in range(N_EDGE_TYPES))
            print(f"  {et_i[:12]:12s} {vals}")

        # Which edge types have the strongest diagonal?
        diag = np.diag(W)
        ranked = np.argsort(-diag)
        print(f"\n  Edge type importance (diagonal of W):")
        for r in ranked:
            print(f"    {EDGE_TYPES[r]:25s}: {diag[r]:.4f}")

        # Strongest off-diagonal interactions
        off_diag = []
        for i in range(N_EDGE_TYPES):
            for j in range(i + 1, N_EDGE_TYPES):
                off_diag.append((EDGE_TYPES[i], EDGE_TYPES[j], W[i, j]))
        off_diag.sort(key=lambda x: -abs(x[2]))
        print(f"\n  Strongest off-diagonal interactions:")
        for et_i, et_j, val in off_diag[:5]:
            sign = "+" if val > 0 else "-"
            print(f"    {et_i:20s} × {et_j:20s}: {sign}{abs(val):.4f}")

        all_results.append({
            'channel': channel_name,
            'n_patients': int(result['n_patients']),
            'mean_c': float(result['mean_c']),
            'holdback_c': float(result['holdback_c']),
            'fold_results': [float(c) for c in result['fold_results']],
        })

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"CHANNEL WALK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Channel':30s} {'N':>7s} {'CV C-index':>11s} {'Holdback':>10s}")
    print("-" * 62)
    for r in sorted(all_results, key=lambda x: -x['holdback_c']):
        print(f"{r['channel']:30s} {r['n_patients']:7d} "
              f"{r['mean_c']:11.4f} {r['holdback_c']:10.4f}")

    # Compare to global model
    global_results = os.path.join(bilinear_dir, "results.json")
    if os.path.exists(global_results):
        with open(global_results) as f:
            global_r = json.load(f)
        print(f"\n{'Global bilinear':30s} {44283:7d} "
              f"{global_r['mean_c_index']:11.4f} {global_r['holdback_c_index']:10.4f}")

    # Save
    with open(os.path.join(RESULTS_DIR, "channel_walk_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
