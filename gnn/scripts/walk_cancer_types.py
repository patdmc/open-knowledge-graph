"""
Walk cancer types — train per-CT bilinear models.

Counter-slice to walk_channels: instead of grouping patients by which
biological channel is disrupted, group by cancer type. The per-CT W matrix
reveals which edge-type interactions are tissue-specific, and comparing
to channel-walk results identifies where tissue context overrides pathway logic.

Anti-concordant CTs from the global model are the most interesting targets:
their W matrices may have inverted signs on specific edge types, pointing
to missing TREATMENT_RESPONSE or TISSUE_CONTEXT edges.

Usage:
    python3 -u -m gnn.scripts.walk_cancer_types
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
from gnn.scripts.train_bilinear_edge import (
    BilinearEdgeModel, EDGE_TYPES, N_EDGE_TYPES,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "ct_walk",
)


def train_ct_model(ct_name, patient_edges, nm, ct, times, events,
                   ct_patient_idx, epochs=100, lr=1e-3, batch_size=512, seed=42):
    """Train a single-CT bilinear model (no CT embedding — just one W)."""

    idx = ct_patient_idx
    n = len(idx)

    if n < 100:
        print(f"  Skipping {ct_name}: only {n} patients")
        return None

    n_events = events[idx].sum().item()
    if n_events < 20:
        print(f"  Skipping {ct_name}: only {n_events} events")
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

        # Single-CT model: n_cancer_types=1, all patients get CT index 0
        model = BilinearEdgeModel(N_EDGE_TYPES, 1).to('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Fake CT indices (all 0 since single-CT model)
        fake_ct = torch.zeros(len(events), dtype=torch.long)

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
                hazard = model(patient_edges[b_abs], nm[b_abs], fake_ct[b_abs])
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
                    h_val = model(patient_edges[val_t], nm[val_t], fake_ct[val_t]
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

    if best_global_state is None:
        return None

    # Holdback
    model.load_state_dict(best_global_state)
    model.eval()
    fake_ct = torch.zeros(len(events), dtype=torch.long)
    with torch.no_grad():
        h_abs = torch.tensor(holdback_abs, dtype=torch.long)
        h_pred = model(patient_edges[h_abs], nm[h_abs], fake_ct[h_abs]).numpy().flatten()

    e_hb = events[holdback_abs].numpy().astype(bool)
    t_hb = times[holdback_abs].numpy()
    valid_hb = t_hb > 0
    try:
        c_hb = concordance_index_censored(e_hb[valid_hb], t_hb[valid_hb], h_pred[valid_hb])[0]
    except Exception:
        c_hb = 0.5

    # Extract the single W matrix
    W = model.get_W_matrix(0)

    return {
        'ct_name': ct_name,
        'n_patients': n,
        'n_events': int(n_events),
        'fold_results': fold_results,
        'mean_c': np.mean(fold_results),
        'holdback_c': c_hb,
        'W_matrix': W,
        'model_state': best_global_state,
    }


def main():
    print("=" * 70)
    print("CANCER TYPE WALK: Per-CT bilinear models")
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

    # === Group patients by cancer type ===
    ct_patients = defaultdict(list)
    for b in range(N_patients):
        ct_patients[ct[b].item()].append(b)

    # Sort by global model performance (worst first) — these are most informative
    bilinear_results_path = os.path.join(bilinear_dir, "results.json")
    ct_global_c = {}
    if os.path.exists(bilinear_results_path):
        with open(bilinear_results_path) as f:
            global_r = json.load(f)
        for ct_key, info in global_r.get('per_ct_holdback', {}).items():
            ct_idx_parsed = int(ct_key.split('_')[1])
            ct_global_c[ct_idx_parsed] = info['c_index']

    # Worst C-index first (most informative), fall back to size
    ct_order = sorted(ct_patients.keys(),
                      key=lambda k: ct_global_c.get(k, 0.5))

    print(f"\n  Training order (worst global C-index first):")
    for i, ct_idx in enumerate(ct_order[:10]):
        ct_name = ct_reverse.get(ct_idx, f"CT_{ct_idx}")
        gc = ct_global_c.get(ct_idx, 0.5)
        print(f"    {i+1}. {ct_name}: global C={gc:.4f}, n={len(ct_patients[ct_idx])}")

    # === Train per-CT models ===
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: load previously completed results
    results_path = os.path.join(RESULTS_DIR, "ct_walk_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        completed = {r['ct_name'] for r in all_results}
        print(f"  Resuming: {len(completed)} CTs already completed")
    else:
        all_results = []
        completed = set()

    for ct_idx in ct_order:
        ct_name = ct_reverse.get(ct_idx, f"CT_{ct_idx}")

        if ct_name in completed:
            continue

        ct_patient_idx = np.array(ct_patients[ct_idx])

        print(f"\n{'='*60}")
        print(f"  CT: {ct_name} (n={len(ct_patient_idx)}, "
              f"events={events[ct_patient_idx].sum().item():.0f})")
        print(f"{'='*60}")

        result = train_ct_model(
            ct_name, patient_edges, nm, ct, times, events,
            ct_patient_idx,
        )

        if result is None:
            continue

        print(f"  Mean CV C-index: {result['mean_c']:.4f}")
        print(f"  Per-fold: {[f'{c:.4f}' for c in result['fold_results']]}")
        print(f"  Holdback C-index: {result['holdback_c']:.4f}")

        W = result['W_matrix']

        # Edge type importance
        diag = np.diag(W)
        ranked = np.argsort(-diag)
        print(f"\n  Edge type importance (diagonal of W):")
        for r in ranked[:5]:
            print(f"    {EDGE_TYPES[r]:25s}: {diag[r]:.4f}")

        # Strongest off-diagonal
        off_diag = []
        for i in range(N_EDGE_TYPES):
            for j in range(i + 1, N_EDGE_TYPES):
                off_diag.append((EDGE_TYPES[i], EDGE_TYPES[j], W[i, j]))
        off_diag.sort(key=lambda x: -abs(x[2]))
        print(f"\n  Strongest off-diagonal:")
        for et_i, et_j, val in off_diag[:3]:
            sign = "+" if val > 0 else "-"
            print(f"    {et_i:20s} × {et_j:20s}: {sign}{abs(val):.4f}")

        # Check for sign inversions vs global model
        n_negative = (diag < 0).sum()
        if n_negative > 0:
            print(f"\n  ⚠ {n_negative} NEGATIVE diagonal entries (sign inversion):")
            for r in range(N_EDGE_TYPES):
                if diag[r] < 0:
                    print(f"    {EDGE_TYPES[r]:25s}: {diag[r]:.4f}")

        all_results.append({
            'ct_name': ct_name,
            'ct_idx': int(ct_idx),
            'n_patients': int(result['n_patients']),
            'n_events': int(result['n_events']),
            'mean_c': float(result['mean_c']),
            'holdback_c': float(result['holdback_c']),
            'fold_results': [float(c) for c in result['fold_results']],
            'W_matrix': W.tolist(),
        })

        # Checkpoint after each CT
        with open(os.path.join(RESULTS_DIR, "ct_walk_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [checkpoint saved]", flush=True)

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"CANCER TYPE WALK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Cancer Type':35s} {'N':>6s} {'Events':>7s} {'CV C':>8s} {'Holdback':>9s}")
    print("-" * 68)
    for r in sorted(all_results, key=lambda x: -x['holdback_c']):
        print(f"{r['ct_name']:35s} {r['n_patients']:6d} {r['n_events']:7d} "
              f"{r['mean_c']:8.4f} {r['holdback_c']:9.4f}")

    # === Cross-reference with global model ===
    bilinear_results = os.path.join(bilinear_dir, "results.json")
    if os.path.exists(bilinear_results):
        with open(bilinear_results) as f:
            global_r = json.load(f)
        print(f"\n{'Global bilinear':35s} {44283:6d} {'':>7s} "
              f"{global_r['mean_c_index']:8.4f} {global_r['holdback_c_index']:9.4f}")

    # === Sign inversion summary ===
    print(f"\n{'='*70}")
    print(f"SIGN INVERSIONS (edge types that flip direction per CT)")
    print(f"{'='*70}")
    for r in all_results:
        W = np.array(r['W_matrix'])
        diag = np.diag(W)
        neg = [(EDGE_TYPES[i], diag[i]) for i in range(N_EDGE_TYPES) if diag[i] < -0.01]
        if neg:
            print(f"\n  {r['ct_name']}:")
            for etype, val in neg:
                print(f"    {etype:25s}: {val:.4f}")

    # Save
    with open(os.path.join(RESULTS_DIR, "ct_walk_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
