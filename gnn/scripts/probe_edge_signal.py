"""
Probe edge signal in anti-concordant cancer types.

For each low-signal CT: amplify one edge type at a time (scale its
weight by 2x, 5x, 10x) and measure C-index change. If a flat diagonal
is actually two signals canceling, amplification will break the tie
and the C-index will move (up or down — both are informative).

Fast: single train/val split, small CTs, 50 epochs. Minutes not hours.

Usage:
    python3 -u -m gnn.scripts.probe_edge_signal
"""

import os
import sys
import json
import time
import numpy as np
import torch
from collections import defaultdict
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES
from gnn.models.cox_sage import cox_ph_loss
from gnn.scripts.train_bilinear_edge import (
    BilinearEdgeModel, EDGE_TYPES, N_EDGE_TYPES,
)
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "edge_probe",
)


def quick_c_index(model, patient_edges, nm, fake_ct, times, events, idx):
    """Single eval pass — no folds, just raw C-index."""
    model.eval()
    with torch.no_grad():
        t_idx = torch.tensor(idx, dtype=torch.long)
        preds = model(patient_edges[t_idx], nm[t_idx], fake_ct[t_idx]).numpy().flatten()

    e = events[idx].numpy().astype(bool)
    t = times[idx].numpy()
    valid = t > 0
    try:
        return concordance_index_censored(e[valid], t[valid], preds[valid])[0]
    except Exception:
        return 0.5


def probe_ct(ct_name, patient_edges, nm, times, events, ct_idx,
             epochs=50, lr=1e-3, seed=42):
    """For one CT: train baseline, then probe each edge type with amplification."""

    idx = ct_idx
    n = len(idx)
    n_events = events[idx].sum().item()

    if n < 80 or n_events < 20:
        return None

    # Simple 70/30 split
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_train = int(n * 0.7)
    train_idx = idx[perm[:n_train]]
    val_idx = idx[perm[n_train:]]

    fake_ct = torch.zeros(len(events), dtype=torch.long)

    def train_model(edge_feats, tag=""):
        model = BilinearEdgeModel(N_EDGE_TYPES, 1).to('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)

        best_c = 0.0
        for epoch in range(epochs):
            model.train()
            ep_perm = np.random.permutation(len(train_idx))
            for b_start in range(0, len(ep_perm), 512):
                b_rel = ep_perm[b_start:b_start + 512]
                b_abs = torch.tensor(train_idx[b_rel], dtype=torch.long)
                optimizer.zero_grad()
                hazard = model(edge_feats[b_abs], nm[b_abs], fake_ct[b_abs])
                loss = cox_ph_loss(hazard, times[b_abs], events[b_abs].float())
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                c = quick_c_index(model, edge_feats, nm, fake_ct, times, events, val_idx)
                best_c = max(best_c, c)

        return best_c

    # Baseline: no amplification
    baseline_c = train_model(patient_edges, "baseline")
    print(f"  Baseline C-index: {baseline_c:.4f}")

    results = {'baseline': baseline_c, 'probes': {}}

    # Probe each edge type
    for k, etype in enumerate(EDGE_TYPES):
        for scale in [3.0, 10.0]:
            # Create amplified copy
            amp_edges = patient_edges.clone()
            amp_edges[:, :, :, k] *= scale

            c = train_model(amp_edges, f"{etype}×{scale}")
            delta = c - baseline_c
            direction = "+" if delta > 0 else "-" if delta < 0 else "="
            sig = "***" if abs(delta) > 0.03 else "**" if abs(delta) > 0.02 else "*" if abs(delta) > 0.01 else ""

            key = f"{etype}×{scale:.0f}"
            results['probes'][key] = {'c': c, 'delta': delta}

            if abs(delta) > 0.005:
                print(f"    {etype:25s} ×{scale:>4.0f}: C={c:.4f} ({direction}{abs(delta):.4f}) {sig}")

    # Also probe pairs — amplify two edge types together
    print(f"\n  Pair probes (top movers):")
    top_singles = sorted(results['probes'].items(),
                         key=lambda x: -abs(x[1]['delta']))[:5]
    top_etypes = set()
    for key, _ in top_singles:
        etype = key.split('×')[0]
        top_etypes.add(etype)

    for et1 in top_etypes:
        k1 = EDGE_TYPES.index(et1)
        for et2 in top_etypes:
            if et2 <= et1:
                continue
            k2 = EDGE_TYPES.index(et2)
            amp_edges = patient_edges.clone()
            amp_edges[:, :, :, k1] *= 5.0
            amp_edges[:, :, :, k2] *= 5.0

            c = train_model(amp_edges, f"{et1}+{et2}")
            delta = c - baseline_c
            direction = "+" if delta > 0 else "-"

            key = f"{et1}+{et2}×5"
            results['probes'][key] = {'c': c, 'delta': delta}

            if abs(delta) > 0.005:
                print(f"    {et1:15s} + {et2:15s}: C={c:.4f} ({direction}{abs(delta):.4f})")

    return results


def main():
    print("=" * 70)
    print("EDGE SIGNAL PROBE: Amplify edge types in anti-concordant CTs")
    print("=" * 70)

    # Load data
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

    # Load 17D edge matrix
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

    # Load CT walk results to find anti-concordant CTs
    ct_walk_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "ct_walk", "ct_walk_results.json",
    )
    with open(ct_walk_path) as f:
        ct_walk = json.load(f)

    # Sort by holdback C — worst first
    ct_walk.sort(key=lambda x: x['holdback_c'])

    # Group patients by CT
    ct_patients = defaultdict(list)
    for b in range(N_patients):
        ct_patients[ct[b].item()].append(b)

    # Probe the worst CTs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: load previously completed probes
    probe_path = os.path.join(RESULTS_DIR, "probe_results.json")
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            all_probes = json.load(f)
        print(f"  Resuming: {len(all_probes)} CTs already probed")
    else:
        all_probes = {}

    for r in ct_walk:
        if r['holdback_c'] > 0.52:
            break  # Only probe anti-concordant / near-chance CTs

        ct_idx_val = r['ct_idx']
        ct_name = r['ct_name']

        if ct_name in all_probes:
            print(f"  Skipping {ct_name} (already probed)")
            continue

        ct_patient_idx = np.array(ct_patients.get(ct_idx_val, []))

        if len(ct_patient_idx) < 80:
            continue

        print(f"\n{'='*60}")
        print(f"  {ct_name} (n={len(ct_patient_idx)}, "
              f"CT walk C={r['holdback_c']:.4f})")
        print(f"{'='*60}")

        result = probe_ct(ct_name, patient_edges, nm, times, events,
                          ct_patient_idx)

        if result is None:
            continue

        # Summary: which edge types moved it most?
        probes = result['probes']
        sorted_probes = sorted(probes.items(), key=lambda x: -abs(x[1]['delta']))

        print(f"\n  TOP MOVERS:")
        for key, info in sorted_probes[:8]:
            if abs(info['delta']) > 0.003:
                d = "+" if info['delta'] > 0 else "-"
                print(f"    {key:30s}: {d}{abs(info['delta']):.4f}")

        all_probes[ct_name] = {
            'n': len(ct_patient_idx),
            'baseline_c': result['baseline'],
            'ct_walk_c': r['holdback_c'],
            'probes': {k: v for k, v in sorted_probes[:15]},
        }

        # Checkpoint after each CT
        with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
            json.dump(all_probes, f, indent=2, default=str)
        print(f"  [checkpoint saved]", flush=True)

    # === Cross-CT summary ===
    print(f"\n{'='*70}")
    print(f"CROSS-CT PROBE SUMMARY")
    print(f"{'='*70}")
    print(f"Which edge types consistently move anti-concordant CTs?\n")

    # Aggregate: for each edge type, average delta across CTs
    etype_deltas = defaultdict(list)
    for ct_name, info in all_probes.items():
        for key, probe in info['probes'].items():
            etype_deltas[key].append(probe['delta'])

    print(f"{'Edge Type Probe':35s} {'Mean Δ':>8s} {'Max Δ':>8s} {'CTs':>5s}")
    print("-" * 60)
    for key in sorted(etype_deltas, key=lambda k: -abs(np.mean(etype_deltas[k]))):
        deltas = etype_deltas[key]
        if len(deltas) >= 2:
            print(f"{key:35s} {np.mean(deltas):+8.4f} {max(deltas, key=abs):+8.4f} {len(deltas):5d}")

    # === CT × Channel intersection probes ===
    print(f"\n{'='*70}")
    print(f"CT × CHANNEL INTERSECTION PROBES")
    print(f"{'='*70}")

    # Build patient → channels mapping
    patient_channels = defaultdict(set)
    for b, patient_genes in enumerate(data['gene_names']):
        for g in patient_genes:
            if g and g != '' and g != 'WT':
                ch = CHANNEL_MAP.get(g)
                if ch:
                    patient_channels[b].add(ch)

    # For each anti-concordant CT: which channels are most represented?
    intersection_probes = {}
    for ct_name, info in all_probes.items():
        ct_idx_val = None
        for r in ct_walk:
            if r['ct_name'] == ct_name:
                ct_idx_val = r['ct_idx']
                break
        if ct_idx_val is None:
            continue

        ct_patient_idx = np.array(ct_patients.get(ct_idx_val, []))

        # Count channels in this CT
        ch_counts = defaultdict(int)
        for b in ct_patient_idx:
            for ch in patient_channels.get(b, set()):
                ch_counts[ch] += 1

        print(f"\n  {ct_name}: channel distribution")
        for ch in sorted(ch_counts, key=ch_counts.get, reverse=True):
            pct = 100 * ch_counts[ch] / len(ct_patient_idx)
            print(f"    {ch:20s}: {ch_counts[ch]:5d} ({pct:.0f}%)")

        # For top 3 channels: probe edge types within that intersection
        for ch in sorted(ch_counts, key=ch_counts.get, reverse=True)[:3]:
            ch_idx = np.array([b for b in ct_patient_idx
                               if ch in patient_channels.get(b, set())])
            n_events_ch = events[ch_idx].sum().item()

            if len(ch_idx) < 50 or n_events_ch < 15:
                continue

            print(f"\n  --- {ct_name} × {ch} (n={len(ch_idx)}, events={n_events_ch:.0f}) ---")

            result = probe_ct(f"{ct_name}×{ch}", patient_edges, nm,
                              times, events, ch_idx, epochs=40)

            if result is None:
                continue

            probes = result['probes']
            sorted_probes = sorted(probes.items(), key=lambda x: -abs(x[1]['delta']))

            for key, pinfo in sorted_probes[:5]:
                if abs(pinfo['delta']) > 0.005:
                    d = "+" if pinfo['delta'] > 0 else "-"
                    print(f"    {key:30s}: {d}{abs(pinfo['delta']):.4f}")

            intersection_probes[f"{ct_name}×{ch}"] = {
                'n': len(ch_idx),
                'baseline_c': result['baseline'],
                'probes': {k: v for k, v in sorted_probes[:10]},
            }

            # Checkpoint after each intersection
            all_probes['intersections'] = intersection_probes
            with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
                json.dump(all_probes, f, indent=2, default=str)
            print(f"  [checkpoint saved]", flush=True)

    all_probes['intersections'] = intersection_probes

    # Save
    with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
        json.dump(all_probes, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
