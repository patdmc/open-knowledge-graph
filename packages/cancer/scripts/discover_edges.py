"""
Edge discovery tool — find missing relationship types from bilinear model residuals.

Process:
  1. Load bilinear model + patient data
  2. Identify worst-performing cancer types (anti-concordant or noise)
  3. For those patients: which mutation pairs does the model get wrong?
  4. Look at existing edge types for those pairs — is there countervalent signal?
  5. Cluster the residual patterns — propose new edge types

Usage:
    python3 -u -m gnn.scripts.discover_edges
"""

import os
import sys
import json
import time
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES
from gnn.scripts.train_bilinear_edge import (
    BilinearEdgeModel, EDGE_TYPES, N_EDGE_TYPES, load_raw_edge_matrix,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "bilinear_edge",
)


def main():
    print("=" * 70)
    print("EDGE DISCOVERY: Finding missing relationship types")
    print("=" * 70)

    # === Load data ===
    print("\nLoading data...", flush=True)
    ds = AtlasDataset()
    data = ds.build_features()
    ct_map = data['cancer_type_map']  # {cancer_type_name: index}
    ct_reverse = {v: k for k, v in ct_map.items()}

    # === Gene vocabulary ===
    gene_vocab = {}
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    idx_to_gene = {i: g for g, i in gene_vocab.items()}
    G = len(gene_vocab)

    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]

    # === Load edge matrix ===
    cache_path = os.path.join(RESULTS_DIR, "raw_edge_matrix.npy")
    edge_matrix = np.load(cache_path)

    # === Load model ===
    n_cancer_types = data['n_cancer_types']
    model = BilinearEdgeModel(N_EDGE_TYPES, n_cancer_types)
    state = torch.load(os.path.join(RESULTS_DIR, "best_model.pt"),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # === Load results ===
    with open(os.path.join(RESULTS_DIR, "results.json")) as f:
        results = json.load(f)

    # === Identify worst cancer types ===
    print("\n" + "=" * 70)
    print("PER-CANCER-TYPE PERFORMANCE")
    print("=" * 70)

    ct_perf = []
    for ct_key, info in results['per_ct_holdback'].items():
        ct_idx = int(ct_key.split('_')[1])
        ct_name = ct_reverse.get(ct_idx, ct_key)
        ct_perf.append({
            'ct_idx': ct_idx,
            'ct_name': ct_name,
            'c_index': info['c_index'],
            'n': info['n'],
            'events': info['events'],
            'signal': abs(info['c_index'] - 0.5),
            'direction': 'concordant' if info['c_index'] > 0.5 else 'anti-concordant',
        })

    ct_perf.sort(key=lambda x: x['c_index'])

    print(f"\n{'Cancer Type':40s} {'C-index':>8} {'N':>6} {'Events':>7} {'Direction':>16}")
    print("-" * 80)
    for r in ct_perf:
        print(f"{r['ct_name']:40s} {r['c_index']:8.4f} {r['n']:6d} {r['events']:7d} {r['direction']:>16}")

    # === Focus on anti-concordant CTs with enough data ===
    bad_cts = [r for r in ct_perf if r['c_index'] < 0.47 and r['n'] >= 50]
    print(f"\n\nAnti-concordant CTs (C < 0.47, n >= 50): {len(bad_cts)}")

    # === For worst CTs: analyze which mutation pairs contribute to error ===
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS: What's the model getting wrong?")
    print("=" * 70)

    nm = data['node_masks']
    ct_tensor = data['cancer_types']
    times = data['times'].numpy()
    events = data['events'].numpy()

    # Holdback indices (same seed as training)
    np.random.seed(42)
    all_idx = np.arange(len(events))
    np.random.shuffle(all_idx)
    n_holdback = int(len(events) * 0.15)
    holdback_idx = all_idx[:n_holdback]

    for bad_ct in bad_cts:
        ct_idx = bad_ct['ct_idx']
        ct_name = bad_ct['ct_name']
        print(f"\n--- {ct_name} (C={bad_ct['c_index']:.4f}, n={bad_ct['n']}) ---")

        # Get holdback patients for this CT
        ct_mask = ct_tensor[holdback_idx].numpy() == ct_idx
        ct_patients = holdback_idx[ct_mask]

        if len(ct_patients) < 20:
            print("  Too few patients, skipping")
            continue

        # Get model predictions
        with torch.no_grad():
            ct_abs = torch.tensor(ct_patients, dtype=torch.long)
            # Gather edge features
            gi = torch.tensor(gene_indices[ct_patients], dtype=torch.long)
            safe_idx = gi.clamp(0, G - 1)
            edge_matrix_t = torch.tensor(edge_matrix, dtype=torch.float32)
            idx_i = safe_idx.unsqueeze(2).expand(-1, MAX_NODES, MAX_NODES)
            idx_j = safe_idx.unsqueeze(1).expand(-1, MAX_NODES, MAX_NODES)
            patient_edges = edge_matrix_t[idx_i, idx_j]
            m = nm[ct_abs]
            pair_mask = (m.unsqueeze(1) * m.unsqueeze(2)).unsqueeze(-1)
            patient_edges *= pair_mask

            preds = model(patient_edges, m, ct_tensor[ct_abs]).numpy().flatten()

        ct_times = times[ct_patients]
        ct_events = events[ct_patients]

        # Find discordant pairs: model says A > B but A dies before B
        n_pts = len(ct_patients)
        discordant_genes = defaultdict(int)
        concordant_genes = defaultdict(int)
        n_disc = 0
        n_conc = 0

        for i in range(n_pts):
            if ct_events[i] != 1:
                continue
            for j in range(n_pts):
                if ct_times[j] <= ct_times[i]:
                    continue
                # Patient i died at time t_i, patient j survived past t_i
                # Model should predict hazard_i > hazard_j
                if preds[i] < preds[j]:
                    # Discordant — model got it wrong
                    n_disc += 1
                    # Which genes does patient i have that j doesn't?
                    genes_i = set(data['gene_names'][ct_patients[i]])
                    genes_j = set(data['gene_names'][ct_patients[j]])
                    for g in genes_i - genes_j:
                        if g and g != '' and g != 'WT':
                            discordant_genes[g] += 1
                    for g in genes_j - genes_i:
                        if g and g != '' and g != 'WT':
                            discordant_genes[g] += 1
                else:
                    n_conc += 1
                    genes_i = set(data['gene_names'][ct_patients[i]])
                    genes_j = set(data['gene_names'][ct_patients[j]])
                    for g in genes_i - genes_j:
                        if g and g != '' and g != 'WT':
                            concordant_genes[g] += 1

        total_pairs = n_disc + n_conc
        if total_pairs == 0:
            continue

        print(f"  Concordant pairs: {n_conc}, Discordant: {n_disc} "
              f"({100*n_disc/total_pairs:.1f}% error rate)")

        # Genes that appear disproportionately in discordant pairs
        print(f"\n  Genes driving discordance (appear in wrong predictions):")
        disc_ratio = {}
        for g in set(list(discordant_genes.keys()) + list(concordant_genes.keys())):
            d = discordant_genes.get(g, 0)
            c = concordant_genes.get(g, 0)
            total = d + c
            if total >= 10:
                ratio = d / total
                disc_ratio[g] = (ratio, d, c, total)

        sorted_genes = sorted(disc_ratio.items(), key=lambda x: -x[1][0])
        print(f"  {'Gene':15s} {'Disc%':>7s} {'Disc':>6s} {'Conc':>6s} {'Total':>6s} {'Channel':>15s}")
        for g, (ratio, d, c, total) in sorted_genes[:15]:
            ch = CHANNEL_MAP.get(g, "—")
            print(f"  {g:15s} {100*ratio:6.1f}% {d:6d} {c:6d} {total:6d} {ch:>15s}")

        # === Edge type analysis for discordant genes ===
        # For the top discordant genes: what's their edge profile?
        print(f"\n  Edge type profiles for top discordant genes:")
        top_disc = [g for g, _ in sorted_genes[:10] if g in gene_vocab]

        for g in top_disc[:5]:
            gi = gene_vocab[g]
            profile = edge_matrix[gi, :, :]  # (G, 9)
            # How connected is this gene through each edge type?
            connected = (profile > 0).sum(axis=0)
            total_weight = profile.sum(axis=0)
            print(f"\n  {g} ({CHANNEL_MAP.get(g, '?')}):")
            for k, etype in enumerate(EDGE_TYPES):
                if connected[k] > 0:
                    print(f"    {etype:25s}: {connected[k]:4d} edges, total weight={total_weight[k]:.2f}")

        # === Look for missing connections between discordant gene pairs ===
        print(f"\n  Missing edges between discordant gene pairs:")
        top_disc_set = set(top_disc[:10])
        missing_pairs = []

        for g1 in top_disc_set:
            if g1 not in gene_vocab:
                continue
            for g2 in top_disc_set:
                if g2 not in gene_vocab or g1 >= g2:
                    continue
                i, j = gene_vocab[g1], gene_vocab[g2]
                profile = edge_matrix[i, j, :]
                n_edges = (profile > 0).sum()
                if n_edges == 0:
                    missing_pairs.append((g1, g2, CHANNEL_MAP.get(g1, '?'),
                                          CHANNEL_MAP.get(g2, '?')))

        if missing_pairs:
            print(f"  Gene pairs with ZERO edges (potential missing relationship types):")
            for g1, g2, ch1, ch2 in missing_pairs[:20]:
                cross = "CROSS" if ch1 != ch2 else "SAME"
                print(f"    {g1:10s} — {g2:10s}  ({ch1} × {ch2}) [{cross}]")
        else:
            print(f"  All top discordant gene pairs have at least one edge type")

    # === W matrix analysis: which off-diagonal entries are near zero? ===
    print("\n" + "=" * 70)
    print("W MATRIX GAP ANALYSIS")
    print("Which edge-type interactions are underweighted across cancer types?")
    print("=" * 70)

    W_all = json.load(open(os.path.join(RESULTS_DIR, "W_matrices.json")))
    n_cts_loaded = len(W_all)

    # Average W across all CTs
    W_mean = np.zeros((N_EDGE_TYPES, N_EDGE_TYPES))
    for ct_name, W_list in W_all.items():
        W_mean += np.array(W_list)
    W_mean /= n_cts_loaded

    # Variance across CTs — high variance means CT-specific signal
    W_var = np.zeros((N_EDGE_TYPES, N_EDGE_TYPES))
    for ct_name, W_list in W_all.items():
        W_var += (np.array(W_list) - W_mean) ** 2
    W_var /= n_cts_loaded

    print(f"\n  Mean W matrix (averaged across {n_cts_loaded} CTs):")
    print(f"  {'':12s} " + " ".join(f"{et[:6]:>7s}" for et in EDGE_TYPES))
    for i, et_i in enumerate(EDGE_TYPES):
        vals = " ".join(f"{W_mean[i,j]:7.4f}" for j in range(N_EDGE_TYPES))
        print(f"  {et_i[:12]:12s} {vals}")

    print(f"\n  W variance (high = CT-specific interaction):")
    print(f"  {'':12s} " + " ".join(f"{et[:6]:>7s}" for et in EDGE_TYPES))
    for i, et_i in enumerate(EDGE_TYPES):
        vals = " ".join(f"{W_var[i,j]:7.5f}" for j in range(N_EDGE_TYPES))
        print(f"  {et_i[:12]:12s} {vals}")

    # === Identify high-variance off-diagonal entries ===
    print(f"\n  High-variance off-diagonal W entries (potential missing decomposition):")
    off_diag = []
    for i in range(N_EDGE_TYPES):
        for j in range(i + 1, N_EDGE_TYPES):
            off_diag.append({
                'edge_i': EDGE_TYPES[i],
                'edge_j': EDGE_TYPES[j],
                'mean': W_mean[i, j],
                'var': W_var[i, j],
                'cv': np.sqrt(W_var[i, j]) / (abs(W_mean[i, j]) + 1e-6),
            })

    off_diag.sort(key=lambda x: -x['var'])
    print(f"  {'Edge Type A':20s} {'Edge Type B':20s} {'Mean':>8s} {'Var':>10s} {'CV':>8s}")
    print("  " + "-" * 70)
    for entry in off_diag[:10]:
        print(f"  {entry['edge_i']:20s} {entry['edge_j']:20s} "
              f"{entry['mean']:8.4f} {entry['var']:10.6f} {entry['cv']:8.2f}")

    # === Summary: proposed investigations ===
    print("\n" + "=" * 70)
    print("PROPOSED INVESTIGATIONS")
    print("=" * 70)

    print("""
    Based on residual analysis, the following should be investigated:

    1. ANTI-CONCORDANT CTs: The model predicts in the wrong direction for some
       cancer types. This suggests a missing edge type that REVERSES the
       relationship between mutations and survival in those contexts.
       → Candidate: TREATMENT_RESPONSE edges (some mutations predict better
         outcome because they respond to targeted therapy)

    2. HIGH-VARIANCE W ENTRIES: Some edge-type interactions vary dramatically
       across cancer types. The current W tries to fit all variation with
       per-CT parameters. If the variance is structured (not noise), there
       may be a latent edge type that explains it.
       → Candidate: TISSUE_CONTEXT edges (gene expression context determines
         whether a mutation is harmful or protective)

    3. DISCONNECTED GENE PAIRS: Genes that drive discordance but have zero
       edges between them in the current graph. These pairs co-determine
       survival but through a mechanism not captured by any current edge type.
       → Candidate: EPIGENETIC_COUPLING, METABOLIC_DEPENDENCY, or
         IMMUNE_INTERACTION edges
    """)


if __name__ == "__main__":
    main()
