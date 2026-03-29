"""
Patient Affinity Graph — connect patients with similar mutation profiles.

Two levels of affinity:
  1. Within-cancer: patients in the same cancer type sharing mutated genes.
     Weighted by Jaccard similarity of gene sets.
  2. Cross-cancer: patients in DIFFERENT cancer types with identical or
     overlapping gene sets. Cancer type is phenotype (tissue of origin),
     not genotype (mutation profile). Same mutations walk the same graph
     topology — tissue shifts the intercept, mutations determine the slope.

Each edge carries:
  - Jaccard similarity (mutation overlap)
  - Tissue intercept delta (how much the cancer types differ at baseline)
  - Same/cross cancer flag

The affinity graph is the second graph in the two-graph model:
  1. Mutation graph (coupling channels) — which mutations interact
  2. Patient affinity graph — which patients inform each other

Combined: mutation-level interactions regularized by patient similarity,
with tissue-specific intercepts on cross-cancer edges.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, NON_SILENT, MSK_DATASETS


# Hub gene set
_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)


class PatientAffinityGraph:
    """Build per-cancer-type patient affinity graphs from mutation profiles."""

    def __init__(self, dataset_name="msk_impact_50k", min_shared_genes=1,
                 min_patients_per_cancer=50):
        self.min_shared = min_shared_genes
        self.min_patients = min_patients_per_cancer

        paths = MSK_DATASETS[dataset_name]
        print("Loading data...", flush=True)
        mutations = pd.read_csv(paths["mutations"])
        clinical = pd.read_csv(paths["clinical"])
        sample_clinical = pd.read_csv(paths["sample_clinical"])

        mutations = mutations[mutations['mutationType'].isin(NON_SILENT)]
        mutations = mutations[mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())]

        clinical = clinical.merge(
            sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
            on='patientId', how='left'
        )
        clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
        clinical['event'] = clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )

        self.clinical = clinical
        self.mutations = mutations

        # Build per-patient gene sets
        gene_muts = mutations[['patientId', 'gene.hugoGeneSymbol']].drop_duplicates()
        self.patient_genes = gene_muts.groupby('patientId')['gene.hugoGeneSymbol'].apply(
            frozenset
        ).to_dict()

        # Channel pattern per patient
        gene_muts['channel'] = gene_muts['gene.hugoGeneSymbol'].map(CHANNEL_MAP)
        self.patient_channels = gene_muts.groupby('patientId')['channel'].apply(
            frozenset
        ).to_dict()

        # Cancer type per patient
        ct_map = clinical[['patientId', 'CANCER_TYPE']].drop_duplicates('patientId')
        self.patient_cancer = dict(zip(ct_map['patientId'], ct_map['CANCER_TYPE']))

        # Group patients by cancer type
        self.cancer_patients = defaultdict(list)
        for pid, ct in self.patient_cancer.items():
            if isinstance(ct, str):
                self.cancer_patients[ct].append(pid)

        # Per-patient gene set as sorted string (for cross-cancer matching)
        self.patient_gene_str = {
            pid: ','.join(sorted(genes))
            for pid, genes in self.patient_genes.items()
        }

        # Compute tissue intercepts: median OS for deceased patients per cancer type
        deceased = clinical[clinical['event'] == 1]
        self.tissue_intercepts = deceased.groupby('CANCER_TYPE')['OS_MONTHS'].median().to_dict()
        global_median = deceased['OS_MONTHS'].median()
        # Normalize to deltas from global median
        self.tissue_deltas = {
            ct: med - global_median for ct, med in self.tissue_intercepts.items()
        }

        # Group patients by gene set string (for cross-cancer lookup)
        self.gene_set_to_patients = defaultdict(list)
        for pid, gs in self.patient_gene_str.items():
            ct = self.patient_cancer.get(pid)
            if isinstance(ct, str):
                self.gene_set_to_patients[gs].append((pid, ct))

        print(f"Patients with mutations: {len(self.patient_genes)}", flush=True)
        print(f"Cancer types: {len(self.cancer_patients)}", flush=True)
        print(f"Global median OS (deceased): {global_median:.1f} months", flush=True)
        print(f"Tissue intercept range: {min(self.tissue_deltas.values()):+.1f} to "
              f"{max(self.tissue_deltas.values()):+.1f} months", flush=True)

    def jaccard(self, set_a, set_b):
        """Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def shared_genes(self, pid_a, pid_b):
        """Number of shared mutated genes between two patients."""
        genes_a = self.patient_genes.get(pid_a, frozenset())
        genes_b = self.patient_genes.get(pid_b, frozenset())
        return len(genes_a & genes_b)

    def build_cancer_graph(self, cancer_type, max_edges_per_node=20):
        """Build affinity graph for one cancer type.

        Returns:
            nodes: list of patient IDs
            edges: list of (i, j, weight) tuples (indices into nodes)
            node_features: dict of per-node features
        """
        patients = [p for p in self.cancer_patients[cancer_type]
                     if p in self.patient_genes]
        if len(patients) < self.min_patients:
            return None

        pid_to_idx = {p: i for i, p in enumerate(patients)}

        # Build gene set vectors for efficient comparison
        # Group patients by gene set for exact-match clustering
        gene_set_groups = defaultdict(list)
        for pid in patients:
            genes = self.patient_genes.get(pid, frozenset())
            gene_set_groups[genes].append(pid)

        # Build edges: connect patients sharing >= min_shared genes
        # Use inverted index for efficiency: gene -> list of patients
        gene_to_patients = defaultdict(set)
        for pid in patients:
            for gene in self.patient_genes.get(pid, frozenset()):
                gene_to_patients[gene].add(pid)

        # For each patient, find candidates via shared genes
        edges = []
        edge_set = set()

        for pid in patients:
            genes = self.patient_genes.get(pid, frozenset())
            if not genes:
                continue

            # Candidate neighbors: anyone sharing at least one gene
            candidates = set()
            for gene in genes:
                candidates.update(gene_to_patients[gene])
            candidates.discard(pid)

            # Score candidates
            scored = []
            for cand in candidates:
                cand_genes = self.patient_genes.get(cand, frozenset())
                n_shared = len(genes & cand_genes)
                if n_shared >= self.min_shared:
                    jacc = n_shared / len(genes | cand_genes)
                    scored.append((cand, jacc, n_shared))

            # Keep top-k by Jaccard
            scored.sort(key=lambda x: -x[1])
            for cand, jacc, n_shared in scored[:max_edges_per_node]:
                i, j = pid_to_idx[pid], pid_to_idx[cand]
                edge_key = (min(i, j), max(i, j))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append((i, j, jacc))

        # Node features
        node_features = []
        for pid in patients:
            genes = self.patient_genes.get(pid, frozenset())
            channels = self.patient_channels.get(pid, frozenset())

            # Channel pattern (6-dim binary)
            ch_pattern = [1.0 if ch in channels else 0.0 for ch in CHANNEL_NAMES]

            # Gene-level features
            n_genes = len(genes)
            n_hub = sum(1 for g in genes if g in _HUB_SET)
            n_leaf = n_genes - n_hub
            hub_ratio = n_hub / max(n_genes, 1)

            # Clinical
            row = self.clinical[self.clinical['patientId'] == pid]
            if len(row) > 0:
                row = row.iloc[0]
                time = float(row['OS_MONTHS'])
                event = int(row['event'])
            else:
                time = 0.0
                event = 0

            node_features.append({
                'patientId': pid,
                'channel_pattern': ch_pattern,
                'n_genes': n_genes,
                'n_hub': n_hub,
                'n_leaf': n_leaf,
                'hub_ratio': hub_ratio,
                'genes': sorted(genes),
                'time': time,
                'event': event,
            })

        return {
            'cancer_type': cancer_type,
            'nodes': patients,
            'n_nodes': len(patients),
            'edges': edges,
            'n_edges': len(edges),
            'node_features': node_features,
            'gene_set_groups': {
                ','.join(sorted(k)): [pid_to_idx[p] for p in v]
                for k, v in gene_set_groups.items() if len(v) >= 2
            },
        }

    def build_all_graphs(self, max_edges_per_node=20):
        """Build affinity graphs for all cancer types with enough patients."""
        graphs = {}
        for ct in sorted(self.cancer_patients.keys()):
            if len(self.cancer_patients[ct]) < self.min_patients:
                continue
            g = self.build_cancer_graph(ct, max_edges_per_node)
            if g is not None:
                graphs[ct] = g
        return graphs

    def build_unified_graph(self, max_within_edges=20, max_cross_edges=5,
                            min_cross_jaccard=0.5, min_cancer_n=50):
        """Build a single graph with all patients, within- and cross-cancer edges.

        Within-cancer edges: Jaccard similarity of gene sets (same as per-cancer).
        Cross-cancer edges: patients with identical or high-overlap gene sets
            in different cancer types. Edge features include tissue intercept delta.

        Returns:
            dict with nodes, within_edges, cross_edges, node_features,
            tissue_intercepts, and summary stats.
        """
        print("Building unified affinity graph...", flush=True)

        # All patients with mutations in cancer types with enough patients
        valid_cts = {ct for ct, pids in self.cancer_patients.items()
                     if len(pids) >= min_cancer_n}
        all_patients = []
        for ct in sorted(valid_cts):
            for pid in self.cancer_patients[ct]:
                if pid in self.patient_genes:
                    all_patients.append(pid)

        pid_to_idx = {p: i for i, p in enumerate(all_patients)}
        print(f"  Nodes: {len(all_patients)} patients in {len(valid_cts)} cancer types",
              flush=True)

        # --- Within-cancer edges (reuse inverted index approach) ---
        print("  Building within-cancer edges...", flush=True)
        within_edges = []
        within_edge_set = set()

        for ct in sorted(valid_cts):
            ct_patients = [p for p in self.cancer_patients[ct]
                           if p in self.patient_genes]

            # Inverted index: gene -> patients in this cancer type
            gene_to_ct_patients = defaultdict(set)
            for pid in ct_patients:
                for gene in self.patient_genes.get(pid, frozenset()):
                    gene_to_ct_patients[gene].add(pid)

            for pid in ct_patients:
                genes = self.patient_genes.get(pid, frozenset())
                if not genes:
                    continue

                candidates = set()
                for gene in genes:
                    candidates.update(gene_to_ct_patients[gene])
                candidates.discard(pid)

                scored = []
                for cand in candidates:
                    cand_genes = self.patient_genes.get(cand, frozenset())
                    n_shared = len(genes & cand_genes)
                    if n_shared >= self.min_shared:
                        jacc = n_shared / len(genes | cand_genes)
                        scored.append((cand, jacc))

                scored.sort(key=lambda x: -x[1])
                for cand, jacc in scored[:max_within_edges]:
                    i, j = pid_to_idx[pid], pid_to_idx[cand]
                    key = (min(i, j), max(i, j))
                    if key not in within_edge_set:
                        within_edge_set.add(key)
                        within_edges.append({
                            'src': i, 'dst': j,
                            'weight': jacc,
                            'type': 'within',
                            'tissue_delta': 0.0,
                        })

        print(f"  Within-cancer edges: {len(within_edges):,}", flush=True)

        # --- Cross-cancer edges ---
        # Connect patients with identical gene sets in different cancer types
        print("  Building cross-cancer edges...", flush=True)
        cross_edges = []
        cross_edge_set = set()
        n_cross_exact = 0
        n_cross_overlap = 0

        # Exact gene set matches across cancer types
        for gs, patients_cts in self.gene_set_to_patients.items():
            if len(patients_cts) < 2:
                continue

            # Group by cancer type
            by_ct = defaultdict(list)
            for pid, ct in patients_cts:
                if ct in valid_cts and pid in pid_to_idx:
                    by_ct[ct].append(pid)

            ct_list = list(by_ct.keys())
            if len(ct_list) < 2:
                continue

            for ci in range(len(ct_list)):
                for cj in range(ci + 1, len(ct_list)):
                    ct1, ct2 = ct_list[ci], ct_list[cj]
                    delta = abs(self.tissue_deltas.get(ct1, 0) -
                                self.tissue_deltas.get(ct2, 0))
                    tissue_delta = (self.tissue_deltas.get(ct1, 0) -
                                    self.tissue_deltas.get(ct2, 0))

                    # Connect up to max_cross_edges per cancer-type pair
                    pids1 = by_ct[ct1][:max_cross_edges * 2]
                    pids2 = by_ct[ct2][:max_cross_edges * 2]

                    n_added = 0
                    for p1 in pids1:
                        if n_added >= max_cross_edges * len(pids1):
                            break
                        for p2 in pids2:
                            i, j = pid_to_idx[p1], pid_to_idx[p2]
                            key = (min(i, j), max(i, j))
                            if key not in cross_edge_set and key not in within_edge_set:
                                cross_edge_set.add(key)
                                cross_edges.append({
                                    'src': i, 'dst': j,
                                    'weight': 1.0,  # exact match
                                    'type': 'cross_exact',
                                    'tissue_delta': tissue_delta,
                                    'ct1': ct1, 'ct2': ct2,
                                })
                                n_cross_exact += 1
                                n_added += 1

        # High-overlap matches across cancer types (Jaccard >= threshold)
        # Use inverted index across all cancer types
        global_gene_to_patients = defaultdict(set)
        for pid in all_patients:
            for gene in self.patient_genes.get(pid, frozenset()):
                global_gene_to_patients[gene].add(pid)

        # Sample patients for cross-cancer overlap (full pairwise is too expensive)
        np.random.seed(42)
        sampled = set()
        for ct in valid_cts:
            ct_pids = [p for p in self.cancer_patients[ct] if p in pid_to_idx]
            if len(ct_pids) > 200:
                ct_pids = list(np.random.choice(ct_pids, 200, replace=False))
            sampled.update(ct_pids)

        for pid in sampled:
            genes = self.patient_genes.get(pid, frozenset())
            if not genes:
                continue
            pid_ct = self.patient_cancer.get(pid)

            candidates = set()
            for gene in genes:
                candidates.update(global_gene_to_patients[gene])
            candidates.discard(pid)

            scored = []
            for cand in candidates:
                cand_ct = self.patient_cancer.get(cand)
                if cand_ct == pid_ct:
                    continue  # within-cancer already handled
                if cand_ct not in valid_cts:
                    continue
                cand_genes = self.patient_genes.get(cand, frozenset())
                n_shared = len(genes & cand_genes)
                jacc = n_shared / len(genes | cand_genes)
                if jacc >= min_cross_jaccard:
                    scored.append((cand, jacc))

            scored.sort(key=lambda x: -x[1])
            for cand, jacc in scored[:max_cross_edges]:
                i, j = pid_to_idx[pid], pid_to_idx[cand]
                key = (min(i, j), max(i, j))
                if key not in cross_edge_set and key not in within_edge_set:
                    cross_edge_set.add(key)
                    cand_ct = self.patient_cancer[cand]
                    tissue_delta = (self.tissue_deltas.get(pid_ct, 0) -
                                    self.tissue_deltas.get(cand_ct, 0))
                    cross_edges.append({
                        'src': i, 'dst': j,
                        'weight': jacc,
                        'type': 'cross_overlap',
                        'tissue_delta': tissue_delta,
                        'ct1': pid_ct, 'ct2': cand_ct,
                    })
                    n_cross_overlap += 1

        print(f"  Cross-cancer edges: {len(cross_edges):,} "
              f"(exact={n_cross_exact:,}, overlap={n_cross_overlap:,})", flush=True)

        # --- Node features ---
        node_features = []
        clinical_lookup = self.clinical.set_index('patientId')

        for pid in all_patients:
            genes = self.patient_genes.get(pid, frozenset())
            channels = self.patient_channels.get(pid, frozenset())
            ct = self.patient_cancer.get(pid, '')

            ch_pattern = [1.0 if ch in channels else 0.0 for ch in CHANNEL_NAMES]
            n_genes = len(genes)
            n_hub = sum(1 for g in genes if g in _HUB_SET)

            if pid in clinical_lookup.index:
                row = clinical_lookup.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                time = float(row['OS_MONTHS'])
                event = int(row['event'])
            else:
                time = 0.0
                event = 0

            tissue_delta = self.tissue_deltas.get(ct, 0.0)

            node_features.append({
                'patientId': pid,
                'cancer_type': ct,
                'channel_pattern': ch_pattern,
                'n_genes': n_genes,
                'n_hub': n_hub,
                'hub_ratio': n_hub / max(n_genes, 1),
                'genes': sorted(genes),
                'time': time,
                'event': event,
                'tissue_delta': tissue_delta,
            })

        return {
            'nodes': all_patients,
            'n_nodes': len(all_patients),
            'within_edges': within_edges,
            'cross_edges': cross_edges,
            'n_within': len(within_edges),
            'n_cross': len(cross_edges),
            'node_features': node_features,
            'tissue_intercepts': dict(self.tissue_intercepts),
            'tissue_deltas': dict(self.tissue_deltas),
            'cancer_types': sorted(valid_cts),
        }

    def affinity_stats(self, graph):
        """Compute summary statistics for an affinity graph."""
        n = graph['n_nodes']
        edges = graph['edges']
        n_e = len(edges)

        if n_e == 0:
            return {
                'n_nodes': n, 'n_edges': 0,
                'density': 0, 'mean_weight': 0,
                'mean_degree': 0, 'n_components': n,
            }

        weights = [e[2] for e in edges]
        degrees = defaultdict(int)
        for i, j, w in edges:
            degrees[i] += 1
            degrees[j] += 1

        # Connected components (simple BFS)
        adj = defaultdict(set)
        for i, j, w in edges:
            adj[i].add(j)
            adj[j].add(i)
        visited = set()
        n_components = 0
        for node in range(n):
            if node not in visited:
                n_components += 1
                stack = [node]
                while stack:
                    v = stack.pop()
                    if v in visited:
                        continue
                    visited.add(v)
                    stack.extend(adj[v] - visited)

        return {
            'n_nodes': n,
            'n_edges': n_e,
            'density': 2 * n_e / (n * (n - 1)) if n > 1 else 0,
            'mean_weight': np.mean(weights),
            'median_weight': np.median(weights),
            'mean_degree': np.mean(list(degrees.values())) if degrees else 0,
            'max_degree': max(degrees.values()) if degrees else 0,
            'n_components': n_components,
            'n_isolates': n - len(degrees),
        }


def main():
    """Build unified affinity graph with within- and cross-cancer edges."""
    pag = PatientAffinityGraph()

    # Build unified graph
    ug = pag.build_unified_graph()

    print(f"\n{'='*70}")
    print("UNIFIED AFFINITY GRAPH")
    print(f"{'='*70}")
    print(f"Nodes: {ug['n_nodes']:,}")
    print(f"Within-cancer edges: {ug['n_within']:,}")
    print(f"Cross-cancer edges: {ug['n_cross']:,}")
    print(f"Total edges: {ug['n_within'] + ug['n_cross']:,}")

    # Tissue intercepts
    print(f"\n{'='*70}")
    print("TISSUE INTERCEPTS (median OS delta from global median, deceased only)")
    print(f"{'='*70}")
    deltas = sorted(ug['tissue_deltas'].items(), key=lambda x: x[1])
    for ct, delta in deltas:
        med = ug['tissue_intercepts'][ct]
        print(f"  {ct[:40]:40s} medOS={med:5.1f}  delta={delta:+5.1f}")

    # Cross-cancer edge analysis
    print(f"\n{'='*70}")
    print("CROSS-CANCER EDGE ANALYSIS")
    print(f"{'='*70}")

    # Group cross edges by cancer type pair
    pair_counts = defaultdict(int)
    pair_deltas = defaultdict(list)
    for e in ug['cross_edges']:
        pair = tuple(sorted([e['ct1'], e['ct2']]))
        pair_counts[pair] += 1
        pair_deltas[pair].append(e['tissue_delta'])

    print(f"\nTop cross-cancer connections:")
    print(f"{'Cancer Type 1':30s} {'Cancer Type 2':30s} {'Edges':>6s} {'|Delta|':>7s}")
    print('-' * 80)
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:25]:
        mean_delta = np.mean(np.abs(pair_deltas[pair]))
        print(f"{pair[0][:30]:30s} {pair[1][:30]:30s} {count:6d} {mean_delta:7.1f}")

    # Show specific cross-cancer clusters
    print(f"\n{'='*70}")
    print("CROSS-CANCER GENE SET CLUSTERS (same mutations, different tissue)")
    print(f"{'='*70}")

    nf = ug['node_features']
    # Group patients by gene set
    gene_set_groups = defaultdict(list)
    for i, feat in enumerate(nf):
        gs = ','.join(feat['genes'])
        if gs:
            gene_set_groups[gs].append(i)

    for gs, indices in sorted(gene_set_groups.items(), key=lambda x: -len(x[1]))[:15]:
        if len(indices) < 10:
            continue
        # Group by cancer type
        by_ct = defaultdict(list)
        for i in indices:
            by_ct[nf[i]['cancer_type']].append(i)
        if len(by_ct) < 2:
            continue

        print(f"\n  [{gs}] n={len(indices)}")
        for ct, ct_indices in sorted(by_ct.items(), key=lambda x: -len(x[1])):
            if len(ct_indices) < 3:
                continue
            times = [nf[i]['time'] for i in ct_indices]
            events = [nf[i]['event'] for i in ct_indices]
            delta = pag.tissue_deltas.get(ct, 0)
            print(f"    {ct[:35]:35s} n={len(ct_indices):4d}  medOS={np.median(times):5.1f}  "
                  f"ER={np.mean(events):.2f}  tissue={delta:+5.1f}")


if __name__ == "__main__":
    main()
