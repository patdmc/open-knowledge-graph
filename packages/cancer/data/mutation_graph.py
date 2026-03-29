"""
Embedded mutation graph — holds the full biological graph in memory.

Node types:
  - Gene: 92 genes with channel, function (GOF/LOF/context), position (hub/leaf)
  - Channel: 6 coupling channels
  - Patient: ~44K patients with cancer_type, survival, mutations

Edge types:
  - BELONGS_TO: Gene -> Channel
  - COUPLES: Gene <-> Gene (within or cross channel, carries direction pair type)
  - HAS_MUTATION: Patient -> Gene (carries atlas_hr, direction)
  - AFFINITY: Patient <-> Patient (Jaccard similarity, within/cross cancer)

The graph is a plain networkx MultiDiGraph. All walks are dict lookups.

Usage:
    g = MutationGraph()
    g.build()
    features = g.walk_patient('P-0000001')
    df = g.walk_all_patients()
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    HUB_GENES, LEAF_GENES, GENE_FUNCTION, GENE_POSITION,
    NON_SILENT, TRUNCATING, MSK_DATASETS, GNN_CACHE,
    ALL_GENES,
)


_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)


class MutationGraph:
    """In-memory mutation graph. All lookups are O(1) dict access."""

    def __init__(self, dataset_name="msk_impact_50k"):
        self.dataset_name = dataset_name
        self.G = nx.MultiDiGraph()
        self._loaded = False

    def build(self, min_jaccard=0.3, max_neighbors=20):
        """Build the full graph: genes, channels, patients, edges."""
        t0 = time.time()

        paths = MSK_DATASETS[self.dataset_name]
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
        clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS', 'CANCER_TYPE'])
        clinical['event'] = clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )

        # --- Load atlas from Neo4j ---
        from gnn.data.graph_snapshot import load_atlas
        t1_raw, t2_raw, t3_raw, t4_raw, _ = load_atlas()

        self._t1, self._t2, self._t3, self._t4 = {}, {}, {}, {}
        for key, entry in t1_raw.items():
            self._t1[key] = np.log(max(entry['hr'], 0.01))
        for key, entry in t2_raw.items():
            self._t2[key] = np.log(max(entry['hr'], 0.01))
        for key, entry in t3_raw.items():
            self._t3[key] = np.log(max(entry['hr'], 0.01))
        for key, entry in t4_raw.items():
            self._t4[key] = np.log(max(entry['hr'], 0.01))

        # --- Tissue intercepts ---
        deceased = clinical[clinical['event'] == 1]
        global_median = deceased['OS_MONTHS'].median()
        ct_medians = deceased.groupby('CANCER_TYPE')['OS_MONTHS'].median()
        self._tissue_deltas = (ct_medians - global_median).to_dict()

        # =====================================================================
        # 1. Channel nodes
        # =====================================================================
        print("Building channel nodes...", flush=True)
        for ch in CHANNEL_NAMES:
            self.G.add_node(f"CH:{ch}", ntype='channel', name=ch,
                            idx=CHANNEL_TO_IDX[ch])

        # =====================================================================
        # 2. Gene nodes + BELONGS_TO edges
        # =====================================================================
        print("Building gene nodes...", flush=True)
        for gene in ALL_GENES:
            ch = CHANNEL_MAP[gene]
            func = GENE_FUNCTION.get(gene, 'context')
            pos = GENE_POSITION.get(gene, 'unclassified')
            self.G.add_node(f"G:{gene}", ntype='gene', name=gene,
                            channel=ch, function=func, position=pos,
                            is_hub=gene in _HUB_SET)
            self.G.add_edge(f"G:{gene}", f"CH:{ch}", etype='BELONGS_TO')

        # =====================================================================
        # 3. COUPLES edges (gene-gene within and cross channel)
        # =====================================================================
        print("Building coupling edges...", flush=True)
        n_couples = 0
        for a, b in combinations(ALL_GENES, 2):
            ch_a, ch_b = CHANNEL_MAP[a], CHANNEL_MAP[b]
            func_a = GENE_FUNCTION.get(a, 'context')
            func_b = GENE_FUNCTION.get(b, 'context')
            locality = 'within' if ch_a == ch_b else 'cross'

            # Determine pair direction type
            dirs = sorted([func_a, func_b])
            if dirs == ['GOF', 'LOF']:
                pair_type = 'GOF_LOF'
            elif dirs == ['GOF', 'GOF']:
                pair_type = 'GOF_GOF'
            elif dirs == ['LOF', 'LOF']:
                pair_type = 'LOF_LOF'
            else:
                pair_type = 'other'

            self.G.add_edge(f"G:{a}", f"G:{b}", etype='COUPLES',
                            locality=locality, pair_type=pair_type,
                            channels=(ch_a, ch_b))
            n_couples += 1

        print(f"  Coupling edges: {n_couples}", flush=True)

        # =====================================================================
        # 4. Patient nodes + HAS_MUTATION edges
        # =====================================================================
        print("Building patient nodes...", flush=True)
        gene_muts = mutations[['patientId', 'gene.hugoGeneSymbol', 'proteinChange',
                                'mutationType']].copy()
        gene_muts.columns = ['patientId', 'gene', 'proteinChange', 'mutationType']

        ct_map = clinical[['patientId', 'CANCER_TYPE']].drop_duplicates('patientId')
        ct_dict = dict(zip(ct_map['patientId'], ct_map['CANCER_TYPE']))

        # Build per-patient gene sets for affinity later
        patient_genes = defaultdict(set)

        n_patients = 0
        n_mut_edges = 0

        for _, row in clinical.iterrows():
            pid = row['patientId']
            ct = row['CANCER_TYPE']
            self.G.add_node(f"P:{pid}", ntype='patient', name=pid,
                            cancer_type=ct,
                            os_months=row['OS_MONTHS'],
                            event=int(row['event']),
                            tissue_delta=self._tissue_deltas.get(ct, 0.0))
            n_patients += 1

        # Add mutation edges
        pid_muts = gene_muts[gene_muts['patientId'].isin(ct_dict)].copy()
        # Deduplicate: one edge per (patient, gene)
        pid_muts = pid_muts.drop_duplicates(subset=['patientId', 'gene'])

        for _, m in pid_muts.iterrows():
            pid, gene = m['patientId'], m['gene']
            ct = ct_dict.get(pid, '')
            func = GENE_FUNCTION.get(gene, 'context')

            # Context genes: truncating -> LOF, else -> GOF
            if func == 'context':
                direction = 'LOF' if m['mutationType'] in TRUNCATING else 'GOF'
            else:
                direction = func

            log_hr = self._atlas_log_hr(ct, gene, m['proteinChange'])

            self.G.add_edge(f"P:{pid}", f"G:{gene}", etype='HAS_MUTATION',
                            direction=direction, log_hr=log_hr,
                            protein_change=m['proteinChange'])
            patient_genes[pid].add(gene)
            n_mut_edges += 1

        print(f"  Patient nodes: {n_patients}", flush=True)
        print(f"  Mutation edges: {n_mut_edges}", flush=True)

        # =====================================================================
        # 5. Affinity index (lazy — computed per-patient during walk)
        # =====================================================================
        # Instead of adding 800K+ edges to NetworkX (slow), we keep an
        # inverted index and compute Jaccard on the fly per patient.
        # For 20 neighbors this is ~microseconds.
        print("Building affinity index...", flush=True)

        self._gene_to_patients = defaultdict(set)
        valid_pids = {pid for pid in patient_genes if f"P:{pid}" in self.G}
        for pid, genes in patient_genes.items():
            if pid in valid_pids:
                for gene in genes:
                    self._gene_to_patients[gene].add(pid)

        print(f"  Indexed {len(self._gene_to_patients)} genes "
              f"across {len(valid_pids)} patients", flush=True)

        # Store patient_genes and config for affinity lookups
        self._patient_genes = {pid: frozenset(g) for pid, g in patient_genes.items()}
        self._ct_dict = ct_dict
        self._affinity_config = {
            'min_jaccard': min_jaccard,
            'max_neighbors': max_neighbors,
        }
        self._loaded = True

        elapsed = time.time() - t0
        print(f"\nGraph built in {elapsed:.1f}s", flush=True)
        print(f"  Total nodes: {self.G.number_of_nodes()}", flush=True)
        print(f"  Total edges: {self.G.number_of_edges()}", flush=True)

    def _atlas_log_hr(self, cancer_type, gene, protein_change):
        """Tiered atlas lookup: T1 > T2 > T3 > T4."""
        key1 = (cancer_type, gene, protein_change)
        if key1 in self._t1:
            return self._t1[key1]
        key2 = (cancer_type, gene)
        if key2 in self._t2:
            return self._t2[key2]
        ch = CHANNEL_MAP.get(gene)
        if ch:
            key3 = (cancer_type, ch)
            if key3 in self._t3:
                return self._t3[key3]
        if key2 in self._t4:
            return self._t4[key2]
        return 0.0

    # -----------------------------------------------------------------
    # Walk queries — all dict lookups, no iteration over full graph
    # -----------------------------------------------------------------

    def walk_patient(self, patient_id):
        """Walk one patient's mutations through the graph.

        Returns dict of features — all computed from edge traversal.
        """
        pid_node = f"P:{patient_id}"
        if pid_node not in self.G:
            return None

        pdata = self.G.nodes[pid_node]
        rec = {
            'patientId': patient_id,
            'cancer_type': pdata['cancer_type'],
            'OS_MONTHS': pdata['os_months'],
            'event': pdata['event'],
            'tissue_delta': pdata['tissue_delta'],
        }

        # Step 1: traverse HAS_MUTATION edges to get this patient's genes
        mut_edges = [(u, v, d) for u, v, d in self.G.out_edges(pid_node, data=True)
                     if d.get('etype') == 'HAS_MUTATION']

        if not mut_edges:
            rec.update(self._empty_features())
            return rec

        # Collect per-gene info from the graph
        gene_info = []
        atlas_sum = 0.0
        for _, gene_node, edata in mut_edges:
            gdata = self.G.nodes[gene_node]
            direction = edata['direction']
            log_hr = edata['log_hr']
            atlas_sum += log_hr

            gene_info.append({
                'gene': gdata['name'],
                'channel': gdata['channel'],
                'direction': direction,
                'is_hub': gdata['is_hub'],
                'log_hr': log_hr,
            })

        rec['atlas_sum'] = atlas_sum
        rec['n_genes'] = len(gene_info)

        # Direction counts
        rec['n_gof'] = sum(1 for g in gene_info if g['direction'] == 'GOF')
        rec['n_lof'] = sum(1 for g in gene_info if g['direction'] == 'LOF')

        # Per-channel direction flags
        for ch in CHANNEL_NAMES:
            rec[f'ch_GOF_{ch}'] = 0
            rec[f'ch_LOF_{ch}'] = 0

        channels_hit = set()
        n_hub = 0
        for g in gene_info:
            ch = g['channel']
            channels_hit.add(ch)
            if g['direction'] == 'GOF':
                rec[f'ch_GOF_{ch}'] = 1
            elif g['direction'] == 'LOF':
                rec[f'ch_LOF_{ch}'] = 1
            if g['is_hub']:
                n_hub += 1

        rec['n_channels_severed'] = len(channels_hit)
        rec['n_hub_hit'] = n_hub
        rec['hub_ratio'] = n_hub / max(len(gene_info), 1)
        total_dir = rec['n_gof'] + rec['n_lof']
        rec['gof_lof_ratio'] = rec['n_gof'] / max(total_dir, 1)

        # Step 2: traverse COUPLES edges between this patient's genes
        pair_counts = defaultdict(int)
        cross_gof_lof_channels = set()

        for a, b in combinations(gene_info, 2):
            same_ch = a['channel'] == b['channel']
            locality = 'within' if same_ch else 'cross'

            dirs = sorted([a['direction'], b['direction']])
            if dirs == ['GOF', 'LOF']:
                pair_type = 'GOF_LOF'
                if not same_ch:
                    cross_gof_lof_channels.add(
                        tuple(sorted([a['channel'], b['channel']])))
            elif dirs == ['GOF', 'GOF']:
                pair_type = 'GOF_GOF'
            elif dirs == ['LOF', 'LOF']:
                pair_type = 'LOF_LOF'
            else:
                pair_type = 'other'

            pair_counts[f'{pair_type}_{locality}'] += 1

        for pt in ['GOF_LOF_within', 'GOF_LOF_cross',
                    'GOF_GOF_within', 'GOF_GOF_cross',
                    'LOF_LOF_within', 'LOF_LOF_cross']:
            rec[pt] = pair_counts.get(pt, 0)

        rec['cross_gof_lof_channels'] = len(cross_gof_lof_channels)

        # Specific pathway flags
        rec['path_PI3K_GOF_x_CC_LOF'] = int(
            rec['ch_GOF_PI3K_Growth'] and rec['ch_LOF_CellCycle'])
        rec['path_PI3K_GOF_x_TA_LOF'] = int(
            rec['ch_GOF_PI3K_Growth'] and rec['ch_LOF_TissueArch'])
        rec['path_CC_GOF_x_DDR_LOF'] = int(
            rec['ch_GOF_CellCycle'] and rec['ch_LOF_DDR'])
        rec['path_anyGOF_x_Immune_LOF'] = int(
            rec['n_gof'] > 0 and rec['ch_LOF_Immune'])

        # Step 3: affinity neighbors (lazy Jaccard from inverted index)
        genes = self._patient_genes.get(patient_id, frozenset())
        min_jacc = self._affinity_config['min_jaccard']
        max_nb = self._affinity_config['max_neighbors']

        if genes and hasattr(self, '_gene_to_patients'):
            candidates = set()
            for gene in genes:
                candidates.update(self._gene_to_patients.get(gene, set()))
            candidates.discard(patient_id)

            scored = []
            for cand in candidates:
                cand_genes = self._patient_genes.get(cand, frozenset())
                jacc = len(genes & cand_genes) / len(genes | cand_genes)
                if jacc >= min_jacc:
                    scored.append(jacc)

            scored.sort(reverse=True)
            neighbors = scored[:max_nb]
            rec['n_neighbors'] = len(neighbors)
            rec['mean_jacc'] = np.mean(neighbors) if neighbors else 0.0
        else:
            rec['n_neighbors'] = 0
            rec['mean_jacc'] = 0.0

        return rec

    def _empty_features(self):
        """Default features for patients with no mutations."""
        rec = {
            'atlas_sum': 0.0, 'n_genes': 0,
            'n_gof': 0, 'n_lof': 0,
            'n_channels_severed': 0, 'n_hub_hit': 0,
            'hub_ratio': 0.0, 'gof_lof_ratio': 0.0,
            'n_neighbors': 0, 'mean_jacc': 0.0,
            'cross_gof_lof_channels': 0,
            'path_PI3K_GOF_x_CC_LOF': 0, 'path_PI3K_GOF_x_TA_LOF': 0,
            'path_CC_GOF_x_DDR_LOF': 0, 'path_anyGOF_x_Immune_LOF': 0,
        }
        for ch in CHANNEL_NAMES:
            rec[f'ch_GOF_{ch}'] = 0
            rec[f'ch_LOF_{ch}'] = 0
        for pt in ['GOF_LOF_within', 'GOF_LOF_cross',
                    'GOF_GOF_within', 'GOF_GOF_cross',
                    'LOF_LOF_within', 'LOF_LOF_cross']:
            rec[pt] = 0
        return rec

    def walk_all_patients(self):
        """Walk all patients. Returns DataFrame."""
        print("Walking all patients...", flush=True)
        t0 = time.time()

        records = []
        patient_nodes = [n for n, d in self.G.nodes(data=True)
                         if d.get('ntype') == 'patient']

        for i, pid_node in enumerate(patient_nodes):
            pid = self.G.nodes[pid_node]['name']
            rec = self.walk_patient(pid)
            if rec:
                records.append(rec)

            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(patient_nodes)}", flush=True)

        df = pd.DataFrame(records)
        elapsed = time.time() - t0
        print(f"Walk complete: {len(df)} patients in {elapsed:.1f}s", flush=True)
        return df

    def save(self, path=None):
        """Serialize graph to disk."""
        if path is None:
            path = os.path.join(GNN_CACHE, "mutation_graph.pkl")
        with open(path, 'wb') as f:
            pickle.dump({
                'G': self.G,
                '_t1': self._t1, '_t2': self._t2, '_t3': self._t3,
                '_tissue_deltas': self._tissue_deltas,
                '_patient_genes': self._patient_genes,
                '_ct_dict': self._ct_dict,
                '_gene_to_patients': dict(self._gene_to_patients),
                '_affinity_config': self._affinity_config,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Graph saved to {path} ({size_mb:.1f} MB)", flush=True)

    def load(self, path=None):
        """Load graph from disk."""
        if path is None:
            path = os.path.join(GNN_CACHE, "mutation_graph.pkl")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.G = data['G']
        self._t1 = data['_t1']
        self._t2 = data['_t2']
        self._t3 = data['_t3']
        self._tissue_deltas = data['_tissue_deltas']
        self._patient_genes = data['_patient_genes']
        self._ct_dict = data['_ct_dict']
        self._gene_to_patients = defaultdict(set, data.get('_gene_to_patients', {}))
        self._affinity_config = data.get('_affinity_config',
                                          {'min_jaccard': 0.3, 'max_neighbors': 20})
        self._loaded = True
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges", flush=True)

    def stats(self):
        """Print graph statistics."""
        ntypes = defaultdict(int)
        for _, d in self.G.nodes(data=True):
            ntypes[d.get('ntype', 'unknown')] += 1

        etypes = defaultdict(int)
        for _, _, d in self.G.edges(data=True):
            etypes[d.get('etype', 'unknown')] += 1

        print(f"\n{'='*50}")
        print("MUTATION GRAPH STATS")
        print(f"{'='*50}")
        print(f"Nodes: {self.G.number_of_nodes()}")
        for nt, count in sorted(ntypes.items()):
            print(f"  {nt}: {count}")
        print(f"Edges: {self.G.number_of_edges()}")
        for et, count in sorted(etypes.items()):
            print(f"  {et}: {count}")


def main():
    from sklearn.model_selection import StratifiedKFold
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as ci

    mg = MutationGraph()
    mg.build()
    mg.stats()
    mg.save()

    # Walk all patients
    df = mg.walk_all_patients()
    df = df[df['OS_MONTHS'] > 0].copy()

    # Standardize
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ('OS_MONTHS', 'event'):
            continue
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    features = [
        'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
        'GOF_LOF_within', 'GOF_LOF_cross',
        'GOF_GOF_cross', 'LOF_LOF_within',
        'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
        'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
        'n_neighbors', 'mean_jacc',
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    valid = [f for f in features if f in df.columns and df[f].std() > 0.001]
    c_indices = []
    for fold, (ti, vi) in enumerate(skf.split(df, df['event'])):
        train, val = df.iloc[ti], df.iloc[vi]
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(train[valid + ['OS_MONTHS', 'event']],
                duration_col='OS_MONTHS', event_col='event')
        h = cph.predict_partial_hazard(val[valid]).values.flatten()
        c_indices.append(ci(val['OS_MONTHS'].values, -h, val['event'].values))

    print(f"\nGraph walk C-index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
    print(f"Per-fold: {[f'{c:.4f}' for c in c_indices]}")
    print(f"\nBASELINES:")
    print(f"  Atlas lookup (zero-param):    C = 0.577")
    print(f"  Directional walk (no graph):  C = 0.636")
    print(f"  AtlasTransformer V1 (neural): C = 0.673")


if __name__ == "__main__":
    main()
