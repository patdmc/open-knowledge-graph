"""
Atlas Dataset — encode patients as sets of atlas graph nodes (vectorized).

Each patient's mutations are looked up in the survival atlas.
Each mutation becomes a node with features:
  [log_hr, ci_width, tier, is_hub, channel_onehot(6), is_harmful, is_protective,
   n_patients_with_mutation, normalized_position_in_protein, biallelic,
   expression_z, gof_lof, is_truncating]

Vectorized approach:
  1. Build mutation→node-features lookup from atlas (once)
  2. Merge mutations table with atlas (single pandas join)
  3. Group by patient, stack node features
  4. Pad/truncate to MAX_NODES

Patients without any atlas hits get a single "wild-type" node with log_hr=0.

The atlas sum (simple sum of log HRs) is passed separately as a skip connection
so the transformer can learn the residual beyond the additive model.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, NON_SILENT, MSK_DATASETS,
)

MAX_NODES = 32  # max mutations per patient (pad/truncate)

# --- MUTATION-LEVEL node feature layout (V6) ---
# Gene-level features (is_hub, channel_onehot) ZEROED OUT.
# Predictive signal comes from the mutation itself + graph topology.
#
# [0] log_hr  — per (gene, CT, protein_change) from atlas
# [1] ci_width (confidence — narrower = more certain)
# [2] tier (1, 2, or 3, normalized to 0.33/0.67/1.0)
# [3] ZEROED (was: is_hub — gene-level, not mutation-level)
# [4-9] ZEROED (was: channel_onehot — gene-level identity)
# [10] is_harmful (hr > 1.1)
# [11] is_protective (hr < 0.9)
# [12] log(n_patients) normalized
# [13] normalized protein position (if available)
# [14] biallelic_status — 1.0 if two-hit knockout, 0 otherwise
# [15] expression_z — signed tissue z-score: negative=underexpressed, positive=overexpressed, 0=normal (clipped [-1,1])
# [16] gof_lof — +1 GOF, -1 LOF, 0 unknown
# [17] is_truncating — 1.0 if nonsense/frameshift/splice, 0 if missense, -1 if in-frame indel
NODE_FEAT_DIM = 18

# Legacy 14-dim layout for backward compat
NODE_FEAT_DIM_V3 = 14

# Mutation-only mode: zero out gene-level features
MUTATION_ONLY = True

PROTEIN_LENGTHS = {
    'TP53': 393, 'KRAS': 189, 'BRAF': 766, 'PIK3CA': 1068,
    'PTEN': 403, 'APC': 2843, 'EGFR': 1210, 'BRCA1': 1863,
    'BRCA2': 3418, 'ATM': 3056, 'RB1': 928, 'SMAD4': 552,
    'NF1': 2818, 'CDH1': 882, 'ARID1A': 2285, 'FBXW7': 707,
    'CTNNB1': 781, 'PIK3R1': 724, 'STK11': 433, 'MAP3K1': 1512,
    'ERBB2': 1255, 'FGFR3': 806, 'ESR1': 595, 'AR': 919,
    'GATA3': 443, 'FOXA1': 472, 'MYC': 439, 'CDKN2A': 156,
    'JAK1': 1154, 'JAK2': 1132, 'B2M': 119, 'NOTCH1': 2555,
    'MSH6': 1360, 'MSH2': 934, 'MLH1': 756, 'POLE': 2286,
    'POLD1': 1107, 'NRAS': 189, 'AKT1': 480, 'MTOR': 2549,
    'ERBB3': 1342, 'FGFR2': 821, 'FGFR1': 822, 'MET': 1390,
}

# Pre-compute hub gene set for fast lookup
_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)


def parse_position(pc):
    if not isinstance(pc, str):
        return None
    m = re.search(r'[A-Z*]?(\d+)', pc)
    return int(m.group(1)) if m else None


def get_channel_pos_id(gene):
    """Return 0-11 index: channel_idx * 2 + (0 if hub, 1 if leaf)."""
    ch = CHANNEL_MAP.get(gene)
    if ch is None:
        return 0
    ch_idx = CHANNEL_NAMES.index(ch) if ch in CHANNEL_NAMES else 0
    is_hub = gene in _HUB_SET
    return ch_idx * 2 + (0 if is_hub else 1)


_TRUNCATING_TYPES = {
    'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site',
}
_INFRAME_TYPES = {'In_Frame_Del', 'In_Frame_Ins'}


def make_node_features(gene, pc, hr, ci_width, tier, n_with,
                       biallelic=False, expression_context=None,
                       expression_z=None, gof_lof=0, mutation_type=None):
    """Create a mutation-level node feature vector.

    Gene-level features (is_hub, channel_onehot) are zeroed out when
    MUTATION_ONLY=True. The predictive signal comes from the mutation
    itself and its interactions, not from gene identity.
    """
    feat = np.zeros(NODE_FEAT_DIM, dtype=np.float32)

    # --- Mutation-level features (always on) ---
    feat[0] = np.log(hr)
    feat[1] = min(ci_width, 3.0) / 3.0
    feat[2] = tier / 3.0
    feat[10] = 1.0 if hr > 1.1 else 0.0
    feat[11] = 1.0 if hr < 0.9 else 0.0
    feat[12] = np.log(max(n_with, 1)) / 10.0

    pos = parse_position(pc)
    plen = PROTEIN_LENGTHS.get(gene)
    feat[13] = pos / plen if (pos and plen) else 0.5

    # --- Gene-level features (zeroed in mutation-only mode) ---
    if not MUTATION_ONLY:
        feat[3] = 1.0 if gene in _HUB_SET else 0.0
        ch = CHANNEL_MAP.get(gene)
        if ch and ch in CHANNEL_NAMES:
            feat[4 + CHANNEL_NAMES.index(ch)] = 1.0
    # else: [3] and [4-9] stay zero

    # --- New mutation-level enrichments ---
    feat[14] = 1.0 if biallelic else 0.0

    # Expression z-score: signed, continuous. Negative = underexpressed,
    # positive = overexpressed. Clipped to [-3, 3] then scaled to [-1, 1].
    # Falls back to categorical magnitude if z-score not available.
    if expression_z is not None:
        feat[15] = np.clip(expression_z, -3.0, 3.0) / 3.0
    else:
        _expr_mag_map = {
            'silent': -1.0, 'low': -0.5, 'normal': 0.0,
            'high': 0.5, 'very_high': 0.75, 'overexpressed': 1.0,
        }
        feat[15] = _expr_mag_map.get(expression_context, 0.0)

    # GOF/LOF: +1 gain-of-function, -1 loss-of-function
    feat[16] = float(gof_lof)

    # Mutation type: truncating (+1), missense (0), in-frame indel (-1)
    if mutation_type in _TRUNCATING_TYPES:
        feat[17] = 1.0
    elif mutation_type in _INFRAME_TYPES:
        feat[17] = -1.0
    # else: 0.0 (missense or unknown)

    return feat


class AtlasDataset:

    def __init__(self, dataset_name="msk_impact_50k"):
        paths = MSK_DATASETS[dataset_name]
        print("Loading data...", flush=True)
        self.mutations = pd.read_csv(paths["mutations"])
        self.clinical = pd.read_csv(paths["clinical"])
        self.sample_clinical = pd.read_csv(paths["sample_clinical"])

        self.mutations = self.mutations[self.mutations['mutationType'].isin(NON_SILENT)]
        self.mutations = self.mutations[
            self.mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())
        ]

        # Drop CANCER_TYPE from clinical if present (sample_clinical is authoritative)
        if 'CANCER_TYPE' in self.clinical.columns:
            self.clinical = self.clinical.drop(columns=['CANCER_TYPE'])
        self.clinical = self.clinical.merge(
            self.sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
            on='patientId', how='left'
        )
        self.clinical = self.clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
        self.clinical['event'] = self.clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )
        age_col = None
        for col_name in ['AGE_AT_DX', 'AGE_AT_SEQUENCING', 'AGE_AT_SURGERY']:
            if col_name in self.clinical.columns:
                age_col = col_name
                break
        if age_col:
            self.clinical['age'] = pd.to_numeric(
                self.clinical[age_col], errors='coerce'
            ).fillna(60)
        else:
            self.clinical['age'] = 60.0

        # Load atlas from Neo4j (includes tier 4 graph-imputed entries)
        from gnn.data.graph_snapshot import load_atlas
        print("Loading atlas from Neo4j...", flush=True)
        self.t1, self.t2, self.t3, self.t4, self._atlas_cancer_types = load_atlas()
        print(f"Atlas: T1={len(self.t1)}, T2={len(self.t2)}, "
              f"T3={len(self.t3)}, T4={len(self.t4)}", flush=True)

        self.patients = self.clinical['patientId'].tolist()
        print(f"Patients: {len(self.patients)}", flush=True)

    def _load_mutation_enrichments(self):
        """Load per-patient×gene mutation enrichments from Neo4j.

        Returns:
            biallelic_lookup: {(patient_id, gene)} → True
            expression_lookup: {(patient_id, gene)} → {'ctx': str, 'z': float or None}
            gof_lof_lookup: {gene} → +1 (GOF), -1 (LOF), 0 (unknown)
        """
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))
        biallelic_lookup = set()
        expression_lookup = {}
        gof_lof_lookup = {}

        with driver.session() as s:
            # Biallelic: per patient×gene
            result = s.run("""
                MATCH (p:Patient)-[r:HAS_MUTATION]->(g:Gene)
                WHERE r.biallelic_status = 'biallelic'
                RETURN p.id AS pid, g.name AS gene
            """)
            for r in result:
                biallelic_lookup.add((r["pid"], r["gene"]))
            print(f"  Biallelic enrichments: {len(biallelic_lookup):,}", flush=True)

            # Expression context + z-score: per patient×gene
            result = s.run("""
                MATCH (p:Patient)-[r:HAS_MUTATION]->(g:Gene)
                WHERE r.expression_context IS NOT NULL
                   OR r.expression_tissue_z IS NOT NULL
                RETURN p.id AS pid, g.name AS gene,
                       r.expression_context AS ctx,
                       r.expression_tissue_z AS z
            """)
            for r in result:
                expression_lookup[(r["pid"], r["gene"])] = {
                    'ctx': r["ctx"],
                    'z': float(r["z"]) if r["z"] is not None else None,
                }
            print(f"  Expression enrichments: {len(expression_lookup):,}", flush=True)

            # GOF/LOF: per gene (from gene.function property)
            result = s.run("""
                MATCH (g:Gene) WHERE g.function IS NOT NULL
                RETURN g.name AS gene, g.function AS func
            """)
            for r in result:
                f = r["func"]
                if f in ("oncogene",):
                    gof_lof_lookup[r["gene"]] = 1
                elif f in ("TSG", "likely_TSG"):
                    gof_lof_lookup[r["gene"]] = -1
                else:
                    gof_lof_lookup[r["gene"]] = 0
            print(f"  GOF/LOF classifications: {len(gof_lof_lookup):,}", flush=True)

        driver.close()
        return biallelic_lookup, expression_lookup, gof_lof_lookup

    def build_features(self):
        """Mutation-level feature building.

        When MUTATION_ONLY=True, gene-level features (is_hub, channel_onehot)
        are zeroed. Mutation-level enrichments (biallelic, expression context,
        GOF/LOF) are added. The predictive signal sits with the mutation and
        mutation interactions, not with gene identity.
        """
        print("Encoding patients as mutation-level nodes...", flush=True)
        if MUTATION_ONLY:
            print("  MODE: mutation-only (gene-level features zeroed)", flush=True)

        # Load mutation enrichments from graph
        biallelic_set, expr_lookup, gof_lof = self._load_mutation_enrichments()

        # Step 1: Merge mutations with clinical to get cancer type
        mut_cols = ['patientId', 'gene.hugoGeneSymbol', 'proteinChange']
        if 'mutationType' in self.mutations.columns:
            mut_cols.append('mutationType')
        muts = self.mutations[mut_cols].copy()
        muts.columns = ['patientId', 'gene', 'proteinChange'] + (['mutationType'] if 'mutationType' in self.mutations.columns else [])

        ct_map = self.clinical[['patientId', 'CANCER_TYPE']].drop_duplicates('patientId')
        muts = muts.merge(ct_map, on='patientId', how='inner')
        muts['channel'] = muts['gene'].map(CHANNEL_MAP)

        # Step 2: For each mutation row, find best atlas match (T1 > T2 > T3)
        print("  Matching mutations to atlas tiers...", flush=True)

        node_feats_list = []
        cp_ids_list = []
        patient_ids_list = []
        log_hrs_list = []
        gene_names_list = []

        for idx, row in muts.iterrows():
            ct = row['CANCER_TYPE']
            gene = row['gene']
            pc = row['proteinChange']
            ch = row['channel']
            pid = row['patientId']

            entry = None
            if (ct, gene, pc) in self.t1:
                entry = self.t1[(ct, gene, pc)]
            elif (ct, gene) in self.t2:
                entry = self.t2[(ct, gene)]
            elif ch and (ct, ch) in self.t3:
                entry = self.t3[(ct, ch)]
            elif (ct, gene) in self.t4:
                entry = self.t4[(ct, gene)]

            if entry is not None:
                expr_info = expr_lookup.get((pid, gene), {})
                if isinstance(expr_info, str):
                    # Legacy format: just the context string
                    expr_ctx = expr_info
                    expr_z = None
                elif isinstance(expr_info, dict):
                    expr_ctx = expr_info.get('ctx')
                    expr_z = expr_info.get('z')
                else:
                    expr_ctx = None
                    expr_z = None

                mut_type = row.get('mutationType') if 'mutationType' in muts.columns else None
                feat = make_node_features(
                    gene, pc, entry['hr'], entry.get('ci_width', 1.0),
                    entry['tier'], entry.get('n_with', 50),
                    biallelic=(pid, gene) in biallelic_set,
                    expression_context=expr_ctx,
                    expression_z=expr_z,
                    gof_lof=gof_lof.get(gene, 0),
                    mutation_type=mut_type,
                )
                node_feats_list.append(feat)
                cp_ids_list.append(get_channel_pos_id(gene) if not MUTATION_ONLY else 0)
                patient_ids_list.append(pid)
                log_hrs_list.append(np.log(entry['hr']))
                gene_names_list.append(gene)

        # Step 3: Build per-patient matched mutations dataframe
        print(f"  Atlas-matched mutations: {len(node_feats_list)}", flush=True)

        matched = pd.DataFrame({
            'patientId': patient_ids_list,
            'cp_id': cp_ids_list,
            'log_hr': log_hrs_list,
            'gene_name': gene_names_list,
        })
        # Store features as numpy array indexed same as matched
        matched_feats = np.array(node_feats_list, dtype=np.float32) if node_feats_list else np.empty((0, NODE_FEAT_DIM), dtype=np.float32)

        # Step 4: Group by patient, pad/truncate to MAX_NODES
        print("  Assembling per-patient tensors...", flush=True)

        # Pre-group matched mutations by patient
        patient_groups = {}
        if len(matched) > 0:
            for i, pid in enumerate(matched['patientId'].values):
                if pid not in patient_groups:
                    patient_groups[pid] = []
                patient_groups[pid].append(i)

        # Cancer type mapping
        cancer_type_map = {}
        ct_idx = 0

        all_node_feats = []
        all_node_masks = []
        all_channel_pos_ids = []
        all_atlas_sums = []
        all_gene_names = []
        all_times = []
        all_events = []
        all_cancer_types = []
        all_ages = []
        all_sexes = []

        matched_genes = matched['gene_name'].values if len(matched) > 0 else []

        for _, row in self.clinical.iterrows():
            pid = row['patientId']
            ct = row['CANCER_TYPE']

            indices = patient_groups.get(pid)

            if indices is not None and len(indices) > 0:
                p_feats = matched_feats[indices]
                p_cpids = matched.iloc[indices]['cp_id'].values.tolist()
                p_genes = [matched_genes[i] for i in indices]
                atlas_sum = float(matched.iloc[indices]['log_hr'].sum())
                n_nodes = len(indices)

                if n_nodes > MAX_NODES:
                    # Keep highest |log_hr| nodes
                    abs_loghr = np.abs(p_feats[:, 0])
                    top_idx = np.argsort(abs_loghr)[-MAX_NODES:]
                    p_feats = p_feats[top_idx]
                    p_cpids = [p_cpids[i] for i in top_idx]
                    p_genes = [p_genes[i] for i in top_idx]
                    n_nodes = MAX_NODES

                nodes = list(p_feats)
                cp_ids = list(p_cpids)
                genes = list(p_genes)
                mask = [1] * n_nodes
            else:
                # Wild-type node
                nodes = [np.zeros(NODE_FEAT_DIM, dtype=np.float32)]
                cp_ids = [0]
                genes = ['WT']
                mask = [1]
                atlas_sum = 0.0

            # Pad to MAX_NODES
            while len(nodes) < MAX_NODES:
                nodes.append(np.zeros(NODE_FEAT_DIM, dtype=np.float32))
                cp_ids.append(0)
                genes.append('')
                mask.append(0)

            all_node_feats.append(np.stack(nodes))
            all_node_masks.append(mask)
            all_channel_pos_ids.append(cp_ids)
            all_atlas_sums.append(atlas_sum)
            all_gene_names.append(genes)

            all_times.append(row['OS_MONTHS'])
            all_events.append(row['event'])

            if ct not in cancer_type_map:
                cancer_type_map[ct] = ct_idx
                ct_idx += 1
            all_cancer_types.append(cancer_type_map[ct])

            age = row['age'] if pd.notna(row['age']) else 60.0
            all_ages.append(age)
            sex = row.get('SEX', 'Unknown')
            all_sexes.append(1.0 if sex == 'Male' else 0.0)

        node_feats = torch.tensor(np.stack(all_node_feats), dtype=torch.float32)
        node_masks = torch.tensor(all_node_masks, dtype=torch.float32)
        channel_pos_ids = torch.tensor(all_channel_pos_ids, dtype=torch.long)
        atlas_sums = torch.tensor(all_atlas_sums, dtype=torch.float32).unsqueeze(1)
        times = torch.tensor(all_times, dtype=torch.float32)
        events = torch.tensor(all_events, dtype=torch.long)
        cancer_types = torch.tensor(all_cancer_types, dtype=torch.long)
        ages = torch.tensor(all_ages, dtype=torch.float32)
        sexes = torch.tensor(all_sexes, dtype=torch.float32)

        ages = (ages - ages.mean()) / (ages.std() + 1e-8)

        # Nodes per patient stats
        nodes_per_patient = node_masks.sum(dim=1)
        print(f"Node features: {node_feats.shape}", flush=True)
        print(f"Mean nodes/patient: {nodes_per_patient.mean():.1f}", flush=True)
        print(f"Max nodes/patient: {nodes_per_patient.max():.0f}", flush=True)
        print(f"Patients with 0 atlas nodes: {(nodes_per_patient <= 1).sum().item()}", flush=True)

        return {
            'node_features': node_feats,
            'node_masks': node_masks,
            'channel_pos_ids': channel_pos_ids,
            'atlas_sums': atlas_sums,
            'gene_names': all_gene_names,
            'times': times,
            'events': events,
            'cancer_types': cancer_types,
            'ages': ages,
            'sexes': sexes,
            'n_cancer_types': ct_idx,
            'cancer_type_map': cancer_type_map,
        }


    def build_v5_features(self, schema=None):
        """Build features for V5 hierarchical transformer.

        Extends build_features() with:
          - Enriched node features (atlas + dynamic graph node properties)
          - Pairwise edge features (dynamic, from all graph edge types)
          - Block IDs (sub-pathway community assignments)
          - Channel IDs (channel assignments)

        Args:
            schema: GraphSchema instance (loaded). If None, loads from graph.
        """
        if schema is None:
            from gnn.data.graph_schema import get_schema
            schema = get_schema()

        # First build base features
        base = self.build_features()

        gene_names = base['gene_names']  # list of lists
        B = len(gene_names)
        N = MAX_NODES

        print(f"\nBuilding V5 features (schema: {schema.edge_feature_dim} edge dims, "
              f"{schema.node_extra_dim} node extra dims)...", flush=True)

        # ─── Enriched node features: concatenate graph node props ───
        all_extra = np.zeros((B, N, schema.node_extra_dim), dtype=np.float32)
        all_block_ids = np.full((B, N), schema.n_blocks, dtype=np.int64)  # default = unassigned
        all_channel_ids = np.full((B, N), schema.n_channels, dtype=np.int64)  # default = unassigned

        for b in range(B):
            for n in range(N):
                gene = gene_names[b][n]
                if gene and gene != 'WT' and gene != '':
                    all_extra[b, n] = schema.get_node_extra_features(gene)
                    bid = schema.get_block_id(gene)
                    if bid >= 0:
                        all_block_ids[b, n] = bid
                    cid = schema.get_channel_id(gene)
                    if cid < schema.n_channels:
                        all_channel_ids[b, n] = cid

        # Concatenate atlas features + extra features
        atlas_feats = base['node_features'].numpy()  # (B, N, 14)
        enriched = np.concatenate([atlas_feats, all_extra], axis=-1)  # (B, N, 14 + extra)
        print(f"  Enriched node features: {enriched.shape}", flush=True)

        # ─── Pairwise edge features ───
        all_edges = np.zeros((B, N, N, schema.edge_feature_dim), dtype=np.float32)

        for b in range(B):
            genes = gene_names[b]
            for i in range(N):
                gi = genes[i]
                if not gi or gi == '' or gi == 'WT':
                    continue
                for j in range(i + 1, N):
                    gj = genes[j]
                    if not gj or gj == '' or gj == 'WT':
                        continue
                    feat = schema.get_edge_feature_vector(gi, gj)
                    all_edges[b, i, j] = feat
                    all_edges[b, j, i] = feat  # symmetric

        # Guard: replace any NaN from graph data with 0
        nan_count = np.isnan(all_edges).sum()
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN in edge features, replacing with 0", flush=True)
            all_edges = np.nan_to_num(all_edges, nan=0.0)

        nan_count = np.isnan(all_extra).sum()
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN in node extra features, replacing with 0", flush=True)
            all_extra = np.nan_to_num(all_extra, nan=0.0)
            enriched = np.concatenate([atlas_feats, all_extra], axis=-1)

        print(f"  Edge features: {all_edges.shape}", flush=True)

        # Return extended dict
        result = dict(base)
        result['node_features'] = torch.tensor(enriched, dtype=torch.float32)
        result['edge_features'] = torch.tensor(all_edges, dtype=torch.float32)
        result['block_ids'] = torch.tensor(all_block_ids, dtype=torch.long)
        result['channel_ids'] = torch.tensor(all_channel_ids, dtype=torch.long)
        result['schema'] = schema
        return result


if __name__ == "__main__":
    ds = AtlasDataset()
    data = ds.build_features()
    print("Done.")
