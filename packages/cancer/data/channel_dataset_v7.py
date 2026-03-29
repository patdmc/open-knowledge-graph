"""
ChannelNet V7 dataset — hotspot mutation encoding.

Extends V5's 12-node (6 channel × 2 position) architecture with
per-mutation hotspot features. Instead of just "is this gene mutated",
each node encodes WHICH specific mutations are present.

Architecture:
  - 12 nodes (6 channels × hub/leaf)
  - Per node: base features (V5) + hotspot binary indicators
  - Hotspots: recurrent mutations with >= MIN_HOTSPOT_COUNT patients
  - Each hotspot is a binary feature: 1 if that specific amino acid
    change is present in this patient, 0 otherwise

Why this matters:
  TP53 R175H (+5.9% death rate) and TP53 G245S (-1.0%) are both
  "TP53 mutations" but carry completely different prognostic weight.
  KRAS G12D (+10.5%) and KRAS G13D (-3.1%) — same gene, opposite
  direction. The gene-level model averages over this signal.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    GENE_FUNCTION, NON_SILENT, TRUNCATING,
    MSK_DATASETS, GNN_CACHE, ANALYSIS_CACHE,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_HOTSPOT_COUNT = 50  # minimum patients for a mutation to be a hotspot
MAX_HOTSPOTS_PER_NODE = 30  # cap per node to keep features manageable

# Base feature dimension (same as V5: counts, VAF, GOF/LOF balance)
BASE_FEAT_DIM = 16

# ---------------------------------------------------------------------------
# Hotspot discovery
# ---------------------------------------------------------------------------

def discover_hotspots(mutations_df, min_count=MIN_HOTSPOT_COUNT,
                      max_per_node=MAX_HOTSPOTS_PER_NODE):
    """
    Find recurrent mutations per channel-position node.

    Returns:
        hotspot_map: dict of (channel, position) -> list of (gene, proteinChange)
        hotspot_to_idx: dict of (channel, position) -> {(gene, pc): idx}
    """
    all_hubs = {g for genes in HUB_GENES.values() for g in genes}
    all_leaves = {g for genes in LEAF_GENES.values() for g in genes}

    hotspot_map = {}
    hotspot_to_idx = {}

    for channel in CHANNEL_NAMES:
        for pos, gene_set in [("hub", HUB_GENES.get(channel, set())),
                               ("leaf", LEAF_GENES.get(channel, set()))]:
            if not gene_set:
                hotspot_map[(channel, pos)] = []
                hotspot_to_idx[(channel, pos)] = {}
                continue

            mask = mutations_df['gene.hugoGeneSymbol'].isin(gene_set)
            if mask.sum() == 0:
                hotspot_map[(channel, pos)] = []
                hotspot_to_idx[(channel, pos)] = {}
                continue

            counts = (mutations_df[mask]
                      .groupby(['gene.hugoGeneSymbol', 'proteinChange'])
                      ['patientId'].nunique()
                      .reset_index())
            counts.columns = ['gene', 'pc', 'n']
            counts = counts[counts['n'] >= min_count]
            counts = counts.sort_values('n', ascending=False)
            counts = counts.head(max_per_node)

            hotspots = list(zip(counts['gene'], counts['pc']))
            hotspot_map[(channel, pos)] = hotspots
            hotspot_to_idx[(channel, pos)] = {
                (g, pc): i for i, (g, pc) in enumerate(hotspots)
            }

    return hotspot_map, hotspot_to_idx


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_patient_v7(patient_id, patient_mutations, hotspot_map,
                      hotspot_to_idx, n_hotspots_per_node):
    """
    Encode a single patient as a 12-node feature matrix.

    Each node has: base_features (16) + hotspot_indicators (n_hotspots_per_node)

    Returns: tensor of shape (12, BASE_FEAT_DIM + n_hotspots_per_node)
    """
    feat_dim = BASE_FEAT_DIM + n_hotspots_per_node
    features = torch.zeros(12, feat_dim)

    # Node ordering: DDR_hub, DDR_leaf, CellCycle_hub, CellCycle_leaf, ...
    node_idx = 0
    for channel in CHANNEL_NAMES:
        for pos_name, gene_set in [("hub", HUB_GENES.get(channel, set())),
                                    ("leaf", LEAF_GENES.get(channel, set()))]:
            if patient_mutations is not None:
                node_muts = patient_mutations[
                    patient_mutations['gene.hugoGeneSymbol'].isin(gene_set)
                ]
            else:
                node_muts = pd.DataFrame()

            # --- Base features (same as V5) ---
            n_muts = len(node_muts)
            n_trunc = node_muts['mutationType'].isin(TRUNCATING).sum() if n_muts > 0 else 0

            if n_muts > 0 and 'tumorAltCount' in node_muts.columns:
                alt = pd.to_numeric(node_muts['tumorAltCount'], errors='coerce').fillna(0)
                ref = pd.to_numeric(node_muts.get('tumorRefCount', 0), errors='coerce').fillna(0)
                total = (alt + ref).clip(lower=1)
                vaf = alt / total
                max_vaf = float(vaf.max()) if len(vaf) > 0 else 0.0
                mean_vaf = float(vaf.mean()) if len(vaf) > 0 else 0.0
            else:
                max_vaf = 0.0
                mean_vaf = 0.0

            # GOF/LOF balance
            n_gof = 0
            n_lof = 0
            for _, m in node_muts.iterrows():
                gene = m['gene.hugoGeneSymbol']
                func = GENE_FUNCTION.get(gene, 'context')
                mtype = m['mutationType']
                if mtype in TRUNCATING:
                    n_lof += 1
                elif func == 'GOF':
                    n_gof += 1
                elif func == 'LOF':
                    n_lof += 1

            gof_frac = n_gof / max(n_gof + n_lof, 1)

            features[node_idx, 0] = n_muts
            features[node_idx, 1] = n_trunc
            features[node_idx, 2] = max_vaf
            features[node_idx, 3] = mean_vaf
            features[node_idx, 4] = gof_frac
            features[node_idx, 5] = 1.0 if n_muts > 0 else 0.0  # any mutation
            features[node_idx, 6] = min(n_muts / 5.0, 1.0)  # normalized count
            # features[7:BASE_FEAT_DIM] reserved for additional base features

            # --- Hotspot features ---
            hs_idx_map = hotspot_to_idx.get((channel, pos_name), {})
            for _, m in node_muts.iterrows():
                gene = m['gene.hugoGeneSymbol']
                pc = m.get('proteinChange', '')
                key = (gene, pc)
                if key in hs_idx_map:
                    hs_offset = BASE_FEAT_DIM + hs_idx_map[key]
                    features[node_idx, hs_offset] = 1.0

            node_idx += 1

    return features


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class ChannelDatasetV7:
    """
    Builds patient feature tensors with hotspot mutation encoding.
    """

    def __init__(self, dataset_name="msk_impact_50k", min_hotspot_count=MIN_HOTSPOT_COUNT,
                 max_hotspots_per_node=MAX_HOTSPOTS_PER_NODE):
        self.dataset_name = dataset_name
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots_per_node = max_hotspots_per_node

        # Load data
        paths = MSK_DATASETS[dataset_name]
        print(f"Loading {dataset_name}...")
        self.mutations = pd.read_csv(paths["mutations"])
        self.clinical = pd.read_csv(paths["clinical"])
        self.sample_clinical = pd.read_csv(paths["sample_clinical"])

        # Filter non-silent
        self.mutations = self.mutations[
            self.mutations['mutationType'].isin(NON_SILENT)
        ]

        # Filter to channel genes
        self.mutations = self.mutations[
            self.mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())
        ]

        # Merge clinical
        self.clinical = self.clinical.merge(
            self.sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
            on='patientId', how='left'
        )
        self.clinical = self.clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
        self.clinical['event'] = self.clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )

        # Discover hotspots
        print("Discovering hotspots...")
        self.hotspot_map, self.hotspot_to_idx = discover_hotspots(
            self.mutations, min_hotspot_count, max_hotspots_per_node
        )

        # Compute max hotspots across all nodes for uniform feature dim
        self.n_hotspots_per_node = max(
            len(hs) for hs in self.hotspot_map.values()
        ) if self.hotspot_map else 0

        self.feat_dim = BASE_FEAT_DIM + self.n_hotspots_per_node

        total_hotspots = sum(len(hs) for hs in self.hotspot_map.values())
        print(f"Found {total_hotspots} hotspots across 12 nodes")
        print(f"Max hotspots per node: {self.n_hotspots_per_node}")
        print(f"Feature dim per node: {self.feat_dim}")

        # Group mutations by patient for fast lookup
        self.patient_mutations = dict(list(
            self.mutations.groupby('patientId')
        ))

        # Patient list
        self.patients = self.clinical['patientId'].tolist()
        print(f"Patients: {len(self.patients)}")

    def build_features(self):
        """Build feature tensor for all patients."""
        print("Encoding patient features...")
        all_features = []
        all_times = []
        all_events = []
        all_cancer_types = []
        all_ages = []
        all_sexes = []

        cancer_type_map = {}
        ct_idx = 0

        for i, row in self.clinical.iterrows():
            pid = row['patientId']
            pmuts = self.patient_mutations.get(pid, None)

            feat = encode_patient_v7(
                pid, pmuts, self.hotspot_map,
                self.hotspot_to_idx, self.n_hotspots_per_node
            )
            all_features.append(feat)
            all_times.append(row['OS_MONTHS'])
            all_events.append(row['event'])

            ct = row.get('CANCER_TYPE', 'Unknown')
            if ct not in cancer_type_map:
                cancer_type_map[ct] = ct_idx
                ct_idx += 1
            all_cancer_types.append(cancer_type_map[ct])

            age = row.get('AGE_AT_DX', 60)
            try:
                age = float(age)
            except (ValueError, TypeError):
                age = 60.0
            all_ages.append(age if pd.notna(age) else 60.0)

            sex = row.get('SEX', 'Unknown')
            all_sexes.append(1.0 if sex == 'Male' else 0.0)

        features = torch.nan_to_num(torch.stack(all_features), nan=0.0)
        times = torch.tensor(all_times, dtype=torch.float32)
        events = torch.tensor(all_events, dtype=torch.long)
        cancer_types = torch.tensor(all_cancer_types, dtype=torch.long)
        ages = torch.tensor(all_ages, dtype=torch.float32)
        sexes = torch.tensor(all_sexes, dtype=torch.float32)

        # Normalize age
        ages = (ages - ages.mean()) / (ages.std() + 1e-8)

        print(f"Features shape: {features.shape}")
        print(f"Cancer types: {ct_idx}")

        return {
            'features': features,
            'times': times,
            'events': events,
            'cancer_types': cancer_types,
            'ages': ages,
            'sexes': sexes,
            'n_cancer_types': ct_idx,
            'feat_dim': self.feat_dim,
            'hotspot_map': self.hotspot_map,
            'cancer_type_map': cancer_type_map,
        }

    def save_hotspot_map(self, path=None):
        """Save hotspot map for interpretation."""
        if path is None:
            path = os.path.join(GNN_CACHE, "hotspot_map_v7.json")

        serializable = {}
        for (channel, pos), hotspots in self.hotspot_map.items():
            key = f"{channel}_{pos}"
            serializable[key] = [
                {"gene": g, "protein_change": pc} for g, pc in hotspots
            ]

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved hotspot map to {path}")


if __name__ == "__main__":
    ds = ChannelDatasetV7()
    data = ds.build_features()
    ds.save_hotspot_map()

    # Save cached tensors
    cache_path = os.path.join(GNN_CACHE, "channel_v7_data.pt")
    torch.save(data, cache_path)
    print(f"Saved to {cache_path}")
