"""
ChannelNet V8 dataset — interaction-aware encoding.

Extends V5's 12-node architecture with:
  1. Per-node hotspot indicators (from V7)
  2. Cross-node interaction features from Cox interaction scan

The interaction features encode the 0/3 taxonomy:
  - Type 0 (counteractive): co-mutation cancels hazard
  - Type 3 (multiplicative): co-mutation compounds hazard

Each interaction is a binary feature: 1 if BOTH mutations in the pair
are present in this patient, 0 otherwise. Weighted by log(HR) so the
model knows direction and magnitude.

Architecture: same 12 nodes, but readout gets an additional interaction
vector encoding which specific cross-channel couplings are active.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    GENE_FUNCTION, NON_SILENT, TRUNCATING,
    MSK_DATASETS, GNN_CACHE, ANALYSIS_CACHE,
)
from gnn.data.channel_dataset_v7 import (
    BASE_FEAT_DIM, discover_hotspots, encode_patient_v7,
)

# ---------------------------------------------------------------------------
# Interaction pairs from Cox scan (p < 0.05, |type| in {0, 3})
# Format: (gene_a, pc_a, gene_b, pc_b, log_hr_interaction)
# ---------------------------------------------------------------------------

INTERACTION_PAIRS = [
    # Cross-channel multiplicative (type 3) — compounding
    ("JAK1", "K860Nfs*16", "ARID1A", "D1850Tfs*33", np.log(4.593)),
    ("JAK1", "K860Nfs*16", "PTEN", "K267Rfs*9", np.log(3.364)),
    ("BRAF", "V600E", "APC", "T1556Nfs*3", np.log(2.498)),
    ("KRAS", "G12V", "TP53", "Y220C", np.log(2.041)),
    ("BRAF", "V600E", "AKT1", "E17K", np.log(1.803)),
    # Cross-channel counteractive (type 0) — cancelling
    ("KRAS", "G12D", "JAK1", "K860Nfs*16", np.log(0.239)),
    ("KRAS", "G13D", "TP53", "R282W", np.log(0.369)),
    # Within-channel counteractive (type 0) — redundancy
    ("KRAS", "G12D", "PIK3CA", "E545K", np.log(0.595)),
    ("KRAS", "G12D", "PIK3CA", "H1047R", np.log(0.523)),
    ("KRAS", "G12D", "PIK3CA", "E542K", np.log(0.607)),
]

N_INTERACTIONS = len(INTERACTION_PAIRS)


# ---------------------------------------------------------------------------
# Encode interaction features for a patient
# ---------------------------------------------------------------------------

def encode_interactions(patient_mutations):
    """
    Return a vector of length N_INTERACTIONS.
    Each entry: 1.0 if both mutations in the pair are present, else 0.0.
    Also return a weight vector (log HR) for scaling.
    """
    present = torch.zeros(N_INTERACTIONS)

    if patient_mutations is None or len(patient_mutations) == 0:
        return present

    # Build set of (gene, proteinChange) for this patient
    patient_muts = set()
    for _, row in patient_mutations.iterrows():
        gene = row['gene.hugoGeneSymbol']
        pc = row.get('proteinChange', '')
        patient_muts.add((gene, pc))

    for i, (ga, pca, gb, pcb, _) in enumerate(INTERACTION_PAIRS):
        if (ga, pca) in patient_muts and (gb, pcb) in patient_muts:
            present[i] = 1.0

    return present


# Precompute weight vector (log HR for each interaction)
INTERACTION_WEIGHTS = torch.tensor(
    [lhr for _, _, _, _, lhr in INTERACTION_PAIRS], dtype=torch.float32
)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class ChannelDatasetV8:

    def __init__(self, dataset_name="msk_impact_50k",
                 min_hotspot_count=50, max_hotspots_per_node=30):
        self.dataset_name = dataset_name

        paths = MSK_DATASETS[dataset_name]
        print(f"Loading {dataset_name}...")
        self.mutations = pd.read_csv(paths["mutations"])
        self.clinical = pd.read_csv(paths["clinical"])
        self.sample_clinical = pd.read_csv(paths["sample_clinical"])

        self.mutations = self.mutations[
            self.mutations['mutationType'].isin(NON_SILENT)
        ]
        self.mutations = self.mutations[
            self.mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())
        ]

        self.clinical = self.clinical.merge(
            self.sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
            on='patientId', how='left'
        )
        self.clinical = self.clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
        self.clinical['event'] = self.clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )

        # Hotspot discovery (same as V7)
        print("Discovering hotspots...")
        self.hotspot_map, self.hotspot_to_idx = discover_hotspots(
            self.mutations, min_hotspot_count, max_hotspots_per_node
        )
        self.n_hotspots_per_node = max(
            len(hs) for hs in self.hotspot_map.values()
        ) if self.hotspot_map else 0
        self.feat_dim = BASE_FEAT_DIM + self.n_hotspots_per_node

        total_hotspots = sum(len(hs) for hs in self.hotspot_map.values())
        print(f"Hotspots: {total_hotspots}, max/node: {self.n_hotspots_per_node}")
        print(f"Node feature dim: {self.feat_dim}")
        print(f"Interaction pairs: {N_INTERACTIONS}")

        self.patient_mutations = dict(list(
            self.mutations.groupby('patientId')
        ))
        self.patients = self.clinical['patientId'].tolist()
        print(f"Patients: {len(self.patients)}")

    def build_features(self):
        print("Encoding patient features + interactions...")
        all_features = []
        all_interactions = []
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

            # Node features (same as V7)
            feat = encode_patient_v7(
                pid, pmuts, self.hotspot_map,
                self.hotspot_to_idx, self.n_hotspots_per_node
            )
            all_features.append(feat)

            # Interaction features (new in V8)
            interact = encode_interactions(pmuts)
            all_interactions.append(interact)

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
        interactions = torch.stack(all_interactions)
        times = torch.tensor(all_times, dtype=torch.float32)
        events = torch.tensor(all_events, dtype=torch.long)
        cancer_types = torch.tensor(all_cancer_types, dtype=torch.long)
        ages = torch.tensor(all_ages, dtype=torch.float32)
        sexes = torch.tensor(all_sexes, dtype=torch.float32)

        ages = (ages - ages.mean()) / (ages.std() + 1e-8)

        # Stats on interaction feature activation
        n_with_any = (interactions.sum(dim=1) > 0).sum().item()
        print(f"Features shape: {features.shape}")
        print(f"Interactions shape: {interactions.shape}")
        print(f"Patients with ≥1 interaction: {n_with_any} ({n_with_any/len(events)*100:.1f}%)")
        for i, (ga, pca, gb, pcb, lhr) in enumerate(INTERACTION_PAIRS):
            n = (interactions[:, i] > 0).sum().item()
            if n > 0:
                direction = "×" if lhr > 0 else "÷"
                print(f"  {ga} {pca} {direction} {gb} {pcb}: {n} patients")

        return {
            'features': features,
            'interactions': interactions,
            'interaction_weights': INTERACTION_WEIGHTS,
            'times': times,
            'events': events,
            'cancer_types': cancer_types,
            'ages': ages,
            'sexes': sexes,
            'n_cancer_types': ct_idx,
            'feat_dim': self.feat_dim,
            'n_interactions': N_INTERACTIONS,
            'hotspot_map': self.hotspot_map,
            'cancer_type_map': cancer_type_map,
        }


if __name__ == "__main__":
    ds = ChannelDatasetV8()
    data = ds.build_features()
    cache_path = os.path.join(GNN_CACHE, "channel_v8_data.pt")
    torch.save(data, cache_path)
    print(f"Saved to {cache_path}")
