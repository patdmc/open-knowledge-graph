"""
Channel dataset v5: hub/leaf split — 12 nodes (6 channels × 2 positions).

Per node features (12-dim):
  [0]  is_severed (any mutation in this channel-position)
  [1]  n_mutations (log-scaled)
  [2]  n_genes
  [3]  frac_truncating
  [4]  frac_missense
  [5]  mean_vaf
  [6]  max_vaf
  [7]  gof_count
  [8]  lof_count
  [9]  is_hub (1 for hub nodes, 0 for leaf nodes)
  [10:16] channel one-hot (6-dim)

Total: 16 dims per node, 12 nodes = 192 raw features per patient.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    HUB_GENES, LEAF_GENES, GENE_FUNCTION, GENE_POSITION,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
    GNN_CACHE,
)

HUBLF_FEAT_DIM = 16
N_NODES = 12  # 6 channels × 2 (hub, leaf)

# Node ordering: [DDR_hub, DDR_leaf, CellCycle_hub, CellCycle_leaf, ...]
NODE_NAMES = []
for ch in CHANNEL_NAMES:
    NODE_NAMES.append(f"{ch}_hub")
    NODE_NAMES.append(f"{ch}_leaf")


def _get_position(gene):
    """Get hub/leaf position for a gene. Unclassified goes to leaf."""
    pos = GENE_POSITION.get(gene, "unclassified")
    return "hub" if pos == "hub" else "leaf"


def build_hublf_features(study_id="msk_impact_50k"):
    """Build 12-node hub/leaf channel features.

    Returns dict with:
        hublf_features: (N, 12, 16)
        cancer_type_idx: (N,) int
        age: (N,) float
        sex: (N,) int
        times: (N,)
        events: (N,)
        cancer_type_vocab: list
    """
    cache_path = os.path.join(GNN_CACHE, f"hublf_features_{study_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    print(f"Building hub/leaf features for {study_id}...")
    paths = MSK_DATASETS[study_id]

    mut = pd.read_csv(paths["mutations"], low_memory=False)
    clin = pd.read_csv(paths["clinical"], low_memory=False)

    # Parse survival
    clin = clin[clin["OS_STATUS"].notna() & clin["OS_MONTHS"].notna()].copy()
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    clin["time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin = clin.dropna(subset=["time"])
    clin = clin[clin["time"] > 0]

    # Clinical covariates
    clin["age"] = pd.to_numeric(clin.get("AGE_AT_DX", pd.Series(dtype=float)),
                                errors="coerce")
    clin["sex_int"] = clin.get("SEX", pd.Series(dtype=str)).map(
        {"Male": 1, "Female": 0}
    ).fillna(-1).astype(int)

    # Cancer type
    if "sample_clinical" in paths and os.path.exists(paths["sample_clinical"]):
        sample_clin = pd.read_csv(paths["sample_clinical"], low_memory=False)
        ct = sample_clin.drop_duplicates("patientId")[["patientId", "CANCER_TYPE"]]
        clin = clin.merge(ct, on="patientId", how="left")
    if "CANCER_TYPE" not in clin.columns:
        clin["CANCER_TYPE"] = "Unknown"
    clin["CANCER_TYPE"] = clin["CANCER_TYPE"].fillna("Unknown")

    ct_counts = clin["CANCER_TYPE"].value_counts()
    common_types = ct_counts[ct_counts >= 50].index.tolist()
    cancer_type_vocab = sorted(common_types) + ["Other"]
    ct_to_idx = {ct: i for i, ct in enumerate(cancer_type_vocab)}
    clin["ct_idx"] = clin["CANCER_TYPE"].map(
        lambda x: ct_to_idx.get(x, ct_to_idx["Other"])
    )

    # Filter mutations
    channel_genes = set(CHANNEL_MAP.keys())
    mut = mut[
        mut["mutationType"].isin(NON_SILENT) &
        mut["gene.hugoGeneSymbol"].isin(channel_genes)
    ].copy()
    mut["channel"] = mut["gene.hugoGeneSymbol"].map(CHANNEL_MAP)
    mut["position"] = mut["gene.hugoGeneSymbol"].map(_get_position)
    mut["is_truncating"] = mut["mutationType"].isin(TRUNCATING)
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")
    mut["vaf"] = pd.to_numeric(mut["tumorAltCount"], errors="coerce") / (
        pd.to_numeric(mut["tumorAltCount"], errors="coerce") +
        pd.to_numeric(mut["tumorRefCount"], errors="coerce")
    )
    mut["vaf"] = mut["vaf"].fillna(0.0)
    mut["is_gof"] = mut["gene.hugoGeneSymbol"].map(lambda g: GENE_FUNCTION.get(g) == "GOF")
    mut["is_lof"] = mut["gene.hugoGeneSymbol"].map(lambda g: GENE_FUNCTION.get(g) == "LOF")

    # Per-patient-per-channel-per-position aggregation
    grouped = mut.groupby(["patientId", "channel", "position"]).agg(
        n_mutations=("gene.hugoGeneSymbol", "size"),
        n_genes=("gene.hugoGeneSymbol", "nunique"),
        frac_truncating=("is_truncating", "mean"),
        frac_missense=("is_missense", "mean"),
        mean_vaf=("vaf", "mean"),
        max_vaf=("vaf", "max"),
        gof_count=("is_gof", "sum"),
        lof_count=("is_lof", "sum"),
    ).reset_index()

    # Build tensors
    patients = clin["patientId"].unique()
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    N = len(patients)

    hublf_features = torch.zeros(N, N_NODES, HUBLF_FEAT_DIM)
    cancer_type_idx = torch.zeros(N, dtype=torch.long)
    age = torch.zeros(N)
    sex = torch.zeros(N, dtype=torch.long)
    times = torch.zeros(N)
    events = torch.zeros(N)

    for _, row in clin.iterrows():
        pid = row["patientId"]
        if pid not in patient_to_idx:
            continue
        idx = patient_to_idx[pid]
        times[idx] = row["time"]
        events[idx] = row["event"]
        cancer_type_idx[idx] = row["ct_idx"]
        if pd.notna(row.get("age")):
            age[idx] = (row["age"] - 60) / 15
        sex[idx] = max(row.get("sex_int", -1), 0)

    # Fill hub/leaf features
    for _, row in grouped.iterrows():
        pid = row["patientId"]
        if pid not in patient_to_idx:
            continue
        p_idx = patient_to_idx[pid]
        c_idx = CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        # Node index: hub = 2*c_idx, leaf = 2*c_idx + 1
        pos_offset = 0 if row["position"] == "hub" else 1
        node_idx = 2 * c_idx + pos_offset

        feat_vec = torch.tensor([
            1.0,  # is_severed
            np.log1p(row["n_mutations"]),
            float(row["n_genes"]),
            float(row["frac_truncating"]),
            float(row["frac_missense"]),
            float(row["mean_vaf"]),
            float(row["max_vaf"]),
            float(row["gof_count"]),
            float(row["lof_count"]),
            1.0 if row["position"] == "hub" else 0.0,  # is_hub
        ])

        hublf_features[p_idx, node_idx, :10] = feat_vec

    # Channel one-hot identity (dims 10-15)
    for c in range(6):
        hublf_features[:, 2 * c, 10 + c] = 1.0      # hub node
        hublf_features[:, 2 * c + 1, 10 + c] = 1.0   # leaf node

    result = {
        "hublf_features": hublf_features,
        "cancer_type_idx": cancer_type_idx,
        "age": age,
        "sex": sex,
        "times": times,
        "events": events,
        "cancer_type_vocab": cancer_type_vocab,
    }

    torch.save(result, cache_path)
    print(f"  {N} patients, {N_NODES} nodes, {HUBLF_FEAT_DIM} features, cached.")
    return result


class HubLeafDataset(Dataset):
    """Dataset with 12 hub/leaf channel nodes."""

    def __init__(self, data_dict, indices=None):
        self.indices = indices
        self.data = data_dict

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.data["times"])

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        return {
            "hublf_features": self.data["hublf_features"][idx],
            "cancer_type_idx": self.data["cancer_type_idx"][idx],
            "age": self.data["age"][idx],
            "sex": self.data["sex"][idx],
            "time": self.data["times"][idx],
            "event": self.data["events"][idx],
        }
