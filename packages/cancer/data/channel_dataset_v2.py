"""
Enhanced channel dataset with:
  - 6-channel features (17 dims each)
  - 3-tier features (cell-intrinsic, tissue, organism — hierarchy grouping)
  - Cancer type embedding index
  - Clinical covariates (age, sex)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    HUB_GENES, LEAF_GENES, GENE_FUNCTION,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
    ANALYSIS_CACHE, GNN_CACHE,
)

CHANNEL_FEAT_DIM = 17

# Severance hierarchy tiers
# Cell-intrinsic: PI3K_Growth + CellCycle (severed first, most common)
# Tissue-level: DDR + TissueArch
# Organism-level: Endocrine + Immune (severed last, rarest)
TIER_MAP = {
    "PI3K_Growth": 0, "CellCycle": 0,   # cell-intrinsic
    "DDR": 1, "TissueArch": 1,           # tissue-level
    "Endocrine": 2, "Immune": 2,         # organism-level
}
TIER_NAMES = ["cell_intrinsic", "tissue_level", "organism_level"]
N_TIERS = 3


def _build_hub_leaf_sets():
    all_hubs, all_leaves = set(), set()
    for genes in HUB_GENES.values():
        all_hubs |= genes
    for genes in LEAF_GENES.values():
        all_leaves |= genes
    return all_hubs, all_leaves


def build_channel_features_v2(study_id="msk_impact_50k"):
    """Build enhanced feature set.

    Returns dict with:
        channel_features: (N, 6, 17)
        tier_features: (N, 3, 17)  — aggregated by hierarchy tier
        cancer_type_idx: (N,) int — index into cancer type vocabulary
        age: (N,) float — normalized age at diagnosis
        sex: (N,) int — 0=female, 1=male
        times: (N,)
        events: (N,)
        cancer_type_vocab: list of cancer type strings
    """
    cache_path = os.path.join(GNN_CACHE, f"channel_features_v2_{study_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    print(f"Building v2 channel features for {study_id}...")
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
    sample_clin = None
    if "sample_clinical" in paths and os.path.exists(paths["sample_clinical"]):
        sample_clin = pd.read_csv(paths["sample_clinical"], low_memory=False)
        ct = sample_clin.drop_duplicates("patientId")[["patientId", "CANCER_TYPE"]]
        clin = clin.merge(ct, on="patientId", how="left")
    if "CANCER_TYPE" not in clin.columns:
        clin["CANCER_TYPE"] = "Unknown"
    clin["CANCER_TYPE"] = clin["CANCER_TYPE"].fillna("Unknown")

    # Build cancer type vocabulary
    ct_counts = clin["CANCER_TYPE"].value_counts()
    # Keep types with >= 50 patients, collapse rest to "Other"
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
    mut["is_truncating"] = mut["mutationType"].isin(TRUNCATING)
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")
    mut["vaf"] = pd.to_numeric(mut["tumorAltCount"], errors="coerce") / (
        pd.to_numeric(mut["tumorAltCount"], errors="coerce") +
        pd.to_numeric(mut["tumorRefCount"], errors="coerce")
    )
    mut["vaf"] = mut["vaf"].fillna(0.0)
    mut["is_gof"] = mut["gene.hugoGeneSymbol"].map(lambda g: GENE_FUNCTION.get(g) == "GOF")
    mut["is_lof"] = mut["gene.hugoGeneSymbol"].map(lambda g: GENE_FUNCTION.get(g) == "LOF")

    all_hubs, all_leaves = _build_hub_leaf_sets()

    # Per-patient-per-channel aggregation
    grouped = mut.groupby(["patientId", "channel"]).agg(
        n_mutations=("gene.hugoGeneSymbol", "size"),
        n_genes=("gene.hugoGeneSymbol", "nunique"),
        has_hub=("gene.hugoGeneSymbol", lambda x: bool(set(x) & all_hubs)),
        has_leaf=("gene.hugoGeneSymbol", lambda x: bool(set(x) & all_leaves)),
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

    channel_features = torch.zeros(N, 6, CHANNEL_FEAT_DIM)
    tier_features = torch.zeros(N, N_TIERS, CHANNEL_FEAT_DIM)
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
            age[idx] = (row["age"] - 60) / 15  # normalize around 60
        sex[idx] = max(row.get("sex_int", -1), 0)

    # Fill channel features
    for _, row in grouped.iterrows():
        pid = row["patientId"]
        if pid not in patient_to_idx:
            continue
        p_idx = patient_to_idx[pid]
        c_idx = CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        feat_vec = torch.tensor([
            1.0,  # is_severed
            np.log1p(row["n_mutations"]),
            float(row["n_genes"]),
            float(row["has_hub"]),
            float(row["has_leaf"]),
            float(row["frac_truncating"]),
            float(row["frac_missense"]),
            float(row["mean_vaf"]),
            float(row["max_vaf"]),
            float(row["gof_count"]),
            float(row["lof_count"]),
        ])

        channel_features[p_idx, c_idx, :11] = feat_vec

    # Channel one-hot identity
    for c in range(6):
        channel_features[:, c, 11 + c] = 1.0

    # Aggregate to 3-tier features
    # For each tier, take element-wise max of its two channels (except counts which sum)
    for tier_idx in range(N_TIERS):
        channels_in_tier = [c for c, t in TIER_MAP.items() if t == tier_idx]
        c_indices = [CHANNEL_TO_IDX[c] for c in channels_in_tier]

        # is_severed: any channel severed
        tier_features[:, tier_idx, 0] = torch.stack(
            [channel_features[:, ci, 0] for ci in c_indices]
        ).max(dim=0).values

        # n_mutations, n_genes: sum
        for feat_idx in [1, 2, 9, 10]:
            tier_features[:, tier_idx, feat_idx] = torch.stack(
                [channel_features[:, ci, feat_idx] for ci in c_indices]
            ).sum(dim=0)

        # has_hub, has_leaf: any
        for feat_idx in [3, 4]:
            tier_features[:, tier_idx, feat_idx] = torch.stack(
                [channel_features[:, ci, feat_idx] for ci in c_indices]
            ).max(dim=0).values

        # fracs and VAFs: mean of non-zero
        for feat_idx in [5, 6, 7, 8]:
            vals = torch.stack([channel_features[:, ci, feat_idx] for ci in c_indices])
            masks = vals > 0
            sums = vals.sum(dim=0)
            counts = masks.float().sum(dim=0).clamp(min=1)
            tier_features[:, tier_idx, feat_idx] = sums / counts

        # Tier one-hot (reuse last 6 dims, put tier identity in first 3)
        tier_features[:, tier_idx, 11 + tier_idx] = 1.0

    result = {
        "channel_features": channel_features,
        "tier_features": tier_features,
        "cancer_type_idx": cancer_type_idx,
        "age": age,
        "sex": sex,
        "times": times,
        "events": events,
        "cancer_type_vocab": cancer_type_vocab,
    }

    torch.save(result, cache_path)
    print(f"  {N} patients, {len(cancer_type_vocab)} cancer types, cached.")
    return result


class ChannelDatasetV2(Dataset):
    """Enhanced channel dataset with tiers and clinical covariates."""

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
            "channel_features": self.data["channel_features"][idx],
            "tier_features": self.data["tier_features"][idx],
            "cancer_type_idx": self.data["cancer_type_idx"][idx],
            "age": self.data["age"][idx],
            "sex": self.data["sex"][idx],
            "time": self.data["times"][idx],
            "event": self.data["events"][idx],
        }
