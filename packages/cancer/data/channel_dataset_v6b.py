"""
Channel dataset v6b: 8 channels, survival-filtered gene set.

Only keeps genes where mutation associates with equal or worse survival
(survival_ratio <= 1.05). Removes compensatory/confounded genes.

Original 99 curated genes + 14 true severance + ~52 weak/neutral = ~165 total.

Same architecture as v6: 8 channels × 9 features + 4 tiers × 5 features.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import (
    CHANNEL_MAP, NON_SILENT, TRUNCATING, MSK_DATASETS,
    GNN_CACHE, GNN_RESULTS,
)
from .channel_dataset_v6 import (
    V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX, N_CHANNELS_V6,
    CHANNEL_FEAT_DIM_V6, N_TIERS_V6, V6_TIER_MAP, V6_TIER_NAMES,
    V6_GENE_FUNCTION,
)

# Load filtered gene set
_FILTER_PATH = os.path.join(GNN_RESULTS, "gene_survival_filter.json")


def _build_v6b_channel_map():
    """Build channel map using only survival-filtered genes."""
    with open(_FILTER_PATH) as f:
        filt = json.load(f)
    return filt["keep_genes"]


V6B_CHANNEL_MAP = _build_v6b_channel_map()


def build_channel_features_v6b(study_id="msk_impact_50k"):
    """Build 8-channel features with survival-filtered gene set.

    Returns dict with:
        channel_features: (N, 8, 9)
        tier_features: (N, 4, 5)
        cancer_type_idx: (N,) int
        age: (N,) float
        sex: (N,) int
        times: (N,)
        events: (N,)
        cancer_type_vocab: list
    """
    cache_path = os.path.join(GNN_CACHE, f"channel_v6b_features_{study_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    v6b_map = V6B_CHANNEL_MAP
    channel_genes = set(v6b_map.keys())

    print(f"Building v6b (8-channel, filtered) features for {study_id}...")
    print(f"  {len(channel_genes)} genes across {N_CHANNELS_V6} channels")
    for ch in V6_CHANNEL_NAMES:
        n = sum(1 for g, c in v6b_map.items() if c == ch)
        print(f"    {ch}: {n} genes")

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
    mut = mut[
        mut["mutationType"].isin(NON_SILENT) &
        mut["gene.hugoGeneSymbol"].isin(channel_genes)
    ].copy()
    mut["channel"] = mut["gene.hugoGeneSymbol"].map(v6b_map)
    mut["is_truncating"] = mut["mutationType"].isin(TRUNCATING)
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")
    mut["vaf"] = pd.to_numeric(mut["tumorAltCount"], errors="coerce") / (
        pd.to_numeric(mut["tumorAltCount"], errors="coerce") +
        pd.to_numeric(mut["tumorRefCount"], errors="coerce")
    )
    mut["vaf"] = mut["vaf"].fillna(0.0)
    mut["is_gof"] = mut["gene.hugoGeneSymbol"].map(
        lambda g: V6_GENE_FUNCTION.get(g) == "GOF"
    )
    mut["is_lof"] = mut["gene.hugoGeneSymbol"].map(
        lambda g: V6_GENE_FUNCTION.get(g) == "LOF"
    )

    # Per-patient-per-channel aggregation
    grouped = mut.groupby(["patientId", "channel"]).agg(
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

    channel_features = torch.zeros(N, N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6)
    tier_features = torch.zeros(N, N_TIERS_V6, 5)
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

    for _, row in grouped.iterrows():
        pid = row["patientId"]
        if pid not in patient_to_idx:
            continue
        p_idx = patient_to_idx[pid]
        c_idx = V6_CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        channel_features[p_idx, c_idx] = torch.tensor([
            1.0,
            np.log1p(row["n_mutations"]),
            float(row["n_genes"]),
            float(row["frac_truncating"]),
            float(row["frac_missense"]),
            float(row["mean_vaf"]),
            float(row["max_vaf"]),
            float(row["gof_count"]),
            float(row["lof_count"]),
        ])

    # Tier features
    for t_idx in range(N_TIERS_V6):
        tier_channels = [V6_CHANNEL_TO_IDX[ch] for ch, t in V6_TIER_MAP.items() if t == t_idx]
        if not tier_channels:
            continue
        tier_ch = channel_features[:, tier_channels, :]
        tier_features[:, t_idx, 0] = tier_ch[:, :, 0].sum(dim=1)
        tier_features[:, t_idx, 1] = tier_ch[:, :, 1].sum(dim=1)
        tier_features[:, t_idx, 2] = tier_ch[:, :, 2].sum(dim=1)
        tier_features[:, t_idx, 3] = tier_ch[:, :, 5].max(dim=1)[0]
        tier_features[:, t_idx, 4] = tier_ch[:, :, 6].max(dim=1)[0]

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
    print(f"  {N} patients, {N_CHANNELS_V6} channels, filtered to {len(channel_genes)} genes, cached.")
    return result


class ChannelDatasetV6b(Dataset):
    """Dataset with 8 channels, survival-filtered genes."""

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
