"""
Channel dataset V2+MSI: 6 channels (V2 architecture) + MSI/TMB clinical features.

Ablation dataset to test whether 8-channel expansion adds value beyond MSI/TMB.
Uses V2's 6-channel gene set and 17-dim features, but adds msi_score, msi_high, tmb.
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
    GNN_CACHE,
)
from .channel_dataset_v2 import CHANNEL_FEAT_DIM, TIER_MAP, N_TIERS


def build_channel_features_v2_msi(study_id="msk_impact_50k"):
    """Build V2 6-channel features + MSI/TMB clinical covariates.

    Returns dict with:
        channel_features: (N, 6, 17)
        tier_features: (N, 3, 17)
        cancer_type_idx: (N,) int
        age: (N,) float
        sex: (N,) int
        msi_score: (N,) float
        msi_high: (N,) int
        tmb: (N,) float
        times: (N,)
        events: (N,)
        cancer_type_vocab: list
    """
    cache_path = os.path.join(GNN_CACHE, f"channel_v2_msi_features_{study_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    print(f"Building V2+MSI (6-channel + MSI/TMB) features for {study_id}...")

    paths = MSK_DATASETS[study_id]
    mut = pd.read_csv(paths["mutations"], low_memory=False)
    clin = pd.read_csv(paths["clinical"], low_memory=False)

    # Load full clinical for MSI + TMB
    full_clin_path = paths["clinical"].replace("_clinical.csv", "_full_clinical.csv")
    if os.path.exists(full_clin_path):
        full_clin = pd.read_csv(full_clin_path, low_memory=False)
        extra = full_clin[["patientId", "MSI_SCORE", "MSI_TYPE", "CVR_TMB_SCORE"]].drop_duplicates("patientId")
        clin = clin.merge(extra, on="patientId", how="left")
    else:
        clin["MSI_SCORE"] = np.nan
        clin["MSI_TYPE"] = "Unknown"
        clin["CVR_TMB_SCORE"] = np.nan

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

    # MSI features
    clin["msi_score_raw"] = pd.to_numeric(clin.get("MSI_SCORE", pd.Series(dtype=float)),
                                           errors="coerce").fillna(0.0).clip(lower=0.0)
    clin["msi_score_log"] = np.log1p(clin["msi_score_raw"])
    msi_max = clin["msi_score_log"].quantile(0.99)
    if msi_max > 0:
        clin["msi_score_norm"] = clin["msi_score_log"] / msi_max
    else:
        clin["msi_score_norm"] = 0.0
    clin["msi_high"] = (clin["msi_score_raw"] > 10).astype(int)

    # TMB features
    clin["tmb_raw"] = pd.to_numeric(clin.get("CVR_TMB_SCORE", pd.Series(dtype=float)),
                                     errors="coerce").fillna(0.0).clip(lower=0.0)
    clin["tmb_log"] = np.log1p(clin["tmb_raw"])
    tmb_max = clin["tmb_log"].quantile(0.99)
    if tmb_max > 0:
        clin["tmb_norm"] = clin["tmb_log"] / tmb_max
    else:
        clin["tmb_norm"] = 0.0

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

    # Filter mutations to V2's 6-channel gene set
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

    all_hubs, all_leaves = set(), set()
    for genes in HUB_GENES.values():
        all_hubs |= genes
    for genes in LEAF_GENES.values():
        all_leaves |= genes

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
    msi_score = torch.zeros(N)
    msi_high = torch.zeros(N, dtype=torch.long)
    tmb = torch.zeros(N)
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
        msi_score[idx] = row["msi_score_norm"]
        msi_high[idx] = row["msi_high"]
        tmb[idx] = row["tmb_norm"]

    for _, row in grouped.iterrows():
        pid = row["patientId"]
        if pid not in patient_to_idx:
            continue
        p_idx = patient_to_idx[pid]
        c_idx = CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        feat_vec = torch.tensor([
            1.0,
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

    # Tier features
    for tier_idx in range(N_TIERS):
        channels_in_tier = [c for c, t in TIER_MAP.items() if t == tier_idx]
        c_indices = [CHANNEL_TO_IDX[c] for c in channels_in_tier]

        tier_features[:, tier_idx, 0] = torch.stack(
            [channel_features[:, ci, 0] for ci in c_indices]
        ).max(dim=0).values

        for feat_idx in [1, 2, 9, 10]:
            tier_features[:, tier_idx, feat_idx] = torch.stack(
                [channel_features[:, ci, feat_idx] for ci in c_indices]
            ).sum(dim=0)

        for feat_idx in [3, 4]:
            tier_features[:, tier_idx, feat_idx] = torch.stack(
                [channel_features[:, ci, feat_idx] for ci in c_indices]
            ).max(dim=0).values

        for feat_idx in [5, 6, 7, 8]:
            vals = torch.stack([channel_features[:, ci, feat_idx] for ci in c_indices])
            masks = vals > 0
            sums = vals.sum(dim=0)
            counts = masks.float().sum(dim=0).clamp(min=1)
            tier_features[:, tier_idx, feat_idx] = sums / counts

        tier_features[:, tier_idx, 11 + tier_idx] = 1.0

    result = {
        "channel_features": channel_features,
        "tier_features": tier_features,
        "cancer_type_idx": cancer_type_idx,
        "age": age,
        "sex": sex,
        "msi_score": msi_score,
        "msi_high": msi_high,
        "tmb": tmb,
        "times": times,
        "events": events,
        "cancer_type_vocab": cancer_type_vocab,
    }

    torch.save(result, cache_path)
    n_msi = int(msi_high.sum())
    print(f"  {N} patients, 6 channels, {n_msi} MSI-high, cached.")
    return result


class ChannelDatasetV2MSI(Dataset):
    """V2 6-channel dataset with MSI/TMB clinical features."""

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
            "msi_score": self.data["msi_score"][idx],
            "msi_high": self.data["msi_high"][idx],
            "tmb": self.data["tmb"][idx],
            "time": self.data["times"][idx],
            "event": self.data["events"][idx],
        }
