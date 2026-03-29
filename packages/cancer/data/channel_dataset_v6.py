"""
Channel dataset v6: 8 channels (original 6 + 2 epigenetic meta-channels).

Adds two meta-regulatory channels discovered from error analysis:
  7. ChromatinRemodel — structural access control (histone modification, SWI/SNF)
  8. DNAMethylation — chemical state inheritance (methylation, metabolic epigenetics)

These mirror the cell_intrinsic split (PI3K_Growth / CellCycle) —
two distinct mechanisms within the same regulatory tier.

Per-channel features (same as v2):
  [0]  is_severed (any mutation in this channel)
  [1]  n_mutations (log-scaled)
  [2]  n_genes
  [3]  frac_truncating
  [4]  frac_missense
  [5]  mean_vaf
  [6]  max_vaf
  [7]  gof_count
  [8]  lof_count

Total: 9 dims per channel, 8 channels = 72 raw features per patient.

Tier features: 4 tiers × 5 features = 20
  Tier 0: cell_intrinsic (PI3K_Growth + CellCycle)
  Tier 1: tissue_level (DDR + TissueArch)
  Tier 2: organism_level (Endocrine + Immune)
  Tier 3: meta_regulatory (ChromatinRemodel + DNAMethylation)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    GENE_FUNCTION, NON_SILENT, TRUNCATING, MSK_DATASETS,
    GNN_CACHE, GNN_RESULTS,
)

# ── Channel 7: Chromatin remodeling / histone modification (structural access) ──
CHROMATIN_REMODEL_GENES = {
    # Histone methyltransferases
    "KMT2D", "KMT2C", "KMT2A", "KMT2B",
    "SETD2", "NSD1",
    # Histone acetyltransferases
    "CREBBP", "EP300",
    # SWI/SNF chromatin remodeling
    "ARID1B", "ARID2", "SMARCA4",
    # Histone demethylases
    "KDM6A",
    # Polycomb
    "BCOR",
    # Histone variants
    "H3C7",
    # Coactivator
    "ANKRD11",
}

# ── Channel 8: DNA methylation / metabolic epigenetics (chemical state) ──
DNA_METHYLATION_GENES = {
    # DNA methyltransferases
    "DNMT3A", "DNMT3B",
    # TET demethylases (5mC → 5hmC)
    "TET1", "TET2",
    # Metabolic drivers of epigenetic state (IDH → 2-HG → blocks TET/KDM)
    "IDH1", "IDH2",
    # Chromatin/telomere (ATRX-DAXX deposits H3.3, maintains ALT telomeres)
    "ATRX", "DAXX",
}

# Expanded channel map: original 99 + high-confidence Hallmark additions + chromatin
# Load the expanded map for high-confidence additions
_EXPANDED_PATH = os.path.join(GNN_RESULTS, "expanded_channel_map.json")

def _build_v6_channel_map():
    """Build the 8-channel map with two epigenetic meta-channels."""
    v6_map = dict(CHANNEL_MAP)  # Start with curated 99

    # Add ChromatinRemodel channel
    for gene in CHROMATIN_REMODEL_GENES:
        if gene not in v6_map:
            v6_map[gene] = "ChromatinRemodel"

    # Add DNAMethylation channel
    for gene in DNA_METHYLATION_GENES:
        if gene not in v6_map:
            v6_map[gene] = "DNAMethylation"

    # Add high-confidence Hallmark expansions for existing channels
    all_new = CHROMATIN_REMODEL_GENES | DNA_METHYLATION_GENES
    if os.path.exists(_EXPANDED_PATH):
        with open(_EXPANDED_PATH) as f:
            expanded = json.load(f)
        for gene, info in expanded.items():
            if gene in v6_map:
                continue  # Already assigned
            if gene in all_new:
                continue  # Goes to epigenetic channels
            if info["confidence"] == "high" and info["source"] == "hallmark":
                v6_map[gene] = info["channel"]

    return v6_map


V6_CHANNEL_NAMES = [
    "DDR", "CellCycle", "PI3K_Growth", "Endocrine", "Immune", "TissueArch",
    "ChromatinRemodel", "DNAMethylation",
]
V6_CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(V6_CHANNEL_NAMES)}
N_CHANNELS_V6 = 8

V6_CHANNEL_MAP = _build_v6_channel_map()

CHANNEL_FEAT_DIM_V6 = 9
N_TIERS_V6 = 4

V6_TIER_MAP = {
    "PI3K_Growth": 0, "CellCycle": 0,                   # cell_intrinsic
    "DDR": 1, "TissueArch": 1,                          # tissue_level
    "Endocrine": 2, "Immune": 2,                        # organism_level
    "ChromatinRemodel": 3, "DNAMethylation": 3,         # meta_regulatory
}
V6_TIER_NAMES = ["cell_intrinsic", "tissue_level", "organism_level", "meta_regulatory"]

# GOF/LOF for new channel genes (predominantly LOF — tumor suppressors)
V6_GENE_FUNCTION = dict(GENE_FUNCTION)
for gene in CHROMATIN_REMODEL_GENES | DNA_METHYLATION_GENES:
    if gene not in V6_GENE_FUNCTION:
        V6_GENE_FUNCTION[gene] = "LOF"
# IDH1/IDH2 are GOF (neomorphic gain-of-function producing 2-HG)
V6_GENE_FUNCTION["IDH1"] = "GOF"
V6_GENE_FUNCTION["IDH2"] = "GOF"


def build_channel_features_v6(study_id="msk_impact_50k"):
    """Build 7-channel features with expanded gene map.

    Returns dict with:
        channel_features: (N, 7, 9)
        tier_features: (N, 4, 5)
        cancer_type_idx: (N,) int
        age: (N,) float
        sex: (N,) int
        times: (N,)
        events: (N,)
        cancer_type_vocab: list
    """
    cache_path = os.path.join(GNN_CACHE, f"channel_v6_features_{study_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    print(f"Building v6 (7-channel) features for {study_id}...")

    v6_map = V6_CHANNEL_MAP
    channel_genes = set(v6_map.keys())
    print(f"  {len(channel_genes)} genes across {N_CHANNELS_V6} channels")
    for ch in V6_CHANNEL_NAMES:
        n = sum(1 for g, c in v6_map.items() if c == ch)
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

    # Filter mutations to channel genes
    mut = mut[
        mut["mutationType"].isin(NON_SILENT) &
        mut["gene.hugoGeneSymbol"].isin(channel_genes)
    ].copy()
    mut["channel"] = mut["gene.hugoGeneSymbol"].map(v6_map)
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
            1.0,  # is_severed
            np.log1p(row["n_mutations"]),
            float(row["n_genes"]),
            float(row["frac_truncating"]),
            float(row["frac_missense"]),
            float(row["mean_vaf"]),
            float(row["max_vaf"]),
            float(row["gof_count"]),
            float(row["lof_count"]),
        ])

    # Tier features: aggregate channels within each tier
    for t_idx in range(N_TIERS_V6):
        tier_channels = [V6_CHANNEL_TO_IDX[ch] for ch, t in V6_TIER_MAP.items() if t == t_idx]
        if not tier_channels:
            continue
        tier_ch = channel_features[:, tier_channels, :]  # (N, n_ch_in_tier, 9)
        tier_features[:, t_idx, 0] = tier_ch[:, :, 0].sum(dim=1)     # n_channels_severed
        tier_features[:, t_idx, 1] = tier_ch[:, :, 1].sum(dim=1)     # total_mutations
        tier_features[:, t_idx, 2] = tier_ch[:, :, 2].sum(dim=1)     # total_genes
        tier_features[:, t_idx, 3] = tier_ch[:, :, 5].max(dim=1)[0]  # max_mean_vaf
        tier_features[:, t_idx, 4] = tier_ch[:, :, 6].max(dim=1)[0]  # max_max_vaf

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
    print(f"  {N} patients, {N_CHANNELS_V6} channels, {N_TIERS_V6} tiers, cached.")
    return result


class ChannelDatasetV6(Dataset):
    """Dataset with 7 channels + 4 tiers."""

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
