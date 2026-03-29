"""
Channel-level dataset: pre-aggregate patient mutations to 6 channel feature vectors.

Per-channel features (16-dim):
  [0]  is_severed (binary: any mutation in this channel)
  [1]  n_mutations (count, log-scaled)
  [2]  n_genes_mutated (count)
  [3]  has_hub (binary)
  [4]  has_leaf (binary)
  [5]  frac_truncating (fraction of mutations that are truncating)
  [6]  frac_missense (fraction that are missense)
  [7]  mean_vaf (mean variant allele frequency)
  [8]  max_vaf (max VAF — dominant clone signal)
  [9]  gof_count (number of GOF gene mutations)
  [10] lof_count (number of LOF gene mutations)
  [11:17] channel one-hot (6-dim, identity)

Total: 17 dims per channel, 6 channels = 102 raw features per patient.
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


def _build_hub_leaf_sets():
    all_hubs = set()
    all_leaves = set()
    for genes in HUB_GENES.values():
        all_hubs |= genes
    for genes in LEAF_GENES.values():
        all_leaves |= genes
    return all_hubs, all_leaves


def build_channel_features(study_id="msk_impact_50k"):
    """Build channel-level feature matrix for all patients.

    Returns:
        features: (N, 6, 17) float tensor
        times: (N,) float tensor
        events: (N,) float tensor
    """
    cache_path = os.path.join(GNN_CACHE, f"channel_features_{study_id}.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        return data["features"], data["times"], data["events"]

    print(f"Building channel features for {study_id}...")
    paths = MSK_DATASETS[study_id]

    mut = pd.read_csv(paths["mutations"], low_memory=False)
    clin = pd.read_csv(paths["clinical"], low_memory=False)

    # Parse survival
    clin = clin[clin["OS_STATUS"].notna() & clin["OS_MONTHS"].notna()].copy()
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    clin["time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin = clin.dropna(subset=["time"])
    clin = clin[clin["time"] > 0]

    # Filter mutations
    channel_genes = set(CHANNEL_MAP.keys())
    mut = mut[
        mut["mutationType"].isin(NON_SILENT) &
        mut["gene.hugoGeneSymbol"].isin(channel_genes)
    ].copy()
    mut["channel"] = mut["gene.hugoGeneSymbol"].map(CHANNEL_MAP)
    mut["is_truncating"] = mut["mutationType"].isin(TRUNCATING)
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")

    # VAF
    mut["vaf"] = pd.to_numeric(mut["tumorAltCount"], errors="coerce") / (
        pd.to_numeric(mut["tumorAltCount"], errors="coerce") +
        pd.to_numeric(mut["tumorRefCount"], errors="coerce")
    )
    mut["vaf"] = mut["vaf"].fillna(0.0)

    # GOF/LOF
    mut["is_gof"] = mut["gene.hugoGeneSymbol"].map(
        lambda g: GENE_FUNCTION.get(g) == "GOF"
    )
    mut["is_lof"] = mut["gene.hugoGeneSymbol"].map(
        lambda g: GENE_FUNCTION.get(g) == "LOF"
    )

    all_hubs, all_leaves = _build_hub_leaf_sets()

    # Vectorized per-patient-per-channel aggregation
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

    # Build feature tensors
    patients = clin["patientId"].unique()
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    N = len(patients)

    features = torch.zeros(N, 6, CHANNEL_FEAT_DIM)
    times = torch.zeros(N)
    events = torch.zeros(N)

    # Fill survival data
    for _, row in clin.iterrows():
        pid = row["patientId"]
        if pid in patient_to_idx:
            idx = patient_to_idx[pid]
            times[idx] = row["time"]
            events[idx] = row["event"]

    # Fill channel features
    for _, row in grouped.iterrows():
        pid = row["patientId"]
        if pid not in patient_to_idx:
            continue
        p_idx = patient_to_idx[pid]
        c_idx = CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        features[p_idx, c_idx, 0] = 1.0  # is_severed
        features[p_idx, c_idx, 1] = np.log1p(row["n_mutations"])
        features[p_idx, c_idx, 2] = row["n_genes"]
        features[p_idx, c_idx, 3] = float(row["has_hub"])
        features[p_idx, c_idx, 4] = float(row["has_leaf"])
        features[p_idx, c_idx, 5] = row["frac_truncating"]
        features[p_idx, c_idx, 6] = row["frac_missense"]
        features[p_idx, c_idx, 7] = row["mean_vaf"]
        features[p_idx, c_idx, 8] = row["max_vaf"]
        features[p_idx, c_idx, 9] = row["gof_count"]
        features[p_idx, c_idx, 10] = row["lof_count"]

    # Channel one-hot identity (always set, even for unsevered channels)
    for c in range(6):
        features[:, c, 11 + c] = 1.0

    # Cache
    torch.save({"features": features, "times": times, "events": events}, cache_path)
    print(f"  {N} patients, cached to {cache_path}")

    return features, times, events


class ChannelDataset(Dataset):
    """Simple tensor dataset for channel-level features."""

    def __init__(self, features, times, events, indices=None):
        """
        Args:
            features: (N, 6, 17) channel feature tensor
            times: (N,) survival times
            events: (N,) event indicators
            indices: optional subset indices
        """
        if indices is not None:
            self.features = features[indices]
            self.times = times[indices]
            self.events = events[indices]
        else:
            self.features = features
            self.times = times
            self.events = events

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return {
            "channel_features": self.features[idx],  # (6, 17)
            "time": self.times[idx],
            "event": self.events[idx],
        }
