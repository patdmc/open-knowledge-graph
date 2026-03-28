#!/usr/bin/env python3
"""
Baseline: Cox PH with channel count and hub/leaf features.
This is what we need to beat with the GNN.

Usage:
    python3 -m gnn.scripts.baseline_cox
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, LEAF_GENES,
    NON_SILENT, ANALYSIS_CACHE, GNN_RESULTS,
)


def build_features(study_id="msk_impact_50k"):
    """Build patient-level feature matrix for Cox PH baseline."""
    prefix = {
        "msk_impact_50k": "msk_impact_50k_2026",
        "msk_met_2021": "msk_met_2021",
        "msk_impact_2017": "msk_impact_2017",
    }[study_id]

    mut = pd.read_csv(os.path.join(ANALYSIS_CACHE, f"{prefix}_mutations.csv"), low_memory=False)
    clin = pd.read_csv(os.path.join(ANALYSIS_CACHE, f"{prefix}_clinical.csv"), low_memory=False)
    sample = pd.read_csv(os.path.join(ANALYSIS_CACHE, f"{prefix}_sample_clinical.csv"), low_memory=False)

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

    # Hub/leaf classification
    all_hubs = set()
    all_leaves = set()
    for genes in HUB_GENES.values():
        all_hubs |= genes
    for genes in LEAF_GENES.values():
        all_leaves |= genes

    # Vectorized feature construction
    # Channel count per patient
    channel_count = mut.groupby("patientId")["channel"].nunique().reset_index()
    channel_count.columns = ["patientId", "channel_count"]

    # Mutation count per patient
    mut_count = mut.groupby("patientId").size().reset_index(name="mutation_count")

    # Hub/leaf flags
    mut["is_hub"] = mut["gene.hugoGeneSymbol"].isin(all_hubs)
    mut["is_leaf"] = mut["gene.hugoGeneSymbol"].isin(all_leaves)
    hub_flag = mut.groupby("patientId")["is_hub"].any().reset_index()
    hub_flag.columns = ["patientId", "has_hub"]
    leaf_flag = mut.groupby("patientId")["is_leaf"].any().reset_index()
    leaf_flag.columns = ["patientId", "has_leaf"]

    # Per-channel binary features via pivot
    channel_pivot = mut.groupby(["patientId", "channel"]).size().unstack(fill_value=0)
    for ch in CHANNEL_NAMES:
        if ch not in channel_pivot.columns:
            channel_pivot[ch] = 0
    channel_binary = (channel_pivot[CHANNEL_NAMES] > 0).astype(int).reset_index()
    channel_binary.columns = ["patientId"] + [f"ch_{ch}" for ch in CHANNEL_NAMES]

    # Merge everything onto clinical
    df = clin[["patientId", "time", "event"]].copy()
    df = df.merge(channel_count, on="patientId", how="left")
    df = df.merge(mut_count, on="patientId", how="left")
    df = df.merge(hub_flag, on="patientId", how="left")
    df = df.merge(leaf_flag, on="patientId", how="left")
    df = df.merge(channel_binary, on="patientId", how="left")

    # Fill NaN for patients with no channel mutations
    df["channel_count"] = df["channel_count"].fillna(0).astype(int)
    df["mutation_count"] = df["mutation_count"].fillna(0).astype(int)
    df["has_hub"] = df["has_hub"].fillna(False).astype(int)
    df["has_leaf"] = df["has_leaf"].fillna(False).astype(int)
    for ch in CHANNEL_NAMES:
        df[f"ch_{ch}"] = df[f"ch_{ch}"].fillna(0).astype(int)
    df["hub_only"] = ((df["has_hub"] == 1) & (df["has_leaf"] == 0)).astype(int)

    # Merge cancer type
    ct = sample.drop_duplicates("patientId")[["patientId", "CANCER_TYPE"]]
    df = df.merge(ct, on="patientId", how="left")

    return df


def run_cox_baseline():
    """Run 5-fold CV Cox PH baselines."""
    print("Building features...")
    df = build_features()
    print(f"  {len(df)} patients")

    # Stratification key
    strat = df["event"].astype(str) + "_" + df["channel_count"].clip(upper=3).astype(str)

    models = {
        "Channel Count Only": ["channel_count"],
        "Channel Count + Hub/Leaf": ["channel_count", "has_hub"],
        "Per-Channel Binary": [f"ch_{ch}" for ch in CHANNEL_NAMES],
        "Per-Channel + Hub/Leaf": [f"ch_{ch}" for ch in CHANNEL_NAMES] + ["has_hub", "has_leaf"],
        "Full (channels + hub + mut_count)": [f"ch_{ch}" for ch in CHANNEL_NAMES] + ["has_hub", "has_leaf", "mutation_count"],
    }

    print(f"\n{'Model':<45} {'C-index':<20}")
    print(f"{'-'*45} {'-'*20}")

    results = {}
    for name, features in models.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_cis = []

        for train_idx, val_idx in skf.split(df, strat):
            train = df.iloc[train_idx]
            val = df.iloc[val_idx]

            cox_df = train[["time", "event"] + features].copy()
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(cox_df, duration_col="time", event_col="event")

                val_hazard = cph.predict_partial_hazard(val[features])
                ci = concordance_index(val["time"], -val_hazard.values.flatten(), val["event"])
                fold_cis.append(ci)
            except Exception as e:
                fold_cis.append(0.5)

        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        print(f"  {name:<43} {mean_ci:.4f} +/- {std_ci:.4f}")
        results[name] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis}

    # Save results
    import json
    save_dir = os.path.join(GNN_RESULTS, "baselines")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cox_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_dir}")

    return results


if __name__ == "__main__":
    run_cox_baseline()
