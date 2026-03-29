#!/usr/bin/env python3
"""
Signal decomposition: where does the predictive signal actually live?

No neural network. Just simple scoring functions applied to each patient's
mutation profile, scored against survival with C-index.

Levels:
  1. n_channels         — flat count of severed channels
  2. weighted_channels   — sum of per-channel hazard weights
  3. channel_pairs       — adds pairwise channel interaction terms
  4. gene_weighted       — sum of per-gene hazard weights
  5. gene_pairs          — adds top pairwise gene interaction terms

Each level adds complexity. The C-index at each level shows where
the signal concentrates.

Also tests: demographics alone (age + sex + cancer_type + MSI + TMB)

Usage:
    python3 -u -m gnn.scripts.signal_decomposition
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX,
    N_CHANNELS_V6, V6_TIER_MAP,
)
from gnn.training.metrics import concordance_index

SAVE_BASE = os.path.join(GNN_RESULTS, "signal_decomposition")


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  SIGNAL DECOMPOSITION — WHERE DOES THE PREDICTION LIVE?")
    print("  No neural network. Simple scoring functions vs C-index.")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])

    times = data["times"].numpy()
    events = data["events"].numpy().astype(int)
    ch_feats = data["channel_features"].numpy()  # (N, 8, 9)

    # Load per-gene hazard weights
    with open(os.path.join(GNN_RESULTS, "per_mutation_impact", "gene_results.json")) as f:
        gene_results = json.load(f)
    gene_hazard = {r["gene"]: r["hazard_shift"] for r in gene_results}
    gene_abs_hazard = {r["gene"]: abs(r["hazard_shift"]) for r in gene_results}

    # Load per-gene mutation data
    import pandas as pd
    from gnn.config import ANALYSIS_CACHE

    mut_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    mutations = pd.read_csv(mut_path, low_memory=False,
                            usecols=["patientId", "gene.hugoGeneSymbol"])

    # Get patient ordering
    patient_ids = data.get("patient_ids")
    if patient_ids is None:
        clin_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv")
        clin = pd.read_csv(clin_path, low_memory=False)
        clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
        clin["time"] = clin["OS_MONTHS"].astype(float)
        clin["event"] = clin["OS_STATUS"].apply(
            lambda x: 1 if "DECEASED" in str(x).upper() else 0
        )
        clin = clin[clin["time"] > 0]
        patient_ids = clin["patientId"].unique()

    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mutations[mutations["gene.hugoGeneSymbol"].isin(channel_genes)].copy()

    # Build per-patient data
    print("Building per-patient profiles...")
    patient_genes = defaultdict(set)
    patient_channels = defaultdict(set)
    for _, row in mut.iterrows():
        pid = row["patientId"]
        if pid in pid_to_idx:
            idx = pid_to_idx[pid]
            gene = row["gene.hugoGeneSymbol"]
            patient_genes[idx].add(gene)
            patient_channels[idx].add(V6_CHANNEL_MAP[gene])

    # Compute per-channel hazard weight (mean absolute gene hazard in channel)
    channel_hazard = {}
    for ch_name in V6_CHANNEL_NAMES:
        ch_genes = [g for g, c in V6_CHANNEL_MAP.items() if c == ch_name]
        ch_shifts = [abs(gene_hazard.get(g, 0)) for g in ch_genes if g in gene_hazard]
        channel_hazard[ch_name] = np.mean(ch_shifts) if ch_shifts else 0.0

    print(f"  Per-channel hazard weights:")
    for ch in V6_CHANNEL_NAMES:
        print(f"    {ch:<20} {channel_hazard[ch]:.4f}")

    # Load pairwise synergy data
    synergy_path = os.path.join(GNN_RESULTS, "mutation_synergy", "pair_results.json")
    pair_synergy = {}
    if os.path.exists(synergy_path):
        with open(synergy_path) as f:
            pairs = json.load(f)
        for p in pairs:
            key = tuple(sorted([p["gene_a"], p["gene_b"]]))
            pair_synergy[key] = p["synergy"]
        print(f"  Loaded {len(pair_synergy)} pairwise synergy scores")

    # Channel pair synergy (mean synergy for each channel combination)
    ch_pair_synergy = defaultdict(list)
    if os.path.exists(synergy_path):
        for p in pairs:
            key = tuple(sorted([p["channel_a"], p["channel_b"]]))
            ch_pair_synergy[key].append(p["synergy"])
    ch_pair_mean = {k: np.mean(v) for k, v in ch_pair_synergy.items()}

    # Same folds as unified ablation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))

    # =========================================================================
    # Compute scores for each level
    # =========================================================================
    print("\nComputing scores for each level...")

    scores = {}

    # Level 0: Demographics only (age + sex + cancer_type + MSI + TMB)
    # Use a simple linear combination — higher age, MSI, TMB = higher hazard
    demo_score = np.zeros(N)
    demo_score += data["age"].numpy() * 0.5  # older = higher hazard
    demo_score += data["msi_score"].numpy() * -0.1  # MSI-high = better (immunotherapy)
    demo_score += data["tmb"].numpy() * -0.1  # high TMB = better (immunotherapy)
    scores["demographics"] = demo_score

    # Level 1: n_channels (flat count)
    n_ch = ch_feats[:, :, 0].sum(axis=1)  # is_severed flags
    scores["n_channels"] = n_ch

    # Level 1b: n_channels with saturation (quadratic — captures the reversal)
    # Fit: hazard peaks around 4-5 channels then drops
    # Use -(n-4.5)^2 + constant to capture the inverted U
    scores["n_channels_quad"] = -(n_ch - 4.5) ** 2

    # Level 2: weighted channels (sum of per-channel hazard * is_severed)
    weighted_ch = np.zeros(N)
    for ch_name in V6_CHANNEL_NAMES:
        c_idx = V6_CHANNEL_TO_IDX[ch_name]
        weighted_ch += ch_feats[:, c_idx, 0] * channel_hazard[ch_name]
    scores["weighted_channels"] = weighted_ch

    # Level 3: weighted channels + pairwise channel interactions
    ch_pair_score = np.zeros(N)
    for i in range(N):
        channels = patient_channels.get(i, set())
        if len(channels) >= 2:
            for ch_a, ch_b in __import__("itertools").combinations(sorted(channels), 2):
                key = tuple(sorted([ch_a, ch_b]))
                ch_pair_score[i] += ch_pair_mean.get(key, 0)
    scores["channel_pairs"] = weighted_ch + ch_pair_score

    # Level 4: per-gene hazard weights (sum of individual gene hazard shifts)
    gene_score = np.zeros(N)
    for i in range(N):
        genes = patient_genes.get(i, set())
        for g in genes:
            gene_score[i] += gene_hazard.get(g, 0)
    scores["gene_weighted"] = gene_score

    # Level 5: per-gene + pairwise gene synergy (top interactions)
    gene_pair_score = np.zeros(N)
    for i in range(N):
        genes = patient_genes.get(i, set())
        gene_list = sorted(genes)
        if len(gene_list) >= 2:
            for ga, gb in __import__("itertools").combinations(gene_list, 2):
                key = tuple(sorted([ga, gb]))
                gene_pair_score[i] += pair_synergy.get(key, 0)
    scores["gene_pairs"] = gene_score + gene_pair_score

    # Level 6: Combined — gene weighted + n_channels quadratic (saturation)
    scores["gene_weighted+sat"] = gene_score + scores["n_channels_quad"] * 0.1

    # =========================================================================
    # Evaluate each score via 5-fold C-index
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  RESULTS — C-INDEX PER SCORING LEVEL")
    print(f"{'='*90}")

    level_order = [
        "demographics",
        "n_channels",
        "n_channels_quad",
        "weighted_channels",
        "channel_pairs",
        "gene_weighted",
        "gene_pairs",
        "gene_weighted+sat",
    ]

    results = {}
    for level in level_order:
        s = scores[level]
        fold_cis = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ci = concordance_index(
                torch.tensor(s[val_idx].astype(np.float32)),
                torch.tensor(times[val_idx].astype(np.float32)),
                torch.tensor(events[val_idx].astype(np.float32)),
            )
            fold_cis.append(ci)

        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        results[level] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis}

    # Also compute full-dataset C-index for reference
    print(f"\n  {'Level':<25} {'Mean CI':>8} {'Std':>7}  {'Fold CIs'}")
    print(f"  {'-'*25} {'-'*8} {'-'*7}  {'-'*40}")

    for level in level_order:
        r = results[level]
        folds_str = "  ".join(f"{ci:.4f}" for ci in r["folds"])
        print(f"  {level:<25} {r['mean']:>8.4f} {r['std']:>7.4f}  {folds_str}")

    # Delta from n_channels baseline
    base = results["n_channels"]["mean"]
    print(f"\n  Deltas from n_channels ({base:.4f}):")
    for level in level_order:
        if level == "n_channels":
            continue
        delta = results[level]["mean"] - base
        print(f"    {level:<25} {delta:+.4f}")

    # Compare with V6c model
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    if os.path.exists(ua_path):
        with open(ua_path) as f:
            ua = json.load(f)
        v6c_mean = ua["means"].get("V6c", 0)
        print(f"\n  V6c model (unified ablation): {v6c_mean:.4f}")
        print(f"  Best simple score:            {max(r['mean'] for r in results.values()):.4f}")
        gap = v6c_mean - max(r["mean"] for r in results.values())
        print(f"  Gap (what the model adds):    {gap:+.4f}")

    print(f"\n{'='*90}")

    # Save
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
