#!/usr/bin/env python3
"""
Directional edge analysis: order-dependent mutation interactions.

The same two mutations produce opposite outcomes depending on which occurs first.
We use mutation frequency as a proxy for temporal order (more common = likely earlier).

For each pair, compute:
  forward_hazard  = actual outcome when A is primary (more common), B is secondary
  reverse_hazard  = actual outcome when B is primary, A is secondary
  directional_gap = forward_hazard - reverse_hazard

Then build a directional scoring function and test against C-index to see
if directional edges compete with the transformer model.

Usage:
    python3 -u -m gnn.scripts.directional_edges
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, ANALYSIS_CACHE
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, CHANNEL_FEAT_DIM_V6, V6_TIER_MAP,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "directional_edges")
MIN_HOTSPOT_PATIENTS = 30
MIN_CO_OCCUR = 15


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 110)
    print("  DIRECTIONAL EDGE ANALYSIS")
    print("  Order-dependent mutation interactions as a scoring function")
    print("=" * 110)

    # Load data + predictions
    print("\nLoading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)

    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
        "hidden_dim": 128,
        "cross_channel_heads": 4,
        "cross_channel_layers": 2,
        "dropout": 0.3,
        "n_cancer_types": n_ct,
    }

    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {
                "channel_features": data["channel_features"][idx],
                "tier_features": data["tier_features"][idx],
                "cancer_type_idx": data["cancer_type_idx"][idx],
                "age": data["age"][idx],
                "sex": data["sex"][idx],
                "msi_score": data["msi_score"][idx],
                "msi_high": data["msi_high"][idx],
                "tmb": data["tmb"][idx],
            }
            with torch.no_grad():
                h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    print(f"  {all_in_val.sum().item()} patients with predictions")

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline_hazard = hazards[valid_mask].mean()

    # Load mutations
    print("\nLoading mutations...")
    import pandas as pd

    mut_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    mutations = pd.read_csv(mut_path, low_memory=False,
                            usecols=["patientId", "gene.hugoGeneSymbol",
                                     "proteinChange", "mutationType"])

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
    mut = mutations[
        mutations["gene.hugoGeneSymbol"].isin(channel_genes) &
        mutations["proteinChange"].notna() &
        (mutations["proteinChange"] != "")
    ].copy()
    mut["hotspot"] = mut["gene.hugoGeneSymbol"] + " " + mut["proteinChange"].astype(str)
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)

    # Hotspot frequency (global count = proxy for evolutionary age / order)
    hotspot_freq = mut.groupby("hotspot")["patient_idx"].nunique()
    valid_hotspots = hotspot_freq[hotspot_freq >= MIN_HOTSPOT_PATIENTS].index

    mut_valid = mut[mut["hotspot"].isin(valid_hotspots)]

    # Per-patient hotspot sets
    patient_hotspots = defaultdict(set)
    for _, row in mut_valid.iterrows():
        patient_hotspots[row["patient_idx"]].add(row["hotspot"])

    # Per-hotspot patient sets and shifts
    hotspot_patients = {}
    hotspot_shift = {}
    for hs, group in mut_valid.groupby("hotspot"):
        pts = set(group["patient_idx"].unique())
        hotspot_patients[hs] = pts
        pts_valid = np.array(sorted(pts))
        pts_valid = pts_valid[valid_mask[pts_valid]]
        if len(pts_valid) >= 10:
            hotspot_shift[hs] = float(hazards[pts_valid].mean() - baseline_hazard)

    hotspot_list = sorted(hotspot_shift.keys())
    freq_lookup = hotspot_freq.to_dict()

    # Per-gene shift (for gene-level scoring)
    gene_patients = defaultdict(set)
    for _, row in mut.iterrows():
        if row["patient_idx"] in range(N):
            gene_patients[row["gene.hugoGeneSymbol"]].add(row["patient_idx"])

    gene_shift = {}
    for gene, pts in gene_patients.items():
        pts_valid = np.array(sorted(pts))
        pts_valid = pts_valid[valid_mask[pts_valid]]
        if len(pts_valid) >= 20:
            gene_shift[gene] = float(hazards[pts_valid].mean() - baseline_hazard)

    # =========================================================================
    # Build directional edge table
    # =========================================================================
    print("\nBuilding directional edge table...")

    # For each pair with co-occurrence, compute directional synergy
    # Primary = more frequent mutation (likely earlier)
    # Secondary = less frequent (likely later)
    edges = []

    for i in range(len(hotspot_list)):
        ha = hotspot_list[i]
        pts_a = hotspot_patients[ha]
        fa = freq_lookup.get(ha, 0)

        for j in range(i + 1, len(hotspot_list)):
            hb = hotspot_list[j]
            pts_b = hotspot_patients[hb]
            fb = freq_lookup.get(hb, 0)

            co = pts_a & pts_b
            co_valid = np.array(sorted(co))
            if len(co_valid) == 0:
                continue
            co_valid = co_valid[valid_mask[co_valid]]
            if len(co_valid) < MIN_CO_OCCUR:
                continue

            sa = hotspot_shift[ha]
            sb = hotspot_shift[hb]
            actual = float(hazards[co_valid].mean() - baseline_hazard)
            expected = sa + sb
            synergy = actual - expected

            # Direction: more frequent = primary
            if fa >= fb:
                primary, secondary = ha, hb
                primary_shift, secondary_shift = sa, sb
                primary_freq, secondary_freq = fa, fb
            else:
                primary, secondary = hb, ha
                primary_shift, secondary_shift = sb, sa
                primary_freq, secondary_freq = fb, fa

            # Directional edge weight:
            # How much does the secondary mutation change the outcome relative to
            # what we'd expect from the primary alone?
            # rescue_value = primary_shift - actual (positive = secondary helped)
            rescue_value = primary_shift - actual + secondary_shift

            meta_a = mut_valid[mut_valid["hotspot"] == ha].iloc[0]
            meta_b = mut_valid[mut_valid["hotspot"] == hb].iloc[0]
            gene_a = meta_a["gene.hugoGeneSymbol"]
            gene_b = meta_b["gene.hugoGeneSymbol"]
            ch_a = V6_CHANNEL_MAP.get(gene_a, "?")
            ch_b = V6_CHANNEL_MAP.get(gene_b, "?")
            tier_a = V6_TIER_MAP.get(ch_a, -1)
            tier_b = V6_TIER_MAP.get(ch_b, -1)

            edges.append({
                "primary": primary,
                "secondary": secondary,
                "primary_shift": primary_shift,
                "secondary_shift": secondary_shift,
                "primary_freq": int(primary_freq),
                "secondary_freq": int(secondary_freq),
                "expected": expected,
                "actual": actual,
                "synergy": synergy,
                "rescue_value": rescue_value,
                "n_co_occur": len(co_valid),
                "gene_primary": gene_a if primary == ha else gene_b,
                "gene_secondary": gene_b if primary == ha else gene_a,
                "channel_primary": ch_a if primary == ha else ch_b,
                "channel_secondary": ch_b if primary == ha else ch_a,
                "tier_primary": tier_a if primary == ha else tier_b,
                "tier_secondary": tier_b if primary == ha else tier_a,
                "cross_channel": ch_a != ch_b,
                "tier_direction": "up" if (tier_b if primary == ha else tier_a) > (tier_a if primary == ha else tier_b) else
                                  "down" if (tier_b if primary == ha else tier_a) < (tier_a if primary == ha else tier_b) else "same",
            })

    print(f"  {len(edges)} directional edges")

    # =========================================================================
    # Tier direction analysis
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  TIER DIRECTION vs SYNERGY")
    print(f"  Does escalating UP the tier hierarchy (lower→higher) produce rescue?")
    print(f"{'='*110}")

    for direction in ["up", "down", "same"]:
        dir_edges = [e for e in edges if e["tier_direction"] == direction]
        if not dir_edges:
            continue
        syns = [e["synergy"] for e in dir_edges]
        rescues = [e["rescue_value"] for e in dir_edges]
        pct_rescue = 100 * sum(1 for s in syns if s < 0) / len(syns)
        print(f"  {direction:>6}: N={len(dir_edges):>4}, mean_synergy={np.mean(syns):+.4f}, "
              f"mean_rescue={np.mean(rescues):+.4f}, {pct_rescue:.1f}% rescue")

    # =========================================================================
    # Build per-patient directional score
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  BUILDING PER-PATIENT DIRECTIONAL SCORE")
    print(f"{'='*110}")

    # Edge lookup: (primary, secondary) -> synergy
    edge_lookup = {}
    for e in edges:
        edge_lookup[(e["primary"], e["secondary"])] = e["synergy"]

    # For each patient with 2+ hotspots:
    # 1. Order their hotspots by frequency (most frequent first = likely primary)
    # 2. Sum the directional edge weights for each consecutive pair
    # 3. Add to the gene-level hazard sum

    # Scoring functions to compare:
    scores = {}

    # Score 1: gene_weighted (baseline from signal_decomposition)
    gene_score = np.zeros(N)
    patient_genes_map = defaultdict(set)
    for _, row in mut.iterrows():
        idx = row.get("patient_idx")
        if idx is not None and 0 <= idx < N:
            gene = row["gene.hugoGeneSymbol"]
            patient_genes_map[idx].add(gene)
            if gene in gene_shift:
                gene_score[idx] += gene_shift[gene]  # only count once per gene
    # Deduplicate: only count each gene once
    gene_score_dedup = np.zeros(N)
    for idx, genes in patient_genes_map.items():
        for g in genes:
            if g in gene_shift:
                gene_score_dedup[idx] += gene_shift[g]
    scores["gene_weighted"] = gene_score_dedup

    # Score 2: hotspot_weighted (sum of per-hotspot shifts)
    hotspot_score = np.zeros(N)
    for idx in range(N):
        for hs in patient_hotspots.get(idx, set()):
            if hs in hotspot_shift:
                hotspot_score[idx] += hotspot_shift[hs]
    scores["hotspot_weighted"] = hotspot_score

    # Score 3: directional (hotspot_weighted + directional edge corrections)
    directional_score = hotspot_score.copy()
    n_edges_applied = 0
    for idx in range(N):
        hs_list = sorted(patient_hotspots.get(idx, set()),
                        key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) < 2:
            continue
        for i in range(len(hs_list)):
            for j in range(i + 1, len(hs_list)):
                # hs_list[i] is more frequent (primary), hs_list[j] is secondary
                key = (hs_list[i], hs_list[j])
                if key in edge_lookup:
                    directional_score[idx] += edge_lookup[key]
                    n_edges_applied += 1
    scores["directional"] = directional_score
    print(f"  Applied {n_edges_applied} directional edge corrections across {N} patients")

    # Score 4: directional + tier bonus
    # Add bonus for secondary mutations that escalate UP the tier hierarchy
    tier_bonus_score = directional_score.copy()
    for idx in range(N):
        hs_list = sorted(patient_hotspots.get(idx, set()),
                        key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) < 2:
            continue
        for i in range(len(hs_list)):
            for j in range(i + 1, len(hs_list)):
                hi = hs_list[i]
                hj = hs_list[j]
                # Get tiers
                gi = mut_valid[mut_valid["hotspot"] == hi]
                gj = mut_valid[mut_valid["hotspot"] == hj]
                if len(gi) == 0 or len(gj) == 0:
                    continue
                gene_i = gi.iloc[0]["gene.hugoGeneSymbol"]
                gene_j = gj.iloc[0]["gene.hugoGeneSymbol"]
                ch_i = V6_CHANNEL_MAP.get(gene_i, "?")
                ch_j = V6_CHANNEL_MAP.get(gene_j, "?")
                ti = V6_TIER_MAP.get(ch_i, -1)
                tj = V6_TIER_MAP.get(ch_j, -1)
                if tj > ti:
                    # Secondary escalates UP — this is the defense direction
                    tier_bonus_score[idx] -= 0.1  # protective bonus
                elif tj < ti:
                    # Secondary goes DOWN — collapse direction
                    tier_bonus_score[idx] += 0.1  # harmful penalty
    scores["directional+tier"] = tier_bonus_score

    # Score 5: gene_weighted + simple directional correction at gene level
    gene_dir_score = gene_score_dedup.copy()
    # Use gene-level synergy from mutation_synergy results
    synergy_path = os.path.join(GNN_RESULTS, "mutation_synergy", "pair_results.json")
    if os.path.exists(synergy_path):
        with open(synergy_path) as f:
            gene_pairs = json.load(f)
        gene_pair_lookup = {}
        for p in gene_pairs:
            fa_g = freq_lookup.get(p["gene_a"], 0)
            fb_g = freq_lookup.get(p["gene_b"], 0)
            # Use gene frequency sum as proxy
            if fa_g >= fb_g:
                key = (p["gene_a"], p["gene_b"])
            else:
                key = (p["gene_b"], p["gene_a"])
            gene_pair_lookup[key] = p["synergy"]

        for idx, genes in patient_genes_map.items():
            gene_list = sorted(genes)
            for i in range(len(gene_list)):
                for j in range(i + 1, len(gene_list)):
                    key = tuple(sorted([gene_list[i], gene_list[j]]))
                    # Try both orderings
                    syn = gene_pair_lookup.get((gene_list[i], gene_list[j]),
                          gene_pair_lookup.get((gene_list[j], gene_list[i]), None))
                    if syn is not None:
                        gene_dir_score[idx] += syn
    scores["gene_weighted+synergy"] = gene_dir_score

    # =========================================================================
    # Evaluate all scoring functions via 5-fold C-index
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  C-INDEX COMPARISON: SCORING FUNCTIONS")
    print(f"{'='*110}")

    folds = list(skf.split(np.arange(N), events))

    level_order = [
        "gene_weighted",
        "hotspot_weighted",
        "directional",
        "directional+tier",
        "gene_weighted+synergy",
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

    print(f"\n  {'Level':<25} {'Mean CI':>8} {'Std':>7}  Fold CIs")
    print(f"  {'-'*25} {'-'*8} {'-'*7}  {'-'*40}")
    for level in level_order:
        r = results[level]
        folds_str = "  ".join(f"{ci:.4f}" for ci in r["folds"])
        print(f"  {level:<25} {r['mean']:>8.4f} {r['std']:>7.4f}  {folds_str}")

    # Compare with V6c model
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    if os.path.exists(ua_path):
        with open(ua_path) as f:
            ua = json.load(f)
        v6c_mean = ua["means"].get("V6c", 0)
        best_simple = max(r["mean"] for r in results.values())
        print(f"\n  V6c transformer:     {v6c_mean:.4f}")
        print(f"  Best directional:    {best_simple:.4f}")
        print(f"  Gap:                 {v6c_mean - best_simple:+.4f}")

    # Also compare with signal_decomposition baselines
    sd_path = os.path.join(GNN_RESULTS, "signal_decomposition", "results.json")
    if os.path.exists(sd_path):
        with open(sd_path) as f:
            sd = json.load(f)
        for level in ["gene_weighted", "gene_pairs", "gene_weighted+sat"]:
            if level in sd:
                print(f"  signal_decomp {level}: {sd[level]['mean']:.4f}")

    # =========================================================================
    # Directional edge examples
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  TOP DIRECTIONAL EDGES (by absolute synergy)")
    print(f"{'='*110}")

    edges.sort(key=lambda x: -abs(x["synergy"]))

    print(f"  {'Primary':<22} {'Secondary':<22} {'PriSh':>7} {'SecSh':>7} "
          f"{'Actual':>7} {'Syn':>7} {'Dir':>5} {'N':>4}")
    print(f"  {'-'*22} {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*4}")

    for e in edges[:40]:
        print(f"  {e['primary']:<22} {e['secondary']:<22} "
              f"{e['primary_shift']:>+7.3f} {e['secondary_shift']:>+7.3f} "
              f"{e['actual']:>+7.3f} {e['synergy']:>+7.3f} "
              f"{e['tier_direction']:>5} {e['n_co_occur']:>4}")

    # =========================================================================
    # Save
    # =========================================================================
    with open(os.path.join(SAVE_BASE, "directional_edges.json"), "w") as f:
        json.dump(edges, f, indent=2, default=str)

    with open(os.path.join(SAVE_BASE, "scoring_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary = {
        "n_edges": len(edges),
        "n_edges_applied": n_edges_applied,
        "scoring_results": {k: v["mean"] for k, v in results.items()},
    }
    with open(os.path.join(SAVE_BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
