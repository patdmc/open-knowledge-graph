#!/usr/bin/env python3
"""
Mutation position analysis: is the last mutation in a gene more protective?

For genes with multiple hotspots, extract the amino acid position from the
proteinChange string and test whether mutations later in the protein sequence
(closer to C-terminus) have different hazard shifts than earlier ones.

Hypothesis: late-position mutations are more protective because they disrupt
less of the protein — partial function vs complete loss.

Also: within a gene's mutation cluster, do later-position mutations show
the compensatory pattern expected from the escalation cascade?

Usage:
    python3 -u -m gnn.scripts.cluster_position
"""

import sys
import os
import re
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
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "cluster_position")
TIER_NAMES = {0: "Cell Intrinsic", 1: "Tissue Level", 2: "Organism Level", 3: "Meta Regulatory"}


def extract_position(protein_change):
    """Extract the amino acid position number from a proteinChange string.
    E.g. 'R175H' -> 175, 'K860Nfs*16' -> 860, 'X987_splice' -> 987
    """
    if not protein_change or protein_change == "nan":
        return None
    m = re.search(r'[A-Z*](\d+)', str(protein_change))
    if m:
        return int(m.group(1))
    return None


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 110)
    print("  MUTATION POSITION ANALYSIS")
    print("  Is the last mutation in a gene's sequence more protective?")
    print("=" * 110)

    # Load data + predictions
    print("\nLoading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

    events = data["events"].numpy().astype(int)
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
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)
    mut["position"] = mut["proteinChange"].apply(extract_position)
    mut = mut[mut["position"].notna()]
    mut["hotspot"] = mut["gene.hugoGeneSymbol"] + " " + mut["proteinChange"].astype(str)

    hazards = all_hazards.numpy()
    times_np = data["times"].numpy()
    events_np = data["events"].numpy()
    valid_mask = all_in_val.numpy()
    baseline_hazard = hazards[valid_mask].mean()

    # =========================================================================
    # Per-gene: get protein length estimate (max observed position)
    # =========================================================================
    gene_max_pos = mut.groupby("gene.hugoGeneSymbol")["position"].max().to_dict()

    # =========================================================================
    # For each hotspot: compute normalized position (0=N-term, 1=C-term)
    # =========================================================================
    print("\nComputing position-normalized hazard shifts...")

    hotspot_groups = mut.groupby("hotspot")
    hotspot_data = []

    for hotspot, group in hotspot_groups:
        pts = group["patient_idx"].unique()
        pts_valid = pts[valid_mask[pts]]
        if len(pts_valid) < 30:
            continue

        gene = group.iloc[0]["gene.hugoGeneSymbol"]
        position = group.iloc[0]["position"]
        max_pos = gene_max_pos.get(gene, position)
        norm_pos = position / max_pos if max_pos > 0 else 0.5

        hazard_shift = float(hazards[pts_valid].mean() - baseline_hazard)
        med_surv = float(np.median(times_np[pts_valid]))
        ev_rate = float(events_np[pts_valid].mean())

        channel = V6_CHANNEL_MAP.get(gene, "unknown")
        tier_idx = V6_TIER_MAP.get(channel, -1)

        hotspot_data.append({
            "hotspot": hotspot,
            "gene": gene,
            "channel": channel,
            "tier": TIER_NAMES.get(tier_idx, "?"),
            "mutationType": group.iloc[0]["mutationType"],
            "position": int(position),
            "max_position": int(max_pos),
            "norm_position": norm_pos,
            "n_patients": len(pts_valid),
            "hazard_shift": hazard_shift,
            "median_survival": med_surv,
            "event_rate": ev_rate,
        })

    print(f"  {len(hotspot_data)} hotspots with position data")

    # =========================================================================
    # Global correlation: position vs hazard shift
    # =========================================================================
    from scipy import stats as scipy_stats

    positions = np.array([h["norm_position"] for h in hotspot_data])
    shifts = np.array([h["hazard_shift"] for h in hotspot_data])
    abs_positions = np.array([h["position"] for h in hotspot_data])

    r_norm, p_norm = scipy_stats.pearsonr(positions, shifts)
    r_abs, p_abs = scipy_stats.pearsonr(abs_positions, shifts)

    print(f"\n{'='*110}")
    print(f"  GLOBAL CORRELATION: SEQUENCE POSITION vs HAZARD SHIFT")
    print(f"{'='*110}")
    print(f"  Normalized position (0=N, 1=C)  vs  shift:  r = {r_norm:+.4f}  p = {p_norm:.2e}")
    print(f"  Absolute position               vs  shift:  r = {r_abs:+.4f}  p = {p_abs:.2e}")
    print(f"\n  Negative r = C-terminal mutations are more protective")

    # =========================================================================
    # Bin by position (quintiles)
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  HAZARD SHIFT BY NORMALIZED POSITION (quintiles)")
    print(f"{'='*110}")

    quantiles = np.percentile(positions, [20, 40, 60, 80])
    bins = np.digitize(positions, quantiles)
    bin_labels = ["Q1 (N-term)", "Q2", "Q3", "Q4", "Q5 (C-term)"]

    print(f"  {'Quintile':<15} {'Pos range':>12} {'N_hot':>6} {'MeanShift':>10} "
          f"{'MedShift':>9} {'%Protective':>11} {'MeanEvR':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*6} {'-'*10} {'-'*9} {'-'*11} {'-'*8}")

    for b in range(5):
        mask = bins == b
        if mask.sum() == 0:
            continue
        b_shifts = shifts[mask]
        b_pos = positions[mask]
        b_ev = np.array([h["event_rate"] for h in hotspot_data])[mask]
        pct_prot = 100 * (b_shifts < 0).sum() / len(b_shifts)
        print(f"  {bin_labels[b]:<15} {np.min(b_pos):.2f}-{np.max(b_pos):.2f}"
              f"     {mask.sum():>6} {np.mean(b_shifts):>+10.4f} "
              f"{np.median(b_shifts):>+9.4f} {pct_prot:>10.1f}% {np.mean(b_ev):>8.3f}")

    # =========================================================================
    # Per-gene within-gene analysis: for genes with 3+ hotspots
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  WITHIN-GENE POSITION ANALYSIS (genes with 3+ hotspots)")
    print(f"  Is the last mutation in each gene more protective?")
    print(f"{'='*110}")

    gene_hotspots = defaultdict(list)
    for h in hotspot_data:
        gene_hotspots[h["gene"]].append(h)

    # Sort each gene's hotspots by position
    gene_results = []
    for gene, hs in gene_hotspots.items():
        if len(hs) < 3:
            continue
        hs_sorted = sorted(hs, key=lambda x: x["position"])
        positions_g = [x["position"] for x in hs_sorted]
        shifts_g = [x["hazard_shift"] for x in hs_sorted]

        # Correlation within gene
        if len(set(positions_g)) >= 3:
            r, p = scipy_stats.pearsonr(positions_g, shifts_g)
        else:
            r, p = 0, 1

        first_shift = shifts_g[0]
        last_shift = shifts_g[-1]
        channel = hs_sorted[0]["channel"]

        gene_results.append({
            "gene": gene,
            "channel": channel,
            "n_hotspots": len(hs),
            "first_pos": positions_g[0],
            "last_pos": positions_g[-1],
            "first_shift": first_shift,
            "last_shift": last_shift,
            "delta": last_shift - first_shift,
            "r": r,
            "p": p,
        })

    gene_results.sort(key=lambda x: x["delta"])

    print(f"\n  {'Gene':<12} {'Channel':<18} {'#Hot':>4} {'First':>6} {'Last':>6} "
          f"{'Sh_first':>9} {'Sh_last':>8} {'Delta':>8} {'r':>7}")
    print(f"  {'-'*12} {'-'*18} {'-'*4} {'-'*6} {'-'*6} "
          f"{'-'*9} {'-'*8} {'-'*8} {'-'*7}")

    for g in gene_results:
        print(f"  {g['gene']:<12} {g['channel']:<18} {g['n_hotspots']:>4} "
              f"{g['first_pos']:>6} {g['last_pos']:>6} "
              f"{g['first_shift']:>+9.4f} {g['last_shift']:>+8.4f} "
              f"{g['delta']:>+8.4f} {g['r']:>+7.3f}")

    # Summary stats
    deltas = [g["delta"] for g in gene_results]
    n_last_more_protective = sum(1 for d in deltas if d < 0)
    print(f"\n  Summary across {len(gene_results)} genes:")
    print(f"    Last mutation more protective: {n_last_more_protective}/{len(gene_results)} "
          f"({100*n_last_more_protective/len(gene_results):.0f}%)")
    print(f"    Mean delta (last - first): {np.mean(deltas):+.4f}")
    print(f"    Median delta: {np.median(deltas):+.4f}")

    # Within-gene correlations
    rs = [g["r"] for g in gene_results]
    print(f"    Mean within-gene r: {np.mean(rs):+.4f}")
    sig_neg = sum(1 for g in gene_results if g["r"] < -0.3)
    sig_pos = sum(1 for g in gene_results if g["r"] > 0.3)
    print(f"    Genes with r < -0.3 (C-term protective): {sig_neg}")
    print(f"    Genes with r > +0.3 (C-term harmful):    {sig_pos}")

    # =========================================================================
    # Per-gene detail: show position vs shift for top genes
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  DETAILED POSITION MAP FOR KEY GENES")
    print(f"{'='*110}")

    # Top genes by number of hotspots
    for gene in sorted(gene_hotspots.keys(),
                       key=lambda g: -len(gene_hotspots[g]))[:15]:
        hs = sorted(gene_hotspots[gene], key=lambda x: x["position"])
        channel = hs[0]["channel"]
        print(f"\n  {gene} ({channel}, {len(hs)} hotspots):")
        print(f"    {'Position':>8} {'Change':<20} {'Type':<15} {'N':>5} "
              f"{'Shift':>8} {'EvRate':>7}")

        for h in hs:
            change = h["hotspot"].split(" ", 1)[1] if " " in h["hotspot"] else "?"
            print(f"    {h['position']:>8} {change:<20} {h['mutationType']:<15} "
                  f"{h['n_patients']:>5} {h['hazard_shift']:>+8.4f} {h['event_rate']:>7.3f}")

    # =========================================================================
    # Mutation type at each position quintile
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  MUTATION TYPE DISTRIBUTION BY POSITION")
    print(f"{'='*110}")

    for b in range(5):
        mask = bins == b
        if mask.sum() == 0:
            continue
        types = [hotspot_data[i]["mutationType"] for i in range(len(hotspot_data)) if mask[i]]
        n_miss = sum(1 for t in types if "Missense" in str(t))
        n_trunc = sum(1 for t in types if any(x in str(t) for x in ["Frame_Shift", "Nonsense", "Splice"]))
        n_total = len(types)
        print(f"  {bin_labels[b]:<15}  Missense: {100*n_miss/n_total:.0f}%  "
              f"Truncating: {100*n_trunc/n_total:.0f}%  (N={n_total})")

    print(f"\n{'='*110}")

    # Save
    with open(os.path.join(SAVE_BASE, "hotspot_position_data.json"), "w") as f:
        json.dump(hotspot_data, f, indent=2, default=str)

    with open(os.path.join(SAVE_BASE, "gene_position_results.json"), "w") as f:
        json.dump(gene_results, f, indent=2, default=str)

    summary = {
        "n_hotspots": len(hotspot_data),
        "n_genes_analyzed": len(gene_results),
        "global_correlation": {"r": r_norm, "p": p_norm},
        "pct_last_more_protective": 100 * n_last_more_protective / max(len(gene_results), 1),
        "mean_delta_last_minus_first": float(np.mean(deltas)),
    }
    with open(os.path.join(SAVE_BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
