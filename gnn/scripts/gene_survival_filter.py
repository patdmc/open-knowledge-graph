#!/usr/bin/env python3
"""
Gene survival filter: which expanded genes worsen vs improve survival?

For each gene added in V6 (beyond the original 99), compute a per-gene
hazard ratio using simple Cox-style comparison:
  - Patients WITH mutation in gene vs WITHOUT
  - Within each cancer type (to avoid confounding)
  - Pooled across types via inverse-variance weighting

Genes where mutation associates with BETTER survival are either:
  1. Compensatory (body fighting back)
  2. Passenger (noise correlated with good-prognosis types)
  3. Wrong channel assignment

These get flagged for removal or sign-flip.

Usage:
    python3 -u -m gnn.scripts.gene_survival_filter
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, NON_SILENT, MSK_DATASETS, GNN_RESULTS
from gnn.data.channel_dataset_v6 import V6_CHANNEL_MAP, V6_CHANNEL_NAMES


def compute_gene_hazard_ratio(gene, mut_df, clin_df, min_mutated=20, min_types=2):
    """Compute pooled hazard ratio for a gene across cancer types.

    Returns:
        hr: hazard ratio (>1 = worse survival with mutation, <1 = better)
        n_types: number of cancer types contributing
        n_mutated: total mutated patients
        detail: per-type breakdown
    """
    # Patients with this gene mutated
    mutated_patients = set(mut_df[mut_df["gene.hugoGeneSymbol"] == gene]["patientId"].unique())

    if len(mutated_patients) < min_mutated:
        return None

    # Per cancer type: compare median survival
    type_results = []

    for ct, ct_group in clin_df.groupby("CANCER_TYPE"):
        ct_patients = set(ct_group["patientId"])
        mut_in_type = mutated_patients & ct_patients
        unmut_in_type = ct_patients - mutated_patients

        if len(mut_in_type) < 5 or len(unmut_in_type) < 20:
            continue

        # Get survival for mutated vs unmutated
        mut_surv = ct_group[ct_group["patientId"].isin(mut_in_type)]
        unmut_surv = ct_group[ct_group["patientId"].isin(unmut_in_type)]

        # Use restricted mean survival time (RMST) — robust, no model assumptions
        # Truncate at median follow-up to avoid censoring bias
        t_max = min(ct_group["time"].median() * 2, ct_group["time"].quantile(0.9))

        def rmst(df, t_max):
            """Crude RMST: mean of min(time, t_max) weighted by event."""
            t = df["time"].clip(upper=t_max)
            # For censored patients, use their censored time as lower bound
            return t.mean()

        rmst_mut = rmst(mut_surv, t_max)
        rmst_unmut = rmst(unmut_surv, t_max)

        if rmst_unmut == 0:
            continue

        # Log ratio: >0 means mutation associated with LONGER survival
        log_ratio = np.log(rmst_mut / rmst_unmut) if rmst_mut > 0 else -2.0

        # Weight by sample size (inverse variance approximation)
        weight = 1.0 / (1.0/len(mut_in_type) + 1.0/len(unmut_in_type))

        type_results.append({
            "cancer_type": ct,
            "n_mut": len(mut_in_type),
            "n_unmut": len(unmut_in_type),
            "rmst_mut": rmst_mut,
            "rmst_unmut": rmst_unmut,
            "log_ratio": log_ratio,
            "weight": weight,
        })

    if len(type_results) < min_types:
        return None

    # Inverse-variance weighted pooled log-ratio
    total_weight = sum(r["weight"] for r in type_results)
    pooled_log_ratio = sum(r["log_ratio"] * r["weight"] for r in type_results) / total_weight

    # Convert back: >1 means mutation = longer survival (compensatory)
    # <1 means mutation = shorter survival (true severance)
    survival_ratio = np.exp(pooled_log_ratio)

    return {
        "gene": gene,
        "survival_ratio": survival_ratio,  # >1 = better survival with mut
        "n_types": len(type_results),
        "n_mutated": len(mutated_patients),
        "detail": type_results,
    }


def main():
    print("=" * 75)
    print("  GENE SURVIVAL FILTER")
    print("  Which expanded genes are severance vs compensatory?")
    print("=" * 75)

    # Load data
    paths = MSK_DATASETS["msk_impact_50k"]
    mut = pd.read_csv(paths["mutations"], low_memory=False)
    clin = pd.read_csv(paths["clinical"], low_memory=False)

    # Parse survival
    clin = clin[clin["OS_STATUS"].notna() & clin["OS_MONTHS"].notna()].copy()
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    clin["time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin = clin.dropna(subset=["time"])
    clin = clin[clin["time"] > 0]

    # Add cancer type
    if "sample_clinical" in paths and os.path.exists(paths["sample_clinical"]):
        sample_clin = pd.read_csv(paths["sample_clinical"], low_memory=False)
        ct = sample_clin.drop_duplicates("patientId")[["patientId", "CANCER_TYPE"]]
        clin = clin.merge(ct, on="patientId", how="left")
    clin["CANCER_TYPE"] = clin["CANCER_TYPE"].fillna("Unknown")

    # Non-silent mutations only
    mut_ns = mut[mut["mutationType"].isin(NON_SILENT)].copy()

    # Identify new genes (in V6 but not original 99)
    original_genes = set(CHANNEL_MAP.keys())
    v6_genes = set(V6_CHANNEL_MAP.keys())
    new_genes = v6_genes - original_genes

    print(f"  Original curated: {len(original_genes)} genes")
    print(f"  V6 total: {len(v6_genes)} genes")
    print(f"  New genes to evaluate: {len(new_genes)}")

    # Also evaluate original genes as calibration
    print(f"\n  Evaluating survival associations...")

    all_results = []

    for i, gene in enumerate(sorted(v6_genes)):
        result = compute_gene_hazard_ratio(gene, mut_ns, clin)
        if result:
            result["is_new"] = gene in new_genes
            result["channel"] = V6_CHANNEL_MAP.get(gene, "?")
            all_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(v6_genes)} genes evaluated...")

    print(f"  {len(all_results)} genes with sufficient data")

    # Separate new vs original
    new_results = [r for r in all_results if r["is_new"]]
    orig_results = [r for r in all_results if not r["is_new"]]

    # Calibration: original genes should mostly be <1 (mutation = worse)
    orig_ratios = [r["survival_ratio"] for r in orig_results]
    print(f"\n  CALIBRATION (original 99 curated genes):")
    print(f"  Mean survival ratio: {np.mean(orig_ratios):.3f} (expect <1.0)")
    print(f"  Genes where mutation = worse survival (ratio < 1.0): "
          f"{sum(1 for r in orig_ratios if r < 1.0)}/{len(orig_ratios)}")
    print(f"  Genes where mutation = better survival (ratio > 1.0): "
          f"{sum(1 for r in orig_ratios if r > 1.0)}/{len(orig_ratios)}")

    # New genes
    print(f"\n  NEW GENES ({len(new_results)} evaluated):")
    severance = [r for r in new_results if r["survival_ratio"] < 1.0]
    compensatory = [r for r in new_results if r["survival_ratio"] >= 1.0]
    print(f"  Severance (mutation = worse, ratio < 1.0): {len(severance)}")
    print(f"  Compensatory (mutation = better, ratio >= 1.0): {len(compensatory)}")

    # Print compensatory genes — these should be removed
    print(f"\n  {'─'*72}")
    print(f"  COMPENSATORY GENES (remove from channel map)")
    print(f"  {'─'*72}")
    compensatory.sort(key=lambda r: -r["survival_ratio"])
    print(f"  {'Gene':<15} {'Channel':<18} {'Ratio':>7} {'N_mut':>6} {'Types':>6}  Direction")
    print(f"  {'─'*15} {'─'*18} {'─'*7} {'─'*6} {'─'*6}  {'─'*20}")
    for r in compensatory:
        direction = "STRONG compensatory" if r["survival_ratio"] > 1.10 else "weak compensatory"
        print(f"  {r['gene']:<15} {r['channel']:<18} {r['survival_ratio']:>7.3f} "
              f"{r['n_mutated']:>6} {r['n_types']:>6}  {direction}")

    # Print strongest severance genes — these are keepers
    print(f"\n  {'─'*72}")
    print(f"  STRONGEST SEVERANCE GENES (keep in channel map)")
    print(f"  {'─'*72}")
    severance.sort(key=lambda r: r["survival_ratio"])
    print(f"  {'Gene':<15} {'Channel':<18} {'Ratio':>7} {'N_mut':>6} {'Types':>6}")
    print(f"  {'─'*15} {'─'*18} {'─'*7} {'─'*6} {'─'*6}")
    for r in severance[:30]:
        print(f"  {r['gene']:<15} {r['channel']:<18} {r['survival_ratio']:>7.3f} "
              f"{r['n_mutated']:>6} {r['n_types']:>6}")

    # Per-channel summary
    print(f"\n  {'─'*72}")
    print(f"  PER-CHANNEL SUMMARY")
    print(f"  {'─'*72}")
    for ch in V6_CHANNEL_NAMES:
        ch_new = [r for r in new_results if r["channel"] == ch]
        if not ch_new:
            continue
        n_sev = sum(1 for r in ch_new if r["survival_ratio"] < 1.0)
        n_comp = sum(1 for r in ch_new if r["survival_ratio"] >= 1.0)
        mean_ratio = np.mean([r["survival_ratio"] for r in ch_new])
        print(f"  {ch:<18} {len(ch_new):>3} new genes | "
              f"{n_sev} severance, {n_comp} compensatory | "
              f"mean ratio: {mean_ratio:.3f}")

    # Build filtered gene list
    keep_genes = set(CHANNEL_MAP.keys())  # Always keep original 99
    removed_genes = set()

    for r in new_results:
        if r["survival_ratio"] < 1.05:  # Keep if mutation = worse or neutral
            keep_genes.add(r["gene"])
        else:
            removed_genes.add(r["gene"])

    # Also keep new genes we couldn't evaluate (low frequency) — conservative
    unevaluated = new_genes - {r["gene"] for r in new_results}
    # Don't add unevaluated — they're low-frequency and likely noise

    print(f"\n{'='*75}")
    print(f"  FILTERED GENE SET")
    print(f"  Original: {len(original_genes)}")
    print(f"  V6 expanded: {len(v6_genes)}")
    print(f"  After filter: {len(keep_genes)}")
    print(f"  Removed (compensatory): {len(removed_genes)}")
    print(f"  Removed (unevaluated/low-freq): {len(unevaluated)}")
    print(f"{'='*75}")

    # Save
    out = {
        "keep_genes": {g: V6_CHANNEL_MAP[g] for g in sorted(keep_genes) if g in V6_CHANNEL_MAP},
        "removed_compensatory": {g: V6_CHANNEL_MAP[g] for g in sorted(removed_genes) if g in V6_CHANNEL_MAP},
        "removed_unevaluated": list(sorted(unevaluated)),
        "all_results": [
            {"gene": r["gene"], "channel": r["channel"], "survival_ratio": r["survival_ratio"],
             "n_mutated": r["n_mutated"], "n_types": r["n_types"], "is_new": r["is_new"]}
            for r in sorted(all_results, key=lambda r: r["survival_ratio"])
        ],
    }
    out_path = os.path.join(GNN_RESULTS, "gene_survival_filter.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
