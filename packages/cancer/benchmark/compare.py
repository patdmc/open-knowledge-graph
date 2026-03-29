#!/usr/bin/env python3
"""
Head-to-head benchmark comparison against published GNN models.

Compares our Coupling-Channel GNN against:
  - Cox-Sage (C-index on 7 TCGA cohorts)
  - Multilevel GNN (AUC on GBM, LGG, KIRC)
  - Simple baselines (Cox PH with channel count, random forest)
"""

import os
import json
import numpy as np
import pandas as pd

from ..config import GNN_RESULTS
from .cox_sage import get_cox_sage_reported_results


def get_multilevel_gnn_reported_results():
    """Published results from Prior Knowledge-Guided Multilevel GNN.

    From Table 3 of the paper (Briefings in Bioinformatics, 2024):
    5-year AUC values.
    """
    return {
        "GBM": {"auc_5yr": 0.827, "std": 0.021},
        "LGG": {"auc_5yr": 0.862, "std": 0.015},
        "KIRC": {"auc_5yr": 0.798, "std": 0.033},
    }


def get_mogonet_reported_results():
    """Published results from MOGONET.

    From Table 2 of the paper (Nature Communications, 2021):
    Classification accuracy for subtype prediction.
    """
    return {
        "BRCA": {"accuracy": 0.853, "f1": 0.845},
        "LGG": {"accuracy": 0.872, "f1": 0.861},
        "KIPAN": {"accuracy": 0.964, "f1": 0.962},
    }


def compile_comparison_table(our_results_path=None):
    """Build comparison table of our results vs published benchmarks.

    Args:
        our_results_path: path to our cv_summary.json

    Returns:
        DataFrame with comparison
    """
    rows = []

    # Cox-Sage benchmarks
    cox_sage = get_cox_sage_reported_results()
    for cancer, r in cox_sage.items():
        rows.append({
            "Cancer Type": cancer,
            "Metric": "C-index",
            "Cox-Sage": f"{r['c_index']:.3f} +/- {r['std']:.3f}",
            "Coupling-Channel GNN": "TBD",
            "Dataset": "TCGA",
        })

    # Multilevel GNN benchmarks
    mlgnn = get_multilevel_gnn_reported_results()
    for cancer, r in mlgnn.items():
        rows.append({
            "Cancer Type": cancer,
            "Metric": "5yr AUC",
            "Multilevel GNN": f"{r['auc_5yr']:.3f} +/- {r['std']:.3f}",
            "Coupling-Channel GNN": "TBD",
            "Dataset": "TCGA",
        })

    df = pd.DataFrame(rows)

    # Load our results if available
    if our_results_path and os.path.exists(our_results_path):
        with open(our_results_path) as f:
            our = json.load(f)
        ci = our.get("mean_c_index", 0)
        std = our.get("std_c_index", 0)
        n = our.get("n_patients", 0)
        print(f"\nOur results: C-index = {ci:.4f} +/- {std:.4f} (N={n})")

    return df


def print_comparison_table():
    """Pretty-print the full comparison."""
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPARISON: Coupling-Channel GNN vs Published Models")
    print("=" * 80)

    print("\n--- Cox-Sage (Briefings in Bioinformatics, 2025) ---")
    print("  Task: Survival prediction (C-index)")
    print("  Their data: TCGA, 7 cancer types, ~3K patients total")
    print("  Our data: MSK-IMPACT, pan-cancer, 43,872 patients")
    print()
    cox_sage = get_cox_sage_reported_results()
    print(f"  {'Cancer':<12} {'Cox-Sage C-index':<20}")
    print(f"  {'-'*12} {'-'*20}")
    for cancer, r in cox_sage.items():
        print(f"  {cancer:<12} {r['c_index']:.3f} +/- {r['std']:.3f}")

    print("\n--- Multilevel GNN (Briefings in Bioinformatics, 2024) ---")
    print("  Task: Risk stratification (5-year AUC)")
    print("  Their data: TCGA multi-omics, 3 cancer types")
    print()
    mlgnn = get_multilevel_gnn_reported_results()
    print(f"  {'Cancer':<12} {'Multilevel GNN 5yr AUC':<25}")
    print(f"  {'-'*12} {'-'*25}")
    for cancer, r in mlgnn.items():
        print(f"  {cancer:<12} {r['auc_5yr']:.3f} +/- {r['std']:.3f}")

    print("\n--- Key advantage ---")
    print("  These models use gene expression (continuous, expensive)")
    print("  We use somatic mutations only (binary, standard NGS panel)")
    print("  If we match or beat them with less data, the framework wins.")


if __name__ == "__main__":
    print_comparison_table()
