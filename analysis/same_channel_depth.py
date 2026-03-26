"""
Plot: within a fixed channel count, more mutations = better survival.

This is the direct visualization of the protective mutation-count effect.
For patients with the same number of channels severed, stratify by
mutation count and show that more mutations (deeper same-channel commitment)
predicts better survival.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os


def plot_same_channel_depth(study_id, study_label):
    """For a fixed channel count, show that more mutations = better survival."""
    df = pd.read_csv(f"analysis/results/{study_id}/patient_data.csv")
    df = df.dropna(subset=["os_months", "event"])

    # Focus on patients with exactly 2 channels severed (largest group with meaningful variation)
    # and also do 1 channel
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for idx, n_ch in enumerate([1, 2, 3]):
        ax = axes[idx]
        kmf = KaplanMeierFitter()
        subset = df[df["channel_count"] == n_ch].copy()

        if len(subset) < 50:
            ax.set_title(f"{n_ch} channel(s): too few patients")
            continue

        # Split by mutation count: low vs high within this channel group
        median_mut = subset["driver_mutation_count"].median()
        q25 = subset["driver_mutation_count"].quantile(0.25)
        q75 = subset["driver_mutation_count"].quantile(0.75)

        groups = [
            (f"Few mutations (≤{int(q25)})", subset["driver_mutation_count"] <= q25, "#e74c3c"),
            (f"Many mutations (>{int(q75)})", subset["driver_mutation_count"] > q75, "#2ecc71"),
        ]

        for label, mask, color in groups:
            sub = subset[mask]
            if len(sub) < 20:
                continue
            kmf.fit(sub["os_months"], sub["event"],
                    label=f"{label} (n={len(sub)}, d={int(sub['event'].sum())})")
            kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

        # Log-rank
        low = subset[subset["driver_mutation_count"] <= q25]
        high = subset[subset["driver_mutation_count"] > q75]
        if len(low) > 20 and len(high) > 20:
            lr = logrank_test(low["os_months"], high["os_months"],
                            low["event"], high["event"])
            ax.text(0.95, 0.05, f"p={lr.p_value:.4e}",
                    transform=ax.transAxes, ha="right", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_title(f"{n_ch} Channel{'s' if n_ch > 1 else ''} Severed\n"
                     f"(same channels, different mutation depth)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Months", fontsize=11)
        ax.set_ylabel("Overall Survival", fontsize=11)
        ax.legend(fontsize=9, loc="lower left")
        ax.set_ylim(0, 1.05)

    plt.suptitle(f"Same Channel Count, Different Mutation Depth — {study_label}\n"
                 f"More mutations in the same channels = better survival",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    outdir = f"analysis/results/{study_id}"
    for ext in [".png", ".pdf"]:
        plt.savefig(f"{outdir}/same_channel_depth{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {outdir}/same_channel_depth")

    # Print the numbers
    print(f"\n{'='*60}")
    print(f"SAME-CHANNEL DEPTH: {study_label}")
    print(f"{'='*60}")
    for n_ch in [1, 2, 3]:
        subset = df[df["channel_count"] == n_ch]
        if len(subset) < 50:
            continue
        q25 = subset["driver_mutation_count"].quantile(0.25)
        q75 = subset["driver_mutation_count"].quantile(0.75)
        low = subset[subset["driver_mutation_count"] <= q25]
        high = subset[subset["driver_mutation_count"] > q75]
        if len(low) > 10 and len(high) > 10:
            lr = logrank_test(low["os_months"], high["os_months"],
                            low["event"], high["event"])
            print(f"  {n_ch} ch: few muts (≤{int(q25)}): {len(low)} pts, "
                  f"{100*low['event'].mean():.1f}% mort | "
                  f"many muts (>{int(q75)}): {len(high)} pts, "
                  f"{100*high['event'].mean():.1f}% mort | "
                  f"p={lr.p_value:.4e}")


def main():
    for study_id, label in [
        ("msk_impact_2017", "MSK-IMPACT 2017"),
        ("msk_met_2021", "MSK MetTropism"),
        ("msk_impact_50k_2026", "MSK-IMPACT 50K"),
    ]:
        path = f"analysis/results/{study_id}/patient_data.csv"
        if os.path.exists(path):
            plot_same_channel_depth(study_id, label)
        else:
            print(f"Skipping {study_id}: no patient_data.csv")


if __name__ == "__main__":
    main()
