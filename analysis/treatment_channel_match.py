"""
Treatment-channel matching analysis.

Tests whether therapy efficacy depends on channel matching:
- METABRIC: Does hormone therapy help when endocrine channel is load-bearing?
- TMB/immunotherapy: Does immunotherapy help when immune channel is load-bearing?
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Import channel mapping from main analysis
import sys
sys.path.insert(0, 'analysis')
from channel_analysis import (
    CHANNEL_MAP, CHANNEL_NAMES, DRIVER_MUTATION_TYPES,
    BASE, HEADERS, fetch_clinical, fetch_mutations
)


def load_data(study_id):
    """Load cached data or fetch."""
    cache_dir = "analysis/cache"
    os.makedirs(cache_dir, exist_ok=True)
    clinical_df = fetch_clinical(study_id, cache_dir)
    mut_df = fetch_mutations(study_id, cache_dir)
    return clinical_df, mut_df


def get_patient_channels(mut_df):
    """Get channel info per patient."""
    gene_col = "gene.hugoGeneSymbol"
    if gene_col not in mut_df.columns and "keyword" in mut_df.columns:
        mut_df[gene_col] = mut_df["keyword"].apply(
            lambda k: str(k).split()[0] if pd.notna(k) else None
        )

    driver = mut_df[mut_df["mutationType"].isin(DRIVER_MUTATION_TYPES)].copy()
    driver["channel"] = driver[gene_col].map(CHANNEL_MAP)
    mapped = driver.dropna(subset=["channel"])

    # Per patient
    all_patients = pd.DataFrame({"patientId": mut_df["patientId"].unique()})

    ch_count = mapped.groupby("patientId")["channel"].nunique().reset_index()
    ch_count.columns = ["patientId", "channel_count"]

    ch_set = mapped.groupby("patientId")["channel"].apply(set).reset_index()
    ch_set.columns = ["patientId", "channels_severed"]

    # Per-channel binary flags
    for channel in CHANNEL_NAMES:
        has = mapped[mapped["channel"] == channel].groupby("patientId").size().reset_index()
        has.columns = ["patientId", f"has_{channel}"]
        has[f"has_{channel}"] = 1
        all_patients = all_patients.merge(has[["patientId", f"has_{channel}"]], on="patientId", how="left")
        all_patients[f"has_{channel}"] = all_patients[f"has_{channel}"].fillna(0).astype(int)

    all_patients = all_patients.merge(ch_count, on="patientId", how="left")
    all_patients["channel_count"] = all_patients["channel_count"].fillna(0).astype(int)
    all_patients = all_patients.merge(ch_set, on="patientId", how="left")
    all_patients["channels_severed"] = all_patients["channels_severed"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    return all_patients


def analyze_metabric():
    """METABRIC: hormone therapy × endocrine channel matching."""
    print("\n" + "=" * 70)
    print("METABRIC: Hormone Therapy × Endocrine Channel")
    print("=" * 70)

    clinical_df, mut_df = load_data("brca_metabric")
    patients = get_patient_channels(mut_df)

    # Survival
    surv = clinical_df[["OS_STATUS", "OS_MONTHS"]].copy().reset_index()
    surv.columns = ["patientId", "os_status", "os_months"]
    surv["os_months"] = pd.to_numeric(surv["os_months"], errors="coerce")
    surv["event"] = surv["os_status"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    surv = surv.dropna(subset=["os_months"])

    # Treatment
    treat = clinical_df[["CHEMOTHERAPY", "HORMONE_THERAPY", "RADIO_THERAPY"]].copy().reset_index()
    treat.columns = ["patientId", "chemo", "hormone", "radiation"]

    # Merge all
    df = patients.merge(surv, on="patientId").merge(treat, on="patientId")
    df = df.dropna(subset=["hormone"])
    print(f"Patients with all data: {len(df)}")
    print(f"Deaths: {int(df['event'].sum())}")

    # The endocrine channel in breast cancer:
    # GATA3, FOXA1, ESR1 mutations indicate the endocrine channel is ACTIVE/load-bearing
    # (these are ER+ pathway genes - mutations here mean the tumor is endocrine-dependent)
    # Actually, the logic is:
    # - ER+ tumors: endocrine channel is load-bearing → hormone therapy should help
    # - TNBC: endocrine channel was never load-bearing → hormone therapy shouldn't help
    # The proxy: if tumor has NO endocrine-channel mutations (GATA3, FOXA1, ESR1),
    # the endocrine channel may not be the primary axis.
    # But actually in breast cancer, GATA3/FOXA1 mutations are found IN ER+ tumors —
    # they're drivers within the endocrine-coupled phenotype.

    # Better approach: use whether OTHER channels are severed.
    # Hormone therapy targets endocrine channel.
    # It should help more when fewer OTHER channels are severed (tumor is more dependent on endocrine).
    # It should help less when many channels are severed (tumor has already escaped multiple axes).

    # Split: hormone therapy × channel count
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Low channel count (0-1) — hormone therapy should help most
    ax = axes[0]
    kmf = KaplanMeierFitter()
    low_ch = df[df["channel_count"] <= 1]

    for label, mask, color in [
        ("Hormone therapy (n={n}, d={d})", low_ch["hormone"] == "YES", "#2ecc71"),
        ("No hormone therapy (n={n}, d={d})", low_ch["hormone"] == "NO", "#e74c3c"),
    ]:
        subset = low_ch[mask]
        actual_label = label.format(n=len(subset), d=int(subset["event"].sum()))
        if len(subset) < 10:
            continue
        kmf.fit(subset["os_months"], subset["event"], label=actual_label)
        kmf.plot_survival_function(ax=ax, color=color)

    ht_yes = low_ch[low_ch["hormone"] == "YES"]
    ht_no = low_ch[low_ch["hormone"] == "NO"]
    if len(ht_yes) > 10 and len(ht_no) > 10:
        lr = logrank_test(ht_yes["os_months"], ht_no["os_months"], ht_yes["event"], ht_no["event"])
        ax.text(0.95, 0.05, f"p={lr.p_value:.4e}", transform=ax.transAxes, ha="right",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("0-1 Channels Severed\n(Endocrine channel likely load-bearing)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Months", fontsize=11)
    ax.set_ylabel("Overall Survival", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    # Panel B: High channel count (3+) — hormone therapy should help less
    ax = axes[1]
    kmf = KaplanMeierFitter()
    high_ch = df[df["channel_count"] >= 3]

    for label, mask, color in [
        ("Hormone therapy (n={n}, d={d})", high_ch["hormone"] == "YES", "#2ecc71"),
        ("No hormone therapy (n={n}, d={d})", high_ch["hormone"] == "NO", "#e74c3c"),
    ]:
        subset = high_ch[mask]
        actual_label = label.format(n=len(subset), d=int(subset["event"].sum()))
        if len(subset) < 10:
            continue
        kmf.fit(subset["os_months"], subset["event"], label=actual_label)
        kmf.plot_survival_function(ax=ax, color=color)

    ht_yes = high_ch[high_ch["hormone"] == "YES"]
    ht_no = high_ch[high_ch["hormone"] == "NO"]
    if len(ht_yes) > 10 and len(ht_no) > 10:
        lr = logrank_test(ht_yes["os_months"], ht_no["os_months"], ht_yes["event"], ht_no["event"])
        ax.text(0.95, 0.05, f"p={lr.p_value:.4e}", transform=ax.transAxes, ha="right",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("3+ Channels Severed\n(Multiple escape routes, endocrine less dominant)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Months", fontsize=11)
    ax.set_ylabel("Overall Survival", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    plt.suptitle("METABRIC: Hormone Therapy Benefit by Channel Count", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    outdir = "analysis/results/metabric_treatment"
    os.makedirs(outdir, exist_ok=True)
    for ext in [".png", ".pdf"]:
        plt.savefig(f"{outdir}/hormone_therapy_by_channels{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {outdir}/")

    # Cox interaction model
    print("\n--- Cox PH: Hormone therapy × channel count interaction ---")
    cox_df = df[["os_months", "event", "channel_count"]].copy()
    cox_df["hormone_tx"] = (df["hormone"] == "YES").astype(int)
    cox_df["interaction"] = cox_df["hormone_tx"] * cox_df["channel_count"]
    cox_df = cox_df.dropna()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="os_months", event_col="event")
    cph.print_summary()

    # Summary stats
    print("\n--- Mortality by channel count × hormone therapy ---")
    for n_ch in range(5):
        for ht in ["YES", "NO"]:
            sub = df[(df["channel_count"] == n_ch) & (df["hormone"] == ht)]
            if len(sub) > 5:
                mort = 100 * sub["event"].mean()
                print(f"  {n_ch} ch, HT={ht}: {len(sub):4d} pts, {mort:5.1f}% mort")


def analyze_immunotherapy():
    """TMB dataset: immunotherapy × immune channel."""
    print("\n" + "=" * 70)
    print("TMB DATASET: Immunotherapy × Channel Architecture")
    print("=" * 70)

    clinical_df, mut_df = load_data("tmb_mskcc_2018")
    patients = get_patient_channels(mut_df)

    # Survival
    surv = clinical_df[["OS_STATUS", "OS_MONTHS"]].copy().reset_index()
    surv.columns = ["patientId", "os_status", "os_months"]
    surv["os_months"] = pd.to_numeric(surv["os_months"], errors="coerce")
    surv["event"] = surv["os_status"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    surv = surv.dropna(subset=["os_months"])

    # Drug type
    if "DRUG_TYPE" in clinical_df.columns:
        drug = clinical_df[["DRUG_TYPE"]].copy().reset_index()
        drug.columns = ["patientId", "drug_type"]
        surv = surv.merge(drug, on="patientId", how="left")

    # Cancer type
    if "CANCER_TYPE" in clinical_df.columns:
        ct = clinical_df[["CANCER_TYPE"]].copy().reset_index()
        ct.columns = ["patientId", "cancer_type"]
        surv = surv.merge(ct, on="patientId", how="left")

    df = patients.merge(surv, on="patientId")
    print(f"Patients: {len(df)}, Deaths: {int(df['event'].sum())}")

    # All patients got immunotherapy — so we test:
    # Does channel count predict response to immunotherapy?
    # And does having immune-channel mutations predict worse response?

    outdir = "analysis/results/tmb_immunotherapy"
    os.makedirs(outdir, exist_ok=True)

    # Figure 1: KM by channel count (all immunotherapy patients)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    kmf = KaplanMeierFitter()
    for label, mask, color in [
        ("0-1 channels", df["channel_count"] <= 1, "#2ecc71"),
        ("2 channels", df["channel_count"] == 2, "#f39c12"),
        ("3+ channels", df["channel_count"] >= 3, "#e74c3c"),
    ]:
        subset = df[mask]
        if len(subset) < 10:
            continue
        kmf.fit(subset["os_months"], subset["event"],
                label=f"{label} (n={len(subset)}, d={int(subset['event'].sum())})")
        kmf.plot_survival_function(ax=ax, color=color)

    low = df[df["channel_count"] <= 1]
    high = df[df["channel_count"] >= 3]
    if len(low) > 10 and len(high) > 10:
        lr = logrank_test(low["os_months"], high["os_months"], low["event"], high["event"])
        ax.text(0.95, 0.05, f"0-1 vs 3+: p={lr.p_value:.4e}", transform=ax.transAxes, ha="right",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("Immunotherapy Patients by Channel Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Months", fontsize=11)
    ax.set_ylabel("Overall Survival", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    # Figure 2: immune channel mutations vs not
    ax = axes[1]
    kmf = KaplanMeierFitter()

    has_immune = df["has_Immune"] == 1
    for label, mask, color in [
        ("Immune channel mutated", has_immune, "#e74c3c"),
        ("Immune channel intact", ~has_immune, "#2ecc71"),
    ]:
        subset = df[mask]
        if len(subset) < 10:
            continue
        kmf.fit(subset["os_months"], subset["event"],
                label=f"{label} (n={len(subset)}, d={int(subset['event'].sum())})")
        kmf.plot_survival_function(ax=ax, color=color)

    immune_yes = df[has_immune]
    immune_no = df[~has_immune]
    if len(immune_yes) > 10 and len(immune_no) > 10:
        lr = logrank_test(immune_yes["os_months"], immune_no["os_months"],
                         immune_yes["event"], immune_no["event"])
        ax.text(0.95, 0.05, f"p={lr.p_value:.4e}", transform=ax.transAxes, ha="right",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("Immunotherapy: Immune Channel Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Months", fontsize=11)
    ax.set_ylabel("Overall Survival", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    plt.suptitle("Immunotherapy Response by Coupling Architecture (TMB Cohort)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(f"{outdir}/immunotherapy_channels{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {outdir}/")

    # Cox
    print("\n--- Cox PH: Channel count for immunotherapy patients ---")
    cox_df = df[["os_months", "event", "channel_count", "has_Immune"]].copy().dropna()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="os_months", event_col="event")
    cph.print_summary()

    # Summary
    print("\n--- Mortality by immune channel status ---")
    for immune in [0, 1]:
        sub = df[df["has_Immune"] == immune]
        print(f"  Immune {'mutated' if immune else 'intact'}: {len(sub)} pts, "
              f"{100*sub['event'].mean():.1f}% mort")


def main():
    analyze_metabric()
    analyze_immunotherapy()


if __name__ == "__main__":
    main()
