"""
Coupling-channel survival analysis — reusable across any cBioPortal study.

Usage:
    python3 analysis/channel_analysis.py <study_id> [--cache-dir analysis/cache]

Examples:
    python3 analysis/channel_analysis.py brca_tcga_pan_can_atlas_2018
    python3 analysis/channel_analysis.py msk_impact_2017
    python3 analysis/channel_analysis.py msk_met_2021
"""

import argparse
import os
import sys
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ============================================================
# cBioPortal API
# ============================================================

BASE = "https://www.cbioportal.org/api"
HEADERS = {"Accept": "application/json"}


def api_get(path, params=None):
    resp = requests.get(f"{BASE}{path}", headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def api_post(path, body):
    resp = requests.post(
        f"{BASE}{path}",
        json=body,
        headers={**HEADERS, "Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return resp.json()


# ============================================================
# Coupling-channel mapping
# ============================================================

CHANNEL_MAP = {
    # DDR channel
    "BRCA1": "DDR", "BRCA2": "DDR", "PALB2": "DDR",
    "RAD51C": "DDR", "RAD51D": "DDR", "RAD51B": "DDR",
    "ATM": "DDR", "ATR": "DDR", "CHEK2": "DDR", "CHEK1": "DDR",
    "FANCA": "DDR", "FANCC": "DDR", "FANCD2": "DDR",
    "BAP1": "DDR", "BARD1": "DDR",
    "MLH1": "DDR", "MSH2": "DDR", "MSH6": "DDR", "PMS2": "DDR",
    "POLE": "DDR", "POLD1": "DDR",

    # Cell cycle / apoptosis channel
    "TP53": "CellCycle", "RB1": "CellCycle",
    "CDKN1A": "CellCycle", "CDKN1B": "CellCycle",
    "CDKN2A": "CellCycle", "CDKN2B": "CellCycle",
    "CDK4": "CellCycle", "CDK6": "CellCycle",
    "CCND1": "CellCycle", "CCNE1": "CellCycle",
    "MDM2": "CellCycle", "MDM4": "CellCycle",
    "MYC": "CellCycle", "MYCN": "CellCycle",

    # PI3K / Growth signaling channel
    "PIK3CA": "PI3K_Growth", "PIK3R1": "PI3K_Growth",
    "PTEN": "PI3K_Growth", "AKT1": "PI3K_Growth",
    "AKT2": "PI3K_Growth", "AKT3": "PI3K_Growth",
    "MTOR": "PI3K_Growth",
    "KRAS": "PI3K_Growth", "NRAS": "PI3K_Growth", "HRAS": "PI3K_Growth",
    "BRAF": "PI3K_Growth", "RAF1": "PI3K_Growth",
    "MAP2K1": "PI3K_Growth", "MAP2K2": "PI3K_Growth",
    "MAP3K1": "PI3K_Growth", "MAP3K13": "PI3K_Growth",
    "ERBB2": "PI3K_Growth", "ERBB3": "PI3K_Growth",
    "EGFR": "PI3K_Growth", "FGFR1": "PI3K_Growth",
    "FGFR2": "PI3K_Growth", "FGFR3": "PI3K_Growth",
    "IGF1R": "PI3K_Growth", "MET": "PI3K_Growth",
    "NF1": "PI3K_Growth", "NF2": "PI3K_Growth",
    "TSC1": "PI3K_Growth", "TSC2": "PI3K_Growth",
    "STK11": "PI3K_Growth",
    "ARID1A": "PI3K_Growth",

    # Endocrine / hormone receptor channel
    "ESR1": "Endocrine", "ESR2": "Endocrine",
    "PGR": "Endocrine", "AR": "Endocrine",
    "FOXA1": "Endocrine",
    "GATA3": "Endocrine",

    # Immune surveillance channel
    "B2M": "Immune", "HLA-A": "Immune", "HLA-B": "Immune",
    "HLA-C": "Immune",
    "JAK1": "Immune", "JAK2": "Immune",
    "STAT1": "Immune",
    "CD274": "Immune",
    "PDCD1LG2": "Immune",
    "CTLA4": "Immune",

    # Tissue architecture channel
    "CDH1": "TissueArch", "CDH2": "TissueArch",
    "CTNNB1": "TissueArch",
    "APC": "TissueArch",
    "AXIN1": "TissueArch", "AXIN2": "TissueArch",
    "SMAD2": "TissueArch", "SMAD3": "TissueArch", "SMAD4": "TissueArch",
    "TGFBR1": "TissueArch", "TGFBR2": "TissueArch",
    "NOTCH1": "TissueArch", "NOTCH2": "TissueArch",
    "NOTCH3": "TissueArch", "NOTCH4": "TissueArch",
    "FBXW7": "TissueArch",
    "GJA1": "TissueArch",
    "GJB2": "TissueArch",
}

CHANNEL_NAMES = {
    "DDR": "DNA Damage Response",
    "CellCycle": "Cell Cycle / Apoptosis",
    "PI3K_Growth": "PI3K / Growth Signaling",
    "Endocrine": "Endocrine / Hormone",
    "Immune": "Immune Surveillance",
    "TissueArch": "Tissue Architecture",
}

CHANNEL_COLORS = {
    "DDR": "#e74c3c",
    "CellCycle": "#e67e22",
    "PI3K_Growth": "#f1c40f",
    "Endocrine": "#2ecc71",
    "Immune": "#3498db",
    "TissueArch": "#9b59b6",
}

DRIVER_MUTATION_TYPES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
}


# ============================================================
# Data fetching
# ============================================================

def fetch_clinical(study_id, cache_dir):
    """Fetch patient-level clinical data."""
    cache_path = os.path.join(cache_dir, f"{study_id}_clinical.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached clinical data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0)

    print(f"  Fetching clinical data for {study_id}...")
    data = api_get(f"/studies/{study_id}/clinical-data",
                   {"clinicalDataType": "PATIENT", "projection": "DETAILED"})
    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index="patientId", columns="clinicalAttributeId",
        values="value", aggfunc="first"
    )
    pivot.to_csv(cache_path)
    print(f"  Cached {len(pivot)} patients to {cache_path}")
    return pivot


def fetch_mutations(study_id, cache_dir):
    """Fetch somatic mutations."""
    cache_path = os.path.join(cache_dir, f"{study_id}_mutations.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached mutations from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Fetching mutations for {study_id}...")

    # Get samples
    samples = api_get(f"/studies/{study_id}/samples")
    sample_ids = [s["sampleId"] for s in samples]
    print(f"  {len(sample_ids)} samples")

    # Find mutation profile
    profiles = api_get(f"/studies/{study_id}/molecular-profiles")
    mut_profile = None
    for p in profiles:
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            mut_profile = p["molecularProfileId"]
            break
    if not mut_profile:
        raise ValueError("No mutation profile found")
    print(f"  Profile: {mut_profile}")

    # Fetch in batches
    batch_size = 500
    all_mutations = []
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i+batch_size]
        muts = api_post(
            f"/molecular-profiles/{mut_profile}/mutations/fetch",
            {"sampleIds": batch}
        )
        all_mutations.extend(muts)
        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i//batch_size + 1}/{(len(sample_ids)-1)//batch_size + 1}, "
                  f"mutations so far: {len(all_mutations)}")

    df = pd.json_normalize(all_mutations)

    # Extract gene symbol from keyword
    if "gene.hugoGeneSymbol" not in df.columns:
        if "keyword" in df.columns:
            df["gene.hugoGeneSymbol"] = df["keyword"].apply(
                lambda k: str(k).split()[0] if pd.notna(k) else None
            )

    df.to_csv(cache_path, index=False)
    print(f"  Cached {len(df)} mutations to {cache_path}")
    return df


# ============================================================
# Analysis
# ============================================================

def build_patient_df(clinical_df, mut_df):
    """Build patient-level dataframe with channel counts and survival."""
    gene_col = "gene.hugoGeneSymbol"

    # Filter to non-silent
    driver_mut = mut_df[mut_df["mutationType"].isin(DRIVER_MUTATION_TYPES)].copy()
    print(f"  Total mutations: {len(mut_df)}, non-silent: {len(driver_mut)}")

    # Map to channels
    driver_mut["channel"] = driver_mut[gene_col].map(CHANNEL_MAP)
    mapped = driver_mut.dropna(subset=["channel"])

    # Per-patient aggregations
    all_patients = pd.DataFrame({"patientId": mut_df["patientId"].unique()})

    # Channel count
    ch_count = mapped.groupby("patientId")["channel"].nunique().reset_index()
    ch_count.columns = ["patientId", "channel_count"]

    # Channel set
    ch_set = mapped.groupby("patientId")["channel"].apply(set).reset_index()
    ch_set.columns = ["patientId", "channels_severed"]

    # Driver mutation count
    drv_count = driver_mut.groupby("patientId")[gene_col].count().reset_index()
    drv_count.columns = ["patientId", "driver_mutation_count"]

    # Total mutation count
    tot_count = mut_df.groupby("patientId")[gene_col].count().reset_index()
    tot_count.columns = ["patientId", "total_mutation_count"]

    # Merge
    pdf = all_patients
    for right in [ch_count, ch_set, drv_count, tot_count]:
        pdf = pdf.merge(right, on="patientId", how="left")
    pdf["channel_count"] = pdf["channel_count"].fillna(0).astype(int)
    pdf["driver_mutation_count"] = pdf["driver_mutation_count"].fillna(0).astype(int)
    pdf["total_mutation_count"] = pdf["total_mutation_count"].fillna(0).astype(int)
    pdf["channels_severed"] = pdf["channels_severed"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    # Survival
    surv = clinical_df[["OS_STATUS", "OS_MONTHS"]].copy().reset_index()
    surv.columns = ["patientId", "os_status", "os_months"]
    surv["os_months"] = pd.to_numeric(surv["os_months"], errors="coerce")
    surv["event"] = surv["os_status"].apply(
        lambda x: 1 if "DECEASED" in str(x) else 0
    )
    surv = surv.dropna(subset=["os_months"])

    # Cancer type if available
    if "CANCER_TYPE" in clinical_df.columns:
        ct = clinical_df[["CANCER_TYPE"]].reset_index()
        ct.columns = ["patientId", "cancer_type"]
        surv = surv.merge(ct, on="patientId", how="left")

    if "CANCER_TYPE_DETAILED" in clinical_df.columns:
        ctd = clinical_df[["CANCER_TYPE_DETAILED"]].reset_index()
        ctd.columns = ["patientId", "cancer_type_detailed"]
        surv = surv.merge(ctd, on="patientId", how="left")

    df = pdf.merge(surv, on="patientId", how="inner")
    print(f"  Final: {len(df)} patients, {int(df['event'].sum())} deaths")
    return df


# ============================================================
# Plotting
# ============================================================

def plot_km_channels(df, title_suffix, output_path):
    """KM curves: channel count vs mutation count side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Channels
    ax = axes[0]
    kmf = KaplanMeierFitter()
    groups = [
        ("0 channels", df["channel_count"] == 0, "#3498db"),
        ("1 channel", df["channel_count"] == 1, "#2ecc71"),
        ("2 channels", df["channel_count"] == 2, "#f39c12"),
        ("3+ channels", df["channel_count"] >= 3, "#e74c3c"),
    ]
    for label, mask, color in groups:
        subset = df[mask]
        if len(subset) < 10:
            continue
        kmf.fit(subset["os_months"], subset["event"],
                label=f"{label} (n={len(subset)}, d={int(subset['event'].sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Months", fontsize=12)
    ax.set_ylabel("Overall Survival", fontsize=12)
    ax.set_title(f"By Coupling Channels Severed\n{title_suffix}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    # p-value
    low = df[df["channel_count"] <= 1]
    high = df[df["channel_count"] >= 3]
    if len(low) > 10 and len(high) > 10:
        lr = logrank_test(low["os_months"], high["os_months"], low["event"], high["event"])
        ax.text(0.95, 0.05, f"0-1 vs 3+: p={lr.p_value:.2e}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Panel B: Mutation count
    ax = axes[1]
    kmf = KaplanMeierFitter()
    q33 = df["driver_mutation_count"].quantile(0.33)
    q67 = df["driver_mutation_count"].quantile(0.67)
    if q33 == q67:
        q33 = df["driver_mutation_count"].median() - 1
        q67 = df["driver_mutation_count"].median() + 1

    for label, mask, color in [
        (f"Low (≤{int(q33)})", df["driver_mutation_count"] <= q33, "#2ecc71"),
        (f"Med ({int(q33)+1}-{int(q67)})", (df["driver_mutation_count"] > q33) & (df["driver_mutation_count"] <= q67), "#f39c12"),
        (f"High (>{int(q67)})", df["driver_mutation_count"] > q67, "#e74c3c"),
    ]:
        subset = df[mask]
        if len(subset) < 10:
            continue
        kmf.fit(subset["os_months"], subset["event"],
                label=f"{label} (n={len(subset)}, d={int(subset['event'].sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Months", fontsize=12)
    ax.set_ylabel("Overall Survival", fontsize=12)
    ax.set_title(f"By Non-Silent Mutation Count\n{title_suffix}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    low_m = df[df["driver_mutation_count"] <= q33]
    high_m = df[df["driver_mutation_count"] > q67]
    if len(low_m) > 10 and len(high_m) > 10:
        lr = logrank_test(low_m["os_months"], high_m["os_months"], low_m["event"], high_m["event"])
        ax.text(0.95, 0.05, f"Low vs High: p={lr.p_value:.2e}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_path + ext, dpi=300, bbox_inches="tight")
    plt.close()


def plot_by_channel(df, title_suffix, output_path):
    """KM curves for each specific channel."""
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()

    for channel, name in CHANNEL_NAMES.items():
        has = df["channels_severed"].apply(lambda s: channel in s)
        subset = df[has]
        if len(subset) < 15:
            continue
        kmf.fit(subset["os_months"], subset["event"],
                label=f"{name} (n={len(subset)}, d={int(subset['event'].sum())})")
        kmf.plot_survival_function(ax=ax, color=CHANNEL_COLORS.get(channel, "gray"))

    # Baseline
    no_ch = df[df["channel_count"] == 0]
    if len(no_ch) >= 15:
        kmf.fit(no_ch["os_months"], no_ch["event"],
                label=f"No channel mutations (n={len(no_ch)}, d={int(no_ch['event'].sum())})")
        kmf.plot_survival_function(ax=ax, color="gray", linestyle="--")

    ax.set_xlabel("Months", fontsize=12)
    ax.set_ylabel("Overall Survival", fontsize=12)
    ax.set_title(f"By Specific Channel Severed\n{title_suffix}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_path + ext, dpi=300, bbox_inches="tight")
    plt.close()


def run_cox(df, label):
    """Cox PH regression comparing channel count vs mutation count."""
    cox_df = df[["os_months", "event", "channel_count", "driver_mutation_count"]].copy()
    cox_df = cox_df.dropna()
    cox_df["log_mut"] = np.log1p(cox_df["driver_mutation_count"])

    results = {}

    print(f"\n{'='*60}")
    print(f"COX PH: {label}")
    print(f"{'='*60}")

    for name, cols in [
        ("Channel count only", ["channel_count"]),
        ("Log mutation count only", ["log_mut"]),
        ("Both (multivariate)", ["channel_count", "log_mut"]),
    ]:
        cph = CoxPHFitter()
        try:
            cph.fit(cox_df[["os_months", "event"] + cols],
                    duration_col="os_months", event_col="event")
            print(f"\n--- {name} ---")
            cph.print_summary()
            results[name] = {
                "concordance": cph.concordance_index_,
                "AIC": cph.AIC_partial_,
            }
        except Exception as e:
            print(f"\n--- {name} --- FAILED: {e}")

    return results


def print_summary(df, label):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {label}")
    print(f"{'='*60}")
    print(f"Patients: {len(df)}")
    print(f"Deaths: {int(df['event'].sum())} ({100*df['event'].mean():.1f}%)")
    print(f"Median follow-up: {df['os_months'].median():.1f} months")

    print(f"\nChannel count → mortality:")
    for n in sorted(df["channel_count"].unique()):
        sub = df[df["channel_count"] == n]
        d = sub["event"].sum()
        print(f"  {n} ch: {len(sub):5d} pts, {int(d):4d} deaths ({100*d/len(sub):5.1f}%)")

    print(f"\nChannel frequency:")
    all_ch = []
    for s in df["channels_severed"]:
        all_ch.extend(list(s))
    freq = pd.Series(all_ch).value_counts()
    for ch, count in freq.items():
        print(f"  {CHANNEL_NAMES.get(ch, ch):30s}: {count:5d} ({100*count/len(df):.1f}%)")

    # Cross-channel vs same-channel
    cross = df[df["channel_count"] >= 2]
    same = df[(df["driver_mutation_count"] >= 2) & (df["channel_count"] == 1)]
    print(f"\nCross-channel (2+ channels): {len(cross)} pts, "
          f"{100*cross['event'].mean():.1f}% mortality")
    print(f"Same-channel (2+ muts, 1 ch): {len(same)} pts, "
          f"{100*same['event'].mean():.1f}% mortality")
    if len(cross) > 10 and len(same) > 10:
        lr = logrank_test(cross["os_months"], same["os_months"],
                          cross["event"], same["event"])
        print(f"Log-rank p: {lr.p_value:.4e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Coupling-channel survival analysis")
    parser.add_argument("study_id", help="cBioPortal study ID")
    parser.add_argument("--cache-dir", default="analysis/cache", help="Cache directory")
    args = parser.parse_args()

    study_id = args.study_id
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    output_dir = f"analysis/results/{study_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Analyzing {study_id} ===\n")

    # Fetch
    clinical_df = fetch_clinical(study_id, cache_dir)
    mut_df = fetch_mutations(study_id, cache_dir)

    # Build
    df = build_patient_df(clinical_df, mut_df)
    df.to_csv(f"{output_dir}/patient_data.csv", index=False)

    # Analyze
    print_summary(df, study_id)
    cox_results = run_cox(df, study_id)

    # Plot
    plot_km_channels(df, study_id, f"{output_dir}/km_channels_vs_mutations")
    plot_by_channel(df, study_id, f"{output_dir}/km_by_channel")

    print(f"\nResults saved to {output_dir}/")

    # Save summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(df, study_id)
        f.write(buf.getvalue())

    return df


if __name__ == "__main__":
    main()
