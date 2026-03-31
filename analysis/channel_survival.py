"""
Channel-count vs survival analysis using TCGA-BRCA data from cBioPortal.

Maps somatic mutations to coupling channels and tests whether the number
of distinct coupling channels severed predicts overall survival better
than raw mutation count.
"""

import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import json
import os

BASE = "https://www.cbioportal.org/api"
STUDY = "brca_tcga_pan_can_atlas_2018"
HEADERS = {"Accept": "application/json"}

# ============================================================
# Coupling-channel mapping
# ============================================================
# Each gene is assigned to the coupling channel it participates in.
# A mutation in a gene severs (or weakens) that channel.
# Genes can appear in multiple channels if they are cross-channel hubs.
#
# Channels:
#   1. DDR        — DNA damage response / homologous recombination
#   2. Cell cycle  — cell cycle checkpoints / apoptosis
#   3. PI3K/Growth — PI3K-AKT-mTOR / RAS-MAPK growth signaling
#   4. Endocrine   — hormone receptor signaling
#   5. Immune       — immune surveillance / antigen presentation
#   6. Tissue arch  — cell adhesion, tissue architecture, Wnt, Notch, TGF-beta

def _load_channel_map():
    """Load canonical 8-channel gene map from CSV."""
    import csv as _csv
    _csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "channel_gene_map.csv")
    if not os.path.exists(_csv_path):
        _csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "channel_gene_map.csv")
    result = {}
    with open(_csv_path) as f:
        for row in _csv.DictReader(f):
            ch = row["channel"]
            if ch == "TissueArchitecture":
                ch = "TissueArch"
            result[row["gene"]] = ch
    return result


CHANNEL_MAP = _load_channel_map()

CHANNEL_NAMES = {
    "DDR": "DNA Damage Response",
    "CellCycle": "Cell Cycle / Apoptosis",
    "PI3K_Growth": "PI3K / Growth Signaling",
    "Endocrine": "Endocrine / Hormone",
    "Immune": "Immune Surveillance",
    "TissueArch": "Tissue Architecture",
    "ChromatinRemodel": "Chromatin Remodeling",
    "DNAMethylation": "DNA Methylation",
}


def fetch_clinical_data():
    """Fetch patient-level clinical data from cBioPortal."""
    print("Fetching clinical data...")
    url = f"{BASE}/studies/{STUDY}/clinical-data"
    params = {"clinicalDataType": "PATIENT", "projection": "DETAILED"}
    resp = requests.get(url, params=params, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index="patientId",
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first"
    )
    print(f"  Got clinical data for {len(pivot)} patients")
    return pivot


def fetch_mutations():
    """Fetch somatic mutations from cBioPortal."""
    print("Fetching mutation data...")

    # Get sample list
    url = f"{BASE}/studies/{STUDY}/samples"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    samples = resp.json()
    sample_ids = [s["sampleId"] for s in samples]
    print(f"  Found {len(sample_ids)} samples")

    # Find mutation profile
    url = f"{BASE}/studies/{STUDY}/molecular-profiles"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    profiles = resp.json()
    mut_profile = None
    for p in profiles:
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            mut_profile = p["molecularProfileId"]
            break

    if not mut_profile:
        raise ValueError("No mutation profile found")
    print(f"  Using profile: {mut_profile}")

    # Fetch mutations in batches
    batch_size = 500
    all_mutations = []
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i+batch_size]
        url = f"{BASE}/molecular-profiles/{mut_profile}/mutations/fetch"
        body = {
            "sampleIds": batch,
        }
        resp = requests.post(
            url,
            json=body,
            headers={**HEADERS, "Content-Type": "application/json"}
        )
        resp.raise_for_status()
        all_mutations.extend(resp.json())
        print(f"  Fetched batch {i//batch_size + 1}, total mutations so far: {len(all_mutations)}")

    df = pd.json_normalize(all_mutations)

    # Gene symbol is not directly in the response; extract from 'keyword' field
    # keyword format: "GENENAME PXXX missense" or similar
    if "gene.hugoGeneSymbol" not in df.columns:
        if "keyword" in df.columns:
            df["gene.hugoGeneSymbol"] = df["keyword"].apply(
                lambda k: str(k).split()[0] if pd.notna(k) else None
            )
        else:
            # Fallback: resolve entrezGeneId via API
            raise ValueError("No keyword or gene symbol in mutation data")

    print(f"  Total mutations: {len(df)}")
    print(f"  Unique genes: {df['gene.hugoGeneSymbol'].nunique()}")
    return df


def map_to_channels(mut_df):
    """Map mutations to coupling channels per patient."""
    # Get gene symbol column
    gene_col = "gene.hugoGeneSymbol" if "gene.hugoGeneSymbol" in mut_df.columns else "hugoGeneSymbol"
    patient_col = "patientId"

    # Map each mutation to a channel
    mut_df["channel"] = mut_df[gene_col].map(CHANNEL_MAP)

    # Count distinct channels per patient (only mapped genes)
    mapped = mut_df.dropna(subset=["channel"])
    channels_per_patient = mapped.groupby(patient_col)["channel"].nunique().reset_index()
    channels_per_patient.columns = [patient_col, "channel_count"]

    # Also count total mutations per patient (all genes)
    total_muts = mut_df.groupby(patient_col)[gene_col].count().reset_index()
    total_muts.columns = [patient_col, "mutation_count"]

    # Count distinct mutated genes per patient
    gene_count = mut_df.groupby(patient_col)[gene_col].nunique().reset_index()
    gene_count.columns = [patient_col, "gene_count"]

    # Which channels are severed per patient
    channel_detail = mapped.groupby(patient_col)["channel"].apply(set).reset_index()
    channel_detail.columns = [patient_col, "channels_severed"]

    result = channels_per_patient.merge(total_muts, on=patient_col)
    result = result.merge(gene_count, on=patient_col)
    result = result.merge(channel_detail, on=patient_col)

    return result


def build_survival_df(clinical_df, channel_df):
    """Merge clinical and channel data into survival dataframe."""
    # Extract OS data — clinical_df index is patientId
    surv = clinical_df[["OS_STATUS", "OS_MONTHS"]].copy()
    surv = surv.reset_index()  # patientId becomes a column
    surv.columns = ["patientId", "os_status", "os_months"]
    surv["os_months"] = pd.to_numeric(surv["os_months"], errors="coerce")

    surv["event"] = surv["os_status"].apply(
        lambda x: 1 if "DECEASED" in str(x) or str(x) == "1" or str(x) == "1:DECEASED" else 0
    )
    surv = surv.dropna(subset=["os_months"])

    # Merge
    merged = surv.merge(channel_df, on="patientId", how="inner")
    print(f"\nMerged dataset: {len(merged)} patients with both survival and mutation data")
    print(f"  Events (deaths): {merged['event'].sum()}")
    print(f"  Channel count distribution:")
    print(merged["channel_count"].value_counts().sort_index().to_string())

    return merged


def plot_km_by_channels(df, output_path):
    """Kaplan-Meier curves stratified by number of coupling channels severed."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: By channel count ---
    ax = axes[0]
    kmf = KaplanMeierFitter()

    # Group: 0-1 channels, 2 channels, 3+ channels
    bins = {
        "0-1 channels": df["channel_count"] <= 1,
        "2 channels": df["channel_count"] == 2,
        "3+ channels": df["channel_count"] >= 3,
    }
    colors = {"0-1 channels": "#2ecc71", "2 channels": "#f39c12", "3+ channels": "#e74c3c"}

    for label, mask in bins.items():
        subset = df[mask]
        if len(subset) < 5:
            continue
        kmf.fit(subset["os_months"], subset["event"], label=f"{label} (n={len(subset)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[label])

    ax.set_xlabel("Months", fontsize=12)
    ax.set_ylabel("Overall Survival Probability", fontsize=12)
    ax.set_title("Survival by Coupling Channels Severed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 200)

    # Log-rank test: 0-1 vs 3+
    low = df[df["channel_count"] <= 1]
    high = df[df["channel_count"] >= 3]
    if len(low) > 5 and len(high) > 5:
        result = logrank_test(low["os_months"], high["os_months"], low["event"], high["event"])
        ax.text(0.95, 0.05, f"0-1 vs 3+: p={result.p_value:.4f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Panel B: By mutation count (comparison) ---
    ax = axes[1]
    kmf = KaplanMeierFitter()

    # Use similar tercile grouping for mutation count
    q33 = df["mutation_count"].quantile(0.33)
    q67 = df["mutation_count"].quantile(0.67)

    mut_bins = {
        f"Low mutations (≤{int(q33)})": df["mutation_count"] <= q33,
        f"Med mutations ({int(q33)+1}-{int(q67)})": (df["mutation_count"] > q33) & (df["mutation_count"] <= q67),
        f"High mutations (>{int(q67)})": df["mutation_count"] > q67,
    }
    mut_colors = list(colors.values())

    for (label, mask), color in zip(mut_bins.items(), mut_colors):
        subset = df[mask]
        if len(subset) < 5:
            continue
        kmf.fit(subset["os_months"], subset["event"], label=f"{label} (n={len(subset)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Months", fontsize=12)
    ax.set_ylabel("Overall Survival Probability", fontsize=12)
    ax.set_title("Survival by Total Mutation Count", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 200)

    # Log-rank test: low vs high mutations
    low_m = df[df["mutation_count"] <= q33]
    high_m = df[df["mutation_count"] > q67]
    if len(low_m) > 5 and len(high_m) > 5:
        result = logrank_test(low_m["os_months"], high_m["os_months"], low_m["event"], high_m["event"])
        ax.text(0.95, 0.05, f"Low vs High: p={result.p_value:.4f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {output_path}")
    plt.close()


def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal patients: {len(df)}")
    print(f"Deaths: {df['event'].sum()} ({100*df['event'].mean():.1f}%)")
    print(f"Median follow-up: {df['os_months'].median():.1f} months")

    print(f"\nChannel count distribution:")
    for n in sorted(df["channel_count"].unique()):
        subset = df[df["channel_count"] == n]
        deaths = subset["event"].sum()
        print(f"  {n} channels: {len(subset)} patients, {deaths} deaths ({100*deaths/len(subset):.1f}%)")

    print(f"\nMutation count: median={df['mutation_count'].median():.0f}, "
          f"mean={df['mutation_count'].mean():.1f}, "
          f"range=[{df['mutation_count'].min()}, {df['mutation_count'].max()}]")

    print(f"\nChannel frequency (how often each channel is hit):")
    all_channels = []
    for channels in df["channels_severed"]:
        all_channels.extend(list(channels))
    channel_freq = pd.Series(all_channels).value_counts()
    for ch, count in channel_freq.items():
        print(f"  {CHANNEL_NAMES.get(ch, ch)}: {count} patients ({100*count/len(df):.1f}%)")


def main():
    os.makedirs("analysis", exist_ok=True)

    # Fetch data
    clinical_df = fetch_clinical_data()
    mut_df = fetch_mutations()

    # Save raw data for reproducibility
    clinical_df.to_csv("analysis/clinical_raw.csv")
    mut_df.to_csv("analysis/mutations_raw.csv", index=False)
    print("Saved raw data to analysis/")

    # Map to channels
    channel_df = map_to_channels(mut_df)

    # Build survival dataframe
    surv_df = build_survival_df(clinical_df, channel_df)
    surv_df.to_csv("analysis/survival_channels.csv", index=False)

    # Print stats
    print_summary_stats(surv_df)

    # Plot
    plot_km_by_channels(surv_df, "analysis/channel_vs_mutation_survival.png")

    # Also save a version suitable for the paper
    plot_km_by_channels(surv_df, "analysis/channel_vs_mutation_survival.pdf")


if __name__ == "__main__":
    main()
