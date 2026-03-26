"""
Pathogenic-variant sensitivity analysis for coupling-channel model.

Addresses reviewer concern that VUS inflate channel counts by running
the core analysis under three increasingly stringent mutation filters:

  Tier 1 — All non-silent (baseline)
  Tier 2 — Truncating only (Nonsense, Frameshift, Splice_Site)
  Tier 3 — Truncating + recurrent hotspot missense (position with >=5 patients)

For each tier: Cox PH (channel_count + log mutation count), cross-vs-same
channel comparison, and (Tier 2) severance hierarchy.
"""

import os
import sys
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# Import channel definitions from existing module
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from channel_analysis import CHANNEL_MAP, CHANNEL_NAMES, DRIVER_MUTATION_TYPES

# ============================================================
# Constants
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "pathogenic_variants")

TRUNCATING_TYPES = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"
}

HOTSPOT_THRESHOLD = 5  # min patients at a protein position to call it a hotspot

# ============================================================
# Data loading
# ============================================================

def load_data():
    mut_path = os.path.join(DATA_DIR, "msk_impact_50k_2026_mutations.csv")
    clin_path = os.path.join(DATA_DIR, "msk_impact_50k_2026_clinical.csv")

    mut = pd.read_csv(mut_path)
    clin = pd.read_csv(clin_path)

    # Clinical: survival
    clin["os_months"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin["event"] = clin["OS_STATUS"].apply(
        lambda x: 1 if "DECEASED" in str(x) else 0
    )
    clin = clin[["patientId", "os_months", "event"]].dropna(subset=["os_months"])

    return mut, clin

# ============================================================
# Tier filters
# ============================================================

def filter_tier1(mut):
    """All non-silent mutations (baseline)."""
    return mut[mut["mutationType"].isin(DRIVER_MUTATION_TYPES)].copy()


def filter_tier2(mut):
    """Truncating mutations only."""
    return mut[mut["mutationType"].isin(TRUNCATING_TYPES)].copy()


def find_hotspots(mut):
    """Identify hotspot missense positions: gene + proteinPosStart with >= HOTSPOT_THRESHOLD patients."""
    missense = mut[mut["mutationType"] == "Missense_Mutation"].copy()
    missense = missense.dropna(subset=["proteinPosStart"])
    missense["proteinPosStart"] = missense["proteinPosStart"].astype(int)

    # Count distinct patients per gene+position
    pos_counts = (
        missense
        .groupby(["gene.hugoGeneSymbol", "proteinPosStart"])["patientId"]
        .nunique()
        .reset_index()
    )
    pos_counts.columns = ["gene", "pos", "n_patients"]
    hotspots = pos_counts[pos_counts["n_patients"] >= HOTSPOT_THRESHOLD].copy()
    hotspots = hotspots.sort_values("n_patients", ascending=False)
    return hotspots


def filter_tier3(mut, hotspots):
    """Truncating + hotspot missense."""
    trunc = mut[mut["mutationType"].isin(TRUNCATING_TYPES)]

    # Hotspot missense
    missense = mut[mut["mutationType"] == "Missense_Mutation"].copy()
    missense = missense.dropna(subset=["proteinPosStart"])
    missense["proteinPosStart"] = missense["proteinPosStart"].astype(int)

    hotspot_keys = set(zip(hotspots["gene"], hotspots["pos"]))
    is_hotspot = missense.apply(
        lambda r: (r["gene.hugoGeneSymbol"], int(r["proteinPosStart"])) in hotspot_keys,
        axis=1
    )
    hotspot_missense = missense[is_hotspot]

    combined = pd.concat([trunc, hotspot_missense], ignore_index=True)
    return combined

# ============================================================
# Patient-level aggregation
# ============================================================

def build_patient_df(filtered_mut, clin):
    """Map filtered mutations to channels and merge with survival."""
    gene_col = "gene.hugoGeneSymbol"

    filtered_mut = filtered_mut.copy()
    filtered_mut["channel"] = filtered_mut[gene_col].map(CHANNEL_MAP)
    mapped = filtered_mut.dropna(subset=["channel"])

    # Unique patients from filtered mutations
    all_patients = pd.DataFrame({"patientId": filtered_mut["patientId"].unique()})

    # Channel count per patient
    ch_count = mapped.groupby("patientId")["channel"].nunique().reset_index()
    ch_count.columns = ["patientId", "channel_count"]

    # Channel set per patient
    ch_set = mapped.groupby("patientId")["channel"].apply(set).reset_index()
    ch_set.columns = ["patientId", "channels_severed"]

    # Driver mutation count (all filtered, not just in channel genes)
    drv_count = filtered_mut.groupby("patientId")[gene_col].count().reset_index()
    drv_count.columns = ["patientId", "driver_mutation_count"]

    pdf = all_patients
    for right in [ch_count, ch_set, drv_count]:
        pdf = pdf.merge(right, on="patientId", how="left")
    pdf["channel_count"] = pdf["channel_count"].fillna(0).astype(int)
    pdf["driver_mutation_count"] = pdf["driver_mutation_count"].fillna(0).astype(int)
    pdf["channels_severed"] = pdf["channels_severed"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    df = pdf.merge(clin, on="patientId", how="inner")
    return df

# ============================================================
# Analysis functions
# ============================================================

def run_cox_multivariate(df, label):
    """Run multivariate Cox: channel_count + log(driver_mutation_count + 1)."""
    cox_df = df[["os_months", "event", "channel_count", "driver_mutation_count"]].copy()
    cox_df = cox_df.dropna()
    cox_df = cox_df[cox_df["os_months"] > 0]
    cox_df["log_mut"] = np.log1p(cox_df["driver_mutation_count"])

    results = {}

    print(f"\n--- Cox PH: {label} ---")
    print(f"N = {len(cox_df)}, events = {int(cox_df['event'].sum())}")

    # Multivariate
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df[["os_months", "event", "channel_count", "log_mut"]],
                duration_col="os_months", event_col="event")

        for var in ["channel_count", "log_mut"]:
            hr = np.exp(cph.params_[var])
            p = cph.summary.loc[var, "p"]
            ci_lo = np.exp(cph.confidence_intervals_.loc[var].iloc[0])
            ci_hi = np.exp(cph.confidence_intervals_.loc[var].iloc[1])
            nice_name = "Channel count" if var == "channel_count" else "log(mut+1)"
            print(f"  {nice_name}: HR={hr:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f}), p={p:.2e}")
            results[var] = {"HR": hr, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi}

        results["concordance"] = cph.concordance_index_
        print(f"  Concordance: {cph.concordance_index_:.4f}")

    except Exception as e:
        print(f"  Cox FAILED: {e}")

    return results


def cross_vs_same(df, label):
    """Compare cross-channel (2+ channels) vs same-channel (2+ muts in 1 channel)."""
    cross = df[df["channel_count"] >= 2]
    same = df[(df["driver_mutation_count"] >= 2) & (df["channel_count"] == 1)]

    print(f"\n--- Cross vs Same Channel: {label} ---")
    print(f"  Cross-channel (2+ ch): {len(cross)} pts, "
          f"{100*cross['event'].mean():.1f}% mortality")
    print(f"  Same-channel (2+ mut, 1 ch): {len(same)} pts, "
          f"{100*same['event'].mean():.1f}% mortality")

    if len(cross) > 10 and len(same) > 10:
        lr = logrank_test(cross["os_months"], same["os_months"],
                          cross["event"], same["event"])
        print(f"  Log-rank p: {lr.p_value:.4e}")
        return lr.p_value
    else:
        print(f"  Insufficient sample size for log-rank test")
        return None


def severance_hierarchy(df, label):
    """Channel frequency at 1, 2, 3, 4 channels -- does the same hierarchy emerge?"""
    print(f"\n--- Severance Hierarchy: {label} ---")
    print(f"{'Channel':<30s}  {'1ch':>8s}  {'2ch':>8s}  {'3ch':>8s}  {'4+ch':>8s}")
    print("-" * 70)

    channel_names_sorted = sorted(CHANNEL_NAMES.keys())
    for ch in channel_names_sorted:
        row = []
        for n_ch in [1, 2, 3, 4]:
            if n_ch < 4:
                subset = df[df["channel_count"] == n_ch]
            else:
                subset = df[df["channel_count"] >= n_ch]
            if len(subset) == 0:
                row.append("--")
                continue
            has_ch = subset["channels_severed"].apply(lambda s: ch in s).sum()
            pct = 100 * has_ch / len(subset)
            row.append(f"{pct:5.1f}%")
        print(f"  {CHANNEL_NAMES[ch]:<28s}  {row[0]:>8s}  {row[1]:>8s}  {row[2]:>8s}  {row[3]:>8s}")


def print_tier_summary(df, label):
    """Quick summary for a tier."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    n_with_mut = (df["driver_mutation_count"] > 0).sum()
    print(f"  Patients with survival data: {len(df)}")
    print(f"  Patients with >= 1 mutation: {n_with_mut}")
    print(f"  Deaths: {int(df['event'].sum())} ({100*df['event'].mean():.1f}%)")
    print(f"  Median follow-up: {df['os_months'].median():.1f} months")

    print(f"\n  Channel count distribution:")
    for n in sorted(df["channel_count"].unique()):
        sub = df[df["channel_count"] == n]
        d = sub["event"].sum()
        mort = 100 * d / len(sub) if len(sub) > 0 else 0
        print(f"    {n} ch: {len(sub):6d} pts, {int(d):5d} deaths ({mort:5.1f}%)")

    print(f"\n  Channel frequency (among patients with >= 1 channel mutation):")
    ch_patients = df[df["channel_count"] >= 1]
    all_ch = []
    for s in ch_patients["channels_severed"]:
        all_ch.extend(list(s))
    if all_ch:
        freq = pd.Series(all_ch).value_counts()
        for ch_name, count in freq.items():
            print(f"    {CHANNEL_NAMES.get(ch_name, ch_name):30s}: {count:6d} "
                  f"({100*count/len(ch_patients):.1f}% of pts with mutations)")

# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PATHOGENIC VARIANT SENSITIVITY ANALYSIS")
    print("Addressing VUS inflation concern")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    mut, clin = load_data()
    print(f"  Total mutations: {len(mut)}")
    print(f"  Patients with clinical data: {len(clin)}")

    # --------------------------------------------------------
    # Identify hotspots for Tier 3
    # --------------------------------------------------------
    hotspots = find_hotspots(mut)
    print(f"\n--- Hotspot Missense Positions (>= {HOTSPOT_THRESHOLD} patients) ---")
    print(f"  Total hotspot positions: {len(hotspots)}")
    print(f"\n  Top 20 hotspots by frequency:")
    for _, row in hotspots.head(20).iterrows():
        # Find most common protein change at this position
        pos_muts = mut[
            (mut["gene.hugoGeneSymbol"] == row["gene"]) &
            (mut["proteinPosStart"] == row["pos"]) &
            (mut["mutationType"] == "Missense_Mutation")
        ]
        top_change = pos_muts["proteinChange"].value_counts().index[0] if len(pos_muts) > 0 else "?"
        print(f"    {row['gene']:10s} pos {int(row['pos']):5d} "
              f"({top_change:15s}): {int(row['n_patients']):5d} patients")

    # --------------------------------------------------------
    # Build tiers
    # --------------------------------------------------------
    tier_configs = [
        ("Tier 1 — All non-silent (baseline)", filter_tier1(mut)),
        ("Tier 2 — Truncating only", filter_tier2(mut)),
        ("Tier 3 — Truncating + hotspot missense", filter_tier3(mut, hotspots)),
    ]

    all_results = {}

    for tier_label, tier_mut in tier_configs:
        n_mut = len(tier_mut)
        n_patients = tier_mut["patientId"].nunique()
        print(f"\n\n{'#'*70}")
        print(f"# {tier_label}")
        print(f"# Mutations: {n_mut:,}, Patients: {n_patients:,}")
        print(f"{'#'*70}")

        df = build_patient_df(tier_mut, clin)

        print_tier_summary(df, tier_label)
        cox_results = run_cox_multivariate(df, tier_label)
        cross_p = cross_vs_same(df, tier_label)

        all_results[tier_label] = {
            "n_mutations": n_mut,
            "n_patients_with_mut": n_patients,
            "n_patients_survival": len(df),
            "cox": cox_results,
            "cross_vs_same_p": cross_p,
        }

    # --------------------------------------------------------
    # Severance hierarchy for Tier 2 (truncating only)
    # --------------------------------------------------------
    tier2_mut = filter_tier2(mut)
    tier2_df = build_patient_df(tier2_mut, clin)
    severance_hierarchy(tier2_df, "Tier 2 — Truncating only")

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE: Channel count HR across tiers")
    print(f"{'='*70}")
    print(f"{'Tier':<45s}  {'N pts':>7s}  {'CC HR':>8s}  {'CC p':>10s}  {'Mut HR':>8s}  {'Mut p':>10s}")
    print("-" * 95)
    for tier_label, res in all_results.items():
        cox = res.get("cox", {})
        cc = cox.get("channel_count", {})
        mt = cox.get("log_mut", {})
        cc_hr = f"{cc['HR']:.4f}" if "HR" in cc else "--"
        cc_p = f"{cc['p']:.2e}" if "p" in cc else "--"
        mt_hr = f"{mt['HR']:.4f}" if "HR" in mt else "--"
        mt_p = f"{mt['p']:.2e}" if "p" in mt else "--"
        n = res["n_patients_survival"]
        print(f"  {tier_label:<43s}  {n:>7d}  {cc_hr:>8s}  {cc_p:>10s}  {mt_hr:>8s}  {mt_p:>10s}")

    print(f"\n{'='*70}")
    print("CROSS-CHANNEL vs SAME-CHANNEL p-values")
    print(f"{'='*70}")
    for tier_label, res in all_results.items():
        p = res.get("cross_vs_same_p")
        p_str = f"{p:.4e}" if p is not None else "--"
        print(f"  {tier_label:<43s}  p = {p_str}")

    # --------------------------------------------------------
    # Key finding
    # --------------------------------------------------------
    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")
    t1_cc = all_results.get("Tier 1 — All non-silent (baseline)", {}).get("cox", {}).get("channel_count", {})
    t2_cc = all_results.get("Tier 2 — Truncating only", {}).get("cox", {}).get("channel_count", {})
    t3_cc = all_results.get("Tier 3 — Truncating + hotspot missense", {}).get("cox", {}).get("channel_count", {})

    if t1_cc and t3_cc:
        print(f"  Baseline (all non-silent):        channel HR = {t1_cc.get('HR', 0):.4f}, p = {t1_cc.get('p', 1):.2e}")
    if t2_cc:
        print(f"  Truncating only:                  channel HR = {t2_cc.get('HR', 0):.4f}, p = {t2_cc.get('p', 1):.2e}")
    if t3_cc:
        print(f"  Truncating + hotspot missense:    channel HR = {t3_cc.get('HR', 0):.4f}, p = {t3_cc.get('p', 1):.2e}")
    print()
    print("  If channel_count HR remains significant across tiers, VUS are not")
    print("  inflating the signal. The coupling-channel effect is driven by")
    print("  high-confidence pathogenic variants, not noise from VUS.")

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    buf = io.StringIO()
    # Re-run all prints into buffer
    with redirect_stdout(buf):
        main_inner(mut, clin, hotspots, tier_configs, all_results, tier2_df)
    with open(summary_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved summary to {summary_path}")


def main_inner(mut, clin, hotspots, tier_configs, all_results, tier2_df):
    """Reprint all results for file capture."""
    print("=" * 70)
    print("PATHOGENIC VARIANT SENSITIVITY ANALYSIS")
    print("Addressing VUS inflation concern")
    print("=" * 70)

    print(f"\n--- Hotspot Missense Positions (>= {HOTSPOT_THRESHOLD} patients) ---")
    print(f"  Total hotspot positions: {len(hotspots)}")
    print(f"\n  Top 20 hotspots by frequency:")
    for _, row in hotspots.head(20).iterrows():
        pos_muts = mut[
            (mut["gene.hugoGeneSymbol"] == row["gene"]) &
            (mut["proteinPosStart"] == row["pos"]) &
            (mut["mutationType"] == "Missense_Mutation")
        ]
        top_change = pos_muts["proteinChange"].value_counts().index[0] if len(pos_muts) > 0 else "?"
        print(f"    {row['gene']:10s} pos {int(row['pos']):5d} "
              f"({top_change:15s}): {int(row['n_patients']):5d} patients")

    for tier_label, tier_mut in tier_configs:
        df = build_patient_df(tier_mut, clin)
        print_tier_summary(df, tier_label)
        run_cox_multivariate(df, tier_label)
        cross_vs_same(df, tier_label)

    severance_hierarchy(tier2_df, "Tier 2 — Truncating only")

    # Comparison table
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE: Channel count HR across tiers")
    print(f"{'='*70}")
    print(f"{'Tier':<45s}  {'N pts':>7s}  {'CC HR':>8s}  {'CC p':>10s}  {'Mut HR':>8s}  {'Mut p':>10s}")
    print("-" * 95)
    for tier_label, res in all_results.items():
        cox = res.get("cox", {})
        cc = cox.get("channel_count", {})
        mt = cox.get("log_mut", {})
        cc_hr = f"{cc['HR']:.4f}" if "HR" in cc else "--"
        cc_p = f"{cc['p']:.2e}" if "p" in cc else "--"
        mt_hr = f"{mt['HR']:.4f}" if "HR" in mt else "--"
        mt_p = f"{mt['p']:.2e}" if "p" in mt else "--"
        n = res["n_patients_survival"]
        print(f"  {tier_label:<43s}  {n:>7d}  {cc_hr:>8s}  {cc_p:>10s}  {mt_hr:>8s}  {mt_p:>10s}")

    print(f"\n{'='*70}")
    print("CROSS-CHANNEL vs SAME-CHANNEL p-values")
    print(f"{'='*70}")
    for tier_label, res in all_results.items():
        p = res.get("cross_vs_same_p")
        p_str = f"{p:.4e}" if p is not None else "--"
        print(f"  {tier_label:<43s}  p = {p_str}")

    # Key finding
    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")
    t1_cc = all_results.get("Tier 1 — All non-silent (baseline)", {}).get("cox", {}).get("channel_count", {})
    t2_cc = all_results.get("Tier 2 — Truncating only", {}).get("cox", {}).get("channel_count", {})
    t3_cc = all_results.get("Tier 3 — Truncating + hotspot missense", {}).get("cox", {}).get("channel_count", {})

    if t1_cc and t3_cc:
        print(f"  Baseline (all non-silent):        channel HR = {t1_cc.get('HR', 0):.4f}, p = {t1_cc.get('p', 1):.2e}")
    if t2_cc:
        print(f"  Truncating only:                  channel HR = {t2_cc.get('HR', 0):.4f}, p = {t2_cc.get('p', 1):.2e}")
    if t3_cc:
        print(f"  Truncating + hotspot missense:    channel HR = {t3_cc.get('HR', 0):.4f}, p = {t3_cc.get('p', 1):.2e}")
    print()
    print("  If channel_count HR remains significant across tiers, VUS are not")
    print("  inflating the signal. The coupling-channel effect is driven by")
    print("  high-confidence pathogenic variants, not noise from VUS.")


if __name__ == "__main__":
    main()
