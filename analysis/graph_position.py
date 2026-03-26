"""
graph_position.py — Test whether hub mutations (high connectivity in signaling
networks) predict worse survival than leaf mutations (low connectivity) within
the same coupling channel.

Uses curated hub/leaf classification based on known pathway topology.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
OUT = os.path.join(BASE, "results", "graph_position")
os.makedirs(OUT, exist_ok=True)

CHANNEL_MAP = {
    "BRCA1": "DDR", "BRCA2": "DDR", "PALB2": "DDR",
    "RAD51C": "DDR", "RAD51D": "DDR", "RAD51B": "DDR",
    "ATM": "DDR", "ATR": "DDR", "CHEK2": "DDR", "CHEK1": "DDR",
    "FANCA": "DDR", "FANCC": "DDR", "FANCD2": "DDR",
    "BAP1": "DDR", "BARD1": "DDR",
    "MLH1": "DDR", "MSH2": "DDR", "MSH6": "DDR", "PMS2": "DDR",
    "POLE": "DDR", "POLD1": "DDR",
    "TP53": "CellCycle", "RB1": "CellCycle",
    "CDKN1A": "CellCycle", "CDKN1B": "CellCycle",
    "CDKN2A": "CellCycle", "CDKN2B": "CellCycle",
    "CDK4": "CellCycle", "CDK6": "CellCycle",
    "CCND1": "CellCycle", "CCNE1": "CellCycle",
    "MDM2": "CellCycle", "MDM4": "CellCycle",
    "MYC": "CellCycle", "MYCN": "CellCycle",
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
    "STK11": "PI3K_Growth", "ARID1A": "PI3K_Growth",
    "ESR1": "Endocrine", "ESR2": "Endocrine",
    "PGR": "Endocrine", "AR": "Endocrine",
    "FOXA1": "Endocrine", "GATA3": "Endocrine",
    "B2M": "Immune", "HLA-A": "Immune", "HLA-B": "Immune",
    "HLA-C": "Immune",
    "JAK1": "Immune", "JAK2": "Immune",
    "STAT1": "Immune", "CD274": "Immune",
    "PDCD1LG2": "Immune", "CTLA4": "Immune",
    "CDH1": "TissueArch", "CDH2": "TissueArch",
    "CTNNB1": "TissueArch",
    "APC": "TissueArch",
    "AXIN1": "TissueArch", "AXIN2": "TissueArch",
    "SMAD2": "TissueArch", "SMAD3": "TissueArch", "SMAD4": "TissueArch",
    "TGFBR1": "TissueArch", "TGFBR2": "TissueArch",
    "NOTCH1": "TissueArch", "NOTCH2": "TissueArch",
    "NOTCH3": "TissueArch", "NOTCH4": "TissueArch",
    "FBXW7": "TissueArch",
    "GJA1": "TissueArch", "GJB2": "TissueArch",
}

# Curated hub/leaf mapping based on pathway topology
HUB_GENES = {
    "DDR": {"ATM", "ATR", "BRCA1"},
    "CellCycle": {"TP53", "RB1", "MYC"},
    "PI3K_Growth": {"KRAS", "PTEN", "PIK3CA", "EGFR", "NF1"},
    "Endocrine": {"ESR1", "AR"},
    "Immune": {"B2M", "JAK1", "JAK2"},
    "TissueArch": {"APC", "CDH1", "SMAD4"},
}

LEAF_GENES = {
    "DDR": {"BRCA2", "RAD51C", "RAD51D", "CHEK2", "PALB2", "FANCA", "FANCC"},
    "CellCycle": {"CDK4", "CDK6", "CDKN2A", "CDKN2B", "MDM2", "CCND1"},
    "PI3K_Growth": {"AKT1", "BRAF", "MTOR", "ERBB2", "MAP2K1", "FGFR1", "FGFR2", "FGFR3"},
    "Endocrine": {"FOXA1", "GATA3", "PGR"},
    "Immune": {"HLA-A", "HLA-B", "HLA-C", "STAT1"},
    "TissueArch": {"CTNNB1", "AXIN1", "NOTCH1", "FBXW7", "TGFBR2"},
}

# Flatten for quick lookup: gene -> "hub" or "leaf"
GENE_POSITION = {}
for ch, genes in HUB_GENES.items():
    for g in genes:
        GENE_POSITION[g] = "hub"
for ch, genes in LEAF_GENES.items():
    for g in genes:
        GENE_POSITION[g] = "leaf"

NON_SILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "Splice_Region",
    "Nonstop_Mutation", "Translation_Start_Site",
}

DATASETS = {
    "MSK-IMPACT-50K": {
        "mutations": os.path.join(CACHE, "msk_impact_50k_2026_mutations.csv"),
        "clinical": os.path.join(CACHE, "msk_impact_50k_2026_clinical.csv"),
    },
    "MSK-MetTropism": {
        "mutations": os.path.join(CACHE, "msk_met_2021_mutations.csv"),
        "clinical": os.path.join(CACHE, "msk_met_2021_clinical.csv"),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(name):
    """Load and preprocess one dataset; return (mut_df, clin_df)."""
    paths = DATASETS[name]
    mut = pd.read_csv(paths["mutations"], low_memory=False)
    clin = pd.read_csv(paths["clinical"], low_memory=False)

    # Filter to non-silent, channel-mapped, position-classified mutations
    mut = mut[mut["mutationType"].isin(NON_SILENT)].copy()
    mut["channel"] = mut["gene.hugoGeneSymbol"].map(CHANNEL_MAP)
    mut["position"] = mut["gene.hugoGeneSymbol"].map(GENE_POSITION)
    mut = mut.dropna(subset=["channel", "position"])

    # Parse survival
    clin = clin[clin["OS_STATUS"].notna() & clin["OS_MONTHS"].notna()].copy()
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    clin["time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin = clin.dropna(subset=["time"])
    clin = clin[clin["time"] > 0]

    return mut, clin


def patient_channel_positions(mut):
    """
    For each patient, determine:
    - which channels are severed (have a classified mutation)
    - whether each channel has hub, leaf, or both mutations
    Returns DataFrame: patientId, channel, has_hub, has_leaf
    """
    grouped = (
        mut.groupby(["patientId", "channel"])["position"]
        .agg(lambda x: set(x))
        .reset_index()
    )
    grouped["has_hub"] = grouped["position"].apply(lambda s: "hub" in s)
    grouped["has_leaf"] = grouped["position"].apply(lambda s: "leaf" in s)
    return grouped


def patient_summary(pcp):
    """
    For each patient, compute:
    - channel_count: number of distinct channels severed
    - channels with hub-only, leaf-only, or mixed
    - max_position: "hub" if any hub mutation across all channels, else "leaf"
    """
    per_patient = pcp.groupby("patientId").agg(
        channel_count=("channel", "nunique"),
        any_hub=("has_hub", "any"),
        any_leaf=("has_leaf", "any"),
        channels=("channel", lambda x: set(x)),
    ).reset_index()
    per_patient["max_position"] = per_patient["any_hub"].apply(
        lambda x: "hub" if x else "leaf"
    )
    return per_patient


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

results = []

def log(msg):
    results.append(msg)
    print(msg)


def run_analysis(dataset_name):
    log(f"\n{'='*70}")
    log(f"  DATASET: {dataset_name}")
    log(f"{'='*70}\n")

    mut, clin = load_dataset(dataset_name)
    pcp = patient_channel_positions(mut)
    psumm = patient_summary(pcp)

    log(f"Patients with classified mutations: {len(psumm):,}")
    log(f"  Hub-only (max_position=hub): {(psumm['max_position']=='hub').sum():,}")
    log(f"  Leaf-only (max_position=leaf): {(psumm['max_position']=='leaf').sum():,}")

    # Merge with clinical
    df_all = psumm.merge(clin[["patientId", "time", "event"]], on="patientId", how="inner")
    log(f"Patients with survival data: {len(df_all):,}")

    # ----- Analysis 1: 1-channel patients, hub-only vs leaf-only -----
    log(f"\n--- Analysis 1: Single-channel patients, hub-only vs leaf-only ---")

    one_ch = pcp.copy()
    # Identify 1-channel patients
    ch_counts = one_ch.groupby("patientId")["channel"].nunique().reset_index()
    ch_counts.columns = ["patientId", "n_channels"]
    single_ch_patients = ch_counts[ch_counts["n_channels"] == 1]["patientId"]
    one_ch = one_ch[one_ch["patientId"].isin(single_ch_patients)].copy()

    # Hub-only: has_hub=True, has_leaf=False; Leaf-only: opposite
    hub_only_pts = one_ch[one_ch["has_hub"] & ~one_ch["has_leaf"]]["patientId"]
    leaf_only_pts = one_ch[~one_ch["has_hub"] & one_ch["has_leaf"]]["patientId"]

    df_1ch = one_ch[one_ch["patientId"].isin(hub_only_pts) | one_ch["patientId"].isin(leaf_only_pts)].copy()
    df_1ch["hub_flag"] = df_1ch["patientId"].isin(hub_only_pts).astype(int)
    df_1ch = df_1ch.merge(clin[["patientId", "time", "event"]], on="patientId", how="inner")

    log(f"  Hub-only patients (1 channel): {df_1ch['hub_flag'].sum():,}")
    log(f"  Leaf-only patients (1 channel): {(df_1ch['hub_flag']==0).sum():,}")

    if len(df_1ch) > 50 and df_1ch["hub_flag"].nunique() == 2:
        # Log-rank test
        hub_s = df_1ch[df_1ch["hub_flag"] == 1]
        leaf_s = df_1ch[df_1ch["hub_flag"] == 0]
        lr = logrank_test(hub_s["time"], leaf_s["time"], hub_s["event"], leaf_s["event"])
        log(f"  Log-rank p-value: {lr.p_value:.4e}")
        log(f"  Median OS hub-only: {hub_s['time'].median():.1f} mo")
        log(f"  Median OS leaf-only: {leaf_s['time'].median():.1f} mo")

        # Cox PH with channel one-hot
        cox_df = df_1ch[["time", "event", "hub_flag", "channel"]].copy()
        channel_dummies = pd.get_dummies(cox_df["channel"], prefix="ch", drop_first=True)
        cox_df = pd.concat([cox_df.drop(columns=["channel"]).reset_index(drop=True),
                            channel_dummies.reset_index(drop=True)], axis=1)
        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="time", event_col="event")
            hub_row = cph.summary.loc["hub_flag"]
            log(f"  Cox PH hub_flag HR: {hub_row['exp(coef)']:.3f} "
                f"(95% CI {hub_row['exp(coef) lower 95%']:.3f}-{hub_row['exp(coef) upper 95%']:.3f}), "
                f"p={hub_row['p']:.4e}")
        except Exception as e:
            log(f"  Cox PH failed: {e}")
    else:
        log(f"  Insufficient data for 1-channel analysis")

    # ----- Analysis 2: Within-channel hub vs leaf for large channels -----
    log(f"\n--- Analysis 2: Within-channel hub vs leaf (largest channels) ---")

    for ch_name in ["PI3K_Growth", "CellCycle", "DDR"]:
        log(f"\n  Channel: {ch_name}")
        ch_data = pcp[pcp["channel"] == ch_name].copy()
        hub_pts = ch_data[ch_data["has_hub"] & ~ch_data["has_leaf"]]["patientId"]
        leaf_pts = ch_data[~ch_data["has_hub"] & ch_data["has_leaf"]]["patientId"]

        ch_surv = pd.DataFrame({
            "patientId": pd.concat([hub_pts, leaf_pts]),
            "hub_flag": [1]*len(hub_pts) + [0]*len(leaf_pts),
        })
        ch_surv = ch_surv.merge(clin[["patientId", "time", "event"]], on="patientId", how="inner")

        n_hub = (ch_surv["hub_flag"] == 1).sum()
        n_leaf = (ch_surv["hub_flag"] == 0).sum()
        log(f"    Hub-only: {n_hub:,}  |  Leaf-only: {n_leaf:,}")

        if n_hub >= 20 and n_leaf >= 20:
            h = ch_surv[ch_surv["hub_flag"] == 1]
            l = ch_surv[ch_surv["hub_flag"] == 0]
            lr = logrank_test(h["time"], l["time"], h["event"], l["event"])
            log(f"    Log-rank p: {lr.p_value:.4e}")
            log(f"    Median OS hub: {h['time'].median():.1f} mo  |  leaf: {l['time'].median():.1f} mo")

            try:
                cph = CoxPHFitter()
                cph.fit(ch_surv[["time", "event", "hub_flag"]], duration_col="time", event_col="event")
                r = cph.summary.loc["hub_flag"]
                log(f"    Cox HR: {r['exp(coef)']:.3f} "
                    f"(95% CI {r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f}), "
                    f"p={r['p']:.4e}")
            except Exception as e:
                log(f"    Cox failed: {e}")
        else:
            log(f"    Insufficient data (need >=20 per group)")

    # ----- Analysis 3: All patients, max_position with channel_count -----
    log(f"\n--- Analysis 3: All patients — max_position + channel_count multivariate Cox ---")

    df_mv = df_all[["time", "event", "max_position", "channel_count"]].copy()
    df_mv["hub_any"] = (df_mv["max_position"] == "hub").astype(int)
    df_mv = df_mv.drop(columns=["max_position"])

    log(f"  N = {len(df_mv):,}")
    log(f"  hub_any=1: {df_mv['hub_any'].sum():,}  |  hub_any=0: {(df_mv['hub_any']==0).sum():,}")
    log(f"  Mean channel_count: {df_mv['channel_count'].mean():.2f}")

    if len(df_mv) > 100:
        try:
            cph = CoxPHFitter()
            cph.fit(df_mv, duration_col="time", event_col="event")
            log(f"\n  Multivariate Cox results:")
            for var in ["hub_any", "channel_count"]:
                r = cph.summary.loc[var]
                log(f"    {var}: HR={r['exp(coef)']:.3f} "
                    f"(95% CI {r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f}), "
                    f"p={r['p']:.4e}")
        except Exception as e:
            log(f"  Multivariate Cox failed: {e}")

    return df_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log("GRAPH POSITION ANALYSIS: Hub vs Leaf Mutations and Survival")
    log("=" * 70)

    all_dfs = []
    for ds_name in DATASETS:
        df = run_analysis(ds_name)
        df["dataset"] = ds_name
        all_dfs.append(df)

    # ----- KM Plot: pooled across datasets -----
    log(f"\n{'='*70}")
    log("  KAPLAN-MEIER PLOT (pooled)")
    log(f"{'='*70}\n")

    pooled = pd.concat(all_dfs, ignore_index=True)
    hub_grp = pooled[pooled["max_position"] == "hub"]
    leaf_grp = pooled[pooled["max_position"] == "leaf"]

    log(f"Pooled hub: {len(hub_grp):,}  |  leaf: {len(leaf_grp):,}")

    lr = logrank_test(hub_grp["time"], leaf_grp["time"], hub_grp["event"], leaf_grp["event"])
    log(f"Pooled log-rank p: {lr.p_value:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()

    kmf.fit(hub_grp["time"], hub_grp["event"], label=f"Hub (n={len(hub_grp):,})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#c0392b")

    kmf.fit(leaf_grp["time"], leaf_grp["event"], label=f"Leaf (n={len(leaf_grp):,})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2980b9")

    ax.set_xlabel("Overall Survival (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Hub vs Leaf Mutation Survival\n(pooled MSK-IMPACT-50K + MetTropism)", fontsize=13)
    ax.legend(fontsize=11, loc="lower left")
    ax.annotate(f"Log-rank p = {lr.p_value:.2e}", xy=(0.98, 0.98),
                xycoords="axes fraction", ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))
    ax.set_xlim(0, min(pooled["time"].quantile(0.95), 120))
    plt.tight_layout()
    km_path = os.path.join(OUT, "km_hub_vs_leaf.png")
    fig.savefig(km_path, dpi=150)
    plt.close(fig)
    log(f"\nKM plot saved to: {km_path}")

    # Write summary
    summary_path = os.path.join(OUT, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(results))
    log(f"Summary saved to: {summary_path}")
    print("\nDone.")
