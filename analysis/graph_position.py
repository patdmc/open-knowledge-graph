"""
graph_position.py — Test whether graph position (STRING PPI degree) predicts
mutation severity within coupling channels.

Hub/leaf classification derived from STRING PPI within-channel degree.
Top-3 genes per channel by within-channel degree are hubs; rest are leaves.
Continuous degree also available as GENE_DEGREE for regression models.
"""

import os
import sys
import warnings
import json
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

def _load_channel_map():
    """Load canonical 8-channel gene map from CSV."""
    import csv as _csv
    csv_path = os.path.join(os.path.dirname(BASE), "data", "channel_gene_map.csv")
    result = {}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            ch = row["channel"]
            # Normalize TissueArchitecture → TissueArch for backwards compat
            if ch == "TissueArchitecture":
                ch = "TissueArch"
            result[row["gene"]] = ch
    return result


CHANNEL_MAP = _load_channel_map()

# ---------------------------------------------------------------------------
# STRING PPI degree-based hub/leaf classification
# ---------------------------------------------------------------------------
GNN_CACHE = os.path.join(os.path.dirname(BASE), "gnn", "data", "cache")
STRING_CACHE = os.path.join(GNN_CACHE, "string_ppi_edges.json")

def _compute_string_hub_leaf():
    """Derive hub/leaf from STRING PPI within-channel degree (top 3 per channel)."""
    if not os.path.exists(STRING_CACHE):
        # Fallback to legacy curated if STRING cache missing
        print("WARNING: STRING cache not found, using legacy curated hub/leaf")
        return _legacy_hub_leaf()

    with open(STRING_CACHE) as f:
        edges_data = json.load(f)

    # Compute total and within-channel degree
    total_degree = {}
    within_ch_degree = {}  # (gene, channel) -> degree

    for key, edge_list in edges_data.items():
        channel = key if key != "cross_channel" else None
        for edge in edge_list:
            a, b = edge[0], edge[1]
            if a in CHANNEL_MAP:
                total_degree[a] = total_degree.get(a, 0) + 1
            if b in CHANNEL_MAP:
                total_degree[b] = total_degree.get(b, 0) + 1
            if channel and channel != "cross_channel":
                within_ch_degree[(a, channel)] = within_ch_degree.get((a, channel), 0) + 1
                within_ch_degree[(b, channel)] = within_ch_degree.get((b, channel), 0) + 1

    # Top 3 per channel by within-channel degree
    hub_genes = {}
    for ch in set(CHANNEL_MAP.values()):
        ch_genes = [(g, d) for (g, c), d in within_ch_degree.items() if c == ch]
        ch_genes.sort(key=lambda x: -x[1])
        hub_genes[ch] = set(g for g, d in ch_genes[:3])

    # Everything else is leaf
    leaf_genes = {}
    hub_flat = set()
    for genes in hub_genes.values():
        hub_flat |= genes
    for ch in set(CHANNEL_MAP.values()):
        ch_all = {g for g, c in CHANNEL_MAP.items() if c == ch}
        leaf_genes[ch] = ch_all - hub_genes.get(ch, set())

    return hub_genes, leaf_genes, total_degree

def _legacy_hub_leaf():
    """Legacy curated classification (kept for comparison only)."""
    hub = {
        "DDR": {"ATM", "ATR", "BRCA1"},
        "CellCycle": {"TP53", "RB1", "MYC"},
        "PI3K_Growth": {"KRAS", "PTEN", "PIK3CA", "EGFR", "NF1"},
        "Endocrine": {"ESR1", "AR"},
        "Immune": {"B2M", "JAK1", "JAK2"},
        "TissueArch": {"APC", "CDH1", "SMAD4"},
    }
    leaf = {
        "DDR": {"BRCA2", "RAD51C", "RAD51D", "CHEK2", "PALB2", "FANCA", "FANCC"},
        "CellCycle": {"CDK4", "CDK6", "CDKN2A", "CDKN2B", "MDM2", "CCND1"},
        "PI3K_Growth": {"AKT1", "BRAF", "MTOR", "ERBB2", "MAP2K1", "FGFR1", "FGFR2", "FGFR3"},
        "Endocrine": {"FOXA1", "GATA3", "PGR"},
        "Immune": {"HLA-A", "HLA-B", "HLA-C", "STAT1"},
        "TissueArch": {"CTNNB1", "AXIN1", "NOTCH1", "FBXW7", "TGFBR2"},
    }
    degree = {g: 0 for g in CHANNEL_MAP}
    return hub, leaf, degree

HUB_GENES, LEAF_GENES, GENE_DEGREE = _compute_string_hub_leaf()

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

    # ----- Analysis 4: TP53/KRAS Sensitivity -----
    log(f"\n--- Analysis 4: Degree effect EXCLUDING TP53 and KRAS ---")

    # Get patients whose ONLY hub mutations are TP53 or KRAS
    # We want to test: does degree predict survival without these two genes?
    tp53_kras = {"TP53", "KRAS"}

    # For continuous degree: assign max mutated gene degree per patient
    mut_with_degree = mut.copy()
    mut_with_degree["degree"] = mut_with_degree["gene.hugoGeneSymbol"].map(GENE_DEGREE).fillna(0)

    # Patient max degree (all genes)
    pat_degree_all = mut_with_degree.groupby("patientId")["degree"].max().reset_index()
    pat_degree_all.columns = ["patientId", "max_degree"]

    # Patient max degree EXCLUDING TP53/KRAS
    mut_no_tk = mut_with_degree[~mut_with_degree["gene.hugoGeneSymbol"].isin(tp53_kras)]
    pat_degree_notk = mut_no_tk.groupby("patientId")["degree"].max().reset_index()
    pat_degree_notk.columns = ["patientId", "max_degree_no_tk"]

    # Also: mean degree per patient (weighted by all mutated genes)
    pat_mean_degree = mut_with_degree.groupby("patientId")["degree"].mean().reset_index()
    pat_mean_degree.columns = ["patientId", "mean_degree"]

    pat_mean_notk = mut_no_tk.groupby("patientId")["degree"].mean().reset_index()
    pat_mean_notk.columns = ["patientId", "mean_degree_no_tk"]

    # Merge
    deg_df = clin[["patientId", "time", "event"]].copy()
    deg_df = deg_df.merge(pat_degree_all, on="patientId", how="inner")
    deg_df = deg_df.merge(pat_degree_notk, on="patientId", how="left")
    deg_df = deg_df.merge(pat_mean_degree, on="patientId", how="left")
    deg_df = deg_df.merge(pat_mean_notk, on="patientId", how="left")
    deg_df = deg_df.merge(psumm[["patientId", "channel_count"]], on="patientId", how="left")
    deg_df["channel_count"] = deg_df["channel_count"].fillna(0)

    log(f"  Patients with degree data: {len(deg_df):,}")

    # 4a: Continuous degree (all genes) — Cox
    log(f"\n  4a. Continuous max degree (all genes):")
    try:
        from lifelines.utils import concordance_index
        c_all = concordance_index(deg_df["time"], -deg_df["max_degree"], deg_df["event"])
        log(f"    C-index (max degree): {c_all:.4f}")

        cph = CoxPHFitter()
        cox_in = deg_df[["time", "event", "max_degree", "channel_count"]].dropna()
        cph.fit(cox_in, duration_col="time", event_col="event")
        for var in ["max_degree", "channel_count"]:
            r = cph.summary.loc[var]
            log(f"    {var}: HR={r['exp(coef)']:.4f} "
                f"(95% CI {r['exp(coef) lower 95%']:.4f}-{r['exp(coef) upper 95%']:.4f}), "
                f"p={r['p']:.4e}")
    except Exception as e:
        log(f"    Cox with degree failed: {e}")

    # 4b: Continuous degree EXCLUDING TP53/KRAS
    log(f"\n  4b. Continuous max degree (EXCLUDING TP53/KRAS):")
    deg_notk = deg_df.dropna(subset=["max_degree_no_tk"])
    log(f"    Patients with non-TP53/KRAS mutations: {len(deg_notk):,}")

    if len(deg_notk) > 100:
        try:
            c_notk = concordance_index(deg_notk["time"], -deg_notk["max_degree_no_tk"], deg_notk["event"])
            log(f"    C-index (max degree excl TP53/KRAS): {c_notk:.4f}")

            cph = CoxPHFitter()
            cox_in = deg_notk[["time", "event", "max_degree_no_tk", "channel_count"]].dropna()
            cph.fit(cox_in, duration_col="time", event_col="event")
            for var in ["max_degree_no_tk", "channel_count"]:
                r = cph.summary.loc[var]
                log(f"    {var}: HR={r['exp(coef)']:.4f} "
                    f"(95% CI {r['exp(coef) lower 95%']:.4f}-{r['exp(coef) upper 95%']:.4f}), "
                    f"p={r['p']:.4e}")
        except Exception as e:
            log(f"    Cox excl TP53/KRAS failed: {e}")

    # 4c: Hub/leaf EXCLUDING TP53/KRAS
    log(f"\n  4c. Hub/leaf binary (EXCLUDING TP53/KRAS patients):")
    # Find patients whose ONLY hub mutations are in TP53/KRAS
    hub_genes_per_patient = mut[mut["position"] == "hub"].groupby("patientId")["gene.hugoGeneSymbol"].apply(set).reset_index()
    hub_genes_per_patient.columns = ["patientId", "hub_genes"]

    # Patients with hub mutations in non-TP53/KRAS genes
    hub_genes_per_patient["has_other_hub"] = hub_genes_per_patient["hub_genes"].apply(
        lambda s: len(s - tp53_kras) > 0
    )

    # Keep: patients with no hub mutations (leaf-only), OR patients with non-TP53/KRAS hubs
    leaf_patients = set(psumm[psumm["max_position"] == "leaf"]["patientId"])
    other_hub_patients = set(hub_genes_per_patient[hub_genes_per_patient["has_other_hub"]]["patientId"])

    df_notk = df_all[df_all["patientId"].isin(leaf_patients | other_hub_patients)].copy()
    df_notk["hub_any"] = df_notk["patientId"].isin(other_hub_patients).astype(int)

    n_hub_notk = df_notk["hub_any"].sum()
    n_leaf_notk = (df_notk["hub_any"] == 0).sum()
    log(f"    Hub (non-TP53/KRAS): {n_hub_notk:,}  |  Leaf: {n_leaf_notk:,}")

    if n_hub_notk >= 20 and n_leaf_notk >= 20:
        try:
            cph = CoxPHFitter()
            cox_in = df_notk[["time", "event", "hub_any", "channel_count"]].dropna()
            cph.fit(cox_in, duration_col="time", event_col="event")
            for var in ["hub_any", "channel_count"]:
                r = cph.summary.loc[var]
                log(f"    {var}: HR={r['exp(coef)']:.3f} "
                    f"(95% CI {r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f}), "
                    f"p={r['p']:.4e}")
        except Exception as e:
            log(f"    Cox excl TP53/KRAS failed: {e}")

    # 4d: Per-channel degree analysis
    log(f"\n  4d. Per-channel continuous degree (within-channel degree):")
    for ch_name in ["PI3K_Growth", "CellCycle", "DDR"]:
        ch_mut = mut_with_degree[mut_with_degree["channel"] == ch_name].copy()
        # Within-channel degree
        ch_mut["within_degree"] = ch_mut["gene.hugoGeneSymbol"].apply(
            lambda g: within_ch_degree.get((g, ch_name), 0)
            if (g, ch_name) in within_ch_degree else 0
        )
        pat_ch_deg = ch_mut.groupby("patientId")["within_degree"].max().reset_index()
        pat_ch_deg.columns = ["patientId", "wc_degree"]
        ch_df = pat_ch_deg.merge(clin[["patientId", "time", "event"]], on="patientId", how="inner")

        if len(ch_df) > 100:
            try:
                c = concordance_index(ch_df["time"], -ch_df["wc_degree"], ch_df["event"])
                cph = CoxPHFitter()
                cph.fit(ch_df[["time", "event", "wc_degree"]], duration_col="time", event_col="event")
                r = cph.summary.loc["wc_degree"]
                log(f"    {ch_name}: C={c:.4f}, HR={r['exp(coef)']:.4f}, p={r['p']:.4e}")
            except Exception as e:
                log(f"    {ch_name}: failed — {e}")

    # Need within_ch_degree accessible here
    return df_all


# Expose within_ch_degree at module level for Analysis 4d
_string_within_ch_degree = {}
if os.path.exists(STRING_CACHE):
    with open(STRING_CACHE) as _f:
        _edges_data = json.load(_f)
    for _key, _edge_list in _edges_data.items():
        if _key != "cross_channel":
            for _edge in _edge_list:
                _a, _b = _edge[0], _edge[1]
                _string_within_ch_degree[(_a, _key)] = _string_within_ch_degree.get((_a, _key), 0) + 1
                _string_within_ch_degree[(_b, _key)] = _string_within_ch_degree.get((_b, _key), 0) + 1
within_ch_degree = _string_within_ch_degree


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
