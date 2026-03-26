#!/usr/bin/env python3
"""
BRCA1 vs BRCA2 phenotype comparison across MSK datasets.

Tests coupling-channel predictions:
  BRCA1 = hub/upstream DDR = complete DDR severance = triple-negative, visceral-tropic
  BRCA2 = downstream DDR = partial DDR severance = HR+, bone-tropic
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
CACHE = "/Users/patdmccarthy/open-knowledge-graph/analysis/cache"
OUT   = "/Users/patdmccarthy/open-knowledge-graph/analysis/results/brca_comparison"
os.makedirs(OUT, exist_ok=True)

# ── channel map (from channel_analysis.py) ─────────────────────────────
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

PATHOGENIC_TYPES = [
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "Splice_Region",
    "Nonstop_Mutation", "Translation_Start_Site",
]

TARGET_GENES = ["BRCA1", "BRCA2", "RAD51C"]

# ── helpers ────────────────────────────────────────────────────────────
def load_mutations(path):
    df = pd.read_csv(path)
    return df

def get_mutant_patients(mut_df, gene, pathogenic_types=PATHOGENIC_TYPES):
    """Return set of patientIds with a pathogenic mutation in gene."""
    mask = (mut_df["gene.hugoGeneSymbol"] == gene) & (mut_df["mutationType"].isin(pathogenic_types))
    return set(mut_df.loc[mask, "patientId"].unique())

def parse_survival(clin_df):
    """Add event/time columns from OS_STATUS/OS_MONTHS."""
    df = clin_df.copy()
    if "OS_STATUS" in df.columns:
        df["event"] = df["OS_STATUS"].apply(lambda x: 1 if str(x).startswith("1") else 0)
    if "OS_MONTHS" in df.columns:
        df["time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    return df

def assign_brca_group(row, brca1_set, brca2_set, rad51c_set):
    pid = row["patientId"]
    groups = []
    if pid in brca1_set:
        groups.append("BRCA1")
    if pid in brca2_set:
        groups.append("BRCA2")
    if pid in rad51c_set:
        groups.append("RAD51C")
    if len(groups) == 1:
        return groups[0]
    elif len(groups) > 1:
        return "Multi"
    return None

# ── load all datasets ──────────────────────────────────────────────────
print("Loading datasets...")

# 50K
mut_50k = load_mutations(f"{CACHE}/msk_impact_50k_2026_mutations.csv")
clin_50k = pd.read_csv(f"{CACHE}/msk_impact_50k_2026_clinical.csv")
full_clin_50k = pd.read_csv(f"{CACHE}/msk_impact_50k_2026_full_clinical.csv")

# Met 2021
mut_met = load_mutations(f"{CACHE}/msk_met_2021_mutations.csv")
clin_met = pd.read_csv(f"{CACHE}/msk_met_2021_full_clinical.csv")

# 2017
mut_2017 = load_mutations(f"{CACHE}/msk_impact_2017_mutations.csv")
clin_2017 = pd.read_csv(f"{CACHE}/msk_impact_2017_clinical.csv")

# ── identify mutant patients per dataset ───────────────────────────────
print("Identifying mutant patients...")

datasets = {
    "MSK-IMPACT-50K": {"mut": mut_50k, "clin": full_clin_50k, "has_subtype": False, "has_met": False},
    "MSK-MET-2021":   {"mut": mut_met, "clin": clin_met,      "has_subtype": True,  "has_met": True},
    "MSK-IMPACT-2017":{"mut": mut_2017,"clin": clin_2017,      "has_subtype": False, "has_met": False},
}

# Build patient gene-group assignments per dataset
for name, d in datasets.items():
    brca1_pts = get_mutant_patients(d["mut"], "BRCA1")
    brca2_pts = get_mutant_patients(d["mut"], "BRCA2")
    rad51c_pts = get_mutant_patients(d["mut"], "RAD51C")
    d["brca1"] = brca1_pts
    d["brca2"] = brca2_pts
    d["rad51c"] = rad51c_pts
    clin = d["clin"].copy()
    clin["brca_group"] = clin.apply(lambda r: assign_brca_group(r, brca1_pts, brca2_pts, rad51c_pts), axis=1)
    clin = parse_survival(clin)
    d["clin"] = clin
    print(f"  {name}: BRCA1={len(brca1_pts)}, BRCA2={len(brca2_pts)}, RAD51C={len(rad51c_pts)}")

# ── merge all datasets for pooled analyses ─────────────────────────────
all_muts = []
all_clin = []
for name, d in datasets.items():
    c = d["clin"].copy()
    c["dataset"] = name
    # Normalize column names
    if "CANCER_TYPE" not in c.columns:
        c["CANCER_TYPE"] = np.nan
    if "CANCER_TYPE_DETAILED" not in c.columns:
        c["CANCER_TYPE_DETAILED"] = np.nan
    if "SUBTYPE" not in c.columns:
        c["SUBTYPE"] = np.nan
    all_clin.append(c)
    m = d["mut"].copy()
    m["dataset"] = name
    all_muts.append(m)

pooled_clin = pd.concat(all_clin, ignore_index=True)
pooled_mut  = pd.concat(all_muts, ignore_index=True)

# Deduplicate patients across datasets (keep the one with most info)
# Priority: MET > 50K > 2017
priority = {"MSK-MET-2021": 0, "MSK-IMPACT-50K": 1, "MSK-IMPACT-2017": 2}
pooled_clin["_prio"] = pooled_clin["dataset"].map(priority)
pooled_clin = pooled_clin.sort_values("_prio").drop_duplicates(subset="patientId", keep="first").drop(columns="_prio")

# Only patients in one of our groups
grouped = pooled_clin[pooled_clin["brca_group"].isin(["BRCA1", "BRCA2", "RAD51C"])].copy()

report = []
def rprint(s=""):
    print(s)
    report.append(str(s))

# ═══════════════════════════════════════════════════════════════════════
# 1. CANCER TYPE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
rprint("=" * 80)
rprint("1. CANCER TYPE DISTRIBUTION (pooled, deduplicated)")
rprint("=" * 80)

for gene in TARGET_GENES:
    sub = grouped[grouped["brca_group"] == gene]
    n = len(sub)
    if n == 0:
        continue
    rprint(f"\n{gene}-mutant patients: N={n}")
    ct = sub["CANCER_TYPE"].value_counts()
    top = ct.head(10)
    for cancer, count in top.items():
        rprint(f"  {cancer}: {count} ({100*count/n:.1f}%)")

# Breast subtype breakdown
rprint("\n--- Breast cancer subtype breakdown ---")
breast = grouped[grouped["CANCER_TYPE"] == "Breast Cancer"].copy()

# Use CANCER_TYPE_DETAILED and SUBTYPE to classify
def classify_breast_subtype(row):
    sub = str(row.get("SUBTYPE", "")).lower() if pd.notna(row.get("SUBTYPE")) else ""
    det = str(row.get("CANCER_TYPE_DETAILED", "")).lower() if pd.notna(row.get("CANCER_TYPE_DETAILED")) else ""
    combined = sub + " " + det
    if "triple" in combined or "tnbc" in combined or "basal" in combined:
        return "Triple-Negative"
    if "her2" in combined and ("hr+" in combined or "er+" in combined or "hr-" not in combined):
        # Might be HER2+ with HR+
        if "hr-" in combined or "er-" in combined:
            return "HER2+"
        return "HER2+"
    if "hr+" in combined or "er+" in combined:
        if "her2-" in combined or "her2" not in combined:
            return "HR+/HER2-"
        return "HER2+"
    if "her2" in combined and "her2-" not in combined:
        return "HER2+"
    if "lobular" in det or "ductal" in det:
        # No specific subtype info
        return "Unspecified"
    return "Unspecified"

breast["breast_subtype"] = breast.apply(classify_breast_subtype, axis=1)

for gene in TARGET_GENES:
    sub = breast[breast["brca_group"] == gene]
    n = len(sub)
    if n == 0:
        continue
    rprint(f"\n{gene}-mutant breast cancer: N={n}")
    st = sub["breast_subtype"].value_counts()
    for stype, count in st.items():
        rprint(f"  {stype}: {count} ({100*count/n:.1f}%)")

# Also show raw SUBTYPE values for the met dataset patients
rprint("\n--- Raw SUBTYPE values (MSK-MET-2021 breast patients) ---")
met_breast = datasets["MSK-MET-2021"]["clin"]
met_breast = met_breast[(met_breast["CANCER_TYPE"] == "Breast Cancer") &
                        (met_breast["brca_group"].isin(TARGET_GENES))]
for gene in TARGET_GENES:
    sub = met_breast[met_breast["brca_group"] == gene]
    n = len(sub)
    if n == 0:
        continue
    rprint(f"\n{gene} (MET data): N={n}")
    if "SUBTYPE" in sub.columns:
        st = sub["SUBTYPE"].value_counts()
        for stype, count in st.items():
            rprint(f"  {stype}: {count} ({100*count/n:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 2. SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
rprint("\n" + "=" * 80)
rprint("2. SURVIVAL ANALYSIS")
rprint("=" * 80)

surv = grouped.dropna(subset=["time", "event"]).copy()
surv = surv[surv["time"] > 0]

def plot_km(df, group_col, groups, title, filename, restrict_cancer=None):
    """Plot KM curves and return logrank p-value."""
    if restrict_cancer:
        df = df[df["CANCER_TYPE"] == restrict_cancer]

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    results = {}
    for g in groups:
        sub = df[df[group_col] == g]
        if len(sub) < 5:
            continue
        kmf.fit(sub["time"], sub["event"], label=f"{g} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax)
        results[g] = sub

    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close(fig)

    # Logrank between first two groups if both exist
    g_keys = [g for g in groups if g in results and len(results[g]) >= 5]
    if len(g_keys) >= 2:
        a = results[g_keys[0]]
        b = results[g_keys[1]]
        lr = logrank_test(a["time"], b["time"], a["event"], b["event"])
        return lr.p_value, {g: len(results[g]) for g in g_keys}
    return None, {}

# Overall KM: BRCA1 vs BRCA2
p_overall, ns = plot_km(surv, "brca_group", ["BRCA1", "BRCA2", "RAD51C"],
                        "Overall Survival: BRCA1 vs BRCA2 vs RAD51C",
                        "km_overall.png")
rprint(f"\nOverall KM (BRCA1 vs BRCA2): logrank p={p_overall:.4f}" if p_overall else "\nInsufficient data for overall KM")
if ns:
    rprint(f"  Sample sizes: {ns}")

# Breast-only KM
p_breast, ns_b = plot_km(surv, "brca_group", ["BRCA1", "BRCA2", "RAD51C"],
                         "Overall Survival (Breast Cancer): BRCA1 vs BRCA2 vs RAD51C",
                         "km_breast.png", restrict_cancer="Breast Cancer")
rprint(f"\nBreast-only KM (BRCA1 vs BRCA2): logrank p={p_breast:.4f}" if p_breast else "\nInsufficient data for breast-only KM")
if ns_b:
    rprint(f"  Sample sizes: {ns_b}")

# Cox PH: BRCA1 vs BRCA2 controlling for cancer type
rprint("\n--- Cox PH: BRCA1 vs BRCA2 controlling for cancer type ---")
cox_df = surv[surv["brca_group"].isin(["BRCA1", "BRCA2"])].copy()
cox_df["is_BRCA1"] = (cox_df["brca_group"] == "BRCA1").astype(int)

# Top cancer types for dummies
top_cancers = cox_df["CANCER_TYPE"].value_counts().head(5).index.tolist()
for c in top_cancers:
    safe = c.replace(" ", "_").replace("/", "_")[:20]
    cox_df[f"ct_{safe}"] = (cox_df["CANCER_TYPE"] == c).astype(int)

cox_cols = ["time", "event", "is_BRCA1"] + [c for c in cox_df.columns if c.startswith("ct_")]
cox_clean = cox_df[cox_cols].dropna()

if len(cox_clean) >= 20:
    cph = CoxPHFitter()
    try:
        cph.fit(cox_clean, duration_col="time", event_col="event")
        rprint(cph.summary[["coef", "exp(coef)", "p"]].to_string())
    except Exception as e:
        rprint(f"Cox PH failed: {e}")
else:
    rprint("Insufficient data for Cox PH")

# ═══════════════════════════════════════════════════════════════════════
# 3. METASTATIC TROPISM (MSK-MET-2021 only)
# ═══════════════════════════════════════════════════════════════════════
rprint("\n" + "=" * 80)
rprint("3. METASTATIC TROPISM (MSK-MET-2021)")
rprint("=" * 80)

met_clin = datasets["MSK-MET-2021"]["clin"].copy()
met_grouped = met_clin[met_clin["brca_group"].isin(TARGET_GENES)]

MET_SITES = [
    "DMETS_DX_BONE", "DMETS_DX_LIVER", "DMETS_DX_LUNG",
    "DMETS_DX_CNS_BRAIN", "DMETS_DX_DIST_LN", "DMETS_DX_PLEURA",
    "DMETS_DX_ADRENAL_GLAND", "DMETS_DX_SKIN",
]

def met_site_analysis(df, label, genes=TARGET_GENES):
    rprint(f"\n--- {label} ---")
    results = {}
    for gene in genes:
        sub = df[df["brca_group"] == gene]
        n = len(sub)
        if n == 0:
            continue
        results[gene] = {}
        rprint(f"\n{gene}: N={n}")
        for site in MET_SITES:
            if site not in sub.columns:
                continue
            pos = (sub[site] == "Yes").sum()
            pct = 100 * pos / n if n > 0 else 0
            results[gene][site] = (pos, n, pct)
            rprint(f"  {site}: {pos}/{n} ({pct:.1f}%)")

    # Fisher exact tests BRCA1 vs BRCA2
    if "BRCA1" in results and "BRCA2" in results:
        rprint(f"\nFisher exact tests (BRCA1 vs BRCA2):")
        for site in MET_SITES:
            if site not in results["BRCA1"] or site not in results["BRCA2"]:
                continue
            a_pos, a_n, _ = results["BRCA1"][site]
            b_pos, b_n, _ = results["BRCA2"][site]
            table = [[a_pos, a_n - a_pos], [b_pos, b_n - b_pos]]
            _, p = stats.fisher_exact(table)
            sig = " *" if p < 0.05 else ""
            rprint(f"  {site}: p={p:.4f}{sig}")
    return results

# All cancer types
met_site_analysis(met_grouped, "All cancer types (MSK-MET-2021)")

# Breast only
met_breast_only = met_grouped[met_grouped["CANCER_TYPE"] == "Breast Cancer"]
met_site_analysis(met_breast_only, "Breast cancer only (MSK-MET-2021)")

# ═══════════════════════════════════════════════════════════════════════
# 4. CO-MUTATION / CHANNEL PATTERNS
# ═══════════════════════════════════════════════════════════════════════
rprint("\n" + "=" * 80)
rprint("4. CO-MUTATION PATTERNS (channel-level)")
rprint("=" * 80)

# Use pooled mutations, filter to pathogenic
path_mut = pooled_mut[pooled_mut["mutationType"].isin(PATHOGENIC_TYPES)].copy()

# Build patient -> set of mutated channels
# Deduplicate patients: keep unique (patientId, gene) pairs
path_mut_dedup = path_mut.drop_duplicates(subset=["patientId", "gene.hugoGeneSymbol"])
path_mut_dedup["channel"] = path_mut_dedup["gene.hugoGeneSymbol"].map(CHANNEL_MAP)

# For each target gene group, what channels are co-mutated?
for gene in TARGET_GENES:
    gene_patients = set()
    for d in datasets.values():
        gene_patients |= d.get(gene.lower().replace("brca", "brca"), set())

    # Re-collect properly
    gene_patients = set()
    for name, d in datasets.items():
        if gene == "BRCA1":
            gene_patients |= d["brca1"]
        elif gene == "BRCA2":
            gene_patients |= d["brca2"]
        elif gene == "RAD51C":
            gene_patients |= d["rad51c"]

    n = len(gene_patients)
    if n == 0:
        continue

    # Get all mutations for these patients (exclude the target gene itself)
    pt_muts = path_mut_dedup[(path_mut_dedup["patientId"].isin(gene_patients)) &
                             (path_mut_dedup["gene.hugoGeneSymbol"] != gene)]

    rprint(f"\n{gene}-mutant patients (N={n}): co-mutated channels")

    # Channel frequencies
    ch_counts = pt_muts.groupby("channel")["patientId"].nunique()
    ch_counts = ch_counts.sort_values(ascending=False)
    for ch, count in ch_counts.items():
        if pd.isna(ch):
            continue
        rprint(f"  {ch}: {count}/{n} ({100*count/n:.1f}%)")

    # Top co-mutated genes
    rprint(f"\n{gene}-mutant: top 15 co-mutated genes")
    gene_counts = pt_muts.groupby("gene.hugoGeneSymbol")["patientId"].nunique()
    gene_counts = gene_counts.sort_values(ascending=False).head(15)
    for g, count in gene_counts.items():
        ch = CHANNEL_MAP.get(g, "?")
        rprint(f"  {g} ({ch}): {count}/{n} ({100*count/n:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 5. BRCA1 vs BRCA2 STATISTICAL COMPARISON (key predictions)
# ═══════════════════════════════════════════════════════════════════════
rprint("\n" + "=" * 80)
rprint("5. KEY PREDICTION TESTS")
rprint("=" * 80)

# Prediction 1: BRCA1 -> more triple-negative than BRCA2
rprint("\nPrediction: BRCA1 enriched for triple-negative breast cancer")
brca1_breast = breast[breast["brca_group"] == "BRCA1"]
brca2_breast = breast[breast["brca_group"] == "BRCA2"]
rad51c_breast = breast[breast["brca_group"] == "RAD51C"]

def tn_fraction(df):
    n = len(df)
    tn = (df["breast_subtype"] == "Triple-Negative").sum()
    return tn, n

for gene, sub in [("BRCA1", brca1_breast), ("BRCA2", brca2_breast), ("RAD51C", rad51c_breast)]:
    tn, n = tn_fraction(sub)
    if n > 0:
        rprint(f"  {gene}: {tn}/{n} triple-negative ({100*tn/n:.1f}%)")

if len(brca1_breast) > 0 and len(brca2_breast) > 0:
    tn1, n1 = tn_fraction(brca1_breast)
    tn2, n2 = tn_fraction(brca2_breast)
    table = [[tn1, n1 - tn1], [tn2, n2 - tn2]]
    _, p = stats.fisher_exact(table)
    rprint(f"  Fisher exact (BRCA1 vs BRCA2 triple-neg): p={p:.6f}")

# Prediction 2: BRCA2 -> more HR+ than BRCA1
rprint("\nPrediction: BRCA2 enriched for HR+/HER2- breast cancer")
def hrpos_fraction(df):
    n = len(df)
    hr = (df["breast_subtype"] == "HR+/HER2-").sum()
    return hr, n

for gene, sub in [("BRCA1", brca1_breast), ("BRCA2", brca2_breast), ("RAD51C", rad51c_breast)]:
    hr, n = hrpos_fraction(sub)
    if n > 0:
        rprint(f"  {gene}: {hr}/{n} HR+/HER2- ({100*hr/n:.1f}%)")

if len(brca1_breast) > 0 and len(brca2_breast) > 0:
    hr1, n1 = hrpos_fraction(brca1_breast)
    hr2, n2 = hrpos_fraction(brca2_breast)
    table = [[hr1, n1 - hr1], [hr2, n2 - hr2]]
    _, p = stats.fisher_exact(table)
    rprint(f"  Fisher exact (BRCA1 vs BRCA2 HR+): p={p:.6f}")

# Prediction 3: BRCA1 -> more TP53 co-mutation (complete DDR severance)
rprint("\nPrediction: BRCA1 more TP53 co-mutation than BRCA2")
for gene in TARGET_GENES:
    gene_patients = set()
    for name, d in datasets.items():
        if gene == "BRCA1":
            gene_patients |= d["brca1"]
        elif gene == "BRCA2":
            gene_patients |= d["brca2"]
        elif gene == "RAD51C":
            gene_patients |= d["rad51c"]
    n = len(gene_patients)
    tp53 = path_mut_dedup[(path_mut_dedup["patientId"].isin(gene_patients)) &
                          (path_mut_dedup["gene.hugoGeneSymbol"] == "TP53")]["patientId"].nunique()
    if n > 0:
        rprint(f"  {gene}: TP53 co-mut {tp53}/{n} ({100*tp53/n:.1f}%)")

# Fisher exact BRCA1 vs BRCA2 for TP53
brca1_all = set()
brca2_all = set()
for d in datasets.values():
    brca1_all |= d["brca1"]
    brca2_all |= d["brca2"]
tp53_b1 = path_mut_dedup[(path_mut_dedup["patientId"].isin(brca1_all)) &
                          (path_mut_dedup["gene.hugoGeneSymbol"] == "TP53")]["patientId"].nunique()
tp53_b2 = path_mut_dedup[(path_mut_dedup["patientId"].isin(brca2_all)) &
                          (path_mut_dedup["gene.hugoGeneSymbol"] == "TP53")]["patientId"].nunique()
n1, n2 = len(brca1_all), len(brca2_all)
table = [[tp53_b1, n1 - tp53_b1], [tp53_b2, n2 - tp53_b2]]
_, p = stats.fisher_exact(table)
rprint(f"  Fisher exact (BRCA1 vs BRCA2 TP53 co-mut): p={p:.6f}")

# Prediction 4: RAD51C should resemble BRCA2 not BRCA1
rprint("\nPrediction: RAD51C phenotype resembles BRCA2 (not BRCA1)")
rprint("  (See cancer type distribution, breast subtype, and co-mutation data above)")
rad51c_all = set()
for d in datasets.values():
    rad51c_all |= d["rad51c"]
rprint(f"  RAD51C total patients: {len(rad51c_all)}")
if len(rad51c_all) > 0:
    # RAD51C TP53 rate
    tp53_r = path_mut_dedup[(path_mut_dedup["patientId"].isin(rad51c_all)) &
                             (path_mut_dedup["gene.hugoGeneSymbol"] == "TP53")]["patientId"].nunique()
    rprint(f"  RAD51C TP53 co-mut: {tp53_r}/{len(rad51c_all)} ({100*tp53_r/len(rad51c_all):.1f}%)")
    rprint(f"  Compare: BRCA1 TP53={100*tp53_b1/n1:.1f}%, BRCA2 TP53={100*tp53_b2/n2:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# 6. METASTATIC TROPISM BAR CHART
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating tropism bar chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (title, df) in enumerate([
    ("All Cancers", met_grouped),
    ("Breast Cancer Only", met_breast_only)
]):
    ax = axes[idx]
    sites_short = [s.replace("DMETS_DX_", "") for s in MET_SITES]
    x = np.arange(len(MET_SITES))
    width = 0.25
    for i, gene in enumerate(["BRCA1", "BRCA2", "RAD51C"]):
        sub = df[df["brca_group"] == gene]
        n = len(sub)
        if n == 0:
            continue
        pcts = []
        for site in MET_SITES:
            if site in sub.columns:
                pcts.append(100 * (sub[site] == "Yes").sum() / n)
            else:
                pcts.append(0)
        ax.bar(x + i * width, pcts, width, label=f"{gene} (n={n})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sites_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% with metastasis")
    ax.set_title(title)
    ax.legend()

fig.suptitle("Metastatic Tropism: BRCA1 vs BRCA2 vs RAD51C", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/met_tropism_bars.png", dpi=150)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# 7. CANCER TYPE DISTRIBUTION BAR CHART
# ═══════════════════════════════════════════════════════════════════════
print("Generating cancer type bar chart...")
fig, ax = plt.subplots(figsize=(12, 6))
top_types = grouped["CANCER_TYPE"].value_counts().head(8).index.tolist()
x = np.arange(len(top_types))
width = 0.25
for i, gene in enumerate(["BRCA1", "BRCA2", "RAD51C"]):
    sub = grouped[grouped["brca_group"] == gene]
    n = len(sub)
    if n == 0:
        continue
    pcts = [100 * (sub["CANCER_TYPE"] == ct).sum() / n for ct in top_types]
    ax.bar(x + i * width, pcts, width, label=f"{gene} (n={n})")
ax.set_xticks(x + width)
ax.set_xticklabels(top_types, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("% of patients")
ax.set_title("Cancer Type Distribution: BRCA1 vs BRCA2 vs RAD51C")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/cancer_type_dist.png", dpi=150)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# 8. BREAST SUBTYPE PIE/BAR CHART
# ═══════════════════════════════════════════════════════════════════════
print("Generating breast subtype chart...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for i, gene in enumerate(["BRCA1", "BRCA2", "RAD51C"]):
    ax = axes[i]
    sub = breast[breast["brca_group"] == gene]
    n = len(sub)
    if n == 0:
        ax.set_title(f"{gene} (n=0)")
        ax.axis("off")
        continue
    vc = sub["breast_subtype"].value_counts()
    colors = {"Triple-Negative": "#e74c3c", "HR+/HER2-": "#3498db",
              "HER2+": "#2ecc71", "Unspecified": "#95a5a6"}
    c = [colors.get(k, "#bdc3c7") for k in vc.index]
    ax.pie(vc.values, labels=[f"{k}\n({v})" for k, v in vc.items()],
           colors=c, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"{gene} (n={n})")
fig.suptitle("Breast Cancer Subtypes by BRCA Gene", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/breast_subtypes.png", dpi=150)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# WRITE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
summary_path = f"{OUT}/summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(report))
print(f"\nSummary written to {summary_path}")
print(f"Plots saved to {OUT}/")
print("Done.")
