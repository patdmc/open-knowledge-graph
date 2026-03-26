"""
Sensitivity analysis: does channel count outpredict mutation count
across different channel granularities (4, 6, 8, 10)?

Tests whether the core result is robust to the specific channel mapping.
"""

import os
import sys
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

sys.path.insert(0, 'analysis')
from channel_analysis import (
    DRIVER_MUTATION_TYPES, BASE, HEADERS,
    fetch_clinical, fetch_mutations
)


# ============================================================
# Channel mappings at different granularities
# ============================================================

# 4 CHANNELS: coarse grouping
CHANNELS_4 = {
    # Genome integrity (DDR + cell cycle)
    "BRCA1": "GenomeIntegrity", "BRCA2": "GenomeIntegrity", "PALB2": "GenomeIntegrity",
    "RAD51C": "GenomeIntegrity", "RAD51D": "GenomeIntegrity", "RAD51B": "GenomeIntegrity",
    "ATM": "GenomeIntegrity", "ATR": "GenomeIntegrity", "CHEK2": "GenomeIntegrity", "CHEK1": "GenomeIntegrity",
    "FANCA": "GenomeIntegrity", "FANCC": "GenomeIntegrity", "FANCD2": "GenomeIntegrity",
    "BAP1": "GenomeIntegrity", "BARD1": "GenomeIntegrity",
    "MLH1": "GenomeIntegrity", "MSH2": "GenomeIntegrity", "MSH6": "GenomeIntegrity", "PMS2": "GenomeIntegrity",
    "POLE": "GenomeIntegrity", "POLD1": "GenomeIntegrity",
    "TP53": "GenomeIntegrity", "RB1": "GenomeIntegrity",
    "CDKN1A": "GenomeIntegrity", "CDKN1B": "GenomeIntegrity",
    "CDKN2A": "GenomeIntegrity", "CDKN2B": "GenomeIntegrity",
    "CDK4": "GenomeIntegrity", "CDK6": "GenomeIntegrity",
    "CCND1": "GenomeIntegrity", "CCNE1": "GenomeIntegrity",
    "MDM2": "GenomeIntegrity", "MDM4": "GenomeIntegrity",
    "MYC": "GenomeIntegrity", "MYCN": "GenomeIntegrity",
    # Growth signaling (PI3K + RAS + receptors)
    "PIK3CA": "GrowthSignaling", "PIK3R1": "GrowthSignaling",
    "PTEN": "GrowthSignaling", "AKT1": "GrowthSignaling",
    "AKT2": "GrowthSignaling", "AKT3": "GrowthSignaling", "MTOR": "GrowthSignaling",
    "KRAS": "GrowthSignaling", "NRAS": "GrowthSignaling", "HRAS": "GrowthSignaling",
    "BRAF": "GrowthSignaling", "RAF1": "GrowthSignaling",
    "MAP2K1": "GrowthSignaling", "MAP2K2": "GrowthSignaling",
    "MAP3K1": "GrowthSignaling", "MAP3K13": "GrowthSignaling",
    "ERBB2": "GrowthSignaling", "ERBB3": "GrowthSignaling",
    "EGFR": "GrowthSignaling", "FGFR1": "GrowthSignaling",
    "FGFR2": "GrowthSignaling", "FGFR3": "GrowthSignaling",
    "IGF1R": "GrowthSignaling", "MET": "GrowthSignaling",
    "NF1": "GrowthSignaling", "NF2": "GrowthSignaling",
    "TSC1": "GrowthSignaling", "TSC2": "GrowthSignaling",
    "STK11": "GrowthSignaling", "ARID1A": "GrowthSignaling",
    # Hormonal + immune (organism-level coupling)
    "ESR1": "OrganismCoupling", "ESR2": "OrganismCoupling",
    "PGR": "OrganismCoupling", "AR": "OrganismCoupling",
    "FOXA1": "OrganismCoupling", "GATA3": "OrganismCoupling",
    "B2M": "OrganismCoupling", "HLA-A": "OrganismCoupling", "HLA-B": "OrganismCoupling",
    "HLA-C": "OrganismCoupling", "JAK1": "OrganismCoupling", "JAK2": "OrganismCoupling",
    "STAT1": "OrganismCoupling", "CD274": "OrganismCoupling",
    "PDCD1LG2": "OrganismCoupling", "CTLA4": "OrganismCoupling",
    # Tissue architecture
    "CDH1": "TissueArch", "CDH2": "TissueArch", "CTNNB1": "TissueArch",
    "APC": "TissueArch", "AXIN1": "TissueArch", "AXIN2": "TissueArch",
    "SMAD2": "TissueArch", "SMAD3": "TissueArch", "SMAD4": "TissueArch",
    "TGFBR1": "TissueArch", "TGFBR2": "TissueArch",
    "NOTCH1": "TissueArch", "NOTCH2": "TissueArch",
    "NOTCH3": "TissueArch", "NOTCH4": "TissueArch",
    "FBXW7": "TissueArch", "GJA1": "TissueArch", "GJB2": "TissueArch",
}

# 6 CHANNELS: the baseline (same as channel_analysis.py)
CHANNELS_6 = {
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
    "AKT2": "PI3K_Growth", "AKT3": "PI3K_Growth", "MTOR": "PI3K_Growth",
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
    "HLA-C": "Immune", "JAK1": "Immune", "JAK2": "Immune",
    "STAT1": "Immune", "CD274": "Immune",
    "PDCD1LG2": "Immune", "CTLA4": "Immune",
    "CDH1": "TissueArch", "CDH2": "TissueArch", "CTNNB1": "TissueArch",
    "APC": "TissueArch", "AXIN1": "TissueArch", "AXIN2": "TissueArch",
    "SMAD2": "TissueArch", "SMAD3": "TissueArch", "SMAD4": "TissueArch",
    "TGFBR1": "TissueArch", "TGFBR2": "TissueArch",
    "NOTCH1": "TissueArch", "NOTCH2": "TissueArch",
    "NOTCH3": "TissueArch", "NOTCH4": "TissueArch",
    "FBXW7": "TissueArch", "GJA1": "TissueArch", "GJB2": "TissueArch",
}

# 8 CHANNELS: split growth signaling into PI3K and RAS-MAPK,
# split DDR into HRR and mismatch repair
CHANNELS_8 = {
    # HRR (homologous recombination repair)
    "BRCA1": "HRR", "BRCA2": "HRR", "PALB2": "HRR",
    "RAD51C": "HRR", "RAD51D": "HRR", "RAD51B": "HRR",
    "ATM": "HRR", "ATR": "HRR", "CHEK2": "HRR", "CHEK1": "HRR",
    "FANCA": "HRR", "FANCC": "HRR", "FANCD2": "HRR",
    "BAP1": "HRR", "BARD1": "HRR",
    # Mismatch repair
    "MLH1": "MMR", "MSH2": "MMR", "MSH6": "MMR", "PMS2": "MMR",
    "POLE": "MMR", "POLD1": "MMR",
    # Cell cycle
    "TP53": "CellCycle", "RB1": "CellCycle",
    "CDKN1A": "CellCycle", "CDKN1B": "CellCycle",
    "CDKN2A": "CellCycle", "CDKN2B": "CellCycle",
    "CDK4": "CellCycle", "CDK6": "CellCycle",
    "CCND1": "CellCycle", "CCNE1": "CellCycle",
    "MDM2": "CellCycle", "MDM4": "CellCycle",
    "MYC": "CellCycle", "MYCN": "CellCycle",
    # PI3K-AKT-mTOR
    "PIK3CA": "PI3K", "PIK3R1": "PI3K",
    "PTEN": "PI3K", "AKT1": "PI3K",
    "AKT2": "PI3K", "AKT3": "PI3K", "MTOR": "PI3K",
    "TSC1": "PI3K", "TSC2": "PI3K",
    "STK11": "PI3K", "ARID1A": "PI3K",
    # RAS-MAPK
    "KRAS": "RAS_MAPK", "NRAS": "RAS_MAPK", "HRAS": "RAS_MAPK",
    "BRAF": "RAS_MAPK", "RAF1": "RAS_MAPK",
    "MAP2K1": "RAS_MAPK", "MAP2K2": "RAS_MAPK",
    "MAP3K1": "RAS_MAPK", "MAP3K13": "RAS_MAPK",
    "NF1": "RAS_MAPK", "NF2": "RAS_MAPK",
    # Receptor tyrosine kinases
    "ERBB2": "RAS_MAPK", "ERBB3": "RAS_MAPK",
    "EGFR": "RAS_MAPK", "FGFR1": "RAS_MAPK",
    "FGFR2": "RAS_MAPK", "FGFR3": "RAS_MAPK",
    "IGF1R": "RAS_MAPK", "MET": "RAS_MAPK",
    # Endocrine
    "ESR1": "Endocrine", "ESR2": "Endocrine",
    "PGR": "Endocrine", "AR": "Endocrine",
    "FOXA1": "Endocrine", "GATA3": "Endocrine",
    # Immune
    "B2M": "Immune", "HLA-A": "Immune", "HLA-B": "Immune",
    "HLA-C": "Immune", "JAK1": "Immune", "JAK2": "Immune",
    "STAT1": "Immune", "CD274": "Immune",
    "PDCD1LG2": "Immune", "CTLA4": "Immune",
    # Tissue architecture
    "CDH1": "TissueArch", "CDH2": "TissueArch", "CTNNB1": "TissueArch",
    "APC": "TissueArch", "AXIN1": "TissueArch", "AXIN2": "TissueArch",
    "SMAD2": "TissueArch", "SMAD3": "TissueArch", "SMAD4": "TissueArch",
    "TGFBR1": "TissueArch", "TGFBR2": "TissueArch",
    "NOTCH1": "TissueArch", "NOTCH2": "TissueArch",
    "NOTCH3": "TissueArch", "NOTCH4": "TissueArch",
    "FBXW7": "TissueArch", "GJA1": "TissueArch", "GJB2": "TissueArch",
}

# 10 CHANNELS: further splits — tissue arch into Wnt/Notch and ECM/adhesion,
# cell cycle into p53/Rb and CDK/cyclin, add chromatin remodeling
CHANNELS_10 = {
    # HRR
    "BRCA1": "HRR", "BRCA2": "HRR", "PALB2": "HRR",
    "RAD51C": "HRR", "RAD51D": "HRR", "RAD51B": "HRR",
    "ATM": "HRR", "ATR": "HRR", "CHEK2": "HRR", "CHEK1": "HRR",
    "FANCA": "HRR", "FANCC": "HRR", "FANCD2": "HRR",
    "BAP1": "HRR", "BARD1": "HRR",
    # MMR
    "MLH1": "MMR", "MSH2": "MMR", "MSH6": "MMR", "PMS2": "MMR",
    "POLE": "MMR", "POLD1": "MMR",
    # p53/Rb checkpoint
    "TP53": "p53_Rb", "RB1": "p53_Rb",
    "MDM2": "p53_Rb", "MDM4": "p53_Rb",
    "CDKN2A": "p53_Rb", "CDKN2B": "p53_Rb",
    # CDK/cyclin proliferation
    "CDK4": "CDK_Cyclin", "CDK6": "CDK_Cyclin",
    "CCND1": "CDK_Cyclin", "CCNE1": "CDK_Cyclin",
    "CDKN1A": "CDK_Cyclin", "CDKN1B": "CDK_Cyclin",
    "MYC": "CDK_Cyclin", "MYCN": "CDK_Cyclin",
    # PI3K-AKT-mTOR
    "PIK3CA": "PI3K", "PIK3R1": "PI3K",
    "PTEN": "PI3K", "AKT1": "PI3K",
    "AKT2": "PI3K", "AKT3": "PI3K", "MTOR": "PI3K",
    "TSC1": "PI3K", "TSC2": "PI3K",
    "STK11": "PI3K",
    # RAS-MAPK
    "KRAS": "RAS_MAPK", "NRAS": "RAS_MAPK", "HRAS": "RAS_MAPK",
    "BRAF": "RAS_MAPK", "RAF1": "RAS_MAPK",
    "MAP2K1": "RAS_MAPK", "MAP2K2": "RAS_MAPK",
    "MAP3K1": "RAS_MAPK", "MAP3K13": "RAS_MAPK",
    "NF1": "RAS_MAPK", "NF2": "RAS_MAPK",
    "ERBB2": "RAS_MAPK", "ERBB3": "RAS_MAPK",
    "EGFR": "RAS_MAPK", "FGFR1": "RAS_MAPK",
    "FGFR2": "RAS_MAPK", "FGFR3": "RAS_MAPK",
    "IGF1R": "RAS_MAPK", "MET": "RAS_MAPK",
    # Endocrine
    "ESR1": "Endocrine", "ESR2": "Endocrine",
    "PGR": "Endocrine", "AR": "Endocrine",
    "FOXA1": "Endocrine", "GATA3": "Endocrine",
    # Immune
    "B2M": "Immune", "HLA-A": "Immune", "HLA-B": "Immune",
    "HLA-C": "Immune", "JAK1": "Immune", "JAK2": "Immune",
    "STAT1": "Immune", "CD274": "Immune",
    "PDCD1LG2": "Immune", "CTLA4": "Immune",
    # Wnt/Notch signaling
    "APC": "Wnt_Notch", "AXIN1": "Wnt_Notch", "AXIN2": "Wnt_Notch",
    "CTNNB1": "Wnt_Notch",
    "NOTCH1": "Wnt_Notch", "NOTCH2": "Wnt_Notch",
    "NOTCH3": "Wnt_Notch", "NOTCH4": "Wnt_Notch",
    "FBXW7": "Wnt_Notch",
    # Adhesion/ECM (tissue structure)
    "CDH1": "Adhesion_ECM", "CDH2": "Adhesion_ECM",
    "SMAD2": "Adhesion_ECM", "SMAD3": "Adhesion_ECM", "SMAD4": "Adhesion_ECM",
    "TGFBR1": "Adhesion_ECM", "TGFBR2": "Adhesion_ECM",
    "GJA1": "Adhesion_ECM", "GJB2": "Adhesion_ECM",
    # Chromatin remodeling (moved from PI3K_Growth)
    "ARID1A": "Chromatin",
}


# Helper: remap an existing dict to new channel names
def _remap(base, mapping):
    """Remap channel names: mapping = {old_channel: new_channel}."""
    return {gene: mapping.get(ch, ch) for gene, ch in base.items()}


# 2 CHANNELS: cell-intrinsic vs cell-extrinsic
CHANNELS_2 = _remap(CHANNELS_6, {
    "DDR": "CellIntrinsic", "CellCycle": "CellIntrinsic", "PI3K_Growth": "CellIntrinsic",
    "Endocrine": "CellExtrinsic", "Immune": "CellExtrinsic", "TissueArch": "CellExtrinsic",
})

# 3 CHANNELS: genome integrity, growth signaling, microenvironment
CHANNELS_3 = _remap(CHANNELS_6, {
    "DDR": "GenomeIntegrity", "CellCycle": "GenomeIntegrity",
    "PI3K_Growth": "GrowthSignaling",
    "Endocrine": "Microenvironment", "Immune": "Microenvironment", "TissueArch": "Microenvironment",
})

# 5 CHANNELS: same as 6 but merge endocrine + immune
CHANNELS_5 = _remap(CHANNELS_6, {
    "Endocrine": "OrganismCoupling", "Immune": "OrganismCoupling",
})

# 7 CHANNELS: same as 6 but split DDR into HRR and MMR
CHANNELS_7 = dict(CHANNELS_6)
for gene in ["MLH1", "MSH2", "MSH6", "PMS2", "POLE", "POLD1"]:
    CHANNELS_7[gene] = "MMR"
for gene in ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "RAD51B",
             "ATM", "ATR", "CHEK2", "CHEK1", "FANCA", "FANCC", "FANCD2",
             "BAP1", "BARD1"]:
    CHANNELS_7[gene] = "HRR"


ALL_MAPPINGS = {
    2: CHANNELS_2,
    3: CHANNELS_3,
    4: CHANNELS_4,
    5: CHANNELS_5,
    6: CHANNELS_6,
    7: CHANNELS_7,
    8: CHANNELS_8,
    10: CHANNELS_10,
}


def compute_channel_count(mut_df, channel_map):
    """For each patient, count distinct channels hit."""
    gene_col = "gene.hugoGeneSymbol"
    if gene_col not in mut_df.columns and "keyword" in mut_df.columns:
        mut_df = mut_df.copy()
        mut_df[gene_col] = mut_df["keyword"].apply(
            lambda k: str(k).split()[0] if pd.notna(k) else None
        )

    driver = mut_df[mut_df["mutationType"].isin(DRIVER_MUTATION_TYPES)].copy()
    driver["channel"] = driver[gene_col].map(channel_map)
    mapped = driver.dropna(subset=["channel"])

    ch_count = mapped.groupby("patientId")["channel"].nunique().reset_index()
    ch_count.columns = ["patientId", "channel_count"]

    mut_count = driver.groupby("patientId").size().reset_index(name="driver_mutation_count")

    result = ch_count.merge(mut_count, on="patientId", how="outer")
    result["channel_count"] = result["channel_count"].fillna(0).astype(int)
    result["driver_mutation_count"] = result["driver_mutation_count"].fillna(0).astype(int)
    return result


def run_cox(df):
    """Run multivariate Cox: channel_count + log_mutation_count."""
    cox_df = df[["os_months", "event", "channel_count", "log_mut"]].dropna()
    cox_df = cox_df[cox_df["os_months"] > 0]

    results = {}

    # Channel only
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df[["os_months", "event", "channel_count"]],
                duration_col="os_months", event_col="event")
        s = cph.summary
        results["ch_uni_hr"] = s.loc["channel_count", "exp(coef)"]
        results["ch_uni_p"] = s.loc["channel_count", "p"]
    except Exception:
        results["ch_uni_hr"] = np.nan
        results["ch_uni_p"] = np.nan

    # Mutation only
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df[["os_months", "event", "log_mut"]],
                duration_col="os_months", event_col="event")
        s = cph.summary
        results["mut_uni_hr"] = s.loc["log_mut", "exp(coef)"]
        results["mut_uni_p"] = s.loc["log_mut", "p"]
    except Exception:
        results["mut_uni_hr"] = np.nan
        results["mut_uni_p"] = np.nan

    # Both
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df[["os_months", "event", "channel_count", "log_mut"]],
                duration_col="os_months", event_col="event")
        s = cph.summary
        results["ch_multi_hr"] = s.loc["channel_count", "exp(coef)"]
        results["ch_multi_p"] = s.loc["channel_count", "p"]
        results["mut_multi_hr"] = s.loc["log_mut", "exp(coef)"]
        results["mut_multi_p"] = s.loc["log_mut", "p"]
    except Exception:
        results["ch_multi_hr"] = np.nan
        results["ch_multi_p"] = np.nan
        results["mut_multi_hr"] = np.nan
        results["mut_multi_p"] = np.nan

    # AIC from multivariate model
    try:
        results["aic"] = cph.AIC_partial_
    except Exception:
        results["aic"] = np.nan

    return results


def main():
    studies = [
        ("msk_impact_2017", "MSK-IMPACT 2017"),
        ("msk_met_2021", "MSK MetTropism"),
        ("msk_impact_50k_2026", "MSK-IMPACT 50K"),
    ]

    cache_dir = "analysis/cache"

    print("=" * 80)
    print("CHANNEL SENSITIVITY ANALYSIS: 4, 6, 8, 10 channels")
    print("=" * 80)

    all_results = []

    for study_id, study_label in studies:
        print(f"\n{'='*60}")
        print(f"DATASET: {study_label} ({study_id})")
        print(f"{'='*60}")

        # Load data
        clinical_df = fetch_clinical(study_id, cache_dir)
        mut_df = fetch_mutations(study_id, cache_dir)

        # Build survival data
        surv = clinical_df[["OS_STATUS", "OS_MONTHS"]].copy().reset_index()
        surv.columns = ["patientId", "os_status", "os_months"]
        surv["os_months"] = pd.to_numeric(surv["os_months"], errors="coerce")
        surv["event"] = surv["os_status"].apply(
            lambda x: 1 if "DECEASED" in str(x) else 0
        )
        surv = surv.dropna(subset=["os_months"])

        for n_channels, channel_map in sorted(ALL_MAPPINGS.items()):
            ch_data = compute_channel_count(mut_df, channel_map)
            df = surv.merge(ch_data, on="patientId", how="left")
            df["channel_count"] = df["channel_count"].fillna(0).astype(int)
            df["driver_mutation_count"] = df["driver_mutation_count"].fillna(0).astype(int)
            df["log_mut"] = np.log1p(df["driver_mutation_count"])
            df = df.dropna(subset=["os_months", "event"])

            results = run_cox(df)
            results["study"] = study_label
            results["n_channels"] = n_channels
            results["n_patients"] = len(df)
            all_results.append(results)

            sig_ch = "***" if results["ch_multi_p"] < 0.005 else ("*" if results["ch_multi_p"] < 0.05 else "ns")
            sig_mut = "***" if results["mut_multi_p"] < 0.005 else ("*" if results["mut_multi_p"] < 0.05 else "ns")

            print(f"\n  {n_channels} channels:")
            print(f"    Channel count (multivariate): HR={results['ch_multi_hr']:.3f}, p={results['ch_multi_p']:.2e} {sig_ch}")
            print(f"    Log mut count (multivariate): HR={results['mut_multi_hr']:.3f}, p={results['mut_multi_p']:.2e} {sig_mut}")

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Study':<20} {'#Ch':>4} {'Ch HR':>7} {'Ch p':>12} {'Mut HR':>7} {'Mut p':>12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['study']:<20} {r['n_channels']:>4} "
              f"{r['ch_multi_hr']:>7.3f} {r['ch_multi_p']:>12.2e} "
              f"{r['mut_multi_hr']:>7.3f} {r['mut_multi_p']:>12.2e}")

    # Save results
    outdir = "analysis/results/sensitivity"
    os.makedirs(outdir, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{outdir}/channel_sensitivity.csv", index=False)
    print(f"\nSaved to {outdir}/channel_sensitivity.csv")


if __name__ == "__main__":
    main()
