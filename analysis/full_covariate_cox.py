"""
Full covariate Cox regression analysis.
Tests whether channel_count remains significant after controlling for
ALL available clinical covariates from cBioPortal.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / "cache"
RESULTS_DIR = BASE / "results" / "full_covariate"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API = "https://www.cbioportal.org/api"

STUDIES = ["msk_impact_2017", "msk_met_2021", "msk_impact_50k_2026", "brca_metabric"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_clinical(study_id, data_type, max_retries=3):
    """Fetch clinical data from cBioPortal API with retries."""
    url = f"{API}/studies/{study_id}/clinical-data"
    params = {"clinicalDataType": data_type, "projection": "DETAILED"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt+1} for {study_id} {data_type}: {e}")
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  FAILED to fetch {study_id} {data_type}: {e}")
                return None


def pivot_clinical(records, id_col="patientId"):
    """Pivot long-form clinical JSON to wide DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    # Each record has patientId (and possibly sampleId), clinicalAttributeId, value
    if id_col == "sampleId" and "sampleId" not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=id_col,
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first"
    ).reset_index()
    return pivot


def get_full_clinical(study_id):
    """Fetch + cache full clinical data for a study."""
    cache_path = CACHE_DIR / f"{study_id}_full_clinical.csv"
    if cache_path.exists():
        print(f"  Using cached {cache_path.name}")
        return pd.read_csv(cache_path)

    print(f"  Fetching PATIENT clinical data...")
    patient_raw = fetch_clinical(study_id, "PATIENT")
    patient_df = pivot_clinical(patient_raw, "patientId")
    print(f"    Patient data: {patient_df.shape}")

    print(f"  Fetching SAMPLE clinical data...")
    sample_raw = fetch_clinical(study_id, "SAMPLE")
    sample_df = pivot_clinical(sample_raw, "sampleId")
    print(f"    Sample data: {sample_df.shape}")

    # For sample data, we need patientId. Extract from raw records.
    if not sample_df.empty and sample_raw:
        # Build sampleId -> patientId map
        sid_pid = {}
        for rec in sample_raw:
            sid = rec.get("sampleId", "")
            pid = rec.get("patientId", "")
            if sid and pid:
                sid_pid[sid] = pid
        sample_df["patientId"] = sample_df["sampleId"].map(sid_pid)
        # Take first sample per patient
        sample_df = sample_df.sort_values("sampleId").groupby("patientId").first().reset_index()
        sample_df = sample_df.drop(columns=["sampleId"], errors="ignore")

    # Merge patient + sample
    if not patient_df.empty and not sample_df.empty:
        # Avoid column collisions — sample cols override patient if both exist
        overlap = set(patient_df.columns) & set(sample_df.columns) - {"patientId"}
        if overlap:
            sample_df = sample_df.rename(columns={c: f"{c}_SAMPLE" for c in overlap})
        merged = patient_df.merge(sample_df, on="patientId", how="outer")
    elif not patient_df.empty:
        merged = patient_df
    elif not sample_df.empty:
        merged = sample_df
    else:
        return pd.DataFrame()

    merged.to_csv(cache_path, index=False)
    print(f"    Cached to {cache_path.name}: {merged.shape}")
    return merged


def try_numeric(series):
    """Try to convert a series to numeric."""
    return pd.to_numeric(series, errors="coerce")


def prepare_covariates(clinical_df, patient_df):
    """
    Merge clinical with patient_data, prepare covariates for Cox model.
    Returns (df_ready, covariate_names) where df_ready has os_months, event,
    channel_count, driver_mutation_count, and all covariates.
    """
    # Merge
    merged = patient_df.merge(clinical_df, on="patientId", how="inner")
    print(f"    Merged N = {len(merged)}")

    if len(merged) < 50:
        print(f"    Too few patients after merge, skipping")
        return None, None

    # Start building covariate matrix
    # Keep our core columns
    core_cols = ["patientId", "os_months", "event", "channel_count", "driver_mutation_count"]
    result = merged[core_cols].copy()

    # Identify potential numeric covariates
    numeric_attrs = [
        "CVR_TMB_SCORE", "TMB_NONSYNONYMOUS", "TMB_SCORE",
        "MSI_SCORE", "FRACTION_GENOME_ALTERED", "TUMOR_PURITY",
        "MUTATION_COUNT", "AGE", "AGE_AT_DX", "AGE_AT_DIAGNOSIS",
        "AGE_AT_SEQ_REPORT", "AGE_AT_SEQUENCING", "AGE_AT_PROCUREMENT",
        "SAMPLE_COVERAGE", "GENOME_DOUBLING", "ANEUPLOIDY_SCORE",
        "BUFFA_HYPOXIA_SCORE", "LYMPH_NODES_EXAMINED_POSITIVE",
        "LYMPH_NODES_EXAMINED_NUMBER", "TUMOR_SIZE",
        "NEOPLASM_HISTOLOGIC_GRADE", "NOTTINGHAM_PROGNOSTIC_INDEX",
    ]

    added_covariates = []

    for attr in numeric_attrs:
        # Check for exact match or case-insensitive match
        matches = [c for c in merged.columns if c.upper() == attr.upper()]
        if not matches:
            continue
        col = matches[0]
        vals = try_numeric(merged[col])
        frac_missing = vals.isna().mean()
        if frac_missing > 0.50:
            continue
        # Check variance
        if vals.std() < 1e-10:
            continue
        name = attr.upper()
        result[name] = vals.values
        added_covariates.append(name)

    # Categorical covariates
    cat_attrs = {
        "SEX": 10,
        "MSI_TYPE": 10,
        "MSI_STATUS": 10,
        "SAMPLE_TYPE": 10,
        "SMOKING_HISTORY": 10,
        "CANCER_TYPE": 100,
        "CANCER_TYPE_DETAILED": 200,
        "SAMPLE_CLASS": 10,
        "METASTATIC_SITE": 50,
        "PRIMARY_SITE": 50,
        "TUMOR_SITE": 50,
        "ONCOTREE_CODE": 200,
        "GENE_PANEL": 20,
        "ER_STATUS": 5,
        "HER2_STATUS": 5,
        "PR_STATUS": 5,
        "HORMONE_THERAPY": 5,
        "CHEMOTHERAPY": 5,
        "RADIO_THERAPY": 5,
        "CELLULARITY": 5,
        "HISTOLOGICAL_SUBTYPE": 50,
        "BREAST_SURGERY": 10,
        "INFERRED_MENOPAUSAL_STATE": 5,
        "INTCLUST": 15,
        "CLAUDIN_SUBTYPE": 10,
        "THREEGENE": 10,
    }

    for attr, max_levels in cat_attrs.items():
        matches = [c for c in merged.columns if c.upper() == attr.upper()]
        if not matches:
            continue
        col = matches[0]
        vals = merged[col].copy()
        # Drop NA-like values
        vals = vals.replace(["NA", "Unknown", "", "Not Available", "Not Collected",
                             "[Not Available]", "[Not Applicable]", "unknown", "UNKNOWN"],
                            np.nan)
        frac_missing = vals.isna().mean()
        if frac_missing > 0.50:
            continue
        # Count levels
        vc = vals.value_counts()
        if len(vc) < 2:
            continue
        # For many-level vars, drop rare categories
        if len(vc) > max_levels:
            # Keep only categories with >= 50 patients
            keep = vc[vc >= 50].index
            if len(keep) < 2:
                keep = vc.head(10).index
            vals[~vals.isin(keep)] = np.nan

        # Drop rare levels (<50 patients) for any categorical
        vc2 = vals.value_counts()
        keep_levels = vc2[vc2 >= 50].index if len(vc2[vc2 >= 50]) >= 2 else vc2.head(5).index
        if len(keep_levels) < 2:
            continue
        vals[~vals.isin(keep_levels)] = np.nan

        # One-hot encode
        dummies = pd.get_dummies(vals, prefix=attr.upper(), drop_first=True, dummy_na=False)
        for dc in dummies.columns:
            result[dc] = dummies[dc].values
            added_covariates.append(dc)

    print(f"    Covariates added: {len(added_covariates)}")
    print(f"    Columns: {added_covariates[:15]}{'...' if len(added_covariates) > 15 else ''}")

    return result, added_covariates


def run_cox(df, duration_col, event_col, covariates, label=""):
    """Run Cox PH model, return results dict or None."""
    cols = [duration_col, event_col] + covariates
    sub = df[cols].dropna().copy()
    n = len(sub)
    if n < 50:
        print(f"    {label}: N={n} too small, skipping")
        return None

    # Check for zero variance
    for c in covariates:
        if sub[c].std() < 1e-10:
            sub = sub.drop(columns=[c])
            covariates = [x for x in covariates if x != c]

    if "channel_count" not in covariates:
        print(f"    {label}: channel_count dropped (zero var?), skipping")
        return None

    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(sub, duration_col=duration_col, event_col=event_col)
    except Exception as e:
        print(f"    {label}: FAILED to converge — {e}")
        return None

    row = cph.summary.loc["channel_count"]
    hr = row["exp(coef)"]
    ci_low = row["exp(coef) lower 95%"]
    ci_high = row["exp(coef) upper 95%"]
    pval = row["p"]

    print(f"    {label} (N={n}): HR={hr:.4f} [{ci_low:.4f}–{ci_high:.4f}], p={pval:.2e}")
    return {
        "label": label,
        "N": n,
        "HR": hr,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "p": pval,
        "n_covariates": len(covariates),
        "covariates": covariates,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_results = []
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("FULL COVARIATE COX REGRESSION ANALYSIS")
    log("Does channel_count remain significant after controlling for ALL covariates?")
    log("=" * 80)

    for study_id in STUDIES:
        log(f"\n{'─' * 70}")
        log(f"STUDY: {study_id}")
        log(f"{'─' * 70}")

        # Check patient_data exists
        pd_path = BASE / "results" / study_id / "patient_data.csv"
        if not pd_path.exists():
            log(f"  patient_data.csv not found at {pd_path}, skipping.")
            continue

        patient_df = pd.read_csv(pd_path)
        # Parse event column
        if "event" in patient_df.columns:
            patient_df["event"] = patient_df["event"].astype(int)
        elif "os_status" in patient_df.columns:
            patient_df["event"] = patient_df["os_status"].apply(
                lambda x: 1 if "DECEASED" in str(x) else 0
            )
        patient_df["os_months"] = pd.to_numeric(patient_df["os_months"], errors="coerce")
        patient_df = patient_df.dropna(subset=["os_months", "event"])
        patient_df = patient_df[patient_df["os_months"] > 0]
        log(f"  Patient data: {len(patient_df)} patients with survival data")

        # Fetch full clinical
        clinical_df = get_full_clinical(study_id)
        if clinical_df.empty:
            log(f"  No clinical data fetched, skipping.")
            continue
        log(f"  Full clinical data: {clinical_df.shape}")
        log(f"  Clinical columns: {list(clinical_df.columns)[:20]}...")

        # Prepare covariates
        prepped, covariate_names = prepare_covariates(clinical_df, patient_df)
        if prepped is None:
            continue

        # ── Model A: channel_count only ──
        res_a = run_cox(prepped, "os_months", "event", ["channel_count"],
                        label=f"{study_id} — Model A (channel_count only)")

        # ── Model B: channel_count + driver_mutation_count ──
        res_b = run_cox(prepped, "os_months", "event",
                        ["channel_count", "driver_mutation_count"],
                        label=f"{study_id} — Model B (+driver_mutation_count)")

        # ── Model C: kitchen sink ──
        all_covs = ["channel_count", "driver_mutation_count"] + \
                   [c for c in covariate_names if c not in ["channel_count", "driver_mutation_count"]]
        res_c = run_cox(prepped, "os_months", "event", all_covs,
                        label=f"{study_id} — Model C (ALL covariates)")

        for res in [res_a, res_b, res_c]:
            if res:
                res["study"] = study_id
                all_results.append(res)

    # ── Summary Table ──
    log("\n" + "=" * 80)
    log("SUMMARY: channel_count hazard ratio across models")
    log("=" * 80)
    log(f"{'Study':<25} {'Model':<12} {'N':>6} {'HR':>7} {'95% CI':>20} {'p-value':>12} {'#Covs':>6}")
    log("-" * 90)

    for r in all_results:
        model = r["label"].split("—")[1].strip().split("(")[0].strip()
        ci_str = f"[{r['CI_low']:.3f}–{r['CI_high']:.3f}]"
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        log(f"{r['study']:<25} {model:<12} {r['N']:>6} {r['HR']:>7.3f} {ci_str:>20} {r['p']:>10.2e} {sig:>2} {r['n_covariates']:>5}")

    log("\n" + "-" * 90)
    log("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # ── Key finding ──
    log("\n" + "=" * 80)
    log("KEY FINDING")
    log("=" * 80)
    model_c_results = [r for r in all_results if "Model C" in r["label"]]
    if model_c_results:
        all_sig = all(r["p"] < 0.05 for r in model_c_results)
        if all_sig:
            log("channel_count is SIGNIFICANT (p<0.05) in ALL full-covariate models.")
            log("This means channel count predicts survival INDEPENDENTLY of:")
            log("  TMB, MSI status, cancer type, age, sex, tumor purity,")
            log("  fraction genome altered, and all other available covariates.")
        else:
            for r in model_c_results:
                status = "SIGNIFICANT" if r["p"] < 0.05 else "NOT significant"
                log(f"  {r['study']}: {status} (p={r['p']:.2e})")

    # Model C covariate details
    log("\n" + "=" * 80)
    log("FULL MODEL COVARIATES PER STUDY")
    log("=" * 80)
    for r in model_c_results:
        log(f"\n{r['study']} Model C ({r['n_covariates']} covariates, N={r['N']}):")
        covs = r["covariates"]
        for i in range(0, len(covs), 5):
            log(f"  {', '.join(covs[i:i+5])}")

    # Save
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
