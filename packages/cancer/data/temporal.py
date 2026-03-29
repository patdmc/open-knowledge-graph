"""
Temporal estimation for MSK-IMPACT patients.

MSK-IMPACT patient IDs are sequential (P-0000004 through P-0121625).
No explicit enrollment dates exist in the public data. We estimate
enrollment year from alive (censored) patients: their OS_MONTHS equals
time from enrollment to data cutoff, so enrollment ≈ cutoff - OS_MONTHS/12.

This gives us a patient_id → estimated_year mapping with per-patient
confidence (tighter for recent, denser enrollment periods).

The enrollment year is a proxy for treatment era. Standard-of-care changes
are temporal: checkpoint inhibitors (2014-2015), CDK4/6 inhibitors (2015),
PARP inhibitors (2014-2018), etc. Patients with the same mutations but
different enrollment years received different treatments, so their outcomes
differ for reasons invisible to the mutation profile alone.

Usage:
    from gnn.data.temporal import TemporalEstimator

    te = TemporalEstimator(clinical_df)  # df with patientId, OS_MONTHS, OS_STATUS
    te.fit()

    year = te.estimate_year(patient_id)           # single patient
    years, confs = te.estimate_all(patient_ids)    # batch

    # Per-cancer-type era baselines (death rate by era)
    baselines = te.era_baselines(clinical_df)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import defaultdict


# Approximate data cutoff for MSK-IMPACT 50K dataset
DATA_CUTOFF = 2023.5

# Key drug approval dates (FDA) — used as validation anchors and era boundaries
# Treatment era changes affect outcomes for patients with the same mutations.
DRUG_APPROVALS = {
    "olaparib_ovarian": 2014.9,       # Dec 2014
    "pembrolizumab_melanoma": 2014.7,  # Sep 2014
    "nivolumab_nsclc": 2015.2,         # Mar 2015
    "palbociclib_breast": 2015.1,      # Feb 2015
    "atezolizumab_bladder": 2016.4,    # May 2016
    "olaparib_breast": 2018.0,         # Jan 2018
    "entrectinib_ntrk": 2019.0,        # 2019
    "alpelisib_pik3ca_breast": 2019.4,  # May 2019
    "sotorasib_krasg12c": 2021.4,      # May 2021
}

# Treatment era breakpoints — major shifts in standard of care
# Used for era_baselines() to segment temporal analysis
TREATMENT_ERAS = [
    (2011.0, "pre-immunotherapy"),
    (2014.7, "early-checkpoint-inhibitors"),
    (2017.0, "combination-IO-era"),
    (2019.0, "precision-oncology-expansion"),
    (2021.0, "late-precision-era"),
]


class TemporalEstimator:
    """Estimates enrollment year from patient ID using censored survival data."""

    def __init__(self, clinical_df, cutoff=DATA_CUTOFF, n_bins=100):
        """
        Args:
            clinical_df: DataFrame with patientId, OS_MONTHS, OS_STATUS
            cutoff: approximate calendar date of data cutoff
            n_bins: number of bins for interpolation (more = smoother)
        """
        self.cutoff = cutoff
        self.n_bins = n_bins
        self._fit(clinical_df)

    def _fit(self, df):
        """Build patient_id → year interpolation from alive patients."""
        df = df.copy()
        df['pid_num'] = df['patientId'].str.extract(r'P-(\d+)').astype(float)
        df['deceased'] = df['OS_STATUS'].str.contains('DECEASED', na=False)

        # Alive patients with valid OS give us enrollment anchors
        alive = df[
            (~df['deceased']) &
            (df['OS_MONTHS'].notna()) &
            (df['OS_MONTHS'] > 0)
        ].copy()
        alive['est_enroll'] = self.cutoff - alive['OS_MONTHS'] / 12

        # Bin by patient ID, take median enrollment year per bin
        alive_sorted = alive.sort_values('pid_num')
        alive_sorted['pid_bin'] = pd.qcut(
            alive_sorted['pid_num'], self.n_bins, labels=False, duplicates='drop'
        )

        bin_stats = alive_sorted.groupby('pid_bin').agg(
            pid_median=('pid_num', 'median'),
            year_median=('est_enroll', 'median'),
            year_std=('est_enroll', 'std'),
            count=('pid_num', 'count'),
        ).reset_index()

        # Interpolation functions
        self._f_year = interp1d(
            bin_stats['pid_median'].values,
            bin_stats['year_median'].values,
            fill_value='extrapolate', kind='linear',
        )
        self._f_std = interp1d(
            bin_stats['pid_median'].values,
            bin_stats['year_std'].values,
            fill_value='extrapolate', kind='linear',
        )

        # Store range for clamping
        self._pid_min = df['pid_num'].min()
        self._pid_max = df['pid_num'].max()
        self._year_min = bin_stats['year_median'].min()
        self._year_max = bin_stats['year_median'].max()

        # Store all patient IDs for batch lookups
        self._pid_to_num = dict(zip(df['patientId'], df['pid_num']))

    def estimate_year(self, patient_id):
        """Estimate enrollment year for a single patient ID string."""
        pid_num = self._pid_to_num.get(patient_id)
        if pid_num is None:
            # Try parsing directly
            import re
            m = re.search(r'P-(\d+)', str(patient_id))
            if m:
                pid_num = float(m.group(1))
            else:
                return self.cutoff, 3.0  # unknown, high uncertainty

        year = float(self._f_year(pid_num))
        year = np.clip(year, self._year_min, self._year_max)
        conf = float(np.clip(self._f_std(pid_num), 0.3, 3.0))
        return year, conf

    def estimate_all(self, patient_ids):
        """Batch estimate for a list of patient IDs.

        Returns:
            years: np.array of estimated enrollment years
            confidences: np.array of uncertainty (std in years)
        """
        pid_nums = np.array([
            self._pid_to_num.get(pid, np.nan) for pid in patient_ids
        ])

        # Handle missing
        valid = ~np.isnan(pid_nums)
        years = np.full(len(patient_ids), self.cutoff)
        confs = np.full(len(patient_ids), 3.0)

        if valid.any():
            years[valid] = np.clip(
                self._f_year(pid_nums[valid]),
                self._year_min, self._year_max,
            )
            confs[valid] = np.clip(self._f_std(pid_nums[valid]), 0.3, 3.0)

        return years, confs

    def estimate_by_index(self, pid_nums_array):
        """Estimate for an array of numeric patient IDs (already extracted).

        Args:
            pid_nums_array: np.array of numeric patient IDs

        Returns:
            years: np.array
            confidences: np.array
        """
        years = np.clip(
            self._f_year(pid_nums_array),
            self._year_min, self._year_max,
        )
        confs = np.clip(self._f_std(pid_nums_array), 0.3, 3.0)
        return years, confs


def build_era_baselines(clinical_df, temporal_estimator, min_patients=30):
    """Build per-cancer-type, per-era baseline survival statistics.

    Uses mature patients only (enrolled >= 2 years before cutoff) so that
    death rates are meaningful rather than dominated by censoring.

    Returns:
        dict: {cancer_type: [{year, death_rate, median_os_dead, n}, ...]}
        Also returns a lookup function: era_baseline(cancer_type, year) → death_rate
    """
    df = clinical_df.copy()
    df['pid_num'] = df['patientId'].str.extract(r'P-(\d+)').astype(float)
    years, confs = temporal_estimator.estimate_by_index(df['pid_num'].values)
    df['est_year'] = years
    df['year_conf'] = confs
    df['deceased'] = df['OS_STATUS'].str.contains('DECEASED', na=False).astype(int)

    # Mature patients only: at least 2 years before cutoff
    mature = df[df['est_year'] <= temporal_estimator.cutoff - 2.0].copy()
    mature['est_year_half'] = (mature['est_year'] * 2).round() / 2

    baselines = {}
    lookup_data = {}  # (ct, year) → death_rate

    for ct in mature['CANCER_TYPE'].unique():
        ct_data = mature[mature['CANCER_TYPE'] == ct]
        eras = []
        for yr in sorted(ct_data['est_year_half'].unique()):
            sub = ct_data[ct_data['est_year_half'] == yr]
            if len(sub) < min_patients:
                continue
            dr = sub['deceased'].mean()
            dead = sub[sub['deceased'] == 1]
            med_os = dead['OS_MONTHS'].median() if len(dead) > 5 else np.nan
            eras.append({
                'year': yr, 'death_rate': dr,
                'median_os_dead': med_os, 'n': len(sub),
            })
            lookup_data[(ct, yr)] = dr

        if eras:
            baselines[ct] = eras

    # Build interpolation for each CT
    ct_interps = {}
    for ct, eras in baselines.items():
        if len(eras) >= 2:
            yrs = [e['year'] for e in eras]
            drs = [e['death_rate'] for e in eras]
            ct_interps[ct] = interp1d(
                yrs, drs, fill_value=(drs[0], drs[-1]),
                bounds_error=False, kind='linear',
            )

    def era_baseline(cancer_type, year):
        """Look up expected death rate for (cancer_type, year)."""
        if cancer_type in ct_interps:
            return float(ct_interps[cancer_type](year))
        # Fallback: overall death rate for that year
        return 0.4  # rough average

    return baselines, era_baseline


def calibrate_with_clinical(temporal_estimator, clinical_csv_path):
    """Tighten temporal estimates using cBioPortal clinical data.

    Uses AGE_AT_SEQUENCING, AGE_AT_SURGERY, AGE_AT_EVIDENCE_OF_METS, and
    AGE_AT_DEATH to cross-validate and refine enrollment year estimates.

    Logic:
      - If AGE_AT_SEQUENCING and AGE_AT_DEATH are both known, the gap gives
        survival time, which must be consistent with OS_MONTHS.
      - If AGE_AT_SURGERY < AGE_AT_SEQUENCING, the gap gives pre-sequencing
        treatment duration — earlier enrollment than sequencing.
      - MSK-IMPACT sequencing typically happens within 1-2 years of enrollment,
        so AGE_AT_SEQUENCING anchors the calendar year via birth year inference.

    Returns:
        adjustments: dict of {patient_id: (adjusted_year, adjusted_conf)}
        stats: dict with calibration statistics
    """
    df = pd.read_csv(clinical_csv_path)

    adjustments = {}
    stats = {"n_patients": len(df), "n_adjusted": 0, "mean_shift": 0.0}
    shifts = []

    for _, row in df.iterrows():
        pid = row.get("patient_id", "")
        if not pid or pid not in temporal_estimator._pid_to_num:
            continue

        orig_year, orig_conf = temporal_estimator.estimate_year(pid)

        # Use age-at-death + OS_MONTHS for cross-validation
        age_seq = _safe_float(row.get("AGE_AT_SEQUENCING"))
        age_death = _safe_float(row.get("AGE_AT_DEATH"))
        age_surgery = _safe_float(row.get("AGE_AT_SURGERY"))
        os_months = _safe_float(row.get("OS_MONTHS"))

        constraints = []

        if age_seq is not None and age_death is not None and os_months is not None:
            # Cross-check: age_death - age_seq should ≈ os_months/12
            # (sequencing happens near enrollment)
            implied_survival = age_death - age_seq
            actual_survival = os_months / 12
            if abs(implied_survival - actual_survival) < 1.0:
                # Consistent — sequencing ≈ enrollment
                # birth_year ≈ cutoff - os_months/12 - age_at_death... but we
                # don't know birth year. The consistency check itself tightens
                # confidence.
                constraints.append(("age_consistent", 0.0, 0.5))

        if age_surgery is not None and age_seq is not None:
            gap = age_seq - age_surgery
            if 0 < gap < 10:
                # Surgery happened before sequencing — enrollment likely closer
                # to surgery date. Shift estimate earlier by gap/2.
                constraints.append(("surgery_gap", -gap / 4, 0.3))

        if constraints:
            total_shift = sum(c[1] for c in constraints)
            conf_tightening = min(c[2] for c in constraints)
            adj_year = orig_year + total_shift
            adj_conf = min(orig_conf, conf_tightening + orig_conf * 0.5)
            adjustments[pid] = (adj_year, adj_conf)
            shifts.append(total_shift)

    if shifts:
        stats["n_adjusted"] = len(shifts)
        stats["mean_shift"] = float(np.mean(shifts))
        stats["median_shift"] = float(np.median(shifts))
        stats["mean_conf_before"] = float(np.mean([
            temporal_estimator.estimate_year(pid)[1]
            for pid in list(adjustments.keys())[:100]
        ]))

    return adjustments, stats


def _safe_float(val):
    """Convert to float, returning None if not possible."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def normalize_year(year, center=2019.5, scale=3.0):
    """Normalize enrollment year to roughly [-1, 1] range for model input."""
    return (year - center) / scale


def year_features(years, confs):
    """Build feature vector from temporal estimates.

    Returns (N, 3) array:
        [0] normalized year
        [1] confidence (inverse uncertainty, higher = more certain)
        [2] is_mature (1 if enrolled >= 2 years before cutoff)
    """
    n_year = normalize_year(years)
    confidence = 1.0 / np.clip(confs, 0.3, 3.0)
    is_mature = (years <= DATA_CUTOFF - 2.0).astype(np.float32)
    return np.stack([n_year, confidence, is_mature], axis=1).astype(np.float32)
