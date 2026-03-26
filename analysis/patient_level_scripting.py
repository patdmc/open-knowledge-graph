"""
Patient-level escalation scripting score.

For each patient, compute how "on-script" their mutations are:
  - Look up each mutation in the atlas (tier 1 > tier 2 > tier 3)
  - Count harmful vs protective atlas hits
  - Compute patient-level H/P ratio
  - Score = sum(log HR) across all atlas-matched mutations

Then: Cox PH of scripting score vs survival, within each cancer type
and pan-cancer. This tests the hypothesis at the patient level, not
just the cancer-type level.

Also: for patients with multiple mutations, measure whether the
second/third mutations are on-script (same channel as first, hub)
or off-script (different channel, leaf). Off-script additions should
predict better survival.
"""

import os
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    NON_SILENT, MSK_DATASETS,
)


def get_channel(gene):
    return CHANNEL_MAP.get(gene)


def get_position(gene):
    for ch, hubs in HUB_GENES.items():
        if gene in hubs:
            return "hub"
    if gene in CHANNEL_MAP:
        return "leaf"
    return None


def main():
    paths = MSK_DATASETS["msk_impact_50k"]
    print("Loading data...", flush=True)
    mutations = pd.read_csv(paths["mutations"])
    clinical = pd.read_csv(paths["clinical"])
    sample_clinical = pd.read_csv(paths["sample_clinical"])

    mutations = mutations[mutations['mutationType'].isin(NON_SILENT)]
    mutations = mutations[mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())]

    clinical = clinical.merge(
        sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
        on='patientId', how='left'
    )
    clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    clinical['event'] = clinical['OS_STATUS'].apply(
        lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
    )
    clinical['age'] = pd.to_numeric(clinical.get('AGE_AT_DX', 60), errors='coerce').fillna(60)
    clinical['age_z'] = (clinical['age'] - clinical['age'].mean()) / (clinical['age'].std() + 1e-8)

    # Load atlas
    atlas_path = os.path.join(os.path.dirname(__file__), "survival_atlas_full.csv")
    atlas = pd.read_csv(atlas_path)

    # Build atlas lookups: (cancer_type, gene, pc) -> hr for tier 1
    #                       (cancer_type, gene) -> hr for tier 2
    #                       (cancer_type, channel) -> hr for tier 3
    t1_lookup = {}
    t2_lookup = {}
    t3_lookup = {}

    for _, r in atlas.iterrows():
        if r['p_value'] > 0.10:  # only use marginally significant
            continue
        if r['tier'] == 1:
            t1_lookup[(r['cancer_type'], r['gene'], r['protein_change'])] = r['hr']
        elif r['tier'] == 2:
            t2_lookup[(r['cancer_type'], r['gene'])] = r['hr']
        elif r['tier'] == 3:
            t3_lookup[(r['cancer_type'], r['channel'])] = r['hr']

    print(f"Atlas lookups: T1={len(t1_lookup)}, T2={len(t2_lookup)}, T3={len(t3_lookup)}", flush=True)

    # Group mutations by patient
    patient_muts = {}
    for pid, group in mutations.groupby('patientId'):
        patient_muts[pid] = group

    # ===================================================================
    # Compute per-patient scores
    # ===================================================================
    print("Computing patient scores...", flush=True)

    records = []

    for _, row in clinical.iterrows():
        pid = row['patientId']
        ct = row['CANCER_TYPE']
        pmuts = patient_muts.get(pid)

        if pmuts is None or len(pmuts) == 0:
            records.append({
                'patientId': pid,
                'cancer_type': ct,
                'OS_MONTHS': row['OS_MONTHS'],
                'event': row['event'],
                'age_z': row['age_z'],
                'n_mutations': 0,
                'n_channels_hit': 0,
                'n_harmful': 0,
                'n_protective': 0,
                'sum_log_hr': 0.0,
                'mean_log_hr': 0.0,
                'hub_fraction': 0.0,
                'n_on_script': 0,
                'n_off_script': 0,
                'off_script_ratio': 0.0,
                'has_atlas_hit': 0,
            })
            continue

        n_muts = len(pmuts)
        channels_hit = set()
        log_hrs = []
        n_harmful = 0
        n_protective = 0
        n_hub = 0
        n_leaf = 0

        # Track channels for on/off script
        mutation_channels = []

        for _, m in pmuts.iterrows():
            gene = m['gene.hugoGeneSymbol']
            pc = m.get('proteinChange', '')
            ch = get_channel(gene)
            pos = get_position(gene)

            if ch:
                channels_hit.add(ch)
                mutation_channels.append(ch)
            if pos == 'hub':
                n_hub += 1
            else:
                n_leaf += 1

            # Look up HR (most specific tier first)
            hr = None
            if (ct, gene, pc) in t1_lookup:
                hr = t1_lookup[(ct, gene, pc)]
            elif (ct, gene) in t2_lookup:
                hr = t2_lookup[(ct, gene)]
            elif ch and (ct, ch) in t3_lookup:
                hr = t3_lookup[(ct, ch)]

            if hr is not None:
                log_hrs.append(np.log(hr))
                if hr > 1.1:
                    n_harmful += 1
                elif hr < 0.9:
                    n_protective += 1

        # On-script vs off-script: after the first mutation's channel,
        # are subsequent mutations in the same channel (on-script) or different?
        n_on_script = 0
        n_off_script = 0
        if len(mutation_channels) > 1:
            first_ch = mutation_channels[0]
            for ch in mutation_channels[1:]:
                if ch == first_ch:
                    n_on_script += 1
                else:
                    n_off_script += 1

        off_script_ratio = n_off_script / max(n_on_script + n_off_script, 1)

        records.append({
            'patientId': pid,
            'cancer_type': ct,
            'OS_MONTHS': row['OS_MONTHS'],
            'event': row['event'],
            'age_z': row['age_z'],
            'n_mutations': n_muts,
            'n_channels_hit': len(channels_hit),
            'n_harmful': n_harmful,
            'n_protective': n_protective,
            'sum_log_hr': sum(log_hrs) if log_hrs else 0.0,
            'mean_log_hr': np.mean(log_hrs) if log_hrs else 0.0,
            'hub_fraction': n_hub / max(n_hub + n_leaf, 1),
            'n_on_script': n_on_script,
            'n_off_script': n_off_script,
            'off_script_ratio': off_script_ratio,
            'has_atlas_hit': 1 if log_hrs else 0,
        })

    df = pd.DataFrame(records)
    df = df[df['OS_MONTHS'] > 0]

    print(f"Patients: {len(df)}", flush=True)
    print(f"With atlas hits: {df['has_atlas_hit'].sum()} ({df['has_atlas_hit'].mean():.1%})", flush=True)
    print(f"With ≥2 mutations: {(df['n_mutations'] >= 2).sum()}", flush=True)

    # ===================================================================
    # Pan-cancer Cox models
    # ===================================================================
    print(f"\n{'='*70}")
    print("PAN-CANCER COX MODELS")
    print(f"{'='*70}", flush=True)

    scored = df[df['has_atlas_hit'] == 1].copy()

    for metric in ['sum_log_hr', 'mean_log_hr', 'n_harmful', 'n_protective',
                    'hub_fraction', 'off_script_ratio', 'n_channels_hit']:
        test_df = scored[['OS_MONTHS', 'event', 'age_z', metric]].copy()
        test_df = test_df.dropna()
        if test_df[metric].std() < 1e-8:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')
            hr = np.exp(cph.params_[metric])
            p = cph.summary.loc[metric, 'p']
            ci = cph.confidence_intervals_.loc[metric]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {metric:25s}: HR={hr:.3f} [{np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f}] "
                  f"p={p:.2e} {sig}", flush=True)
        except Exception as e:
            print(f"  {metric:25s}: FAILED ({e})", flush=True)

    # ===================================================================
    # Per-cancer-type Cox for sum_log_hr
    # ===================================================================
    print(f"\n{'='*70}")
    print("PER-CANCER-TYPE: sum_log_hr PREDICTING SURVIVAL")
    print(f"{'='*70}", flush=True)

    valid_cts = df['cancer_type'].value_counts()
    valid_cts = valid_cts[valid_cts >= 200].index.tolist()

    ct_results = []
    for ct in sorted(valid_cts):
        ct_df = scored[scored['cancer_type'] == ct].copy()
        if len(ct_df) < 50 or ct_df['sum_log_hr'].std() < 1e-8:
            continue

        test_df = ct_df[['OS_MONTHS', 'event', 'age_z', 'sum_log_hr']].copy()
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')
            hr = np.exp(cph.params_['sum_log_hr'])
            p = cph.summary.loc['sum_log_hr', 'p']
            ci = cph.confidence_intervals_.loc['sum_log_hr']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            ct_results.append({
                'cancer_type': ct, 'hr': hr, 'p': p, 'n': len(ct_df),
                'ci_low': np.exp(ci.iloc[0]), 'ci_high': np.exp(ci.iloc[1]),
            })
            print(f"  {ct:40s} n={len(ct_df):5d} HR={hr:.3f} "
                  f"[{np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f}] "
                  f"p={p:.2e} {sig}", flush=True)
        except Exception:
            continue

    # ===================================================================
    # Off-script analysis: patients with ≥2 mutations
    # ===================================================================
    print(f"\n{'='*70}")
    print("OFF-SCRIPT ANALYSIS (patients with ≥2 channel mutations)")
    print(f"{'='*70}", flush=True)

    multi = df[df['n_mutations'] >= 2].copy()
    print(f"Patients with ≥2 mutations: {len(multi)}", flush=True)

    # Quartiles of off-script ratio
    try:
        multi['off_script_q'] = pd.qcut(multi['off_script_ratio'], q=4,
                                          labels=False, duplicates='drop')
        q_labels = {i: f"Q{i+1}" for i in range(multi['off_script_q'].nunique())}
        q_labels[min(q_labels)] = f"Q1_scripted"
        q_labels[max(q_labels)] = f"Q{max(q_labels)+1}_diffuse"
        multi['off_script_q'] = multi['off_script_q'].map(q_labels)
    except Exception:
        multi['off_script_q'] = pd.cut(multi['off_script_ratio'],
                                        bins=[-0.01, 0.0, 0.5, 1.01],
                                        labels=['all_same_ch', 'mixed', 'all_diff_ch'])

    print(f"\nSurvival by off-script quartile:")
    for q in sorted(multi['off_script_q'].unique()):
        qdf = multi[multi['off_script_q'] == q]
        med_os = qdf['OS_MONTHS'].median()
        evt = qdf['event'].mean()
        mean_ch = qdf['n_channels_hit'].mean()
        print(f"  {q:15s}: n={len(qdf):5d}, medOS={med_os:5.1f}m, "
              f"event={evt:.1%}, channels={mean_ch:.1f}", flush=True)

    # Cox for off_script_ratio (adjusted for n_mutations and age)
    test_df = multi[['OS_MONTHS', 'event', 'age_z', 'off_script_ratio',
                      'n_mutations']].copy()
    test_df['n_mutations_z'] = (test_df['n_mutations'] - test_df['n_mutations'].mean()) / (
        test_df['n_mutations'].std() + 1e-8)
    test_df = test_df.drop(columns='n_mutations')

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')
        print(f"\nCox PH (off_script_ratio, adjusted for n_mutations + age):")
        for var in ['off_script_ratio', 'n_mutations_z', 'age_z']:
            hr = np.exp(cph.params_[var])
            p = cph.summary.loc[var, 'p']
            ci = cph.confidence_intervals_.loc[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:25s}: HR={hr:.3f} [{np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f}] "
                  f"p={p:.2e} {sig}", flush=True)
    except Exception as e:
        print(f"  Cox failed: {e}", flush=True)

    # ===================================================================
    # Channel-spread analysis
    # ===================================================================
    print(f"\n{'='*70}")
    print("CHANNEL SPREAD: more channels hit = ?")
    print(f"{'='*70}", flush=True)

    mutated = df[df['n_mutations'] > 0].copy()

    for n_ch in range(1, 7):
        ch_df = mutated[mutated['n_channels_hit'] == n_ch]
        if len(ch_df) < 50:
            continue
        med_os = ch_df['OS_MONTHS'].median()
        evt = ch_df['event'].mean()
        mean_muts = ch_df['n_mutations'].mean()
        print(f"  {n_ch} channels: n={len(ch_df):5d}, medOS={med_os:5.1f}m, "
              f"event={evt:.1%}, mean_muts={mean_muts:.1f}", flush=True)

    # Cox for n_channels_hit (adjusted for n_mutations + age)
    test_df = mutated[['OS_MONTHS', 'event', 'age_z', 'n_channels_hit',
                        'n_mutations']].copy()
    test_df['n_mutations_z'] = (test_df['n_mutations'] - test_df['n_mutations'].mean()) / (
        test_df['n_mutations'].std() + 1e-8)
    test_df = test_df.drop(columns='n_mutations')

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')
        print(f"\nCox PH (n_channels_hit, adjusted for n_mutations + age):")
        for var in ['n_channels_hit', 'n_mutations_z', 'age_z']:
            hr = np.exp(cph.params_[var])
            p = cph.summary.loc[var, 'p']
            ci = cph.confidence_intervals_.loc[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:25s}: HR={hr:.3f} [{np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f}] "
                  f"p={p:.2e} {sig}", flush=True)
    except Exception as e:
        print(f"  Cox failed: {e}", flush=True)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "patient_scripting_scores.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
