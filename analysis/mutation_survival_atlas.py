"""
Mutation Survival Atlas — per-cancer, per-mutation hazard ratios.

For each (cancer_type, mutation) with sufficient patients:
  - Cox PH hazard ratio vs wild-type (adjusted for age)
  - 95% CI
  - Direction: protective / neutral / harmful
  - If key interaction pairs co-occur: interaction-adjusted HR

For each (cancer_type, mutation_pair) in the 0/3 taxonomy:
  - Multiplicative excess or counteractive reduction

Output: lookup table that covers as much of the patient population
as possible with confident predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
)

# ---------------------------------------------------------------------------
MIN_PATIENTS_WITH_MUT = 20      # per cancer type
MIN_PATIENTS_CANCER_TYPE = 200  # minimum patients in a cancer type
MIN_EVENTS = 10                 # minimum deaths for Cox to converge
CI_THRESHOLD = 2.0              # max CI width for "confident" prediction
# ---------------------------------------------------------------------------


def get_channel_and_position(gene):
    channel = CHANNEL_MAP.get(gene)
    if channel is None:
        return None, None
    for ch, hubs in HUB_GENES.items():
        if gene in hubs:
            return channel, "hub"
    return channel, "leaf"


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

    # Get (patient, mutation) pairs
    patient_muts = mutations.groupby('patientId').apply(
        lambda df: set(zip(df['gene.hugoGeneSymbol'], df['proteinChange']))
    ).to_dict()

    # Cancer types with enough patients
    ct_counts = clinical['CANCER_TYPE'].value_counts()
    valid_cts = ct_counts[ct_counts >= MIN_PATIENTS_CANCER_TYPE].index.tolist()
    print(f"\nCancer types with ≥{MIN_PATIENTS_CANCER_TYPE} patients: {len(valid_cts)}", flush=True)

    # Find recurrent mutations per cancer type
    print("Finding recurrent mutations per cancer type...", flush=True)
    ct_patients = clinical.groupby('CANCER_TYPE')['patientId'].apply(set).to_dict()

    # Count mutations per cancer type
    mut_ct_counts = {}
    for ct in valid_cts:
        ct_pids = ct_patients.get(ct, set())
        ct_muts = mutations[mutations['patientId'].isin(ct_pids)]
        counts = (ct_muts
                  .groupby(['gene.hugoGeneSymbol', 'proteinChange'])
                  ['patientId'].nunique()
                  .reset_index())
        counts.columns = ['gene', 'pc', 'n']
        counts = counts[counts['n'] >= MIN_PATIENTS_WITH_MUT]
        for _, row in counts.iterrows():
            mut_ct_counts[(ct, row['gene'], row['pc'])] = row['n']

    print(f"Total (cancer_type, mutation) pairs to test: {len(mut_ct_counts)}", flush=True)

    # -----------------------------------------------------------------------
    # Per-mutation, per-cancer Cox PH
    # -----------------------------------------------------------------------
    print("\nRunning per-mutation Cox models...", flush=True)

    results = []
    n_tested = 0
    n_failed = 0

    for (ct, gene, pc), n_with in sorted(mut_ct_counts.items(),
                                           key=lambda x: -x[1]):
        # Get patients in this cancer type
        ct_clinical = clinical[clinical['CANCER_TYPE'] == ct].copy()
        ct_pids = set(ct_clinical['patientId'])

        # Binary: does this patient have this mutation?
        ct_clinical['has_mut'] = ct_clinical['patientId'].apply(
            lambda pid: 1 if (gene, pc) in patient_muts.get(pid, set()) else 0
        )

        n_mut = ct_clinical['has_mut'].sum()
        n_wt = len(ct_clinical) - n_mut
        n_events_mut = ct_clinical[ct_clinical['has_mut'] == 1]['event'].sum()
        n_events_wt = ct_clinical[ct_clinical['has_mut'] == 0]['event'].sum()

        if n_events_mut < 3 or n_events_wt < MIN_EVENTS:
            continue

        test_df = ct_clinical[['OS_MONTHS', 'event', 'age_z', 'has_mut']].copy()
        test_df = test_df[test_df['OS_MONTHS'] > 0]

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')

            hr = np.exp(cph.params_['has_mut'])
            ci_low = np.exp(cph.confidence_intervals_.loc['has_mut'].iloc[0])
            ci_high = np.exp(cph.confidence_intervals_.loc['has_mut'].iloc[1])
            p_val = cph.summary.loc['has_mut', 'p']

            channel, position = get_channel_and_position(gene)
            ci_width = ci_high - ci_low

            # Classify
            if p_val < 0.05:
                if hr > 1.1:
                    direction = "harmful"
                elif hr < 0.9:
                    direction = "protective"
                else:
                    direction = "neutral"
            else:
                direction = "neutral"

            confident = ci_width < CI_THRESHOLD and p_val < 0.05

            results.append({
                'cancer_type': ct,
                'gene': gene,
                'protein_change': pc,
                'channel': channel,
                'position': position,
                'n_patients_ct': len(ct_clinical),
                'n_mutated': n_mut,
                'n_wildtype': n_wt,
                'n_events_mut': n_events_mut,
                'n_events_wt': n_events_wt,
                'hr': hr,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_width': ci_width,
                'p_value': p_val,
                'direction': direction,
                'confident': confident,
                'mutation_label': f"{gene} {pc}",
            })
            n_tested += 1

        except Exception:
            n_failed += 1
            continue

        if n_tested % 50 == 0:
            print(f"  {n_tested} tested, {len([r for r in results if r['confident']])} confident...",
                  flush=True)

    print(f"\nTested: {n_tested}, Failed: {n_failed}", flush=True)

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    confident = results_df[results_df['confident']]
    harmful = confident[confident['direction'] == 'harmful']
    protective = confident[confident['direction'] == 'protective']

    print(f"\n{'='*70}")
    print(f"MUTATION SURVIVAL ATLAS")
    print(f"{'='*70}")
    print(f"Total tested: {len(results_df)}")
    print(f"Confident predictions (p<0.05, CI width<{CI_THRESHOLD}): {len(confident)}")
    print(f"  Harmful: {len(harmful)}")
    print(f"  Protective: {len(protective)}")

    # Coverage: what fraction of patients have ≥1 confident prediction?
    confident_mutations = set(zip(confident['gene'], confident['protein_change'],
                                   confident['cancer_type']))
    patients_covered = 0
    patients_total = len(clinical)
    for _, row in clinical.iterrows():
        pid = row['patientId']
        ct = row['CANCER_TYPE']
        pmuts = patient_muts.get(pid, set())
        for (g, pc) in pmuts:
            if (g, pc, ct) in confident_mutations:
                patients_covered += 1
                break

    print(f"\nPopulation coverage: {patients_covered}/{patients_total} "
          f"({patients_covered/patients_total*100:.1f}%)")

    # Top harmful mutations (cross cancer types)
    print(f"\n--- TOP HARMFUL (by HR, confident) ---")
    for _, r in harmful.sort_values('hr', ascending=False).head(25).iterrows():
        print(f"  {r['cancer_type']:35s} {r['mutation_label']:20s} "
              f"({r['channel']}/{r['position']}) "
              f"HR={r['hr']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
              f"p={r['p_value']:.4f} n={r['n_mutated']}", flush=True)

    print(f"\n--- TOP PROTECTIVE (by HR, confident) ---")
    for _, r in protective.sort_values('hr').head(25).iterrows():
        print(f"  {r['cancer_type']:35s} {r['mutation_label']:20s} "
              f"({r['channel']}/{r['position']}) "
              f"HR={r['hr']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
              f"p={r['p_value']:.4f} n={r['n_mutated']}", flush=True)

    # Per-channel summary
    print(f"\n--- BY CHANNEL ---")
    for ch in CHANNEL_NAMES:
        ch_conf = confident[confident['channel'] == ch]
        if len(ch_conf) == 0:
            continue
        n_h = len(ch_conf[ch_conf['direction'] == 'harmful'])
        n_p = len(ch_conf[ch_conf['direction'] == 'protective'])
        mean_hr = ch_conf['hr'].mean()
        print(f"  {ch:15s}: {len(ch_conf)} confident ({n_h} harmful, {n_p} protective), "
              f"mean HR={mean_hr:.2f}")

    # Hub vs leaf
    print(f"\n--- HUB vs LEAF ---")
    for pos in ['hub', 'leaf']:
        pos_conf = confident[confident['position'] == pos]
        if len(pos_conf) == 0:
            continue
        n_h = len(pos_conf[pos_conf['direction'] == 'harmful'])
        n_p = len(pos_conf[pos_conf['direction'] == 'protective'])
        print(f"  {pos}: {len(pos_conf)} confident ({n_h} harmful, {n_p} protective)")

    # Cross-channel interaction scan (per cancer type)
    # For the significant interaction pairs, check per-cancer-type
    print(f"\n{'='*70}")
    print("CROSS-CHANNEL INTERACTIONS BY CANCER TYPE")
    print(f"{'='*70}")

    INTERACTION_PAIRS = [
        ("JAK1", "K860Nfs*16", "ARID1A", "D1850Tfs*33", "multiplicative"),
        ("JAK1", "K860Nfs*16", "PTEN", "K267Rfs*9", "multiplicative"),
        ("BRAF", "V600E", "APC", "T1556Nfs*3", "multiplicative"),
        ("KRAS", "G12V", "TP53", "Y220C", "multiplicative"),
        ("KRAS", "G12D", "JAK1", "K860Nfs*16", "counteractive"),
        ("KRAS", "G13D", "TP53", "R282W", "counteractive"),
        ("KRAS", "G12D", "PIK3CA", "E545K", "counteractive"),
        ("KRAS", "G12D", "PIK3CA", "H1047R", "counteractive"),
    ]

    for ga, pca, gb, pcb, itype in INTERACTION_PAIRS:
        # Find cancer types where both mutations appear
        for ct in valid_cts:
            ct_clinical = clinical[clinical['CANCER_TYPE'] == ct].copy()

            ct_clinical['has_a'] = ct_clinical['patientId'].apply(
                lambda pid: 1 if (ga, pca) in patient_muts.get(pid, set()) else 0
            )
            ct_clinical['has_b'] = ct_clinical['patientId'].apply(
                lambda pid: 1 if (gb, pcb) in patient_muts.get(pid, set()) else 0
            )

            n_both = ((ct_clinical['has_a'] == 1) & (ct_clinical['has_b'] == 1)).sum()
            if n_both < 5:
                continue

            ct_clinical['interact'] = ct_clinical['has_a'] * ct_clinical['has_b']
            test_df = ct_clinical[['OS_MONTHS', 'event', 'age_z',
                                    'has_a', 'has_b', 'interact']].copy()
            test_df = test_df[test_df['OS_MONTHS'] > 0]

            try:
                cph = CoxPHFitter(penalizer=0.05)
                cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')
                hr_int = np.exp(cph.params_['interact'])
                p_int = cph.summary.loc['interact', 'p']

                if p_int < 0.10:
                    print(f"  {ct:30s} {ga} {pca} × {gb} {pcb}: "
                          f"n_both={n_both}, HR_int={hr_int:.2f}, p={p_int:.3f} "
                          f"[{itype}]", flush=True)
            except Exception:
                continue

    # Save
    out_dir = os.path.dirname(__file__)
    results_df.to_csv(os.path.join(out_dir, "mutation_survival_atlas.csv"), index=False)
    confident.to_csv(os.path.join(out_dir, "mutation_survival_atlas_confident.csv"), index=False)
    print(f"\nSaved atlas to {out_dir}/mutation_survival_atlas*.csv", flush=True)


if __name__ == "__main__":
    main()
