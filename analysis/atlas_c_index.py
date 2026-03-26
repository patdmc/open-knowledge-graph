"""
C-index from pure atlas graph traversal — no neural network.

For each patient:
  1. Look up each mutation in the atlas (tier 1 > tier 2 > tier 3)
  2. Sum log(HR) values = risk score
  3. Compute concordance index

Compare: atlas lookup vs ChannelNetV5 (0.677) vs Cox-SAGE (0.627-0.716)
"""

import os
import sys
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as c_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_MAP, NON_SILENT, MSK_DATASETS,
)


def main():
    paths = MSK_DATASETS["msk_impact_50k"]
    print("Loading...", flush=True)
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

    # Load full atlas
    atlas_path = os.path.join(os.path.dirname(__file__), "survival_atlas_full.csv")
    atlas = pd.read_csv(atlas_path)

    # Build lookups (use all entries with p < 0.10)
    t1 = {}
    t2 = {}
    t3 = {}
    for _, r in atlas.iterrows():
        if r['p_value'] > 0.10:
            continue
        if r['tier'] == 1:
            t1[(r['cancer_type'], r['gene'], r['protein_change'])] = np.log(r['hr'])
        elif r['tier'] == 2:
            t2[(r['cancer_type'], r['gene'])] = np.log(r['hr'])
        elif r['tier'] == 3:
            t3[(r['cancer_type'], r['channel'])] = np.log(r['hr'])

    print(f"Atlas: T1={len(t1)}, T2={len(t2)}, T3={len(t3)}", flush=True)

    # Group mutations by patient
    patient_muts = {}
    for pid, group in mutations.groupby('patientId'):
        patient_muts[pid] = list(zip(
            group['gene.hugoGeneSymbol'],
            group['proteinChange'],
        ))

    # Score each patient
    scores = []
    scores_t1only = []
    scores_t12 = []
    times = []
    events = []
    cancer_types = []

    for _, row in clinical.iterrows():
        pid = row['patientId']
        ct = row['CANCER_TYPE']
        pmuts = patient_muts.get(pid, [])

        score = 0.0
        score_t1 = 0.0
        score_t12 = 0.0

        for gene, pc in pmuts:
            ch = CHANNEL_MAP.get(gene)

            # Tier 1: mutation-specific
            if (ct, gene, pc) in t1:
                lhr = t1[(ct, gene, pc)]
                score += lhr
                score_t1 += lhr
                score_t12 += lhr
            # Tier 2: gene-level
            elif (ct, gene) in t2:
                lhr = t2[(ct, gene)]
                score += lhr
                score_t12 += lhr
            # Tier 3: channel-level
            elif ch and (ct, ch) in t3:
                lhr = t3[(ct, ch)]
                score += lhr

        scores.append(score)
        scores_t1only.append(score_t1)
        scores_t12.append(score_t12)
        times.append(row['OS_MONTHS'])
        events.append(row['event'])
        cancer_types.append(ct)

    scores = np.array(scores)
    scores_t1only = np.array(scores_t1only)
    scores_t12 = np.array(scores_t12)
    times = np.array(times)
    events = np.array(events)

    # Filter valid
    valid = times > 0
    scores = scores[valid]
    scores_t1only = scores_t1only[valid]
    scores_t12 = scores_t12[valid]
    times = times[valid]
    events = events[valid]
    cancer_types = np.array(cancer_types)[valid]

    # ===================================================================
    # PAN-CANCER C-INDEX
    # ===================================================================
    print(f"\n{'='*70}")
    print("PAN-CANCER C-INDEX (atlas graph traversal)")
    print(f"{'='*70}")

    c_all = c_index(times, -scores, events)  # negative because higher score = higher risk
    c_t1 = c_index(times, -scores_t1only, events)
    c_t12 = c_index(times, -scores_t12, events)

    print(f"  All tiers (T1+T2+T3):  C = {c_all:.4f}")
    print(f"  T1+T2 only:            C = {c_t12:.4f}")
    print(f"  T1 only (mutation):    C = {c_t1:.4f}")
    print(f"")
    print(f"  ChannelNetV5 (GNN):    C = 0.6770")
    print(f"  Cox-SAGE (published):  C = 0.627-0.716")
    print(f"  Random:                C = 0.500")

    # ===================================================================
    # PER-CANCER-TYPE C-INDEX
    # ===================================================================
    print(f"\n{'='*70}")
    print("PER-CANCER-TYPE C-INDEX")
    print(f"{'='*70}")

    ct_counts = pd.Series(cancer_types).value_counts()
    valid_cts = ct_counts[ct_counts >= 200].index

    ct_results = []
    for ct in sorted(valid_cts):
        mask = cancer_types == ct
        ct_scores = scores[mask]
        ct_times = times[mask]
        ct_events = events[mask]

        if ct_events.sum() < 10 or ct_scores.std() < 1e-8:
            continue

        try:
            c = c_index(ct_times, -ct_scores, ct_events)
            ct_results.append({'cancer_type': ct, 'c_index': c, 'n': mask.sum(),
                               'n_events': int(ct_events.sum())})
            print(f"  {ct:40s} n={mask.sum():5d} events={int(ct_events.sum()):4d} C={c:.4f}",
                  flush=True)
        except Exception:
            continue

    ct_df = pd.DataFrame(ct_results)
    weighted_c = np.average(ct_df['c_index'], weights=ct_df['n'])
    print(f"\n  Weighted mean C-index: {weighted_c:.4f}")
    print(f"  Unweighted mean:      {ct_df['c_index'].mean():.4f}")

    # ===================================================================
    # BOOTSTRAP CI
    # ===================================================================
    print(f"\n{'='*70}")
    print("BOOTSTRAP 95% CI (1000 iterations)")
    print(f"{'='*70}", flush=True)

    np.random.seed(42)
    boot_cs = []
    n = len(scores)
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        try:
            bc = c_index(times[idx], -scores[idx], events[idx])
            boot_cs.append(bc)
        except Exception:
            continue

    boot_cs = np.array(boot_cs)
    ci_low = np.percentile(boot_cs, 2.5)
    ci_high = np.percentile(boot_cs, 97.5)
    print(f"  Pan-cancer C = {c_all:.4f} [{ci_low:.4f} - {ci_high:.4f}]")

    # ===================================================================
    # COMBINED: atlas score + age
    # ===================================================================
    print(f"\n{'='*70}")
    print("ATLAS SCORE + AGE (simple linear combination)")
    print(f"{'='*70}", flush=True)

    ages = pd.to_numeric(clinical['AGE_AT_DX'], errors='coerce').fillna(60).values
    ages = ages[clinical['OS_MONTHS'].notna() & clinical['OS_STATUS'].notna()]
    # Re-filter to match
    clin_valid = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    clin_valid = clin_valid[clin_valid['OS_MONTHS'] > 0]
    ages_valid = pd.to_numeric(clin_valid['AGE_AT_DX'], errors='coerce').fillna(60).values
    ages_z = (ages_valid - ages_valid.mean()) / (ages_valid.std() + 1e-8)

    if len(ages_z) == len(scores):
        # Simple combination: score + 0.5*age (age is a known predictor)
        combined = scores + 0.5 * ages_z
        c_combined = c_index(times, -combined, events)
        print(f"  Atlas alone:           C = {c_all:.4f}")
        print(f"  Atlas + age:           C = {c_combined:.4f}")
    else:
        print(f"  Length mismatch: ages={len(ages_z)}, scores={len(scores)}")

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
