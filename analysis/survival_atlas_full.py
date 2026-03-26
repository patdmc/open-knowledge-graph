"""
Full Survival Atlas — hierarchical lookup from biopsy to prediction.

Three tiers of prediction, each per cancer type:
  1. Mutation-specific: HR for (cancer_type, gene, proteinChange)
  2. Gene-level: HR for (cancer_type, gene, any non-silent mutation)
  3. Channel-level: HR for (cancer_type, channel severed)

For a patient, use the most specific tier available.
Also: interaction adjustments for known 0/3 pairs.

Output: complete lookup table + population coverage at each tier.
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
MIN_MUT_PATIENTS = 20
MIN_GENE_PATIENTS = 30
MIN_CHANNEL_PATIENTS = 50
MIN_CANCER_TYPE = 200
MIN_EVENTS = 10
CI_MAX = 2.5
# ---------------------------------------------------------------------------


def get_channel_and_position(gene):
    channel = CHANNEL_MAP.get(gene)
    if channel is None:
        return None, None
    for ch, hubs in HUB_GENES.items():
        if gene in hubs:
            return channel, "hub"
    return channel, "leaf"


def fit_cox_binary(df, covariate_col, min_events=MIN_EVENTS, penalizer=0.01):
    """Fit Cox PH for a single binary covariate + age. Returns dict or None."""
    n_with = df[covariate_col].sum()
    n_without = len(df) - n_with
    events_with = df[df[covariate_col] == 1]['event'].sum()
    events_without = df[df[covariate_col] == 0]['event'].sum()

    if events_with < 3 or events_without < min_events or n_with < 5:
        return None

    test_df = df[['OS_MONTHS', 'event', 'age_z', covariate_col]].copy()
    test_df = test_df[test_df['OS_MONTHS'] > 0]

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')

        hr = np.exp(cph.params_[covariate_col])
        ci = cph.confidence_intervals_.loc[covariate_col]
        ci_low = np.exp(ci.iloc[0])
        ci_high = np.exp(ci.iloc[1])
        p_val = cph.summary.loc[covariate_col, 'p']

        return {
            'hr': hr,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_high - ci_low,
            'p_value': p_val,
            'n_with': int(n_with),
            'n_without': int(n_without),
            'events_with': int(events_with),
            'events_without': int(events_without),
        }
    except Exception:
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

    # Build patient lookup structures
    print("Building patient lookups...", flush=True)
    patient_muts = {}  # pid -> set of (gene, pc)
    patient_genes = {}  # pid -> set of genes
    patient_channels = {}  # pid -> set of channels

    for pid, group in mutations.groupby('patientId'):
        muts = set(zip(group['gene.hugoGeneSymbol'], group['proteinChange']))
        genes = set(group['gene.hugoGeneSymbol'])
        channels = set(CHANNEL_MAP.get(g) for g in genes if g in CHANNEL_MAP)
        channels.discard(None)
        patient_muts[pid] = muts
        patient_genes[pid] = genes
        patient_channels[pid] = channels

    valid_cts = clinical['CANCER_TYPE'].value_counts()
    valid_cts = valid_cts[valid_cts >= MIN_CANCER_TYPE].index.tolist()
    print(f"Cancer types: {len(valid_cts)}", flush=True)
    print(f"Total patients: {len(clinical)}", flush=True)

    # ===================================================================
    # TIER 1: Mutation-specific
    # ===================================================================
    print(f"\n{'='*70}")
    print("TIER 1: MUTATION-SPECIFIC")
    print(f"{'='*70}", flush=True)

    tier1_results = []
    n_tested = 0

    for ct in valid_cts:
        ct_clin = clinical[clinical['CANCER_TYPE'] == ct].copy()
        ct_pids = set(ct_clin['patientId'])
        ct_muts_df = mutations[mutations['patientId'].isin(ct_pids)]

        # Count mutations
        counts = (ct_muts_df
                  .groupby(['gene.hugoGeneSymbol', 'proteinChange'])
                  ['patientId'].nunique()
                  .reset_index())
        counts.columns = ['gene', 'pc', 'n']
        counts = counts[counts['n'] >= MIN_MUT_PATIENTS]

        for _, row in counts.iterrows():
            gene, pc = row['gene'], row['pc']
            ct_clin['has_mut'] = ct_clin['patientId'].apply(
                lambda pid: 1 if (gene, pc) in patient_muts.get(pid, set()) else 0
            )

            result = fit_cox_binary(ct_clin, 'has_mut')
            if result is None:
                continue

            ch, pos = get_channel_and_position(gene)
            confident = result['p_value'] < 0.05 and result['ci_width'] < CI_MAX

            tier1_results.append({
                'tier': 1,
                'cancer_type': ct,
                'gene': gene,
                'protein_change': pc,
                'channel': ch,
                'position': pos,
                'label': f"{gene} {pc}",
                'confident': confident,
                **result,
            })
            n_tested += 1

        if n_tested % 100 == 0:
            print(f"  Tier 1: {n_tested} tested...", flush=True)

    tier1_df = pd.DataFrame(tier1_results)
    tier1_conf = tier1_df[tier1_df['confident']] if len(tier1_df) > 0 else pd.DataFrame()
    print(f"Tier 1: {len(tier1_df)} tested, {len(tier1_conf)} confident", flush=True)

    # ===================================================================
    # TIER 2: Gene-level
    # ===================================================================
    print(f"\n{'='*70}")
    print("TIER 2: GENE-LEVEL")
    print(f"{'='*70}", flush=True)

    tier2_results = []

    all_genes = sorted(CHANNEL_MAP.keys())

    for ct in valid_cts:
        ct_clin = clinical[clinical['CANCER_TYPE'] == ct].copy()

        for gene in all_genes:
            ct_clin['has_gene'] = ct_clin['patientId'].apply(
                lambda pid, g=gene: 1 if g in patient_genes.get(pid, set()) else 0
            )

            n_with = ct_clin['has_gene'].sum()
            if n_with < MIN_GENE_PATIENTS:
                continue

            result = fit_cox_binary(ct_clin, 'has_gene')
            if result is None:
                continue

            ch, pos = get_channel_and_position(gene)
            confident = result['p_value'] < 0.05 and result['ci_width'] < CI_MAX

            tier2_results.append({
                'tier': 2,
                'cancer_type': ct,
                'gene': gene,
                'protein_change': '*',
                'channel': ch,
                'position': pos,
                'label': f"{gene} (any)",
                'confident': confident,
                **result,
            })

    tier2_df = pd.DataFrame(tier2_results)
    tier2_conf = tier2_df[tier2_df['confident']] if len(tier2_df) > 0 else pd.DataFrame()
    print(f"Tier 2: {len(tier2_df)} tested, {len(tier2_conf)} confident", flush=True)

    # ===================================================================
    # TIER 3: Channel-level
    # ===================================================================
    print(f"\n{'='*70}")
    print("TIER 3: CHANNEL-LEVEL")
    print(f"{'='*70}", flush=True)

    tier3_results = []

    for ct in valid_cts:
        ct_clin = clinical[clinical['CANCER_TYPE'] == ct].copy()

        for channel in CHANNEL_NAMES:
            ct_clin['has_channel'] = ct_clin['patientId'].apply(
                lambda pid, ch=channel: 1 if ch in patient_channels.get(pid, set()) else 0
            )

            n_with = ct_clin['has_channel'].sum()
            if n_with < MIN_CHANNEL_PATIENTS:
                continue

            result = fit_cox_binary(ct_clin, 'has_channel')
            if result is None:
                continue

            confident = result['p_value'] < 0.05 and result['ci_width'] < CI_MAX

            tier3_results.append({
                'tier': 3,
                'cancer_type': ct,
                'gene': '*',
                'protein_change': '*',
                'channel': channel,
                'position': '*',
                'label': f"{channel} (any gene)",
                'confident': confident,
                **result,
            })

    tier3_df = pd.DataFrame(tier3_results)
    tier3_conf = tier3_df[tier3_df['confident']] if len(tier3_df) > 0 else pd.DataFrame()
    print(f"Tier 3: {len(tier3_df)} tested, {len(tier3_conf)} confident", flush=True)

    # ===================================================================
    # COVERAGE ANALYSIS
    # ===================================================================
    print(f"\n{'='*70}")
    print("POPULATION COVERAGE")
    print(f"{'='*70}", flush=True)

    # Build confident lookup sets
    tier1_keys = set()
    if len(tier1_conf) > 0:
        for _, r in tier1_conf.iterrows():
            tier1_keys.add((r['cancer_type'], r['gene'], r['protein_change']))

    tier2_keys = set()
    if len(tier2_conf) > 0:
        for _, r in tier2_conf.iterrows():
            tier2_keys.add((r['cancer_type'], r['gene']))

    tier3_keys = set()
    if len(tier3_conf) > 0:
        for _, r in tier3_conf.iterrows():
            tier3_keys.add((r['cancer_type'], r['channel']))

    covered_t1 = 0
    covered_t2 = 0
    covered_t3 = 0
    covered_any = 0
    total = len(clinical)

    for _, row in clinical.iterrows():
        pid = row['patientId']
        ct = row['CANCER_TYPE']
        pmuts = patient_muts.get(pid, set())
        pgenes = patient_genes.get(pid, set())
        pchannels = patient_channels.get(pid, set())

        hit_t1 = any((ct, g, pc) in tier1_keys for g, pc in pmuts)
        hit_t2 = any((ct, g) in tier2_keys for g in pgenes)
        hit_t3 = any((ct, ch) in tier3_keys for ch in pchannels)

        if hit_t1:
            covered_t1 += 1
        if hit_t2:
            covered_t2 += 1
        if hit_t3:
            covered_t3 += 1
        if hit_t1 or hit_t2 or hit_t3:
            covered_any += 1

    print(f"Tier 1 (mutation-specific): {covered_t1}/{total} ({covered_t1/total*100:.1f}%)")
    print(f"Tier 2 (gene-level):        {covered_t2}/{total} ({covered_t2/total*100:.1f}%)")
    print(f"Tier 3 (channel-level):     {covered_t3}/{total} ({covered_t3/total*100:.1f}%)")
    print(f"Any tier:                   {covered_any}/{total} ({covered_any/total*100:.1f}%)")

    # Per cancer type coverage
    print(f"\n--- PER CANCER TYPE COVERAGE ---")
    for ct in sorted(valid_cts):
        ct_clin = clinical[clinical['CANCER_TYPE'] == ct]
        ct_total = len(ct_clin)
        ct_covered = 0
        for _, row in ct_clin.iterrows():
            pid = row['patientId']
            pmuts = patient_muts.get(pid, set())
            pgenes = patient_genes.get(pid, set())
            pchannels = patient_channels.get(pid, set())
            hit = (any((ct, g, pc) in tier1_keys for g, pc in pmuts) or
                   any((ct, g) in tier2_keys for g in pgenes) or
                   any((ct, ch) in tier3_keys for ch in pchannels))
            if hit:
                ct_covered += 1
        pct = ct_covered / ct_total * 100 if ct_total > 0 else 0
        print(f"  {ct:40s} {ct_covered:5d}/{ct_total:5d} ({pct:5.1f}%)", flush=True)

    # ===================================================================
    # SUMMARY TABLES
    # ===================================================================
    print(f"\n{'='*70}")
    print("TIER 1 — TOP MUTATIONS BY CANCER TYPE")
    print(f"{'='*70}")

    if len(tier1_conf) > 0:
        # Most harmful per cancer type
        for ct in sorted(valid_cts):
            ct_rows = tier1_conf[tier1_conf['cancer_type'] == ct].sort_values('hr', ascending=False)
            if len(ct_rows) == 0:
                continue
            print(f"\n  {ct}:")
            harmful = ct_rows[ct_rows['hr'] > 1.1].head(5)
            protective = ct_rows[ct_rows['hr'] < 0.9].sort_values('hr').head(5)
            for _, r in harmful.iterrows():
                print(f"    HARMFUL  {r['label']:25s} ({r['channel']}/{r['position']}) "
                      f"HR={r['hr']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
                      f"n={r['n_with']}")
            for _, r in protective.iterrows():
                print(f"    PROTECT  {r['label']:25s} ({r['channel']}/{r['position']}) "
                      f"HR={r['hr']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
                      f"n={r['n_with']}")

    print(f"\n{'='*70}")
    print("TIER 2 — GENE-LEVEL BY CANCER TYPE")
    print(f"{'='*70}")

    if len(tier2_conf) > 0:
        for ct in sorted(valid_cts):
            ct_rows = tier2_conf[tier2_conf['cancer_type'] == ct].sort_values('hr', ascending=False)
            if len(ct_rows) == 0:
                continue
            print(f"\n  {ct}:")
            for _, r in ct_rows.iterrows():
                direction = "HARMFUL" if r['hr'] > 1.1 else "PROTECT" if r['hr'] < 0.9 else "NEUTRAL"
                print(f"    {direction:8s} {r['gene']:15s} ({r['channel']}/{r['position']}) "
                      f"HR={r['hr']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
                      f"n={r['n_with']}")

    print(f"\n{'='*70}")
    print("TIER 3 — CHANNEL-LEVEL BY CANCER TYPE")
    print(f"{'='*70}")

    if len(tier3_conf) > 0:
        for ct in sorted(valid_cts):
            ct_rows = tier3_conf[tier3_conf['cancer_type'] == ct].sort_values('hr', ascending=False)
            if len(ct_rows) == 0:
                continue
            print(f"\n  {ct}:")
            for _, r in ct_rows.iterrows():
                direction = "HARMFUL" if r['hr'] > 1.1 else "PROTECT" if r['hr'] < 0.9 else "NEUTRAL"
                print(f"    {direction:8s} {r['channel']:15s} "
                      f"HR={r['hr']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
                      f"n={r['n_with']}")

    # ===================================================================
    # SAVE
    # ===================================================================
    out_dir = os.path.dirname(__file__)

    all_tiers = pd.concat([tier1_df, tier2_df, tier3_df], ignore_index=True)
    all_tiers.to_csv(os.path.join(out_dir, "survival_atlas_full.csv"), index=False)

    all_confident = pd.concat([tier1_conf, tier2_conf, tier3_conf], ignore_index=True)
    all_confident.to_csv(os.path.join(out_dir, "survival_atlas_confident.csv"), index=False)

    print(f"\nTotal atlas entries: {len(all_tiers)}")
    print(f"Confident entries: {len(all_confident)}")
    print(f"Saved to {out_dir}/survival_atlas_*.csv", flush=True)


if __name__ == "__main__":
    main()
