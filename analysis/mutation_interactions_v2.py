"""
Mutation-pair interaction scan with 4-level interaction taxonomy.

Interaction types (by hazard ratio of interaction term):
  0. Counteractive  — HR_int < 0.7: mutations cancel each other out
  1. Redundant      — 0.7 ≤ HR_int < 0.95: overlapping damage, less than sum
  2. Additive       — 0.95 ≤ HR_int ≤ 1.05: independent, sum of parts
  3. Multiplicative  — HR_int > 1.05: compounding, worse than sum

For each pair: Cox PH with interaction term, classify by HR_interaction.
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
)

# ---------------------------------------------------------------------------
MIN_PATIENTS_WITH_MUTATION = 100
MIN_PATIENTS_WITH_BOTH = 20
# ---------------------------------------------------------------------------


def get_channel_and_position(gene):
    channel = CHANNEL_MAP.get(gene)
    if channel is None:
        return None, None
    for ch, hubs in HUB_GENES.items():
        if gene in hubs:
            return channel, "hub"
    return channel, "leaf"


def classify_interaction(hr_int, p_val):
    """Classify interaction by HR and significance."""
    if p_val > 0.10:
        return "2_additive"  # not significant = consistent with additive
    if hr_int < 0.70:
        return "0_counteractive"
    elif hr_int < 0.95:
        return "1_redundant"
    elif hr_int <= 1.05:
        return "2_additive"
    else:
        return "3_multiplicative"


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

    # Find recurrent mutations
    mut_counts = (mutations
                  .groupby(['gene.hugoGeneSymbol', 'proteinChange'])
                  ['patientId'].nunique()
                  .reset_index())
    mut_counts.columns = ['gene', 'pc', 'n_patients']
    recurrent = mut_counts[mut_counts['n_patients'] >= MIN_PATIENTS_WITH_MUTATION].copy()
    recurrent = recurrent.sort_values('n_patients', ascending=False)

    # Add channel info
    recurrent['channel'] = recurrent['gene'].map(lambda g: get_channel_and_position(g)[0])
    recurrent['position'] = recurrent['gene'].map(lambda g: get_channel_and_position(g)[1])
    recurrent['label'] = recurrent['gene'] + '_' + recurrent['pc']

    print(f"\nRecurrent mutations (≥{MIN_PATIENTS_WITH_MUTATION} patients): {len(recurrent)}", flush=True)
    for ch in CHANNEL_NAMES:
        subset = recurrent[recurrent['channel'] == ch]
        if len(subset) > 0:
            print(f"  {ch}: {len(subset)} mutations", flush=True)
            for _, r in subset.head(5).iterrows():
                print(f"    {r['label']} (n={r['n_patients']}, {r['position']})", flush=True)

    # Build patient × mutation matrix efficiently
    print("\nBuilding mutation matrix...", flush=True)
    patient_ids = clinical['patientId'].values
    labels = recurrent['label'].tolist()
    label_to_gene_pc = dict(zip(recurrent['label'], zip(recurrent['gene'], recurrent['pc'])))

    # Create lookup: (gene, pc) -> label
    gpc_to_label = {(g, pc): lab for lab, (g, pc) in label_to_gene_pc.items()}

    # Build matrix via groupby
    mut_matrix = pd.DataFrame(0, index=patient_ids, columns=labels, dtype=np.int8)

    recurrent_keys = set(zip(recurrent['gene'], recurrent['pc']))
    for _, row in mutations.iterrows():
        gene = row['gene.hugoGeneSymbol']
        pc = row.get('proteinChange', '')
        pid = row['patientId']
        key = (gene, pc)
        if key in gpc_to_label and pid in mut_matrix.index:
            mut_matrix.at[pid, gpc_to_label[key]] = 1

    print(f"Matrix shape: {mut_matrix.shape}", flush=True)

    # Merge with clinical
    df = clinical.set_index('patientId')[['OS_MONTHS', 'event', 'age_z']].copy()
    df = df.join(mut_matrix, how='inner')
    df = df[df['OS_MONTHS'] > 0]
    print(f"Patients in analysis: {len(df)}", flush=True)

    # Channel lookup
    label_to_channel = dict(zip(recurrent['label'], recurrent['channel']))
    label_to_position = dict(zip(recurrent['label'], recurrent['position']))

    # Scan all pairs
    results = []
    n_tested = 0
    n_skipped = 0

    pairs = list(combinations(labels, 2))
    print(f"\nTotal candidate pairs: {len(pairs)}", flush=True)

    for i, (la, lb) in enumerate(pairs):
        both = ((df[la] == 1) & (df[lb] == 1)).sum()
        if both < MIN_PATIENTS_WITH_BOTH:
            n_skipped += 1
            continue

        interaction_col = "interact"
        test_df = df[['OS_MONTHS', 'event', 'age_z', la, lb]].copy()
        test_df[interaction_col] = test_df[la] * test_df[lb]

        # Skip if interaction column is constant
        if test_df[interaction_col].nunique() < 2:
            continue

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')

            coef_int = cph.params_[interaction_col]
            p_int = cph.summary.loc[interaction_col, 'p']
            hr_int = np.exp(coef_int)

            coef_a = cph.params_[la]
            coef_b = cph.params_[lb]

            interaction_type = classify_interaction(hr_int, p_int)

            ch_a = label_to_channel[la]
            ch_b = label_to_channel[lb]

            results.append({
                'mut_a': la,
                'mut_b': lb,
                'channel_a': ch_a,
                'channel_b': ch_b,
                'pos_a': label_to_position[la],
                'pos_b': label_to_position[lb],
                'cross_channel': ch_a != ch_b,
                'n_a': (df[la] == 1).sum(),
                'n_b': (df[lb] == 1).sum(),
                'n_both': both,
                'hr_a': np.exp(coef_a),
                'hr_b': np.exp(coef_b),
                'hr_interaction': hr_int,
                'coef_interaction': coef_int,
                'p_interaction': p_int,
                'interaction_type': interaction_type,
            })
            n_tested += 1

        except Exception:
            continue

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pairs)} pairs, {n_tested} tested, "
                  f"{sum(1 for r in results if r['p_interaction'] < 0.05)} significant...",
                  flush=True)

    print(f"\nTested: {n_tested}, Skipped (low co-occurrence): {n_skipped}", flush=True)

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("No interactions found.")
        return

    # Classify
    sig = results_df[results_df['p_interaction'] < 0.05].copy()
    print(f"\nSignificant (p<0.05): {len(sig)} / {len(results_df)}")

    # ---- Summary by type ----
    print(f"\n{'='*70}")
    print("INTERACTION TAXONOMY — ALL TESTED PAIRS")
    print(f"{'='*70}")

    for itype in ['0_counteractive', '1_redundant', '2_additive', '3_multiplicative']:
        subset = results_df[results_df['interaction_type'] == itype]
        cross = subset[subset['cross_channel']]
        within = subset[~subset['cross_channel']]
        print(f"\n{itype}: {len(subset)} total ({len(cross)} cross-channel, {len(within)} within-channel)")

    # ---- Significant non-additive interactions ----
    nonadd = sig[sig['interaction_type'] != '2_additive'].sort_values('p_interaction')

    print(f"\n{'='*70}")
    print(f"SIGNIFICANT NON-ADDITIVE INTERACTIONS (p<0.05): {len(nonadd)}")
    print(f"{'='*70}")

    for itype in ['0_counteractive', '1_redundant', '3_multiplicative']:
        subset = nonadd[nonadd['interaction_type'] == itype]
        if len(subset) == 0:
            continue

        label = itype.split('_')[1].upper()
        print(f"\n--- {label} ({len(subset)}) ---")

        for _, r in subset.iterrows():
            cross_tag = "CROSS" if r['cross_channel'] else "within"
            pos_a = "H" if r['pos_a'] == 'hub' else "L"
            pos_b = "H" if r['pos_b'] == 'hub' else "L"
            print(f"  [{cross_tag}] {r['mut_a']}({pos_a},{r['channel_a']}) × "
                  f"{r['mut_b']}({pos_b},{r['channel_b']})")
            print(f"    n_both={r['n_both']}, HR_a={r['hr_a']:.2f}, HR_b={r['hr_b']:.2f}, "
                  f"HR_int={r['hr_interaction']:.3f}, p={r['p_interaction']:.4f}")

    # ---- Cross-channel summary ----
    cross_sig = nonadd[nonadd['cross_channel']]
    if len(cross_sig) > 0:
        print(f"\n{'='*70}")
        print("CROSS-CHANNEL INTERACTION MATRIX")
        print(f"{'='*70}")

        # Build channel × channel interaction count matrix
        for _, r in cross_sig.iterrows():
            ch_pair = tuple(sorted([r['channel_a'], r['channel_b']]))
            print(f"  {ch_pair[0]} × {ch_pair[1]}: "
                  f"{r['mut_a']} × {r['mut_b']} → {r['interaction_type'].split('_')[1]} "
                  f"(HR={r['hr_interaction']:.3f})")

    # ---- Hub × Hub vs Hub × Leaf vs Leaf × Leaf ----
    if len(nonadd) > 0:
        print(f"\n{'='*70}")
        print("POSITION ANALYSIS")
        print(f"{'='*70}")

        for pos_pair in [('hub', 'hub'), ('hub', 'leaf'), ('leaf', 'leaf')]:
            mask = (
                ((nonadd['pos_a'] == pos_pair[0]) & (nonadd['pos_b'] == pos_pair[1])) |
                ((nonadd['pos_a'] == pos_pair[1]) & (nonadd['pos_b'] == pos_pair[0]))
            )
            subset = nonadd[mask]
            if len(subset) > 0:
                types = subset['interaction_type'].value_counts().to_dict()
                print(f"  {pos_pair[0]}×{pos_pair[1]}: {len(subset)} interactions — {types}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "mutation_interaction_results_v2.csv")
    results_df.to_csv(out_path, index=False)

    sig_path = os.path.join(os.path.dirname(__file__), "mutation_interaction_significant.csv")
    nonadd.to_csv(sig_path, index=False)

    print(f"\nAll results: {out_path}")
    print(f"Significant non-additive: {sig_path}")


if __name__ == "__main__":
    main()
