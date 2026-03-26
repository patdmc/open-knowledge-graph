"""
Scan for multiplicative mutation-pair interactions on survival.

For each pair of recurrent mutations (≥30 patients with both):
  - Fit Cox PH: hazard ~ mut_A + mut_B + mut_A*mut_B + age + cancer_type
  - If interaction term p < 0.05 and coefficient magnitude > 0.3,
    the pair has a non-additive (multiplicative) effect on survival.

Focus: cross-channel pairs (the theory predicts coupling interactions).
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_PATIENTS_WITH_MUTATION = 200    # minimum patients carrying a mutation
MIN_PATIENTS_WITH_BOTH = 30         # minimum co-occurrence for interaction test
P_THRESHOLD = 0.05
COEF_THRESHOLD = 0.2                # |beta_interaction| > this

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_name="msk_impact_50k"):
    paths = MSK_DATASETS[dataset_name]
    print(f"Loading {dataset_name}...")
    mutations = pd.read_csv(paths["mutations"])
    clinical = pd.read_csv(paths["clinical"])
    sample_clinical = pd.read_csv(paths["sample_clinical"])

    # Filter non-silent, channel genes
    mutations = mutations[mutations['mutationType'].isin(NON_SILENT)]
    mutations = mutations[mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())]

    # Clinical prep
    clinical = clinical.merge(
        sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
        on='patientId', how='left'
    )
    clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    clinical['event'] = clinical['OS_STATUS'].apply(
        lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
    )

    # Age
    clinical['age'] = pd.to_numeric(clinical.get('AGE_AT_DX', 60), errors='coerce').fillna(60)
    clinical['age_z'] = (clinical['age'] - clinical['age'].mean()) / (clinical['age'].std() + 1e-8)

    return mutations, clinical


def get_channel_and_position(gene):
    """Return (channel, position) for a gene."""
    channel = CHANNEL_MAP.get(gene)
    if channel is None:
        return None, None
    for ch, hubs in HUB_GENES.items():
        if gene in hubs:
            return channel, "hub"
    return channel, "leaf"


# ---------------------------------------------------------------------------
# Find recurrent mutations and build patient × mutation matrix
# ---------------------------------------------------------------------------

def build_mutation_matrix(mutations, clinical):
    """Build a sparse patient × mutation binary matrix for recurrent mutations."""

    # Count patients per (gene, proteinChange)
    mut_counts = (mutations
                  .groupby(['gene.hugoGeneSymbol', 'proteinChange'])
                  ['patientId'].nunique()
                  .reset_index())
    mut_counts.columns = ['gene', 'pc', 'n_patients']

    # Filter to recurrent
    recurrent = mut_counts[mut_counts['n_patients'] >= MIN_PATIENTS_WITH_MUTATION]
    recurrent = recurrent.sort_values('n_patients', ascending=False)

    print(f"Recurrent mutations (≥{MIN_PATIENTS_WITH_MUTATION} patients): {len(recurrent)}")

    # Build mutation labels with channel info
    mut_labels = []
    for _, row in recurrent.iterrows():
        ch, pos = get_channel_and_position(row['gene'])
        label = f"{row['gene']}_{row['pc']}"
        mut_labels.append({
            'label': label,
            'gene': row['gene'],
            'pc': row['pc'],
            'channel': ch,
            'position': pos,
            'n_patients': row['n_patients'],
        })

    mut_labels = pd.DataFrame(mut_labels)
    print(f"Mutations to scan: {len(mut_labels)}")
    for ch in CHANNEL_NAMES:
        n = (mut_labels['channel'] == ch).sum()
        print(f"  {ch}: {n} mutations")

    # Build patient × mutation binary matrix
    patient_ids = clinical['patientId'].unique()
    patient_set = set(patient_ids)

    # Filter mutations to recurrent ones
    recurrent_keys = set(zip(recurrent['gene'], recurrent['pc']))
    filtered = mutations[
        mutations.apply(
            lambda r: (r['gene.hugoGeneSymbol'], r.get('proteinChange', '')) in recurrent_keys,
            axis=1
        )
    ]

    # Pivot: for each patient, which mutations are present
    mut_matrix = pd.DataFrame(0, index=patient_ids, columns=mut_labels['label'].tolist())

    for _, row in filtered.iterrows():
        pid = row['patientId']
        gene = row['gene.hugoGeneSymbol']
        pc = row.get('proteinChange', '')
        label = f"{gene}_{pc}"
        if pid in patient_set and label in mut_matrix.columns:
            mut_matrix.loc[pid, label] = 1

    return mut_matrix, mut_labels


# ---------------------------------------------------------------------------
# Scan for multiplicative interactions
# ---------------------------------------------------------------------------

def scan_interactions(mut_matrix, mut_labels, clinical):
    """Test all cross-channel mutation pairs for multiplicative interaction."""

    # Merge clinical with mutation matrix
    df = clinical.set_index('patientId')[['OS_MONTHS', 'event', 'age_z']].copy()
    df = df.join(mut_matrix, how='inner')
    df = df[df['OS_MONTHS'] > 0]

    print(f"\nPatients in analysis: {len(df)}")

    labels = mut_labels['label'].tolist()
    channels = dict(zip(mut_labels['label'], mut_labels['channel']))

    results = []
    n_pairs = 0
    n_tested = 0

    # All pairs, prioritizing cross-channel
    for i, label_a in enumerate(labels):
        for label_b in labels[i+1:]:
            ch_a = channels[label_a]
            ch_b = channels[label_b]

            # Count co-occurrence
            both = ((df[label_a] == 1) & (df[label_b] == 1)).sum()
            if both < MIN_PATIENTS_WITH_BOTH:
                continue

            n_pairs += 1

            # Build interaction term
            interaction_col = f"{label_a}_x_{label_b}"
            test_df = df[['OS_MONTHS', 'event', 'age_z', label_a, label_b]].copy()
            test_df[interaction_col] = test_df[label_a] * test_df[label_b]

            # Fit Cox PH
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(test_df, duration_col='OS_MONTHS', event_col='event')

                interaction_coef = cph.params_[interaction_col]
                interaction_p = cph.summary.loc[interaction_col, 'p']
                interaction_hr = np.exp(interaction_coef)

                coef_a = cph.params_[label_a]
                coef_b = cph.params_[label_b]
                hr_a = np.exp(coef_a)
                hr_b = np.exp(coef_b)

                n_tested += 1

                if interaction_p < P_THRESHOLD and abs(interaction_coef) > COEF_THRESHOLD:
                    cross_channel = ch_a != ch_b
                    results.append({
                        'mut_a': label_a,
                        'mut_b': label_b,
                        'channel_a': ch_a,
                        'channel_b': ch_b,
                        'cross_channel': cross_channel,
                        'n_both': both,
                        'n_a': (df[label_a] == 1).sum(),
                        'n_b': (df[label_b] == 1).sum(),
                        'hr_a': hr_a,
                        'hr_b': hr_b,
                        'hr_interaction': interaction_hr,
                        'coef_interaction': interaction_coef,
                        'p_interaction': interaction_p,
                        'additive_expected_hr': hr_a * hr_b,
                        'multiplicative_excess': interaction_hr,
                        'direction': 'synergistic' if interaction_coef > 0 else 'protective',
                    })

            except Exception as e:
                continue

            if n_pairs % 100 == 0:
                print(f"  Tested {n_pairs} pairs, {len(results)} significant...", flush=True)

    print(f"\nTotal pairs with ≥{MIN_PATIENTS_WITH_BOTH} co-occurrences: {n_pairs}")
    print(f"Successfully tested: {n_tested}")
    print(f"Significant interactions: {len(results)}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('p_interaction')
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mutations, clinical = load_data()
    mut_matrix, mut_labels = build_mutation_matrix(mutations, clinical)
    results = scan_interactions(mut_matrix, mut_labels, clinical)

    if len(results) > 0:
        print(f"\n{'='*70}")
        print(f"TOP MULTIPLICATIVE INTERACTIONS")
        print(f"{'='*70}")

        # Cross-channel first
        cross = results[results['cross_channel']]
        within = results[~results['cross_channel']]

        print(f"\nCross-channel interactions: {len(cross)}")
        print(f"Within-channel interactions: {len(within)}")

        if len(cross) > 0:
            print(f"\n--- Cross-channel (theory-predicted) ---")
            for _, r in cross.head(20).iterrows():
                print(f"  {r['mut_a']} ({r['channel_a']}) × {r['mut_b']} ({r['channel_b']})")
                print(f"    n={r['n_both']}, HR_int={r['hr_interaction']:.3f}, "
                      f"p={r['p_interaction']:.4f}, {r['direction']}")

        if len(within) > 0:
            print(f"\n--- Within-channel ---")
            for _, r in within.head(10).iterrows():
                print(f"  {r['mut_a']} ({r['channel_a']}) × {r['mut_b']} ({r['channel_b']})")
                print(f"    n={r['n_both']}, HR_int={r['hr_interaction']:.3f}, "
                      f"p={r['p_interaction']:.4f}, {r['direction']}")

        # Summary stats
        if len(cross) > 0:
            print(f"\nCross-channel protective: {(cross['direction'] == 'protective').sum()}")
            print(f"Cross-channel synergistic: {(cross['direction'] == 'synergistic').sum()}")

        # Save
        out_path = os.path.join(os.path.dirname(__file__), "mutation_interaction_results.csv")
        results.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
    else:
        print("\nNo significant multiplicative interactions found.")
