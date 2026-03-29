"""
Directional Walk — encode GOF/LOF travel direction on the mutation graph.

Each patient's mutations are walked through the coupling-channel graph.
The walk encodes three things:
  1. Which channels are severed (topology)
  2. How they're severed — GOF vs LOF (direction)
  3. Cross-channel pair types (interaction)

GOF pushes signal forward. LOF removes a node. The combination matters:
  - GOF+LOF cross-channel: multiplicative (activating + removing brake)
  - GOF+GOF: redundant (parallel activation, saturating)
  - LOF+LOF: saturating (channel already severed)

Features per patient:
  - atlas_sum: additive log-HR baseline
  - tissue_intercept: cancer-type baseline shift
  - n_gof, n_lof: mutation direction counts
  - Per-channel GOF/LOF flags (6 channels × 2 directions = 12)
  - Cross-channel pair type counts (GOF_LOF, GOF_GOF, LOF_LOF × within/cross = 6)
  - Specific high-signal pathway interactions
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, NON_SILENT,
    MSK_DATASETS, GENE_FUNCTION,
)


_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)


class DirectionalWalk:
    """Compute directional walk features for all patients."""

    def __init__(self, dataset_name="msk_impact_50k"):
        paths = MSK_DATASETS[dataset_name]
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
        clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS', 'CANCER_TYPE'])
        clinical['event'] = clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )

        self.clinical = clinical
        self.mutations = mutations

        # Load atlas from Neo4j
        from gnn.data.graph_snapshot import load_atlas
        t1_raw, t2_raw, t3_raw, t4_raw, _ = load_atlas()

        # Convert to format expected here: {key: {'hr': ..., 'log_hr': ...}}
        self.t1 = {}
        for key, entry in t1_raw.items():
            self.t1[key] = {'hr': entry['hr'], 'log_hr': np.log(max(entry['hr'], 0.01))}
        self.t2 = {}
        for key, entry in t2_raw.items():
            self.t2[key] = {'hr': entry['hr'], 'log_hr': np.log(max(entry['hr'], 0.01))}
        self.t3 = {}
        for key, entry in t3_raw.items():
            self.t3[key] = {'hr': entry['hr'], 'log_hr': np.log(max(entry['hr'], 0.01))}
        self.t4 = {}
        for key, entry in t4_raw.items():
            self.t4[key] = {'hr': entry['hr'], 'log_hr': np.log(max(entry['hr'], 0.01))}

        # Tissue intercepts
        deceased = clinical[clinical['event'] == 1]
        global_median = deceased['OS_MONTHS'].median()
        ct_medians = deceased.groupby('CANCER_TYPE')['OS_MONTHS'].median()
        self.tissue_deltas = (ct_medians - global_median).to_dict()
        self.global_median = global_median

        # Per-patient gene sets with metadata
        gene_muts = mutations[['patientId', 'gene.hugoGeneSymbol', 'proteinChange']].copy()
        gene_muts.columns = ['patientId', 'gene', 'proteinChange']
        gene_muts['channel'] = gene_muts['gene'].map(CHANNEL_MAP)
        gene_muts['function'] = gene_muts['gene'].map(GENE_FUNCTION).fillna('context')
        gene_muts['is_hub'] = gene_muts['gene'].isin(_HUB_SET)

        ct_map = clinical[['patientId', 'CANCER_TYPE']].drop_duplicates('patientId')
        gene_muts = gene_muts.merge(ct_map, on='patientId', how='inner')

        self.gene_muts = gene_muts
        print(f"Patients: {clinical['patientId'].nunique()}", flush=True)
        print(f"Mutations: {len(gene_muts)}", flush=True)

    def _atlas_log_hr(self, cancer_type, gene, protein_change):
        """Look up log-HR from atlas (tier 1 > 2 > 3 > 4)."""
        ch = CHANNEL_MAP.get(gene)
        key1 = (cancer_type, gene, protein_change)
        if key1 in self.t1:
            return self.t1[key1]['log_hr']
        key2 = (cancer_type, gene)
        if key2 in self.t2:
            return self.t2[key2]['log_hr']
        if ch:
            key3 = (cancer_type, ch)
            if key3 in self.t3:
                return self.t3[key3]['log_hr']
        if key2 in self.t4:
            return self.t4[key2]['log_hr']
        return 0.0

    def compute_features(self):
        """Compute directional walk features for all patients.

        Returns DataFrame with one row per patient and columns:
            patientId, cancer_type, OS_MONTHS, event,
            atlas_sum, tissue_delta,
            n_gof, n_lof, n_context,
            ch_GOF_DDR ... ch_LOF_TissueArch (12 columns),
            GOF_LOF_within, GOF_LOF_cross, GOF_GOF_within, GOF_GOF_cross,
            LOF_LOF_within, LOF_LOF_cross,
            n_channels_severed, n_hub_hit, hub_ratio,
            gof_lof_ratio,
            cross_gof_lof_channels (count of unique cross-channel GOF+LOF pairs)
        """
        print("Computing directional walk features...", flush=True)

        # Group mutations by patient
        patient_groups = self.gene_muts.groupby('patientId')

        records = []
        for _, row in self.clinical.iterrows():
            pid = row['patientId']
            ct = row['CANCER_TYPE']

            rec = {
                'patientId': pid,
                'cancer_type': ct,
                'OS_MONTHS': row['OS_MONTHS'],
                'event': row['event'],
                'tissue_delta': self.tissue_deltas.get(ct, 0.0),
            }

            if pid in patient_groups.groups:
                grp = patient_groups.get_group(pid)
                # Deduplicate by gene (keep first)
                genes_seen = set()
                gene_list = []
                for _, m in grp.iterrows():
                    if m['gene'] not in genes_seen:
                        genes_seen.add(m['gene'])
                        gene_list.append(m)

                # Atlas sum
                atlas_sum = sum(
                    self._atlas_log_hr(ct, m['gene'], m['proteinChange'])
                    for m in gene_list
                )
                rec['atlas_sum'] = atlas_sum

                # Direction counts
                funcs = [m['function'] for m in gene_list]
                rec['n_gof'] = sum(1 for f in funcs if f == 'GOF')
                rec['n_lof'] = sum(1 for f in funcs if f == 'LOF')
                rec['n_context'] = sum(1 for f in funcs if f == 'context')

                # Per-channel direction flags
                for ch in CHANNEL_NAMES:
                    rec[f'ch_GOF_{ch}'] = 0
                    rec[f'ch_LOF_{ch}'] = 0

                channels_severed = set()
                n_hub = 0
                for m in gene_list:
                    ch = m['channel']
                    if ch:
                        channels_severed.add(ch)
                    if m['function'] == 'GOF' and ch:
                        rec[f'ch_GOF_{ch}'] = 1
                    elif m['function'] == 'LOF' and ch:
                        rec[f'ch_LOF_{ch}'] = 1
                    if m['is_hub']:
                        n_hub += 1

                rec['n_channels_severed'] = len(channels_severed)
                rec['n_hub_hit'] = n_hub
                rec['n_genes'] = len(gene_list)
                rec['hub_ratio'] = n_hub / max(len(gene_list), 1)

                # GOF/LOF ratio
                total_dir = rec['n_gof'] + rec['n_lof']
                rec['gof_lof_ratio'] = rec['n_gof'] / max(total_dir, 1)

                # Pair type counts
                pair_counts = defaultdict(int)
                cross_gof_lof_channels = set()

                for a, b in combinations(gene_list, 2):
                    fa, fb = a['function'], b['function']
                    ca, cb = a['channel'], b['channel']
                    same_ch = ca == cb

                    if (fa == 'GOF' and fb == 'LOF') or (fa == 'LOF' and fb == 'GOF'):
                        ptype = 'GOF_LOF'
                        if not same_ch and ca and cb:
                            # Track which channel pairs have GOF+LOF
                            cross_gof_lof_channels.add(tuple(sorted([ca, cb])))
                    elif fa == 'GOF' and fb == 'GOF':
                        ptype = 'GOF_GOF'
                    elif fa == 'LOF' and fb == 'LOF':
                        ptype = 'LOF_LOF'
                    else:
                        ptype = 'other'

                    locality = 'within' if same_ch else 'cross'
                    pair_counts[f'{ptype}_{locality}'] += 1

                for pt in ['GOF_LOF_within', 'GOF_LOF_cross',
                           'GOF_GOF_within', 'GOF_GOF_cross',
                           'LOF_LOF_within', 'LOF_LOF_cross']:
                    rec[pt] = pair_counts.get(pt, 0)

                rec['cross_gof_lof_channels'] = len(cross_gof_lof_channels)

                # Specific high-signal paths (binary flags)
                # PI3K_Growth GOF + CellCycle LOF (the universal bad path)
                rec['path_PI3K_GOF_x_CC_LOF'] = int(
                    rec['ch_GOF_PI3K_Growth'] and rec['ch_LOF_CellCycle']
                )
                # PI3K_Growth GOF + TissueArch LOF
                rec['path_PI3K_GOF_x_TA_LOF'] = int(
                    rec['ch_GOF_PI3K_Growth'] and rec['ch_LOF_TissueArch']
                )
                # CellCycle GOF + DDR LOF
                rec['path_CC_GOF_x_DDR_LOF'] = int(
                    rec['ch_GOF_CellCycle'] and rec['ch_LOF_DDR']
                )
                # Any GOF + Immune LOF
                rec['path_anyGOF_x_Immune_LOF'] = int(
                    rec['n_gof'] > 0 and rec['ch_LOF_Immune']
                )

            else:
                # No mutations — wild type
                rec['atlas_sum'] = 0.0
                rec['n_gof'] = 0
                rec['n_lof'] = 0
                rec['n_context'] = 0
                for ch in CHANNEL_NAMES:
                    rec[f'ch_GOF_{ch}'] = 0
                    rec[f'ch_LOF_{ch}'] = 0
                rec['n_channels_severed'] = 0
                rec['n_hub_hit'] = 0
                rec['n_genes'] = 0
                rec['hub_ratio'] = 0.0
                rec['gof_lof_ratio'] = 0.0
                for pt in ['GOF_LOF_within', 'GOF_LOF_cross',
                           'GOF_GOF_within', 'GOF_GOF_cross',
                           'LOF_LOF_within', 'LOF_LOF_cross']:
                    rec[pt] = 0
                rec['cross_gof_lof_channels'] = 0
                rec['path_PI3K_GOF_x_CC_LOF'] = 0
                rec['path_PI3K_GOF_x_TA_LOF'] = 0
                rec['path_CC_GOF_x_DDR_LOF'] = 0
                rec['path_anyGOF_x_Immune_LOF'] = 0

            records.append(rec)

        df = pd.DataFrame(records)
        print(f"Features computed: {df.shape}", flush=True)
        return df


def main():
    """Compute directional walk features and test prediction improvement."""
    from sklearn.model_selection import StratifiedKFold
    from lifelines import CoxPHFitter

    dw = DirectionalWalk()
    df = dw.compute_features()

    # Standardize continuous features
    for col in ['atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_genes',
                'n_channels_severed', 'n_hub_hit']:
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    df = df[df['OS_MONTHS'] > 0].copy()

    # Define model feature sets
    models = {
        'atlas_only': ['atlas_sum'],
        'atlas+tissue': ['atlas_sum', 'tissue_delta'],
        'atlas+tissue+counts': ['atlas_sum', 'tissue_delta', 'n_gof', 'n_lof',
                                 'n_channels_severed'],
        'atlas+tissue+direction': [
            'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
            'GOF_LOF_within', 'GOF_LOF_cross',
            'GOF_GOF_cross', 'LOF_LOF_within',
        ],
        'atlas+tissue+direction+paths': [
            'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
            'GOF_LOF_within', 'GOF_LOF_cross',
            'GOF_GOF_cross', 'LOF_LOF_within',
            'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
            'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
        ],
        'full_walk': [
            'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
            'n_hub_hit', 'hub_ratio', 'gof_lof_ratio',
            'GOF_LOF_within', 'GOF_LOF_cross',
            'GOF_GOF_within', 'GOF_GOF_cross',
            'LOF_LOF_within', 'LOF_LOF_cross',
            'cross_gof_lof_channels',
            'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
            'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
        ] + [f'ch_GOF_{ch}' for ch in CHANNEL_NAMES]
          + [f'ch_LOF_{ch}' for ch in CHANNEL_NAMES],
    }

    # 5-fold CV with concordance index
    print(f"\n{'='*80}")
    print("COX REGRESSION: DIRECTIONAL WALK FEATURES")
    print(f"{'='*80}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, features in models.items():
        c_indices = []

        for fold, (train_idx, val_idx) in enumerate(
                skf.split(df, df['event'])):

            train = df.iloc[train_idx]
            val = df.iloc[val_idx]

            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(
                    train[features + ['OS_MONTHS', 'event']],
                    duration_col='OS_MONTHS',
                    event_col='event',
                )

                # Predict on validation
                val_hazard = cph.predict_partial_hazard(val[features]).values.flatten()
                val_times = val['OS_MONTHS'].values
                val_events = val['event'].values

                # Concordance index
                from lifelines.utils import concordance_index as ci
                c = ci(val_times, -val_hazard, val_events)
                c_indices.append(c)
            except Exception as e:
                print(f"  Fold {fold} error: {e}")
                c_indices.append(0.5)

        mean_c = np.mean(c_indices)
        std_c = np.std(c_indices)
        print(f"  {model_name:40s} C={mean_c:.4f} ± {std_c:.4f}  "
              f"folds={[f'{c:.4f}' for c in c_indices]}")

    # Per-cancer-type comparison: atlas_only vs full_walk
    print(f"\n{'='*80}")
    print("PER CANCER TYPE: ATLAS vs FULL DIRECTIONAL WALK")
    print(f"{'='*80}")

    full_features = models['full_walk']

    print(f"{'Cancer Type':35s} {'N':>5s} {'C_atlas':>7s} {'C_walk':>7s} {'Lift':>7s}")
    print('-' * 65)

    for ct in ['Non-Small Cell Lung Cancer', 'Breast Cancer', 'Colorectal Cancer',
               'Pancreatic Cancer', 'Endometrial Cancer', 'Bladder Cancer',
               'Ovarian Cancer', 'Glioma', 'Melanoma', 'Esophagogastric Cancer',
               'Hepatobiliary Cancer', 'Prostate Cancer', 'Soft Tissue Sarcoma']:
        ct_df = df[df['cancer_type'] == ct].copy()
        if len(ct_df) < 200 or ct_df['event'].sum() < 50:
            continue

        c_atlas_list = []
        c_walk_list = []

        for fold, (train_idx, val_idx) in enumerate(
                skf.split(ct_df, ct_df['event'])):

            train = ct_df.iloc[train_idx]
            val = ct_df.iloc[val_idx]

            try:
                # Atlas only
                cph1 = CoxPHFitter(penalizer=0.01)
                cph1.fit(train[['atlas_sum', 'OS_MONTHS', 'event']],
                         duration_col='OS_MONTHS', event_col='event')
                h1 = cph1.predict_partial_hazard(val[['atlas_sum']]).values.flatten()

                # Full walk
                # Drop features with zero variance in this cancer type
                valid_feats = [f for f in full_features
                               if train[f].std() > 0.001]
                cph2 = CoxPHFitter(penalizer=0.01)
                cph2.fit(train[valid_feats + ['OS_MONTHS', 'event']],
                         duration_col='OS_MONTHS', event_col='event')
                h2 = cph2.predict_partial_hazard(val[valid_feats]).values.flatten()

                from lifelines.utils import concordance_index as ci
                c_atlas_list.append(ci(val['OS_MONTHS'].values, -h1, val['event'].values))
                c_walk_list.append(ci(val['OS_MONTHS'].values, -h2, val['event'].values))
            except Exception:
                pass

        if c_atlas_list and c_walk_list:
            ca = np.mean(c_atlas_list)
            cw = np.mean(c_walk_list)
            lift = cw - ca
            print(f"{ct[:35]:35s} {len(ct_df):5d} {ca:7.4f} {cw:7.4f} {lift:+7.4f}")

    # Print Cox coefficients for the full model
    print(f"\n{'='*80}")
    print("FULL WALK COX COEFFICIENTS (all patients)")
    print(f"{'='*80}")

    valid_feats = [f for f in models['full_walk'] if df[f].std() > 0.001]
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df[valid_feats + ['OS_MONTHS', 'event']],
            duration_col='OS_MONTHS', event_col='event')
    summary = cph.summary[['coef', 'exp(coef)', 'p']].sort_values('p')
    print(summary.to_string())


if __name__ == "__main__":
    main()
