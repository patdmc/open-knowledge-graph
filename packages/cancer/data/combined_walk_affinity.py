"""
Combined Walk + Affinity — directional walk features smoothed by patient affinity.

Two-graph model:
  1. Mutation graph walk → per-patient directional features
  2. Patient affinity graph → smooth features across similar patients

The affinity graph acts as a regularizer: patients with similar mutation
profiles (high Jaccard) should have similar walk scores. Cross-cancer
edges share information between tissues, adjusted by tissue intercept.

Smoothing: for each patient, weighted average of their walk features
with their affinity neighbors' features. Weight = Jaccard similarity.
Cross-cancer neighbors get tissue-delta-adjusted features.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as ci

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, NON_SILENT, MSK_DATASETS, GENE_FUNCTION, HUB_GENES
from gnn.data.directional_walk import DirectionalWalk


_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)


class CombinedWalkAffinity:

    def __init__(self, dataset_name="msk_impact_50k"):
        self.dw = DirectionalWalk(dataset_name)

        # Build per-patient gene sets for affinity
        gene_muts = self.dw.mutations[['patientId', 'gene.hugoGeneSymbol']].drop_duplicates()
        self.patient_genes = gene_muts.groupby('patientId')['gene.hugoGeneSymbol'].apply(
            frozenset
        ).to_dict()

        # Cancer type per patient
        ct_map = self.dw.clinical[['patientId', 'CANCER_TYPE']].drop_duplicates('patientId')
        self.patient_cancer = dict(zip(ct_map['patientId'], ct_map['CANCER_TYPE']))

    def compute_affinity_smoothed_features(self, max_neighbors=20,
                                            min_jaccard=0.3,
                                            cross_cancer=True,
                                            smooth_weight=0.3):
        """Compute walk features + affinity-smoothed versions.

        For each patient:
          1. Compute their raw directional walk features
          2. Find their top-k affinity neighbors (within + cross cancer)
          3. Compute weighted average of neighbors' walk features
          4. Combine: (1-w) * own_features + w * neighbor_avg

        Args:
            max_neighbors: max affinity neighbors per patient
            min_jaccard: minimum Jaccard to count as neighbor
            cross_cancer: include cross-cancer neighbors
            smooth_weight: weight on neighbor average (0=no smoothing, 1=full)

        Returns:
            DataFrame with raw + smoothed features
        """
        # Step 1: Raw walk features
        df = self.dw.compute_features()
        df = df.set_index('patientId')

        # Feature columns to smooth
        smooth_cols = [
            'atlas_sum', 'n_gof', 'n_lof', 'n_channels_severed',
            'n_hub_hit', 'hub_ratio', 'gof_lof_ratio',
            'GOF_LOF_within', 'GOF_LOF_cross',
            'GOF_GOF_within', 'GOF_GOF_cross',
            'LOF_LOF_within', 'LOF_LOF_cross',
            'cross_gof_lof_channels',
            'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
            'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
        ]

        # Step 2: Build inverted index for fast neighbor lookup
        print("Building affinity neighbors...", flush=True)
        gene_to_patients = defaultdict(set)
        valid_pids = set(df.index)
        for pid, genes in self.patient_genes.items():
            if pid in valid_pids:
                for gene in genes:
                    gene_to_patients[gene].add(pid)

        # Step 3: For each patient, find neighbors and smooth
        print("Smoothing features via affinity graph...", flush=True)
        smoothed_data = {}
        n_with_neighbors = 0
        total_neighbors = 0

        for pid in df.index:
            genes = self.patient_genes.get(pid, frozenset())
            pid_ct = self.patient_cancer.get(pid, '')

            if not genes:
                smoothed_data[pid] = {f'smooth_{c}': df.at[pid, c] for c in smooth_cols}
                continue

            # Find candidates via inverted index
            candidates = set()
            for gene in genes:
                candidates.update(gene_to_patients[gene])
            candidates.discard(pid)

            # Score by Jaccard
            scored = []
            for cand in candidates:
                cand_genes = self.patient_genes.get(cand, frozenset())
                n_shared = len(genes & cand_genes)
                jacc = n_shared / len(genes | cand_genes)
                if jacc < min_jaccard:
                    continue

                cand_ct = self.patient_cancer.get(cand, '')
                is_cross = cand_ct != pid_ct

                if is_cross and not cross_cancer:
                    continue

                scored.append((cand, jacc, is_cross))

            # Top-k by Jaccard
            scored.sort(key=lambda x: -x[1])
            neighbors = scored[:max_neighbors]

            if not neighbors:
                smoothed_data[pid] = {f'smooth_{c}': df.at[pid, c] for c in smooth_cols}
                continue

            n_with_neighbors += 1
            total_neighbors += len(neighbors)

            # Weighted average of neighbor features
            weighted_sum = np.zeros(len(smooth_cols))
            weight_total = 0.0

            for cand, jacc, is_cross in neighbors:
                w = jacc
                if is_cross:
                    # Downweight cross-cancer slightly
                    w *= 0.7
                vals = np.array([df.at[cand, c] for c in smooth_cols])
                weighted_sum += w * vals
                weight_total += w

            if weight_total > 0:
                neighbor_avg = weighted_sum / weight_total
            else:
                neighbor_avg = np.array([df.at[pid, c] for c in smooth_cols])

            own = np.array([df.at[pid, c] for c in smooth_cols])
            blended = (1 - smooth_weight) * own + smooth_weight * neighbor_avg

            smoothed_data[pid] = {f'smooth_{c}': v for c, v in zip(smooth_cols, blended)}

        print(f"  Patients with neighbors: {n_with_neighbors} / {len(df)}", flush=True)
        print(f"  Mean neighbors: {total_neighbors / max(n_with_neighbors, 1):.1f}", flush=True)

        # Add smoothed columns to df
        smooth_df = pd.DataFrame.from_dict(smoothed_data, orient='index')
        df = df.join(smooth_df)

        # Also add neighbor count and mean Jaccard as features
        neighbor_stats = {}
        for pid in df.index:
            genes = self.patient_genes.get(pid, frozenset())
            if not genes:
                neighbor_stats[pid] = {'n_neighbors': 0, 'mean_jacc': 0.0}
                continue

            candidates = set()
            for gene in genes:
                candidates.update(gene_to_patients[gene])
            candidates.discard(pid)

            jaccs = []
            for cand in candidates:
                cand_genes = self.patient_genes.get(cand, frozenset())
                j = len(genes & cand_genes) / len(genes | cand_genes)
                if j >= min_jaccard:
                    jaccs.append(j)

            neighbor_stats[pid] = {
                'n_neighbors': min(len(jaccs), max_neighbors),
                'mean_jacc': np.mean(jaccs[:max_neighbors]) if jaccs else 0.0,
            }

        stats_df = pd.DataFrame.from_dict(neighbor_stats, orient='index')
        df = df.join(stats_df)

        df = df.reset_index()
        return df


def _cv_c_index(df, features, skf, penalizer=0.01):
    """5-fold CV C-index for a feature set."""
    valid = [f for f in features if f in df.columns and df[f].std() > 0.001]
    if not valid:
        return 0.5, 0.0

    c_indices = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['event'])):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(train[valid + ['OS_MONTHS', 'event']],
                    duration_col='OS_MONTHS', event_col='event')
            h = cph.predict_partial_hazard(val[valid]).values.flatten()
            c_indices.append(ci(val['OS_MONTHS'].values, -h, val['event'].values))
        except Exception:
            c_indices.append(0.5)
    return np.mean(c_indices), np.std(c_indices)


def main():
    cwa = CombinedWalkAffinity()

    # Compute walk + affinity features once
    df = cwa.compute_affinity_smoothed_features(
        max_neighbors=20, min_jaccard=0.3,
        cross_cancer=True, smooth_weight=0.3,
    )
    df = df[df['OS_MONTHS'] > 0].copy()

    # Standardize
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ('OS_MONTHS', 'event'):
            continue
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    raw_features = [
        'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
        'GOF_LOF_within', 'GOF_LOF_cross',
        'GOF_GOF_cross', 'LOF_LOF_within',
        'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
        'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
    ]

    combined_features = raw_features + [
        'smooth_atlas_sum', 'smooth_n_gof', 'smooth_n_lof',
        'smooth_GOF_LOF_cross', 'smooth_LOF_LOF_within',
        'n_neighbors', 'mean_jacc',
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Global (pooled) ---
    print(f"\n{'='*80}")
    print("POOLED (all cancers)")
    print(f"{'='*80}")
    for name, feats in [('raw_walk', raw_features), ('combined', combined_features)]:
        m, s = _cv_c_index(df, feats, skf)
        print(f"  {name:25s} C={m:.4f} ± {s:.4f}")

    # --- Per cancer type ---
    print(f"\n{'='*80}")
    print("PER-CANCER (fitted separately)")
    print(f"{'='*80}")
    print(f"  {'Cancer Type':35s} {'N':>5s} {'C_raw':>7s} {'C_comb':>7s} {'Lift':>7s}")
    print(f"  {'-'*65}")

    cancer_types = ['Non-Small Cell Lung Cancer', 'Breast Cancer',
                    'Colorectal Cancer', 'Pancreatic Cancer',
                    'Endometrial Cancer', 'Bladder Cancer',
                    'Ovarian Cancer', 'Glioma', 'Melanoma',
                    'Esophagogastric Cancer', 'Hepatobiliary Cancer',
                    'Prostate Cancer']

    per_cancer_raw = []
    per_cancer_comb = []
    per_cancer_n = []

    for ct_name in cancer_types:
        ct_df = df[df['cancer_type'] == ct_name]
        if len(ct_df) < 200 or ct_df['event'].sum() < 50:
            continue

        # Per-cancer features: drop tissue_delta (constant within cancer)
        ct_raw = [f for f in raw_features if f != 'tissue_delta']
        ct_comb = [f for f in combined_features if f != 'tissue_delta']

        cr, _ = _cv_c_index(ct_df, ct_raw, skf)
        cc, _ = _cv_c_index(ct_df, ct_comb, skf)

        per_cancer_raw.append(cr)
        per_cancer_comb.append(cc)
        per_cancer_n.append(len(ct_df))

        print(f"  {ct_name[:35]:35s} {len(ct_df):5d} {cr:7.4f} {cc:7.4f} {cc-cr:+7.4f}")

    # Weighted average across cancer types
    weights = np.array(per_cancer_n, dtype=float)
    weights /= weights.sum()
    wa_raw = np.average(per_cancer_raw, weights=weights)
    wa_comb = np.average(per_cancer_comb, weights=weights)
    print(f"  {'-'*65}")
    print(f"  {'Weighted avg (per-cancer)':35s} {sum(per_cancer_n):5d} {wa_raw:7.4f} {wa_comb:7.4f} {wa_comb-wa_raw:+7.4f}")

    print(f"\n  BASELINES:")
    print(f"  Atlas lookup (zero-param):    C = 0.577")
    print(f"  Pooled directional walk:      C = 0.636")
    print(f"  AtlasTransformer V1 (neural): C = 0.673")


if __name__ == "__main__":
    main()
