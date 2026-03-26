"""
Escalation Entropy — measure how scripted each cancer's mutation path is.

Hypothesis: cancers with low mutation path entropy (tight, scripted
escalation through specific channels) are the deadliest. Cancers with
high entropy (diffuse, unscripted mutations across many channels) are
more survivable because off-script mutations represent failed escalation.

Metrics per cancer type:
  1. Channel entropy: H(channel distribution of mutations)
  2. Gene concentration: fraction of mutations in top-3 genes
  3. Path stereotypy: how similar are patients' mutation profiles?
  4. Harmful/protective ratio from the atlas
  5. Median survival

Then: correlate entropy with survival across cancer types.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, LEAF_GENES,
    NON_SILENT, MSK_DATASETS,
)


def shannon_entropy(counts):
    """Shannon entropy of a count vector."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


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
    mutations['channel'] = mutations['gene.hugoGeneSymbol'].map(get_channel)
    mutations['position'] = mutations['gene.hugoGeneSymbol'].map(get_position)

    clinical = clinical.merge(
        sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
        on='patientId', how='left'
    )
    clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    clinical['event'] = clinical['OS_STATUS'].apply(
        lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
    )

    valid_cts = clinical['CANCER_TYPE'].value_counts()
    valid_cts = valid_cts[valid_cts >= 200].index.tolist()

    # Load atlas for harmful/protective counts
    atlas_path = os.path.join(os.path.dirname(__file__), "survival_atlas_confident.csv")
    atlas = pd.read_csv(atlas_path) if os.path.exists(atlas_path) else None

    results = []

    for ct in valid_cts:
        ct_pids = set(clinical[clinical['CANCER_TYPE'] == ct]['patientId'])
        ct_muts = mutations[mutations['patientId'].isin(ct_pids)]
        ct_clin = clinical[clinical['CANCER_TYPE'] == ct]

        if len(ct_muts) == 0:
            continue

        n_patients = len(ct_clin)
        n_mutated_patients = ct_muts['patientId'].nunique()

        # --- 1. Channel entropy ---
        channel_counts = ct_muts['channel'].value_counts()
        ch_entropy = shannon_entropy(channel_counts.values)
        # Normalize by max possible (log2 of 6 channels)
        max_entropy = np.log2(len(CHANNEL_NAMES))
        ch_entropy_norm = ch_entropy / max_entropy if max_entropy > 0 else 0

        # --- 2. Gene concentration (top-3 share) ---
        gene_counts = ct_muts['gene.hugoGeneSymbol'].value_counts()
        top3_share = gene_counts.head(3).sum() / gene_counts.sum()
        top1_gene = gene_counts.index[0]
        top1_share = gene_counts.iloc[0] / gene_counts.sum()

        # --- 3. Hub concentration (fraction of mutations in hub genes) ---
        hub_frac = (ct_muts['position'] == 'hub').mean()

        # --- 4. Patient-level path stereotypy ---
        # For each patient, compute channel vector, then measure pairwise similarity
        patient_channels = {}
        for pid, group in ct_muts.groupby('patientId'):
            vec = np.zeros(len(CHANNEL_NAMES))
            for ch in group['channel']:
                idx = CHANNEL_NAMES.index(ch) if ch in CHANNEL_NAMES else -1
                if idx >= 0:
                    vec[idx] = 1  # binary: channel hit or not
            patient_channels[pid] = vec

        if len(patient_channels) > 10:
            vecs = np.array(list(patient_channels.values()))
            # Mean pairwise cosine similarity (sample for speed)
            n_sample = min(500, len(vecs))
            idx = np.random.choice(len(vecs), n_sample, replace=False)
            sample_vecs = vecs[idx]
            norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            normed = sample_vecs / norms
            sim_matrix = normed @ normed.T
            # Mean off-diagonal
            mask = ~np.eye(n_sample, dtype=bool)
            stereotypy = sim_matrix[mask].mean()
        else:
            stereotypy = np.nan

        # --- 5. Per-gene mutation specificity ---
        # How concentrated is each gene's mutation spectrum?
        gene_mut_entropies = []
        for gene, gdf in ct_muts.groupby('gene.hugoGeneSymbol'):
            pc_counts = gdf['proteinChange'].value_counts()
            if len(pc_counts) > 1:
                gene_mut_entropies.append(shannon_entropy(pc_counts.values))
        mean_gene_mut_entropy = np.mean(gene_mut_entropies) if gene_mut_entropies else 0

        # --- 6. Survival ---
        median_os = ct_clin['OS_MONTHS'].median()
        event_rate = ct_clin['event'].mean()
        # 3-year survival rate
        survived_36 = ((ct_clin['OS_MONTHS'] >= 36) |
                       ((ct_clin['event'] == 0) & (ct_clin['OS_MONTHS'] >= 36))).mean()

        # --- 7. Atlas harmful/protective ratio ---
        if atlas is not None:
            ct_atlas = atlas[atlas['cancer_type'] == ct]
            n_harmful = len(ct_atlas[ct_atlas['hr'] > 1.1])
            n_protective = len(ct_atlas[ct_atlas['hr'] < 0.9])
            hp_ratio = n_harmful / max(n_protective, 1)
        else:
            n_harmful = n_protective = 0
            hp_ratio = np.nan

        # --- 8. Dominant channel(s) ---
        dominant_channels = channel_counts.head(2).index.tolist()

        # --- 9. Number of distinct channels hit per patient ---
        channels_per_patient = []
        for pid, group in ct_muts.groupby('patientId'):
            channels_per_patient.append(group['channel'].nunique())
        mean_channels = np.mean(channels_per_patient)
        std_channels = np.std(channels_per_patient)

        results.append({
            'cancer_type': ct,
            'n_patients': n_patients,
            'n_mutated': n_mutated_patients,
            'mutation_rate': n_mutated_patients / n_patients,
            'channel_entropy': ch_entropy,
            'channel_entropy_norm': ch_entropy_norm,
            'top3_gene_share': top3_share,
            'top1_gene': top1_gene,
            'top1_gene_share': top1_share,
            'hub_fraction': hub_frac,
            'path_stereotypy': stereotypy,
            'mean_gene_mutation_entropy': mean_gene_mut_entropy,
            'mean_channels_per_patient': mean_channels,
            'std_channels_per_patient': std_channels,
            'dominant_channels': ', '.join(dominant_channels),
            'median_os_months': median_os,
            'event_rate': event_rate,
            'survival_36m': survived_36,
            'n_harmful_atlas': n_harmful,
            'n_protective_atlas': n_protective,
            'harmful_protective_ratio': hp_ratio,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('channel_entropy_norm')

    # ===================================================================
    # RESULTS
    # ===================================================================
    print(f"\n{'='*90}")
    print("ESCALATION ENTROPY BY CANCER TYPE")
    print(f"{'='*90}")
    print(f"{'Cancer Type':40s} {'ChEnt':>6s} {'Top1%':>6s} {'Hub%':>5s} "
          f"{'Stereo':>6s} {'MedOS':>6s} {'EvtRt':>6s} {'H/P':>5s} {'Path':>20s}")
    print("-" * 110)

    for _, r in df.iterrows():
        print(f"{r['cancer_type']:40s} {r['channel_entropy_norm']:6.3f} "
              f"{r['top1_gene_share']:6.1%} {r['hub_fraction']:5.1%} "
              f"{r['path_stereotypy']:6.3f} {r['median_os_months']:6.1f} "
              f"{r['event_rate']:6.1%} {r['harmful_protective_ratio']:5.2f} "
              f"{r['dominant_channels']:>20s}",
              flush=True)

    # ===================================================================
    # CORRELATIONS
    # ===================================================================
    print(f"\n{'='*70}")
    print("CORRELATIONS WITH SURVIVAL")
    print(f"{'='*70}")

    for metric in ['channel_entropy_norm', 'top1_gene_share', 'hub_fraction',
                    'path_stereotypy', 'mean_channels_per_patient',
                    'harmful_protective_ratio', 'mean_gene_mutation_entropy']:
        valid = df[[metric, 'median_os_months']].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.spearmanr(valid[metric], valid['median_os_months'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {metric:35s} vs median OS: rho={r:+.3f}, p={p:.4f} {sig}")

    for metric in ['channel_entropy_norm', 'top1_gene_share', 'hub_fraction',
                    'path_stereotypy', 'mean_channels_per_patient',
                    'harmful_protective_ratio', 'mean_gene_mutation_entropy']:
        valid = df[[metric, 'event_rate']].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.spearmanr(valid[metric], valid['event_rate'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {metric:35s} vs event rate: rho={r:+.3f}, p={p:.4f} {sig}")

    # ===================================================================
    # SCRIPTED vs DIFFUSE CLASSIFICATION
    # ===================================================================
    print(f"\n{'='*70}")
    print("SCRIPTED vs DIFFUSE CANCERS")
    print(f"{'='*70}")

    median_entropy = df['channel_entropy_norm'].median()
    scripted = df[df['channel_entropy_norm'] < median_entropy]
    diffuse = df[df['channel_entropy_norm'] >= median_entropy]

    print(f"\nScripted (low entropy, n={len(scripted)}):")
    print(f"  Mean median OS: {scripted['median_os_months'].mean():.1f} months")
    print(f"  Mean event rate: {scripted['event_rate'].mean():.1%}")
    print(f"  Mean H/P ratio: {scripted['harmful_protective_ratio'].mean():.2f}")
    for _, r in scripted.sort_values('channel_entropy_norm').iterrows():
        print(f"    {r['cancer_type']:35s} entropy={r['channel_entropy_norm']:.3f} "
              f"medOS={r['median_os_months']:.0f}m top1={r['top1_gene']}({r['top1_gene_share']:.0%})")

    print(f"\nDiffuse (high entropy, n={len(diffuse)}):")
    print(f"  Mean median OS: {diffuse['median_os_months'].mean():.1f} months")
    print(f"  Mean event rate: {diffuse['event_rate'].mean():.1%}")
    print(f"  Mean H/P ratio: {diffuse['harmful_protective_ratio'].mean():.2f}")
    for _, r in diffuse.sort_values('channel_entropy_norm').iterrows():
        print(f"    {r['cancer_type']:35s} entropy={r['channel_entropy_norm']:.3f} "
              f"medOS={r['median_os_months']:.0f}m top1={r['top1_gene']}({r['top1_gene_share']:.0%})")

    # Mann-Whitney U test
    u, p = stats.mannwhitneyu(
        scripted['median_os_months'], diffuse['median_os_months'],
        alternative='less'
    )
    print(f"\nMann-Whitney U (scripted OS < diffuse OS): U={u:.0f}, p={p:.4f}")

    u2, p2 = stats.mannwhitneyu(
        scripted['event_rate'], diffuse['event_rate'],
        alternative='greater'
    )
    print(f"Mann-Whitney U (scripted events > diffuse events): U={u2:.0f}, p={p2:.4f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "escalation_entropy.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
