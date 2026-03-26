"""
Channel discovery: let co-mutation patterns find gene groupings.

Instead of declaring biological channels a priori, cluster genes by
Jaccard similarity of their co-mutation profiles, cut at various k,
and test whether the data-driven cluster count predicts survival.
Compare to the hand-picked 6-channel mapping.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
def adjusted_rand_score(labels_true, labels_pred):
    """Manual ARI to avoid sklearn dependency."""
    from collections import Counter
    pairs = Counter(zip(labels_true, labels_pred))
    a_c = Counter(labels_true)
    b_c = Counter(labels_pred)
    n = len(labels_true)
    c2 = lambda x: x * (x - 1) / 2
    s_nij = sum(c2(v) for v in pairs.values())
    s_ai = sum(c2(v) for v in a_c.values())
    s_bi = sum(c2(v) for v in b_c.values())
    cn = c2(n)
    if cn == 0: return 0.0
    exp = s_ai * s_bi / cn
    mx = (s_ai + s_bi) / 2
    if mx == exp: return 1.0
    return (s_nij - exp) / (mx - exp)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from channel_analysis import CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_COLORS

# ============================================================
# Paths
# ============================================================
CACHE = os.path.join(os.path.dirname(__file__), 'cache')
OUT = os.path.join(os.path.dirname(__file__), 'results', 'channel_discovery')
os.makedirs(OUT, exist_ok=True)

MUT_FILE = os.path.join(CACHE, 'msk_impact_50k_2026_mutations.csv')
CLIN_FILE = os.path.join(CACHE, 'msk_impact_50k_2026_clinical.csv')

DRIVER_TYPES = [
    'Missense_Mutation', 'Nonsense_Mutation',
    'Frame_Shift_Del', 'Frame_Shift_Ins',
    'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
]

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 10]

# ============================================================
# 1. Load data and build gene co-mutation matrix
# ============================================================
print("Loading mutations...")
mut = pd.read_csv(MUT_FILE, usecols=['patientId', 'gene.hugoGeneSymbol', 'mutationType'])
mut = mut[mut['mutationType'].isin(DRIVER_TYPES)]

channel_genes = set(CHANNEL_MAP.keys())
mut = mut[mut['gene.hugoGeneSymbol'].isin(channel_genes)]

print("Loading clinical data...")
clin = pd.read_csv(CLIN_FILE, usecols=['patientId', 'OS_MONTHS', 'OS_STATUS'])
clin = clin.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
clin['event'] = clin['OS_STATUS'].apply(lambda x: 1 if '1:' in str(x) or 'DECEASED' in str(x) else 0)
clin = clin[clin['OS_MONTHS'] > 0]

# Patient x gene binary matrix
print("Building patient x gene binary matrix...")
mut_dedup = mut[['patientId', 'gene.hugoGeneSymbol']].drop_duplicates()
patient_gene = mut_dedup.assign(val=1).pivot_table(
    index='patientId', columns='gene.hugoGeneSymbol', values='val',
    fill_value=0, aggfunc='max'
)

# Filter genes with >= 100 patients mutated
gene_counts = patient_gene.sum()
keep_genes = gene_counts[gene_counts >= 100].index.tolist()
patient_gene = patient_gene[keep_genes]

print(f"Genes in CHANNEL_MAP: {len(channel_genes)}")
print(f"Genes with >= 100 mutated patients: {len(keep_genes)}")
dropped = channel_genes - set(keep_genes)
if dropped:
    print(f"Dropped (< 100 mutations): {sorted(dropped)}")

# ============================================================
# 2. Jaccard distance and hierarchical clustering
# ============================================================
print("\nComputing Jaccard distances...")
gene_matrix = patient_gene.values.T  # genes x patients
dist_vec = pdist(gene_matrix, metric='jaccard')
dist_mat = squareform(dist_vec)

print("Running hierarchical clustering (average linkage)...")
Z = linkage(dist_vec, method='average')

gene_names = list(patient_gene.columns)

# Build biological channel labels for the kept genes
bio_labels = [CHANNEL_MAP.get(g, 'Unknown') for g in gene_names]

# ============================================================
# 3. For each k, test cluster count as survival predictor
# ============================================================
print("\n" + "=" * 60)
print("SURVIVAL ANALYSIS: data-driven clusters")
print("=" * 60)

results = []

for k in K_VALUES:
    clusters = fcluster(Z, t=k, criterion='maxclust')
    gene_to_cluster = dict(zip(gene_names, clusters))

    # For each patient, count distinct clusters with a mutation
    cluster_cols = []
    for g in gene_names:
        c = gene_to_cluster[g]
        cluster_cols.append(f'c{c}')

    # Build patient-level data
    pg = patient_gene.copy()
    pg.columns = cluster_cols

    # For each patient, which clusters are hit?
    cluster_ids = sorted(set(cluster_cols))
    patient_cluster_hit = pd.DataFrame(index=pg.index)
    for cid in cluster_ids:
        cols_for_cluster = [c for c in pg.columns if c == cid]
        # Sum across all genes in this cluster, binarize
        patient_cluster_hit[cid] = (pg[cols_for_cluster].sum(axis=1) > 0).astype(int)

    # Actually, simpler: for each patient, count number of distinct clusters hit
    # Rebuild properly
    patient_cluster_count = pd.Series(0, index=patient_gene.index)
    for clust_id in sorted(set(clusters)):
        genes_in_cluster = [g for g, c in gene_to_cluster.items() if c == clust_id]
        hit = (patient_gene[genes_in_cluster].sum(axis=1) > 0).astype(int)
        patient_cluster_count += hit

    # Total mutation count
    patient_mut_count = patient_gene.sum(axis=1)

    # Build survival dataframe
    surv = pd.DataFrame({
        'patientId': patient_gene.index,
        'cluster_count': patient_cluster_count.values,
        'mutation_count': patient_mut_count.values,
    }).merge(clin[['patientId', 'OS_MONTHS', 'event']], on='patientId', how='inner')

    surv['log_mutation_count'] = np.log(surv['mutation_count'] + 1)

    # Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(
            surv[['OS_MONTHS', 'event', 'cluster_count', 'log_mutation_count']],
            duration_col='OS_MONTHS', event_col='event'
        )
        cc_hr = np.exp(cph.params_['cluster_count'])
        cc_p = cph.summary.loc['cluster_count', 'p']
        mc_hr = np.exp(cph.params_['log_mutation_count'])
        mc_p = cph.summary.loc['log_mutation_count', 'p']
    except Exception as e:
        print(f"  k={k}: Cox failed: {e}")
        cc_hr, cc_p, mc_hr, mc_p = np.nan, np.nan, np.nan, np.nan

    results.append({
        'k': k, 'cc_hr': cc_hr, 'cc_p': cc_p, 'mc_hr': mc_hr, 'mc_p': mc_p
    })

    print(f"  k={k:2d}: cluster_count HR={cc_hr:.4f} (p={cc_p:.2e})"
          f"  |  log_mut HR={mc_hr:.4f} (p={mc_p:.2e})")

results_df = pd.DataFrame(results)

# ============================================================
# 4. Hand-picked channel comparison
# ============================================================
print("\n" + "=" * 60)
print("HAND-PICKED 6-CHANNEL COMPARISON")
print("=" * 60)

# Hand-picked channel count for each patient
patient_handpicked_count = pd.Series(0, index=patient_gene.index)
for ch_name in set(CHANNEL_MAP.values()):
    genes_in_ch = [g for g in gene_names if CHANNEL_MAP.get(g) == ch_name]
    if genes_in_ch:
        hit = (patient_gene[genes_in_ch].sum(axis=1) > 0).astype(int)
        patient_handpicked_count += hit

patient_mut_count = patient_gene.sum(axis=1)

surv_hp = pd.DataFrame({
    'patientId': patient_gene.index,
    'channel_count': patient_handpicked_count.values,
    'mutation_count': patient_mut_count.values,
}).merge(clin[['patientId', 'OS_MONTHS', 'event']], on='patientId', how='inner')

surv_hp['log_mutation_count'] = np.log(surv_hp['mutation_count'] + 1)

cph_hp = CoxPHFitter()
cph_hp.fit(
    surv_hp[['OS_MONTHS', 'event', 'channel_count', 'log_mutation_count']],
    duration_col='OS_MONTHS', event_col='event'
)
hp_hr = np.exp(cph_hp.params_['channel_count'])
hp_p = cph_hp.summary.loc['channel_count', 'p']
hp_mc_hr = np.exp(cph_hp.params_['log_mutation_count'])
hp_mc_p = cph_hp.summary.loc['log_mutation_count', 'p']

print(f"  Hand-picked (k=6): channel_count HR={hp_hr:.4f} (p={hp_p:.2e})"
      f"  |  log_mut HR={hp_mc_hr:.4f} (p={hp_mc_p:.2e})")

# Compare data-driven k=6 vs hand-picked k=6
dd_k6 = results_df[results_df['k'] == 6].iloc[0]
print(f"  Data-driven (k=6): cluster_count HR={dd_k6['cc_hr']:.4f} (p={dd_k6['cc_p']:.2e})"
      f"  |  log_mut HR={dd_k6['mc_hr']:.4f} (p={dd_k6['mc_p']:.2e})")

# Adjusted Rand Index between data-driven k=6 and hand-picked
clusters_k6 = fcluster(Z, t=6, criterion='maxclust')
dd_labels = list(clusters_k6)
hp_labels = [CHANNEL_MAP.get(g, 'Unknown') for g in gene_names]

ari = adjusted_rand_score(hp_labels, dd_labels)
print(f"\n  Adjusted Rand Index (data-driven k=6 vs hand-picked): {ari:.4f}")

# ============================================================
# 5. Optimal k
# ============================================================
print("\n" + "=" * 60)
print("OPTIMAL k")
print("=" * 60)

best_row = results_df.loc[results_df['cc_p'].idxmin()]
print(f"  Strongest cluster_count signal: k={int(best_row['k'])}"
      f"  HR={best_row['cc_hr']:.4f}  p={best_row['cc_p']:.2e}")

# Reference: sensitivity analysis peaked at k=3 for MSK-IMPACT 50K
sens_file = os.path.join(os.path.dirname(__file__), 'results', 'sensitivity', 'channel_sensitivity.csv')
if os.path.exists(sens_file):
    sens = pd.read_csv(sens_file)
    sens_50k = sens[sens['study'] == 'MSK-IMPACT 50K']
    if len(sens_50k) > 0:
        best_sens = sens_50k.loc[sens_50k['ch_multi_p'].idxmin()]
        print(f"  Sensitivity analysis (hand-picked) peaked at: k={int(best_sens['n_channels'])}"
              f"  HR={best_sens['ch_multi_hr']:.4f}  p={best_sens['ch_multi_p']:.2e}")

# ============================================================
# 6. Visualize
# ============================================================

# --- Dendrogram colored by biological channel ---
print("\nGenerating dendrogram...")

channel_color_map = {}
for ch_key, color in CHANNEL_COLORS.items():
    channel_color_map[ch_key] = color
channel_color_map['Unknown'] = '#999999'

leaf_colors = {}
for i, g in enumerate(gene_names):
    ch = CHANNEL_MAP.get(g, 'Unknown')
    leaf_colors[g] = channel_color_map.get(ch, '#999999')

fig, ax = plt.subplots(figsize=(16, 8))

# Use default dendrogram coloring, then we'll label by channel
dendro = dendrogram(
    Z, labels=gene_names, leaf_rotation=90, leaf_font_size=7,
    ax=ax, color_threshold=0, above_threshold_color='#333333'
)

# Color the x-axis labels by biological channel
xlabels = ax.get_xticklabels()
for label in xlabels:
    gene = label.get_text()
    ch = CHANNEL_MAP.get(gene, 'Unknown')
    label.set_color(channel_color_map.get(ch, '#999999'))
    label.set_fontweight('bold')

# Add legend for channels
legend_patches = []
for ch_key in ['DDR', 'CellCycle', 'PI3K_Growth', 'Endocrine', 'Immune', 'TissueArch']:
    if ch_key in CHANNEL_NAMES:
        legend_patches.append(
            mpatches.Patch(color=CHANNEL_COLORS[ch_key], label=CHANNEL_NAMES[ch_key])
        )
ax.legend(handles=legend_patches, loc='upper right', fontsize=8, title='Biological Channel')

ax.set_title('Gene Clustering by Co-Mutation (Jaccard Distance, Average Linkage)\n'
             'X-axis labels colored by hand-picked biological channel', fontsize=12)
ax.set_ylabel('Jaccard Distance')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'dendrogram.png'), dpi=150)
fig.savefig(os.path.join(OUT, 'dendrogram.pdf'))
plt.close(fig)

# --- HR vs k plot ---
print("Generating HR vs k plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: HR
ax1.plot(results_df['k'], results_df['cc_hr'], 'o-', color='#2980b9',
         linewidth=2, markersize=8, label='Data-driven clusters')

# Overlay hand-picked result at k=6
ax1.plot(6, hp_hr, 's', color='#e74c3c', markersize=12, zorder=5,
         label=f'Hand-picked channels (HR={hp_hr:.3f})')

# Overlay sensitivity analysis hand-picked results if available
if os.path.exists(sens_file):
    sens = pd.read_csv(sens_file)
    sens_50k = sens[sens['study'] == 'MSK-IMPACT 50K']
    if len(sens_50k) > 0:
        ax1.plot(sens_50k['n_channels'], sens_50k['ch_multi_hr'], 'D--',
                 color='#e67e22', linewidth=1.5, markersize=6,
                 label='Sensitivity (hand-picked)', alpha=0.8)

ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Hazard Ratio (cluster/channel count)')
ax1.set_title('Cluster Count HR vs k')
ax1.legend(fontsize=8)
ax1.set_xticks(K_VALUES)
ax1.axhline(1.0, color='grey', linestyle=':', alpha=0.5)

# Right panel: -log10(p)
ax2.plot(results_df['k'], -np.log10(results_df['cc_p']), 'o-', color='#2980b9',
         linewidth=2, markersize=8, label='Data-driven clusters')
ax2.plot(6, -np.log10(hp_p), 's', color='#e74c3c', markersize=12, zorder=5,
         label='Hand-picked channels')

if os.path.exists(sens_file) and len(sens_50k) > 0:
    ax2.plot(sens_50k['n_channels'], -np.log10(sens_50k['ch_multi_p']), 'D--',
             color='#e67e22', linewidth=1.5, markersize=6,
             label='Sensitivity (hand-picked)', alpha=0.8)

ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('-log10(p-value)')
ax2.set_title('Statistical Significance vs k')
ax2.legend(fontsize=8)
ax2.set_xticks(K_VALUES)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'hr_vs_k.png'), dpi=150)
fig.savefig(os.path.join(OUT, 'hr_vs_k.pdf'))
plt.close(fig)

# ============================================================
# 7. Save results
# ============================================================

# Cluster-gene mapping
print("\nSaving cluster-gene mapping...")
with open(os.path.join(OUT, 'cluster_gene_mapping.txt'), 'w') as f:
    for k in K_VALUES:
        clusters = fcluster(Z, t=k, criterion='maxclust')
        gene_to_cluster = dict(zip(gene_names, clusters))
        f.write(f"\n{'=' * 60}\n")
        f.write(f"k = {k}\n")
        f.write(f"{'=' * 60}\n")
        for clust_id in sorted(set(clusters)):
            genes_in = sorted([g for g, c in gene_to_cluster.items() if c == clust_id])
            channels_in = [CHANNEL_MAP.get(g, '?') for g in genes_in]
            f.write(f"\n  Cluster {clust_id} ({len(genes_in)} genes):\n")
            for g, ch in zip(genes_in, channels_in):
                f.write(f"    {g:15s}  [{ch}]\n")

# Summary
print("Saving summary...")
with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("CHANNEL DISCOVERY: Data-Driven Gene Groupings\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset: MSK-IMPACT 50K\n")
    f.write(f"Genes in CHANNEL_MAP: {len(channel_genes)}\n")
    f.write(f"Genes with >= 100 mutated patients: {len(keep_genes)}\n")
    if dropped:
        f.write(f"Dropped (< 100 mutations): {sorted(dropped)}\n")
    f.write(f"\nClustering: Jaccard distance, average linkage\n")
    f.write(f"Survival model: Cox PH, cluster_count + log(mutation_count + 1)\n\n")

    f.write("-" * 60 + "\n")
    f.write("DATA-DRIVEN CLUSTERING RESULTS\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'k':>3s}  {'CC HR':>8s}  {'CC p-val':>12s}  {'Mut HR':>8s}  {'Mut p-val':>12s}\n")
    for _, row in results_df.iterrows():
        f.write(f"{int(row['k']):3d}  {row['cc_hr']:8.4f}  {row['cc_p']:12.2e}  "
                f"{row['mc_hr']:8.4f}  {row['mc_p']:12.2e}\n")

    f.write(f"\nOptimal k (lowest p): k={int(best_row['k'])}"
            f"  HR={best_row['cc_hr']:.4f}  p={best_row['cc_p']:.2e}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("HAND-PICKED vs DATA-DRIVEN (k=6)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Hand-picked:  channel_count HR={hp_hr:.4f} (p={hp_p:.2e})\n")
    f.write(f"Data-driven:  cluster_count HR={dd_k6['cc_hr']:.4f} (p={dd_k6['cc_p']:.2e})\n")
    f.write(f"\nAdjusted Rand Index: {ari:.4f}\n")
    f.write("(1.0 = identical, 0.0 = random, can be negative)\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("DATA-DRIVEN k=6 CLUSTER COMPOSITION\n")
    f.write("-" * 60 + "\n")
    clusters_k6 = fcluster(Z, t=6, criterion='maxclust')
    gene_to_cluster_k6 = dict(zip(gene_names, clusters_k6))
    for clust_id in sorted(set(clusters_k6)):
        genes_in = sorted([g for g, c in gene_to_cluster_k6.items() if c == clust_id])
        channels_in = [CHANNEL_MAP.get(g, '?') for g in genes_in]
        channel_summary = {}
        for ch in channels_in:
            channel_summary[ch] = channel_summary.get(ch, 0) + 1
        f.write(f"\n  Cluster {clust_id} ({len(genes_in)} genes):\n")
        f.write(f"    Channel composition: {dict(sorted(channel_summary.items(), key=lambda x: -x[1]))}\n")
        for g, ch in zip(genes_in, channels_in):
            f.write(f"    {g:15s}  [{ch}]\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("INTERPRETATION\n")
    f.write("-" * 60 + "\n")
    if best_row['cc_p'] < 0.05:
        f.write("Data-driven clustering produces significant survival signal.\n")
    if ari > 0.3:
        f.write(f"ARI={ari:.3f} indicates substantial overlap with hand-picked channels.\n")
        f.write("The biological channel mapping is NOT arbitrary -- data-driven\n")
        f.write("clustering recovers similar groupings.\n")
    elif ari > 0.1:
        f.write(f"ARI={ari:.3f} indicates moderate overlap with hand-picked channels.\n")
        f.write("Some biological groupings are recovered by co-mutation patterns.\n")
    else:
        f.write(f"ARI={ari:.3f} indicates limited overlap with hand-picked channels.\n")
        f.write("Co-mutation patterns suggest different groupings than biology alone.\n")

print("\n" + "=" * 60)
print("DONE. Results saved to analysis/results/channel_discovery/")
print("=" * 60)
