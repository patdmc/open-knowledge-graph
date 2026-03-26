"""
Channel Co-occurrence Graph Analysis
=====================================
Treats the 6 coupling channels as a graph and tests whether
graph-theoretic properties of severed channels predict survival
beyond simple channel count.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import ast
import os
from itertools import combinations
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__),
                         'results/msk_impact_50k_2026/patient_data.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results/channel_graph')
os.makedirs(OUT_DIR, exist_ok=True)

CHANNELS = ['DDR', 'CellCycle', 'PI3K_Growth', 'Endocrine', 'Immune', 'TissueArch']
CHANNEL_IDX = {ch: i for i, ch in enumerate(CHANNELS)}
N_CH = len(CHANNELS)

# ── Load & parse ───────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['channels_set'] = df['channels_severed'].apply(ast.literal_eval)
print(f"  {len(df):,} patients loaded")

# =====================================================================
# 1. BUILD CHANNEL CO-OCCURRENCE GRAPH
# =====================================================================
print("\n=== Step 1: Channel co-occurrence graph ===")

# Count how often each channel is severed
channel_counts = {ch: 0 for ch in CHANNELS}
pair_counts = {(a, b): 0 for a, b in combinations(CHANNELS, 2)}
either_counts = {(a, b): 0 for a, b in combinations(CHANNELS, 2)}

for s in df['channels_set']:
    for ch in CHANNELS:
        if ch in s:
            channel_counts[ch] += 1
    for a, b in combinations(CHANNELS, 2):
        a_in = a in s
        b_in = b in s
        if a_in or b_in:
            either_counts[(a, b)] += 1
        if a_in and b_in:
            pair_counts[(a, b)] += 1

print("\nChannel frequencies:")
for ch in CHANNELS:
    print(f"  {ch}: {channel_counts[ch]:,} ({100*channel_counts[ch]/len(df):.1f}%)")

# Jaccard-style co-occurrence: P(both | either)
cooccurrence = np.zeros((N_CH, N_CH))
for (a, b), cnt in pair_counts.items():
    i, j = CHANNEL_IDX[a], CHANNEL_IDX[b]
    val = cnt / either_counts[(a, b)] if either_counts[(a, b)] > 0 else 0
    cooccurrence[i, j] = val
    cooccurrence[j, i] = val
np.fill_diagonal(cooccurrence, 1.0)

print("\nCo-occurrence matrix (P(both | either)):")
co_df = pd.DataFrame(cooccurrence, index=CHANNELS, columns=CHANNELS)
print(co_df.round(3).to_string())

# ── Heatmap ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cooccurrence, cmap='YlOrRd', vmin=0, vmax=cooccurrence[np.triu_indices(N_CH, k=1)].max() * 1.1)
ax.set_xticks(range(N_CH))
ax.set_yticks(range(N_CH))
ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(CHANNELS, fontsize=10)
for i in range(N_CH):
    for j in range(N_CH):
        if i != j:
            ax.text(j, i, f'{cooccurrence[i,j]:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if cooccurrence[i, j] > 0.3 else 'black')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('P(both severed | either severed)', fontsize=10)
ax.set_title('Channel Co-occurrence Graph\n(Jaccard-style co-severance)', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'channel_cooccurrence.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'channel_cooccurrence.pdf'))
plt.close(fig)
print("  Saved channel_cooccurrence.png/pdf")

# ── Network visualization (pure matplotlib) ────────────────────────────
import networkx as nx

G = nx.Graph()
for ch in CHANNELS:
    G.add_node(ch)
for (a, b), cnt in pair_counts.items():
    i, j = CHANNEL_IDX[a], CHANNEL_IDX[b]
    w = cooccurrence[i, j]
    if w > 0:
        G.add_edge(a, b, weight=w)

pos = nx.spring_layout(G, seed=42, k=2.5)

fig, ax = plt.subplots(figsize=(8, 7))
# Draw edges with width proportional to weight
edges = G.edges(data=True)
weights = [d['weight'] for _, _, d in edges]
max_w = max(weights) if weights else 1
for (u, v, d) in edges:
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    lw = 0.5 + 6 * (d['weight'] / max_w)
    alpha = 0.3 + 0.7 * (d['weight'] / max_w)
    ax.plot(x, y, '-', color='steelblue', linewidth=lw, alpha=alpha, zorder=1)
    mid_x, mid_y = (x[0]+x[1])/2, (y[0]+y[1])/2
    ax.text(mid_x, mid_y, f"{d['weight']:.2f}", fontsize=7, ha='center', va='center',
            color='navy', alpha=0.8)

# Draw nodes
node_sizes = [channel_counts[ch] / len(df) * 4000 + 300 for ch in CHANNELS]
for i, ch in enumerate(CHANNELS):
    ax.scatter(pos[ch][0], pos[ch][1], s=node_sizes[i], c='coral',
               edgecolors='darkred', linewidths=1.5, zorder=2)
    ax.text(pos[ch][0], pos[ch][1] + 0.12, ch, fontsize=10, ha='center',
            fontweight='bold', zorder=3)

ax.set_title('Channel Co-occurrence Network\n(edge width ~ co-occurrence strength, node size ~ frequency)',
             fontsize=12, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'channel_network.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'channel_network.pdf'))
plt.close(fig)
print("  Saved channel_network.png/pdf")

# =====================================================================
# 2. PER-PATIENT GRAPH PROPERTIES
# =====================================================================
print("\n=== Step 2: Per-patient graph properties ===")

# Build thresholded adjacency for shortest-path computation
threshold = np.median(cooccurrence[np.triu_indices(N_CH, k=1)])
adj_binary = (cooccurrence > threshold).astype(float)
np.fill_diagonal(adj_binary, 0)
G_thresh = nx.from_numpy_array(adj_binary)
# Relabel nodes
mapping = {i: ch for i, ch in enumerate(CHANNELS)}
G_thresh = nx.relabel_nodes(G_thresh, mapping)

# Compute shortest paths (using inverse weight for distance on full graph)
# For distance: use 1/weight so high co-occurrence = short distance
G_dist = nx.Graph()
for (a, b), cnt in pair_counts.items():
    i, j = CHANNEL_IDX[a], CHANNEL_IDX[b]
    w = cooccurrence[i, j]
    if w > 0:
        G_dist.add_edge(a, b, weight=1.0/w)
for ch in CHANNELS:
    if ch not in G_dist:
        G_dist.add_node(ch)

def patient_graph_props(channels_set):
    chs = [c for c in channels_set if c in CHANNEL_IDX]
    if len(chs) < 2:
        return np.nan, np.nan
    # Mean co-occurrence weight
    pairs = list(combinations(chs, 2))
    weights = [cooccurrence[CHANNEL_IDX[a], CHANNEL_IDX[b]] for a, b in pairs]
    mean_w = np.mean(weights)
    # Mean shortest-path distance (in thresholded graph)
    dists = []
    for a, b in pairs:
        try:
            d = nx.shortest_path_length(G_dist, a, b, weight='weight')
            dists.append(d)
        except nx.NetworkXNoPath:
            dists.append(np.nan)
    mean_dist = np.nanmean(dists) if dists else np.nan
    return mean_w, mean_dist

props = df['channels_set'].apply(patient_graph_props)
df['mean_cooccurrence'] = props.apply(lambda x: x[0])
df['mean_graph_dist'] = props.apply(lambda x: x[1])

print(f"  Patients with 2+ channels: {df['mean_cooccurrence'].notna().sum():,}")
print(f"  Mean co-occurrence (2+ ch): {df.loc[df['mean_cooccurrence'].notna(), 'mean_cooccurrence'].mean():.4f}")
print(f"  Mean graph distance (2+ ch): {df.loc[df['mean_graph_dist'].notna(), 'mean_graph_dist'].mean():.2f}")

# =====================================================================
# 3. KM PLOT: HIGH vs LOW CO-OCCURRENCE AT EXACTLY 2 CHANNELS
# =====================================================================
print("\n=== Step 3: Graph independence KM analysis (2-channel patients) ===")

df2 = df[df['channel_count'] == 2].copy()
print(f"  Patients with exactly 2 channels severed: {len(df2):,}")

median_cooc = df2['mean_cooccurrence'].median()
df2['cooc_group'] = np.where(df2['mean_cooccurrence'] >= median_cooc, 'High co-occurrence\n(correlated channels)',
                             'Low co-occurrence\n(independent channels)')

# Log-rank test
high = df2[df2['mean_cooccurrence'] >= median_cooc]
low = df2[df2['mean_cooccurrence'] < median_cooc]

lr = logrank_test(high['os_months'], low['os_months'], high['event'], low['event'])
print(f"  Median co-occurrence split: {median_cooc:.4f}")
print(f"  High co-occurrence group: n={len(high):,}, events={high['event'].sum():,}")
print(f"  Low co-occurrence group:  n={len(low):,}, events={low['event'].sum():,}")
print(f"  Log-rank p-value: {lr.p_value:.2e}")
print(f"  Test statistic: {lr.test_statistic:.2f}")

# KM plot
fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()

colors = {'High co-occurrence\n(correlated channels)': '#2196F3',
          'Low co-occurrence\n(independent channels)': '#F44336'}

for label, grp in df2.groupby('cooc_group'):
    kmf.fit(grp['os_months'], grp['event'], label=label)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[label], linewidth=2)

ax.set_xlabel('Overall Survival (months)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title(f'Channel Independence and Survival\n(patients with exactly 2 channels severed, n={len(df2):,})',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')

# Add stats annotation
txt = f"Log-rank p = {lr.p_value:.2e}\nMedian co-occurrence split = {median_cooc:.3f}"
ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=9,
        va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'graph_independence_km.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'graph_independence_km.pdf'))
plt.close(fig)
print("  Saved graph_independence_km.png/pdf")

# =====================================================================
# 4. COX MODEL: CHANNEL COUNT + CO-OCCURRENCE + DRIVER MUTATIONS
# =====================================================================
print("\n=== Step 4: Cox proportional hazards model ===")

df_cox = df[df['mean_cooccurrence'].notna()].copy()
df_cox = df_cox[['os_months', 'event', 'channel_count', 'mean_cooccurrence',
                  'driver_mutation_count']].dropna()
# Remove zero-duration rows
df_cox = df_cox[df_cox['os_months'] > 0]

print(f"  Cox model cohort: {len(df_cox):,} patients (2+ channels, valid data)")

# Standardize for comparability (manual z-score to avoid sklearn dependency)
df_cox_scaled = df_cox.copy()
cols_to_scale = ['channel_count', 'mean_cooccurrence', 'driver_mutation_count']
for col in cols_to_scale:
    mu, sigma = df_cox[col].mean(), df_cox[col].std()
    df_cox_scaled[col] = (df_cox[col] - mu) / sigma

cph = CoxPHFitter()
cph.fit(df_cox_scaled, duration_col='os_months', event_col='event')
print("\nCox PH Model (standardized covariates):")
cph.print_summary()

# Also fit without co-occurrence to test added value
cph_base = CoxPHFitter()
df_base = df_cox_scaled[['os_months', 'event', 'channel_count', 'driver_mutation_count']].copy()
cph_base.fit(df_base, duration_col='os_months', event_col='event')

print(f"\n  Full model AIC:     {cph.AIC_partial_:.1f}")
print(f"  Base model AIC:     {cph_base.AIC_partial_:.1f}")
print(f"  Delta AIC:          {cph_base.AIC_partial_ - cph.AIC_partial_:.1f} (positive = co-occurrence helps)")

# Concordance
print(f"  Full model concordance:  {cph.concordance_index_:.4f}")
print(f"  Base model concordance:  {cph_base.concordance_index_:.4f}")

# =====================================================================
# 5. SPECIFIC CHANNEL PAIR ANALYSIS
# =====================================================================
print("\n=== Step 5: Specific channel pair survival analysis ===")

# Get patients with exactly 2 channels
df2_pairs = df[df['channel_count'] == 2].copy()
df2_pairs['pair'] = df2_pairs['channels_set'].apply(lambda s: tuple(sorted(s)))

pair_stats = []
all_pairs = list(combinations(CHANNELS, 2))
for pair in all_pairs:
    mask = df2_pairs['pair'] == pair
    sub = df2_pairs[mask]
    if len(sub) < 10:
        pair_stats.append({'pair': pair, 'n': len(sub), 'mortality_rate': np.nan,
                           'median_os': np.nan, 'cooccurrence': cooccurrence[CHANNEL_IDX[pair[0]], CHANNEL_IDX[pair[1]]]})
        continue
    mort_rate = sub['event'].mean()
    median_os = sub['os_months'].median()
    pair_stats.append({
        'pair': pair,
        'n': len(sub),
        'mortality_rate': mort_rate,
        'median_os': median_os,
        'cooccurrence': cooccurrence[CHANNEL_IDX[pair[0]], CHANNEL_IDX[pair[1]]]
    })

pair_df = pd.DataFrame(pair_stats)
pair_df = pair_df.sort_values('mortality_rate', ascending=False)

print("\nChannel pair survival statistics (sorted by mortality rate):")
print(f"{'Pair':<35} {'n':>6} {'Mortality':>10} {'Med OS':>8} {'Co-occ':>8}")
print("-" * 70)
for _, row in pair_df.iterrows():
    p = f"{row['pair'][0]} + {row['pair'][1]}"
    mort = f"{row['mortality_rate']:.3f}" if not np.isnan(row['mortality_rate']) else "N/A"
    med = f"{row['median_os']:.1f}" if not np.isnan(row['median_os']) else "N/A"
    print(f"  {p:<33} {row['n']:>6} {mort:>10} {med:>8} {row['cooccurrence']:>8.3f}")

# Correlation between co-occurrence and mortality
valid = pair_df.dropna(subset=['mortality_rate'])
if len(valid) >= 3:
    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(valid['cooccurrence'], valid['mortality_rate'])
    r_s, p_s = spearmanr(valid['cooccurrence'], valid['mortality_rate'])
    print(f"\n  Pearson r (co-occurrence vs mortality): {r_p:.3f}, p={p_p:.4f}")
    print(f"  Spearman rho: {r_s:.3f}, p={p_s:.4f}")

# ── Heatmap of mortality rate by channel pair ──────────────────────────
mort_matrix = np.full((N_CH, N_CH), np.nan)
n_matrix = np.zeros((N_CH, N_CH), dtype=int)
for _, row in pair_df.iterrows():
    i, j = CHANNEL_IDX[row['pair'][0]], CHANNEL_IDX[row['pair'][1]]
    mort_matrix[i, j] = row['mortality_rate']
    mort_matrix[j, i] = row['mortality_rate']
    n_matrix[i, j] = row['n']
    n_matrix[j, i] = row['n']

fig, ax = plt.subplots(figsize=(9, 7))
masked = np.ma.masked_invalid(mort_matrix)
im = ax.imshow(masked, cmap='RdYlGn_r', vmin=masked.min() * 0.95 if masked.count() > 0 else 0,
               vmax=masked.max() * 1.05 if masked.count() > 0 else 1)
ax.set_xticks(range(N_CH))
ax.set_yticks(range(N_CH))
ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(CHANNELS, fontsize=10)
for i in range(N_CH):
    for j in range(N_CH):
        if i != j and not np.isnan(mort_matrix[i, j]):
            ax.text(j, i, f'{mort_matrix[i,j]:.2f}\n(n={n_matrix[i,j]})',
                    ha='center', va='center', fontsize=8,
                    color='white' if mort_matrix[i, j] > 0.55 else 'black')
        elif i == j:
            ax.text(j, i, '---', ha='center', va='center', fontsize=9, color='gray')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Mortality Rate', fontsize=10)
ax.set_title('Mortality Rate by Severed Channel Pair\n(patients with exactly 2 channels severed)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'channel_pair_survival.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'channel_pair_survival.pdf'))
plt.close(fig)
print("  Saved channel_pair_survival.png/pdf")

# ── Median OS heatmap by channel pair ──────────────────────────────────
os_matrix = np.full((N_CH, N_CH), np.nan)
for _, row in pair_df.iterrows():
    i, j = CHANNEL_IDX[row['pair'][0]], CHANNEL_IDX[row['pair'][1]]
    os_matrix[i, j] = row['median_os']
    os_matrix[j, i] = row['median_os']

fig, ax = plt.subplots(figsize=(9, 7))
masked_os = np.ma.masked_invalid(os_matrix)
im = ax.imshow(masked_os, cmap='RdYlGn',
               vmin=masked_os.min() * 0.95 if masked_os.count() > 0 else 0,
               vmax=masked_os.max() * 1.05 if masked_os.count() > 0 else 100)
ax.set_xticks(range(N_CH))
ax.set_yticks(range(N_CH))
ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(CHANNELS, fontsize=10)
for i in range(N_CH):
    for j in range(N_CH):
        if i != j and not np.isnan(os_matrix[i, j]):
            ax.text(j, i, f'{os_matrix[i,j]:.0f}mo\n(n={n_matrix[i,j]})',
                    ha='center', va='center', fontsize=8,
                    color='white' if os_matrix[i, j] < 40 else 'black')
        elif i == j:
            ax.text(j, i, '---', ha='center', va='center', fontsize=9, color='gray')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Median OS (months)', fontsize=10)
ax.set_title('Median Overall Survival by Severed Channel Pair\n(patients with exactly 2 channels severed)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'channel_pair_median_os.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'channel_pair_median_os.pdf'))
plt.close(fig)
print("  Saved channel_pair_median_os.png/pdf")

# =====================================================================
# 6. SUMMARY
# =====================================================================
print("\n=== Writing summary ===")

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("CHANNEL CO-OCCURRENCE GRAPH ANALYSIS — SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append(f"\nDataset: {len(df):,} patients from MSK-IMPACT")
summary_lines.append(f"Patients with 2+ channels: {df['mean_cooccurrence'].notna().sum():,}")
summary_lines.append(f"Patients with exactly 2 channels: {len(df2):,}")

summary_lines.append("\n--- Channel Frequencies ---")
for ch in CHANNELS:
    summary_lines.append(f"  {ch}: {channel_counts[ch]:,} ({100*channel_counts[ch]/len(df):.1f}%)")

summary_lines.append("\n--- Co-occurrence Matrix (P(both | either)) ---")
summary_lines.append(co_df.round(3).to_string())

# Top/bottom pairs
summary_lines.append("\n--- Channel Pair Survival (sorted by mortality) ---")
summary_lines.append(f"{'Pair':<35} {'n':>6} {'Mortality':>10} {'Med OS':>8} {'Co-occ':>8}")
summary_lines.append("-" * 70)
for _, row in pair_df.iterrows():
    p = f"{row['pair'][0]} + {row['pair'][1]}"
    mort = f"{row['mortality_rate']:.3f}" if not np.isnan(row['mortality_rate']) else "N/A"
    med = f"{row['median_os']:.1f}" if not np.isnan(row['median_os']) else "N/A"
    summary_lines.append(f"  {p:<33} {row['n']:>6} {mort:>10} {med:>8} {row['cooccurrence']:>8.3f}")

if len(valid) >= 3:
    summary_lines.append(f"\n  Pearson r (co-occurrence vs mortality): {r_p:.3f}, p={p_p:.4f}")
    summary_lines.append(f"  Spearman rho (co-occurrence vs mortality): {r_s:.3f}, p={p_s:.4f}")

summary_lines.append("\n--- Graph Independence KM Test (2-channel patients) ---")
summary_lines.append(f"  Median co-occurrence split: {median_cooc:.4f}")
summary_lines.append(f"  High co-occurrence: n={len(high):,}, events={high['event'].sum():,}")
summary_lines.append(f"  Low co-occurrence:  n={len(low):,}, events={low['event'].sum():,}")
summary_lines.append(f"  Log-rank test statistic: {lr.test_statistic:.2f}")
summary_lines.append(f"  Log-rank p-value: {lr.p_value:.2e}")

# Median survival for each group
kmf_h = KaplanMeierFitter()
kmf_h.fit(high['os_months'], high['event'])
kmf_l = KaplanMeierFitter()
kmf_l.fit(low['os_months'], low['event'])
med_h = kmf_h.median_survival_time_
med_l = kmf_l.median_survival_time_
summary_lines.append(f"  Median survival (high co-occurrence): {med_h:.1f} months")
summary_lines.append(f"  Median survival (low co-occurrence):  {med_l:.1f} months")

summary_lines.append("\n--- Cox PH Model (2+ channels, standardized) ---")
summary_lines.append(f"  Covariates: channel_count, mean_cooccurrence, driver_mutation_count")
for var in ['channel_count', 'mean_cooccurrence', 'driver_mutation_count']:
    row_cox = cph.summary.loc[var]
    summary_lines.append(f"  {var}:")
    summary_lines.append(f"    HR = {row_cox['exp(coef)']:.3f} (95% CI: {row_cox['exp(coef) lower 95%']:.3f}-{row_cox['exp(coef) upper 95%']:.3f})")
    summary_lines.append(f"    p = {row_cox['p']:.2e}")

summary_lines.append(f"\n  Full model concordance:  {cph.concordance_index_:.4f}")
summary_lines.append(f"  Base model concordance:  {cph_base.concordance_index_:.4f}")
summary_lines.append(f"  Full model AIC:  {cph.AIC_partial_:.1f}")
summary_lines.append(f"  Base model AIC:  {cph_base.AIC_partial_:.1f}")
summary_lines.append(f"  Delta AIC: {cph_base.AIC_partial_ - cph.AIC_partial_:.1f}")

summary_lines.append("\n--- Key Findings ---")
if lr.p_value < 0.05:
    if med_l < med_h:
        summary_lines.append("  1. Patients with LOW co-occurrence (independent channels) have")
        summary_lines.append("     significantly WORSE survival than those with high co-occurrence.")
        summary_lines.append("     This supports the hypothesis that severing functionally independent")
        summary_lines.append("     channels removes more distinct organizational capacity.")
    else:
        summary_lines.append("  1. Patients with HIGH co-occurrence (correlated channels) have")
        summary_lines.append("     worse survival. This may indicate that correlated channel severance")
        summary_lines.append("     reflects more aggressive underlying biology.")
else:
    summary_lines.append("  1. No significant difference in survival between high and low")
    summary_lines.append("     co-occurrence groups at the 2-channel level.")

cooc_p = cph.summary.loc['mean_cooccurrence', 'p']
cooc_hr = cph.summary.loc['mean_cooccurrence', 'exp(coef)']
if cooc_p < 0.05:
    direction = "protective" if cooc_hr < 1 else "risk-increasing"
    summary_lines.append(f"  2. Co-occurrence weight is a significant predictor in the Cox model")
    summary_lines.append(f"     (HR={cooc_hr:.3f}, p={cooc_p:.2e}), {direction} beyond channel count.")
else:
    summary_lines.append(f"  2. Co-occurrence weight is NOT significant in the Cox model")
    summary_lines.append(f"     (HR={cooc_hr:.3f}, p={cooc_p:.2e}) after controlling for count.")

if len(valid) >= 3:
    if p_s < 0.05:
        direction = "lower" if r_s < 0 else "higher"
        summary_lines.append(f"  3. Channel pairs with higher co-occurrence have {direction} mortality")
        summary_lines.append(f"     (Spearman rho={r_s:.3f}, p={p_s:.4f}).")
    else:
        summary_lines.append(f"  3. No significant correlation between pair co-occurrence and mortality")
        summary_lines.append(f"     at the individual pair level (rho={r_s:.3f}, p={p_s:.4f}).")

summary_lines.append("\n" + "=" * 70)
summary_lines.append("Output files:")
summary_lines.append("  channel_cooccurrence.png/pdf  — Co-occurrence heatmap")
summary_lines.append("  channel_network.png/pdf       — Network visualization")
summary_lines.append("  graph_independence_km.png/pdf  — KM: high vs low co-occurrence")
summary_lines.append("  channel_pair_survival.png/pdf  — Mortality rate by pair")
summary_lines.append("  channel_pair_median_os.png/pdf — Median OS by pair")
summary_lines.append("=" * 70)

summary_text = "\n".join(summary_lines)
with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
    f.write(summary_text)

print(summary_text)
print("\nDone. All outputs saved to:", OUT_DIR)
