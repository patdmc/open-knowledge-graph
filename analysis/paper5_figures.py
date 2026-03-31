#!/usr/bin/env python3
"""Generate all publication-quality figures for McCarthy2026_5_Cancer.tex.

Figures:
  1. km_by_channel_count.pdf         — KM curves by channel count
  2. severance_hierarchy.pdf          — Grouped bar: channel % at each stratum
  3. channel_frequency_by_ct.pdf      — Heatmap: channel × cancer type
  4. channel_coseverance.pdf          — Conditional P(col|row) matrix
  5. sl_exclusivity_by_ct.pdf         — SL mutual exclusivity by cancer type

All use 8-channel / 122-gene curated map from data/channel_gene_map.csv.
Data source: MSK-IMPACT 50K (analysis/cache/).
"""

import csv as _csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
CACHE = Path(__file__).resolve().parent / "cache"
FIGS  = Path(__file__).resolve().parent.parent / "figures"
FIGS.mkdir(exist_ok=True)

MUT_FILE = CACHE / "msk_impact_50k_2026_mutations.csv"
CLI_FILE = CACHE / "msk_impact_50k_2026_clinical.csv"

# ── Channel map (loaded from CSV) ─────────────────────────────────
def _load_channel_map():
    _csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "channel_gene_map.csv")
    result = {}
    with open(_csv_path) as f:
        for row in _csv.DictReader(f):
            ch = row["channel"]
            if ch == "TissueArchitecture":
                ch = "TissueArch"
            result[row["gene"]] = ch
    return result


CHANNEL_MAP = _load_channel_map()

CHANNEL_NAMES = {
    "PI3K_Growth": "PI3K / Growth",
    "CellCycle": "Cell Cycle",
    "ChromatinRemodel": "Chromatin Remodel",
    "TissueArch": "Tissue Arch",
    "DDR": "DDR",
    "Endocrine": "Endocrine",
    "Immune": "Immune",
    "DNAMethylation": "DNA Methylation",
}

# Display order: by tier then frequency
CHANNEL_ORDER = [
    "PI3K_Growth", "CellCycle",          # Tier 0
    "DDR", "TissueArch",                 # Tier 1
    "Endocrine", "Immune",               # Tier 2
    "ChromatinRemodel", "DNAMethylation", # Tier 3
]

NON_SILENT = {
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "Splice_Region",
    "Nonstop_Mutation", "Translation_Start_Site",
}

# ── 8-color palette (colorblind-friendly) ─────────────────────────
CHANNEL_COLORS = {
    "PI3K_Growth":      "#0072B2",
    "CellCycle":        "#E69F00",
    "DDR":              "#CC79A7",
    "TissueArch":       "#009E73",
    "Endocrine":        "#D55E00",
    "Immune":           "#56B4E9",
    "ChromatinRemodel": "#882255",
    "DNAMethylation":   "#999999",
}


# ── Load data ─────────────────────────────────────────────────────
print("Loading data...")
mut = pd.read_csv(MUT_FILE, usecols=["patientId", "mutationType", "gene.hugoGeneSymbol"],
                  low_memory=False)
mut = mut.rename(columns={"gene.hugoGeneSymbol": "gene"})
mut = mut[mut["mutationType"].isin(NON_SILENT)]
mut["channel"] = mut["gene"].map(CHANNEL_MAP)
mapped_mut = mut.dropna(subset=["channel"])

cli = pd.read_csv(CLI_FILE, usecols=["patientId", "OS_MONTHS", "OS_STATUS", "CANCER_TYPE"],
                  low_memory=False)
cli = cli.dropna(subset=["OS_MONTHS", "OS_STATUS"])
cli["event"] = cli["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
cli["time"] = pd.to_numeric(cli["OS_MONTHS"], errors="coerce")
cli = cli.dropna(subset=["time"])
cli = cli[cli["time"] > 0]

# ── Compute per-patient channel sets ──────────────────────────────
patient_channels = (
    mapped_mut.groupby("patientId")["channel"]
    .apply(lambda x: set(x))
    .reset_index()
)
patient_channels.columns = ["patientId", "channels_set"]
patient_channels["n_channels"] = patient_channels["channels_set"].apply(len)

df = cli.merge(patient_channels[["patientId", "n_channels", "channels_set"]],
               on="patientId", how="left")
df["n_channels"] = df["n_channels"].fillna(0).astype(int)
df["channels_set"] = df["channels_set"].apply(lambda x: x if isinstance(x, set) else set())

print(f"Patients with survival + channel data: {len(df)}")
print(f"Channels: {len(set(CHANNEL_MAP.values()))} ({len(CHANNEL_MAP)} genes)")


# ══════════════════════════════════════════════════════════════════
# Figure 1: KM curves by channel count
# ══════════════════════════════════════════════════════════════════
print("\n[1/5] KM by channel count...")

fig, ax = plt.subplots(figsize=(6, 4.5))
kmf = KaplanMeierFitter()

groups = {
    "0": df["n_channels"] == 0,
    "1": df["n_channels"] == 1,
    "2": df["n_channels"] == 2,
    "3": df["n_channels"] == 3,
    "4+": df["n_channels"] >= 4,
}
km_colors = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7"]

for (label, mask), color in zip(groups.items(), km_colors):
    sub = df[mask]
    kmf.fit(sub["time"], sub["event"], label=f"{label} ch (n={len(sub):,})")
    kmf.plot_survival_function(ax=ax, color=color, linewidth=1.5, ci_show=False)

ax.set_xlabel("Months", fontsize=11)
ax.set_ylabel("Overall Survival", fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower left", frameon=True, edgecolor="0.8")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(FIGS / f"km_by_channel_count.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved km_by_channel_count.pdf")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Severance hierarchy (8 channels)
# ══════════════════════════════════════════════════════════════════
print("\n[2/5] Severance hierarchy...")

strata = [1, 2, 3, 4]
strata_labels = ["1 ch", "2 ch", "3 ch", "4+ ch"]
n_ch = len(CHANNEL_ORDER)

pct_matrix = np.zeros((n_ch, len(strata)))
for j, s in enumerate(strata):
    mask = df["n_channels"] >= s if s == 4 else df["n_channels"] == s
    sub = df[mask]
    sub = sub[sub["channels_set"].apply(len) > 0]
    n_stratum = len(sub)
    if n_stratum == 0:
        continue
    for i, ch in enumerate(CHANNEL_ORDER):
        n_with = sub["channels_set"].apply(lambda cset: ch in cset).sum()
        pct_matrix[i, j] = 100.0 * n_with / n_stratum

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.10
x = np.arange(len(strata))

for i, ch in enumerate(CHANNEL_ORDER):
    offset = (i - n_ch / 2 + 0.5) * bar_width
    ax.bar(x + offset, pct_matrix[i, :], bar_width * 0.9,
           label=CHANNEL_NAMES[ch], color=CHANNEL_COLORS[ch],
           edgecolor="white", linewidth=0.5)

ax.set_xlabel("Total channels severed", fontsize=11)
ax.set_ylabel("% of patients with channel severed", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(strata_labels, fontsize=10)
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc="upper left", frameon=True, edgecolor="0.8", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(FIGS / f"severance_hierarchy.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved severance_hierarchy.pdf")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Channel frequency by cancer type (heatmap)
# ══════════════════════════════════════════════════════════════════
print("\n[3/5] Channel frequency by cancer type...")

ct_channel_counts = defaultdict(lambda: defaultdict(int))
ct_totals = defaultdict(int)

for _, row in df.iterrows():
    ct = row.get("CANCER_TYPE")
    if pd.isna(ct):
        continue
    ct_totals[ct] += 1
    for ch in row["channels_set"]:
        ct_channel_counts[ct][ch] += 1

# Filter to cancer types with >= 200 patients
cts = sorted([ct for ct, n in ct_totals.items() if n >= 200])

matrix = np.zeros((len(cts), len(CHANNEL_ORDER)))
for i, ct in enumerate(cts):
    for j, ch in enumerate(CHANNEL_ORDER):
        matrix[i, j] = ct_channel_counts[ct].get(ch, 0) / ct_totals[ct]

# Sort by total channel burden
burden = matrix.sum(axis=1)
sort_idx = np.argsort(burden)
matrix = matrix[sort_idx]
cts_sorted = [cts[i] for i in sort_idx]

fig, ax = plt.subplots(figsize=(9, max(8, len(cts_sorted) * 0.4)))

im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=min(matrix.max() * 1.1, 0.85),
               interpolation="nearest")

ax.set_xticks(range(len(CHANNEL_ORDER)))
ax.set_xticklabels([CHANNEL_NAMES[ch] for ch in CHANNEL_ORDER],
                   rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(cts_sorted)))
ax.set_yticklabels([f"{ct} ({ct_totals[ct]:,})" for ct in cts_sorted], fontsize=7)

for i in range(len(cts_sorted)):
    for j in range(len(CHANNEL_ORDER)):
        val = matrix[i, j]
        if val > 0.01:
            color = "white" if val > 0.4 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=5, color=color)

fig.colorbar(im, ax=ax, shrink=0.6, label="Fraction of patients")

plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(FIGS / f"channel_frequency_by_ct.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved channel_frequency_by_ct.pdf ({len(cts_sorted)} cancer types)")


# ══════════════════════════════════════════════════════════════════
# Figure 4: Channel co-severance matrix (8×8)
# ══════════════════════════════════════════════════════════════════
print("\n[4/5] Channel co-severance matrix...")

ch_idx = {ch: i for i, ch in enumerate(CHANNEL_ORDER)}
co_matrix = np.zeros((n_ch, n_ch))
ch_counts = np.zeros(n_ch)

for _, row in df.iterrows():
    chs = row["channels_set"]
    ch_list = [ch for ch in chs if ch in ch_idx]
    for ch in ch_list:
        ch_counts[ch_idx[ch]] += 1
    for a in ch_list:
        for b in ch_list:
            co_matrix[ch_idx[a], ch_idx[b]] += 1

# P(col | row)
cond_matrix = np.zeros((n_ch, n_ch))
for i in range(n_ch):
    if ch_counts[i] > 0:
        for j in range(n_ch):
            cond_matrix[i, j] = co_matrix[i, j] / ch_counts[i]

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(cond_matrix, cmap="Blues", vmin=0, vmax=1,
               interpolation="nearest")

ch_labels = [CHANNEL_NAMES[ch] for ch in CHANNEL_ORDER]
ax.set_xticks(range(n_ch))
ax.set_xticklabels(ch_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_ch))
ax.set_yticklabels(ch_labels, fontsize=8)

for i in range(n_ch):
    for j in range(n_ch):
        val = cond_matrix[i, j]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                fontsize=8, color=color)

fig.colorbar(im, ax=ax, shrink=0.8, label="P(column | row)")
ax.set_xlabel("Also severed", fontsize=10)
ax.set_ylabel("Given severed", fontsize=10)

plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(FIGS / f"channel_coseverance.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved channel_coseverance.pdf")


# ══════════════════════════════════════════════════════════════════
# Figure 5: SL mutual exclusivity by cancer type
# ══════════════════════════════════════════════════════════════════
print("\n[5/5] SL mutual exclusivity by cancer type...")

# Known SL pairs (DDR channel: BRCA1/2, RAD51C/D, PARP1 partners)
SL_PAIRS = [
    ("BRCA1", "BRCA2"), ("BRCA1", "RAD51C"), ("BRCA1", "RAD51D"),
    ("BRCA1", "PALB2"), ("BRCA2", "RAD51C"), ("BRCA2", "RAD51D"),
    ("ATM", "CHEK2"), ("ATM", "ATR"), ("BRCA1", "ATM"),
]

# Build patient-gene matrix for SL genes
sl_genes = set()
for a, b in SL_PAIRS:
    sl_genes.add(a)
    sl_genes.add(b)

# Use all non-silent mutations (not just mapped)
all_driver = mut[mut["gene"].isin(sl_genes)].copy()
patient_sl_genes = all_driver.groupby("patientId")["gene"].apply(set).to_dict()

# Per cancer type, compute obs/exp for each SL pair
ct_col = "CANCER_TYPE"
patient_ct = cli.set_index("patientId")[ct_col].to_dict()

cts_for_sl = [ct for ct, n in ct_totals.items() if n >= 500]

sl_results = []
for ct in cts_for_sl:
    ct_patients = [pid for pid, c in patient_ct.items() if c == ct]
    n_ct = len(ct_patients)
    if n_ct < 500:
        continue

    exclusivity_scores = []
    for gA, gB in SL_PAIRS:
        n_A = sum(1 for pid in ct_patients if gA in patient_sl_genes.get(pid, set()))
        n_B = sum(1 for pid in ct_patients if gB in patient_sl_genes.get(pid, set()))
        n_AB = sum(1 for pid in ct_patients
                   if gA in patient_sl_genes.get(pid, set())
                   and gB in patient_sl_genes.get(pid, set()))

        if n_A < 5 or n_B < 5:
            continue

        expected = (n_A / n_ct) * (n_B / n_ct) * n_ct
        if expected > 0:
            obs_exp = n_AB / expected
            exclusivity_scores.append(obs_exp)

    if exclusivity_scores:
        mean_excl = np.mean(exclusivity_scores)
        sl_results.append((ct, n_ct, mean_excl, len(exclusivity_scores)))

sl_results.sort(key=lambda x: x[2])

if sl_results:
    fig, ax = plt.subplots(figsize=(8, max(5, len(sl_results) * 0.35)))

    y = range(len(sl_results))
    vals = [r[2] for r in sl_results]
    labels = [f"{r[0]} (n={r[1]:,})" for r in sl_results]
    colors_bar = ["#0072B2" if v < 1 else "#D55E00" for v in vals]

    ax.barh(y, vals, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.axvline(x=1.0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Obs/Exp co-mutation ratio (lower = more exclusive)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS / f"sl_exclusivity_by_ct.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sl_exclusivity_by_ct.pdf ({len(sl_results)} cancer types)")
else:
    print("  SKIP: not enough SL pair data")


print(f"\nAll figures saved to {FIGS}/")
print("Done.")
