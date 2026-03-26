#!/usr/bin/env python3
"""Generate publication-quality figures for McCarthy2026_5_Cancer.tex."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
CACHE = Path(__file__).resolve().parent / "cache"
FIGS  = Path(__file__).resolve().parent.parent / "figures"
FIGS.mkdir(exist_ok=True)

MUT_FILE = CACHE / "msk_impact_50k_2026_mutations.csv"
CLI_FILE = CACHE / "msk_impact_50k_2026_clinical.csv"

# ── Channel map (from channel_analysis.py lines 54-118) ───────────
CHANNEL_MAP = {
    # DDR channel
    "BRCA1": "DDR", "BRCA2": "DDR", "PALB2": "DDR",
    "RAD51C": "DDR", "RAD51D": "DDR", "RAD51B": "DDR",
    "ATM": "DDR", "ATR": "DDR", "CHEK2": "DDR", "CHEK1": "DDR",
    "FANCA": "DDR", "FANCC": "DDR", "FANCD2": "DDR",
    "BAP1": "DDR", "BARD1": "DDR",
    "MLH1": "DDR", "MSH2": "DDR", "MSH6": "DDR", "PMS2": "DDR",
    "POLE": "DDR", "POLD1": "DDR",
    # Cell cycle / apoptosis channel
    "TP53": "CellCycle", "RB1": "CellCycle",
    "CDKN1A": "CellCycle", "CDKN1B": "CellCycle",
    "CDKN2A": "CellCycle", "CDKN2B": "CellCycle",
    "CDK4": "CellCycle", "CDK6": "CellCycle",
    "CCND1": "CellCycle", "CCNE1": "CellCycle",
    "MDM2": "CellCycle", "MDM4": "CellCycle",
    "MYC": "CellCycle", "MYCN": "CellCycle",
    # PI3K / Growth signaling channel
    "PIK3CA": "PI3K_Growth", "PIK3R1": "PI3K_Growth",
    "PTEN": "PI3K_Growth", "AKT1": "PI3K_Growth",
    "AKT2": "PI3K_Growth", "AKT3": "PI3K_Growth",
    "MTOR": "PI3K_Growth",
    "KRAS": "PI3K_Growth", "NRAS": "PI3K_Growth", "HRAS": "PI3K_Growth",
    "BRAF": "PI3K_Growth", "RAF1": "PI3K_Growth",
    "MAP2K1": "PI3K_Growth", "MAP2K2": "PI3K_Growth",
    "MAP3K1": "PI3K_Growth", "MAP3K13": "PI3K_Growth",
    "ERBB2": "PI3K_Growth", "ERBB3": "PI3K_Growth",
    "EGFR": "PI3K_Growth", "FGFR1": "PI3K_Growth",
    "FGFR2": "PI3K_Growth", "FGFR3": "PI3K_Growth",
    "IGF1R": "PI3K_Growth", "MET": "PI3K_Growth",
    "NF1": "PI3K_Growth", "NF2": "PI3K_Growth",
    "TSC1": "PI3K_Growth", "TSC2": "PI3K_Growth",
    "STK11": "PI3K_Growth",
    "ARID1A": "PI3K_Growth",
    # Endocrine / hormone receptor channel
    "ESR1": "Endocrine", "ESR2": "Endocrine",
    "PGR": "Endocrine", "AR": "Endocrine",
    "FOXA1": "Endocrine",
    "GATA3": "Endocrine",
    # Immune surveillance channel
    "B2M": "Immune", "HLA-A": "Immune", "HLA-B": "Immune",
    "HLA-C": "Immune",
    "JAK1": "Immune", "JAK2": "Immune",
    "STAT1": "Immune",
    "CD274": "Immune",
    "PDCD1LG2": "Immune",
    "CTLA4": "Immune",
    # Tissue architecture channel
    "CDH1": "TissueArch", "CDH2": "TissueArch",
    "CTNNB1": "TissueArch",
    "APC": "TissueArch",
    "AXIN1": "TissueArch", "AXIN2": "TissueArch",
    "SMAD2": "TissueArch", "SMAD3": "TissueArch", "SMAD4": "TissueArch",
    "TGFBR1": "TissueArch", "TGFBR2": "TissueArch",
    "NOTCH1": "TissueArch", "NOTCH2": "TissueArch",
    "NOTCH3": "TissueArch", "NOTCH4": "TissueArch",
    "FBXW7": "TissueArch",
    "GJA1": "TissueArch",
    "GJB2": "TissueArch",
}

CHANNEL_NAMES = {
    "DDR": "DNA Damage Response",
    "CellCycle": "Cell Cycle / Apoptosis",
    "PI3K_Growth": "PI3K / Growth Signaling",
    "Endocrine": "Endocrine / Hormone",
    "Immune": "Immune Surveillance",
    "TissueArch": "Tissue Architecture",
}

NON_SILENT = {
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "Splice_Region",
    "Nonstop_Mutation", "Translation_Start_Site",
}

# ── Colorblind-friendly palette (Okabe-Ito) ───────────────────────
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
CHANNEL_COLORS = {
    "PI3K_Growth": "#0072B2",
    "CellCycle":   "#E69F00",
    "TissueArch":  "#009E73",
    "DDR":         "#CC79A7",
    "Endocrine":   "#D55E00",
    "Immune":      "#56B4E9",
}

# ── Load data ──────────────────────────────────────────────────────
print("Loading mutations...")
mut = pd.read_csv(MUT_FILE, usecols=["patientId", "mutationType", "gene.hugoGeneSymbol"])
mut = mut.rename(columns={"gene.hugoGeneSymbol": "gene"})
mut = mut[mut["mutationType"].isin(NON_SILENT)]
mut["channel"] = mut["gene"].map(CHANNEL_MAP)
mut = mut.dropna(subset=["channel"])

print("Loading clinical data...")
cli = pd.read_csv(CLI_FILE, usecols=["patientId", "OS_MONTHS", "OS_STATUS"])
cli = cli.dropna(subset=["OS_MONTHS", "OS_STATUS"])
cli["event"] = cli["OS_STATUS"].apply(lambda x: 1 if str(x).startswith("1") else 0)
cli["time"] = cli["OS_MONTHS"].astype(float)
cli = cli[cli["time"] > 0]

# ── Compute channel count per patient ──────────────────────────────
patient_channels = (
    mut.groupby("patientId")["channel"]
    .apply(lambda x: set(x))
    .reset_index()
)
patient_channels["n_channels"] = patient_channels["channel"].apply(len)
patient_channels["channels_set"] = patient_channels["channel"]

df = cli.merge(patient_channels[["patientId", "n_channels", "channels_set"]],
               on="patientId", how="left")
df["n_channels"] = df["n_channels"].fillna(0).astype(int)

# Cap at 4+
df["ch_group"] = df["n_channels"].apply(lambda x: "4+" if x >= 4 else str(x))

print(f"Total patients with survival data: {len(df)}")
for g in ["0", "1", "2", "3", "4+"]:
    n = (df["ch_group"] == g).sum()
    print(f"  {g} channels: n={n}")


# ══════════════════════════════════════════════════════════════════
# Figure 1: KM curves by channel count
# ══════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1: KM by channel count...")

fig, ax = plt.subplots(figsize=(6, 4.5))
kmf = KaplanMeierFitter()

groups = ["0", "1", "2", "3", "4+"]
colors = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7"]

for grp, color in zip(groups, colors):
    mask = df["ch_group"] == grp
    sub = df[mask]
    n = len(sub)
    d = sub["event"].sum()
    kmf.fit(sub["time"], sub["event"], label=f"{grp} ch (n={n:,})")
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
fig.savefig(FIGS / "km_by_channel_count.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIGS / "km_by_channel_count.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIGS / 'km_by_channel_count.pdf'}")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Severance hierarchy grouped bar chart
# ══════════════════════════════════════════════════════════════════
print("\nGenerating Figure 2: Severance hierarchy...")

# For patients with channels_set, compute % with each channel at each stratum
channel_order = ["PI3K_Growth", "CellCycle", "TissueArch", "DDR", "Endocrine", "Immune"]
channel_labels = ["PI3K/\nGrowth", "Cell\nCycle", "Tissue\nArch", "DDR", "Endo-\ncrine", "Immune"]
strata = [1, 2, 3, 4]  # 4 means 4+
strata_labels = ["1 ch", "2 ch", "3 ch", "4+ ch"]

# Build the data matrix: rows=channels, cols=strata
pct_matrix = np.zeros((len(channel_order), len(strata)))

for j, s in enumerate(strata):
    if s == 4:
        mask = df["n_channels"] >= 4
    else:
        mask = df["n_channels"] == s
    sub = df[mask].dropna(subset=["channels_set"])
    n_stratum = len(sub)
    if n_stratum == 0:
        continue
    for i, ch in enumerate(channel_order):
        n_with = sub["channels_set"].apply(lambda cset: ch in cset).sum()
        pct_matrix[i, j] = 100.0 * n_with / n_stratum

fig, ax = plt.subplots(figsize=(7, 4.5))

n_channels = len(channel_order)
n_strata = len(strata)
bar_width = 0.15
x = np.arange(n_strata)

strata_colors = ["#56B4E9", "#E69F00", "#D55E00", "#CC79A7"]

for i, (ch, label) in enumerate(zip(channel_order, channel_labels)):
    offset = (i - n_channels / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, pct_matrix[i, :], bar_width * 0.9,
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
fig.savefig(FIGS / "severance_hierarchy.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIGS / "severance_hierarchy.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIGS / 'severance_hierarchy.pdf'}")

print("\nDone.")
