#!/usr/bin/env python3
"""
Generate channel severance heatmap by cancer type.

Shows which coupling channels are most frequently severed in each
cancer type. Supports the empirical claim that channels are
organizational (tissue-specific) not statistical.

Also generates:
- Cross-channel vs same-channel mortality by cancer type
- Channel co-severance matrix (which channels are severed together)

Usage:
    python3 -u -m gnn.scripts.channel_by_cancer_type
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CHANNEL_ORDER = ["PI3K/Growth", "Cell Cycle", "Tissue Architecture",
                 "DDR", "Endocrine", "Immune"]


def load_data():
    """Load patient data with channel assignments."""
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    print("  Loading patient mutations with channels...")
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)-[b:BELONGS_TO]->(c:Channel)
            RETURN p.id AS pid, p.cancer_type AS ct,
                   p.os_months AS os, p.event AS status,
                   g.name AS gene, c.name AS channel
        """)
        rows = [(r["pid"], r["ct"], r["os"], r["status"],
                 r["gene"], r["channel"]) for r in result]

    driver.close()

    # Build patient structures
    patient_ct = {}
    patient_os = {}
    patient_status = {}
    patient_channels = defaultdict(set)
    patient_genes = defaultdict(set)

    for pid, ct, os_m, status, gene, channel in rows:
        if ct is None:
            continue
        patient_ct[pid] = ct
        if os_m is not None:
            patient_os[pid] = os_m
        if status is not None:
            patient_status[pid] = status
        patient_channels[pid].add(channel)
        patient_genes[pid].add(gene)

    print(f"  {len(patient_ct)} patients with cancer type")
    print(f"  {len(patient_os)} patients with OS data")
    return patient_ct, patient_os, patient_status, patient_channels, patient_genes


def channel_frequency_heatmap(patient_ct, patient_channels, min_ct_size=200):
    """Heatmap: fraction of patients with each channel severed, by CT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Count per CT per channel
    ct_channel_counts = defaultdict(lambda: defaultdict(int))
    ct_counts = defaultdict(int)

    for pid, ct in patient_ct.items():
        ct_counts[ct] += 1
        for ch in patient_channels.get(pid, set()):
            ct_channel_counts[ct][ch] += 1

    # Filter to large CTs
    cts = [ct for ct, n in ct_counts.items() if n >= min_ct_size]
    cts.sort()

    channels = [ch for ch in CHANNEL_ORDER if any(
        ct_channel_counts[ct].get(ch, 0) > 0 for ct in cts)]

    # Build matrix: fraction of patients with channel severed
    matrix = np.zeros((len(cts), len(channels)))
    for i, ct in enumerate(cts):
        for j, ch in enumerate(channels):
            matrix[i, j] = ct_channel_counts[ct].get(ch, 0) / ct_counts[ct]

    # Sort CTs by total channel burden
    ct_burden = matrix.sum(axis=1)
    sort_idx = np.argsort(ct_burden)
    matrix = matrix[sort_idx]
    cts = [cts[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(8, max(8, len(cts) * 0.35)))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=min(matrix.max(), 0.8),
                   interpolation="nearest")

    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels([f"{ct} ({ct_counts[ct]})" for ct in cts], fontsize=7)

    # Add text annotations
    for i in range(len(cts)):
        for j in range(len(channels)):
            val = matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Fraction of patients")
    ax.set_title("Channel severance frequency by cancer type")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "channel_frequency_by_ct.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved channel frequency heatmap to {path}")


def channel_coseverance_matrix(patient_channels):
    """Heatmap: how often channels are severed together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = CHANNEL_ORDER
    n_ch = len(channels)
    ch_idx = {ch: i for i, ch in enumerate(channels)}

    # Count co-severance
    co_matrix = np.zeros((n_ch, n_ch))
    ch_counts = np.zeros(n_ch)

    for pid, chs in patient_channels.items():
        for ch in chs:
            if ch in ch_idx:
                ch_counts[ch_idx[ch]] += 1
        ch_list = [ch for ch in chs if ch in ch_idx]
        for a in ch_list:
            for b in ch_list:
                co_matrix[ch_idx[a], ch_idx[b]] += 1

    # Normalize to conditional probability P(col | row)
    n_patients = len(patient_channels)
    cond_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        if ch_counts[i] > 0:
            for j in range(n_ch):
                cond_matrix[i, j] = co_matrix[i, j] / ch_counts[i]

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cond_matrix, cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(channels, fontsize=9)

    for i in range(n_ch):
        for j in range(n_ch):
            val = cond_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8,
                        label="P(column channel | row channel)")
    ax.set_title("Channel co-severance: P(column | row)")
    ax.set_xlabel("Also severed")
    ax.set_ylabel("Given severed")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "channel_coseverance.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved co-severance matrix to {path}")


def channel_mortality_by_ct(patient_ct, patient_os, patient_status,
                            patient_channels, min_ct_size=300):
    """Bar chart: cross-channel vs same-channel mortality by CT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct_cross = defaultdict(lambda: {"dead": 0, "total": 0})
    ct_same = defaultdict(lambda: {"dead": 0, "total": 0})

    for pid, ct in patient_ct.items():
        if pid not in patient_os or pid not in patient_status:
            continue
        chs = patient_channels.get(pid, set())
        n_ch = len(chs)
        if n_ch < 1:
            continue

        dead = 1 if patient_status[pid] in (1, "1", "DECEASED", True) else 0

        if n_ch >= 2:
            ct_cross[ct]["dead"] += dead
            ct_cross[ct]["total"] += 1
        elif n_ch == 1:
            ct_same[ct]["dead"] += dead
            ct_same[ct]["total"] += 1

    # Filter to CTs with enough in both groups
    cts = []
    for ct in set(ct_cross.keys()) & set(ct_same.keys()):
        if ct_cross[ct]["total"] >= 30 and ct_same[ct]["total"] >= 30:
            cts.append(ct)

    if not cts:
        print("  Not enough data for cross/same mortality by CT")
        return

    # Compute mortality rates
    data = []
    for ct in cts:
        cross_mort = ct_cross[ct]["dead"] / ct_cross[ct]["total"]
        same_mort = ct_same[ct]["dead"] / ct_same[ct]["total"]
        diff = cross_mort - same_mort
        data.append((ct, cross_mort, same_mort, diff,
                     ct_cross[ct]["total"], ct_same[ct]["total"]))

    data.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.4)))

    y = range(len(data))
    ct_labels = [f"{d[0]} (n={d[4]}+{d[5]})" for d in data]
    cross_vals = [d[1] for d in data]
    same_vals = [d[2] for d in data]

    ax.barh([yi - 0.15 for yi in y], cross_vals, 0.3,
            label="Cross-channel (2+ channels)", color="#d62728", alpha=0.8)
    ax.barh([yi + 0.15 for yi in y], same_vals, 0.3,
            label="Same-channel (1 channel)", color="#1f77b4", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(ct_labels, fontsize=7)
    ax.set_xlabel("Mortality rate")
    ax.set_title("Cross-channel vs same-channel mortality by cancer type")
    ax.legend(loc="lower right")
    ax.axvline(x=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "cross_same_mortality_by_ct.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross/same mortality chart to {path}")

    # Print summary
    n_cross_worse = sum(1 for d in data if d[3] > 0)
    print(f"  Cross-channel mortality higher in {n_cross_worse}/{len(data)} cancer types")


def main():
    print("=" * 70)
    print("  CHANNEL ANALYSIS BY CANCER TYPE")
    print("=" * 70)

    patient_ct, patient_os, patient_status, patient_channels, patient_genes = load_data()

    print("\n  --- Channel frequency by cancer type ---")
    channel_frequency_heatmap(patient_ct, patient_channels)

    print("\n  --- Channel co-severance matrix ---")
    channel_coseverance_matrix(patient_channels)

    print("\n  --- Cross vs same channel mortality by CT ---")
    channel_mortality_by_ct(patient_ct, patient_os, patient_status, patient_channels)

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
