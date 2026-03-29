#!/usr/bin/env python3
"""
Compute synthetic lethality mutual exclusivity by cancer type.

For each known SL pair in the DDR channel, compute obs/exp ratio
per cancer type. Low obs/exp = strong mutual exclusivity = channel
is load-bearing in that tissue.

Generates a heatmap figure for paper 5.

Usage:
    python3 -u -m gnn.scripts.sl_by_cancer_type
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "sl_by_ct")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """Load patient mutations and SL pairs from Neo4j."""
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    # Patient mutations with cancer type
    print("  Loading patient mutations...")
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)
            RETURN p.id AS pid, p.cancer_type AS ct, g.name AS gene
        """)
        rows = [(r["pid"], r["ct"], r["gene"]) for r in result]

    # SL pairs
    print("  Loading SL pairs...")
    with driver.session() as session:
        result = session.run("""
            MATCH (g1:Gene)-[r:SL_PARTNER]-(g2:Gene)
            WHERE g1.name < g2.name
            RETURN DISTINCT g1.name AS g1, g2.name AS g2,
                   r.source AS source
        """)
        sl_pairs = [(r["g1"], r["g2"]) for r in result]

    # Channel assignments
    print("  Loading channel assignments...")
    with driver.session() as session:
        result = session.run("""
            MATCH (g:Gene)-[b:BELONGS_TO]->(c:Channel)
            RETURN g.name AS gene, c.name AS channel, b.weight AS weight
        """)
        gene_channel = {}
        for r in result:
            gene = r["gene"]
            ch = r["channel"]
            w = r["weight"] or 1.0
            if gene not in gene_channel or w > gene_channel[gene][1]:
                gene_channel[gene] = (ch, w)

    driver.close()

    # Build structures
    patient_ct = {}
    patient_genes = defaultdict(set)
    ct_patients = defaultdict(set)
    ct_gene_patients = defaultdict(lambda: defaultdict(set))

    for pid, ct, gene in rows:
        if ct is None:
            continue
        patient_ct[pid] = ct
        patient_genes[pid].add(gene)
        ct_patients[ct].add(pid)
        ct_gene_patients[ct][gene].add(pid)

    # Filter SL pairs to DDR channel
    ddr_genes = {g for g, (ch, _) in gene_channel.items() if ch == "DDR"}
    ddr_sl = [(g1, g2) for g1, g2 in sl_pairs
              if g1 in ddr_genes or g2 in ddr_genes]

    # Also include all SL pairs for broader analysis
    print(f"  {len(rows)} mutation edges, {len(ct_patients)} cancer types")
    print(f"  {len(sl_pairs)} SL pairs total, {len(ddr_sl)} involving DDR genes")
    print(f"  {len(ddr_genes)} DDR genes")

    return ct_patients, ct_gene_patients, sl_pairs, ddr_sl, gene_channel


def compute_obs_exp_by_ct(ct_patients, ct_gene_patients, pairs, min_ct_size=200,
                          min_expected=2.0):
    """Compute obs/exp for each gene pair in each cancer type."""
    # Filter to large enough CTs
    cts = [ct for ct, ps in ct_patients.items() if len(ps) >= min_ct_size]
    cts.sort()
    print(f"  {len(cts)} cancer types with >= {min_ct_size} patients")

    results = []
    for ct in cts:
        n = len(ct_patients[ct])
        gp = ct_gene_patients[ct]

        for g1, g2 in pairs:
            ps1 = gp.get(g1, set())
            ps2 = gp.get(g2, set())
            n1 = len(ps1)
            n2 = len(ps2)
            if n1 < 5 or n2 < 5:
                continue

            both = len(ps1 & ps2)
            expected = (n1 / n) * (n2 / n) * n
            if expected < min_expected:
                continue

            obs_exp = both / max(expected, 1e-6)

            results.append({
                "cancer_type": ct,
                "gene1": g1,
                "gene2": g2,
                "pair": f"{g1}-{g2}",
                "observed": both,
                "expected": round(expected, 1),
                "obs_exp": round(obs_exp, 3),
                "n_patients": n,
                "n_g1": n1,
                "n_g2": n2,
            })

    return pd.DataFrame(results)


def make_heatmap(df, output_path):
    """Generate heatmap of obs/exp by cancer type x SL pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if df.empty:
        print("  No data for heatmap!")
        return

    # Aggregate: mean obs/exp per (cancer_type, pair)
    pivot = df.pivot_table(index="cancer_type", columns="pair",
                           values="obs_exp", aggfunc="mean")

    # Filter: keep pairs present in >= 5 CTs, CTs with >= 3 pairs
    pair_counts = pivot.notna().sum(axis=0)
    good_pairs = pair_counts[pair_counts >= 3].index
    pivot = pivot[good_pairs]

    ct_counts = pivot.notna().sum(axis=1)
    good_cts = ct_counts[ct_counts >= 2].index
    pivot = pivot.loc[good_cts]

    if pivot.empty:
        print("  Not enough data for filtered heatmap")
        # Try unfiltered
        pivot = df.pivot_table(index="cancer_type", columns="pair",
                               values="obs_exp", aggfunc="mean")

    # Sort CTs by mean obs/exp (most exclusive at top)
    ct_means = pivot.mean(axis=1).sort_values()
    pivot = pivot.loc[ct_means.index]

    # Sort pairs by mean obs/exp
    pair_means = pivot.mean(axis=0).sort_values()
    pivot = pivot[pair_means.index]

    # Cap obs/exp for display
    pivot_display = pivot.clip(upper=3.0)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6),
                                     max(6, len(pivot.index) * 0.35)))

    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=3.0)
    im = ax.imshow(pivot_display.values, aspect="auto", cmap="RdBu_r",
                   norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="obs/exp ratio")

    ax.set_xlabel("SL gene pair")
    ax.set_ylabel("Cancer type")
    ax.set_title("Synthetic lethality mutual exclusivity by cancer type\n"
                 "(obs/exp < 1 = exclusive, > 1 = co-occurring)")

    # Mark cells with no data
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if pd.isna(pivot.iloc[i, j]):
                ax.text(j, i, "·", ha="center", va="center",
                        color="gray", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap to {output_path}")


def make_channel_summary(df, gene_channel, output_path):
    """Summary: mean obs/exp per cancer type, grouped by channel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return

    # Assign channel to each pair (use gene1's channel)
    def pair_channel(row):
        ch1 = gene_channel.get(row["gene1"], ("Unknown", 0))[0]
        ch2 = gene_channel.get(row["gene2"], ("Unknown", 0))[0]
        if ch1 == ch2:
            return ch1
        return f"{ch1}/{ch2}"

    df = df.copy()
    df["channel"] = df.apply(pair_channel, axis=1)

    # Mean obs/exp per cancer type per channel
    summary = df.groupby(["cancer_type", "channel"])["obs_exp"].mean().reset_index()
    pivot = summary.pivot_table(index="cancer_type", columns="channel",
                                values="obs_exp", aggfunc="mean")

    if pivot.empty:
        return

    ct_means = pivot.mean(axis=1).sort_values()
    pivot = pivot.loc[ct_means.index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                     max(6, len(pivot.index) * 0.35)))

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=3.0)
    im = ax.imshow(pivot.clip(upper=3.0).values, aspect="auto",
                   cmap="RdBu_r", norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="mean obs/exp")
    ax.set_title("SL exclusivity by cancer type and channel\n"
                 "(blue = exclusive/load-bearing, red = co-occurring/bypassed)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved channel summary to {output_path}")


def main():
    print("=" * 70)
    print("  SL MUTUAL EXCLUSIVITY BY CANCER TYPE")
    print("=" * 70)

    ct_patients, ct_gene_patients, all_sl, ddr_sl, gene_channel = load_data()

    # All SL pairs
    print("\n  --- All SL pairs ---")
    df_all = compute_obs_exp_by_ct(ct_patients, ct_gene_patients, all_sl)
    print(f"  {len(df_all)} observations")

    if not df_all.empty:
        # Summary stats
        print(f"\n  obs/exp distribution:")
        print(f"    mean: {df_all['obs_exp'].mean():.3f}")
        print(f"    median: {df_all['obs_exp'].median():.3f}")
        print(f"    < 0.5 (strong ME): {(df_all['obs_exp'] < 0.5).sum()}")
        print(f"    > 2.0 (strong co-occ): {(df_all['obs_exp'] > 2.0).sum()}")

        # By cancer type: which CTs show strongest ME?
        ct_mean = df_all.groupby("cancer_type")["obs_exp"].agg(["mean", "count"])
        ct_mean = ct_mean.sort_values("mean")
        print(f"\n  Most exclusive cancer types (lowest mean obs/exp):")
        for ct, row in ct_mean.head(10).iterrows():
            print(f"    {ct:>35s}  mean={row['mean']:.3f}  n_pairs={int(row['count'])}")
        print(f"\n  Least exclusive (highest mean obs/exp):")
        for ct, row in ct_mean.tail(10).iterrows():
            print(f"    {ct:>35s}  mean={row['mean']:.3f}  n_pairs={int(row['count'])}")

        # Heatmap
        make_heatmap(df_all, os.path.join(FIGURES_DIR, "sl_exclusivity_by_ct.pdf"))
        make_channel_summary(df_all, gene_channel,
                            os.path.join(FIGURES_DIR, "sl_channel_summary_by_ct.pdf"))

    # Save raw data
    if not df_all.empty:
        df_all.to_csv(os.path.join(RESULTS_DIR, "sl_obs_exp_by_ct.csv"), index=False)

    # Save summary
    summary = {}
    if not df_all.empty:
        ct_mean = df_all.groupby("cancer_type")["obs_exp"].agg(["mean", "median", "count"])
        summary = {
            "n_observations": len(df_all),
            "n_cancer_types": df_all["cancer_type"].nunique(),
            "n_pairs": df_all["pair"].nunique(),
            "global_mean_obs_exp": round(df_all["obs_exp"].mean(), 4),
            "global_median_obs_exp": round(df_all["obs_exp"].median(), 4),
            "most_exclusive_cts": ct_mean.sort_values("mean").head(5).to_dict("index"),
            "least_exclusive_cts": ct_mean.sort_values("mean").tail(5).to_dict("index"),
        }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
