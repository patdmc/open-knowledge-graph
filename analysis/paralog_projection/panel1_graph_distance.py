"""Panel 1b: Graph distance vs sequence distance — where does 3D contact decouple?

The original Panel 1 stratified by Ensembl subtype (sequence divergence bins).
The noise diagnosis showed that each bin is dominated by different repeat/tandem
families (ZNF, snoRNA, Ig loci, Y_RNA) that have trivially high O/E from
genomic proximity, not from fold-mediated contact.

This script stratifies by TWO axes simultaneously:
  - Sequence divergence (perc_id, continuous)
  - Graph distance (functional proximity from GO, co-essentiality, PLM, PPI)

The prediction: 3D contact (O/E) should track GRAPH distance, not sequence
distance. Pairs that are sequence-distant but graph-close should show elevated
O/E. Pairs that are sequence-close but graph-distant should show baseline O/E.
The decoupling between the two axes IS the signal.

Usage:
    python panel1_graph_distance.py \
        --pair-table data/pair_table_gm12878_primary_heavy_full.parquet
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRAPH_FEATURES = {
    "coess_corr": "Co-essentiality (DepMap)",
    "go_bp_sim": "GO Biological Process similarity",
    "go_mf_sim": "GO Molecular Function similarity",
    "plm_cosine": "PLM embedding cosine similarity",
    "ppi_jaccard": "PPI Jaccard (STRING/BioGRID)",
}


def load_and_filter(path: Path) -> pd.DataFrame:
    """Load pair table, filter to intra-chrom pairs with valid O/E and at
    least one graph-distance feature."""
    df = pd.read_parquet(path)
    print(f"[gd] loaded {len(df):,} total pairs", file=sys.stderr)

    # Need valid O/E
    has_oe = df.hic_oe.notna()
    # Need at least one graph feature
    graph_cols = [c for c in GRAPH_FEATURES if c in df.columns]
    has_graph = df[graph_cols].notna().any(axis=1)

    filtered = df[has_oe & has_graph].copy()
    print(f"[gd] after filter (valid O/E + any graph feature): {len(filtered):,}", file=sys.stderr)
    print(f"[gd]   same_chrom: {filtered.same_chrom.sum():,}  cross_chrom: {(~filtered.same_chrom).sum():,}",
          file=sys.stderr)

    # Report per-feature coverage
    for col in graph_cols:
        n = filtered[col].notna().sum()
        print(f"[gd]   {col}: {n:,} valid ({100*n/len(filtered):.1f}%)", file=sys.stderr)

    return filtered


def make_heatmap(df: pd.DataFrame, graph_col: str, graph_label: str,
                 out_dir: Path, tag: str, n_seq_bins: int = 8, n_graph_bins: int = 8):
    """2D heatmap: sequence divergence (perc_id) vs graph distance, colored by median O/E."""
    sub = df[df[graph_col].notna()].copy()
    if len(sub) < 50:
        print(f"[gd] {graph_col}: only {len(sub)} pairs, skipping", file=sys.stderr)
        return None

    # Bucketize sequence divergence (perc_id)
    # Higher perc_id = more similar = more recent. Invert for "divergence" axis.
    sub["seq_bin"] = pd.qcut(sub.perc_id, n_seq_bins, duplicates="drop")

    # Bucketize graph distance
    # For coess_corr and similarity measures: higher = closer in graph
    sub["graph_bin"] = pd.qcut(sub[graph_col], n_graph_bins, duplicates="drop")

    # Compute stats per cell
    grouped = sub.groupby(["seq_bin", "graph_bin"], observed=True)
    stats = grouped.agg(
        oe_median=("hic_oe", "median"),
        oe_mean=("hic_oe", "mean"),
        n_pairs=("hic_oe", "count"),
        pct_above_2=("hic_oe", lambda x: (x > 2).mean() * 100),
    ).reset_index()

    # Pivot for heatmap
    pivot_oe = stats.pivot_table(index="graph_bin", columns="seq_bin",
                                  values="oe_median", observed=True)
    pivot_n = stats.pivot_table(index="graph_bin", columns="seq_bin",
                                 values="n_pairs", observed=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # O/E heatmap
    ax = axes[0]
    im = ax.imshow(pivot_oe.values, aspect="auto", cmap="RdYlBu_r",
                   vmin=0.5, vmax=3.0, origin="lower")
    ax.set_xlabel("Sequence identity (perc_id) →")
    ax.set_ylabel(f"{graph_label} →")
    ax.set_title(f"Median Hi-C O/E\n({graph_label} vs sequence identity)")
    ax.set_xticks(range(len(pivot_oe.columns)))
    ax.set_xticklabels([f"{x.left:.0f}-{x.right:.0f}" if hasattr(x, 'left')
                        else str(x) for x in pivot_oe.columns],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot_oe.index)))
    ax.set_yticklabels([f"{x.left:.2f}-{x.right:.2f}" if hasattr(x, 'left')
                        else str(x) for x in pivot_oe.index], fontsize=7)
    plt.colorbar(im, ax=ax, label="Median O/E", shrink=0.8)

    # Annotate cells with N
    for i in range(pivot_oe.shape[0]):
        for j in range(pivot_oe.shape[1]):
            n_val = pivot_n.values[i, j] if not np.isnan(pivot_n.values[i, j]) else 0
            oe_val = pivot_oe.values[i, j]
            if not np.isnan(oe_val):
                ax.text(j, i, f"{oe_val:.1f}\nn={int(n_val)}",
                       ha="center", va="center", fontsize=6,
                       color="white" if oe_val > 2.0 else "black")

    # N heatmap (pair count per cell — shows where we have power)
    ax2 = axes[1]
    im2 = ax2.imshow(np.log10(pivot_n.values + 1), aspect="auto", cmap="viridis",
                     origin="lower")
    ax2.set_xlabel("Sequence identity (perc_id) →")
    ax2.set_ylabel(f"{graph_label} →")
    ax2.set_title(f"log10(pair count)\n({graph_label} vs sequence identity)")
    ax2.set_xticks(range(len(pivot_n.columns)))
    ax2.set_xticklabels([f"{x.left:.0f}-{x.right:.0f}" if hasattr(x, 'left')
                         else str(x) for x in pivot_n.columns],
                        rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(pivot_n.index)))
    ax2.set_yticklabels([f"{x.left:.2f}-{x.right:.2f}" if hasattr(x, 'left')
                         else str(x) for x in pivot_n.index], fontsize=7)
    plt.colorbar(im2, ax=ax2, label="log10(N+1)", shrink=0.8)

    out_path = out_dir / f"panel1b_graph_dist_{graph_col}_{tag}.png"
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[gd] wrote {out_path}", file=sys.stderr)

    # Print the marginals — does graph distance predict O/E better than sequence?
    print(f"\n[gd] === {graph_label} ===")
    print(f"[gd] Pairs with valid {graph_col}: {len(sub):,}")

    # Marginal by graph distance (collapse across sequence bins)
    graph_marginal = sub.groupby("graph_bin", observed=True).agg(
        oe_median=("hic_oe", "median"),
        n=("hic_oe", "count"),
    )
    print(f"\n[gd] Marginal by {graph_col} (collapsing sequence):")
    print(graph_marginal.to_string())

    # Marginal by sequence (collapse across graph bins)
    seq_marginal = sub.groupby("seq_bin", observed=True).agg(
        oe_median=("hic_oe", "median"),
        n=("hic_oe", "count"),
    )
    print(f"\n[gd] Marginal by perc_id (collapsing {graph_col}):")
    print(seq_marginal.to_string())

    # Quick correlation check: which axis predicts O/E better?
    from scipy.stats import spearmanr
    r_graph, p_graph = spearmanr(sub[graph_col], sub.hic_oe)
    r_seq, p_seq = spearmanr(sub.perc_id, sub.hic_oe)
    print(f"\n[gd] Spearman with O/E:")
    print(f"[gd]   {graph_col}: r={r_graph:.4f}, p={p_graph:.2e}")
    print(f"[gd]   perc_id:    r={r_seq:.4f}, p={p_seq:.2e}")

    return stats


def composite_graph_distance(df: pd.DataFrame) -> pd.Series:
    """Compute a composite graph distance from all available features.
    Each feature is rank-normalized to [0,1], then averaged across
    non-null features per pair."""
    graph_cols = [c for c in GRAPH_FEATURES if c in df.columns]
    ranks = pd.DataFrame(index=df.index)
    for col in graph_cols:
        valid = df[col].notna()
        ranks.loc[valid, col] = df.loc[valid, col].rank(pct=True)
    composite = ranks.mean(axis=1)
    return composite


def make_composite_heatmap(df: pd.DataFrame, out_dir: Path, tag: str,
                           n_seq_bins: int = 8, n_graph_bins: int = 8):
    """Same heatmap but using the composite graph distance."""
    sub = df.copy()
    sub["graph_composite"] = composite_graph_distance(sub)
    valid = sub.graph_composite.notna() & sub.hic_oe.notna()
    sub = sub[valid]
    if len(sub) < 50:
        print("[gd] composite: too few pairs", file=sys.stderr)
        return

    make_heatmap_from_series(
        sub, "graph_composite", "Composite graph proximity (rank-avg)",
        out_dir, tag, n_seq_bins, n_graph_bins
    )


def make_heatmap_from_series(df, graph_col, graph_label, out_dir, tag,
                              n_seq_bins=8, n_graph_bins=8):
    """Reusable heatmap for any numeric column."""
    # This just calls make_heatmap with the column already present
    make_heatmap(df, graph_col, graph_label, out_dir, tag, n_seq_bins, n_graph_bins)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-table", type=Path,
                    default=Path("data/pair_table_gm12878_primary_heavy_full.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    pt = rel(args.pair_table)
    out_dir = rel(args.out_dir)
    tag = args.tag or pt.stem.replace("pair_table_", "").replace("_heavy", "").replace("_full", "")

    df = load_and_filter(pt)

    # Per-feature heatmaps
    for col, label in GRAPH_FEATURES.items():
        if col in df.columns and df[col].notna().sum() > 50:
            make_heatmap(df, col, label, out_dir, tag)

    # Composite heatmap
    df["graph_composite"] = composite_graph_distance(df)
    if df.graph_composite.notna().sum() > 50:
        make_heatmap(df, "graph_composite", "Composite graph proximity (rank-avg)",
                     out_dir, tag)

    # Summary: which axis wins?
    print("\n" + "=" * 60)
    print("[gd] SUMMARY: which axis predicts O/E better?")
    print("=" * 60)
    from scipy.stats import spearmanr
    valid = df.hic_oe.notna()
    for col, label in list(GRAPH_FEATURES.items()) + [("graph_composite", "Composite")]:
        if col in df.columns:
            mask = valid & df[col].notna()
            if mask.sum() > 50:
                r, p = spearmanr(df.loc[mask, col], df.loc[mask, "hic_oe"])
                print(f"  {label:45s}  r={r:+.4f}  p={p:.2e}  N={mask.sum():,}")
    mask = valid & df.perc_id.notna()
    r, p = spearmanr(df.loc[mask, "perc_id"], df.loc[mask, "hic_oe"])
    print(f"  {'Sequence identity (perc_id)':45s}  r={r:+.4f}  p={p:.2e}  N={mask.sum():,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
