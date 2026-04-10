"""Panel 1: Dual-proximity handoff.

The gate panel for the projection-theory test.

X-axis: divergence (Ensembl `subtype` ladder — named discrete bins:
  Homo sapiens → Hominidae → ... → Opisthokonta → Bilateria).
Y-axes (twinned): median linear chromosomal distance (same-chrom subset)
  and median Hi-C O/E contact enrichment (all pairs with a valid O/E value).

Projection theory predicts a handoff: as divergence increases,
  (a) linear distance rises toward the cross-chromosomal limit,
  (b) 3D contact enrichment stays elevated / rises for pairs still in the
      same equivalence class.

Outcomes per NEXT_STEPS.md §6:
  A. Clean handoff — write up the paper.
  B. Parallel decay — retreat the framing.
  C. Channel-dependent — run Panel 3 next.

This script reads the joined pair table produced by build_pair_table_heavy.py
plus concat_heavy_shards.py. It does not refit the Hi-C data.

Usage:
  python panel1_dual_proximity.py \\
      --pair-table data/pair_table_gm12878_dpnII_heavy.parquet \\
      --out-dir data/figures/gm12878_dpnII
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Order matters — this is the ladder from recent to ancient.
# Anything not in this list is dropped from the plot.
SUBTYPE_LADDER = [
    "Homo sapiens",
    "Hominidae",
    "Hominoidea",
    "Catarrhini",
    "Simiiformes",
    "Haplorrhini",
    "Primates",
    "Euarchontoglires",
    "Boreoeutheria",
    "Eutheria",
    "Theria",
    "Mammalia",
    "Amniota",
    "Tetrapoda",
    "Sarcopterygii",
    "Euteleostomi",
    "Vertebrata",
    "Gnathostomata",
    "Chordata",
    "Bilateria",
    "Opisthokonta",
]


def load_pair_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[panel1] loaded {len(df):,} pairs from {path}", file=sys.stderr)
    return df


def bin_by_subtype(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each subtype into summary statistics for Panel 1."""
    present = [s for s in SUBTYPE_LADDER if s in df.subtype.unique()]
    print(f"[panel1] subtype bins present: {present}", file=sys.stderr)

    rows = []
    for s in present:
        sub = df[df.subtype == s]
        sc = sub[sub.same_chrom]
        # Linear distance: only defined for same-chrom pairs. Use median to
        # resist tail effects. Report cross-chrom fraction separately.
        lin_median = sc.linear_distance_bp.median() if len(sc) else np.nan
        lin_mean = sc.linear_distance_bp.mean() if len(sc) else np.nan
        cross_frac = 1.0 - len(sc) / len(sub) if len(sub) else np.nan

        # 3D contact: use O/E where available (intra-chrom only), otherwise
        # fall back to observed counts for inter-chrom. For Panel 1's
        # headline comparison we want the intra-chrom O/E median as the
        # primary "3D proximity" metric because it removes the distance-
        # decay confound that would otherwise trivially track linear distance.
        intra_oe = sub.hic_oe.dropna()
        oe_median = intra_oe.median() if len(intra_oe) else np.nan
        oe_q75 = intra_oe.quantile(0.75) if len(intra_oe) else np.nan

        # Inter-chrom observed contact fraction (> 0) — is there ANY signal?
        inter = sub[~sub.same_chrom]
        inter_contact_frac = (inter.hic_obs > 0).mean() if len(inter) else np.nan

        rows.append({
            "subtype": s,
            "n_pairs": len(sub),
            "n_same_chrom": len(sc),
            "cross_chrom_frac": cross_frac,
            "lin_distance_median_mb": lin_median / 1e6,
            "lin_distance_mean_mb": lin_mean / 1e6,
            "intra_oe_median": oe_median,
            "intra_oe_q75": oe_q75,
            "inter_contact_frac": inter_contact_frac,
        })
    return pd.DataFrame(rows)


def plot_panel1(summary: pd.DataFrame, same_channel: pd.DataFrame, out_path: Path) -> None:
    fig, ax_left = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax_right = ax_left.twinx()

    x = np.arange(len(summary))
    labels = summary.subtype.tolist()

    # Left axis: linear chrom distance (median, same-chrom only), Mb
    line_lin = ax_left.plot(
        x, summary.lin_distance_median_mb, "o-",
        color="tab:blue", lw=2, ms=6, label="Linear distance (median, same-chrom, Mb)"
    )
    ax_left.set_ylabel("Linear distance (Mb, same-chrom median)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=45, ha="right")
    ax_left.set_xlabel("Divergence (Ensembl subtype ladder, recent → ancient)")
    ax_left.grid(True, axis="x", alpha=0.2)

    # Right axis: intra-chrom O/E median — the 3D proximity metric
    line_oe = ax_right.plot(
        x, summary.intra_oe_median, "s-",
        color="tab:red", lw=2, ms=6, label="Intra-chrom O/E (median)"
    )
    ax_right.axhline(1.0, color="tab:red", ls=":", lw=1, alpha=0.5)
    ax_right.set_ylabel("Hi-C O/E (intra-chrom median)", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")

    # Same-channel overlay — if enough data points, plot the same-channel-only
    # O/E median as a darker line to show the projection-theory cohort.
    if len(same_channel) > 0:
        sc_present = [s for s in SUBTYPE_LADDER if s in same_channel.subtype.unique()]
        sc_x = [SUBTYPE_LADDER.index(s) for s in sc_present if s in labels]
        sc_x = [labels.index(s) for s in sc_present if s in labels]
        sc_oe = []
        for s in sc_present:
            if s not in labels:
                continue
            sub = same_channel[same_channel.subtype == s].hic_oe.dropna()
            sc_oe.append(sub.median() if len(sub) else np.nan)
        if sc_x:
            ax_right.plot(
                sc_x, sc_oe, "D-",
                color="darkred", lw=2, ms=8, alpha=0.9,
                label="Same-channel O/E (median)",
            )

    # Title and combined legend
    ax_left.set_title("Panel 1: Dual-proximity handoff across divergence")
    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_l + handles_r, labels_l + labels_r, loc="upper left", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[panel1] wrote {out_path} (+ .pdf)", file=sys.stderr)


def classify_outcome(summary: pd.DataFrame) -> str:
    """Lightweight, explicit heuristic. Not statistical — just the eye-test."""
    # Drop rows with no data
    s = summary.dropna(subset=["lin_distance_median_mb", "intra_oe_median"])
    if len(s) < 3:
        return "UNDETERMINED — fewer than 3 subtype bins have both metrics"
    lin = s.lin_distance_median_mb.values
    oe = s.intra_oe_median.values
    # Compare ends of the ladder.
    lin_recent = np.nanmean(lin[: max(1, len(lin) // 4)])
    lin_ancient = np.nanmean(lin[-max(1, len(lin) // 4):])
    oe_recent = np.nanmean(oe[: max(1, len(oe) // 4)])
    oe_ancient = np.nanmean(oe[-max(1, len(oe) // 4):])
    lin_rising = lin_ancient > 1.5 * lin_recent
    oe_holding = oe_ancient >= 0.8 * oe_recent  # doesn't collapse
    if lin_rising and oe_holding:
        return "A — clean handoff (linear distance rising, O/E holding)"
    if lin_rising and not oe_holding:
        return "B — parallel decay (linear distance rising, O/E collapsing)"
    if not lin_rising:
        return "C-candidate — linear distance not rising, re-check subtype ladder orientation"
    return "UNDETERMINED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair-table",
        type=Path,
        default=Path("data/pair_table_gm12878_dpnII_heavy.parquet"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    ap.add_argument("--tag", default=None,
                    help="Figure file tag. Default: derived from pair-table name.")
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    pt = rel(args.pair_table)
    out_dir = rel(args.out_dir)
    tag = args.tag or pt.stem.replace("pair_table_", "").replace("_heavy", "")

    df = load_pair_table(pt)
    summary = bin_by_subtype(df)

    # Print the summary as stdout so we can eyeball it without opening a PDF.
    print("\n[panel1] summary by subtype:")
    print(summary.to_string(index=False))
    print()

    # Same-channel subset, if any
    sc = df[df.same_channel].copy() if df.same_channel.any() else pd.DataFrame()
    if len(sc):
        print(f"[panel1] same-channel pairs: {len(sc)}")
        print(sc.groupby("subtype", observed=True)[["hic_obs", "hic_oe"]].agg(
            ["count", "median"]
        ).to_string())
        print()

    outcome = classify_outcome(summary)
    print(f"[panel1] outcome: {outcome}")

    out_path = out_dir / f"panel1_dual_proximity_{tag}.png"
    plot_panel1(summary, sc, out_path)

    # Also write the summary table next to the figure for downstream reuse.
    summary_path = out_dir / f"panel1_summary_{tag}.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[panel1] wrote {summary_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
