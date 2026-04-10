"""Build the master paralog-pair table for the projection test.

Joins:
  - Ensembl paralogs with Ks and gene coordinates (fetch_paralogs.py)
  - Channel membership (../../data/channel_gene_map.csv)
  - Hi-C loop calls (fetch_hic.py) — boolean in-loop flag per pair
  - Co-essentiality correlation (fetch_coessentiality.py) — optional, pairwise

Output columns:
  gene_a, gene_b,
  chrom_a, chrom_b, start_a, start_b,
  same_chrom, linear_distance_bp,
  ks, dn, perc_id, subtype,
  channel_a, channel_b, same_channel,
  in_hic_loop, coess_corr
  bin_ks  (pre-binned for plotting)

This script does not fit models or generate figures. It only builds the
joined table. Downstream scripts (not yet written) will consume it for
Panel 1 (dual-proximity handoff), Panel 2 (equivalence-class AUROC), and
Panel 3 (channel-stratified elbow).

Usage:
  python build_pair_table.py --out data/pair_table.parquet
"""

import argparse
import sys
from pathlib import Path


# TODO implement once fetches have been run and input files exist.
# Leaving as scaffold so the pipeline shape is visible end-to-end before
# any data is pulled. The join logic is straightforward pandas merges:
#
#   paralogs = pd.read_csv(paralogs_tsv, sep="\t")
#   channels = pd.read_csv(channel_map_csv)
#   loops    = read_loops_bedpe(hic_loops_path)
#
#   pairs = paralogs.rename(...)  # normalize column names
#   pairs["same_chrom"] = pairs.chrom_a == pairs.chrom_b
#   pairs["linear_distance_bp"] = (
#       (pairs.start_a - pairs.start_b).abs().where(pairs.same_chrom)
#   )
#   pairs = pairs.merge(channels, left_on="gene_a", right_on="gene",
#                       how="left").rename(columns={"channel": "channel_a"})
#   pairs = pairs.merge(channels, left_on="gene_b", right_on="gene",
#                       how="left").rename(columns={"channel": "channel_b"})
#   pairs["same_channel"] = pairs.channel_a == pairs.channel_b
#   pairs["in_hic_loop"] = pair_in_loop(pairs, loops)  # spatial join
#   pairs["bin_ks"] = pd.cut(pairs.ks, bins=KS_BINS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paralogs", type=Path, default=Path("data/paralogs.tsv"))
    ap.add_argument(
        "--channels",
        type=Path,
        default=Path("../../data/channel_gene_map.csv"),
    )
    ap.add_argument(
        "--hic-loops",
        type=Path,
        default=Path("data/gm12878_primary_loops.txt.gz"),
    )
    ap.add_argument(
        "--coess",
        type=Path,
        default=None,
        help="Optional DepMap gene-effect CSV. Skip if not provided.",
    )
    ap.add_argument("--out", type=Path, default=Path("data/pair_table.parquet"))
    args = ap.parse_args()

    print(
        "[build_pair_table] scaffold — join logic pending first fetch run",
        file=sys.stderr,
    )
    print(f"[build_pair_table] inputs: {args.paralogs} {args.channels} {args.hic_loops}", file=sys.stderr)
    print(f"[build_pair_table] output: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
