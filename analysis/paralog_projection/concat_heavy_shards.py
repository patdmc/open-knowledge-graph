"""Reduce step: concat per-chrom-pair heavy shards into a single pair table.

Reads every parquet under data/heavy/<cell>/shards/ and writes
data/pair_table_<cell>_heavy.parquet.

Kept intentionally separate from build_pair_table_heavy.py so the reduce
step can be iterated without re-mapping.

Usage:
  python concat_heavy_shards.py --cell-line gm12878_dpnII
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    here = Path(__file__).parent
    shards_dir = here / "data" / "heavy" / args.cell_line / "shards"
    if not shards_dir.exists():
        print(f"error: {shards_dir} does not exist", file=sys.stderr)
        return 2

    out = args.out or (here / "data" / f"pair_table_{args.cell_line}_heavy.parquet")

    shard_files = sorted(shards_dir.glob("*.parquet"))
    if not shard_files:
        print(f"error: no shards in {shards_dir}", file=sys.stderr)
        return 2

    print(f"[concat] {len(shard_files)} shards", file=sys.stderr)
    frames = []
    total_rows = 0
    for sf in shard_files:
        df = pd.read_parquet(sf)
        total_rows += len(df)
        if len(df):
            frames.append(df)
        print(f"[concat] {sf.name}: {len(df)} rows", file=sys.stderr)

    if not frames:
        print("error: every shard is empty", file=sys.stderr)
        return 2

    full = pd.concat(frames, ignore_index=True)

    # Canonicalize orientation so each unordered pair appears exactly once.
    # Intra-chrom shards previously kept both (A,B) and (B,A); inter-chrom
    # shards were already oriented. Normalize everything here.
    n_pre = len(full)
    swap = full.gene_id_a > full.gene_id_b
    if swap.any():
        swap_cols = [
            ("gene_id_a", "gene_id_b"),
            ("gene_a", "gene_b"),
            ("chrom_a", "chrom_b"),
            ("start_a", "start_b"),
            ("end_a", "end_b"),
            ("mid_a", "mid_b"),
            ("channel_a", "channel_b"),
        ]
        for ca, cb in swap_cols:
            if ca in full.columns and cb in full.columns:
                a = full.loc[swap, ca].copy()
                full.loc[swap, ca] = full.loc[swap, cb]
                full.loc[swap, cb] = a
        # perc_id / perc_id_r1 are paired — swap reciprocal meaning along with the genes
        if "perc_id" in full.columns and "perc_id_r1" in full.columns:
            a = full.loc[swap, "perc_id"].copy()
            full.loc[swap, "perc_id"] = full.loc[swap, "perc_id_r1"]
            full.loc[swap, "perc_id_r1"] = a

    full = full.drop_duplicates(subset=["gene_id_a", "gene_id_b"]).reset_index(drop=True)
    n_post = len(full)
    print(f"[concat] canonicalized: {n_pre} → {n_post} unique pairs", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out, index=False)

    n = len(full)
    n_nonzero = int((full.hic_obs > 0).sum())
    n_sc = int(full.same_chrom.sum())
    n_ch = int(full.same_channel.sum())
    n_ch_nonzero = int(((full.same_channel) & (full.hic_obs > 0)).sum())
    print(f"[concat] wrote {out}", file=sys.stderr)
    print(f"[concat] rows: {n}", file=sys.stderr)
    print(f"[concat] hic_obs>0: {n_nonzero} ({100*n_nonzero/n:.1f}%)", file=sys.stderr)
    print(f"[concat] same_chrom: {n_sc}", file=sys.stderr)
    print(f"[concat] same_channel: {n_ch}; of those with obs>0: {n_ch_nonzero}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
