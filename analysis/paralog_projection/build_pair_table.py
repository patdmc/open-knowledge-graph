"""Build the master paralog-pair table for the projection test.

Joins:
  - Ensembl paralogs with gene coordinates (fetch_paralogs.py)
  - Channel membership (../../data/channel_gene_map.csv)
  - Hi-C loop calls (fetch_hic.py) — boolean in-loop flag per pair
  - Co-essentiality correlation (fetch_coessentiality.py) — optional, pairwise

Output columns:
  gene_a, gene_b,
  chrom_a, chrom_b, start_a, start_b, mid_a, mid_b,
  same_chrom, linear_distance_bp,
  perc_id, perc_id_r1, orthology_type, subtype,
  channel_a, channel_b, same_channel,
  in_hic_loop, coess_corr

Usage:
  python build_pair_table.py --out data/pair_table_gm12878.parquet
  python build_pair_table.py --hic-loops data/imr90_loops.txt.gz --out data/pair_table_imr90.parquet
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


PARALOG_COLS = {
    "Gene stable ID": "gene_id_a",
    "Gene name": "gene_a",
    "Chromosome/scaffold name": "chrom_a",
    "Gene start (bp)": "start_a",
    "Gene end (bp)": "end_a",
    "Human paralogue gene stable ID": "gene_id_b",
    "Human paralogue associated gene name": "gene_b",
    "Human paralogue chromosome/scaffold name": "chrom_b",
    "Human paralogue chromosome/scaffold start (bp)": "start_b",
    "Human paralogue chromosome/scaffold end (bp)": "end_b",
    "Paralogue %id. target Human gene identical to query gene": "perc_id",
    "Paralogue %id. query gene identical to target Human gene": "perc_id_r1",
    "Human paralogue homology type": "orthology_type",
    "Paralogue last common ancestor with Human": "subtype",
}


def load_paralogs(path: Path) -> pd.DataFrame:
    print(f"[build] reading {path}", file=sys.stderr)
    df = pd.read_csv(path, sep="\t", dtype={
        "Chromosome/scaffold name": str,
        "Human paralogue chromosome/scaffold name": str,
    })
    df = df.rename(columns=PARALOG_COLS)
    # Drop unpaired rows (genes with no paralog) and non-canonical chroms.
    df = df.dropna(subset=["gene_id_b"])
    n_before = len(df)
    df = df[df.chrom_a.isin(CANONICAL_CHROMS) & df.chrom_b.isin(CANONICAL_CHROMS)].copy()
    print(f"[build] paralogs: {n_before} → {len(df)} after canonical-chrom filter", file=sys.stderr)
    # Gene midpoints for Hi-C containment test.
    df["mid_a"] = ((df.start_a + df.end_a) // 2).astype(np.int64)
    df["mid_b"] = ((df.start_b + df.end_b) // 2).astype(np.int64)
    df["same_chrom"] = df.chrom_a == df.chrom_b
    df["linear_distance_bp"] = np.where(
        df.same_chrom,
        (df.mid_a - df.mid_b).abs(),
        np.nan,
    )
    return df


def attach_channels(pairs: pd.DataFrame, channel_map: Path) -> pd.DataFrame:
    print(f"[build] reading {channel_map}", file=sys.stderr)
    ch = pd.read_csv(channel_map)
    # channel_map has (gene, channel) — may list a gene in multiple channels.
    # Collapse to one channel per gene for the same_channel test; keep the
    # lexicographically first for determinism. (Revisit if moonlighting matters.)
    ch = ch.drop_duplicates("gene").rename(columns={"gene": "g"})
    pairs = pairs.merge(
        ch.rename(columns={"g": "gene_a", "channel": "channel_a"}),
        on="gene_a",
        how="left",
    )
    pairs = pairs.merge(
        ch.rename(columns={"g": "gene_b", "channel": "channel_b"}),
        on="gene_b",
        how="left",
    )
    pairs["same_channel"] = (
        pairs.channel_a.notna()
        & pairs.channel_b.notna()
        & (pairs.channel_a == pairs.channel_b)
    )
    n_ch = pairs.same_channel.sum()
    print(f"[build] same-channel pairs: {n_ch}", file=sys.stderr)
    return pairs


def load_loops(path: Path) -> pd.DataFrame:
    print(f"[build] reading {path}", file=sys.stderr)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", dtype={"chr1": str, "chr2": str})
    # Rao HiCCUPS output uses 1/10/X style, matching Ensembl.
    needed = ["chr1", "x1", "x2", "chr2", "y1", "y2"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"loops file missing columns: {missing}")
    df = df[needed].copy()
    # Enforce intra-chrom only; HiCCUPS output is already intra but be defensive.
    df = df[df.chr1 == df.chr2].reset_index(drop=True)
    df["x1"] = df.x1.astype(np.int64)
    df["x2"] = df.x2.astype(np.int64)
    df["y1"] = df.y1.astype(np.int64)
    df["y2"] = df.y2.astype(np.int64)
    print(f"[build] loops (intra): {len(df)}", file=sys.stderr)
    return df


def mark_hic_loop_contact(pairs: pd.DataFrame, loops: pd.DataFrame) -> np.ndarray:
    """Return boolean mask (len = len(pairs)): True if a pair's gene midpoints
    fall inside any Hi-C loop bounding box.

    Loops are intra-chromosomal. For each same-chrom pair we check whether
    (mid_a, mid_b) or (mid_b, mid_a) lies inside some loop's (x1..x2) x (y1..y2).
    Either orientation counts — HiCCUPS records only the upper-triangle anchor.

    Positional-indexing throughout — no pandas .index gotchas.
    """
    n = len(pairs)
    in_loop = np.zeros(n, dtype=bool)
    same_chrom = pairs.same_chrom.values
    chrom_a = pairs.chrom_a.values
    mid_a = pairs.mid_a.values
    mid_b = pairs.mid_b.values

    loops_by_chrom: dict[str, dict[str, np.ndarray]] = {}
    for chrom, g in loops.groupby("chr1", sort=False):
        loops_by_chrom[chrom] = {
            "x1": g.x1.values,
            "x2": g.x2.values,
            "y1": g.y1.values,
            "y2": g.y2.values,
        }

    # Unique same-chrom chromosomes in pairs
    sc_positions = np.flatnonzero(same_chrom)
    if sc_positions.size == 0:
        return in_loop

    for chrom in np.unique(chrom_a[sc_positions]):
        if chrom not in loops_by_chrom:
            continue
        lg = loops_by_chrom[chrom]
        x1, x2, y1, y2 = lg["x1"], lg["x2"], lg["y1"], lg["y2"]
        # Positions in the full pairs array that are same-chrom and on this chrom
        positions = sc_positions[chrom_a[sc_positions] == chrom]
        if positions.size == 0:
            continue
        ma = mid_a[positions]
        mb = mid_b[positions]
        hit = np.zeros(positions.size, dtype=bool)
        chunk = 20000
        for start in range(0, positions.size, chunk):
            stop = min(start + chunk, positions.size)
            mas = ma[start:stop, None]
            mbs = mb[start:stop, None]
            h1 = (
                (x1[None, :] <= mas)
                & (mas <= x2[None, :])
                & (y1[None, :] <= mbs)
                & (mbs <= y2[None, :])
            ).any(axis=1)
            h2 = (
                (x1[None, :] <= mbs)
                & (mbs <= x2[None, :])
                & (y1[None, :] <= mas)
                & (mas <= y2[None, :])
            ).any(axis=1)
            hit[start:stop] = h1 | h2
        in_loop[positions] = hit
    return in_loop


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
        default=Path("data/gm12878_loops.txt.gz"),
    )
    ap.add_argument(
        "--coess",
        type=Path,
        default=None,
        help="Optional DepMap gene-effect CSV. Not wired yet.",
    )
    ap.add_argument("--out", type=Path, default=Path("data/pair_table.parquet"))
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    paralogs_path = rel(args.paralogs)
    channels_path = rel(args.channels)
    loops_path = rel(args.hic_loops)
    out_path = rel(args.out)

    pairs = load_paralogs(paralogs_path)
    pairs = attach_channels(pairs, channels_path)
    loops = load_loops(loops_path)
    pairs["in_hic_loop"] = mark_hic_loop_contact(pairs, loops)
    pairs["coess_corr"] = np.nan  # placeholder until fetch_coessentiality runs

    # Summary stats before writing.
    n = len(pairs)
    n_sc = int(pairs.same_chrom.sum())
    n_loop = int(pairs.in_hic_loop.sum())
    n_loop_sc = int(pairs.loc[pairs.same_chrom, "in_hic_loop"].sum())
    n_ch = int(pairs.same_channel.sum())
    n_ch_loop = int(pairs.loc[pairs.same_channel, "in_hic_loop"].sum())
    print(f"[build] rows: {n}", file=sys.stderr)
    print(f"[build] same_chrom: {n_sc} ({100*n_sc/n:.1f}%)", file=sys.stderr)
    print(f"[build] in_hic_loop: {n_loop} (same-chrom only: {n_loop_sc})", file=sys.stderr)
    print(f"[build] same_channel: {n_ch}; of those in loop: {n_ch_loop}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"[build] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
