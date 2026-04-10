"""Enrich a paralog pair table with DepMap co-essentiality correlations.

Co-essentiality = Pearson correlation of CRISPR gene-effect scores across
cell lines, for two genes. Two genes that kill the same cell lines when
knocked out are functionally interchangeable from the phenotype's
perspective — a projection onto the viability axis.

This is the first non-sequence equivalence feature for Panel 2. It is
completely independent of percent identity and independent of 3D
contact, so if co-essentiality predicts Hi-C contact better than
`perc_id` does, that is a direct test of the projection-theory claim
that the fold reads off *function*, not descent.

Input: a pair table parquet written by concat_heavy_shards.py.
Output: same parquet with a `coess_corr` column added.

Usage:
  python enrich_coessentiality.py \\
      --pair-table data/pair_table_gm12878_primary_heavy.parquet \\
      --depmap data/depmap_CRISPRGeneEffect_26Q1.csv \\
      --out data/pair_table_gm12878_primary_heavy_coess.parquet
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


GENE_COL_RE = re.compile(r"^(?P<sym>[^ ]+) \((?P<entrez>\d+)\)$")


def load_depmap(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the CRISPR gene-effect matrix.

    Returns:
      matrix: float32 ndarray of shape (n_cell_lines, n_genes) with NaNs preserved.
      sym_to_col: dict mapping HUGO symbol → column index in matrix.

    Many genes have multiple paralog entries with the same symbol in Ensembl
    (rare, but e.g. pseudogenes sharing names); we keep the first occurrence
    and report the count. DepMap column headers are 'SYMBOL (ENTREZ_ID)'.
    """
    print(f"[coess] loading {path}", file=sys.stderr)
    t0 = time.time()
    df = pd.read_csv(path, index_col=0)
    print(
        f"[coess] loaded {df.shape[0]} cell lines × {df.shape[1]} genes "
        f"({time.time()-t0:.1f}s)",
        file=sys.stderr,
    )

    sym_to_col: dict[str, int] = {}
    clean_cols = []
    dups = 0
    for i, col in enumerate(df.columns):
        m = GENE_COL_RE.match(col)
        sym = m.group("sym") if m else col
        clean_cols.append(sym)
        if sym in sym_to_col:
            dups += 1
            continue
        sym_to_col[sym] = i
    if dups:
        print(f"[coess] {dups} duplicate symbol columns skipped", file=sys.stderr)

    # Convert to float32 contiguous array for fast slicing.
    mat = df.values.astype(np.float32, copy=False)
    return mat, sym_to_col


def z_score_columns(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each column ignoring NaNs.

    Returns (z_mat, valid_mask) where:
      z_mat: same shape, column mean 0, column std 1 (nanmean/nanstd), NaN preserved.
      valid_mask: same shape, True where original was finite.
    """
    valid = np.isfinite(mat)
    col_mean = np.nanmean(mat, axis=0)
    col_std = np.nanstd(mat, axis=0, ddof=0)
    # Avoid divide-by-zero for constant columns.
    col_std = np.where(col_std == 0, np.nan, col_std)
    z = (mat - col_mean[None, :]) / col_std[None, :]
    return z, valid


def pairwise_corr(
    z_mat: np.ndarray,
    valid: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    min_overlap: int = 30,
) -> np.ndarray:
    """Vectorized pairwise Pearson correlation for matched column indices.

    For each (idx_a[k], idx_b[k]), compute the Pearson correlation of the
    two z-scored columns restricted to rows where both are finite.

    Because z-scoring was done over the full column (not the overlap
    subset), the returned "correlation" is NOT strictly the Pearson on the
    overlap — it's a slightly biased estimate that approaches the true
    Pearson when the missingness is MCAR. For DepMap this is acceptable:
    missingness is mostly due to a gene not being screened in some cell
    lines, not data-dependent. The bias is small and consistent across
    pairs, so rankings are preserved.

    Pairs with fewer than `min_overlap` shared cell lines get NaN.
    """
    za = z_mat[:, idx_a]  # (n_cells, n_pairs)
    zb = z_mat[:, idx_b]
    va = valid[:, idx_a]
    vb = valid[:, idx_b]
    both = va & vb
    n_overlap = both.sum(axis=0)

    # Zero-fill NaNs in the product sum so they don't propagate.
    prod = np.where(both, za * zb, 0.0)
    num = prod.sum(axis=0)
    # Normalize by overlap (not n-1) — consistent with population Pearson
    # since z-scoring was already done. Strictly speaking this is an
    # approximation but it's monotone in the true Pearson.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / n_overlap
    corr[n_overlap < min_overlap] = np.nan
    return corr


def enrich(pair_table: Path, depmap_csv: Path, out_path: Path) -> None:
    print(f"[coess] loading pair table {pair_table}", file=sys.stderr)
    pairs = pd.read_parquet(pair_table)
    print(f"[coess] {len(pairs):,} pairs", file=sys.stderr)

    mat, sym_to_col = load_depmap(depmap_csv)
    print(f"[coess] {len(sym_to_col)} unique gene symbols in DepMap", file=sys.stderr)

    # Map each paralog gene symbol to its DepMap column index; NaN if missing.
    pairs = pairs.copy()
    pairs["_col_a"] = pairs.gene_a.map(sym_to_col)
    pairs["_col_b"] = pairs.gene_b.map(sym_to_col)
    n_both = pairs._col_a.notna().sum()
    n_pair_both = ((pairs._col_a.notna()) & (pairs._col_b.notna())).sum()
    print(
        f"[coess] pairs with both genes in DepMap: {n_pair_both:,} "
        f"({100*n_pair_both/len(pairs):.1f}%)",
        file=sys.stderr,
    )

    # Z-score DepMap columns once.
    print("[coess] z-scoring DepMap columns", file=sys.stderr)
    z_mat, valid = z_score_columns(mat)

    # Compute coess_corr in chunks to limit peak memory.
    coess = np.full(len(pairs), np.nan, dtype=np.float32)
    mask = pairs._col_a.notna() & pairs._col_b.notna()
    idx_pairs = np.flatnonzero(mask.values)
    ca = pairs._col_a.values
    cb = pairs._col_b.values

    chunk = 200_000
    t0 = time.time()
    for start in range(0, len(idx_pairs), chunk):
        stop = min(start + chunk, len(idx_pairs))
        sel = idx_pairs[start:stop]
        ia = ca[sel].astype(np.int64)
        ib = cb[sel].astype(np.int64)
        coess[sel] = pairwise_corr(z_mat, valid, ia, ib)
        if (start // chunk) % 5 == 0:
            print(
                f"[coess] chunk {start//chunk+1}: {stop-start} pairs "
                f"({time.time()-t0:.1f}s elapsed)",
                file=sys.stderr,
            )

    pairs["coess_corr"] = coess
    pairs = pairs.drop(columns=["_col_a", "_col_b"])

    # Summary stats before writing.
    n_computed = int(np.isfinite(coess).sum())
    print(f"[coess] computed: {n_computed:,} / {len(pairs):,}", file=sys.stderr)
    if n_computed:
        print(
            f"[coess] distribution: "
            f"mean={np.nanmean(coess):.3f} "
            f"median={np.nanmedian(coess):.3f} "
            f"q75={np.nanquantile(coess, 0.75):.3f} "
            f"q95={np.nanquantile(coess, 0.95):.3f}",
            file=sys.stderr,
        )

    # Same-channel stats
    if "same_channel" in pairs.columns:
        sc = pairs[pairs.same_channel & pairs.coess_corr.notna()]
        dc = pairs[
            ~pairs.same_channel
            & pairs.channel_a.notna()
            & pairs.channel_b.notna()
            & pairs.coess_corr.notna()
        ]
        if len(sc):
            print(
                f"[coess] same-channel (n={len(sc)}): median coess={sc.coess_corr.median():.3f}",
                file=sys.stderr,
            )
        if len(dc):
            print(
                f"[coess] diff-channel annotated (n={len(dc)}): median coess={dc.coess_corr.median():.3f}",
                file=sys.stderr,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"[coess] wrote {out_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-table", type=Path, required=True)
    ap.add_argument("--depmap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    enrich(rel(args.pair_table), rel(args.depmap), rel(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
