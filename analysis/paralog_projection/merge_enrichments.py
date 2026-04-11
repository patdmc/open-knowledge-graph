"""Merge enrichment parquets into a single master feature-cube parquet.

Each enrich_*.py writes its own parquet: base columns + that enrichment's
new columns. They were run in parallel from the same base table, so any
of them is a valid join source; we pick new columns from each and left-
join them onto the base by (gene_id_a, gene_id_b).

Usage:
  python merge_enrichments.py \\
      --base data/pair_table_gm12878_primary_heavy_coess.parquet \\
      --enrichments \\
          data/pair_table_gm12878_primary_heavy_coess_go.parquet \\
          data/pair_table_gm12878_primary_heavy_coess_string.parquet \\
          data/pair_table_gm12878_primary_heavy_coess_plm.parquet \\
      --out data/pair_table_gm12878_primary_heavy_full.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


JOIN_KEYS = ["gene_id_a", "gene_id_b"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--enrichments", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p

    base = pd.read_parquet(rel(args.base))
    print(f"[merge] base: {len(base):,} rows, {len(base.columns)} cols", file=sys.stderr)
    for c in JOIN_KEYS:
        if c not in base.columns:
            print(f"error: base table missing join key {c}", file=sys.stderr)
            return 2

    base_cols = set(base.columns)
    merged = base
    for ep in args.enrichments:
        ep = rel(ep)
        if not ep.exists():
            print(f"[merge] skipping missing: {ep}", file=sys.stderr)
            continue
        df = pd.read_parquet(ep)
        new_cols = [c for c in df.columns if c not in base_cols and c not in JOIN_KEYS]
        if not new_cols:
            print(f"[merge] {ep.name}: no new columns, skipping", file=sys.stderr)
            continue
        print(f"[merge] {ep.name}: adding {new_cols}", file=sys.stderr)
        merged = merged.merge(
            df[JOIN_KEYS + new_cols], on=JOIN_KEYS, how="left",
        )
        base_cols.update(new_cols)

    print(
        f"[merge] final: {len(merged):,} rows, {len(merged.columns)} cols",
        file=sys.stderr,
    )

    # Report feature coverage
    feature_cols = [
        "perc_id", "coess_corr", "ppi_jaccard",
        "go_bp_sim", "go_mf_sim", "plm_cosine",
        "hic_obs", "hic_oe", "linear_distance_bp",
    ]
    print("[merge] coverage (non-null fraction per feature):", file=sys.stderr)
    for c in feature_cols:
        if c in merged.columns:
            frac = merged[c].notna().mean()
            print(f"  {c:20s}  {merged[c].notna().sum():>10,}  ({100*frac:5.1f}%)",
                  file=sys.stderr)

    out = rel(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"[merge] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
