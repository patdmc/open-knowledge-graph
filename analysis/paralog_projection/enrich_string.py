"""Enrich a paralog pair table with STRING PPI Jaccard similarity.

STRING is a curated protein-protein interaction database. Two genes are
"close in the interaction graph" if they share many binding partners —
this is a different projection from sequence identity or co-essentiality.
We measure it as Jaccard similarity of the two genes' neighbor sets.

Data: STRING v12.0 (or latest), species 9606 (human), combined_score ≥ 700.
Downloaded fresh from the STRING API if not already cached.

  https://string-db.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
  https://string-db.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz

Aliases file maps STRING identifiers (9606.ENSPxxxxx) to HUGO symbols /
Ensembl gene ids so we can merge against the paralog table.

Usage:
  python enrich_string.py \\
      --pair-table data/pair_table_gm12878_primary_heavy_coess.parquet \\
      --out data/pair_table_gm12878_primary_heavy_coess_string.parquet
"""

import argparse
import gzip
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


STRING_VERSION = "12.0"
LINKS_URL = (
    f"https://stringdb-downloads.org/download/protein.links.v{STRING_VERSION}/"
    f"9606.protein.links.v{STRING_VERSION}.txt.gz"
)
ALIASES_URL = (
    f"https://stringdb-downloads.org/download/protein.aliases.v{STRING_VERSION}/"
    f"9606.protein.aliases.v{STRING_VERSION}.txt.gz"
)


def download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[string] cached {out_path} ({out_path.stat().st_size/1e6:.1f} MB)",
              file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[string] downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "paralog-projection/0.1"})
    with urllib.request.urlopen(req, timeout=600) as resp, out_path.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    print(f"[string] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)",
          file=sys.stderr)


def load_aliases(aliases_gz: Path) -> dict[str, str]:
    """Map STRING protein id (9606.ENSPxxx) → HUGO gene symbol.

    The aliases file has columns: string_protein_id, alias, source.
    We prefer aliases sourced from 'Ensembl_HGNC_symbol' or 'BioMart_HUGO',
    falling back to the first Ensembl_gene_name entry.
    """
    print(f"[string] parsing {aliases_gz}", file=sys.stderr)
    primary: dict[str, str] = {}
    fallback: dict[str, str] = {}
    with gzip.open(aliases_gz, "rt") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            stid, alias, source = parts[0], parts[1], parts[2]
            if "HGNC_symbol" in source or "HUGO" in source:
                primary[stid] = alias
            elif "gene_name" in source.lower() and stid not in fallback:
                fallback[stid] = alias
    merged = {**fallback, **primary}  # primary takes precedence
    print(f"[string] {len(merged)} protein→symbol mappings", file=sys.stderr)
    return merged


def load_neighbors(
    links_gz: Path, id_to_sym: dict[str, str], min_score: int = 700
) -> dict[str, set[str]]:
    """Build symbol → set(neighbor symbols) from the links file.

    STRING links file columns: protein1, protein2, combined_score (0-1000).
    Keep only edges where both proteins map to a symbol and the score
    passes the threshold.
    """
    print(f"[string] parsing {links_gz} with min_score={min_score}", file=sys.stderr)
    t0 = time.time()
    neighbors: dict[str, set[str]] = defaultdict(set)
    n_edges = 0
    n_kept = 0
    with gzip.open(links_gz, "rt") as fh:
        header = fh.readline()
        for line in fh:
            n_edges += 1
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 3:
                continue
            a, b, score = parts[0], parts[1], parts[2]
            try:
                s = int(score)
            except ValueError:
                continue
            if s < min_score:
                continue
            sa = id_to_sym.get(a)
            sb = id_to_sym.get(b)
            if not sa or not sb:
                continue
            neighbors[sa].add(sb)
            neighbors[sb].add(sa)
            n_kept += 1
    print(
        f"[string] kept {n_kept:,}/{n_edges:,} edges; "
        f"{len(neighbors)} genes with ≥1 neighbor ({time.time()-t0:.1f}s)",
        file=sys.stderr,
    )
    return neighbors


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return np.nan
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return np.nan
    return inter / union


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-table", type=Path, required=True)
    ap.add_argument(
        "--string-dir", type=Path, default=Path("data/string")
    )
    ap.add_argument("--min-score", type=int, default=700)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    string_dir = rel(args.string_dir)
    links_path = string_dir / f"9606.protein.links.v{STRING_VERSION}.txt.gz"
    aliases_path = string_dir / f"9606.protein.aliases.v{STRING_VERSION}.txt.gz"

    download_if_missing(LINKS_URL, links_path)
    download_if_missing(ALIASES_URL, aliases_path)

    id_to_sym = load_aliases(aliases_path)
    neighbors = load_neighbors(links_path, id_to_sym, min_score=args.min_score)

    print("[string] loading pair table", file=sys.stderr)
    pairs = pd.read_parquet(rel(args.pair_table))
    n = len(pairs)
    print(f"[string] {n:,} pairs", file=sys.stderr)

    print("[string] computing per-pair jaccard", file=sys.stderr)
    t0 = time.time()
    out = np.full(n, np.nan, dtype=np.float32)
    ga = pairs.gene_a.values
    gb = pairs.gene_b.values
    for i in range(n):
        a = neighbors.get(ga[i])
        b = neighbors.get(gb[i])
        if a is not None and b is not None:
            out[i] = jaccard(a, b)
        if i % 200_000 == 0 and i > 0:
            print(f"[string] {i:,}/{n:,} ({time.time()-t0:.1f}s)", file=sys.stderr)

    pairs = pairs.copy()
    pairs["ppi_jaccard"] = out

    n_defined = int(np.isfinite(out).sum())
    print(f"[string] pairs with ppi_jaccard: {n_defined:,}", file=sys.stderr)
    if n_defined:
        print(
            f"[string] distribution: "
            f"median={np.nanmedian(out):.3f} "
            f"q75={np.nanquantile(out,0.75):.3f} "
            f"q95={np.nanquantile(out,0.95):.3f} "
            f"max={np.nanmax(out):.3f}",
            file=sys.stderr,
        )

    if "same_channel" in pairs.columns:
        sc = pairs[pairs.same_channel & pairs.ppi_jaccard.notna()]
        if len(sc):
            print(f"[string] same-channel (n={len(sc)}): "
                  f"median={sc.ppi_jaccard.median():.3f}", file=sys.stderr)

    out_path = rel(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"[string] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
