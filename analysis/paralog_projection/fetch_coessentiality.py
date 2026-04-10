"""Fetch DepMap CRISPR gene-effect matrix for co-essentiality features.

DepMap releases are versioned. The CRISPRGeneEffect.csv file is the per-cell-line
gene effect matrix (Chronos-scaled). Co-essentiality is computed downstream in
build_pair_table.py as pairwise Pearson correlation across cell lines.

WARNING: the gene-effect matrix is ~150 MB. Do not run without confirming.

Usage:
  python fetch_coessentiality.py --release 24Q4 --out data/depmap_gene_effect.csv

Release download URLs are listed on https://depmap.org/portal/download/all/
and change per release. Pass --url to override if the release URL pattern shifts.
"""

import argparse
import sys
import urllib.request
from pathlib import Path


DEFAULT_URLS = {
    # Fill with known-good URLs as they are verified manually from depmap.org.
    # Left empty so the script fails loudly until a release is chosen.
}


def fetch(url: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "paralog-projection/0.1"})
    bytes_written = 0
    with urllib.request.urlopen(req, timeout=1800) as resp, out_path.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            bytes_written += len(chunk)
    return bytes_written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="24Q4")
    ap.add_argument("--url", default=None, help="Override URL for the gene-effect CSV")
    ap.add_argument("--out", type=Path, default=Path("data/depmap_gene_effect.csv"))
    args = ap.parse_args()

    url = args.url or DEFAULT_URLS.get(args.release)
    if not url:
        print(
            f"[fetch_coessentiality] no URL for release {args.release}. "
            "Pass --url with the CRISPRGeneEffect.csv download link from "
            "https://depmap.org/portal/download/all/",
            file=sys.stderr,
        )
        return 2

    here = Path(__file__).parent
    out = args.out if args.out.is_absolute() else here / args.out

    print(f"[fetch_coessentiality] downloading {url}", file=sys.stderr)
    n = fetch(url, out)
    print(f"[fetch_coessentiality] wrote {out} ({n / 1e6:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
