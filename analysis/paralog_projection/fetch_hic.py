"""Fetch Hi-C contact data for the paralog projection test.

Default source is the Rao 2014 GM12878 dataset (GSE63525), the field-standard
contact map at 5-25 kb resolution. For the first-pass analysis we use the
published loop calls (BEDPE) rather than the full contact matrix: loops are
high-confidence contact pairs already thresholded and annotated, and the file
is under 1 MB versus gigabytes for the raw matrix.

Loops give us the yes/no "is this pair in contact" signal per pair. The full
contact matrix is needed later for the continuous contact-frequency axis.

Usage:
  python fetch_hic.py --out data/gm12878_loops.bedpe
  python fetch_hic.py --resolution 10kb --out data/gm12878_loops_10kb.bedpe

WARNING: if --matrix is set, the full contact matrix download is multiple GB.
"""

import argparse
import sys
import urllib.request
from pathlib import Path


# Rao 2014 loop calls from the GSE63525 supplementary files on GEO.
# These URLs are the public ftp links; verify before first run.
GSE63525_LOOPS = {
    "gm12878_primary": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
        "GSE63525_GM12878_primary%2Breplicate_HiCCUPS_looplist.txt.gz"
    ),
    "imr90": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
        "GSE63525_IMR90_HiCCUPS_looplist.txt.gz"
    ),
    "k562": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
        "GSE63525_K562_HiCCUPS_looplist.txt.gz"
    ),
}


def fetch(url: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "paralog-projection/0.1"})
    n = 0
    with urllib.request.urlopen(req, timeout=600) as resp, out_path.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
            n += len(chunk)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cell-line",
        default="gm12878_primary",
        choices=sorted(GSE63525_LOOPS.keys()),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    url = GSE63525_LOOPS[args.cell_line]
    here = Path(__file__).parent
    out = args.out or (here / f"data/{args.cell_line}_loops.txt.gz")
    out = out if out.is_absolute() else here / out

    print(f"[fetch_hic] downloading {url}", file=sys.stderr)
    n = fetch(url, out)
    print(f"[fetch_hic] wrote {out} ({n / 1e3:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
