"""Fetch human paralog pairs from Ensembl BioMart.

Pulls the full human paralog table in one BioMart query. Output columns:
  gene_id, gene_name, chromosome, start, end,
  paralog_id, paralog_name, paralog_chromosome, paralog_start, paralog_end,
  perc_id, perc_id_reciprocal, orthology_type, subtype

Note: Ensembl removed dN/dS from the BioMart paralog schema in recent releases.
The divergence axis for panel 1 uses two substitutes that ARE in the current
schema and are arguably cleaner for presentation:

  subtype  — last common ancestor on the species tree (named discrete ladder,
             e.g. Homo sapiens / Primates / Eutheria / Mammalia / Vertebrata).
             Use for named divergence bins on the x-axis.
  perc_id  — paralog percent identity. Use for continuous within-bin sorting
             and as a monotone proxy for Ks.

If true Ks is needed later, pull from the Ensembl Compara FTP homology TSVs
in a separate script (heavier lift, multi-GB).

Usage:
  python fetch_paralogs.py --out data/paralogs.tsv
  python fetch_paralogs.py --out data/paralogs.tsv --mart-host http://www.ensembl.org
"""

import argparse
import sys
import urllib.request
import urllib.parse
from pathlib import Path


BIOMART_QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="ensembl_gene_id" />
    <Attribute name="external_gene_name" />
    <Attribute name="chromosome_name" />
    <Attribute name="start_position" />
    <Attribute name="end_position" />
    <Attribute name="hsapiens_paralog_ensembl_gene" />
    <Attribute name="hsapiens_paralog_associated_gene_name" />
    <Attribute name="hsapiens_paralog_chromosome" />
    <Attribute name="hsapiens_paralog_chrom_start" />
    <Attribute name="hsapiens_paralog_chrom_end" />
    <Attribute name="hsapiens_paralog_perc_id" />
    <Attribute name="hsapiens_paralog_perc_id_r1" />
    <Attribute name="hsapiens_paralog_orthology_type" />
    <Attribute name="hsapiens_paralog_subtype" />
  </Dataset>
</Query>"""


def fetch(mart_host: str, out_path: Path) -> int:
    url = f"{mart_host.rstrip('/')}/biomart/martservice"
    data = urllib.parse.urlencode({"query": BIOMART_QUERY}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with urllib.request.urlopen(req, timeout=600) as resp, out_path.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
            rows += chunk.count(b"\n")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/paralogs.tsv"))
    ap.add_argument("--mart-host", default="http://www.ensembl.org")
    args = ap.parse_args()

    here = Path(__file__).parent
    out = args.out if args.out.is_absolute() else here / args.out

    print(f"[fetch_paralogs] querying BioMart at {args.mart_host}", file=sys.stderr)
    rows = fetch(args.mart_host, out)
    print(f"[fetch_paralogs] wrote {out} ({rows} lines incl. header)", file=sys.stderr)

    # Sanity check: first two lines
    with out.open() as fh:
        for i, line in enumerate(fh):
            if i >= 2:
                break
            print(f"[fetch_paralogs] {line.rstrip()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
