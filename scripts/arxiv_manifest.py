#!/usr/bin/env python3
"""
arxiv_manifest.py — Find which tarballs to download for papers in the graph.

Reads the arXiv manifest JSONL where each line is:
    {"tarball_path": ["YYMM/categoryYYMMNNN.pdf", ...]}

Cross-references against REF nodes in the graph to produce:
  - A download list (tarball URLs or paths)
  - A per-tarball extraction list (which PDFs to pull out)

Usage:
    python scripts/arxiv_manifest.py <manifest.jsonl> [options]

Options:
    --ref-dir     Directory containing REF YAML nodes
                  Default: knowledge-graph/nodes/references/
    --base-url    Base URL to prepend to tarball paths for download
                  Default: https://arxiv.org/e-print/
    --out         Output file for download script (default: print to stdout)
    --format      Output format: shell | list | json  (default: shell)
    --dry-run     Print summary without writing output file

Examples:
    # Print shell download script
    python scripts/arxiv_manifest.py manifest.jsonl

    # Save as a shell script
    python scripts/arxiv_manifest.py manifest.jsonl --out download.sh

    # Just a plain list of tarball paths needed
    python scripts/arxiv_manifest.py manifest.jsonl --format list

    # JSON map of {tarball: [pdf_paths_needed]}
    python scripts/arxiv_manifest.py manifest.jsonl --format json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_REF_DIR = REPO_ROOT / "knowledge-graph" / "nodes" / "references"
DEFAULT_BASE_URL = "https://arxiv.org/e-print/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_graph_arxiv_ids(ref_dir: Path) -> set[str]:
    """Return the set of all arXiv IDs present in REF nodes."""
    ids = set()
    for yaml_path in ref_dir.glob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        m = re.search(r"arxiv_id:\s*[\"']?([^\s\"'\n]+)[\"']?", text)
        if m:
            aid = m.group(1).strip()
            if aid:
                ids.add(aid)
    return ids


def pdf_path_to_arxiv_id(pdf_path: str) -> str | None:
    """
    Convert a manifest PDF path to a normalised arXiv ID.

    Examples:
      0001/astro-ph0001001.pdf  ->  astro-ph/0001001
      2301/2301.04567.pdf       ->  2301.04567
      1234/hep-th1234567.pdf    ->  hep-th/1234567
    """
    filename = Path(pdf_path).stem  # strip directory and .pdf

    # New format: YYMM.NNNNN  (filename is the arXiv ID itself)
    if re.match(r"^\d{4}\.\d{4,5}$", filename):
        return filename

    # Old format: categoryYYMMNNN  e.g. astro-ph0001001, hep-th9212071
    m = re.match(r"^([a-z\-]+)(\d+)$", filename, re.IGNORECASE)
    if m:
        category = m.group(1)
        number = m.group(2)
        return f"{category}/{number}"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find tarballs to download for papers in the knowledge graph."
    )
    parser.add_argument("input", help="Path to arXiv manifest JSONL")
    parser.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Base URL for tarball downloads (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["shell", "list", "json"],
                        default="shell")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only, no output file")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load graph arXiv IDs
    print(f"Scanning REF nodes in {ref_dir} ...", file=sys.stderr)
    graph_ids = load_graph_arxiv_ids(ref_dir)
    print(f"  {len(graph_ids)} arXiv IDs in graph", file=sys.stderr)

    if not graph_ids:
        print("No arXiv IDs in graph. Run arxiv_to_graph.py first.", file=sys.stderr)
        sys.exit(0)

    # Stream manifest, collect needed tarballs
    # {tarball_path: [pdf_paths_needed]}
    needed: dict[str, list[str]] = defaultdict(list)
    matched_ids: set[str] = set()

    lines_read = 0
    errors = 0

    print(f"Streaming {input_path} ...", file=sys.stderr)

    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if args.limit and lines_read >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            lines_read += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            for tarball_path, pdf_list in record.items():
                for pdf_path in pdf_list:
                    arxiv_id = pdf_path_to_arxiv_id(pdf_path)
                    if arxiv_id and arxiv_id in graph_ids:
                        needed[tarball_path].append(pdf_path)
                        matched_ids.add(arxiv_id)

    print(f"  Lines read:     {lines_read}", file=sys.stderr)
    print(f"  Papers matched: {len(matched_ids)}", file=sys.stderr)
    print(f"  Tarballs needed: {len(needed)}", file=sys.stderr)

    # IDs in graph that were NOT found in manifest
    not_found = graph_ids - matched_ids
    if not_found:
        print(f"  Not in manifest: {len(not_found)} paper(s)", file=sys.stderr)

    if errors:
        print(f"  Parse errors:   {errors}", file=sys.stderr)

    print(file=sys.stderr)

    if not needed:
        print("No matching tarballs found.", file=sys.stderr)
        sys.exit(0)

    if args.dry_run:
        print("Tarballs needed:", file=sys.stderr)
        for tarball in sorted(needed):
            print(f"  {tarball}  ({len(needed[tarball])} paper(s))", file=sys.stderr)
        sys.exit(0)

    # Build output
    base_url = args.base_url.rstrip("/") + "/"

    if args.format == "shell":
        lines = ["#!/bin/bash",
                 "# Generated by arxiv_manifest.py",
                 "# Downloads only the tarballs needed for papers in the knowledge graph.",
                 ""]
        for tarball in sorted(needed):
            url = base_url + tarball.lstrip("/")
            lines.append(f"# Contains: {', '.join(needed[tarball][:3])}"
                         + (" ..." if len(needed[tarball]) > 3 else ""))
            lines.append(f"curl -L -O '{url}'")
            lines.append("")
        output = "\n".join(lines)

    elif args.format == "list":
        output = "\n".join(
            base_url + t.lstrip("/") for t in sorted(needed)
        )

    elif args.format == "json":
        # {tarball_url: [pdf_paths]}
        out_dict = {
            base_url + t.lstrip("/"): sorted(v)
            for t, v in sorted(needed.items())
        }
        output = json.dumps(out_dict, indent=2)

    # Write
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(output, encoding="utf-8")
        if args.format == "shell":
            out_path.chmod(0o755)
        print(f"Written to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
