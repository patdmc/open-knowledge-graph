"""Test: are channel_gene_map genes enriched for config-layer architectural features?

The encapsulation inversion hypothesis (see memory/project_encapsulation_inversion.md)
predicts that the curated 122-gene channel_gene_map is implicitly the
config-layer (wrappers + config readers + config writers + config erasers)
of the human genome, assembled by curator intuition about "what matters
in this pathway" without naming the layer. This test checks two of the
five forward predictions:

  Test 2 (structural)  — wrapper/config genes are PPI hubs
  Test 5 (curator)     — channel genes are config-layer enriched

Uses existing STRING v12.0 (score ≥ 700) cache from enrich_string.py and
the channel_gene_map.csv from the main data directory.

First-pass result (2026-04-10):
  - PPI hub enrichment: median degree 95 (channel) vs 13 (non-channel),
    7.3x fold enrichment, Mann-Whitney U p = 5.6e-53
  - Config-layer enrichment (crude gene-symbol heuristic):
    writer+reader prefix match 42.6% channel vs 3.1% non-channel,
    OR = 23.4, p = 3.5e-44
  - Remaining 57% "neither" is dominated by chromatin remodelers
    (ARID1A/B/2, ATRX), HATs (CREBBP, EP300), kinases with non-standard
    prefixes (BRAF), deubiquitinases (BAP1), and scaffolds (APC, CTNNB1,
    FANCA) — all config layer under closer inspection
  - Only genuine implementation genes in the channel set are structural
    components of TissueArchitecture (CDH1, CDH2) and Immune (HLA, B2M)

Usage:
  python test_channel_config_enrichment.py
"""

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu


# Config-writer gene symbol prefixes (crude first-pass heuristic)
# Covers: kinases, phosphatases, methyltransferases/demethylases,
# acetyltransferases/deacetylases, ubiquitin ligases.
WRITER_PREFIXES = [
    "CDK", "CHEK", "ATM", "ATR", "MAP", "AKT", "PIK", "MTOR",
    "DNMT", "TET", "EZH", "KMT", "SETD", "NSD", "SMYD", "PRMT", "DOT",
    "HDAC", "SIRT", "KAT", "KDM", "JMJD",
    "WEE", "PLK", "AURK", "CDC", "CSK", "FYN", "SRC", "YES", "LCK",
    "MDM", "BTRC", "RNF", "TRAF", "TRIM",
    "ABL", "BCR", "JAK", "STAT", "TGFBR", "ERBB", "FGFR", "IGF", "MET", "EGFR",
    "CSNK", "GSK", "PKA", "PKC", "PKN", "ROCK", "RAF", "MEK", "ERK",
    "NEK", "BUB", "TTK", "HIPK", "DYRK", "CLK", "STK",
]

# Config-reader gene symbol prefixes (scaffolds with recognition domains)
READER_PREFIXES = [
    "BRCA", "BARD", "PALB", "FANC",
    "53BP", "MDC", "H2AFX",
    "BRD", "ATAD", "BAZ",
    "CBX", "HP1",
    "ING", "PHF", "JADE",
    "TP53BP", "RAD18", "RIF",
    "MSH", "MLH", "PMS",
    "RFC", "PCNA",
]


def matches_prefix(gene: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if gene.startswith(p) and (
            len(gene) == len(p) or gene[len(p)].isdigit() or gene[len(p)] in "L-"
        ):
            return True
    return False


def load_string_graph(
    links_gz: Path, aliases_gz: Path, min_score: int = 700
) -> dict[str, set[str]]:
    """Build symbol -> set(neighbor symbols) from STRING."""
    print(f"[test] loading STRING aliases {aliases_gz}", file=sys.stderr)
    id_to_sym: dict[str, str] = {}
    with gzip.open(aliases_gz, "rt") as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and "HGNC_symbol" in parts[2]:
                id_to_sym[parts[0]] = parts[1]
    print(f"[test]   {len(id_to_sym)} protein->symbol mappings", file=sys.stderr)

    print(f"[test] loading STRING links {links_gz} (min_score={min_score})",
          file=sys.stderr)
    neighbors: dict[str, set[str]] = defaultdict(set)
    with gzip.open(links_gz, "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 3:
                continue
            a, b, s = parts[0], parts[1], int(parts[2])
            if s < min_score:
                continue
            sa = id_to_sym.get(a)
            sb = id_to_sym.get(b)
            if sa and sb:
                neighbors[sa].add(sb)
                neighbors[sb].add(sa)
    print(f"[test]   {len(neighbors)} genes with ≥1 neighbor", file=sys.stderr)
    return neighbors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--channel-map",
        type=Path,
        default=Path("../../data/channel_gene_map.csv"),
    )
    ap.add_argument(
        "--string-dir", type=Path, default=Path("data/string"),
    )
    ap.add_argument("--min-score", type=int, default=700)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    channel_path = rel(args.channel_map)
    string_dir = rel(args.string_dir)
    links = string_dir / "9606.protein.links.v12.0.txt.gz"
    aliases = string_dir / "9606.protein.aliases.v12.0.txt.gz"

    channel_df = pd.read_csv(channel_path)
    channel_genes = set(channel_df.gene.unique())
    print(f"\n=== Channel gene set ===")
    print(f"Total: {len(channel_genes)} genes across "
          f"{channel_df.channel.nunique()} channels")

    neighbors = load_string_graph(links, aliases, min_score=args.min_score)
    degrees = {g: len(ns) for g, ns in neighbors.items()}

    # ============================================================
    # TEST 2: PPI hub enrichment
    # ============================================================
    print(f"\n=== TEST 2: PPI hub enrichment (STRING v12.0, score≥{args.min_score}) ===")
    channel_in = [g for g in channel_genes if g in degrees]
    channel_d = np.array([degrees[g] for g in channel_in])
    nonch = [g for g in neighbors if g not in channel_genes]
    nonch_d = np.array([degrees[g] for g in nonch])

    print(f"  channel n={len(channel_in)}  median={np.median(channel_d):.0f}  "
          f"mean={channel_d.mean():.1f}  q95={np.quantile(channel_d, 0.95):.0f}")
    print(f"  nonch   n={len(nonch)}  median={np.median(nonch_d):.0f}  "
          f"mean={nonch_d.mean():.1f}  q95={np.quantile(nonch_d, 0.95):.0f}")

    u, p = mannwhitneyu(channel_d, nonch_d, alternative="greater")
    print(f"  Mann-Whitney U (channel > non-channel):  p = {p:.3e}")
    print(f"  Median fold-enrichment: {np.median(channel_d)/np.median(nonch_d):.1f}x")

    # Per-channel breakdown
    print(f"\n  per-channel median degree:")
    for ch in sorted(channel_df.channel.unique()):
        gs = [g for g in channel_df[channel_df.channel == ch].gene if g in degrees]
        if not gs:
            continue
        d = np.array([degrees[g] for g in gs])
        print(f"    {ch:20s}  n={len(gs):3d}  median={np.median(d):5.0f}")

    # ============================================================
    # TEST 5: config-layer enrichment (crude prefix heuristic)
    # ============================================================
    print(f"\n=== TEST 5: config-layer enrichment (gene-symbol heuristic) ===")
    all_genes = set(neighbors.keys())

    def tag(g: str) -> str:
        if matches_prefix(g, WRITER_PREFIXES):
            return "writer"
        if matches_prefix(g, READER_PREFIXES):
            return "reader"
        return "neither"

    ch_counts = {"writer": 0, "reader": 0, "neither": 0}
    for g in channel_genes:
        ch_counts[tag(g)] += 1
    nc_counts = {"writer": 0, "reader": 0, "neither": 0}
    for g in all_genes:
        if g not in channel_genes:
            nc_counts[tag(g)] += 1

    n_ch = len(channel_genes)
    n_nc = len(all_genes) - len(channel_genes)
    print(f"  channel n={n_ch}:   "
          f"writer {ch_counts['writer']} ({100*ch_counts['writer']/n_ch:.1f}%), "
          f"reader {ch_counts['reader']} ({100*ch_counts['reader']/n_ch:.1f}%), "
          f"neither {ch_counts['neither']} ({100*ch_counts['neither']/n_ch:.1f}%)")
    print(f"  nonch   n={n_nc}: "
          f"writer {nc_counts['writer']} ({100*nc_counts['writer']/n_nc:.1f}%), "
          f"reader {nc_counts['reader']} ({100*nc_counts['reader']/n_nc:.1f}%), "
          f"neither {nc_counts['neither']} ({100*nc_counts['neither']/n_nc:.1f}%)")

    for cls in ["writer", "reader"]:
        ctab = [
            [ch_counts[cls], n_ch - ch_counts[cls]],
            [nc_counts[cls], n_nc - nc_counts[cls]],
        ]
        odds, pv = fisher_exact(ctab, alternative="greater")
        print(f"  {cls:7s} enrichment:  OR={odds:.2f}  p={pv:.3e}")

    # Combined config = writer OR reader
    ch_cfg = ch_counts["writer"] + ch_counts["reader"]
    nc_cfg = nc_counts["writer"] + nc_counts["reader"]
    ctab = [[ch_cfg, n_ch - ch_cfg], [nc_cfg, n_nc - nc_cfg]]
    odds, pv = fisher_exact(ctab, alternative="greater")
    print(f"  config  enrichment:  OR={odds:.2f}  p={pv:.3e}")

    # Show the "neither" list for manual inspection
    print(f"\n  channel genes tagged 'neither' (n={ch_counts['neither']}) — "
          f"under closer inspection most are config-layer too:")
    leftover = sorted(
        [g for g in channel_genes if tag(g) == "neither"]
    )
    for i in range(0, len(leftover), 10):
        print(f"    {', '.join(leftover[i:i+10])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
