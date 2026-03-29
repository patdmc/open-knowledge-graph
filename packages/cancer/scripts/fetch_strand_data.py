#!/usr/bin/env python3
"""
Fetch DNA strand information (+1/-1) for all genes in CHANNEL_MAP
using the MyGene.info API, then analyze per-channel strand distributions.
"""

import json
import os
import sys
import time
import requests

# Load channel map from cached config
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "graph_derived_config.json")
with open(CACHE_PATH) as f:
    config = json.load(f)

CHANNEL_MAP = config["channel_map"]
all_genes = sorted(CHANNEL_MAP.keys())
print(f"Total genes in CHANNEL_MAP: {len(all_genes)}")

# ── Batch query MyGene.info ──────────────────────────────────────────────
BATCH_SIZE = 100
results = {}
missing = []

for i in range(0, len(all_genes), BATCH_SIZE):
    batch = all_genes[i:i+BATCH_SIZE]
    print(f"  Querying batch {i//BATCH_SIZE + 1} ({len(batch)} genes)...")

    resp = requests.post(
        "https://mygene.info/v3/query",
        data={
            "q": ",".join(batch),
            "scopes": "symbol",
            "fields": "genomic_pos,symbol",
            "species": "human",
            "size": str(len(batch)),
        },
    )
    resp.raise_for_status()
    hits = resp.json()

    # Build lookup: query symbol -> hit
    for hit in hits:
        query = hit.get("query", "")
        if hit.get("notfound"):
            missing.append(query)
            continue

        gpos = hit.get("genomic_pos")
        if gpos is None:
            missing.append(query)
            continue

        # genomic_pos can be a list (multiple loci) or a dict (single locus)
        if isinstance(gpos, list):
            # Pick the one on a standard chromosome (not alt/patch)
            standard = [g for g in gpos if str(g.get("chr", "")).replace("X","").replace("Y","").replace("MT","").isdigit() or g.get("chr") in ("X","Y","MT")]
            gpos = standard[0] if standard else gpos[0]

        strand = gpos.get("strand")
        chrom = str(gpos.get("chr", ""))
        start = gpos.get("start")
        end = gpos.get("end")

        results[query] = {
            "strand": strand,
            "chr": chrom,
            "start": start,
            "end": end,
            "channel": CHANNEL_MAP.get(query, "unknown"),
        }

    # Be polite to API
    if i + BATCH_SIZE < len(all_genes):
        time.sleep(0.5)

print(f"\nResolved: {len(results)} / {len(all_genes)}")
if missing:
    print(f"Missing/not found: {len(missing)} -> {missing}")

# ── Save JSON ────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "gene_strand_data.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)
print(f"\nSaved to {os.path.abspath(OUT_PATH)}")

# ── Analysis: per-channel strand distribution ────────────────────────────
print("\n" + "="*70)
print("PER-CHANNEL STRAND DISTRIBUTION")
print("="*70)

channel_strands = {}
for gene, info in results.items():
    ch = info["channel"]
    s = info["strand"]
    if s is None:
        continue
    channel_strands.setdefault(ch, {"plus": [], "minus": [], "total": 0})
    channel_strands[ch]["total"] += 1
    if s == 1:
        channel_strands[ch]["plus"].append(gene)
    elif s == -1:
        channel_strands[ch]["minus"].append(gene)

print(f"\n{'Channel':<20} {'Total':>5} {'+ strand':>9} {'- strand':>9} {'+ frac':>8} {'- frac':>8}")
print("-"*65)

overall_plus = 0
overall_minus = 0

for ch in sorted(channel_strands.keys()):
    d = channel_strands[ch]
    np = len(d["plus"])
    nm = len(d["minus"])
    total = np + nm
    overall_plus += np
    overall_minus += nm
    fp = np / total if total else 0
    fm = nm / total if total else 0
    print(f"{ch:<20} {total:>5} {np:>9} {nm:>9} {fp:>8.1%} {fm:>8.1%}")

overall_total = overall_plus + overall_minus
print("-"*65)
print(f"{'OVERALL':<20} {overall_total:>5} {overall_plus:>9} {overall_minus:>9} "
      f"{overall_plus/overall_total:>8.1%} {overall_minus/overall_total:>8.1%}")

# ── Analysis: chromosome distribution per channel ────────────────────────
print("\n" + "="*70)
print("CHROMOSOME DISTRIBUTION (top 3 per channel)")
print("="*70)

for ch in sorted(channel_strands.keys()):
    chr_counts = {}
    for gene, info in results.items():
        if info["channel"] == ch and info.get("chr"):
            chr_counts[info["chr"]] = chr_counts.get(info["chr"], 0) + 1
    top3 = sorted(chr_counts.items(), key=lambda x: -x[1])[:3]
    top3_str = ", ".join(f"chr{c}({n})" for c, n in top3)
    print(f"  {ch:<20}: {top3_str}")

# ── Analysis: strand pairing in defense 2-chains ─────────────────────────
print("\n" + "="*70)
print("STRAND ANALYSIS FOR DEFENSE 2-CHAIN PAIRS")
print("="*70)

# Sub-channel gene groupings for analysis
sub_channels = {
    "DDR_HR": ["BRCA1", "BRCA2", "RAD51", "RAD51B", "RAD51C", "RAD51D", "PALB2", "BARD1", "XRCC2", "RAD54L", "RAD52", "BLM", "RAD50", "MRE11", "NBN"],
    "DDR_MMR": ["MLH1", "MSH2", "MSH3", "MSH6", "PMS1", "PMS2", "EPCAM"],
    "DDR_NER": ["ERCC2", "ERCC3", "ERCC4", "ERCC5"],
    "DDR_BER": ["MUTYH", "NTHL1", "PARP1"],
    "CellCycle_p53": ["TP53", "MDM2", "MDM4", "CDKN1A", "CDKN2A"],
    "CellCycle_RB": ["RB1", "CDK4", "CDK6", "CCND1", "CCND2", "CCND3", "CCNE1", "E2F3"],
    "PI3K_RAS_MAPK": ["KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "NF1", "ARAF", "SOS1"],
    "PI3K_AKT": ["PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR", "TSC1", "TSC2"],
}

for sub_name, genes in sub_channels.items():
    plus_genes = []
    minus_genes = []
    for g in genes:
        if g in results and results[g]["strand"] is not None:
            if results[g]["strand"] == 1:
                plus_genes.append(g)
            else:
                minus_genes.append(g)
    total = len(plus_genes) + len(minus_genes)
    if total == 0:
        continue
    fp = len(plus_genes) / total
    print(f"\n  {sub_name} ({total} genes):")
    print(f"    + strand ({len(plus_genes)}): {', '.join(plus_genes)}")
    print(f"    - strand ({len(minus_genes)}): {', '.join(minus_genes)}")
    print(f"    + fraction: {fp:.1%}")

# ── Cross-channel 2-chain pairs ──────────────────────────────────────────
print("\n" + "="*70)
print("CROSS-CHANNEL STRAND COMPARISON")
print("="*70)

pairs = [
    ("Immune", "PI3K_Growth"),
    ("DDR", "CellCycle"),
    ("CellCycle", "PI3K_Growth"),
    ("TissueArch", "Immune"),
    ("ChromatinRemodel", "DDR"),
]

for ch1, ch2 in pairs:
    d1 = channel_strands.get(ch1, {"plus": [], "minus": []})
    d2 = channel_strands.get(ch2, {"plus": [], "minus": []})
    t1 = len(d1["plus"]) + len(d1["minus"])
    t2 = len(d2["plus"]) + len(d2["minus"])
    f1 = len(d1["plus"]) / t1 if t1 else 0
    f2 = len(d2["plus"]) / t2 if t2 else 0
    print(f"  {ch1:<20} + frac: {f1:.1%}  vs  {ch2:<20} + frac: {f2:.1%}   delta: {abs(f1-f2):.1%}")

print("\nDone.")
