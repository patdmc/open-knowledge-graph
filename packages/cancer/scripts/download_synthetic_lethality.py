"""
Download and process synthetic lethality data for the 509-gene cancer model.

Sources:
1. BioGRID — bulk download, filter for Synthetic Lethality interactions
2. SynLethDB 2.0 — curated synthetic lethality database

Usage:
    python3 -u -m gnn.scripts.download_synthetic_lethality
"""

import csv
import io
import json
import os
import ssl
import urllib.request
import zipfile
from collections import Counter, defaultdict

# Work around macOS Python SSL certificate issues
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GNN_RESULTS = os.path.join(ROOT, "gnn", "results")
SL_DIR = os.path.join(ROOT, "gnn", "data", "cache", "synthetic_lethality")
os.makedirs(SL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load gene list
# ---------------------------------------------------------------------------

def load_gene_set() -> set:
    """Load the 509-gene set from expanded_channel_map.json."""
    path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    with open(path) as f:
        data = json.load(f)
    genes = set(data.keys())
    print(f"Loaded {len(genes)} genes from expanded channel map")
    return genes


def load_channel_map() -> dict:
    """Load gene -> channel mapping."""
    path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    with open(path) as f:
        data = json.load(f)
    return {gene: info["channel"] for gene, info in data.items()}


# ---------------------------------------------------------------------------
# BioGRID download
# ---------------------------------------------------------------------------

BIOGRID_URL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip"

def download_biogrid() -> str:
    """Download BioGRID tab3 zip file. Returns path to zip."""
    zip_path = os.path.join(SL_DIR, "BIOGRID-ALL-LATEST.tab3.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1_000_000:
        print(f"  Using cached BioGRID: {zip_path}")
        return zip_path

    print(f"  Downloading BioGRID from {BIOGRID_URL} ...")
    print("  (This is ~300MB, may take a few minutes)")
    req = urllib.request.Request(BIOGRID_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=SSL_CTX) as resp, open(zip_path, 'wb') as out:
        while True:
            chunk = resp.read(8192 * 16)
            if not chunk:
                break
            out.write(chunk)
    size_mb = os.path.getsize(zip_path) / 1_000_000
    print(f"  Downloaded {size_mb:.1f} MB")
    return zip_path


def extract_sl_from_biogrid(zip_path: str, our_genes: set) -> list[dict]:
    """
    Extract Synthetic Lethality interactions from BioGRID.
    Filter to human (taxon 9606) and our gene set.
    """
    print("  Extracting synthetic lethality from BioGRID ...")
    sl_pairs = []

    with zipfile.ZipFile(zip_path) as zf:
        # Find the main tab3 file
        tab3_files = [n for n in zf.namelist() if n.endswith('.tab3.txt') and 'MULTI' not in n]
        if not tab3_files:
            tab3_files = [n for n in zf.namelist() if '.tab3.txt' in n]
        if not tab3_files:
            tab3_files = [n for n in zf.namelist() if n.endswith('.txt')]

        print(f"  Files in zip: {len(zf.namelist())}")
        for name in zf.namelist()[:5]:
            print(f"    {name}")

        target = tab3_files[0]
        print(f"  Reading: {target}")

        with zf.open(target) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'), delimiter='\t')
            total = 0
            sl_total = 0
            sl_human = 0

            for row in reader:
                total += 1
                if total % 5_000_000 == 0:
                    print(f"    Processed {total:,} rows, found {sl_human} human SL pairs ...")

                exp_system = row.get('Experimental System', '')
                if 'Synthetic' not in exp_system:
                    continue

                sl_total += 1

                # Filter to human
                org_a = row.get('Organism ID Interactor A', '')
                org_b = row.get('Organism ID Interactor B', '')
                if org_a != '9606' or org_b != '9606':
                    continue

                sl_human += 1

                gene_a = row.get('Official Symbol Interactor A', '')
                gene_b = row.get('Official Symbol Interactor B', '')

                # Skip self-interactions
                if gene_a == gene_b:
                    continue

                # Check if at least one gene is in our set
                a_in = gene_a in our_genes
                b_in = gene_b in our_genes
                if not (a_in or b_in):
                    continue

                # Normalize pair order
                if gene_a > gene_b:
                    gene_a, gene_b = gene_b, gene_a

                sl_pairs.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'both_in_set': a_in and b_in,
                    'evidence_type': 'experimental',
                    'experimental_system': exp_system,
                    'source': 'BioGRID',
                    'pubmed_id': row.get('Pubmed ID', ''),
                    'author': row.get('Author', ''),
                    'throughput': row.get('Throughput', ''),
                })

            print(f"  Total BioGRID rows: {total:,}")
            print(f"  Total SL interactions: {sl_total:,}")
            print(f"  Human SL interactions: {sl_human:,}")
            print(f"  SL pairs with our genes: {len(sl_pairs):,}")

    return sl_pairs


# ---------------------------------------------------------------------------
# SynLethDB download (try API)
# ---------------------------------------------------------------------------

SYNLETHDB_URL = "https://synlethdb.sist.shanghaitech.edu.cn/api/pairs"

def download_synlethdb(our_genes: set) -> list[dict]:
    """
    Try to get SL pairs from SynLethDB.
    The API may or may not be accessible; handle gracefully.
    """
    print("\n  Attempting SynLethDB download ...")
    sl_pairs = []

    # Try bulk download first
    bulk_url = "https://synlethdb.sist.shanghaitech.edu.cn/static/SynLethDB2.0_Human.zip"
    bulk_path = os.path.join(SL_DIR, "SynLethDB2.0_Human.zip")

    if os.path.exists(bulk_path) and os.path.getsize(bulk_path) > 10_000:
        print(f"  Using cached SynLethDB: {bulk_path}")
    else:
        try:
            req = urllib.request.Request(bulk_url, headers={
                'User-Agent': 'Mozilla/5.0 (research download)'
            })
            print(f"  Downloading from {bulk_url} ...")
            with urllib.request.urlopen(req, context=SSL_CTX) as resp, open(bulk_path, 'wb') as out:
                while True:
                    chunk = resp.read(8192 * 16)
                    if not chunk:
                        break
                    out.write(chunk)
            size_mb = os.path.getsize(bulk_path) / 1_000_000
            print(f"  Downloaded {size_mb:.1f} MB")
        except Exception as e:
            print(f"  SynLethDB bulk download failed: {e}")

            # Try alternative URL
            alt_url = "https://synlethdb.sist.shanghaitech.edu.cn/static/download/SynLethDB_Human.csv"
            alt_path = os.path.join(SL_DIR, "SynLethDB_Human.csv")
            try:
                req = urllib.request.Request(alt_url, headers={
                    'User-Agent': 'Mozilla/5.0 (research download)'
                })
                print(f"  Trying alternative: {alt_url}")
                with urllib.request.urlopen(req, context=SSL_CTX) as resp, open(alt_path, 'wb') as out:
                    while True:
                        chunk = resp.read(8192 * 16)
                        if not chunk:
                            break
                        out.write(chunk)
                print(f"  Downloaded alternative file")
            except Exception as e2:
                print(f"  Alternative also failed: {e2}")
                print("  SynLethDB not available. Continuing with BioGRID only.")
                return sl_pairs

    # Try to parse whatever we downloaded
    for fpath in [bulk_path, os.path.join(SL_DIR, "SynLethDB_Human.csv")]:
        if not os.path.exists(fpath):
            continue

        try:
            if fpath.endswith('.zip'):
                with zipfile.ZipFile(fpath) as zf:
                    print(f"  SynLethDB zip contents: {zf.namelist()}")
                    csv_files = [n for n in zf.namelist() if n.endswith('.csv') or n.endswith('.txt') or n.endswith('.tsv')]
                    if not csv_files:
                        csv_files = zf.namelist()
                    for csv_file in csv_files:
                        print(f"  Parsing: {csv_file}")
                        sl_pairs.extend(_parse_synlethdb_file(zf.open(csv_file), our_genes))
            else:
                with open(fpath, 'rb') as f:
                    sl_pairs.extend(_parse_synlethdb_file(f, our_genes))
        except Exception as e:
            print(f"  Error parsing {fpath}: {e}")

    print(f"  SynLethDB pairs with our genes: {len(sl_pairs)}")
    return sl_pairs


def _parse_synlethdb_file(fileobj, our_genes: set) -> list[dict]:
    """Parse a SynLethDB CSV/TSV file."""
    pairs = []
    try:
        wrapper = io.TextIOWrapper(fileobj, encoding='utf-8', errors='replace')
        # Read header to detect format
        header_line = wrapper.readline().strip()
        delimiter = '\t' if '\t' in header_line else ','
        headers = header_line.split(delimiter)
        headers_lower = [h.strip().lower() for h in headers]

        print(f"    Headers: {headers_lower[:10]}")

        # Try to identify gene columns
        gene_a_col = None
        gene_b_col = None
        evidence_col = None
        cancer_col = None

        for i, h in enumerate(headers_lower):
            if 'gene_a' in h or 'genea' in h or 'gene1' in h or h == 'symbol_a':
                gene_a_col = i
            elif 'gene_b' in h or 'geneb' in h or 'gene2' in h or h == 'symbol_b':
                gene_b_col = i
            elif 'evidence' in h or 'type' in h or 'method' in h:
                if evidence_col is None:
                    evidence_col = i
            elif 'cancer' in h or 'tissue' in h or 'disease' in h:
                if cancer_col is None:
                    cancer_col = i

        if gene_a_col is None or gene_b_col is None:
            # Try positional (first two columns often gene symbols)
            if len(headers) >= 2:
                gene_a_col = 0
                gene_b_col = 1
                print(f"    Using positional columns: {headers[0]}, {headers[1]}")

        for line in wrapper:
            parts = line.strip().split(delimiter)
            if len(parts) <= max(gene_a_col or 0, gene_b_col or 0):
                continue

            gene_a = parts[gene_a_col].strip().upper()
            gene_b = parts[gene_b_col].strip().upper()

            if gene_a == gene_b:
                continue

            a_in = gene_a in our_genes
            b_in = gene_b in our_genes
            if not (a_in or b_in):
                continue

            # Normalize order
            if gene_a > gene_b:
                gene_a, gene_b = gene_b, gene_a

            evidence = parts[evidence_col].strip() if evidence_col and evidence_col < len(parts) else 'curated'
            cancer = parts[cancer_col].strip() if cancer_col and cancer_col < len(parts) else ''

            pairs.append({
                'gene_a': gene_a,
                'gene_b': gene_b,
                'both_in_set': a_in and b_in,
                'evidence_type': evidence,
                'experimental_system': 'SynLethDB curated',
                'source': 'SynLethDB',
                'pubmed_id': '',
                'author': '',
                'throughput': '',
                'cancer_context': cancer,
            })

    except Exception as e:
        print(f"    Parse error: {e}")

    return pairs


# ---------------------------------------------------------------------------
# Published ISLE pairs (hardcoded high-confidence pairs from literature)
# ---------------------------------------------------------------------------

# Well-known SL pairs from ISLE and clinical literature
# These are the highest-confidence pairs relevant to cancer therapy
CURATED_SL_PAIRS = [
    # BRCA1/2 - PARP (represented by pathway genes)
    ("BRCA1", "PARP1", "clinical", "BRCA-PARP synthetic lethality"),
    ("BRCA2", "PARP1", "clinical", "BRCA-PARP synthetic lethality"),
    ("PALB2", "PARP1", "clinical", "HR-deficiency PARP sensitivity"),
    ("RAD51C", "PARP1", "clinical", "HR-deficiency PARP sensitivity"),
    ("RAD51D", "PARP1", "clinical", "HR-deficiency PARP sensitivity"),
    ("ATM", "PARP1", "experimental", "ATM-PARP SL"),
    # TP53 - related SL pairs
    ("TP53", "PLK1", "experimental", "p53-PLK1 SL"),
    ("TP53", "WEE1", "experimental", "p53-WEE1 SL (G2 checkpoint)"),
    ("TP53", "CHEK1", "experimental", "p53-CHK1 SL"),
    # RAS pathway SL
    ("KRAS", "STK33", "experimental", "KRAS-STK33 SL"),
    ("KRAS", "PLK1", "experimental", "KRAS-PLK1 SL"),
    ("KRAS", "TP53", "computational", "KRAS-TP53 co-vulnerability"),
    # PTEN - related
    ("PTEN", "PIK3CB", "experimental", "PTEN-PIK3CB SL"),
    ("PTEN", "ATR", "experimental", "PTEN-ATR SL"),
    # RB1 - related
    ("RB1", "AURORA_A", "experimental", "RB1-Aurora SL"),
    ("RB1", "TSC1", "experimental", "RB1-TSC1 SL"),
    # APC - related
    ("APC", "CTNNB1", "experimental", "Wnt pathway SL"),
    # STK11 - related
    ("STK11", "MTOR", "experimental", "STK11-mTOR SL"),
    # VHL - related (if in panel)
    ("VHL", "HIF1A", "experimental", "VHL-HIF SL"),
    # MSI-related
    ("MLH1", "POLB", "experimental", "MMR-BER SL"),
    ("MSH2", "POLB", "experimental", "MMR-BER SL"),
    # MYC - related
    ("MYC", "CDK9", "experimental", "MYC-CDK9 SL"),
    ("MYC", "AURKA", "experimental", "MYC-Aurora SL"),
    # ARID1A - related
    ("ARID1A", "ARID1B", "experimental", "ARID1A-ARID1B SL (SWI/SNF)"),
    ("ARID1A", "EZH2", "experimental", "ARID1A-EZH2 SL"),
    ("ARID1A", "ATR", "experimental", "ARID1A-ATR SL"),
    # SMAD4 - related
    ("SMAD4", "AKT1", "computational", "TGFb-AKT cross-talk SL"),
    # CDH1 (E-cadherin) - related
    ("CDH1", "ROS1", "experimental", "CDH1-ROS1 SL (lobular breast)"),
    # NF1 - related
    ("NF1", "BRAF", "experimental", "NF1-BRAF SL (RAS pathway)"),
    # FBXW7 - related
    ("FBXW7", "MYC", "experimental", "FBXW7-MYC SL"),
    # FANCA/FANCC
    ("FANCA", "PARP1", "experimental", "FA-PARP SL"),
    ("FANCC", "PARP1", "experimental", "FA-PARP SL"),
]


def get_curated_pairs(our_genes: set) -> list[dict]:
    """Return curated SL pairs filtered to our gene set."""
    pairs = []
    for gene_a, gene_b, evidence, context in CURATED_SL_PAIRS:
        a_in = gene_a in our_genes
        b_in = gene_b in our_genes
        if not (a_in or b_in):
            continue

        ga, gb = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
        pairs.append({
            'gene_a': ga,
            'gene_b': gb,
            'both_in_set': a_in and b_in,
            'evidence_type': evidence,
            'experimental_system': 'Literature curated',
            'source': 'ISLE/Literature',
            'pubmed_id': '',
            'author': '',
            'throughput': '',
            'cancer_context': context,
        })
    return pairs


# ---------------------------------------------------------------------------
# Deduplication and output
# ---------------------------------------------------------------------------

def deduplicate_pairs(all_pairs: list[dict]) -> list[dict]:
    """Deduplicate SL pairs, keeping best evidence."""
    pair_map = {}
    for p in all_pairs:
        key = (p['gene_a'], p['gene_b'])
        if key not in pair_map:
            pair_map[key] = p
        else:
            # Keep track of multiple sources
            existing = pair_map[key]
            if p['source'] not in existing['source']:
                existing['source'] += f"; {p['source']}"
            if p.get('cancer_context') and not existing.get('cancer_context'):
                existing['cancer_context'] = p['cancer_context']
    return list(pair_map.values())


def write_output(pairs: list[dict], our_genes: set, channel_map: dict):
    """Write clean CSV and summary statistics."""

    # All pairs CSV
    all_csv = os.path.join(SL_DIR, "synthetic_lethality_all.csv")
    fieldnames = ['gene_a', 'gene_b', 'both_in_set', 'evidence_type',
                  'source', 'cancer_context', 'experimental_system',
                  'pubmed_id', 'channel_a', 'channel_b']

    with open(all_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for p in sorted(pairs, key=lambda x: (x['gene_a'], x['gene_b'])):
            p['channel_a'] = channel_map.get(p['gene_a'], '')
            p['channel_b'] = channel_map.get(p['gene_b'], '')
            writer.writerow(p)
    print(f"\nSaved all pairs: {all_csv}")
    print(f"  Total unique pairs: {len(pairs)}")

    # Both-in-set pairs only
    both_pairs = [p for p in pairs if p['both_in_set']]
    both_csv = os.path.join(SL_DIR, "synthetic_lethality_both_in_set.csv")
    with open(both_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for p in sorted(both_pairs, key=lambda x: (x['gene_a'], x['gene_b'])):
            p['channel_a'] = channel_map.get(p['gene_a'], '')
            p['channel_b'] = channel_map.get(p['gene_b'], '')
            writer.writerow(p)
    print(f"\nSaved both-in-set pairs: {both_csv}")
    print(f"  Pairs where BOTH genes in 509: {len(both_pairs)}")

    # Summary stats
    genes_with_sl = set()
    for p in pairs:
        if p['gene_a'] in our_genes:
            genes_with_sl.add(p['gene_a'])
        if p['gene_b'] in our_genes:
            genes_with_sl.add(p['gene_b'])

    both_genes_with_sl = set()
    for p in both_pairs:
        both_genes_with_sl.add(p['gene_a'])
        both_genes_with_sl.add(p['gene_b'])

    # Cross-channel pairs
    cross_channel = [p for p in both_pairs
                     if channel_map.get(p['gene_a']) != channel_map.get(p['gene_b'])
                     and channel_map.get(p['gene_a']) and channel_map.get(p['gene_b'])]

    # Source breakdown
    source_counts = Counter()
    for p in pairs:
        for src in p['source'].split('; '):
            source_counts[src] += 1

    # Evidence breakdown
    evidence_counts = Counter(p['evidence_type'] for p in pairs)

    # Channel pair heatmap data
    channel_pair_counts = Counter()
    for p in both_pairs:
        ch_a = channel_map.get(p['gene_a'], 'unknown')
        ch_b = channel_map.get(p['gene_b'], 'unknown')
        key = tuple(sorted([ch_a, ch_b]))
        channel_pair_counts[key] += 1

    # Genes with most SL partners
    partner_counts = Counter()
    for p in both_pairs:
        partner_counts[p['gene_a']] += 1
        partner_counts[p['gene_b']] += 1

    summary = {
        'total_unique_pairs': len(pairs),
        'both_in_509_pairs': len(both_pairs),
        'genes_with_any_sl_partner': len(genes_with_sl),
        'genes_with_in_set_sl_partner': len(both_genes_with_sl),
        'cross_channel_pairs': len(cross_channel),
        'pct_genes_with_sl': round(100 * len(genes_with_sl) / len(our_genes), 1),
        'source_breakdown': dict(source_counts.most_common()),
        'evidence_breakdown': dict(evidence_counts.most_common()),
        'channel_pair_counts': {f"{k[0]}--{k[1]}": v for k, v in channel_pair_counts.most_common()},
        'top_20_genes_by_sl_partners': dict(partner_counts.most_common(20)),
    }

    summary_path = os.path.join(SL_DIR, "synthetic_lethality_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SYNTHETIC LETHALITY SUMMARY")
    print("=" * 70)
    print(f"\nTotal unique SL pairs (at least 1 gene in 509): {len(pairs)}")
    print(f"Pairs where BOTH genes in 509-gene set:          {len(both_pairs)}")
    print(f"Cross-channel SL pairs (both in 509):            {len(cross_channel)}")
    print(f"\nGenes with any known SL partner:    {len(genes_with_sl)}/{len(our_genes)} ({100*len(genes_with_sl)/len(our_genes):.1f}%)")
    print(f"Genes with in-set SL partner:       {len(both_genes_with_sl)}/{len(our_genes)} ({100*len(both_genes_with_sl)/len(our_genes):.1f}%)")

    print("\nBy source:")
    for src, count in source_counts.most_common():
        print(f"  {src:20s}  {count:6d}")

    print("\nBy evidence type:")
    for ev, count in evidence_counts.most_common():
        print(f"  {ev:20s}  {count:6d}")

    print("\nTop 20 genes by # SL partners (both in 509):")
    for gene, count in partner_counts.most_common(20):
        ch = channel_map.get(gene, '?')
        print(f"  {gene:12s} ({ch:12s})  {count:4d} partners")

    print("\nChannel-pair SL counts (both in 509):")
    for (ch_a, ch_b), count in channel_pair_counts.most_common(15):
        label = f"{ch_a}--{ch_b}" if ch_a != ch_b else f"{ch_a} (within)"
        print(f"  {label:40s}  {count:4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SYNTHETIC LETHALITY DATA DOWNLOAD & PROCESSING")
    print("=" * 70)

    our_genes = load_gene_set()
    channel_map = load_channel_map()

    all_pairs = []

    # 1. BioGRID
    print("\n--- BioGRID ---")
    try:
        zip_path = download_biogrid()
        biogrid_pairs = extract_sl_from_biogrid(zip_path, our_genes)
        all_pairs.extend(biogrid_pairs)
        print(f"  BioGRID contributed {len(biogrid_pairs)} pairs")
    except Exception as e:
        print(f"  BioGRID failed: {e}")

    # 2. SynLethDB
    print("\n--- SynLethDB ---")
    try:
        synleth_pairs = download_synlethdb(our_genes)
        all_pairs.extend(synleth_pairs)
        print(f"  SynLethDB contributed {len(synleth_pairs)} pairs")
    except Exception as e:
        print(f"  SynLethDB failed: {e}")

    # 3. Curated literature pairs
    print("\n--- Curated Literature ---")
    curated_pairs = get_curated_pairs(our_genes)
    all_pairs.extend(curated_pairs)
    print(f"  Curated literature contributed {len(curated_pairs)} pairs")

    # Deduplicate
    print("\n--- Deduplication ---")
    unique_pairs = deduplicate_pairs(all_pairs)
    print(f"  {len(all_pairs)} raw -> {len(unique_pairs)} unique pairs")

    # Write output
    write_output(unique_pairs, our_genes, channel_map)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
