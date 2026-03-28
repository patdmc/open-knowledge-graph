#!/usr/bin/env python3
"""
Download and process DepMap CRISPR dependency data for project genes.

Downloads:
  1. CRISPRGeneEffect.csv — Chronos gene dependency scores per cell line
  2. Model.csv — maps cell lines to cancer lineages

Processes:
  - Filters to ~509 project genes (from expanded_channel_map.json + config.py)
  - Aggregates by cancer lineage (mean dependency per gene per lineage)
  - Saves clean CSVs: long-form and matrix form

Output directory: gnn/data/cache/depmap/

Usage:
    python3 -m gnn.scripts.download_depmap
"""

import os
import sys
import json
import urllib.request
import ssl
import time

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from gnn.config import CHANNEL_MAP, GNN_CACHE, GNN_RESULTS

DEPMAP_DIR = os.path.join(GNN_CACHE, "depmap")
os.makedirs(DEPMAP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Collect all project genes (~509 expanded + 92 core)
# ---------------------------------------------------------------------------

def get_project_genes():
    """Get the full set of project genes from config + expanded map."""
    genes = set(CHANNEL_MAP.keys())  # 92 core genes

    expanded_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    if os.path.exists(expanded_path):
        with open(expanded_path) as f:
            expanded = json.load(f)
        genes.update(expanded.keys())

    # Also add chromatin/methylation genes from v6
    from gnn.data.channel_dataset_v6 import CHROMATIN_REMODEL_GENES, DNA_METHYLATION_GENES
    genes.update(CHROMATIN_REMODEL_GENES)
    genes.update(DNA_METHYLATION_GENES)

    print(f"Total project genes: {len(genes)}")
    return genes


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url, dest_path, desc="file"):
    """Download a file with progress indication."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  {desc} already exists ({size_mb:.1f} MB), skipping download")
        return True

    print(f"  Downloading {desc}...")
    print(f"  URL: {url}")

    # Create SSL context — try certifi first, fall back to unverified
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=120)
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(dest_path + ".tmp", "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)

        print()  # newline after progress
        os.rename(dest_path + ".tmp", dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n  ERROR downloading {desc}: {e}")
        if os.path.exists(dest_path + ".tmp"):
            os.remove(dest_path + ".tmp")
        return False


def try_download_depmap():
    """Try multiple DepMap download URLs."""

    gene_effect_path = os.path.join(DEPMAP_DIR, "CRISPRGeneEffect.csv")
    model_path = os.path.join(DEPMAP_DIR, "Model.csv")

    # URLs discovered from DepMap downloads API (https://depmap.org/portal/download/api/downloads)
    # 25Q3 uses portal API paths, 24Q4 (12/24) and older use figshare
    BASE = "https://depmap.org"
    urls_gene_effect = [
        # 25Q3 (09/25) — latest
        BASE + "/portal/download/api/download?file_name=downloads-by-canonical-id%2F25q3-public-6202.1%2FCRISPRGeneEffect.csv&dl_name=CRISPRGeneEffect.csv&bucket=depmap-external-downloads",
        # 25Q2 (06/25)
        BASE + "/portal/download/api/download?file_name=downloads-by-canonical-id%2F25q2-public-557c.3%2FCRISPRGeneEffect.csv&dl_name=CRISPRGeneEffect.csv&bucket=depmap-external-downloads",
        # 24Q4 (12/24) — figshare
        "https://ndownloader.figshare.com/files/51064667",
        # 24Q2 (05/24) — figshare
        "https://ndownloader.figshare.com/files/46489063",
        # 23Q4 (11/23) — figshare
        "https://ndownloader.figshare.com/files/43346616",
    ]

    urls_model = [
        # 25Q3 (09/25) — latest
        BASE + "/portal/download/api/download?file_name=downloads-by-canonical-id%2Fpublic-25q3-b56c.72%2FModel.csv&dl_name=Model.csv&bucket=depmap-external-downloads",
        # 25Q2 (06/25)
        BASE + "/portal/download/api/download?file_name=downloads-by-canonical-id%2Fpublic-25q2-c5ef.75%2FModel.csv&dl_name=Model.csv&bucket=depmap-external-downloads",
        # 24Q4 (12/24) — figshare
        "https://ndownloader.figshare.com/files/51065297",
        # 24Q2 (05/24) — figshare
        "https://ndownloader.figshare.com/files/46489732",
        # 23Q2 (05/23) — figshare
        "https://ndownloader.figshare.com/files/40448834",
    ]

    # Download Gene Effect
    success_ge = False
    if os.path.exists(gene_effect_path):
        print(f"Gene Effect file already exists, skipping")
        success_ge = True
    else:
        for url in urls_gene_effect:
            success_ge = download_file(url, gene_effect_path, "CRISPRGeneEffect.csv")
            if success_ge:
                break
            time.sleep(2)

    # Download Model info
    success_model = False
    if os.path.exists(model_path):
        print(f"Model file already exists, skipping")
        success_model = True
    else:
        for url in urls_model:
            success_model = download_file(url, model_path, "Model.csv")
            if success_model:
                break
            time.sleep(2)

    return success_ge, success_model


# ---------------------------------------------------------------------------
# Process data
# ---------------------------------------------------------------------------

def process_depmap_data(project_genes):
    """Filter to project genes, aggregate by lineage, save clean CSVs."""

    gene_effect_path = os.path.join(DEPMAP_DIR, "CRISPRGeneEffect.csv")
    model_path = os.path.join(DEPMAP_DIR, "Model.csv")

    # --- Load Gene Effect ---
    print("\nLoading CRISPRGeneEffect.csv...")
    ge = pd.read_csv(gene_effect_path, index_col=0)
    print(f"  Shape: {ge.shape} (cell lines x genes)")
    print(f"  Columns sample: {list(ge.columns[:5])}")

    # Column format is typically "GENE (ENTREZ_ID)" — extract gene symbols
    col_map = {}
    for col in ge.columns:
        # Handle "TP53 (7157)" format
        gene_symbol = col.split(" ")[0].strip()
        col_map[col] = gene_symbol

    ge.columns = [col_map[c] for c in ge.columns]

    # Check for duplicates after renaming
    dupes = ge.columns[ge.columns.duplicated()].tolist()
    if dupes:
        print(f"  Warning: {len(dupes)} duplicate gene symbols, keeping first occurrence")
        ge = ge.loc[:, ~ge.columns.duplicated()]

    # --- Filter to project genes ---
    available = set(ge.columns) & project_genes
    missing = project_genes - set(ge.columns)
    print(f"\n  Project genes found in DepMap: {len(available)} / {len(project_genes)}")
    if missing:
        print(f"  Missing genes ({len(missing)}): {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")

    ge_filtered = ge[sorted(available)]
    print(f"  Filtered shape: {ge_filtered.shape}")

    # --- Load Model info ---
    print("\nLoading Model.csv...")
    model = pd.read_csv(model_path)
    print(f"  Shape: {model.shape}")
    print(f"  Columns: {list(model.columns)}")

    # Find the lineage column
    lineage_col = None
    for candidate in ["OncotreeLineage", "Lineage", "lineage", "primary_disease", "OncotreePrimaryDisease"]:
        if candidate in model.columns:
            lineage_col = candidate
            break

    if lineage_col is None:
        # Try case-insensitive search
        for col in model.columns:
            if "lineage" in col.lower() or "disease" in col.lower():
                lineage_col = col
                break

    if lineage_col is None:
        print("  ERROR: Could not find lineage column. Available columns:")
        for c in sorted(model.columns):
            print(f"    {c}")
        return

    print(f"  Using lineage column: {lineage_col}")

    # Find model ID column
    model_id_col = None
    for candidate in ["ModelID", "DepMap_ID", "model_id", "depmap_id"]:
        if candidate in model.columns:
            model_id_col = candidate
            break

    if model_id_col is None:
        # First column is likely the ID
        model_id_col = model.columns[0]

    print(f"  Using model ID column: {model_id_col}")

    # Build cell line -> lineage mapping
    lineage_map = dict(zip(model[model_id_col], model[lineage_col]))

    # Map cell lines in gene effect to lineages
    ge_filtered = ge_filtered.copy()
    ge_filtered["lineage"] = ge_filtered.index.map(lineage_map)
    n_mapped = ge_filtered["lineage"].notna().sum()
    print(f"  Cell lines mapped to lineage: {n_mapped} / {len(ge_filtered)}")

    # Drop unmapped
    ge_filtered = ge_filtered.dropna(subset=["lineage"])

    # --- Aggregate by lineage ---
    print("\nAggregating by lineage (mean dependency per gene per lineage)...")
    gene_cols = [c for c in ge_filtered.columns if c != "lineage"]
    lineage_agg = ge_filtered.groupby("lineage")[gene_cols].mean()
    print(f"  Lineage-aggregated shape: {lineage_agg.shape}")
    print(f"  Lineages: {sorted(lineage_agg.index.tolist())}")

    # --- Save matrix form (genes x lineages) ---
    matrix_path = os.path.join(DEPMAP_DIR, "depmap_dependency_matrix.csv")
    # Transpose so rows are genes, columns are lineages
    matrix = lineage_agg.T
    matrix.index.name = "gene"
    matrix.to_csv(matrix_path)
    print(f"\n  Saved matrix: {matrix_path}")
    print(f"  Shape: {matrix.shape} (genes x lineages)")

    # --- Save long form ---
    long_path = os.path.join(DEPMAP_DIR, "depmap_dependency_long.csv")
    long_df = lineage_agg.reset_index().melt(
        id_vars=["lineage"],
        var_name="gene",
        value_name="dependency_score"
    )
    # Reorder columns
    long_df = long_df[["gene", "lineage", "dependency_score"]].sort_values(["gene", "lineage"])
    long_df.to_csv(long_path, index=False)
    print(f"  Saved long form: {long_path}")
    print(f"  Shape: {long_df.shape}")

    # --- Save per-cell-line filtered data too (for downstream use) ---
    cell_line_path = os.path.join(DEPMAP_DIR, "depmap_dependency_cell_lines.csv")
    ge_filtered.to_csv(cell_line_path)
    print(f"  Saved cell line data: {cell_line_path}")

    # --- Summary statistics ---
    print("\n=== Summary ===")
    print(f"  Genes: {len(gene_cols)}")
    print(f"  Lineages: {len(lineage_agg)}")
    print(f"  Cell lines: {len(ge_filtered)}")

    # Most essential genes (most negative = most essential)
    mean_dep = matrix.mean(axis=1).sort_values()
    print(f"\n  Top 10 most essential genes (most negative dependency):")
    for gene, score in mean_dep.head(10).items():
        print(f"    {gene}: {score:.4f}")

    print(f"\n  Top 10 least essential genes:")
    for gene, score in mean_dep.tail(10).items():
        print(f"    {gene}: {score:.4f}")

    # Lineage-specific dependencies
    print(f"\n  Lineage count per cell line sample:")
    lineage_counts = ge_filtered["lineage"].value_counts()
    for lineage, count in lineage_counts.head(10).items():
        print(f"    {lineage}: {count} cell lines")

    return matrix, long_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DepMap CRISPR Dependency Data Download & Processing")
    print("=" * 60)

    # Step 1: Get project genes
    project_genes = get_project_genes()

    # Step 2: Download DepMap files
    print("\n--- Downloading DepMap data ---")
    success_ge, success_model = try_download_depmap()

    if not success_ge or not success_model:
        print("\nDownload failed. Trying alternative approach...")
        print("\nYou can manually download the files from:")
        print("  1. https://depmap.org/portal/download/all/")
        print("     - Search for 'CRISPRGeneEffect' and 'Model'")
        print("     - Save to:", DEPMAP_DIR)
        print("\nOr use the DepMap API:")
        print("  curl -L 'https://depmap.org/portal/download/api/download?file_name=CRISPRGeneEffect.csv&release=DepMap+Public+24Q4' -o CRISPRGeneEffect.csv")
        sys.exit(1)

    # Step 3: Process
    print("\n--- Processing DepMap data ---")
    process_depmap_data(project_genes)

    print("\n--- Done! ---")
    print(f"Output directory: {DEPMAP_DIR}")
