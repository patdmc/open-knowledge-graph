#!/usr/bin/env python3
"""
Download Cox-Sage benchmark data and run comparison.

Cox-Sage uses 7 TCGA cohorts: BRCA, BLCA, COADREAD, GBM, LUAD, STAD, UCEC
Data available on Kaggle/Zenodo.
"""

import os
import json
import requests
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO

from ..config import GNN_CACHE, GNN_RESULTS


COX_SAGE_ZENODO = "https://zenodo.org/records/14204893"
COX_SAGE_CANCERS = ["BRCA", "BLCA", "COADREAD", "GBM", "LUAD", "STAD", "UCEC"]

BENCHMARK_DIR = os.path.join(GNN_CACHE, "cox_sage_data")


def download_cox_sage_data():
    """Download Cox-Sage processed TCGA data from Zenodo."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # Check if already downloaded
    marker = os.path.join(BENCHMARK_DIR, ".downloaded")
    if os.path.exists(marker):
        print("Cox-Sage data already downloaded.")
        return BENCHMARK_DIR

    print("Downloading Cox-Sage data from Zenodo...")
    # The Zenodo record has files we need to fetch
    # Try the API to get file URLs
    record_id = "14204893"
    api_url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    record = resp.json()

    for f in record.get("files", []):
        fname = f["key"]
        url = f["links"]["self"]
        fpath = os.path.join(BENCHMARK_DIR, fname)
        if os.path.exists(fpath):
            print(f"  {fname} already exists, skipping")
            continue
        print(f"  Downloading {fname}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(fpath, "wb") as out:
            for chunk in r.iter_content(chunk_size=8192):
                out.write(chunk)

    with open(marker, "w") as f:
        f.write("ok")
    print(f"  Saved to {BENCHMARK_DIR}")
    return BENCHMARK_DIR


def load_cox_sage_cohort(cancer_type):
    """Load a single Cox-Sage cohort's gene expression + clinical data.

    Returns:
        features: (N, G) array of gene expression values
        time: (N,) survival times
        event: (N,) event indicators
    """
    data_dir = BENCHMARK_DIR
    # Cox-Sage stores data as CSVs per cancer type
    # Try common file patterns
    for pattern in [
        f"{cancer_type}_data.csv",
        f"{cancer_type.lower()}_data.csv",
        f"TCGA-{cancer_type}.csv",
    ]:
        fpath = os.path.join(data_dir, pattern)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            break
    else:
        # List what's available
        files = os.listdir(data_dir)
        raise FileNotFoundError(
            f"No data file found for {cancer_type}. Available files: {files}"
        )

    return df


def get_cox_sage_reported_results():
    """Published results from Cox-Sage paper for comparison.

    From Table 2 of the paper (Briefings in Bioinformatics, 2025):
    C-index values with 5-fold CV.
    """
    return {
        "BRCA": {"c_index": 0.669, "std": 0.040},
        "BLCA": {"c_index": 0.627, "std": 0.032},
        "COADREAD": {"c_index": 0.646, "std": 0.054},
        "GBM": {"c_index": 0.675, "std": 0.048},
        "LUAD": {"c_index": 0.630, "std": 0.028},
        "STAD": {"c_index": 0.632, "std": 0.038},
        "UCEC": {"c_index": 0.716, "std": 0.064},
    }


if __name__ == "__main__":
    download_cox_sage_data()
    results = get_cox_sage_reported_results()
    print("\nCox-Sage reported results (C-index):")
    for cancer, r in results.items():
        print(f"  {cancer}: {r['c_index']:.3f} +/- {r['std']:.3f}")
