"""
Extract correlation-based edges from DepMap, TCGA expression, and TCGA CNA.

These are pairwise gene-gene relationships derived from multi-dimensional
biological data. Each edge has a correlation coefficient and p-value.

New edge types:
  - CO_ESSENTIAL: genes with correlated dependency across lineages (DepMap)
  - CO_EXPRESSED: genes with correlated expression across cancer types (TCGA)
  - CO_CNA: genes with correlated copy number across cancer types (TCGA)

All writes through GraphGateway.

Usage:
    python3 -u -m gnn.scripts.integrate_correlation_edges
    python3 -u -m gnn.scripts.integrate_correlation_edges --dry-run
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.graph_changelog import GraphGateway
from gnn.config import CHANNEL_MAP

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "cache")


def _step(name, actual=None, elapsed=None):
    status = f"{actual:,}" if actual is not None else ""
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"  {name:.<50s} {status:>15s}  {t}", flush=True)


def _correlation_edges(matrix, min_abs_corr=0.3, min_obs=10, max_edges=50000):
    """Extract significant correlation pairs from a gene × feature matrix.

    Returns list of (gene_a, gene_b, correlation, p_value) tuples.
    """
    genes = list(matrix.index)
    n = len(genes)
    n_features = matrix.shape[1]

    if n_features < min_obs:
        print(f"    Warning: only {n_features} features, results may be noisy")

    # Compute correlation matrix
    vals = matrix.values.astype(float)
    # Handle NaN: pairwise complete
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            x = vals[i]
            y = vals[j]
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < min_obs:
                continue

            r, p = stats.pearsonr(x[valid], y[valid])
            if abs(r) >= min_abs_corr and p < 0.05:
                edges.append((genes[i], genes[j], float(r), float(p)))

    # Sort by |correlation| descending, keep top max_edges
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    if len(edges) > max_edges:
        edges = edges[:max_edges]

    return edges


def integrate_co_essentiality(gw):
    """DepMap co-essentiality: genes dependent in same lineages."""
    print("\n  [1/3] DepMap co-essentiality", flush=True)
    t0 = time.time()

    dep = pd.read_csv(os.path.join(CACHE, "depmap", "depmap_dependency_matrix.csv"),
                       index_col='gene')
    print(f"    Matrix: {dep.shape[0]} genes × {dep.shape[1]} lineages")

    raw_edges = _correlation_edges(dep, min_abs_corr=0.4, min_obs=10, max_edges=20000)
    print(f"    Raw pairs (|r|>0.4): {len(raw_edges):,}")

    edges = []
    for ga, gb, corr, pval in raw_edges:
        ch_a = CHANNEL_MAP.get(ga)
        ch_b = CHANNEL_MAP.get(gb)
        edges.append({
            'from': ga,
            'to': gb,
            'correlation': round(corr, 4),
            'p_value': round(pval, 6),
            'abs_correlation': round(abs(corr), 4),
            'direction': 'positive' if corr > 0 else 'negative',
            'cross_channel': bool(ch_a != ch_b) if (ch_a and ch_b) else False,
        })

    n = gw.merge_edges("CO_ESSENTIAL", edges,
                       source="DepMap_CRISPR_2024",
                       source_detail=f"{len(edges)} pairs, |r|>0.4, p<0.05")
    _step("CO_ESSENTIAL edges", actual=n, elapsed=time.time() - t0)
    return n


def integrate_co_expression(gw):
    """TCGA co-expression: genes with correlated expression across cancer types."""
    print("\n  [2/3] TCGA co-expression", flush=True)
    t0 = time.time()

    expr = pd.read_csv(os.path.join(CACHE, "tcga", "tcga_expression_summary.csv"))
    pivot = expr.pivot_table(index='gene', columns='cancer_type', values='mean')
    print(f"    Matrix: {pivot.shape[0]} genes × {pivot.shape[1]} cancer types")

    raw_edges = _correlation_edges(pivot, min_abs_corr=0.3, min_obs=10)
    print(f"    Raw pairs (|r|>0.3): {len(raw_edges):,}")

    edges = []
    for ga, gb, corr, pval in raw_edges:
        ch_a = CHANNEL_MAP.get(ga)
        ch_b = CHANNEL_MAP.get(gb)
        edges.append({
            'from': ga,
            'to': gb,
            'correlation': round(corr, 4),
            'p_value': round(pval, 6),
            'abs_correlation': round(abs(corr), 4),
            'direction': 'positive' if corr > 0 else 'negative',
            'cross_channel': bool(ch_a != ch_b) if (ch_a and ch_b) else False,
        })

    n = gw.merge_edges("CO_EXPRESSED", edges,
                       source="TCGA_expression_2024",
                       source_detail=f"{len(edges)} pairs, |r|>0.3, p<0.05",
                       )
    _step("CO_EXPRESSED edges", actual=n, elapsed=time.time() - t0)
    return n


def integrate_co_cna(gw):
    """TCGA co-CNA: genes with correlated copy number across cancer types."""
    print("\n  [3/3] TCGA co-CNA", flush=True)
    t0 = time.time()

    cna = pd.read_csv(os.path.join(CACHE, "tcga", "tcga_cna_summary.csv"))
    pivot = cna.pivot_table(index='gene', columns='cancer_type', values='mean_cna')
    print(f"    Matrix: {pivot.shape[0]} genes × {pivot.shape[1]} cancer types")

    raw_edges = _correlation_edges(pivot, min_abs_corr=0.3, min_obs=10)
    print(f"    Raw pairs (|r|>0.3): {len(raw_edges):,}")

    edges = []
    for ga, gb, corr, pval in raw_edges:
        ch_a = CHANNEL_MAP.get(ga)
        ch_b = CHANNEL_MAP.get(gb)
        edges.append({
            'from': ga,
            'to': gb,
            'correlation': round(corr, 4),
            'p_value': round(pval, 6),
            'abs_correlation': round(abs(corr), 4),
            'direction': 'positive' if corr > 0 else 'negative',
            'cross_channel': bool(ch_a != ch_b) if (ch_a and ch_b) else False,
        })

    n = gw.merge_edges("CO_CNA", edges,
                       source="TCGA_CNA_2024",
                       source_detail=f"{len(edges)} pairs, |r|>0.3, p<0.05")
    _step("CO_CNA edges", actual=n, elapsed=time.time() - t0)
    return n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  EXTRACTING CORRELATION EDGES")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'#'*70}")

    t0 = time.time()
    gw = GraphGateway(dry_run=args.dry_run)

    try:
        integrate_co_essentiality(gw)
        integrate_co_expression(gw)
        integrate_co_cna(gw)
    finally:
        gw.close()

    print(f"\n{'#'*70}")
    print(f"  DONE [{time.time()-t0:.1f}s]")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
