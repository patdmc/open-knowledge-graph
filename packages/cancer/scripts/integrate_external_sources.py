"""
Integrate external data sources into the knowledge graph.

All writes go through GraphGateway — logged, provenance-tracked, never destructive.

Sources:
  1. Synthetic lethality (121 gene pairs) → SYNTHETIC_LETHAL edges
  2. DepMap (501 genes × 30 lineages) → Gene node essentiality properties
  3. CIViC (1,768 variants, 219 genes) → Gene/variant functional annotations
  4. TCGA expression (96 genes × 32 cancer types) → EXPRESSED_IN edges
  5. TCGA CNA (96 genes × 32 cancer types) → CNA_IN edges

Usage:
    python3 -u -m gnn.scripts.integrate_external_sources              # all sources
    python3 -u -m gnn.scripts.integrate_external_sources --source sl  # just synthetic lethality
    python3 -u -m gnn.scripts.integrate_external_sources --dry-run    # preview only
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.graph_changelog import GraphGateway

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "cache")


def _step(name, actual=None, elapsed=None):
    status = f"{actual:,}" if actual is not None else ""
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"  {name:.<50s} {status:>15s}  {t}", flush=True)


# =========================================================================
# 1. Synthetic lethality → SYNTHETIC_LETHAL edges
# =========================================================================

def integrate_synthetic_lethality(gw):
    """Add SYNTHETIC_LETHAL edges between gene pairs.

    Each edge carries evidence type, source, cancer context, and PubMed ID.
    Cross-channel SL pairs are especially valuable — they encode
    inter-pathway dependencies the graph doesn't yet know about.
    """
    print("\n  [1/5] Synthetic lethality", flush=True)
    t0 = time.time()

    path = os.path.join(CACHE, "synthetic_lethality", "synthetic_lethality_both_in_set.csv")
    df = pd.read_csv(path)

    edges = []
    for _, row in df.iterrows():
        edges.append({
            'from': row['gene_a'],
            'to': row['gene_b'],
            'evidence_type': row.get('evidence_type', ''),
            'source_db': row.get('source', ''),
            'cancer_context': row.get('cancer_context', ''),
            'experimental_system': row.get('experimental_system', ''),
            'pubmed_id': str(row.get('pubmed_id', '')),
            'cross_channel': bool(row.get('channel_a', '') != row.get('channel_b', '')),
            'channel_a': row.get('channel_a', ''),
            'channel_b': row.get('channel_b', ''),
        })

    n = gw.merge_edges("SYNTHETIC_LETHAL", edges,
                       source="synthetic_lethality_ISLE_BioGRID",
                       source_detail=f"{len(edges)} pairs, {sum(1 for e in edges if e['cross_channel'])} cross-channel")
    _step("SYNTHETIC_LETHAL edges", actual=n, elapsed=time.time() - t0)
    return n


# =========================================================================
# 2. DepMap → Gene node essentiality properties
# =========================================================================

def integrate_depmap(gw):
    """Add DepMap dependency scores as Gene node properties.

    Negative scores = essential (cell death when knocked out).
    We store: mean essentiality across lineages, most dependent lineage,
    and per-lineage scores for the top 5 most variable lineages.
    """
    print("\n  [2/5] DepMap essentiality", flush=True)
    t0 = time.time()

    path = os.path.join(CACHE, "depmap", "depmap_dependency_matrix.csv")
    df = pd.read_csv(path, index_col='gene')

    lineages = [c for c in df.columns if c != 'gene']

    updates = []
    for gene in df.index:
        scores = df.loc[gene, lineages].values.astype(float)
        valid = scores[~np.isnan(scores)]
        if len(valid) == 0:
            continue

        mean_ess = float(np.mean(valid))
        min_ess = float(np.min(valid))  # most essential (most negative)
        most_essential_lineage = lineages[int(np.nanargmin(scores))]
        std_ess = float(np.std(valid))

        update = {
            'name': gene,
            'depmap_mean_essentiality': round(mean_ess, 4),
            'depmap_min_essentiality': round(min_ess, 4),
            'depmap_std_essentiality': round(std_ess, 4),
            'depmap_most_essential_lineage': most_essential_lineage,
            'depmap_is_pan_essential': bool(mean_ess < -0.5),
            'depmap_n_lineages': len(valid),
        }
        updates.append(update)

    n = gw.set_node_properties("Gene", "name", updates,
                               source="DepMap_CRISPR_2024")
    _step("Gene essentiality properties", actual=n, elapsed=time.time() - t0)
    return n


# =========================================================================
# 3. CIViC → Gene functional annotations
# =========================================================================

def integrate_civic(gw):
    """Add CIViC variant functional annotations to Gene nodes.

    Aggregates per gene: GOF/LOF count, oncogenic effect summary,
    evidence level distribution, and actionability.
    """
    print("\n  [3/5] CIViC variant annotations", flush=True)
    t0 = time.time()

    path = os.path.join(CACHE, "oncokb", "variant_functional_map.csv")
    df = pd.read_csv(path)

    # Aggregate per gene
    gene_stats = {}
    for _, row in df.iterrows():
        gene = row['gene']
        if gene not in gene_stats:
            gene_stats[gene] = {
                'n_variants': 0,
                'gof_count': 0,
                'lof_count': 0,
                'resistance_count': 0,
                'sensitivity_count': 0,
                'evidence_levels': set(),
                'oncogenic_effects': set(),
            }
        gs = gene_stats[gene]
        gs['n_variants'] += 1

        func = str(row.get('inferred_function', ''))
        if 'GOF' in func:
            gs['gof_count'] += 1
        if 'LOF' in func:
            gs['lof_count'] += 1

        effects = str(row.get('oncogenic_effects', ''))
        if 'Resistance' in effects:
            gs['resistance_count'] += 1
        if 'Sensitivity' in effects or 'Response' in effects:
            gs['sensitivity_count'] += 1

        for lvl in str(row.get('evidence_levels', '')).split(';'):
            lvl = lvl.strip()
            if lvl and lvl != 'nan':
                gs['evidence_levels'].add(lvl)

        for eff in effects.split(';'):
            eff = eff.strip()
            if eff and eff != 'nan':
                gs['oncogenic_effects'].add(eff)

    updates = []
    for gene, gs in gene_stats.items():
        # Determine dominant function
        if gs['gof_count'] > gs['lof_count'] * 2:
            dominant = 'GOF'
        elif gs['lof_count'] > gs['gof_count'] * 2:
            dominant = 'LOF'
        elif gs['gof_count'] > 0 and gs['lof_count'] > 0:
            dominant = 'mixed'
        else:
            dominant = 'unknown'

        updates.append({
            'name': gene,
            'civic_n_variants': gs['n_variants'],
            'civic_gof_count': gs['gof_count'],
            'civic_lof_count': gs['lof_count'],
            'civic_dominant_function': dominant,
            'civic_resistance_count': gs['resistance_count'],
            'civic_sensitivity_count': gs['sensitivity_count'],
            'civic_evidence_levels': ';'.join(sorted(gs['evidence_levels'])),
            'civic_is_actionable': bool(gs['sensitivity_count'] > 0 or gs['resistance_count'] > 0),
        })

    n = gw.set_node_properties("Gene", "name", updates,
                               source="CIViC_2024")
    _step("Gene CIViC annotations", actual=n, elapsed=time.time() - t0)
    return n


# =========================================================================
# 4. TCGA expression → EXPRESSED_IN edges (Gene → CancerType context)
# =========================================================================

def integrate_tcga_expression(gw):
    """Add TCGA expression data as Gene node properties.

    Per cancer type: mean expression, std, and z-score relative to pan-cancer.
    High-variance genes in specific cancers = tissue-specific signal.
    """
    print("\n  [4/5] TCGA expression", flush=True)
    t0 = time.time()

    path = os.path.join(CACHE, "tcga", "tcga_expression_summary.csv")
    df = pd.read_csv(path)

    # Compute pan-cancer mean per gene for z-scoring
    pan_cancer = df.groupby('gene').agg({'mean': 'mean', 'std': 'mean'}).reset_index()
    pan_cancer.columns = ['gene', 'pan_mean', 'pan_std']
    pan_map = {row['gene']: (row['pan_mean'], row['pan_std'])
               for _, row in pan_cancer.iterrows()}

    # Create edges: gene → cancer_type with expression context
    edges = []
    for _, row in df.iterrows():
        gene = row['gene']
        ct = row['cancer_type']
        pan_m, pan_s = pan_map.get(gene, (0, 1))

        z_score = (row['mean'] - pan_m) / (pan_s + 1e-8) if pan_s > 0 else 0

        # Only create edges for notable expression (|z| > 1 or high absolute)
        if abs(z_score) > 1.0 or row['mean'] > 10000:
            edges.append({
                'from': gene,
                'to': ct,
                'mean_expr': round(float(row['mean']), 2),
                'std_expr': round(float(row['std']), 2),
                'median_expr': round(float(row['median']), 2),
                'n_samples': int(row['n_samples']),
                'z_vs_pancancer': round(float(z_score), 3),
                'is_overexpressed': bool(z_score > 1.5),
                'is_underexpressed': bool(z_score < -1.5),
            })

    # For now, store as Gene node properties (aggregated)
    # because we don't have CancerType nodes yet
    gene_expr = {}
    for e in edges:
        gene = e['from']
        if gene not in gene_expr:
            gene_expr[gene] = {
                'overexpr_cancers': [],
                'underexpr_cancers': [],
                'max_z': -999,
                'min_z': 999,
            }
        ge = gene_expr[gene]
        if e['is_overexpressed']:
            ge['overexpr_cancers'].append(e['to'])
        if e['is_underexpressed']:
            ge['underexpr_cancers'].append(e['to'])
        ge['max_z'] = max(ge['max_z'], e['z_vs_pancancer'])
        ge['min_z'] = min(ge['min_z'], e['z_vs_pancancer'])

    updates = []
    for gene, ge in gene_expr.items():
        updates.append({
            'name': gene,
            'tcga_n_overexpr_cancers': len(ge['overexpr_cancers']),
            'tcga_n_underexpr_cancers': len(ge['underexpr_cancers']),
            'tcga_overexpr_cancers': ';'.join(sorted(ge['overexpr_cancers'])),
            'tcga_underexpr_cancers': ';'.join(sorted(ge['underexpr_cancers'])),
            'tcga_max_expr_z': round(ge['max_z'], 3),
            'tcga_min_expr_z': round(ge['min_z'], 3),
            'tcga_expr_tissue_specific': bool(len(ge['overexpr_cancers']) > 0 or
                                               len(ge['underexpr_cancers']) > 0),
        })

    n = gw.set_node_properties("Gene", "name", updates,
                               source="TCGA_expression_2024")
    _step("Gene expression properties", actual=n, elapsed=time.time() - t0)

    # Also save the full edge data for future use when we add CancerType nodes
    edge_path = os.path.join(CACHE, "tcga", "tcga_expression_edges.csv")
    pd.DataFrame(edges).to_csv(edge_path, index=False)
    _step("Expression edges saved for future", actual=len(edges))

    return n


# =========================================================================
# 5. TCGA CNA → Gene copy number properties
# =========================================================================

def integrate_tcga_cna(gw):
    """Add TCGA copy number data as Gene node properties.

    Amplification/deletion frequencies per cancer type.
    Genes with frequent amplification = potential oncogenes.
    Genes with frequent deletion = potential tumor suppressors.
    """
    print("\n  [5/5] TCGA copy number alterations", flush=True)
    t0 = time.time()

    path = os.path.join(CACHE, "tcga", "tcga_cna_summary.csv")
    df = pd.read_csv(path)

    # Aggregate per gene across cancer types
    gene_cna = {}
    for _, row in df.iterrows():
        gene = row['gene']
        if gene not in gene_cna:
            gene_cna[gene] = {
                'amp_cancers': [],
                'del_cancers': [],
                'max_amp_freq': 0,
                'max_del_freq': 0,
                'mean_amp_freq': [],
                'mean_del_freq': [],
            }
        gc = gene_cna[gene]

        amp = float(row.get('amp_freq', 0))
        dele = float(row.get('del_freq', 0))
        gc['mean_amp_freq'].append(amp)
        gc['mean_del_freq'].append(dele)

        if amp > 0.05:  # >5% amplification frequency
            gc['amp_cancers'].append(row['cancer_type'])
            gc['max_amp_freq'] = max(gc['max_amp_freq'], amp)

        if dele > 0.05:  # >5% deletion frequency
            gc['del_cancers'].append(row['cancer_type'])
            gc['max_del_freq'] = max(gc['max_del_freq'], dele)

    updates = []
    for gene, gc in gene_cna.items():
        mean_amp = float(np.mean(gc['mean_amp_freq'])) if gc['mean_amp_freq'] else 0
        mean_del = float(np.mean(gc['mean_del_freq'])) if gc['mean_del_freq'] else 0

        # Infer oncogene/TSG from CNA pattern
        if mean_amp > mean_del * 3 and mean_amp > 0.02:
            cna_role = 'oncogene'
        elif mean_del > mean_amp * 3 and mean_del > 0.02:
            cna_role = 'tumor_suppressor'
        elif mean_amp > 0.02 and mean_del > 0.02:
            cna_role = 'context_dependent'
        else:
            cna_role = 'neutral'

        updates.append({
            'name': gene,
            'tcga_mean_amp_freq': round(mean_amp, 4),
            'tcga_mean_del_freq': round(mean_del, 4),
            'tcga_max_amp_freq': round(gc['max_amp_freq'], 4),
            'tcga_max_del_freq': round(gc['max_del_freq'], 4),
            'tcga_n_amp_cancers': len(gc['amp_cancers']),
            'tcga_n_del_cancers': len(gc['del_cancers']),
            'tcga_amp_cancers': ';'.join(sorted(gc['amp_cancers'])),
            'tcga_del_cancers': ';'.join(sorted(gc['del_cancers'])),
            'tcga_cna_role': cna_role,
        })

    n = gw.set_node_properties("Gene", "name", updates,
                               source="TCGA_CNA_2024")
    _step("Gene CNA properties", actual=n, elapsed=time.time() - t0)
    return n


# =========================================================================
# Main
# =========================================================================

SOURCE_MAP = {
    'sl': integrate_synthetic_lethality,
    'depmap': integrate_depmap,
    'civic': integrate_civic,
    'expression': integrate_tcga_expression,
    'cna': integrate_tcga_cna,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=list(SOURCE_MAP.keys()),
                        help='Integrate a single source (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without writing to Neo4j')
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  INTEGRATING EXTERNAL DATA SOURCES")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'#'*70}")

    t0 = time.time()
    gw = GraphGateway(dry_run=args.dry_run)

    try:
        if args.source:
            SOURCE_MAP[args.source](gw)
        else:
            for name, func in SOURCE_MAP.items():
                func(gw)
    finally:
        gw.close()

    print(f"\n{'#'*70}")
    print(f"  INTEGRATION COMPLETE [{time.time()-t0:.1f}s]")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
