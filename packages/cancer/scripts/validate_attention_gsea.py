"""
Validate attention edges from DepMap pre-training against MSigDB Hallmark pathways.

Tests whether the gene pairs discovered by the transformer's attention mechanism
correspond to known biological pathways, and whether intra-channel attention edges
enrich for that channel's expected pathways.

Uses Fisher's exact test for over-representation analysis.

Usage:
    python3 -u -m gnn.scripts.validate_attention_gsea
    python3 -u -m gnn.scripts.validate_attention_gsea --edges path/to/attention_edges.json
"""

import os, sys, json, argparse
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, ALL_GENES, GNN_RESULTS
from gnn.scripts.expand_channel_map import load_hallmark_gmt, HALLMARK_TO_CHANNEL

RESULTS_DIR = os.path.join(GNN_RESULTS, "depmap_pretrain")
GMT_PATH = os.path.join(GNN_RESULTS, "h.all.v2024.1.Hs.symbols.gmt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=str,
                        default=os.path.join(RESULTS_DIR, "attention_edges.json"))
    parser.add_argument("--min-weight", type=float, default=0.05,
                        help="Minimum attention weight to include")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Permutation tests for FDR estimation")
    return parser.parse_args()


def load_attention_edges(path, min_weight=0.05):
    """Load attention edges, filter self-loops and low-weight pairs."""
    with open(path) as f:
        raw = json.load(f)

    edges = []
    for e in raw:
        if e["from"] == e["to"]:
            continue  # skip self-loops
        if e["weight"] < min_weight:
            continue
        edges.append(e)

    return edges


def build_gene_sets_from_edges(edges):
    """Group attention edges by channel relationship.

    Returns:
        per_channel: {channel_name: set of genes in intra-channel edges}
        cross_channel_genes: set of genes in cross-channel edges
        all_attention_genes: set of all genes in any edge
        edge_gene_pairs: set of (gene_a, gene_b) tuples
    """
    per_channel = defaultdict(set)
    cross_channel_genes = set()
    all_attention_genes = set()
    edge_gene_pairs = set()

    for e in edges:
        ga, gb = e["from"], e["to"]
        all_attention_genes.add(ga)
        all_attention_genes.add(gb)
        edge_gene_pairs.add((min(ga, gb), max(ga, gb)))

        ch_a = CHANNEL_MAP.get(ga)
        ch_b = CHANNEL_MAP.get(gb)

        if e.get("cross_channel", False) or (ch_a and ch_b and ch_a != ch_b):
            cross_channel_genes.add(ga)
            cross_channel_genes.add(gb)
        else:
            # Intra-channel: add to both genes' channel
            if ch_a:
                per_channel[ch_a].add(ga)
                per_channel[ch_a].add(gb)
            elif ch_b:
                per_channel[ch_b].add(ga)
                per_channel[ch_b].add(gb)

    return per_channel, cross_channel_genes, all_attention_genes, edge_gene_pairs


def fisher_enrichment(gene_set, pathway_genes, background_size):
    """Fisher's exact test for over-representation.

    Tests: are genes in gene_set over-represented in pathway_genes?

    Returns: odds_ratio, p_value, overlap_genes
    """
    overlap = gene_set & pathway_genes
    a = len(overlap)                              # in set AND in pathway
    b = len(gene_set - pathway_genes)             # in set, NOT in pathway
    c = len(pathway_genes - gene_set)             # NOT in set, in pathway
    d = background_size - a - b - c               # NOT in set, NOT in pathway
    d = max(d, 0)

    table = [[a, b], [c, d]]
    odds, pval = fisher_exact(table, alternative="greater")

    return {
        "odds_ratio": round(float(odds), 3) if np.isfinite(odds) else 999.0,
        "p_value": float(pval),
        "overlap": sorted(overlap),
        "n_overlap": a,
        "n_set": len(gene_set),
        "n_pathway": len(pathway_genes),
    }


def run_enrichment_analysis(gene_sets_dict, hallmark_sets, background_genes):
    """Run Fisher's exact test for each gene set against each Hallmark pathway.

    Args:
        gene_sets_dict: {set_name: set of genes}
        hallmark_sets: {hallmark_name: set of genes}
        background_genes: set of all genes (universe)

    Returns:
        results: list of enrichment hits
    """
    bg_size = len(background_genes)
    results = []

    for set_name, gene_set in gene_sets_dict.items():
        # Intersect with background
        gene_set_bg = gene_set & background_genes
        if len(gene_set_bg) < 2:
            continue

        for hw_name, hw_genes in hallmark_sets.items():
            hw_genes_bg = hw_genes & background_genes
            if len(hw_genes_bg) < 5:
                continue

            enrichment = fisher_enrichment(gene_set_bg, hw_genes_bg, bg_size)

            if enrichment["n_overlap"] > 0:
                expected_channel = HALLMARK_TO_CHANNEL.get(hw_name, "unmapped")
                results.append({
                    "gene_set": set_name,
                    "hallmark": hw_name,
                    "expected_channel": expected_channel,
                    **enrichment,
                })

    # Sort by p-value
    results.sort(key=lambda x: x["p_value"])

    # BH FDR correction
    n_tests = sum(1 for sn in gene_sets_dict for _ in hallmark_sets)
    for i, r in enumerate(results):
        rank = i + 1
        r["q_value"] = min(r["p_value"] * n_tests / rank, 1.0)

    return results


def permutation_test(edges, hallmark_sets, background_genes, n_perms=1000):
    """Shuffle gene labels and re-run enrichment to estimate null distribution.

    Returns the fraction of permutations with more significant hits than real data.
    """
    bg_list = sorted(background_genes)
    bg_size = len(bg_list)

    # Real data: count significant hits (p < 0.05)
    per_channel, _, all_genes, _ = build_gene_sets_from_edges(edges)
    gene_sets = {f"channel_{ch}": genes for ch, genes in per_channel.items()}
    gene_sets["all_attention"] = all_genes
    real_results = run_enrichment_analysis(gene_sets, hallmark_sets, background_genes)
    real_sig = sum(1 for r in real_results if r["p_value"] < 0.05)

    perm_sig_counts = []
    for _ in range(n_perms):
        # Shuffle: randomly relabel genes in edges
        gene_map = dict(zip(sorted(all_genes),
                            np.random.choice(bg_list, size=len(all_genes), replace=False)))
        shuffled_edges = []
        for e in edges:
            if e["from"] == e["to"]:
                continue
            shuffled_edges.append({
                **e,
                "from": gene_map.get(e["from"], e["from"]),
                "to": gene_map.get(e["to"], e["to"]),
            })

        perm_channel, _, perm_all, _ = build_gene_sets_from_edges(shuffled_edges)
        perm_sets = {f"channel_{ch}": genes for ch, genes in perm_channel.items()}
        perm_sets["all_attention"] = perm_all
        perm_results = run_enrichment_analysis(perm_sets, hallmark_sets, background_genes)
        perm_sig = sum(1 for r in perm_results if r["p_value"] < 0.05)
        perm_sig_counts.append(perm_sig)

    empirical_p = np.mean([c >= real_sig for c in perm_sig_counts])

    return {
        "real_significant_hits": real_sig,
        "permutation_mean": float(np.mean(perm_sig_counts)),
        "permutation_std": float(np.std(perm_sig_counts)),
        "empirical_p": float(empirical_p),
        "n_permutations": n_perms,
    }


def channel_affinity_analysis(enrichment_results):
    """Check: do channel attention edges enrich for their own channel's pathways?

    For each channel_X gene set, check if the top enrichment hits map to
    channel X's expected Hallmark pathways via HALLMARK_TO_CHANNEL.
    """
    channel_hits = defaultdict(list)
    for r in enrichment_results:
        if r["gene_set"].startswith("channel_"):
            channel_name = r["gene_set"].replace("channel_", "")
            channel_hits[channel_name].append(r)

    affinity_results = {}
    for channel, hits in channel_hits.items():
        sig_hits = [h for h in hits if h["p_value"] < 0.05]
        matching = [h for h in sig_hits if h["expected_channel"] == channel]
        non_matching = [h for h in sig_hits if h["expected_channel"] != channel]

        affinity_results[channel] = {
            "total_sig_hits": len(sig_hits),
            "matching_channel": len(matching),
            "non_matching": len(non_matching),
            "affinity_ratio": len(matching) / max(len(sig_hits), 1),
            "matching_pathways": [h["hallmark"] for h in matching],
            "non_matching_pathways": [
                {"hallmark": h["hallmark"], "expected": h["expected_channel"],
                 "p": h["p_value"]}
                for h in non_matching
            ],
        }

    return affinity_results


def main():
    args = parse_args()

    print("=" * 60)
    print("  GSEA PATHWAY VALIDATION OF ATTENTION EDGES")
    print("=" * 60)

    # Load attention edges
    edges = load_attention_edges(args.edges, min_weight=args.min_weight)
    print(f"\n  Attention edges: {len(edges)} (after filtering self-loops, min_weight={args.min_weight})")

    if not edges:
        print("  No edges to validate. Run pretrain_depmap.py first.")
        return

    # Load Hallmark gene sets
    if not os.path.exists(GMT_PATH):
        from gnn.scripts.expand_channel_map import download_hallmark_gmt
        download_hallmark_gmt()
    hallmark_sets = load_hallmark_gmt(GMT_PATH)
    print(f"  Hallmark gene sets: {len(hallmark_sets)}")

    # Background: all genes in the graph
    background_genes = set(ALL_GENES)
    print(f"  Background genes: {len(background_genes)}")

    # Build gene sets from attention edges
    per_channel, cross_channel, all_attention, edge_pairs = build_gene_sets_from_edges(edges)

    print(f"\n  Attention gene sets:")
    print(f"    All attention genes: {len(all_attention)}")
    for ch in CHANNEL_NAMES:
        if ch in per_channel:
            print(f"    {ch}: {len(per_channel[ch])} genes")
    print(f"    Cross-channel: {len(cross_channel)} genes")

    # Build test sets
    gene_sets = {}
    gene_sets["all_attention"] = all_attention
    for ch, genes in per_channel.items():
        if len(genes) >= 3:
            gene_sets[f"channel_{ch}"] = genes
    if len(cross_channel) >= 3:
        gene_sets["cross_channel"] = cross_channel

    # Run enrichment
    print(f"\n  Running Fisher's exact enrichment ({len(gene_sets)} sets × {len(hallmark_sets)} pathways)...")
    results = run_enrichment_analysis(gene_sets, hallmark_sets, background_genes)

    # Print significant results
    sig_results = [r for r in results if r["p_value"] < 0.05]
    print(f"\n  Significant enrichments (p < 0.05): {len(sig_results)}")

    if sig_results:
        print(f"\n  {'Gene Set':<25s} {'Hallmark Pathway':<45s} {'p-val':>8s} {'q-val':>8s} "
              f"{'OR':>6s} {'Overlap':>8s} {'Expected Ch':>12s}")
        print("  " + "-" * 120)
        for r in sig_results[:30]:
            hw_short = r["hallmark"].replace("HALLMARK_", "")
            match = "*" if r["gene_set"].replace("channel_", "") == r["expected_channel"] else " "
            print(f"  {r['gene_set']:<25s} {hw_short:<45s} {r['p_value']:>8.4f} {r['q_value']:>8.4f} "
                  f"{r['odds_ratio']:>6.1f} {r['n_overlap']:>3d}/{r['n_set']:<3d} "
                  f"{r['expected_channel']:>12s}{match}")

    # Channel affinity analysis
    print(f"\n  CHANNEL AFFINITY ANALYSIS")
    print(f"  Do attention edges within a channel enrich for that channel's pathways?")
    print(f"  " + "-" * 60)

    affinity = channel_affinity_analysis(results)
    total_matching = 0
    total_sig = 0
    for ch in CHANNEL_NAMES:
        if ch in affinity:
            a = affinity[ch]
            total_matching += a["matching_channel"]
            total_sig += a["total_sig_hits"]
            ratio_str = f"{a['affinity_ratio']:.0%}" if a["total_sig_hits"] > 0 else "n/a"
            print(f"    {ch:<15s}: {a['matching_channel']}/{a['total_sig_hits']} "
                  f"hits match expected channel ({ratio_str})")
            if a["matching_pathways"]:
                for pw in a["matching_pathways"]:
                    print(f"      + {pw.replace('HALLMARK_', '')}")
            if a["non_matching_pathways"]:
                for pw in a["non_matching_pathways"][:3]:
                    print(f"      - {pw['hallmark'].replace('HALLMARK_', '')} "
                          f"(expected: {pw['expected']}, p={pw['p']:.4f})")

    if total_sig > 0:
        print(f"\n    Overall affinity: {total_matching}/{total_sig} "
              f"({total_matching/total_sig:.0%}) significant hits match expected channel")

    # Permutation test
    if args.n_permutations > 0 and len(edges) >= 10:
        print(f"\n  Running permutation test ({args.n_permutations} permutations)...")
        perm = permutation_test(edges, hallmark_sets, background_genes, args.n_permutations)
        print(f"    Real significant hits: {perm['real_significant_hits']}")
        print(f"    Permuted mean: {perm['permutation_mean']:.1f} +/- {perm['permutation_std']:.1f}")
        print(f"    Empirical p-value: {perm['empirical_p']:.4f}")
    else:
        perm = None

    # Save results
    output = {
        "n_edges": len(edges),
        "n_genes": len(all_attention),
        "per_channel_sizes": {ch: len(genes) for ch, genes in per_channel.items()},
        "enrichment_results": results,
        "significant_results": sig_results,
        "channel_affinity": affinity,
        "permutation_test": perm,
    }

    out_path = os.path.join(RESULTS_DIR, "gsea_validation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
