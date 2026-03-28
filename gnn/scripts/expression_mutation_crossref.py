"""
Expression × mutation cross-reference: is the mutated gene actually expressed?

A mutation in a gene that isn't expressed in that tissue is likely a passenger.
A mutation in a highly expressed gene is more likely functional.

Cross-references per-patient expression (TCGA RNA-seq) with per-patient mutations
to flag:
  - "silent mutations": gene mutated but expression < 10th percentile for that tissue
  - "expressed mutations": gene mutated and expression > 50th percentile
  - "overexpressed + mutated": gene mutated AND overexpressed vs pan-cancer (z > 1.5)

Outputs:
  1. Per HAS_MUTATION edge: expression_context property (silent/low/normal/high/over)
  2. Per gene×CT: fraction of mutations that are in expressed vs silent genes

Usage:
    python3 -u -m gnn.scripts.expression_mutation_crossref [--dry-run]
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import ALL_GENES, GNN_CACHE

TCGA_CACHE = os.path.join(GNN_CACHE, "tcga")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "expression_crossref")


def run_crossref(dry_run=False):
    print("=" * 70)
    print("  EXPRESSION × MUTATION CROSS-REFERENCE")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # Load per-patient expression
    print("\n  Loading expression data...")
    expr_raw = pd.read_csv(os.path.join(TCGA_CACHE, "tcga_expression_raw.csv"))
    print(f"  Expression: {len(expr_raw):,} rows, "
          f"{expr_raw['sample_id'].nunique()} samples, "
          f"{expr_raw['gene'].nunique()} genes")

    # Load mutations
    mut = pd.read_csv(os.path.join(TCGA_CACHE, "tcga_mutations.csv"))
    print(f"  Mutations: {len(mut):,} rows, "
          f"{mut['patient_id'].nunique()} patients")

    # Check sample overlap
    expr_samples = set(expr_raw["sample_id"].unique())
    mut_samples = set(mut["sample_id"].unique())
    overlap = expr_samples & mut_samples
    print(f"  Sample overlap: {len(overlap):,} "
          f"(expr={len(expr_samples):,}, mut={len(mut_samples):,})")

    # Build expression lookup: (sample_id, gene) → expression value
    print("\n  Building expression lookup...")
    t1 = time.time()
    expr_lookup = {}
    for _, row in expr_raw.iterrows():
        expr_lookup[(row["sample_id"], row["gene"])] = row["value"]
    print(f"  Lookup: {len(expr_lookup):,} entries [{time.time()-t1:.1f}s]")

    # Compute per-gene×CT expression percentiles
    print("  Computing tissue-specific percentiles...")
    t1 = time.time()
    gene_ct_percentiles = {}
    for (gene, ct), gdf in expr_raw.groupby(["gene", "cancer_type"]):
        vals = gdf["value"].values
        if len(vals) >= 10:
            gene_ct_percentiles[(gene, ct)] = {
                "p10": float(np.percentile(vals, 10)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    print(f"  Percentiles for {len(gene_ct_percentiles):,} gene×CT pairs [{time.time()-t1:.1f}s]")

    # Pan-cancer gene means (for z-score)
    gene_pancancer = {}
    for gene, gdf in expr_raw.groupby("gene"):
        vals = gdf["value"].values
        gene_pancancer[gene] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Cross-reference: for each mutation, what's the expression context?
    print("\n  Cross-referencing mutations with expression...")
    t1 = time.time()

    results = []
    n_matched = 0
    n_unmatched = 0

    for _, row in mut.iterrows():
        sid = row["sample_id"]
        gene = row["gene"]
        ct = row["cancer_type"]

        expr_val = expr_lookup.get((sid, gene))
        if expr_val is None:
            n_unmatched += 1
            continue

        n_matched += 1
        pcts = gene_ct_percentiles.get((gene, ct))
        pancancer = gene_pancancer.get(gene)

        if pcts is None:
            context = "unknown"
            tissue_percentile = None
            z_vs_pancancer = None
        else:
            # Where does this patient's expression fall in the tissue distribution?
            if expr_val < pcts["p10"]:
                context = "silent"
            elif expr_val < pcts["p25"]:
                context = "low"
            elif expr_val < pcts["p75"]:
                context = "normal"
            elif expr_val < pcts["p90"]:
                context = "high"
            else:
                context = "very_high"

            # Approximate percentile
            if pcts["std"] > 0:
                tissue_percentile = (expr_val - pcts["mean"]) / pcts["std"]
            else:
                tissue_percentile = 0.0

            # Z vs pan-cancer
            if pancancer and pancancer["std"] > 0:
                z_vs_pancancer = (expr_val - pancancer["mean"]) / pancancer["std"]
            else:
                z_vs_pancancer = 0.0

            # Override: if z > 1.5 vs pancancer, mark as overexpressed
            if z_vs_pancancer is not None and z_vs_pancancer > 1.5:
                context = "overexpressed"

        results.append({
            "patient_id": row["patient_id"],
            "sample_id": sid,
            "gene": gene,
            "cancer_type": ct,
            "expression": float(expr_val),
            "context": context,
            "tissue_z": round(float(tissue_percentile), 3) if tissue_percentile is not None else None,
            "pancancer_z": round(float(z_vs_pancancer), 3) if z_vs_pancancer is not None else None,
        })

    results_df = pd.DataFrame(results)
    print(f"  Matched: {n_matched:,}, Unmatched: {n_unmatched:,} [{time.time()-t1:.1f}s]")

    # --- Summary ---
    print(f"\n  Expression context distribution:")
    print(f"  {'Context':<15} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    for ctx, count in results_df["context"].value_counts().items():
        print(f"  {ctx:<15} {count:>8,} {count/len(results_df):>8.1%}")

    # Per cancer type: fraction of mutations that are "silent"
    print(f"\n  Silent mutation fraction by cancer type:")
    print(f"  {'Cancer Type':<12} {'Total':>6} {'Silent':>7} {'Frac':>7}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7}")
    ct_silent = []
    for ct, gdf in results_df.groupby("cancer_type"):
        n_total = len(gdf)
        n_silent = (gdf["context"] == "silent").sum()
        if n_total >= 50:
            ct_silent.append({"ct": ct, "total": n_total, "silent": n_silent,
                              "frac": n_silent / n_total})
    ct_silent.sort(key=lambda x: -x["frac"])
    for row in ct_silent[:20]:
        print(f"  {row['ct']:<12} {row['total']:>6} {row['silent']:>7} {row['frac']:>7.1%}")

    # Per gene: which genes have the most silent mutations?
    print(f"\n  Genes with highest silent mutation rate:")
    print(f"  {'Gene':<12} {'Total':>6} {'Silent':>7} {'Frac':>7}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7}")
    gene_silent = []
    for gene, gdf in results_df.groupby("gene"):
        n_total = len(gdf)
        n_silent = (gdf["context"] == "silent").sum()
        if n_total >= 20:
            gene_silent.append({"gene": gene, "total": n_total, "silent": n_silent,
                                "frac": n_silent / n_total})
    gene_silent.sort(key=lambda x: -x["frac"])
    for row in gene_silent[:20]:
        print(f"  {row['gene']:<12} {row['total']:>6} {row['silent']:>7} {row['frac']:>7.1%}")

    # --- Write to Neo4j ---
    print(f"\n  Writing to Neo4j...")
    t1 = time.time()

    if not dry_run:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))

        # Update HAS_MUTATION edges with expression context
        with driver.session() as session:
            batch_size = 500
            n_written = 0
            for i in range(0, len(results_df), batch_size):
                batch = []
                for _, row in results_df.iloc[i:i+batch_size].iterrows():
                    entry = {
                        "pid": row["patient_id"],
                        "gene": row["gene"],
                        "context": row["context"],
                        "expr": round(float(row["expression"]), 2),
                    }
                    if row["tissue_z"] is not None:
                        entry["tissue_z"] = float(row["tissue_z"])
                    if row["pancancer_z"] is not None:
                        entry["pancancer_z"] = float(row["pancancer_z"])
                    batch.append(entry)

                session.run("""
                    UNWIND $batch AS b
                    MATCH (p:Patient {id: b.pid})-[r:HAS_MUTATION]->(g:Gene {name: b.gene})
                    SET r.expression_context = b.context,
                        r.expression_value = b.expr,
                        r.expression_tissue_z = b.tissue_z,
                        r.expression_pancancer_z = b.pancancer_z
                """, batch=batch)
                n_written += len(batch)

            print(f"  Updated {n_written:,} HAS_MUTATION edges [{time.time()-t1:.1f}s]")

        driver.close()
    else:
        print(f"  [DRY RUN] Would update {len(results_df):,} HAS_MUTATION edges")

    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, "expression_crossref.csv"), index=False)

    summary = {
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "context_distribution": results_df["context"].value_counts().to_dict(),
        "n_genes": int(results_df["gene"].nunique()),
        "n_patients": int(results_df["patient_id"].nunique()),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(RESULTS_DIR, "expression_crossref_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total time: {time.time()-t0:.1f}s")
    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_crossref(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
