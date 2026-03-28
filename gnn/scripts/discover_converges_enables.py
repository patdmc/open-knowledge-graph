#!/usr/bin/env python3
"""
Discover CONVERGES and ENABLES edges from co-mutation patterns.

CONVERGES: Mutual exclusivity — same delete at different addresses.
  If genes A and B are rarely mutated together (obs/exp < threshold),
  they likely break the same pathway — the cell only needs one hit.
  Sign: NEGATIVE (anti-correlated, same functional deletion)

ENABLES: Directional co-occurrence — removing A unleashes B.
  If P(B|A) >> P(B) but P(A|B) ~ P(A), then A enables B.
  Sign: POSITIVE but ASYMMETRIC (directional edge)

Usage:
    python3 -u -m gnn.scripts.discover_converges_enables
    python3 -u -m gnn.scripts.discover_converges_enables --dry-run
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_CACHE


def load_comutation_data():
    """Load patient-gene mutation matrix from Neo4j."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    print("  Loading patient × gene mutation data from Neo4j...")
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)
            RETURN p.id AS pid, g.name AS gene
        """)
        edges = [(r["pid"], r["gene"]) for r in result]

    driver.close()

    # Build patient → genes and gene → patients
    patient_genes = defaultdict(set)
    gene_patients = defaultdict(set)
    for pid, gene in edges:
        patient_genes[pid].add(gene)
        gene_patients[gene].add(pid)

    n_patients = len(patient_genes)
    genes = sorted(gene_patients.keys())
    gene_freq = {g: len(ps) / n_patients for g, ps in gene_patients.items()}

    print(f"  {n_patients} patients, {len(genes)} genes")
    return patient_genes, gene_patients, gene_freq, n_patients


def discover_converges(gene_patients, gene_freq, n_patients,
                       max_obs_exp=0.3, min_patients_each=100):
    """Find mutually exclusive gene pairs (CONVERGES).

    obs/exp = observed_co / (freq_A × freq_B × N)
    Low obs/exp means the pair co-occurs far less than expected by chance.
    """
    from scipy.stats import fisher_exact

    genes = [g for g, ps in gene_patients.items() if len(ps) >= min_patients_each]
    print(f"  Testing ME for {len(genes)} genes (>= {min_patients_each} patients each)")

    pairs = []
    n_tested = 0

    for i, g1 in enumerate(genes):
        ps1 = gene_patients[g1]
        for j in range(i + 1, len(genes)):
            g2 = genes[j]
            ps2 = gene_patients[g2]
            n_tested += 1

            # 2×2 contingency table
            both = len(ps1 & ps2)
            only1 = len(ps1) - both
            only2 = len(ps2) - both
            neither = n_patients - only1 - only2 - both

            expected = gene_freq[g1] * gene_freq[g2] * n_patients
            obs_exp = both / max(expected, 1e-6)

            if obs_exp < max_obs_exp and expected >= 5:
                # Fisher exact test for significance
                try:
                    odds_ratio, pval = fisher_exact(
                        [[both, only1], [only2, neither]],
                        alternative="less"
                    )
                except:
                    continue

                pairs.append({
                    "from": g1,
                    "to": g2,
                    "obs_exp": round(obs_exp, 4),
                    "observed": both,
                    "expected": round(expected, 1),
                    "pvalue": float(pval),
                    "odds_ratio": round(float(odds_ratio), 4) if np.isfinite(odds_ratio) else 0.0,
                    "n_patients_a": len(ps1),
                    "n_patients_b": len(ps2),
                })

    # Filter by significance
    pairs = [p for p in pairs if p["pvalue"] < 0.001]
    pairs.sort(key=lambda p: p["obs_exp"])

    print(f"  Tested {n_tested} pairs, found {len(pairs)} CONVERGES "
          f"(obs/exp < {max_obs_exp}, p < 0.001)")
    return pairs


def discover_enables(patient_genes, gene_patients, gene_freq, n_patients,
                     min_lift=1.5, min_asymmetry=1.3, min_patients=100):
    """Find directional co-occurrence (ENABLES).

    Gene A ENABLES gene B if:
      - P(B|A) >> P(B)  (lift > threshold)
      - P(B|A) / P(B) >> P(A|B) / P(A)  (asymmetric)

    This means mutating A creates conditions where B is more likely to be
    mutated — A removes a constraint that B depends on.
    """
    genes = [g for g, ps in gene_patients.items() if len(ps) >= min_patients]
    print(f"  Testing ENABLES for {len(genes)} genes (>= {min_patients} patients each)")

    pairs = []
    n_tested = 0

    for g1 in genes:
        ps1 = gene_patients[g1]
        for g2 in genes:
            if g1 >= g2:
                continue
            n_tested += 1
            ps2 = gene_patients[g2]

            both = len(ps1 & ps2)
            if both < 10:
                continue

            # Conditional probabilities
            p_b_given_a = both / len(ps1)  # P(B|A)
            p_a_given_b = both / len(ps2)  # P(A|B)

            lift_a_to_b = p_b_given_a / max(gene_freq[g2], 1e-6)
            lift_b_to_a = p_a_given_b / max(gene_freq[g1], 1e-6)

            # Check A -> B direction
            if lift_a_to_b >= min_lift and lift_a_to_b >= min_asymmetry * lift_b_to_a:
                pairs.append({
                    "from": g1,
                    "to": g2,
                    "lift": round(lift_a_to_b, 3),
                    "reverse_lift": round(lift_b_to_a, 3),
                    "asymmetry": round(lift_a_to_b / max(lift_b_to_a, 1e-6), 3),
                    "p_b_given_a": round(p_b_given_a, 4),
                    "p_a_given_b": round(p_a_given_b, 4),
                    "n_co": both,
                    "n_patients_a": len(ps1),
                    "n_patients_b": len(ps2),
                })

            # Check B -> A direction
            if lift_b_to_a >= min_lift and lift_b_to_a >= min_asymmetry * lift_a_to_b:
                pairs.append({
                    "from": g2,
                    "to": g1,
                    "lift": round(lift_b_to_a, 3),
                    "reverse_lift": round(lift_a_to_b, 3),
                    "asymmetry": round(lift_b_to_a / max(lift_a_to_b, 1e-6), 3),
                    "p_b_given_a": round(p_a_given_b, 4),
                    "p_a_given_b": round(p_b_given_a, 4),
                    "n_co": both,
                    "n_patients_a": len(ps2),
                    "n_patients_b": len(ps1),
                })

    pairs.sort(key=lambda p: -p["asymmetry"])
    print(f"  Tested {n_tested} pairs, found {len(pairs)} ENABLES "
          f"(lift >= {min_lift}, asymmetry >= {min_asymmetry})")
    return pairs


def write_to_neo4j(converges_pairs, enables_pairs, dry_run=False):
    """Write CONVERGES and ENABLES edges to Neo4j."""
    from gnn.data.graph_changelog import GraphGateway

    gw = GraphGateway(dry_run=dry_run)
    mode = " (dry run)" if dry_run else ""

    if converges_pairs:
        n = gw.merge_edges("CONVERGES", converges_pairs,
                           source="discover_converges_enables",
                           source_detail=f"ME obs/exp < 0.3, p < 0.001")
        print(f"  Wrote {n} CONVERGES edges{mode}")

    if enables_pairs:
        n = gw.merge_edges("ENABLES", enables_pairs,
                           source="discover_converges_enables",
                           source_detail=f"lift >= 1.5, asymmetry >= 1.3")
        print(f"  Wrote {n} ENABLES edges{mode}")

    gw.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  CONVERGES + ENABLES EDGE DISCOVERY")
    print("=" * 70)
    t0 = time.time()

    patient_genes, gene_patients, gene_freq, n_patients = load_comutation_data()

    # --- CONVERGES ---
    print("\n  --- CONVERGES (mutual exclusivity) ---")
    converges = discover_converges(gene_patients, gene_freq, n_patients)
    if converges:
        print(f"\n  Top 15 CONVERGES pairs (most exclusive):")
        for p in converges[:15]:
            print(f"    {p['from']:>10s} × {p['to']:<10s}  obs/exp={p['obs_exp']:.3f}  "
                  f"observed={p['observed']:>3d}  expected={p['expected']:>6.1f}  p={p['pvalue']:.2e}")

    # --- ENABLES ---
    print("\n  --- ENABLES (directional co-occurrence) ---")
    enables = discover_enables(patient_genes, gene_patients, gene_freq, n_patients)
    if enables:
        print(f"\n  Top 15 ENABLES pairs (most asymmetric):")
        for p in enables[:15]:
            print(f"    {p['from']:>10s} -> {p['to']:<10s}  lift={p['lift']:.2f}  "
                  f"reverse={p['reverse_lift']:.2f}  asymmetry={p['asymmetry']:.1f}  "
                  f"n_co={p['n_co']}")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "converges_enables")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({
            "n_converges": len(converges),
            "n_enables": len(enables),
            "top_converges": converges[:50],
            "top_enables": enables[:50],
            "elapsed_seconds": round(time.time() - t0, 1),
        }, f, indent=2)

    # Write to Neo4j
    if not args.stats:
        print("\n  --- Writing to Neo4j ---")
        write_to_neo4j(converges, enables, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    print(f"  DONE [{time.time() - t0:.1f}s]")
    print(f"  {len(converges)} CONVERGES, {len(enables)} ENABLES")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
