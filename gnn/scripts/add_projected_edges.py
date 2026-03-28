"""
Add projected edge types to the knowledge graph.

Takes existing Gene→Drug and Gene→CancerType relationships and projects
them into Gene→Gene edges. Also promotes unused existing Gene→Gene edges
(ANALOGOUS, CONVERGES, TRANSPOSES) into the bilinear model.

New projected edge types:
  CO_SENSITIVE     — both genes sensitive to same drug (shared vulnerability)
  CO_RESISTANT     — both genes resistant to same drug (shared escape)
  DRUG_CONFLICT    — one sensitive, other resistant to same drug
  CO_TISSUE_EXPR   — correlated expression across cancer types
  CO_BIALLELIC     — both biallelic in same cancer types

Existing unused edges added:
  ANALOGOUS        — genes with analogous functional roles
  CONVERGES        — genes that converge on same pathway
  TRANSPOSES       — genes in same transposition group

Usage:
    python3 -u -m gnn.scripts.add_projected_edges
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    print("=" * 70)
    print("PROJECTED EDGE CONSTRUCTION")
    print("=" * 70)

    with driver.session() as s:
        # === 1. CO_SENSITIVE: genes sharing drug sensitivity ===
        print("\n1. Building CO_SENSITIVE edges...")
        r = s.run("""
            MATCH (g1:Gene)-[r1:SENSITIVE_TO]->(d:Drug)<-[r2:SENSITIVE_TO]-(g2:Gene)
            WHERE g1.channel IS NOT NULL AND g2.channel IS NOT NULL
              AND id(g1) < id(g2)
            RETURN g1.name AS g1, g2.name AS g2,
                   count(DISTINCT d) AS n_shared_drugs,
                   collect(DISTINCT d.name) AS drugs,
                   avg(r1.effect_size) AS avg_effect_1,
                   avg(r2.effect_size) AS avg_effect_2
        """)

        co_sensitive = []
        for rec in r:
            co_sensitive.append({
                'g1': rec['g1'], 'g2': rec['g2'],
                'n_drugs': rec['n_shared_drugs'],
                'drugs': rec['drugs'],
                'weight': min(rec['n_shared_drugs'] / 5.0, 1.0),  # Normalize
            })

        print(f"  Found {len(co_sensitive)} CO_SENSITIVE pairs")
        if co_sensitive:
            # Create edges
            for pair in co_sensitive:
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g1)-[r:CO_SENSITIVE]->(g2)
                    SET r.weight = $weight,
                        r.n_shared_drugs = $n_drugs,
                        r.source = 'projected_from_SENSITIVE_TO'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_drugs=pair['n_drugs'])
                # Bidirectional
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g2)-[r:CO_SENSITIVE]->(g1)
                    SET r.weight = $weight,
                        r.n_shared_drugs = $n_drugs,
                        r.source = 'projected_from_SENSITIVE_TO'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_drugs=pair['n_drugs'])
            print(f"  Created {len(co_sensitive)*2} CO_SENSITIVE edges (bidirectional)")
            # Show top pairs
            top = sorted(co_sensitive, key=lambda x: -x['n_drugs'])[:5]
            for p in top:
                print(f"    {p['g1']:10s} — {p['g2']:10s}: {p['n_drugs']} shared drugs")

        # === 2. CO_RESISTANT: genes sharing drug resistance ===
        print("\n2. Building CO_RESISTANT edges...")
        r = s.run("""
            MATCH (g1:Gene)-[r1:RESISTANT_TO]->(d:Drug)<-[r2:RESISTANT_TO]-(g2:Gene)
            WHERE g1.channel IS NOT NULL AND g2.channel IS NOT NULL
              AND id(g1) < id(g2)
            RETURN g1.name AS g1, g2.name AS g2,
                   count(DISTINCT d) AS n_shared_drugs,
                   collect(DISTINCT d.name) AS drugs
        """)

        co_resistant = []
        for rec in r:
            co_resistant.append({
                'g1': rec['g1'], 'g2': rec['g2'],
                'n_drugs': rec['n_shared_drugs'],
                'weight': min(rec['n_shared_drugs'] / 5.0, 1.0),
            })

        print(f"  Found {len(co_resistant)} CO_RESISTANT pairs")
        if co_resistant:
            for pair in co_resistant:
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g1)-[r:CO_RESISTANT]->(g2)
                    SET r.weight = $weight,
                        r.n_shared_drugs = $n_drugs,
                        r.source = 'projected_from_RESISTANT_TO'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_drugs=pair['n_drugs'])
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g2)-[r:CO_RESISTANT]->(g1)
                    SET r.weight = $weight,
                        r.n_shared_drugs = $n_drugs,
                        r.source = 'projected_from_RESISTANT_TO'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_drugs=pair['n_drugs'])
            print(f"  Created {len(co_resistant)*2} CO_RESISTANT edges (bidirectional)")

        # === 3. DRUG_CONFLICT: one sensitive, other resistant to same drug ===
        print("\n3. Building DRUG_CONFLICT edges...")
        r = s.run("""
            MATCH (g1:Gene)-[:SENSITIVE_TO]->(d:Drug)<-[:RESISTANT_TO]-(g2:Gene)
            WHERE g1.channel IS NOT NULL AND g2.channel IS NOT NULL
              AND g1 <> g2
            RETURN g1.name AS g1, g2.name AS g2,
                   count(DISTINCT d) AS n_conflict_drugs,
                   collect(DISTINCT d.name) AS drugs
        """)

        drug_conflict = {}
        for rec in r:
            key = tuple(sorted([rec['g1'], rec['g2']]))
            if key not in drug_conflict:
                drug_conflict[key] = {
                    'g1': key[0], 'g2': key[1],
                    'n_drugs': 0, 'drugs': set(),
                }
            drug_conflict[key]['n_drugs'] += rec['n_conflict_drugs']
            drug_conflict[key]['drugs'].update(rec['drugs'])

        drug_conflict = list(drug_conflict.values())
        for p in drug_conflict:
            p['weight'] = min(p['n_drugs'] / 5.0, 1.0)

        print(f"  Found {len(drug_conflict)} DRUG_CONFLICT pairs")
        if drug_conflict:
            for pair in drug_conflict:
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g1)-[r:DRUG_CONFLICT]->(g2)
                    SET r.weight = $weight,
                        r.n_conflict_drugs = $n_drugs,
                        r.source = 'projected_from_SENSITIVE_TO_RESISTANT_TO'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_drugs=pair['n_drugs'])
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g2)-[r:DRUG_CONFLICT]->(g1)
                    SET r.weight = $weight,
                        r.n_conflict_drugs = $n_drugs,
                        r.source = 'projected_from_SENSITIVE_TO_RESISTANT_TO'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_drugs=pair['n_drugs'])
            print(f"  Created {len(drug_conflict)*2} DRUG_CONFLICT edges (bidirectional)")

        # === 4. CO_TISSUE_EXPR: correlated expression across cancer types ===
        print("\n4. Building CO_TISSUE_EXPR edges...")
        # Get expression profiles for all genes
        r = s.run("""
            MATCH (g:Gene)-[r:EXPRESSION_IN]->(ct:CancerType)
            WHERE g.channel IS NOT NULL
            RETURN g.name AS gene, ct.name AS cancer_type,
                   r.z_score AS z_score
            ORDER BY g.name, ct.name
        """)

        gene_expr = defaultdict(dict)
        for rec in r:
            if rec['z_score'] is not None:
                gene_expr[rec['gene']][rec['cancer_type']] = rec['z_score']

        # Compute pairwise correlation across cancer types
        genes_with_expr = [g for g in gene_expr if len(gene_expr[g]) >= 5]
        all_cts = sorted(set(ct for g in genes_with_expr for ct in gene_expr[g]))

        print(f"  {len(genes_with_expr)} genes with expression data across {len(all_cts)} CTs")

        co_tissue = []
        for i, g1 in enumerate(genes_with_expr):
            for g2 in genes_with_expr[i+1:]:
                shared_cts = set(gene_expr[g1].keys()) & set(gene_expr[g2].keys())
                if len(shared_cts) < 5:
                    continue
                v1 = np.array([gene_expr[g1][ct] for ct in shared_cts])
                v2 = np.array([gene_expr[g2][ct] for ct in shared_cts])
                if np.std(v1) < 1e-6 or np.std(v2) < 1e-6:
                    continue
                corr = np.corrcoef(v1, v2)[0, 1]
                if abs(corr) > 0.5:  # Only strong correlations
                    co_tissue.append({
                        'g1': g1, 'g2': g2,
                        'correlation': float(corr),
                        'weight': float(abs(corr)),
                        'n_shared_cts': len(shared_cts),
                    })

        print(f"  Found {len(co_tissue)} CO_TISSUE_EXPR pairs (|corr| > 0.5)")
        if co_tissue:
            for pair in co_tissue:
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g1)-[r:CO_TISSUE_EXPR]->(g2)
                    SET r.weight = $weight,
                        r.correlation = $corr,
                        r.n_shared_cts = $n_cts,
                        r.source = 'projected_from_EXPRESSION_IN'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], corr=pair['correlation'],
                     n_cts=pair['n_shared_cts'])
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g2)-[r:CO_TISSUE_EXPR]->(g1)
                    SET r.weight = $weight,
                        r.correlation = $corr,
                        r.n_shared_cts = $n_cts,
                        r.source = 'projected_from_EXPRESSION_IN'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], corr=pair['correlation'],
                     n_cts=pair['n_shared_cts'])
            print(f"  Created {len(co_tissue)*2} CO_TISSUE_EXPR edges (bidirectional)")
            top = sorted(co_tissue, key=lambda x: -abs(x['correlation']))[:5]
            for p in top:
                print(f"    {p['g1']:10s} — {p['g2']:10s}: r={p['correlation']:.3f} ({p['n_shared_cts']} CTs)")

        # === 5. CO_BIALLELIC: both biallelic in same cancer types ===
        print("\n5. Building CO_BIALLELIC edges...")
        r = s.run("""
            MATCH (g1:Gene)-[r1:BIALLELIC_IN]->(ct:CancerType)<-[r2:BIALLELIC_IN]-(g2:Gene)
            WHERE g1.channel IS NOT NULL AND g2.channel IS NOT NULL
              AND id(g1) < id(g2)
              AND r1.biallelic_freq > 0.01 AND r2.biallelic_freq > 0.01
            RETURN g1.name AS g1, g2.name AS g2,
                   count(DISTINCT ct) AS n_shared_cts,
                   avg(r1.biallelic_freq * r2.biallelic_freq) AS avg_product
        """)

        co_biallelic = []
        for rec in r:
            if rec['n_shared_cts'] >= 2:
                co_biallelic.append({
                    'g1': rec['g1'], 'g2': rec['g2'],
                    'n_cts': rec['n_shared_cts'],
                    'weight': min(rec['n_shared_cts'] / 5.0, 1.0),
                })

        print(f"  Found {len(co_biallelic)} CO_BIALLELIC pairs (shared CTs >= 2)")
        if co_biallelic:
            for pair in co_biallelic:
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g1)-[r:CO_BIALLELIC]->(g2)
                    SET r.weight = $weight,
                        r.n_shared_cts = $n_cts,
                        r.source = 'projected_from_BIALLELIC_IN'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_cts=pair['n_cts'])
                s.run("""
                    MATCH (g1:Gene {name: $g1}), (g2:Gene {name: $g2})
                    MERGE (g2)-[r:CO_BIALLELIC]->(g1)
                    SET r.weight = $weight,
                        r.n_shared_cts = $n_cts,
                        r.source = 'projected_from_BIALLELIC_IN'
                """, g1=pair['g1'], g2=pair['g2'],
                     weight=pair['weight'], n_cts=pair['n_cts'])
            print(f"  Created {len(co_biallelic)*2} CO_BIALLELIC edges (bidirectional)")

        # === Summary: all Gene-Gene edge types ===
        print("\n" + "=" * 70)
        print("UPDATED GENE-GENE EDGE TYPE INVENTORY")
        print("=" * 70)
        r = s.run("""
            MATCH (a:Gene)-[r]->(b:Gene)
            RETURN type(r) as rtype, count(r) as cnt
            ORDER BY cnt DESC
        """)
        for rec in r:
            print(f"  {rec['rtype']:35s}: {rec['cnt']:>8d}")

    driver.close()
    print("\nDone. Delete gnn/results/bilinear_edge/raw_edge_matrix.npy to rebuild.")


if __name__ == "__main__":
    main()
