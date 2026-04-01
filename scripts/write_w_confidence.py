#!/usr/bin/env python3
"""
Write bilinear W-derived confidence scores back to Neo4j edges.

For each gene-gene edge, compute the per-CT bilinear score:
    confidence_ct = e_ij^T W_ct e_ij

This closes the feedback loop: the model learned which edges matter
per cancer type, and that knowledge goes back onto the edges as
confidence properties. Future models and walks use these weights.

Properties added per edge:
    w_<cancer_type>: float — bilinear score for that CT
    w_max: float — max absolute score across all CTs
    w_mean: float — mean score across all CTs

Original edge properties are preserved (graph durability rule).
"""

import os
import sys
import json
import time
import numpy as np

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "openknowledgegraph")

BILINEAR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gnn", "results", "bilinear_edge",
)

EDGE_TYPES = [
    'PPI', 'COUPLES', 'SYNTHETIC_LETHAL', 'CO_ESSENTIAL',
    'CO_EXPRESSED', 'CO_CNA', 'ATTENDS_TO',
    'HAS_SENSITIVITY_EVIDENCE', 'HAS_RESISTANCE_EVIDENCE',
    'CO_SENSITIVE', 'CO_RESISTANT', 'DRUG_CONFLICT',
    'CO_TISSUE_EXPR', 'CO_BIALLELIC',
    'ANALOGOUS', 'CONVERGES', 'TRANSPOSES',
    'SAME_STRAND',
]

# CT index → clean name for Neo4j property keys
CT_NAMES = {
    0: "nsclc", 1: "colorectal", 2: "breast", 3: "prostate",
    4: "pancreatic", 5: "endometrial", 6: "ovarian", 7: "bladder",
    8: "melanoma", 9: "hepatobiliary", 10: "esophagogastric",
    11: "sarcoma", 12: "thyroid", 13: "renal", 14: "head_neck",
    15: "gist", 16: "germ_cell", 17: "sclc", 18: "mesothelioma",
    19: "appendiceal", 20: "uterine_sarcoma", 21: "salivary",
    22: "gi_neuroendocrine", 23: "skin_nonmelanoma", 24: "cervical",
    25: "small_bowel", 26: "anal",
}


def main():
    t0 = time.time()

    # Load W matrices and edge matrix
    print("Loading bilinear artifacts...")
    with open(os.path.join(BILINEAR_DIR, "W_matrices.json")) as f:
        W_all = json.load(f)

    edge_matrix = np.load(os.path.join(BILINEAR_DIR, "raw_edge_matrix.npy"))

    with open(os.path.join(BILINEAR_DIR, "gene_vocab.json")) as f:
        gene_vocab = json.load(f)

    idx_to_gene = {v: k for k, v in gene_vocab.items()}
    G = len(gene_vocab)
    n_cts = len(W_all)
    print(f"  {G} genes, {n_cts} cancer types")

    # Precompute symmetrized W matrices
    W_mats = {}
    for ct_key, W_raw in W_all.items():
        W = np.array(W_raw)
        W_mats[ct_key] = (W + W.T) / 2

    # Compute per-edge, per-CT bilinear scores
    print("\nComputing per-edge W scores...")
    edge_scores = {}  # (g1, g2) → {ct_name: score, ...}
    n_edges = 0

    for i in range(G):
        for j in range(i + 1, G):
            e = edge_matrix[i, j, :]
            if e.sum() == 0:
                continue

            g1 = idx_to_gene[i]
            g2 = idx_to_gene[j]

            scores = {}
            for ct_key, W in W_mats.items():
                ct_idx = int(ct_key.split("_")[1])
                ct_name = CT_NAMES.get(ct_idx)
                if ct_name is None:
                    continue
                score = float(e @ W @ e)
                scores[ct_name] = score

            if scores:
                scores["w_max"] = max(abs(v) for v in scores.values())
                scores["w_mean"] = np.mean(list(v for k, v in scores.items()
                                                 if k not in ("w_max", "w_mean")))
                edge_scores[(g1, g2)] = scores
                n_edges += 1

    print(f"  {n_edges:,} gene pairs with W scores")

    # Write to Neo4j
    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()

    # Discover gene-gene edge types
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Gene)-[r]->(b:Gene)
            WHERE r.deprecated IS NULL OR r.deprecated = false
            WITH DISTINCT type(r) AS rtype
            RETURN rtype ORDER BY rtype
        """)
        neo4j_edge_types = [r["rtype"] for r in result]

    print(f"  Edge types in Neo4j: {neo4j_edge_types}")

    # Batch update: for each edge type, update all edges
    total_updated = 0
    for etype in neo4j_edge_types:
        print(f"\n  Processing {etype}...")

        with driver.session() as session:
            result = session.run(f"""
                MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                RETURN a.name AS ga, b.name AS gb, elementId(r) AS rid
            """)
            edges = list(result)

        updates = []
        for record in edges:
            ga, gb = record["ga"], record["gb"]
            # Look up in both directions
            scores = edge_scores.get((ga, gb)) or edge_scores.get((gb, ga))
            if scores is None:
                continue

            props = {}
            for ct_name, score in scores.items():
                if ct_name in ("w_max", "w_mean"):
                    props[ct_name] = score
                else:
                    props[f"w_{ct_name}"] = score
            props["w_provenance"] = "bilinear_edge_model_v1"

            updates.append({"rid": record["rid"], "props": props})

        if updates:
            # Batch in chunks
            chunk_size = 1000
            for start in range(0, len(updates), chunk_size):
                chunk = updates[start:start + chunk_size]
                with driver.session() as session:
                    session.run("""
                        UNWIND $updates AS u
                        MATCH ()-[r]->()
                        WHERE elementId(r) = u.rid
                        SET r += u.props
                    """, updates=chunk)

            total_updated += len(updates)
            print(f"    Updated {len(updates)} edges")
        else:
            print(f"    No matching edges")

    driver.close()

    print(f"\n{'='*60}")
    print(f"  Total edges updated: {total_updated:,}")
    print(f"  Properties added per edge: w_<ct>, w_max, w_mean, w_provenance")
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
