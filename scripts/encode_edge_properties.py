#!/usr/bin/env python3
"""
Add numeric one-hot encodings for string edge properties in Neo4j.
Original string properties are preserved (graph durability rule).
"""

import json
import os

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "openknowledgegraph")

STRAND_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gnn", "results", "gene_strand_data.json",
)


COUPLES_QUERIES = [
    (
        "pair_type_gof_lof",
        """
        MATCH ()-[r:COUPLES]->()
        SET r.pair_type_gof_lof = CASE WHEN r.pair_type = 'GOF_LOF' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
    (
        "pair_type_gof_gof",
        """
        MATCH ()-[r:COUPLES]->()
        SET r.pair_type_gof_gof = CASE WHEN r.pair_type = 'GOF_GOF' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
    (
        "pair_type_lof_lof",
        """
        MATCH ()-[r:COUPLES]->()
        SET r.pair_type_lof_lof = CASE WHEN r.pair_type = 'LOF_LOF' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
    (
        "is_cross_channel",
        """
        MATCH ()-[r:COUPLES]->()
        SET r.is_cross_channel = CASE WHEN r.locality = 'cross' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
]

SL_QUERIES = [
    (
        "evidence_clinical",
        """
        MATCH ()-[r:SYNTHETIC_LETHAL]->()
        SET r.evidence_clinical = CASE WHEN r.evidence_type = 'clinical' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
    (
        "evidence_experimental",
        """
        MATCH ()-[r:SYNTHETIC_LETHAL]->()
        SET r.evidence_experimental = CASE WHEN r.evidence_type = 'experimental' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
    (
        "evidence_computational",
        """
        MATCH ()-[r:SYNTHETIC_LETHAL]->()
        SET r.evidence_computational = CASE WHEN r.evidence_type = 'computational' THEN 1.0 ELSE 0.0 END
        RETURN count(r) AS updated
        """,
    ),
]


def encode_strand_attributes(driver):
    """Set strand as Gene node property and same_strand on all gene-gene edges."""
    if not os.path.exists(STRAND_DATA_PATH):
        print("  Strand data not found, skipping")
        return

    with open(STRAND_DATA_PATH) as f:
        strand_data = json.load(f)

    # Set strand on Gene nodes
    print("=== Gene node strand ===")
    with driver.session() as session:
        for gene, info in strand_data.items():
            session.run(
                "MATCH (g:Gene {name: $name}) SET g.strand = $strand",
                name=gene, strand=float(info['strand']),
            )
        print(f"  Set strand on {len(strand_data)} genes")

    # Discover all gene-gene edge types
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Gene)-[r]->(b:Gene)
            WHERE r.deprecated IS NULL OR r.deprecated = false
            WITH DISTINCT type(r) AS rtype
            RETURN rtype ORDER BY rtype
        """)
        edge_types = [r['rtype'] for r in result]

    # Build strand lookup
    strand_lookup = {g: info['strand'] for g, info in strand_data.items()}

    # Set same_strand on all gene-gene edges
    print("\n=== same_strand edge attribute ===")
    for etype in edge_types:
        with driver.session() as session:
            result = session.run(f"""
                MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                RETURN a.name AS ga, b.name AS gb, elementId(r) AS rid
            """)
            edges = list(result)

        same = opposite = unknown = 0
        updates = []
        for record in edges:
            ga, gb = record['ga'], record['gb']
            sa = strand_lookup.get(ga)
            sb = strand_lookup.get(gb)
            if sa is not None and sb is not None:
                val = 1.0 if sa == sb else 0.0
                updates.append((record['rid'], val))
                if val == 1.0:
                    same += 1
                else:
                    opposite += 1
            else:
                unknown += 1

        # Batch update
        if updates:
            with driver.session() as session:
                session.run(
                    """
                    UNWIND $updates AS u
                    MATCH (a:Gene)-[r]->(b:Gene)
                    WHERE elementId(r) = u.rid
                    SET r.same_strand = u.val
                    """,
                    updates=[{'rid': rid, 'val': val} for rid, val in updates],
                )

        total = same + opposite + unknown
        if total > 0:
            print(f"  {etype:25s}: {total:6d} edges  "
                  f"same={same} opposite={opposite} unknown={unknown}")


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Connected to Neo4j\n")

    print("=== COUPLES edges ===")
    for name, query in COUPLES_QUERIES:
        with driver.session() as session:
            result = session.run(query)
            record = result.single()
            print(f"  {name}: {record['updated']} edges updated")

    print("\n=== SYNTHETIC_LETHAL edges ===")
    for name, query in SL_QUERIES:
        with driver.session() as session:
            result = session.run(query)
            record = result.single()
            print(f"  {name}: {record['updated']} edges updated")

    print()
    encode_strand_attributes(driver)

    driver.close()
    print("\nDone. Original string properties preserved.")


if __name__ == "__main__":
    main()
