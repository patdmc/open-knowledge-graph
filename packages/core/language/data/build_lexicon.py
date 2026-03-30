"""
Build the Language Graph lexicon — Lang_Lexeme nodes and semantic edges.

The lexicon is the core vocabulary layer of the language graph. Each node is a
word-sense pair (like a gene in the cancer graph). Semantic edges encode the
structural relationships between word meanings.

Currently sourced from WordNet 3.1. ConceptNet enrichment is planned.

Creates:
  - ~207K Lang_Lexeme nodes (one per word-sense, keyed by sense_id)
  - SYNONYMOUS edges (within synsets — words that share meaning)
  - ANTONYMOUS edges (opposition — the SL_PARTNER equivalent)
  - HYPERNYM_OF edges (IS-A hierarchy — the structural backbone)
  - PART_OF edges (meronymy — part/substance/member)
  - ENTAILS edges (verb entailment — the ENABLES equivalent)

All edges carry provenance per D11 schema.
All nodes follow D11 universal schema: id="language:lexeme:{sense_id}".

Usage:
    python -m language_graph.data.build_lexicon [--dry-run] [--limit N]
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# NLTK data path — use project-local cache
_NLTK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "nltk_data")
if os.path.exists(_NLTK_DIR):
    import nltk
    nltk.data.path.insert(0, _NLTK_DIR)

from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------------
# POS mapping
# ---------------------------------------------------------------------------

WN_POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",  # satellite adjective → adjective
    "r": "adverb",
}


def sense_id(lemma):
    """Stable sense identifier: lemma.pos.synset_offset"""
    syn = lemma.synset()
    return f"{lemma.name()}.{WN_POS_MAP.get(syn.pos(), syn.pos())}.{syn.offset():08d}"


def synset_representative(syn):
    """The first lemma of a synset — used as the canonical representative."""
    return syn.lemmas()[0]


# ---------------------------------------------------------------------------
# Node extraction
# ---------------------------------------------------------------------------

def extract_lexemes(limit=None):
    """
    Extract Lang_Lexeme nodes from WordNet.

    Each lemma-in-synset becomes one node. The sense_id is the unique key.
    Returns list of dicts ready for Neo4j.
    """
    nodes = []
    seen = set()
    count = 0

    for syn in wn.all_synsets():
        definition = syn.definition()
        pos = WN_POS_MAP.get(syn.pos(), syn.pos())

        for lemma in syn.lemmas():
            sid = sense_id(lemma)
            if sid in seen:
                continue
            seen.add(sid)

            node = {
                "id": f"language:lexeme:{sid}",
                "graph": "language",
                "type": "lexeme",
                "name": lemma.name().replace("_", " "),
                "lemma": lemma.name().replace("_", " "),
                "pos": pos,
                "sense_id": sid,
                "definition": definition,
                "synset_id": syn.name(),
                "frequency": lemma.count() or 0,
                "polysemy_count": len(wn.synsets(lemma.name(), pos=syn.pos())),
                "created_at": datetime.utcnow().isoformat(),
                "created_by": "wordnet_ingest.py",
                "source": "wordnet_3.1",
                "source_detail": f"synset={syn.name()}",
                "confidence": 1.0,
            }
            nodes.append(node)
            count += 1
            if limit and count >= limit:
                return nodes

    return nodes


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def extract_edges(limit=None):
    """
    Extract all semantic edges from WordNet.

    Returns dict of {edge_type: [edge_dicts]}.
    """
    edges = defaultdict(list)
    edge_count = 0
    provenance = {
        "source": "wordnet_3.1",
        "evidence_type": "structured_database",
        "created_at": datetime.utcnow().isoformat(),
        "created_by": "wordnet_ingest.py",
        "confidence": 1.0,
    }

    for syn in wn.all_synsets():
        lemmas = syn.lemmas()

        # --- SYNONYMOUS: all pairs within a synset ---
        # Only connect each lemma to the canonical (first) to avoid O(n^2)
        if len(lemmas) > 1:
            canonical = lemmas[0]
            for other in lemmas[1:]:
                edges["SYNONYMOUS"].append({
                    "source_id": f"language:lexeme:{sense_id(canonical)}",
                    "target_id": f"language:lexeme:{sense_id(other)}",
                    "substitutability": 1.0,  # within same synset = perfect synonyms
                    "sign": +1,
                    "weight": 1.0,
                    **provenance,
                })
                edge_count += 1

        # --- ANTONYMOUS: lemma-level antonyms ---
        for lemma in lemmas:
            for ant in lemma.antonyms():
                edges["ANTONYMOUS"].append({
                    "source_id": f"language:lexeme:{sense_id(lemma)}",
                    "target_id": f"language:lexeme:{sense_id(ant)}",
                    "opposition_type": "gradable",  # default; could refine
                    "opposition_strength": 1.0,
                    "sign": -1,
                    "weight": 1.0,
                    **provenance,
                })
                edge_count += 1

        # --- HYPERNYM_OF: IS-A hierarchy ---
        # Connect canonical lemma of child → canonical lemma of parent
        for hyper_syn in syn.hypernyms():
            child_lemma = synset_representative(syn)
            parent_lemma = synset_representative(hyper_syn)
            edges["HYPERNYM_OF"].append({
                "source_id": f"language:lexeme:{sense_id(child_lemma)}",
                "target_id": f"language:lexeme:{sense_id(parent_lemma)}",
                "sign": +1,
                "weight": 1.0,
                **provenance,
            })
            edge_count += 1

        # Instance hypernyms (e.g., Einstein IS-A physicist)
        for hyper_syn in syn.instance_hypernyms():
            child_lemma = synset_representative(syn)
            parent_lemma = synset_representative(hyper_syn)
            edges["HYPERNYM_OF"].append({
                "source_id": f"language:lexeme:{sense_id(child_lemma)}",
                "target_id": f"language:lexeme:{sense_id(parent_lemma)}",
                "sign": +1,
                "weight": 0.9,
                "source_detail": "instance_hypernym",
                **provenance,
            })
            edge_count += 1

        # --- PART_OF: meronymy ---
        for mero_syn in syn.part_meronyms():
            part_lemma = synset_representative(mero_syn)
            whole_lemma = synset_representative(syn)
            edges["PART_OF"].append({
                "source_id": f"language:lexeme:{sense_id(part_lemma)}",
                "target_id": f"language:lexeme:{sense_id(whole_lemma)}",
                "necessity": 0.5,  # default
                "meronym_type": "part",
                "sign": +1,
                "weight": 1.0,
                **provenance,
            })
            edge_count += 1

        for mero_syn in syn.substance_meronyms():
            part_lemma = synset_representative(mero_syn)
            whole_lemma = synset_representative(syn)
            edges["PART_OF"].append({
                "source_id": f"language:lexeme:{sense_id(part_lemma)}",
                "target_id": f"language:lexeme:{sense_id(whole_lemma)}",
                "necessity": 0.7,
                "meronym_type": "substance",
                "sign": +1,
                "weight": 1.0,
                **provenance,
            })
            edge_count += 1

        for mero_syn in syn.member_meronyms():
            part_lemma = synset_representative(mero_syn)
            whole_lemma = synset_representative(syn)
            edges["PART_OF"].append({
                "source_id": f"language:lexeme:{sense_id(part_lemma)}",
                "target_id": f"language:lexeme:{sense_id(whole_lemma)}",
                "necessity": 0.3,
                "meronym_type": "member",
                "sign": +1,
                "weight": 0.8,
                **provenance,
            })
            edge_count += 1

        # --- ENTAILS: verb entailment ---
        for ent_syn in syn.entailments():
            src_lemma = synset_representative(syn)
            tgt_lemma = synset_representative(ent_syn)
            edges["ENTAILS"].append({
                "source_id": f"language:lexeme:{sense_id(src_lemma)}",
                "target_id": f"language:lexeme:{sense_id(tgt_lemma)}",
                "sign": +1,
                "weight": 1.0,
                **provenance,
            })
            edge_count += 1

        if limit and edge_count >= limit:
            break

    return dict(edges)


# ---------------------------------------------------------------------------
# Neo4j commit
# ---------------------------------------------------------------------------

def commit_to_neo4j(nodes, edges, dry_run=False, batch_size=1000):
    """
    Commit nodes and edges to Neo4j.

    Uses MERGE to be idempotent — safe to re-run.
    """
    if dry_run:
        print(f"\n[DRY RUN] Would commit {len(nodes):,} nodes")
        for etype, elist in edges.items():
            print(f"  {etype}: {len(elist):,} edges")
        return

    from neo4j import GraphDatabase
    from language_graph.config import NEO4J_URI, NEO4J_AUTH

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as session:
        # Create constraints for uniqueness
        print("Creating constraints ...")
        session.run(
            "CREATE CONSTRAINT lang_lexeme_id IF NOT EXISTS "
            "FOR (n:Lang_Lexeme) REQUIRE n.id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX lang_lexeme_sense IF NOT EXISTS "
            "FOR (n:Lang_Lexeme) ON (n.sense_id)"
        )
        session.run(
            "CREATE INDEX lang_lexeme_lemma IF NOT EXISTS "
            "FOR (n:Lang_Lexeme) ON (n.lemma)"
        )

        # Batch-insert nodes
        print(f"Committing {len(nodes):,} Lang_Lexeme nodes ...")
        t0 = time.time()
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            session.run(
                """
                UNWIND $batch AS props
                MERGE (n:Lang_Lexeme {id: props.id})
                SET n += props
                """,
                batch=batch,
            )
            if (i // batch_size) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {i + len(batch):,} / {len(nodes):,} ({elapsed:.1f}s)")

        print(f"  Nodes done in {time.time() - t0:.1f}s")

        # Batch-insert edges
        edge_queries = {
            "SYNONYMOUS": """
                UNWIND $batch AS e
                MATCH (a:Lang_Lexeme {id: e.source_id})
                MATCH (b:Lang_Lexeme {id: e.target_id})
                MERGE (a)-[r:SYNONYMOUS]-(b)
                SET r += {
                    substitutability: e.substitutability,
                    sign: e.sign, weight: e.weight,
                    source: e.source, evidence_type: e.evidence_type,
                    created_at: e.created_at, created_by: e.created_by,
                    confidence: e.confidence
                }
            """,
            "ANTONYMOUS": """
                UNWIND $batch AS e
                MATCH (a:Lang_Lexeme {id: e.source_id})
                MATCH (b:Lang_Lexeme {id: e.target_id})
                MERGE (a)-[r:ANTONYMOUS]-(b)
                SET r += {
                    opposition_type: e.opposition_type,
                    opposition_strength: e.opposition_strength,
                    sign: e.sign, weight: e.weight,
                    source: e.source, evidence_type: e.evidence_type,
                    created_at: e.created_at, created_by: e.created_by,
                    confidence: e.confidence
                }
            """,
            "HYPERNYM_OF": """
                UNWIND $batch AS e
                MATCH (a:Lang_Lexeme {id: e.source_id})
                MATCH (b:Lang_Lexeme {id: e.target_id})
                MERGE (a)-[r:HYPERNYM_OF]->(b)
                SET r += {
                    sign: e.sign, weight: e.weight,
                    source: e.source, evidence_type: e.evidence_type,
                    created_at: e.created_at, created_by: e.created_by,
                    confidence: e.confidence
                }
            """,
            "PART_OF": """
                UNWIND $batch AS e
                MATCH (a:Lang_Lexeme {id: e.source_id})
                MATCH (b:Lang_Lexeme {id: e.target_id})
                MERGE (a)-[r:LANG_PART_OF]->(b)
                SET r += {
                    necessity: e.necessity,
                    meronym_type: e.meronym_type,
                    sign: e.sign, weight: e.weight,
                    source: e.source, evidence_type: e.evidence_type,
                    created_at: e.created_at, created_by: e.created_by,
                    confidence: e.confidence
                }
            """,
            "ENTAILS": """
                UNWIND $batch AS e
                MATCH (a:Lang_Lexeme {id: e.source_id})
                MATCH (b:Lang_Lexeme {id: e.target_id})
                MERGE (a)-[r:ENTAILS]->(b)
                SET r += {
                    sign: e.sign, weight: e.weight,
                    source: e.source, evidence_type: e.evidence_type,
                    created_at: e.created_at, created_by: e.created_by,
                    confidence: e.confidence
                }
            """,
        }

        for etype, elist in edges.items():
            query = edge_queries.get(etype)
            if not query:
                print(f"  [SKIP] No query for edge type: {etype}")
                continue

            print(f"Committing {len(elist):,} {etype} edges ...")
            t0 = time.time()
            for i in range(0, len(elist), batch_size):
                batch = elist[i:i + batch_size]
                session.run(query, batch=batch)
                if (i // batch_size) % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  {i + len(batch):,} / {len(elist):,} ({elapsed:.1f}s)")

            print(f"  {etype} done in {time.time() - t0:.1f}s")

    driver.close()
    print("\nAll committed to Neo4j.")


# ---------------------------------------------------------------------------
# Summary / verification
# ---------------------------------------------------------------------------

def verify_neo4j():
    """Quick verification of what's in Neo4j."""
    from neo4j import GraphDatabase
    from language_graph.config import NEO4J_URI, NEO4J_AUTH

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as session:
        n_nodes = session.run(
            "MATCH (n:Lang_Lexeme) RETURN count(n) AS c"
        ).single()["c"]
        print(f"\nLang_Lexeme nodes in Neo4j: {n_nodes:,}")

        for rel_type in ["SYNONYMOUS", "ANTONYMOUS", "HYPERNYM_OF", "LANG_PART_OF", "ENTAILS"]:
            n_edges = session.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c"
            ).single()["c"]
            print(f"  {rel_type}: {n_edges:,}")

        # Sample: show the hypernym chain for "dog"
        result = session.run("""
            MATCH (n:Lang_Lexeme)
            WHERE n.lemma = 'dog' AND n.pos = 'noun'
            RETURN n.sense_id, n.definition
            LIMIT 3
        """)
        print("\nSample — 'dog' senses:")
        for r in result:
            print(f"  {r['n.sense_id']}: {r['n.definition']}")

    driver.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest WordNet into Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't commit")
    parser.add_argument("--limit", type=int, default=None, help="Limit nodes/edges for testing")
    parser.add_argument("--verify-only", action="store_true", help="Just check what's in Neo4j")
    args = parser.parse_args()

    if args.verify_only:
        verify_neo4j()
        return

    print("=== WordNet → Language Graph Ingestion ===\n")

    print("Extracting lexeme nodes ...")
    t0 = time.time()
    nodes = extract_lexemes(limit=args.limit)
    print(f"  {len(nodes):,} nodes extracted in {time.time() - t0:.1f}s")

    print("\nExtracting semantic edges ...")
    t0 = time.time()
    edges = extract_edges(limit=args.limit)
    total_edges = sum(len(v) for v in edges.values())
    print(f"  {total_edges:,} edges extracted in {time.time() - t0:.1f}s")
    for etype, elist in sorted(edges.items()):
        print(f"    {etype}: {len(elist):,}")

    print()
    commit_to_neo4j(nodes, edges, dry_run=args.dry_run)

    if not args.dry_run:
        verify_neo4j()


if __name__ == "__main__":
    main()
