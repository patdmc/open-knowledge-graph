"""
Build the Language Graph semantic frames — Lang_Frame nodes and EVOKES edges.

Frames are the channel equivalent in the language graph. A frame is a structured
situation type (e.g., "Causation", "Commerce_buy") that groups lexemes by the
roles they play. Like how genes belong to channels (DDR, PI3K), words belong
to frames (Causation, Motion, Judgment).

Currently sourced from FrameNet 1.7.

Creates:
  - ~1,200 Lang_Frame nodes (structured situation types)
  - ~13,000 EVOKES edges (Lang_Lexeme → Lang_Frame)
  - Frame-to-frame hierarchy edges (inheritance, subframe, perspective)

Requires build_lexicon.py to have run first (needs Lang_Lexeme nodes).

Usage:
    python -m language_graph.data.build_frames [--dry-run] [--limit N]
"""

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

# NLTK data path
_NLTK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "nltk_data")
if os.path.exists(_NLTK_DIR):
    import nltk
    nltk.data.path.insert(0, _NLTK_DIR)

from nltk.corpus import framenet as fn


# ---------------------------------------------------------------------------
# POS mapping (FrameNet POS → our canonical POS)
# ---------------------------------------------------------------------------

FN_POS_MAP = {
    "N": "noun",
    "V": "verb",
    "A": "adjective",
    "ADV": "adverb",
    "PREP": "preposition",
    "C": "conjunction",
    "PRON": "pronoun",
    "INTJ": "interjection",
    "ART": "determiner",
    "SCON": "conjunction",
    "CCON": "conjunction",
    "NUM": "noun",
    "AVP": "adverb",
    "IDIO": "noun",  # idioms
}


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(limit=None):
    """Extract Lang_Frame nodes from FrameNet."""
    nodes = []
    provenance = {
        "created_at": datetime.utcnow().isoformat(),
        "created_by": "framenet_ingest.py",
        "source": "framenet_1.7",
        "confidence": 1.0,
    }

    for i, frame in enumerate(fn.frames()):
        if limit and i >= limit:
            break

        roles = [fe.name for fe in frame.FE.values()]
        core_roles = [fe.name for fe in frame.FE.values()
                      if fe.coreType == "Core"]

        node = {
            "id": f"language:frame:{frame.name}",
            "graph": "language",
            "type": "frame",
            "name": frame.name,
            "frame_name": frame.name,
            "definition": frame.definition[:500] if frame.definition else "",
            "n_roles": len(roles),
            "n_core_roles": len(core_roles),
            "roles": "|".join(roles),  # Neo4j doesn't do lists cleanly, pipe-delimited
            "core_roles": "|".join(core_roles),
            "n_lexical_units": len(frame.lexUnit),
            **provenance,
        }
        nodes.append(node)

    return nodes


# ---------------------------------------------------------------------------
# EVOKES edge extraction
# ---------------------------------------------------------------------------

def extract_evokes_edges(limit=None):
    """
    Extract EVOKES edges: Lang_Lexeme → Lang_Frame.

    FrameNet lexical units are word.POS pairs that evoke a frame.
    We match these to WordNet Lang_Lexeme nodes by lemma + POS.
    If no exact WordNet match, we still create the edge pointing to
    a lexeme ID (the node may be created later or by another source).
    """
    edges = []
    provenance = {
        "source": "framenet_1.7",
        "evidence_type": "structured_database",
        "created_at": datetime.utcnow().isoformat(),
        "created_by": "framenet_ingest.py",
        "confidence": 1.0,
    }

    # Pre-build WordNet lemma lookup for fast matching
    from nltk.corpus import wordnet as wn

    def best_wordnet_sense(lemma_str, pos_char):
        """Find the most frequent WordNet sense for this lemma+POS."""
        wn_pos = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}.get(pos_char)
        if not wn_pos:
            return None
        synsets = wn.synsets(lemma_str, pos=wn_pos)
        if not synsets:
            return None
        # Take the most frequent sense (first in WordNet ordering)
        best = synsets[0].lemmas()[0]
        for lem in synsets[0].lemmas():
            if lem.name().lower() == lemma_str.lower():
                best = lem
                break
        syn = best.synset()
        pos_name = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}.get(syn.pos(), syn.pos())
        return f"{best.name()}.{pos_name}.{syn.offset():08d}"

    edge_count = 0
    unmatched = 0

    for frame in fn.frames():
        for lu_name, lu in frame.lexUnit.items():
            # lu_name format: "word.pos" e.g. "buy.v", "cost.n"
            parts = lu_name.rsplit(".", 1)
            if len(parts) != 2:
                continue
            lemma_str, fn_pos = parts
            lemma_str = lemma_str.replace(" ", "_")
            our_pos = FN_POS_MAP.get(fn_pos.upper(), "noun")

            # Try to match to a WordNet sense
            wn_sense = best_wordnet_sense(lemma_str, our_pos)

            if wn_sense:
                source_id = f"language:lexeme:{wn_sense}"
            else:
                # Create a framenet-sourced ID — node may not exist yet
                source_id = f"language:lexeme:{lemma_str}.{our_pos}.fn"
                unmatched += 1

            # Determine which frame role this LU typically fills
            role = "lexical_unit"  # default
            # FrameNet LUs don't specify role directly; they evoke the whole frame

            edges.append({
                "source_id": source_id,
                "target_id": f"language:frame:{frame.name}",
                "role": role,
                "lu_name": lu_name,
                "sign": +1,
                "weight": 1.0,
                **provenance,
            })
            edge_count += 1

            if limit and edge_count >= limit:
                print(f"  ({unmatched} LUs had no WordNet match)")
                return edges

    print(f"  ({unmatched} LUs had no WordNet match — will create placeholder lexemes)")
    return edges


# ---------------------------------------------------------------------------
# Frame-to-frame relations
# ---------------------------------------------------------------------------

def extract_frame_relations():
    """Extract frame hierarchy edges from FrameNet."""
    edges = []
    provenance = {
        "source": "framenet_1.7",
        "evidence_type": "structured_database",
        "created_at": datetime.utcnow().isoformat(),
        "created_by": "framenet_ingest.py",
        "confidence": 1.0,
    }

    for rel in fn.frame_relations():
        rel_type = rel.type.name if hasattr(rel.type, 'name') else str(rel.type)

        # Map FrameNet relation types to our vocabulary
        if rel_type in ("Inheritance", "Is Inherited by"):
            edge_type = "FRAME_INHERITS"
            sign = +1
        elif rel_type in ("Subframe", "Has Subframe(s)"):
            edge_type = "FRAME_CONTAINS"
            sign = +1
        elif rel_type in ("Using", "Is Used by"):
            edge_type = "FRAME_USES"
            sign = +1
        elif rel_type in ("Perspective_on", "Is Perspectivized in"):
            edge_type = "FRAME_PERSPECTIVE"
            sign = 0
        elif rel_type in ("Precedes", "Is Preceded by"):
            edge_type = "FRAME_PRECEDES"
            sign = +1
        elif rel_type in ("Causative_of", "Is Causative of"):
            edge_type = "FRAME_CAUSES"
            sign = +1
        elif rel_type in ("Inchoative_of", "Is Inchoative of"):
            edge_type = "FRAME_INCHOATIVE"
            sign = +1
        else:
            edge_type = "FRAME_RELATED"
            sign = 0

        # rel has superFrame and subFrame (or parent/child depending on type)
        if hasattr(rel, 'superFrame') and hasattr(rel, 'subFrame'):
            edges.append({
                "source_id": f"language:frame:{rel.subFrame.name}",
                "target_id": f"language:frame:{rel.superFrame.name}",
                "edge_type": edge_type,
                "fn_relation": rel_type,
                "sign": sign,
                "weight": 1.0,
                **provenance,
            })

    return edges


# ---------------------------------------------------------------------------
# Neo4j commit
# ---------------------------------------------------------------------------

def commit_to_neo4j(frame_nodes, evokes_edges, frame_rel_edges, dry_run=False, batch_size=500):
    """Commit frames, EVOKES edges, and frame relations to Neo4j."""
    if dry_run:
        print(f"\n[DRY RUN] Would commit:")
        print(f"  {len(frame_nodes):,} Lang_Frame nodes")
        print(f"  {len(evokes_edges):,} EVOKES edges")
        print(f"  {len(frame_rel_edges):,} frame relation edges")
        return

    from neo4j import GraphDatabase
    from language_graph.config import NEO4J_URI, NEO4J_AUTH

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as session:
        # Constraints
        print("Creating constraints ...")
        session.run(
            "CREATE CONSTRAINT lang_frame_id IF NOT EXISTS "
            "FOR (n:Lang_Frame) REQUIRE n.id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX lang_frame_name IF NOT EXISTS "
            "FOR (n:Lang_Frame) ON (n.frame_name)"
        )

        # Frame nodes
        print(f"Committing {len(frame_nodes):,} Lang_Frame nodes ...")
        t0 = time.time()
        for i in range(0, len(frame_nodes), batch_size):
            batch = frame_nodes[i:i + batch_size]
            session.run(
                """
                UNWIND $batch AS props
                MERGE (n:Lang_Frame {id: props.id})
                SET n += props
                """,
                batch=batch,
            )
        print(f"  Done in {time.time() - t0:.1f}s")

        # EVOKES edges — need to handle both existing and missing lexeme nodes
        print(f"Committing {len(evokes_edges):,} EVOKES edges ...")
        t0 = time.time()
        for i in range(0, len(evokes_edges), batch_size):
            batch = evokes_edges[i:i + batch_size]
            # Use MERGE on lexeme to create placeholder if needed
            session.run(
                """
                UNWIND $batch AS e
                MERGE (a:Lang_Lexeme {id: e.source_id})
                ON CREATE SET a.graph = 'language', a.type = 'lexeme',
                              a.source = 'framenet_1.7', a.placeholder = true
                WITH a, e
                MATCH (b:Lang_Frame {id: e.target_id})
                MERGE (a)-[r:EVOKES]->(b)
                SET r += {
                    role: e.role, lu_name: e.lu_name,
                    sign: e.sign, weight: e.weight,
                    source: e.source, evidence_type: e.evidence_type,
                    created_at: e.created_at, created_by: e.created_by,
                    confidence: e.confidence
                }
                """,
                batch=batch,
            )
            if (i // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i + len(batch):,} / {len(evokes_edges):,} ({elapsed:.1f}s)")
        print(f"  Done in {time.time() - t0:.1f}s")

        # Frame relation edges
        if frame_rel_edges:
            print(f"Committing {len(frame_rel_edges):,} frame relation edges ...")
            t0 = time.time()
            # Group by edge_type for separate Cypher queries
            by_type = defaultdict(list)
            for e in frame_rel_edges:
                by_type[e["edge_type"]].append(e)

            for etype, elist in by_type.items():
                for i in range(0, len(elist), batch_size):
                    batch = elist[i:i + batch_size]
                    # Dynamic rel type via APOC or just use a generic one
                    # For simplicity, use FRAME_RELATED with a subtype property
                    session.run(
                        """
                        UNWIND $batch AS e
                        MATCH (a:Lang_Frame {id: e.source_id})
                        MATCH (b:Lang_Frame {id: e.target_id})
                        MERGE (a)-[r:FRAME_RELATED]->(b)
                        SET r += {
                            relation_type: e.edge_type,
                            fn_relation: e.fn_relation,
                            sign: e.sign, weight: e.weight,
                            source: e.source, evidence_type: e.evidence_type,
                            created_at: e.created_at, created_by: e.created_by,
                            confidence: e.confidence
                        }
                        """,
                        batch=batch,
                    )
                print(f"  {etype}: {len(elist):,}")
            print(f"  Done in {time.time() - t0:.1f}s")

    driver.close()
    print("\nAll committed to Neo4j.")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_neo4j():
    """Check what FrameNet data is in Neo4j."""
    from neo4j import GraphDatabase
    from language_graph.config import NEO4J_URI, NEO4J_AUTH

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as session:
        n_frames = session.run("MATCH (n:Lang_Frame) RETURN count(n) AS c").single()["c"]
        n_evokes = session.run("MATCH ()-[r:EVOKES]->() RETURN count(r) AS c").single()["c"]
        n_frel = session.run("MATCH ()-[r:FRAME_RELATED]->() RETURN count(r) AS c").single()["c"]

        print(f"\nFrameNet in Neo4j:")
        print(f"  Lang_Frame nodes: {n_frames:,}")
        print(f"  EVOKES edges:     {n_evokes:,}")
        print(f"  FRAME_RELATED:    {n_frel:,}")

        # Sample: Causation frame
        result = session.run("""
            MATCH (f:Lang_Frame {frame_name: 'Causation'})
            OPTIONAL MATCH (l:Lang_Lexeme)-[r:EVOKES]->(f)
            RETURN f.frame_name, f.n_roles, f.core_roles,
                   collect(r.lu_name)[..5] AS sample_lus
        """)
        for r in result:
            print(f"\n  Sample — Causation frame:")
            print(f"    Roles: {r['f.core_roles']}")
            print(f"    LUs: {r['sample_lus']}")

    driver.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest FrameNet into Neo4j")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_neo4j()
        return

    print("=== FrameNet → Language Graph Ingestion ===\n")

    print("Extracting Frame nodes ...")
    frames = extract_frames(limit=args.limit)
    print(f"  {len(frames):,} frames")

    print("\nExtracting EVOKES edges ...")
    evokes = extract_evokes_edges(limit=args.limit)
    print(f"  {len(evokes):,} edges")

    print("\nExtracting frame relations ...")
    frame_rels = extract_frame_relations()
    print(f"  {len(frame_rels):,} frame-frame edges")

    commit_to_neo4j(frames, evokes, frame_rels, dry_run=args.dry_run)

    if not args.dry_run:
        verify_neo4j()


if __name__ == "__main__":
    main()
