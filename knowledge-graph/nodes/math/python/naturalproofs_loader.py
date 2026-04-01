#!/usr/bin/env python3
"""
Load NaturalProofs dataset into lexical skeleton format for the knowledge graph.

Stores assertion skeletons (id, type, depends_on, categories) with pointers
back to source. The full text lives in the source dataset — we store the
graph structure only.

Usage:
    from naturalproofs_loader import NaturalProofsLoader
    loader = NaturalProofsLoader("knowledge-graph/sources/naturalproofs/naturalproofs_proofwiki.json")
    loader.load()
    loader.stats()

    # Get skeleton for a theorem
    sk = loader.skeleton(theorem_id)

    # Get all edges (dependency pairs)
    edges = loader.all_edges()

    # Export to edge discovery format
    nodes = loader.to_discovery_nodes()

    # Run edge discovery on the full corpus
    from edge_discovery import EdgeDiscovery
    ed = EdgeDiscovery()
    ed.nodes.update(nodes)
    ed.discover()
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


def _extract_wiki_links(text):
    """Extract [[Target|Display]] or [[Target]] links from wiki markup."""
    return re.findall(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', text)


def _clean_statement(contents):
    """Extract a clean one-line statement from contents list."""
    text = " ".join(contents)
    # Strip wiki markup
    text = re.sub(r'\[\[[^|\]]*\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # Strip display math markers
    text = re.sub(r':?\$\\displaystyle\s*', '$', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:500]  # cap at 500 chars


class NaturalProofsLoader:
    """Load and index NaturalProofs dataset."""

    def __init__(self, json_path):
        self.path = Path(json_path)
        self.source = self.path.stem  # e.g. "naturalproofs_proofwiki"

        # Indexed data
        self.theorems = {}      # id → theorem dict
        self.definitions = {}   # id → definition dict
        self.others = {}        # id → other dict
        self.all_items = {}     # id → any item

        # Graph structure
        self.edges = set()      # (from_id, to_id) dependency pairs
        self.categories = {}    # id → set of categories

        self._loaded = False

    def load(self):
        """Load and index the dataset."""
        with open(self.path) as f:
            raw = json.load(f)

        ds = raw["dataset"]

        for t in ds["theorems"]:
            tid = t["id"]
            self.theorems[tid] = t
            self.all_items[tid] = t
            self.categories[tid] = set(t.get("toplevel_categories", []))

            # Statement-level references
            for ref_id in t.get("ref_ids", []):
                self.edges.add((tid, ref_id))

            # Proof-level references
            for proof in t.get("proofs", []):
                for ref_id in proof.get("ref_ids", []):
                    self.edges.add((tid, ref_id))

        for d in ds["definitions"]:
            did = d["id"]
            self.definitions[did] = d
            self.all_items[did] = d
            self.categories[did] = set(d.get("toplevel_categories", []))
            for ref_id in d.get("ref_ids", []):
                self.edges.add((did, ref_id))

        for o in ds["others"]:
            oid = o["id"]
            self.others[oid] = o
            self.all_items[oid] = o
            self.categories[oid] = set(o.get("toplevel_categories", []))

        self._loaded = True

    def stats(self):
        """Print dataset statistics."""
        print(f"Source: {self.source}")
        print(f"  Theorems:    {len(self.theorems):,}")
        print(f"  Definitions: {len(self.definitions):,}")
        print(f"  Others:      {len(self.others):,}")
        print(f"  Total items: {len(self.all_items):,}")
        print(f"  Total edges: {len(self.edges):,}")

        # Category distribution
        cat_counts = Counter()
        for cats in self.categories.values():
            for c in cats:
                cat_counts[c] += 1
        print(f"\n  Top categories:")
        for cat, count in cat_counts.most_common(15):
            print(f"    {count:6,}  {cat}")

        # Edge density
        items_with_refs = sum(1 for eid in self.all_items
                              if any(a == eid for a, _ in self.edges))
        print(f"\n  Items with outgoing refs: {items_with_refs:,}")
        print(f"  Avg refs per theorem: {len(self.edges) / max(len(self.theorems), 1):.1f}")

    def skeleton(self, item_id):
        """Return lexical skeleton for an item.

        Skeleton = {id, type, title, statement (truncated), categories,
                    ref_ids, n_proofs, source_pointer}
        """
        item = self.all_items.get(item_id)
        if not item:
            return None

        return {
            "id": f"NP-{self.source}-{item_id}",
            "np_id": item_id,
            "type": item.get("type", "unknown"),
            "title": item.get("title", ""),
            "statement": _clean_statement(item.get("contents", [])),
            "categories": sorted(self.categories.get(item_id, set())),
            "ref_ids": sorted(set(
                item.get("ref_ids", []) +
                [r for p in item.get("proofs", []) for r in p.get("ref_ids", [])]
            )),
            "n_proofs": len(item.get("proofs", [])),
            "source": {
                "dataset": self.source,
                "id": item_id,
            },
        }

    def all_edges(self):
        """Return all dependency edges as (from_id, to_id) pairs."""
        return self.edges

    def to_discovery_nodes(self, categories_filter=None):
        """Convert to format compatible with EdgeDiscovery.

        Returns dict of {node_id: node_data} where node_data has the
        fields EdgeDiscovery expects: id, name, type, domain, concepts,
        edges, references.

        Args:
            categories_filter: if set, only include items in these top-level categories
        """
        nodes = {}

        for item_id, item in self.all_items.items():
            cats = self.categories.get(item_id, set())

            if categories_filter:
                if not cats & set(categories_filter):
                    continue

            # Map NaturalProofs categories to our domain labels
            domain = self._primary_domain(cats)

            # Build ref list as concept-style references
            ref_ids = set(item.get("ref_ids", []))
            for p in item.get("proofs", []):
                ref_ids.update(p.get("ref_ids", []))

            nid = f"NP-{item_id}"
            nodes[nid] = {
                "id": nid,
                "name": item.get("title", f"NP-{item_id}"),
                "type": item.get("type", "unknown"),
                "domain": domain,
                "concepts": [{"id": f"NP-{r}"} for r in sorted(ref_ids)],
                "edges": [
                    {"from": nid, "to": f"NP-{r}", "type": "DEPENDS_ON"}
                    for r in sorted(ref_ids)
                ],
                "source": {"dataset": self.source, "id": item_id},
                # Store categories for feature emission
                "_categories": sorted(cats),
            }

        return nodes

    def _primary_domain(self, categories):
        """Map ProofWiki categories to a primary domain label."""
        priority = [
            ("Number Theory", "number_theory"),
            ("Abstract Algebra", "abstract_algebra"),
            ("Group Theory", "group_theory"),
            ("Linear Algebra", "linear_algebra"),
            ("Topology", "topology"),
            ("Complex Analysis", "complex_analysis"),
            ("Analysis", "analysis"),
            ("Calculus", "calculus"),
            ("Geometry", "geometry"),
            ("Combinatorics", "combinatorics"),
            ("Probability Theory", "probability"),
            ("Set Theory", "set_theory"),
            ("Logic", "logic"),
            ("Algebra", "algebra"),
            ("Arithmetic", "arithmetic"),
        ]
        for cat, domain in priority:
            if cat in categories:
                return domain
        return "mathematics"

    # ── CATEGORY-BASED SLICING ─────────────────────────────────

    def items_in_category(self, category):
        """Return all item IDs in a top-level category."""
        return [iid for iid, cats in self.categories.items() if category in cats]

    def subgraph(self, item_ids):
        """Return edges within a subset of items."""
        id_set = set(item_ids)
        return [(a, b) for a, b in self.edges if a in id_set and b in id_set]

    # ── CROSS-CATEGORY EDGE ANALYSIS ──────────────────────────

    def cross_category_edges(self, top_n=20):
        """Find edges that cross top-level category boundaries.
        These are the most interesting for discovery."""
        results = []
        for a, b in self.edges:
            cats_a = self.categories.get(a, set())
            cats_b = self.categories.get(b, set())
            if cats_a and cats_b:
                shared = cats_a & cats_b
                disjoint_a = cats_a - cats_b
                disjoint_b = cats_b - cats_a
                if disjoint_a and disjoint_b:
                    results.append({
                        "from": a,
                        "to": b,
                        "from_title": self.all_items.get(a, {}).get("title", "?"),
                        "to_title": self.all_items.get(b, {}).get("title", "?"),
                        "shared_cats": sorted(shared),
                        "unique_from": sorted(disjoint_a),
                        "unique_to": sorted(disjoint_b),
                    })

        # Sort by number of disjoint categories (more disjoint = more interesting)
        results.sort(key=lambda r: len(r["unique_from"]) + len(r["unique_to"]), reverse=True)
        return results[:top_n]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "knowledge-graph/sources/naturalproofs/naturalproofs_proofwiki.json"
    loader = NaturalProofsLoader(path)
    loader.load()
    loader.stats()

    print("\n" + "=" * 60)
    print("CROSS-CATEGORY EDGES (most interesting):")
    print("=" * 60)
    for e in loader.cross_category_edges(top_n=15):
        print(f"\n  {e['from_title'][:50]}")
        print(f"    → {e['to_title'][:50]}")
        print(f"    crosses: {e['unique_from'][:3]} ↔ {e['unique_to'][:3]}")
