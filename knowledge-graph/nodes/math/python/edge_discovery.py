#!/usr/bin/env python3
"""
Edge discovery via map/reduce over the knowledge graph.

No embeddings. No models. Exact set intersection.

Map:  each node emits feature keys (references, domains, symbols, lemmas, LaTeX tokens)
Reduce: group by key → pairs with shared features but no existing edge → candidates

Usage:
    from edge_discovery import EdgeDiscovery
    ed = EdgeDiscovery()
    ed.discover()
    ed.print_report()
"""

import os
import re
import yaml
from pathlib import Path
from collections import defaultdict
from itertools import combinations


ROOT = Path(__file__).resolve().parent.parent.parent.parent  # knowledge-graph/
NODES = ROOT / "nodes"
MAX_FILE_SIZE = 500_000

# Directories to scan
SCAN_DIRS = [
    "math", "information-theory", "neurology", "endocrinology", "biology",
    "definitions", "theorems", "emergent", "empirical",
    "equivalency", "novel", "open-questions", "overlap", "references",
]


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _latex_tokens(s):
    """Extract meaningful LaTeX tokens from a string."""
    if not s:
        return set()
    # Pull out \command sequences and variable names
    commands = set(re.findall(r'\\([a-zA-Z]+)', s))
    # Pull out single-letter variables (not in commands)
    stripped = re.sub(r'\\[a-zA-Z]+', '', s)
    variables = set(re.findall(r'\b([A-Za-z])\b', stripped))
    return commands | variables


def _all_yamls():
    """Yield (path, data) for every YAML file in knowledge dirs."""
    for dirname in SCAN_DIRS:
        d = NODES / dirname
        if not d.exists():
            continue
        for p in d.rglob("*.yaml"):
            if p.stat().st_size > MAX_FILE_SIZE:
                continue
            try:
                data = _load_yaml(p)
            except Exception:
                continue
            if isinstance(data, dict) and "id" in data:
                yield p, data


class EdgeDiscovery:
    """Map/reduce edge discovery engine."""

    def __init__(self):
        self.nodes = {}          # id → data
        self.node_files = {}     # id → path
        self.features = {}       # id → set of feature keys
        self.index = defaultdict(set)  # feature_key → set of node ids
        self.existing_edges = set()    # (id_a, id_b) pairs already connected
        self.candidates = []     # discovered candidates

        self._load()

    def _load(self):
        """Load all nodes."""
        for path, data in _all_yamls():
            nid = data["id"]
            self.nodes[nid] = data
            self.node_files[nid] = path

    # ── MAP PHASE ──────────────────────────────────────────────

    def _map_node(self, nid, data):
        """Emit feature keys for a single node."""
        keys = set()
        ntype = data.get("type", "")

        # Feature 1: concept IDs referenced
        for c in data.get("concepts", []):
            if isinstance(c, dict) and "id" in c:
                keys.add(f"concept:{c['id']}")

        # Feature 2: domain
        domain = data.get("domain", "")
        if domain:
            keys.add(f"domain:{domain}")

        # Feature 3: references from proof assertions
        for a in data.get("assertions", []):
            if isinstance(a, dict):
                for ref in a.get("references", []):
                    keys.add(f"ref:{ref}")

        # Feature 4: references from claims
        for claim in data.get("claims", []):
            if isinstance(claim, dict):
                for ref in claim.get("references", []):
                    keys.add(f"ref:{ref}")

        # Feature 5: shared lemma usage
        for u in data.get("used_by", []):
            if isinstance(u, dict):
                keys.add(f"proof:{u.get('proof', '')}")

        # Feature 6: proof links
        for link in data.get("proof_links", []):
            if isinstance(link, dict):
                keys.add(f"proof:{link.get('target', '')}")

        # Feature 7: cross-domain edge targets
        for edge in data.get("cross_domain_edges", []):
            if isinstance(edge, dict):
                keys.add(f"xdomain:{edge.get('target', '')}")

        for edge in data.get("edges", []):
            if isinstance(edge, dict):
                to = edge.get("to", "")
                if to:
                    keys.add(f"edge:{to}")
                    self.existing_edges.add((nid, to))
                    self.existing_edges.add((to, nid))
                fr = edge.get("from", "")
                if fr and fr != nid:
                    keys.add(f"edge:{fr}")
                    self.existing_edges.add((nid, fr))
                    self.existing_edges.add((fr, nid))

        for cd in data.get("cross_domain", []):
            if isinstance(cd, dict):
                target = cd.get("target", "")
                if target:
                    keys.add(f"xdomain:{target}")
                    self.existing_edges.add((nid, target))
                    self.existing_edges.add((target, nid))

        # Feature 8: LaTeX tokens from concepts
        for c in data.get("concepts", []):
            if isinstance(c, dict):
                for tok in _latex_tokens(c.get("latex", "")):
                    keys.add(f"latex:{tok}")

        # Feature 9: LaTeX from proof assertions
        for a in data.get("assertions", []):
            if isinstance(a, dict):
                for tok in _latex_tokens(a.get("statement", "")):
                    keys.add(f"latex:{tok}")

        # Feature 10: LaTeX from claims
        for claim in data.get("claims", []):
            if isinstance(claim, dict):
                for tok in _latex_tokens(claim.get("latex", "")):
                    keys.add(f"latex:{tok}")

        # Feature 11: symbols
        for sym in data.get("symbols", []):
            if isinstance(sym, dict):
                keys.add(f"symbol:{sym.get('sym', '')}")

        # Feature 12: references list (top-level, for shared lemmas)
        for ref in data.get("references", []):
            if isinstance(ref, str):
                keys.add(f"ref:{ref}")

        return keys

    def map_phase(self):
        """Map every node to its feature keys."""
        for nid, data in self.nodes.items():
            keys = self._map_node(nid, data)
            self.features[nid] = keys
            for k in keys:
                self.index[k].add(nid)

    # ── REDUCE PHASE ───────────────────────────────────────────

    def reduce_phase(self, min_shared=2, exclude_latex_only=True):
        """Find node pairs sharing features but lacking edges.

        Args:
            min_shared: minimum shared feature keys to be a candidate
            exclude_latex_only: if True, require at least one non-latex shared feature
        """
        # Collect all pairs and their shared features
        pair_features = defaultdict(set)

        for key, node_ids in self.index.items():
            if len(node_ids) < 2 or len(node_ids) > 50:
                # Skip features that are too common (noise) or singleton
                continue
            for a, b in combinations(sorted(node_ids), 2):
                pair_features[(a, b)].add(key)

        # Filter to candidates
        self.candidates = []
        for (a, b), shared in pair_features.items():
            if len(shared) < min_shared:
                continue
            if (a, b) in self.existing_edges:
                continue

            if exclude_latex_only:
                non_latex = [k for k in shared if not k.startswith("latex:")]
                if not non_latex:
                    continue

            # Score by number of shared features, weighted by specificity
            score = 0
            for k in shared:
                n_nodes = len(self.index[k])
                # Rarer features are more informative
                specificity = 1.0 / n_nodes
                score += specificity

            self.candidates.append({
                "node_a": a,
                "node_b": b,
                "shared_features": sorted(shared),
                "n_shared": len(shared),
                "score": round(score, 3),
                "name_a": self.nodes[a].get("name", a),
                "name_b": self.nodes[b].get("name", b),
            })

        self.candidates.sort(key=lambda c: c["score"], reverse=True)

    # ── DISCOVER ───────────────────────────────────────────────

    def discover(self, min_shared=2):
        """Run full map/reduce pipeline."""
        self.map_phase()
        self.reduce_phase(min_shared=min_shared)
        return self.candidates

    # ── REPORTING ──────────────────────────────────────────────

    def print_report(self, top_n=30):
        """Print the top candidate edges."""
        print("=" * 70)
        print(f"EDGE DISCOVERY — {len(self.candidates)} candidates found")
        print(f"  {len(self.nodes)} nodes, {len(self.index)} feature keys")
        print(f"  {len(self.existing_edges)//2} existing edges")
        print("=" * 70)

        # Group by feature type
        feature_types = defaultdict(int)
        for key in self.index:
            ftype = key.split(":")[0]
            feature_types[ftype] += 1
        print("\nFeature index:")
        for ft, count in sorted(feature_types.items(), key=lambda x: -x[1]):
            print(f"  {ft}: {count} keys")

        print(f"\nTop {top_n} candidate edges:\n")
        for c in self.candidates[:top_n]:
            print(f"  [{c['score']:.2f}] {c['name_a'][:40]}")
            print(f"       ↔ {c['name_b'][:40]}")

            # Show shared features grouped by type
            by_type = defaultdict(list)
            for f in c["shared_features"]:
                ftype, fval = f.split(":", 1)
                by_type[ftype].append(fval)
            for ftype in ["ref", "concept", "xdomain", "symbol", "domain", "proof", "edge", "latex"]:
                if ftype in by_type:
                    vals = by_type[ftype]
                    if len(vals) <= 5:
                        print(f"       {ftype}: {', '.join(vals)}")
                    else:
                        print(f"       {ftype}: {', '.join(vals[:5])} +{len(vals)-5} more")
            print()

    def cross_domain_candidates(self, top_n=20):
        """Return only candidates that cross domain boundaries."""
        cross = []
        for c in self.candidates:
            domain_a = self.nodes[c["node_a"]].get("domain", "")
            domain_b = self.nodes[c["node_b"]].get("domain", "")
            type_a = self.nodes[c["node_a"]].get("type", "")
            type_b = self.nodes[c["node_b"]].get("type", "")
            if domain_a != domain_b or type_a != type_b:
                cross.append(c)
        return cross[:top_n]

    def feature_histogram(self):
        """Show how many features each node emits."""
        counts = [(nid, len(keys)) for nid, keys in self.features.items()]
        counts.sort(key=lambda x: -x[1])
        print("\nFeature counts per node (top 20):")
        for nid, count in counts[:20]:
            name = self.nodes[nid].get("name", nid)[:50]
            print(f"  {count:4d}  {name}")
        print(f"\n  median: {sorted(c for _, c in counts)[len(counts)//2]}")
        print(f"  total nodes: {len(counts)}")


if __name__ == "__main__":
    ed = EdgeDiscovery()
    ed.discover()
    ed.print_report()
