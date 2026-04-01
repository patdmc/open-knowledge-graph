#!/usr/bin/env python3
"""
Three-tier lexical knowledge graph with progressive disclosure.

Level 0: Skeleton — {id, type, categories, ref_ids, pattern_id}
         What map/reduce sees. Fits in memory for entire corpus.

Level 1: Assertion chain — {step, type, depends_on, references}
         Loaded on demand when a candidate is worth investigating.

Level 2: Full text — LaTeX proof, natural language.
         Fetched from source only when a human reads it.

Pattern nodes are first-class entities. A proof method like
induction(base, step) exists once. Every proof using induction
points to it. Transposes = different-domain proofs sharing a parent pattern.

Usage:
    from lexical_graph import LexicalGraph
    lg = LexicalGraph()
    lg.ingest_naturalproofs("path/to/naturalproofs_proofwiki.json")
    lg.ingest_yaml_proofs("knowledge-graph/nodes/math/proofs/")
    lg.build_pattern_index()
    lg.find_transposes()
    lg.print_report()
"""

import json
import re
import yaml
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional, FrozenSet


# ── PATTERN DETECTION ──────────────────────────────────────

MOVES = {
    "contradiction": [
        r"(?:assume|suppos).*(?:not|negat|contrar)",
        r"contradiction",
        r"\\bot",
        r"absurd",
    ],
    "induction": [
        r"(?:base|basis)\s+case",
        r"induct(?:ive|ion)",
        r"P\s*\(\s*[nk]\s*\+\s*1",
    ],
    "construction": [
        r"(?:construct|exhibit)\b",
        r"(?:define|let)\s+\w+\s*=",
    ],
    "counting": [
        r"(?:double|over).?count",
        r"bijection\s+between",
    ],
    "pigeonhole": [
        r"pigeonhole",
    ],
    "diagonalization": [
        r"diagonal(?:iz)",
    ],
    "compactness": [
        r"compact(?:ness)?",
        r"(?:finite|open)\s+(?:sub)?cover",
    ],
    "continuity": [
        r"(?:by\s+)?continuity",
        r"intermediate\s+value",
        r"epsilon.*delta",
    ],
    "linearity": [
        r"(?:by|using)\s+linearity",
        r"linearity\s+of",
    ],
    "symmetry": [
        r"(?:by\s+)?symmetry",
        r"WLOG|without\s+loss",
    ],
    "bounding": [
        r"Markov|Chebyshev|Cauchy.Schwarz|H.lder|Jensen|Minkowski",
        r"(?:upper|lower)\s+bound",
        r"triangle\s+inequality",
    ],
    "decomposition": [
        r"decompos(?:e|ition)",
        r"direct\s+sum",
        r"(?:factor|split)\s+(?:into|as)",
    ],
    "limit": [
        r"\\lim\b",
        r"converge[sd]?\s+to",
        r"\\to\s*(?:\\infty|0)",
    ],
    "fixed_point": [
        r"fixed\s+point",
        r"contraction\s+mapping",
        r"Banach|Brouwer|Kakutani",
    ],
    "spectral": [
        r"eigenvalue|eigenvector|spectrum",
        r"spectral",
        r"characteristic\s+polynomial",
    ],
    "approximation": [
        r"Taylor|Fourier|Laurent",
        r"approximat(?:e|ion)",
        r"series\s+expansion",
    ],
    "duality": [
        r"dual(?:ity)?",
        r"adjoint",
        r"Hahn.Banach|Riesz",
    ],
    "quotient": [
        r"quotient\s+(?:group|ring|space|module)",
        r"factor\s+(?:group|ring)",
        r"modulo|mod\s+",
    ],
    "extension": [
        r"extend(?:s|ed|ing)?\s+(?:to|from)",
        r"Zorn|axiom\s+of\s+choice",
    ],
    "density": [
        r"dense\s+(?:in|subset)",
        r"(?:Stone|Weierstrass)",
        r"approximation\s+(?:by|theorem)",
    ],
}


def _detect_moves(text):
    """Detect proof moves from text."""
    moves = set()
    for move, patterns in MOVES.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                moves.add(move)
                break
    return frozenset(moves)


# ── DATA CLASSES ───────────────────────────────────────────

@dataclass
class PatternNode:
    """First-class proof pattern. Exists once, pointed to by many proofs."""
    id: str
    moves: FrozenSet[str]
    users: list = field(default_factory=list)  # list of skeleton IDs

    @property
    def key(self):
        return "|".join(sorted(self.moves)) if self.moves else "atomic"


@dataclass
class Skeleton:
    """Level 0: Minimal node. What map/reduce operates on."""
    id: str
    title: str
    node_type: str          # theorem, definition, shared_lemma, proof
    categories: frozenset   # domain labels
    ref_ids: frozenset      # dependency edges (other skeleton IDs)
    pattern_id: str         # pointer to PatternNode
    source: dict            # {dataset, id} pointer to Level 2

    @property
    def category_key(self):
        return "|".join(sorted(self.categories))


@dataclass
class AssertionChain:
    """Level 1: Loaded on demand for candidates worth investigating."""
    skeleton_id: str
    steps: list  # [{id, type, depends_on, references, move}]

    @classmethod
    def from_naturalproofs(cls, skeleton_id, proofs, all_items):
        """Extract assertion chain from NaturalProofs proof list."""
        steps = []
        for i, proof in enumerate(proofs):
            text = " ".join(proof.get("contents", []))
            moves = _detect_moves(text)
            ref_titles = []
            for rid in proof.get("ref_ids", []):
                item = all_items.get(rid)
                if item:
                    ref_titles.append(item.get("title", f"#{rid}"))

            steps.append({
                "id": f"proof-{i}",
                "type": "proof",
                "depends_on": [f"proof-{i-1}"] if i > 0 else [],
                "references": ref_titles[:20],
                "moves": sorted(moves),
                "text_length": len(text),
            })
        return cls(skeleton_id=skeleton_id, steps=steps)

    @classmethod
    def from_yaml_proof(cls, skeleton_id, assertions):
        """Extract assertion chain from our YAML proof format."""
        steps = []
        for a in assertions:
            if not isinstance(a, dict):
                continue
            text = a.get("statement", "") + " " + a.get("justification", "")
            moves = _detect_moves(text)
            steps.append({
                "id": a.get("id", "?"),
                "type": a.get("type", "claim"),
                "depends_on": a.get("depends_on", []),
                "references": a.get("references", []),
                "moves": sorted(moves),
            })
        return cls(skeleton_id=skeleton_id, steps=steps)


# ── LEXICAL GRAPH ──────────────────────────────────────────

class LexicalGraph:
    """Three-tier knowledge graph with pattern nodes."""

    def __init__(self):
        self.skeletons = {}       # id → Skeleton
        self.patterns = {}        # pattern_key → PatternNode
        self.chains = {}          # id → AssertionChain (Level 1, loaded lazily)
        self._np_items = {}       # NaturalProofs raw items (for Level 1/2 fetching)
        self._pattern_counter = 0

    # ── INGEST: NaturalProofs ──────────────────────────────

    def ingest_naturalproofs(self, json_path):
        """Ingest NaturalProofs dataset at Level 0."""
        with open(json_path) as f:
            raw = json.load(f)

        ds = raw["dataset"]
        source_name = Path(json_path).stem

        for item in ds["theorems"] + ds["definitions"] + ds["others"]:
            iid = item["id"]
            self._np_items[iid] = item

            # Detect proof pattern from all proof text
            all_text = " ".join(item.get("contents", []))
            for proof in item.get("proofs", []):
                all_text += " " + " ".join(proof.get("contents", []))
            moves = _detect_moves(all_text)

            # Get or create pattern node
            pattern = self._get_or_create_pattern(moves)

            # Collect all ref_ids
            refs = set(item.get("ref_ids", []))
            for proof in item.get("proofs", []):
                refs.update(proof.get("ref_ids", []))

            sid = f"NP-{iid}"
            sk = Skeleton(
                id=sid,
                title=item.get("title", ""),
                node_type=item.get("type", "unknown"),
                categories=frozenset(item.get("toplevel_categories", [])),
                ref_ids=frozenset(f"NP-{r}" for r in refs),
                pattern_id=pattern.id,
                source={"dataset": source_name, "id": iid},
            )
            self.skeletons[sid] = sk
            pattern.users.append(sid)

    # ── INGEST: YAML proofs ────────────────────────────────

    def ingest_yaml_proofs(self, proofs_dir):
        """Ingest our YAML proof files at Level 0."""
        proofs_path = Path(proofs_dir)
        for p in sorted(proofs_path.glob("*.yaml")):
            try:
                with open(p) as f:
                    data = yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") not in ("proof", "shared_lemma"):
                continue

            pid = data["id"]

            # Detect pattern from assertions
            all_text = ""
            for a in data.get("assertions", []):
                if isinstance(a, dict):
                    all_text += " " + a.get("statement", "") + " " + a.get("justification", "")
            moves = _detect_moves(all_text)
            pattern = self._get_or_create_pattern(moves)

            # Collect references
            refs = set()
            for a in data.get("assertions", []):
                if isinstance(a, dict):
                    for r in a.get("references", []):
                        refs.add(r)

            sk = Skeleton(
                id=pid,
                title=data.get("name", pid),
                node_type=data.get("type", "proof"),
                categories=frozenset([data.get("domain", "mathematics")]),
                ref_ids=frozenset(refs),
                pattern_id=pattern.id,
                source={"dataset": "yaml", "path": str(p)},
            )
            self.skeletons[pid] = sk
            pattern.users.append(pid)

    def ingest_yaml_shared(self, shared_dir):
        """Ingest shared lemma files."""
        shared_path = Path(shared_dir)
        for p in sorted(shared_path.glob("*.yaml")):
            try:
                with open(p) as f:
                    data = yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(data, dict) or data.get("type") != "shared_lemma":
                continue

            sid = data["id"]
            text = data.get("statement", "") + " " + data.get("justification", "")
            moves = _detect_moves(text)
            pattern = self._get_or_create_pattern(moves)

            refs = set()
            for r in data.get("references", []):
                if isinstance(r, str):
                    refs.add(r)

            sk = Skeleton(
                id=sid,
                title=data.get("name", sid),
                node_type="shared_lemma",
                categories=frozenset([data.get("domain", "mathematics")]),
                ref_ids=frozenset(refs),
                pattern_id=pattern.id,
                source={"dataset": "yaml", "path": str(p)},
            )
            self.skeletons[sid] = sk
            pattern.users.append(sid)

    # ── PATTERN INDEX ──────────────────────────────────────

    def _get_or_create_pattern(self, moves):
        """Get existing pattern node or create new one."""
        key = "|".join(sorted(moves)) if moves else "atomic"
        if key not in self.patterns:
            self._pattern_counter += 1
            self.patterns[key] = PatternNode(
                id=f"PAT-{self._pattern_counter:04d}",
                moves=moves,
            )
        return self.patterns[key]

    def build_pattern_index(self):
        """Build category index on patterns for fast transpose lookup."""
        # Already built during ingest — patterns have users lists
        pass

    # ── LEVEL 1 LOADING ────────────────────────────────────

    def load_chain(self, skeleton_id):
        """Lazily load Level 1 assertion chain for a skeleton."""
        if skeleton_id in self.chains:
            return self.chains[skeleton_id]

        sk = self.skeletons.get(skeleton_id)
        if not sk:
            return None

        if sk.source.get("dataset", "").startswith("naturalproofs"):
            np_id = sk.source["id"]
            item = self._np_items.get(np_id)
            if item and item.get("proofs"):
                chain = AssertionChain.from_naturalproofs(
                    skeleton_id, item["proofs"], self._np_items
                )
                self.chains[skeleton_id] = chain
                return chain
        elif sk.source.get("dataset") == "yaml":
            path = sk.source.get("path")
            if path:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and data.get("assertions"):
                    chain = AssertionChain.from_yaml_proof(skeleton_id, data["assertions"])
                    self.chains[skeleton_id] = chain
                    return chain
        return None

    # ── TRANSPOSE DISCOVERY ────────────────────────────────

    def find_transposes(self, min_moves=1, min_category_gap=2):
        """Find transposes: different-domain proofs sharing a pattern node.

        No pairwise comparison. Just check which patterns have users
        from different category sets.
        """
        transposes = []

        for pat_key, pattern in self.patterns.items():
            if len(pattern.moves) < min_moves:
                continue
            if len(pattern.users) < 2:
                continue

            # Group users by category set
            cat_groups = defaultdict(list)
            for uid in pattern.users:
                sk = self.skeletons.get(uid)
                if sk:
                    cat_groups[sk.category_key].append(uid)

            if len(cat_groups) < 2:
                continue

            # Find the most distant category pairs
            cat_keys = sorted(cat_groups.keys())
            for i, ck_a in enumerate(cat_keys):
                for ck_b in cat_keys[i+1:]:
                    cats_a = set(ck_a.split("|")) if ck_a else set()
                    cats_b = set(ck_b.split("|")) if ck_b else set()
                    gap = len((cats_a - cats_b) | (cats_b - cats_a))

                    if gap < min_category_gap:
                        continue

                    # Pick representative from each group
                    rep_a = cat_groups[ck_a][0]
                    rep_b = cat_groups[ck_b][0]
                    sk_a = self.skeletons[rep_a]
                    sk_b = self.skeletons[rep_b]

                    transposes.append({
                        "pattern": pattern.key,
                        "pattern_id": pattern.id,
                        "moves": sorted(pattern.moves),
                        "id_a": rep_a,
                        "id_b": rep_b,
                        "title_a": sk_a.title,
                        "title_b": sk_b.title,
                        "cats_a": sorted(cats_a),
                        "cats_b": sorted(cats_b),
                        "unique_a": sorted(cats_a - cats_b),
                        "unique_b": sorted(cats_b - cats_a),
                        "shared_cats": sorted(cats_a & cats_b),
                        "gap": gap,
                        "n_users_a": len(cat_groups[ck_a]),
                        "n_users_b": len(cat_groups[ck_b]),
                    })

        transposes.sort(key=lambda t: (len(t["moves"]), t["gap"]), reverse=True)
        self._transposes = transposes
        return transposes

    # ── REPORTING ──────────────────────────────────────────

    def stats(self):
        """Print graph statistics."""
        print(f"Skeletons:     {len(self.skeletons):,}")
        print(f"Pattern nodes: {len(self.patterns):,}")
        print(f"Chains loaded: {len(self.chains):,}")

        # Pattern distribution
        sizes = sorted((len(p.users), p.key) for p in self.patterns.values())
        print(f"\nLargest patterns (most proofs sharing same moves):")
        for count, key in sizes[-15:]:
            print(f"  {count:6,}  {key}")

        # Category coverage
        all_cats = Counter()
        for sk in self.skeletons.values():
            for c in sk.categories:
                all_cats[c] += 1
        print(f"\nCategory coverage:")
        for cat, count in all_cats.most_common(15):
            print(f"  {count:6,}  {cat}")

    def print_report(self, top_n=30):
        """Print transpose discovery report."""
        transposes = getattr(self, "_transposes", [])
        print("\n" + "=" * 70)
        print(f"TRANSPOSES — {len(transposes):,} found")
        print(f"Same proof pattern, different domains")
        print("=" * 70)

        # Group by pattern
        by_pattern = Counter(t["pattern"] for t in transposes)
        print(f"\nPatterns producing most transposes:")
        for pat, count in by_pattern.most_common(10):
            print(f"  {count:5,}  {pat}")

        # Show top candidates, deduped by pattern
        seen_patterns = set()
        shown = 0
        print(f"\nTop candidates (one per pattern):\n")
        for t in transposes:
            if t["pattern"] in seen_patterns:
                continue
            seen_patterns.add(t["pattern"])

            print(f"  PATTERN: {t['pattern']}")
            print(f"    {t['title_a'][:55]}")
            print(f"      cats: {t['unique_a'][:4]}")
            print(f"    ↔ {t['title_b'][:55]}")
            print(f"      cats: {t['unique_b'][:4]}")
            print(f"    gap: {t['gap']}, users: {t['n_users_a']}+{t['n_users_b']}")
            print()

            shown += 1
            if shown >= top_n:
                break

    def investigate(self, skeleton_id):
        """Progressive disclosure: load Level 1 and show assertion chain."""
        sk = self.skeletons.get(skeleton_id)
        if not sk:
            print(f"Not found: {skeleton_id}")
            return

        print(f"=== {sk.title} ===")
        print(f"  Type: {sk.node_type}")
        print(f"  Categories: {sorted(sk.categories)}")
        print(f"  Pattern: {sk.pattern_id} ({self.patterns.get(sk.pattern_id.replace('PAT-',''), PatternNode('?', frozenset())).key if False else '...'})")
        print(f"  Refs: {len(sk.ref_ids)}")

        # Load Level 1
        chain = self.load_chain(skeleton_id)
        if chain:
            print(f"\n  Assertion chain ({len(chain.steps)} steps):")
            for step in chain.steps:
                moves_str = f" [{', '.join(step['moves'])}]" if step.get("moves") else ""
                deps_str = f" ← {step['depends_on']}" if step.get("depends_on") else ""
                refs_str = f" refs: {step['references'][:3]}" if step.get("references") else ""
                print(f"    {step['id']:20s} {step['type']:12s}{moves_str}{deps_str}{refs_str}")
        else:
            print("  (No assertion chain available)")


if __name__ == "__main__":
    lg = LexicalGraph()

    print("Ingesting NaturalProofs...")
    lg.ingest_naturalproofs("knowledge-graph/sources/naturalproofs/naturalproofs_proofwiki.json")

    print("Ingesting YAML proofs...")
    lg.ingest_yaml_proofs("knowledge-graph/nodes/math/proofs/")
    lg.ingest_yaml_shared("knowledge-graph/nodes/math/proofs/shared/")

    print()
    lg.stats()

    print("\nFinding transposes...")
    lg.find_transposes(min_moves=2, min_category_gap=3)
    lg.print_report(top_n=25)
