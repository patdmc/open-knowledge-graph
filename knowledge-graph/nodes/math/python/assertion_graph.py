#!/usr/bin/env python3
"""
Assertion-level knowledge graph.

The atom is the assertion, not the lemma. One lemma may contain
multiple assertions. Each assertion is its own node in the graph.

Assertion encoding: logical skeleton stripped of domain specifics.
  "bounded(X) → ∃ convergent(sub(X))"
  "closed(S) ∧ bounded(S) → compact(S)"
  "f continuous ∧ f(a)<0 ∧ f(b)>0 → ∃c f(c)=0"

These compress to patterns that match across domains.

Three tiers:
  L0: Assertion skeleton — {id, parent_lemma, logical_form, quantifiers, predicates}
  L1: Full assertion — statement, justification, depends_on, references
  L2: Full proof text — fetched from source on demand

Usage:
    from assertion_graph import AssertionGraph
    ag = AssertionGraph()
    ag.ingest_naturalproofs("path/to/proofwiki.json")
    ag.ingest_yaml_proofs("knowledge-graph/nodes/math/proofs/")
    ag.build_predicate_index()
    candidates = ag.find_assertion_transposes()
    ag.print_report()
"""

import json
import re
import yaml
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import FrozenSet, Optional


# ── LOGICAL FORM EXTRACTION ───────────────────────────────

# Quantifier patterns
QUANTIFIERS = {
    "universal": [
        r"\\forall", r"for\s+(?:all|every|each|any)\b",
        r"(?:let|suppose)\s+\w+\s+be\s+(?:an?\s+)?(?:arbitrary|any)",
    ],
    "existential": [
        r"\\exists", r"there\s+exist",
        r"(?:we\s+can\s+)?find\s+(?:a|an|some)\b",
        r"has\s+(?:a|an)\b",
    ],
    "unique_existential": [
        r"\\exists\s*!", r"there\s+exists?\s+(?:a\s+)?unique",
        r"exactly\s+one",
    ],
}

# Predicate patterns — domain-independent logical properties
PREDICATES = {
    # Boundedness family
    "bounded": [r"\bbounded\b", r"\bfinite\b(?!\s+(?:group|field|ring))"],
    "unbounded": [r"\bunbounded\b", r"\binfinite\b"],

    # Convergence family
    "convergent": [r"\bconverg(?:e[sd]?|ent|ence)\b"],
    "divergent": [r"\bdiverge(?:s|nt|nce)?\b"],
    "limit_exists": [r"\\lim.*(?:exists|=)", r"limit\s+exists"],

    # Continuity family
    "continuous": [r"\bcontinuous\b", r"\bcontinuity\b"],
    "differentiable": [r"\bdifferentiable\b"],
    "integrable": [r"\bintegrable\b"],
    "measurable": [r"\bmeasurable\b"],
    "smooth": [r"\bsmooth\b", r"C\^\\infty", r"C\^\{\\infty\}"],

    # Structural family
    "injective": [r"\binjective\b", r"\bone-to-one\b", r"\b1-1\b"],
    "surjective": [r"\bsurjective\b", r"\bonto\b(?!\s+the)"],
    "bijective": [r"\bbijective\b", r"\bbijection\b"],
    "isomorphic": [r"\bisomorphi(?:c|sm)\b", r"\\cong", r"\\simeq"],
    "homeomorphic": [r"\bhomeomorphi(?:c|sm)\b"],

    # Closure family
    "closed": [r"\bclosed\b(?!\s+form)"],
    "open": [r"\bopen\b(?!\s+(?:source|question|problem))"],
    "compact": [r"\bcompact\b"],
    "dense": [r"\bdense\b"],
    "connected": [r"\bconnected\b"],
    "complete": [r"\bcomplete\b(?!\s+(?:the|this))"],
    "separable": [r"\bseparable\b"],

    # Algebraic family
    "commutative": [r"\bcommutativ(?:e|ity)\b", r"\babelian\b"],
    "normal": [r"\bnormal\s+(?:sub)?group\b", r"\\trianglelefteq"],
    "cyclic": [r"\bcyclic\b"],
    "simple": [r"\bsimple\b(?!\s+(?:way|example|case))"],
    "solvable": [r"\bsolvable\b"],

    # Ordering family
    "monotone": [r"\bmonoton(?:e|ic|ically)\b", r"\b(?:in|de)creasing\b"],
    "maximal": [r"\bmaximal\b", r"\bmaximum\b", r"\bsupremum\b", r"\\sup\b"],
    "minimal": [r"\bminimal\b", r"\bminimum\b", r"\binfimum\b", r"\\inf\b"],

    # Uniqueness / existence
    "unique": [r"\bunique(?:ly|ness)?\b"],
    "exists": [r"\bexist(?:s|ence)?\b"],

    # Equivalence
    "iff": [r"\bif\s+and\s+only\s+if\b", r"\\iff", r"\\Leftrightarrow"],
    "implies": [r"\bimplies\b", r"\\implies", r"\\Rightarrow", r"\bthen\b"],

    # Size / cardinality
    "countable": [r"\bcountabl(?:e|y)\b"],
    "uncountable": [r"\buncountable\b"],
    "equinumerous": [r"\bequinumerous\b", r"\bsame\s+cardinality\b"],

    # Invariance
    "invariant": [r"\binvariant\b"],
    "preserved": [r"\bpreserv(?:e[sd]?|ing)\b"],
    "stable": [r"\bstable\b", r"\bstability\b"],
}


def _extract_quantifiers(text):
    """Extract quantifier types from text."""
    found = set()
    for qtype, patterns in QUANTIFIERS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.add(qtype)
                break
    return frozenset(found)


def _extract_predicates(text):
    """Extract logical predicates from text."""
    found = set()
    for pred, patterns in PREDICATES.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.add(pred)
                break
    return frozenset(found)


def _logical_signature(quantifiers, predicates):
    """Create a canonical logical signature from quantifiers and predicates."""
    parts = []
    if quantifiers:
        parts.append("Q:" + "+".join(sorted(quantifiers)))
    if predicates:
        parts.append("P:" + "+".join(sorted(predicates)))
    return "|".join(parts) if parts else "atomic"


# ── DATA STRUCTURES ───────────────────────────────────────

@dataclass(frozen=True)
class AssertionNode:
    """Level 0 atom: one assertion from one lemma/theorem."""
    id: str                         # globally unique
    parent_id: str                  # lemma/theorem this came from
    parent_title: str
    assertion_index: int            # position within parent
    quantifiers: FrozenSet[str]
    predicates: FrozenSet[str]
    signature: str                  # canonical logical form
    categories: FrozenSet[str]      # domain labels (inherited from parent)
    source: str                     # dataset identifier


class AssertionGraph:
    """Assertion-level map/reduce graph."""

    def __init__(self):
        self.assertions = {}             # id → AssertionNode
        self.signature_index = defaultdict(list)  # signature → [assertion_ids]
        self.predicate_index = defaultdict(set)   # predicate → {assertion_ids}
        self._candidates = []

    # ── INGEST: NaturalProofs ──────────────────────────────

    def ingest_naturalproofs(self, json_path, max_items=None):
        """Ingest NaturalProofs, splitting each item into assertion nodes."""
        with open(json_path) as f:
            raw = json.load(f)

        ds = raw["dataset"]
        count = 0

        for item in ds["theorems"]:
            if max_items and count >= max_items:
                break

            iid = item["id"]
            title = item.get("title", "")
            cats = frozenset(item.get("toplevel_categories", []))

            # Split statement into assertions
            # Each sentence/clause in contents that makes a claim is an assertion
            statement_text = " ".join(item.get("contents", []))
            statement_assertions = self._split_assertions(statement_text)

            # Also split each proof into assertions
            proof_assertions = []
            for pi, proof in enumerate(item.get("proofs", [])):
                proof_text = " ".join(proof.get("contents", []))
                proof_assertions.extend(self._split_assertions(proof_text))

            all_assertions = statement_assertions + proof_assertions

            if not all_assertions:
                # Fallback: treat whole statement as one assertion
                all_assertions = [statement_text]

            for ai, atext in enumerate(all_assertions):
                if len(atext.strip()) < 20:
                    continue

                quants = _extract_quantifiers(atext)
                preds = _extract_predicates(atext)

                if not preds:
                    continue  # Skip assertions with no detectable predicates

                sig = _logical_signature(quants, preds)
                aid = f"NP-{iid}-A{ai}"

                node = AssertionNode(
                    id=aid,
                    parent_id=f"NP-{iid}",
                    parent_title=title,
                    assertion_index=ai,
                    quantifiers=quants,
                    predicates=preds,
                    signature=sig,
                    categories=cats,
                    source="naturalproofs",
                )
                self.assertions[aid] = node
                self.signature_index[sig].append(aid)
                for pred in preds:
                    self.predicate_index[pred].add(aid)

            count += 1

    # ── INGEST: YAML proofs ────────────────────────────────

    def ingest_yaml_proofs(self, proofs_dir):
        """Ingest YAML proofs at assertion level."""
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
            title = data.get("name", pid)
            domain = data.get("domain", "mathematics")

            for ai, a in enumerate(data.get("assertions", [])):
                if not isinstance(a, dict):
                    continue

                text = a.get("statement", "") + " " + a.get("justification", "")
                quants = _extract_quantifiers(text)
                preds = _extract_predicates(text)

                if not preds:
                    continue

                sig = _logical_signature(quants, preds)
                aid = f"{pid}-A{ai}"

                node = AssertionNode(
                    id=aid,
                    parent_id=pid,
                    parent_title=title,
                    assertion_index=ai,
                    quantifiers=quants,
                    predicates=preds,
                    signature=sig,
                    categories=frozenset([domain]),
                    source="yaml",
                )
                self.assertions[aid] = node
                self.signature_index[sig].append(aid)
                for pred in preds:
                    self.predicate_index[pred].add(aid)

    # ── ASSERTION SPLITTING ────────────────────────────────

    def _split_assertions(self, text):
        """Split text into individual assertion chunks.

        Heuristic: split on sentence boundaries that contain logical content.
        Each chunk should be one claim.
        """
        # Clean wiki markup
        clean = re.sub(r'\[\[[^|\]]*\|([^\]]+)\]\]', r'\1', text)
        clean = re.sub(r'\[\[([^\]]+)\]\]', r'\1', clean)

        # Split on :$ (display math delimiter in ProofWiki) and sentence boundaries
        chunks = re.split(r'(?<=\.)\s+(?=[A-Z])|(?<=\})\s*:\s*\$', clean)

        # Filter to chunks that look like assertions (have predicates)
        assertions = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue
            # Must contain at least one predicate-like word
            if _extract_predicates(chunk):
                assertions.append(chunk)

        return assertions

    # ── MAP/REDUCE: Find transposes ────────────────────────

    def find_assertion_transposes(self, min_predicates=2, min_category_gap=2):
        """Map: assertions already indexed by signature.
        Reduce: signatures with users from different categories = transposes.
        """
        candidates = []

        for sig, aids in self.signature_index.items():
            if len(aids) < 2:
                continue

            # Parse signature to check predicate count
            preds_in_sig = sig.count("+") + 1 if "P:" in sig else 0
            if preds_in_sig < min_predicates:
                continue

            # Group by category set
            cat_groups = defaultdict(list)
            for aid in aids:
                node = self.assertions[aid]
                cat_key = "|".join(sorted(node.categories)) if node.categories else "none"
                cat_groups[cat_key].append(aid)

            if len(cat_groups) < 2:
                continue

            # Find most distant category pair
            cat_keys = sorted(cat_groups.keys())
            best_gap = 0
            best_pair = None

            for i, ck_a in enumerate(cat_keys):
                cats_a = set(ck_a.split("|")) if ck_a != "none" else set()
                for ck_b in cat_keys[i+1:]:
                    cats_b = set(ck_b.split("|")) if ck_b != "none" else set()
                    gap = len((cats_a - cats_b) | (cats_b - cats_a))
                    if gap > best_gap:
                        best_gap = gap
                        # Pick representative from each
                        rep_a = cat_groups[ck_a][0]
                        rep_b = cat_groups[ck_b][0]
                        best_pair = (rep_a, cats_a, rep_b, cats_b)

            if best_pair and best_gap >= min_category_gap:
                node_a = self.assertions[best_pair[0]]
                node_b = self.assertions[best_pair[2]]

                candidates.append({
                    "signature": sig,
                    "predicates": sorted(node_a.predicates),
                    "quantifiers": sorted(node_a.quantifiers),
                    "id_a": node_a.id,
                    "id_b": node_b.id,
                    "parent_a": node_a.parent_title,
                    "parent_b": node_b.parent_title,
                    "cats_a": sorted(best_pair[1]),
                    "cats_b": sorted(best_pair[3]),
                    "unique_a": sorted(best_pair[1] - best_pair[3]),
                    "unique_b": sorted(best_pair[3] - best_pair[1]),
                    "gap": best_gap,
                    "n_assertions": len(aids),
                    "n_groups": len(cat_groups),
                })

        candidates.sort(key=lambda c: (len(c["predicates"]), c["gap"]), reverse=True)
        self._candidates = candidates
        return candidates

    # ── HUB SCORING ─────────────────────────────────────────

    def hub_scores(self):
        """Score each predicate as a hub.

        hub_score = log(n_connections) × (n_clusters / n_total_clusters)

        High score = real hub connecting many fields.
        High connections + low spread = field workhorse (not interesting).
        Low connections + high spread = thin bridge (very interesting).
        """
        import math

        # First: cluster assertions by category set
        all_category_sets = set()
        for node in self.assertions.values():
            if node.categories:
                all_category_sets.add(node.categories)
        n_total = max(len(all_category_sets), 1)

        scores = []
        for pred, aids in self.predicate_index.items():
            n_conn = len(aids)
            if n_conn < 2:
                continue

            # How many distinct category sets does this predicate span?
            cat_sets = set()
            for aid in aids:
                node = self.assertions[aid]
                if node.categories:
                    cat_sets.add(node.categories)
            n_clusters = len(cat_sets)

            spread = n_clusters / n_total
            log_conn = math.log2(max(n_conn, 1))
            hub = log_conn * spread
            bridge = spread / max(log_conn, 0.1)  # high spread, low connections

            scores.append({
                "predicate": pred,
                "connections": n_conn,
                "clusters": n_clusters,
                "spread": round(spread, 4),
                "hub_score": round(hub, 3),
                "bridge_score": round(bridge, 3),
            })

        self._hub_scores = sorted(scores, key=lambda s: s["hub_score"], reverse=True)
        self._bridge_scores = sorted(scores, key=lambda s: s["bridge_score"], reverse=True)
        return self._hub_scores

    def signature_hubs(self):
        """Score each SIGNATURE (predicate combination) as a hub.

        More specific than single predicates — 'bounded ∧ convergent'
        is more informative than just 'bounded'.
        """
        import math

        all_category_sets = set()
        for node in self.assertions.values():
            if node.categories:
                all_category_sets.add(node.categories)
        n_total = max(len(all_category_sets), 1)

        scores = []
        for sig, aids in self.signature_index.items():
            n_conn = len(aids)
            if n_conn < 2:
                continue

            # Count predicates in signature
            n_preds = sig.count("+") + 1 if "P:" in sig else 0
            if n_preds < 2:
                continue  # Skip single-predicate signatures

            cat_sets = set()
            for aid in aids:
                node = self.assertions[aid]
                if node.categories:
                    cat_sets.add(node.categories)
            n_clusters = len(cat_sets)

            spread = n_clusters / n_total
            log_conn = math.log2(max(n_conn, 1))
            hub = log_conn * spread
            bridge = spread / max(log_conn, 0.1)

            scores.append({
                "signature": sig,
                "n_predicates": n_preds,
                "connections": n_conn,
                "clusters": n_clusters,
                "spread": round(spread, 4),
                "hub_score": round(hub, 3),
                "bridge_score": round(bridge, 3),
            })

        self._sig_hub_scores = sorted(scores, key=lambda s: s["hub_score"], reverse=True)
        self._sig_bridge_scores = sorted(scores, key=lambda s: s["bridge_score"], reverse=True)
        return self._sig_hub_scores

    def print_hub_report(self, top_n=20):
        """Print hub and bridge analysis."""
        hubs = getattr(self, "_hub_scores", [])
        bridges = getattr(self, "_bridge_scores", [])
        sig_hubs = getattr(self, "_sig_hub_scores", [])
        sig_bridges = getattr(self, "_sig_bridge_scores", [])

        print("\n" + "=" * 70)
        print("PREDICATE HUB ANALYSIS")
        print("=" * 70)

        print(f"\nTop hubs (high connections × high spread):")
        for s in hubs[:top_n]:
            bar = "█" * int(s["hub_score"] * 2)
            print(f"  {s['hub_score']:6.2f}  {s['predicate']:20s}  "
                  f"conn={s['connections']:5,} clusters={s['clusters']:3d} "
                  f"spread={s['spread']:.3f}  {bar}")

        print(f"\nTop bridges (high spread, low connections):")
        for s in bridges[:top_n]:
            bar = "█" * int(s["bridge_score"] * 20)
            print(f"  {s['bridge_score']:6.3f}  {s['predicate']:20s}  "
                  f"conn={s['connections']:5,} clusters={s['clusters']:3d} "
                  f"spread={s['spread']:.3f}  {bar}")

        if sig_hubs:
            print(f"\n{'=' * 70}")
            print("SIGNATURE HUB ANALYSIS (predicate combinations)")
            print("=" * 70)

            print(f"\nTop signature hubs:")
            for s in sig_hubs[:top_n]:
                bar = "█" * int(s["hub_score"] * 3)
                print(f"  {s['hub_score']:6.2f}  {s['signature'][:50]:50s}  "
                      f"conn={s['connections']:4,} clusters={s['clusters']:3d}  {bar}")

            print(f"\nTop signature bridges (thin wires between worlds):")
            for s in sig_bridges[:top_n]:
                if s["connections"] > 100:
                    continue  # Skip the workhorses
                bar = "█" * int(s["bridge_score"] * 30)
                print(f"  {s['bridge_score']:6.3f}  {s['signature'][:50]:50s}  "
                      f"conn={s['connections']:4,} clusters={s['clusters']:3d}  {bar}")

    # ── ASSERTION CHAINING ─────────────────────────────────

    def find_chains(self, max_hops=3, min_predicates_per_link=2):
        """Find assertion chains: A shares pred with B, B shares pred with C.

        Lemma A says {1,2}, Lemma B says {2,3}, Lemma C says {3,4}
        → Chain A-B-C implies {1,4} might be interesting.

        Returns chains as lists of (assertion_id, shared_predicates) tuples.
        """
        # Build predicate overlap graph at the parent (lemma) level
        # Each parent has a set of predicates across all its assertions
        parent_predicates = defaultdict(set)  # parent_id → set of predicates
        parent_categories = {}  # parent_id → categories

        for node in self.assertions.values():
            parent_predicates[node.parent_id] |= node.predicates
            parent_categories[node.parent_id] = node.categories

        # Build adjacency: two parents are adjacent if they share >= min predicates
        # but have different category sets
        adj = defaultdict(list)  # parent_id → [(neighbor_id, shared_preds)]

        # Index: predicate → set of parent_ids
        pred_to_parents = defaultdict(list)
        for pid, preds in parent_predicates.items():
            for pred in preds:
                pred_to_parents[pred].append(pid)

        # For efficiency, only consider predicates with manageable fan-out
        for pred, parents in pred_to_parents.items():
            if len(parents) > 500:
                continue  # Skip ubiquitous predicates for chaining
            for i, pa in enumerate(parents[:200]):
                for pb in parents[i+1:200]:
                    if pa == pb:
                        continue
                    shared = parent_predicates[pa] & parent_predicates[pb]
                    if len(shared) >= min_predicates_per_link:
                        cats_a = parent_categories.get(pa, frozenset())
                        cats_b = parent_categories.get(pb, frozenset())
                        if cats_a != cats_b:  # Must cross domain boundary
                            adj[pa].append((pb, shared))
                            adj[pb].append((pa, shared))

        # BFS for chains up to max_hops
        chains = []
        visited_starts = set()

        for start in list(adj.keys())[:1000]:  # Cap for performance
            if start in visited_starts:
                continue
            visited_starts.add(start)

            # BFS
            queue = [(start, [start], set())]  # (current, path, accumulated_preds)
            while queue and len(chains) < 5000:
                current, path, acc_preds = queue.pop(0)
                if len(path) > max_hops + 1:
                    continue

                for neighbor, shared in adj[current]:
                    if neighbor in path:
                        continue

                    new_path = path + [neighbor]
                    new_preds = acc_preds | shared

                    if len(new_path) >= 3:
                        # Check if endpoints have non-overlapping categories
                        cats_start = parent_categories.get(path[0], frozenset())
                        cats_end = parent_categories.get(neighbor, frozenset())
                        gap = len((cats_start - cats_end) | (cats_end - cats_start))

                        if gap >= 3:
                            # Novel predicates: preds at endpoints not shared by middle
                            start_preds = parent_predicates[path[0]]
                            end_preds = parent_predicates[neighbor]
                            novel = (start_preds | end_preds) - new_preds
                            # The novel combination at the endpoints
                            endpoint_combo = start_preds & end_preds
                            if endpoint_combo - new_preds:
                                chains.append({
                                    "path": new_path,
                                    "length": len(new_path),
                                    "gap": gap,
                                    "shared_predicates": sorted(new_preds),
                                    "endpoint_combo": sorted(endpoint_combo),
                                    "novel_combo": sorted(endpoint_combo - new_preds),
                                    "cats_start": sorted(cats_start),
                                    "cats_end": sorted(cats_end),
                                })

                    if len(new_path) <= max_hops:
                        queue.append((neighbor, new_path, new_preds))

        chains.sort(key=lambda c: (len(c["novel_combo"]), c["gap"], c["length"]),
                    reverse=True)
        self._chains = chains
        return chains

    def print_chain_report(self, top_n=20):
        """Print assertion chain discoveries."""
        chains = getattr(self, "_chains", [])
        print(f"\n{'=' * 70}")
        print(f"ASSERTION CHAINS — {len(chains):,} found")
        print(f"Predicate combinations implied by chaining but not stated in any single paper")
        print("=" * 70)

        for c in chains[:top_n]:
            print(f"\n  Chain length: {c['length']}, gap: {c['gap']}")
            print(f"  Novel combination: {c['novel_combo']}")
            print(f"  Endpoint domains: {c['cats_start'][:3]} ↔ {c['cats_end'][:3]}")
            # Show path with parent titles
            for pid in c["path"]:
                title = ""
                for node in self.assertions.values():
                    if node.parent_id == pid:
                        title = node.parent_title
                        break
                print(f"    → {pid[:30]:30s}  {title[:40]}")

    # ── REPORTING ──────────────────────────────────────────

    def stats(self):
        """Print graph stats."""
        print(f"Assertions:     {len(self.assertions):,}")
        print(f"Signatures:     {len(self.signature_index):,}")
        print(f"Predicates:     {len(self.predicate_index):,}")

        print(f"\nPredicate frequency:")
        for pred, aids in sorted(self.predicate_index.items(),
                                  key=lambda x: -len(x[1]))[:20]:
            print(f"  {len(aids):6,}  {pred}")

        print(f"\nLargest signature groups:")
        for sig, aids in sorted(self.signature_index.items(),
                                 key=lambda x: -len(x[1]))[:15]:
            print(f"  {len(aids):6,}  {sig[:70]}")

    def print_report(self, top_n=30):
        """Print transpose candidates."""
        candidates = self._candidates
        print("\n" + "=" * 70)
        print(f"ASSERTION TRANSPOSES — {len(candidates):,} found")
        print("=" * 70)

        shown = 0
        seen_sigs = set()
        for c in candidates:
            if c["signature"] in seen_sigs:
                continue
            seen_sigs.add(c["signature"])

            preds = c["predicates"]
            quants = c["quantifiers"]

            print(f"\n  CLAIM: {' ∧ '.join(preds)}")
            if quants:
                print(f"  QUANT: {', '.join(quants)}")
            print(f"    {c['parent_a'][:55]}")
            print(f"      [{', '.join(c['unique_a'][:3])}]")
            print(f"    ↔ {c['parent_b'][:55]}")
            print(f"      [{', '.join(c['unique_b'][:3])}]")
            print(f"    gap={c['gap']}, assertions with this signature={c['n_assertions']}, groups={c['n_groups']}")

            shown += 1
            if shown >= top_n:
                break


if __name__ == "__main__":
    ag = AssertionGraph()

    print("Ingesting NaturalProofs...")
    ag.ingest_naturalproofs(
        "knowledge-graph/sources/naturalproofs/naturalproofs_proofwiki.json"
    )

    print("Ingesting YAML proofs...")
    ag.ingest_yaml_proofs("knowledge-graph/nodes/math/proofs/")

    print()
    ag.stats()

    # Hub analysis
    print("\nScoring hubs...")
    ag.hub_scores()
    ag.signature_hubs()
    ag.print_hub_report(top_n=15)

    # Chains
    print("\nFinding assertion chains...")
    ag.find_chains(max_hops=3, min_predicates_per_link=2)
    ag.print_chain_report(top_n=20)
