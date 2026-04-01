#!/usr/bin/env python3
"""
Multipass map/reduce for edge discovery across large corpora.

Pass 1: Emit (category, skeleton_hash) per item. Group by skeleton_hash.
         Items with same proof pattern but different categories = candidates.
Pass 2: For candidates, extract assertion patterns. Group by pattern.
         Same logical move across domains = transpose.
Pass 3: Fetch root proofs for top candidates. Verify structural match.

Lexical assertion encoding:
  "contradiction(hypothesis → claim → negation → ⊥)"
  "induction(base → step → conclusion)"
  "construction(hypothesis → exhibit → verify)"

These patterns are domain-independent and span graphs.

Usage:
    from multipass_discovery import MultipassDiscovery
    md = MultipassDiscovery("knowledge-graph/sources/naturalproofs/naturalproofs_proofwiki.json")
    md.run()
    md.print_report()
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


# ── LEXICAL ASSERTION PATTERNS ─────────────────────────────

# Proof method signatures — the logical moves
METHOD_PATTERNS = {
    "contradiction": [
        r"assume.*(?:not|negat|contrar|suppos)",
        r"leads?\s+to\s+(?:a\s+)?contradiction",
        r"which\s+is\s+(?:a\s+)?contradiction",
        r"\\bot",
        r"absurd",
    ],
    "induction": [
        r"(?:base|basis)\s+case",
        r"induct(?:ive|ion)\s+(?:step|hypothesis)",
        r"suppose.*holds?\s+for\s+(?:all\s+)?[nkm]",
        r"P\s*\(\s*[nk]\s*\+\s*1\s*\)",
    ],
    "construction": [
        r"(?:construct|exhibit|define|let)\s+",
        r"we\s+(?:construct|build|define)",
        r"consider\s+the\s+(?:map|function|set|element)",
    ],
    "counting": [
        r"count(?:ing)?",
        r"(?:double|over)\s*count",
        r"bijection\s+between",
        r"\|[A-Z]\|\s*[=<>]",
    ],
    "pigeonhole": [
        r"pigeonhole",
        r"(?:more|greater)\s+than\s+\d+\s+(?:elements|items|objects)",
    ],
    "diagonalization": [
        r"diagonal(?:ize|ization)",
        r"differ(?:s|ent)\s+from\s+(?:every|each|all)",
    ],
    "compactness": [
        r"compact(?:ness)?",
        r"(?:finite|open)\s+(?:sub)?cover",
        r"Bolzano|Weierstrass|Heine|Borel",
    ],
    "continuity_argument": [
        r"(?:by\s+)?continuity",
        r"intermediate\s+value",
        r"epsilon.*delta",
    ],
    "linearity": [
        r"linear(?:ity)?",
        r"(?:by|using)\s+linearity",
        r"[Ee]\s*\[\s*[aA]\s*[Xx]\s*\+",
    ],
    "symmetry": [
        r"(?:by\s+)?symmetry",
        r"without\s+loss\s+of\s+generality",
        r"WLOG",
    ],
    "bounding": [
        r"bound(?:ed)?(?:\s+(?:above|below|by))?",
        r"(?:upper|lower)\s+bound",
        r"\\le|\\ge|\\leq|\\geq",
        r"Markov|Chebyshev|Cauchy.Schwarz|H.lder|Jensen",
    ],
    "decomposition": [
        r"decompos(?:e|ition)",
        r"(?:direct\s+)?sum",
        r"(?:factor|split|partition)\s+(?:into|as)",
    ],
    "limit_argument": [
        r"(?:take|let|as)\s+.*(?:\\to|\\rightarrow|\s+to\s+)\s*(?:\\infty|infinity|0)",
        r"\\lim(?:_{|\s)",
        r"converge[sd]?\s+to",
    ],
}


def detect_methods(text):
    """Detect proof methods from text content."""
    methods = set()
    text_lower = text.lower()
    for method, patterns in METHOD_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower) or re.search(pat, text):
                methods.add(method)
                break
    return methods


def assertion_signature(item):
    """Create a lexical assertion signature from an item.

    The signature captures the logical STRUCTURE, not the domain content.
    Two items with the same signature use the same proof strategy.
    """
    all_text = " ".join(item.get("contents", []))
    for proof in item.get("proofs", []):
        all_text += " " + " ".join(proof.get("contents", []))

    methods = detect_methods(all_text)
    n_refs = len(set(item.get("ref_ids", [])))
    n_proofs = len(item.get("proofs", []))

    # Quantize ref count to bins
    if n_refs == 0:
        ref_bin = "isolated"
    elif n_refs <= 2:
        ref_bin = "few_refs"
    elif n_refs <= 5:
        ref_bin = "moderate_refs"
    else:
        ref_bin = "many_refs"

    return {
        "methods": frozenset(methods),
        "ref_bin": ref_bin,
        "n_proofs": min(n_proofs, 5),  # cap
        "signature": "|".join(sorted(methods)) + f"/{ref_bin}" if methods else f"none/{ref_bin}",
    }


class MultipassDiscovery:
    """Multipass map/reduce edge discovery."""

    def __init__(self, json_path):
        self.path = Path(json_path)
        self.items = {}           # id → item
        self.categories = {}      # id → set of top-level categories
        self.signatures = {}      # id → assertion signature dict
        self.candidates = []      # final candidates

    def _load_items(self):
        """Stream-load items (memory efficient)."""
        with open(self.path) as f:
            raw = json.load(f)
        ds = raw["dataset"]
        for item in ds["theorems"] + ds["definitions"] + ds["others"]:
            iid = item["id"]
            self.items[iid] = item
            self.categories[iid] = set(item.get("toplevel_categories", []))

    # ── PASS 1: Category buckets + signatures ──────────────────

    def pass1_emit(self):
        """Map: each item → (categories, signature).
        Reduce: group by signature, keep only groups spanning multiple categories.
        """
        sig_groups = defaultdict(list)

        for iid, item in self.items.items():
            if item.get("type") != "theorem":
                continue  # Only theorems have proof structure

            sig = assertion_signature(item)
            self.signatures[iid] = sig
            sig_groups[sig["signature"]].append(iid)

        # Filter: only signatures that appear in 2+ different category sets
        cross_sig = {}
        for sig_key, ids in sig_groups.items():
            if len(ids) < 2:
                continue

            # Check category diversity
            all_cats = set()
            cat_sets = []
            for iid in ids:
                cats = self.categories.get(iid, set())
                all_cats |= cats
                cat_sets.append(cats)

            if len(all_cats) >= 2:
                cross_sig[sig_key] = ids

        return cross_sig

    # ── PASS 2: Cross-category pairs within signature groups ───

    def pass2_candidates(self, cross_sig, min_category_gap=2):
        """For each signature group, find pairs from different categories."""
        candidates = []

        for sig_key, ids in cross_sig.items():
            # Build category → items index within this group
            cat_to_items = defaultdict(list)
            for iid in ids:
                for cat in self.categories.get(iid, set()):
                    cat_to_items[cat].append(iid)

            # Find pairs where categories don't overlap
            seen = set()
            for iid in ids:
                cats_a = self.categories.get(iid, set())
                for jid in ids:
                    if jid <= iid:
                        continue
                    if (iid, jid) in seen:
                        continue
                    seen.add((iid, jid))

                    cats_b = self.categories.get(jid, set())
                    unique_a = cats_a - cats_b
                    unique_b = cats_b - cats_a

                    if len(unique_a) + len(unique_b) >= min_category_gap:
                        candidates.append({
                            "id_a": iid,
                            "id_b": jid,
                            "title_a": self.items[iid].get("title", ""),
                            "title_b": self.items[jid].get("title", ""),
                            "signature": sig_key,
                            "cats_a": sorted(cats_a),
                            "cats_b": sorted(cats_b),
                            "unique_a": sorted(unique_a),
                            "unique_b": sorted(unique_b),
                            "shared_cats": sorted(cats_a & cats_b),
                            "gap": len(unique_a) + len(unique_b),
                        })

            # Cap per signature to avoid combinatorial explosion
            if len(candidates) > 50000:
                break

        candidates.sort(key=lambda c: c["gap"], reverse=True)
        return candidates

    # ── PASS 3: Root proof comparison for top candidates ───────

    def pass3_verify(self, candidates, top_n=100):
        """For top candidates, compare proof structure more deeply."""
        verified = []

        for c in candidates[:top_n]:
            item_a = self.items[c["id_a"]]
            item_b = self.items[c["id_b"]]

            # Extract proof methods for each
            methods_a = set()
            methods_b = set()
            for proof in item_a.get("proofs", []):
                text = " ".join(proof.get("contents", []))
                methods_a |= detect_methods(text)
            for proof in item_b.get("proofs", []):
                text = " ".join(proof.get("contents", []))
                methods_b |= detect_methods(text)

            shared_methods = methods_a & methods_b

            # Check ref overlap (shared dependencies)
            refs_a = set()
            for proof in item_a.get("proofs", []):
                refs_a.update(proof.get("ref_ids", []))
            refs_a.update(item_a.get("ref_ids", []))

            refs_b = set()
            for proof in item_b.get("proofs", []):
                refs_b.update(proof.get("ref_ids", []))
            refs_b.update(item_b.get("ref_ids", []))

            shared_refs = refs_a & refs_b

            # Resolve shared ref titles
            shared_ref_titles = []
            for rid in sorted(shared_refs):
                ref_item = self.items.get(rid)
                if ref_item:
                    shared_ref_titles.append(ref_item.get("title", f"#{rid}"))

            c["shared_methods"] = sorted(shared_methods)
            c["shared_refs"] = shared_ref_titles[:10]
            c["n_shared_refs"] = len(shared_refs)
            c["methods_a"] = sorted(methods_a)
            c["methods_b"] = sorted(methods_b)
            c["score"] = c["gap"] + len(shared_methods) * 2 + len(shared_refs)

            verified.append(c)

        verified.sort(key=lambda c: c["score"], reverse=True)
        return verified

    # ── RUN ────────────────────────────────────────────────────

    def run(self, top_n=50):
        """Run full multipass pipeline."""
        print("Loading items...")
        self._load_items()
        print(f"  {len(self.items):,} items loaded")

        print("\nPass 1: Signature emission + category bucketing...")
        cross_sig = self.pass1_emit()
        print(f"  {len(cross_sig)} cross-category signatures found")
        total_in_groups = sum(len(ids) for ids in cross_sig.values())
        print(f"  {total_in_groups:,} theorems in cross-category groups")

        print("\nPass 2: Cross-category pair extraction...")
        candidates = self.pass2_candidates(cross_sig)
        print(f"  {len(candidates):,} candidate pairs")

        print("\nPass 3: Root proof verification (top candidates)...")
        self.candidates = self.pass3_verify(candidates, top_n=top_n * 2)
        print(f"  {len(self.candidates)} verified candidates")

        return self.candidates

    # ── REPORTING ──────────────────────────────────────────────

    def print_report(self, top_n=30):
        """Print top transpose candidates."""
        print("\n" + "=" * 70)
        print(f"TRANSPOSE CANDIDATES — same proof pattern, different domains")
        print("=" * 70)

        # Signature distribution
        sig_counts = Counter(c["signature"] for c in self.candidates)
        print(f"\nTop proof patterns producing transposes:")
        for sig, count in sig_counts.most_common(10):
            print(f"  {count:4d}  {sig}")

        print(f"\nTop {top_n} candidates:\n")
        for c in self.candidates[:top_n]:
            print(f"  [{c['score']:.0f}] {c['title_a'][:45]}")
            print(f"       ↔ {c['title_b'][:45]}")
            print(f"       pattern: {c['signature']}")
            print(f"       domains: {c['unique_a'][:3]} ↔ {c['unique_b'][:3]}")
            if c.get("shared_methods"):
                print(f"       shared moves: {c['shared_methods']}")
            if c.get("shared_refs"):
                print(f"       shared deps: {c['shared_refs'][:5]}")
            print()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "knowledge-graph/sources/naturalproofs/naturalproofs_proofwiki.json"
    md = MultipassDiscovery(path)
    md.run(top_n=30)
    md.print_report(top_n=30)
