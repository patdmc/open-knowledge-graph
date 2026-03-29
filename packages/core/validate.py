#!/usr/bin/env python3
"""
VALIDATOR — single agent that all other hooks call.

Progressive disclosure: loads only the slice of the knowledge graph
relevant to the claim being validated. No context blowup.

Three faces:
  - Peer review (researcher): is this claim supported by the graph?
  - Editor (writer): is this assertion consistent with existing edges?
  - Index validation (librarian): is this edge/node properly defined?

USAGE:
  from validate import Validator
  v = Validator()

  # Peer review: check a biological claim
  result = v.peer_review("BRCA1 recruits RAD51 to DSBs")

  # Index check: is this edge well-formed?
  result = v.check_edge("BIO-CH01-DDR", "BRCA1", "RAD51C", "COMPENSATES")

  # Gap detection: what does the LLM know that the graph doesn't?
  gaps = v.detect_gaps(claims_list)
"""

import os
import re
import yaml
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGES_ROOT = REPO_ROOT / "packages"
BIO_DIR = PACKAGES_ROOT / "bio"
CACHE_DIR = PACKAGES_ROOT / "core" / "cache"
GAPS_DIR = PACKAGES_ROOT / "core" / "gaps"

for d in [CACHE_DIR, GAPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LAZY GRAPH LOADER — progressive disclosure
# ---------------------------------------------------------------------------
class GraphSlice:
    """Loads only the portion of the knowledge graph needed for validation.
    Caches loaded files so repeated queries in the same session are fast."""

    def __init__(self):
        self._nodes = {}       # node_id → data
        self._genes = {}       # gene_name → {channel, node_id, data}
        self._edges = {}       # (from, to) → [edge_dicts]
        self._loaded_files = set()

    def _load_file(self, path):
        """Load a single YAML file into the graph slice."""
        if str(path) in self._loaded_files:
            return
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception:
            return
        if not isinstance(data, dict) or "id" not in data:
            return

        nid = data["id"]
        self._nodes[nid] = data
        self._loaded_files.add(str(path))

        # Index genes
        channel = data.get("channel", data.get("name", ""))
        for gene in data.get("genes", []):
            if isinstance(gene, dict) and "name" in gene:
                self._genes[gene["name"]] = {
                    "channel": channel,
                    "node_id": nid,
                    "gene_data": gene,
                }

        # Index edges
        for edge in data.get("edges", []):
            if isinstance(edge, dict):
                fr = edge.get("from", nid)
                to = edge.get("to", "")
                if fr and to:
                    self._edges.setdefault((fr, to), []).append(edge)

        for edge in data.get("cross_channel_edges", []):
            if isinstance(edge, dict):
                fr = edge.get("from_gene", "")
                to = edge.get("to_gene", "")
                if fr and to:
                    self._edges.setdefault((fr, to), []).append(edge)

    def load_channel(self, channel_name):
        """Load a specific channel file."""
        for p in BIO_DIR.glob("**/BIO-CH*.yaml"):
            if channel_name.lower() in p.stem.lower():
                self._load_file(p)
                return

    def load_gene(self, gene_name):
        """Load whichever channel file contains this gene."""
        if gene_name in self._genes:
            return  # already loaded
        # Scan all channel files for this gene
        for p in BIO_DIR.glob("**/BIO-CH*.yaml"):
            self._load_file(p)
            if gene_name in self._genes:
                return

    def load_all_biology(self):
        """Load everything — only when needed."""
        for p in BIO_DIR.glob("**/*.yaml"):
            self._load_file(p)

    def get_gene(self, name):
        """Get gene info, loading lazily if needed."""
        self.load_gene(name)
        return self._genes.get(name)

    def get_edge(self, from_node, to_node):
        """Check if an edge exists between two entities."""
        return self._edges.get((from_node, to_node), [])

    def get_gene_interactions(self, gene_name):
        """Get all known interactions for a gene."""
        self.load_gene(gene_name)
        info = self._genes.get(gene_name)
        if not info:
            return []
        gene_data = info["gene_data"]
        interactions = []
        for partner in gene_data.get("interacts_with", []):
            if isinstance(partner, dict):
                interactions.append({
                    "type": "INTERACTS_WITH",
                    "partner": partner.get("gene", ""),
                    "score": partner.get("string_score", 0),
                })
        for partner in gene_data.get("compensates", []):
            if isinstance(partner, dict):
                interactions.append({
                    "type": "COMPENSATES",
                    "partner": partner.get("gene", ""),
                    "source": partner.get("source", ""),
                })
        return interactions

    @property
    def all_gene_names(self):
        """All gene names currently loaded."""
        return set(self._genes.keys())


# ---------------------------------------------------------------------------
# CLAIM PARSER — extract structured claims from text
# ---------------------------------------------------------------------------

# Biological verb patterns
BIO_VERBS = [
    "phosphorylates", "ubiquitinates", "methylates", "acetylates",
    "binds", "recruits", "activates", "inhibits", "stabilizes", "degrades",
    "translocates", "localizes", "exports", "imports",
    "transcribes", "represses", "upregulates", "downregulates",
    "repairs", "remodels", "scaffolds", "dimerizes",
    "compensates", "cooperates", "requires", "drives",
    "interacts with", "is recruited to", "is activated by",
]

# Gene name pattern — uppercase letters/numbers, 2-10 chars
GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?)\b')


def extract_claims(text):
    """Extract biological claims from text.

    Returns list of dicts: {subject, verb, object, raw_text}
    Progressive: only parses what looks like a biological assertion.
    """
    claims = []
    sentences = re.split(r'[.;]\s+', text)

    for sent in sentences:
        # Find gene-like tokens
        genes_in_sent = GENE_PATTERN.findall(sent)
        if len(genes_in_sent) < 2:
            continue

        # Find biological verbs
        sent_lower = sent.lower()
        for verb in BIO_VERBS:
            if verb in sent_lower:
                # Try to find subject and object around the verb
                idx = sent_lower.index(verb)
                before = sent[:idx]
                after = sent[idx + len(verb):]

                subjects = GENE_PATTERN.findall(before)
                objects = GENE_PATTERN.findall(after)

                if subjects and objects:
                    claims.append({
                        "subject": subjects[-1],  # closest to verb
                        "verb": verb,
                        "object": objects[0],      # closest to verb
                        "raw_text": sent.strip(),
                    })
                break  # one verb per sentence

    return claims


# ---------------------------------------------------------------------------
# VALIDATOR — the three faces
# ---------------------------------------------------------------------------
class Validator:
    """Single validator agent. Progressive disclosure — loads only what
    each validation request needs."""

    def __init__(self):
        self.graph = GraphSlice()

    # --- FACE 1: Peer Review (for researcher) ---

    def peer_review(self, claim_text):
        """Check a biological claim against the knowledge graph.

        Returns:
          {
            "status": "confirmed" | "novel" | "contradicted" | "no_data",
            "evidence": [...],
            "claim": {...},
            "needs_source": bool,
          }
        """
        claims = extract_claims(claim_text)
        if not claims:
            return {
                "status": "no_data",
                "evidence": [],
                "claim": {"raw_text": claim_text},
                "needs_source": False,
            }

        results = []
        for claim in claims:
            result = self._check_single_claim(claim)
            results.append(result)

        # Aggregate: worst status wins
        statuses = [r["status"] for r in results]
        if "contradicted" in statuses:
            overall = "contradicted"
        elif "novel" in statuses:
            overall = "novel"
        elif "confirmed" in statuses:
            overall = "confirmed"
        else:
            overall = "no_data"

        return {
            "status": overall,
            "evidence": results,
            "claim": {"raw_text": claim_text},
            "needs_source": overall == "novel",
        }

    def _check_single_claim(self, claim):
        """Check one structured claim against the graph."""
        subj = claim["subject"]
        obj = claim["object"]
        verb = claim["verb"]

        # Load relevant genes
        subj_info = self.graph.get_gene(subj)
        obj_info = self.graph.get_gene(obj)

        if not subj_info and not obj_info:
            return {
                "status": "no_data",
                "claim": claim,
                "reason": f"Neither {subj} nor {obj} found in knowledge graph",
            }

        # Check for existing edges between these genes
        interactions = self.graph.get_gene_interactions(subj)
        matching = [i for i in interactions if i["partner"] == obj]

        if matching:
            return {
                "status": "confirmed",
                "claim": claim,
                "evidence": matching,
                "reason": f"Edge exists: {subj} → {obj} ({matching[0]['type']})",
            }

        # Check reverse direction
        interactions_rev = self.graph.get_gene_interactions(obj)
        matching_rev = [i for i in interactions_rev if i["partner"] == subj]

        if matching_rev:
            return {
                "status": "confirmed",
                "claim": claim,
                "evidence": matching_rev,
                "reason": f"Edge exists: {obj} → {subj} ({matching_rev[0]['type']})",
            }

        # Both genes exist but no edge — this is a novel claim
        if subj_info and obj_info:
            same_channel = subj_info["channel"] == obj_info["channel"]
            return {
                "status": "novel",
                "claim": claim,
                "reason": (
                    f"Both genes in graph ({subj}: {subj_info['channel']}, "
                    f"{obj}: {obj_info['channel']}) but no edge. "
                    f"{'Same channel.' if same_channel else 'Cross-channel.'}"
                ),
                "needs_source": True,
            }

        # One gene missing
        missing = subj if not subj_info else obj
        return {
            "status": "no_data",
            "claim": claim,
            "reason": f"{missing} not found in knowledge graph",
        }

    # --- FACE 2: Editor (for writer) ---

    def check_consistency(self, assertions):
        """Check a list of assertions for internal consistency
        and consistency with the graph.

        Args:
            assertions: list of strings (biological claims)

        Returns:
            list of {claim, status, conflicts}
        """
        results = []
        seen_claims = []

        for text in assertions:
            # Check against graph
            review = self.peer_review(text)

            # Check against previous assertions in this batch
            claims = extract_claims(text)
            conflicts = []
            for claim in claims:
                for prev in seen_claims:
                    if (claim["subject"] == prev["subject"] and
                            claim["object"] == prev["object"] and
                            claim["verb"] != prev["verb"]):
                        conflicts.append({
                            "type": "verb_conflict",
                            "this": claim,
                            "previous": prev,
                        })
                seen_claims.extend(claims)

            results.append({
                "text": text,
                "graph_status": review["status"],
                "needs_source": review.get("needs_source", False),
                "conflicts": conflicts,
            })

        return results

    # --- FACE 3: Index Validation (for librarian) ---

    def check_edge(self, source_node, from_gene, to_gene, edge_type):
        """Validate that a proposed edge is well-formed and non-duplicate.

        Returns:
            {
                "valid": bool,
                "issues": [...],
                "duplicate": bool,
            }
        """
        issues = []

        # Check source node exists
        self.graph.load_all_biology()
        if source_node not in self.graph._nodes:
            issues.append(f"Source node {source_node} not found")

        # Check genes exist
        from_info = self.graph.get_gene(from_gene)
        to_info = self.graph.get_gene(to_gene)

        if not from_info:
            issues.append(f"Gene {from_gene} not in knowledge graph")
        if not to_info:
            issues.append(f"Gene {to_gene} not in knowledge graph")

        # Check edge type is valid
        valid_types = {
            "INTERACTS_WITH", "COMPENSATES", "REGULATES",
            "ACTIVATES", "INHIBITS", "RECRUITS", "BINDS",
            "PHOSPHORYLATES", "UBIQUITINATES",
        }
        if edge_type not in valid_types:
            issues.append(f"Edge type {edge_type} not in valid set: {valid_types}")

        # Check for duplicates
        existing = self.graph.get_edge(from_gene, to_gene)
        duplicate = len(existing) > 0

        if duplicate:
            issues.append(f"Edge {from_gene} → {to_gene} already exists: {existing}")

        return {
            "valid": len(issues) == 0 and not duplicate,
            "issues": issues,
            "duplicate": duplicate,
        }

    # --- GAP DETECTION ---

    def detect_gaps(self, claims_text):
        """Given a block of text (e.g., LLM response), find claims
        that are novel — in the LLM's knowledge but not in the graph.

        Returns list of gap dicts ready for precipitation review.
        """
        claims = extract_claims(claims_text)
        gaps = []

        for claim in claims:
            result = self._check_single_claim(claim)
            if result["status"] == "novel":
                gaps.append({
                    "claim": claim,
                    "reason": result["reason"],
                    "source": "NEEDS_CITATION",
                    "confidence": 0.0,  # zero until source provided
                })

        return gaps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  validate.py review <text>       — peer review a claim")
        print("  validate.py gaps <text>          — find novel claims")
        print("  validate.py edge <node> <from> <to> <type> — check edge")
        print("  validate.py stats                — graph coverage stats")
        sys.exit(1)

    cmd = sys.argv[1]
    v = Validator()

    if cmd == "review":
        text = " ".join(sys.argv[2:])
        result = v.peer_review(text)
        print(json.dumps(result, indent=2))

    elif cmd == "gaps":
        text = " ".join(sys.argv[2:])
        gaps = v.detect_gaps(text)
        if gaps:
            print(f"Found {len(gaps)} novel claims:")
            for g in gaps:
                c = g["claim"]
                print(f"  {c['subject']} {c['verb']} {c['object']}")
                print(f"    Reason: {g['reason']}")
                print(f"    Source: {g['source']}")
        else:
            print("No novel claims detected.")

    elif cmd == "edge":
        if len(sys.argv) < 6:
            print("Usage: validate.py edge <node> <from> <to> <type>")
            sys.exit(1)
        result = v.check_edge(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        print(json.dumps(result, indent=2))

    elif cmd == "stats":
        v.graph.load_all_biology()
        print(f"Loaded nodes: {len(v.graph._nodes)}")
        print(f"Loaded genes: {len(v.graph._genes)}")
        print(f"Loaded edges: {len(v.graph._edges)}")
        print(f"Files loaded: {len(v.graph._loaded_files)}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
