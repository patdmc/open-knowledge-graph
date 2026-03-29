#!/usr/bin/env python3
"""
PRECIPITATE — convert validated claims into durable graph edges.

The ratchet: once a claim is validated with a source, it becomes
a permanent edge in the knowledge graph YAML files. No going back
without explicit provenance of why.

USAGE:
  from precipitate import Precipitator
  p = Precipitator()
  p.add_edge("BRCA1", "RAD51", "RECRUITS",
             source="PMID:12345678",
             context="ddr",
             confidence=0.9)

CLI:
  python precipitate.py add <from> <to> <type> --source <src> --context <ctx>
  python precipitate.py review               — show pending gaps
  python precipitate.py promote <gap_id>     — promote a gap to edge
"""

import json
import os
import yaml
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGES_ROOT = REPO_ROOT / "packages"
BIO_DIR = PACKAGES_ROOT / "bio"
GAPS_DIR = PACKAGES_ROOT / "core" / "gaps"
GAPS_FILE = GAPS_DIR / "pending_gaps.json"

GAPS_DIR.mkdir(parents=True, exist_ok=True)


def _load_gaps():
    if GAPS_FILE.exists():
        with open(GAPS_FILE) as f:
            return json.load(f)
    return []


def _save_gaps(gaps):
    tmp = str(GAPS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(gaps, f, indent=2)
    os.replace(tmp, str(GAPS_FILE))


def _find_channel_file(gene_name):
    """Find which channel YAML file contains a gene."""
    for p in BIO_DIR.glob("**/BIO-CH*.yaml"):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            for gene in data.get("genes", []):
                if isinstance(gene, dict) and gene.get("name") == gene_name:
                    return p, data
        except Exception:
            continue
    return None, None


class Precipitator:
    """Converts validated claims into durable graph edges."""

    def save_gap(self, claim, reason, source="NEEDS_CITATION"):
        """Save a novel claim as a pending gap for review."""
        gaps = _load_gaps()
        gap = {
            "id": len(gaps),
            "subject": claim.get("subject", ""),
            "verb": claim.get("verb", ""),
            "object": claim.get("object", ""),
            "raw_text": claim.get("raw_text", ""),
            "reason": reason,
            "source": source,
            "confidence": 0.0,
            "detected": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        gaps.append(gap)
        _save_gaps(gaps)
        return gap

    def review_gaps(self):
        """Return all pending gaps."""
        gaps = _load_gaps()
        return [g for g in gaps if g.get("status") == "pending"]

    def promote_gap(self, gap_id, source, confidence=0.8):
        """Promote a pending gap to a confirmed edge.

        Args:
            gap_id: integer gap ID
            source: citation (e.g., "PMID:12345678")
            confidence: float 0-1
        """
        gaps = _load_gaps()
        gap = None
        for g in gaps:
            if g["id"] == gap_id:
                gap = g
                break

        if not gap:
            raise ValueError(f"Gap {gap_id} not found")

        if gap["status"] != "pending":
            raise ValueError(f"Gap {gap_id} is {gap['status']}, not pending")

        # Precipitate into the graph
        success = self.add_edge(
            from_gene=gap["subject"],
            to_gene=gap["object"],
            edge_type=self._verb_to_edge_type(gap["verb"]),
            source=source,
            confidence=confidence,
        )

        if success:
            gap["status"] = "precipitated"
            gap["source"] = source
            gap["confidence"] = confidence
            gap["precipitated"] = datetime.now(timezone.utc).isoformat()
            _save_gaps(gaps)
            return True

        return False

    def reject_gap(self, gap_id, reason=""):
        """Reject a pending gap — mark it as reviewed and dismissed."""
        gaps = _load_gaps()
        for g in gaps:
            if g["id"] == gap_id:
                g["status"] = "rejected"
                g["rejection_reason"] = reason
                g["rejected"] = datetime.now(timezone.utc).isoformat()
                _save_gaps(gaps)
                return True
        return False

    def add_edge(self, from_gene, to_gene, edge_type, source,
                 context="", confidence=0.8):
        """Add a new edge to the appropriate channel YAML file.

        Finds the channel file containing from_gene and adds
        the edge with full provenance.
        """
        path, data = _find_channel_file(from_gene)
        if not path or not data:
            print(f"WARNING: {from_gene} not found in any channel file")
            return False

        # Find the gene entry
        for gene in data.get("genes", []):
            if isinstance(gene, dict) and gene.get("name") == from_gene:
                # Add to cross_channel_edges or interacts_with
                edge = {
                    "to_gene": to_gene,
                    "type": edge_type,
                    "source": source,
                    "confidence": confidence,
                    "provenance": "knowledge_graph:hook:precipitate",
                    "added": datetime.now(timezone.utc).isoformat(),
                }
                if context:
                    edge["context"] = context

                # Determine where to put it
                if edge_type in ("INTERACTS_WITH",):
                    gene.setdefault("interacts_with", []).append({
                        "gene": to_gene,
                        "string_score": 0,
                        "source": source,
                        "provenance": "precipitated",
                    })
                elif edge_type in ("COMPENSATES",):
                    gene.setdefault("compensates", []).append({
                        "gene": to_gene,
                        "source": source,
                        "provenance": "precipitated",
                    })
                else:
                    # General edge — add to cross_channel_edges
                    data.setdefault("cross_channel_edges", []).append({
                        "from_gene": from_gene,
                        "to_gene": to_gene,
                        "type": edge_type,
                        "source": source,
                        "confidence": confidence,
                        "provenance": "precipitated",
                        "added": datetime.now(timezone.utc).isoformat(),
                    })

                # Write back
                with open(path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False,
                              allow_unicode=True, sort_keys=False)

                print(f"Precipitated: {from_gene} --[{edge_type}]--> {to_gene}")
                print(f"  Source: {source}")
                print(f"  File: {path.name}")
                return True

        return False

    @staticmethod
    def _verb_to_edge_type(verb):
        """Map a natural language verb to an edge type."""
        mapping = {
            "phosphorylates": "PHOSPHORYLATES",
            "ubiquitinates": "UBIQUITINATES",
            "methylates": "METHYLATES",
            "acetylates": "ACETYLATES",
            "binds": "BINDS",
            "recruits": "RECRUITS",
            "activates": "ACTIVATES",
            "inhibits": "INHIBITS",
            "stabilizes": "STABILIZES",
            "degrades": "DEGRADES",
            "compensates": "COMPENSATES",
            "cooperates": "COOPERATES",
            "interacts with": "INTERACTS_WITH",
            "is recruited to": "RECRUITS",
            "is activated by": "ACTIVATES",
            "represses": "REPRESSES",
            "upregulates": "ACTIVATES",
            "downregulates": "INHIBITS",
        }
        return mapping.get(verb, "RELATES_TO")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  precipitate.py review                    — show pending gaps")
        print("  precipitate.py promote <id> --source <s> — promote gap to edge")
        print("  precipitate.py reject <id> [reason]      — reject a gap")
        print("  precipitate.py add <from> <to> <type> --source <s>")
        sys.exit(1)

    cmd = sys.argv[1]
    p = Precipitator()

    if cmd == "review":
        gaps = p.review_gaps()
        if not gaps:
            print("No pending gaps.")
        else:
            print(f"{len(gaps)} pending gaps:\n")
            for g in gaps:
                print(f"  [{g['id']}] {g['subject']} {g['verb']} {g['object']}")
                print(f"       {g['reason']}")
                print(f"       Source: {g['source']}")
                print(f"       Detected: {g['detected']}")
                print()

    elif cmd == "promote":
        if len(sys.argv) < 3:
            print("Usage: precipitate.py promote <id> --source <source>")
            sys.exit(1)
        gap_id = int(sys.argv[2])
        source = "unknown"
        for i, arg in enumerate(sys.argv):
            if arg == "--source" and i + 1 < len(sys.argv):
                source = sys.argv[i + 1]
        p.promote_gap(gap_id, source)

    elif cmd == "reject":
        gap_id = int(sys.argv[2])
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        p.reject_gap(gap_id, reason)

    elif cmd == "add":
        if len(sys.argv) < 5:
            print("Usage: precipitate.py add <from> <to> <type> --source <s>")
            sys.exit(1)
        from_gene = sys.argv[2]
        to_gene = sys.argv[3]
        edge_type = sys.argv[4]
        source = "manual"
        context = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--source" and i + 1 < len(sys.argv):
                source = sys.argv[i + 1]
            if arg == "--context" and i + 1 < len(sys.argv):
                context = sys.argv[i + 1]
        p.add_edge(from_gene, to_gene, edge_type, source, context)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
