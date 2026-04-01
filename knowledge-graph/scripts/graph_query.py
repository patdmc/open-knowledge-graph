#!/usr/bin/env python3
"""Graph query interface for the knowledge graph.

Translates natural-language benchmark questions into graph traversals.
Returns structured subgraphs that an LLM can format into answers
using ~90% fewer tokens than pure context-stuffing.

Usage:
    python3 graph_query.py "What channel is BRCA1 in?"
    python3 graph_query.py "What are SL partners of RAD51C?"
    python3 graph_query.py "Which cancer type has the worst survival?"
"""

import sys
import re
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NODES_DIR = ROOT / "knowledge-graph" / "nodes"


class KnowledgeGraph:
    """In-memory graph loaded from YAML node files."""

    def __init__(self):
        self.nodes = {}       # id -> full node dict
        self.genes = {}       # gene_symbol -> {channel, degree, position, mutations, cross_edges}
        self.channels = {}    # channel_id -> node dict
        self.cancer_types = {}  # ct_id -> node dict
        self.empirical = {}   # emp_id -> node dict
        self._load()

    def _load(self):
        for yaml_path in NODES_DIR.rglob("*.yaml"):
            try:
                with open(yaml_path) as f:
                    node = yaml.safe_load(f)
            except Exception:
                continue
            if not node or "id" not in node:
                continue

            node_id = node["id"]
            self.nodes[node_id] = node

            # Index channel nodes and their embedded genes
            if node_id.startswith("CHAN-"):
                self.channels[node_id] = node
                for gene in node.get("genes", []):
                    gid = gene.get("id", "")
                    self.genes[gid] = {
                        "channel": node_id,
                        "channel_name": node.get("name", ""),
                        "string_ppi_degree": gene.get("string_ppi_degree", 0),
                        "position": gene.get("position", "leaf"),
                        "mutations": gene.get("mutations", []),
                        "cross_edges": gene.get("cross_edges", []),
                    }

            elif node_id.startswith("CT-"):
                self.cancer_types[node_id] = node

            elif node_id.startswith("EMP"):
                self.empirical[node_id] = node

    # --- Query Methods ---

    def gene_channel(self, gene_symbol):
        """What channel is gene X in?"""
        gene = self.genes.get(gene_symbol)
        if not gene:
            return None
        return {
            "gene": gene_symbol,
            "channel": gene["channel"],
            "channel_name": gene["channel_name"],
            "position": gene["position"],
            "string_ppi_degree": gene["string_ppi_degree"],
        }

    def gene_partners(self, gene_symbol, relation=None):
        """What are the interaction partners of gene X?"""
        gene = self.genes.get(gene_symbol)
        if not gene:
            return []
        edges = gene.get("cross_edges", [])
        if relation:
            edges = [e for e in edges if e.get("relation", "").upper() == relation.upper()]
        return [{"gene": gene_symbol, **e} for e in edges]

    def gene_mutations(self, gene_symbol):
        """What are the key mutations for gene X?"""
        gene = self.genes.get(gene_symbol)
        if not gene:
            return []
        return [{"gene": gene_symbol, **m} for m in gene.get("mutations", [])]

    def channel_genes(self, channel_id):
        """What genes are in channel X?"""
        channel = self.channels.get(channel_id)
        if not channel:
            # Try partial match
            for cid, cnode in self.channels.items():
                if channel_id.upper() in cid.upper() or channel_id.upper() in cnode.get("name", "").upper():
                    channel = cnode
                    break
        if not channel:
            return []
        return [g.get("id") for g in channel.get("genes", [])]

    def channel_hubs(self, channel_id):
        """Which genes are hubs in channel X?"""
        genes = self.channel_genes(channel_id)
        return [g for g in genes if self.genes.get(g, {}).get("position") == "hub"]

    def cancer_type_info(self, query):
        """Get cancer type node by name or abbreviation."""
        query_upper = query.upper().replace(" ", "")
        for ct_id, ct in self.cancer_types.items():
            if query_upper in ct_id.upper() or query_upper in ct.get("name", "").upper().replace(" ", ""):
                return ct
        return None

    def worst_survival(self, n=5):
        """Which cancer types have worst median survival?"""
        cts = []
        for ct_id, ct in self.cancer_types.items():
            mos = ct.get("median_os_months")
            if mos:
                cts.append({"id": ct_id, "name": ct.get("name"), "median_os_months": float(mos)})
        return sorted(cts, key=lambda x: x["median_os_months"])[:n]

    def best_survival(self, n=5):
        """Which cancer types have best median survival?"""
        cts = []
        for ct_id, ct in self.cancer_types.items():
            mos = ct.get("median_os_months")
            if mos:
                cts.append({"id": ct_id, "name": ct.get("name"), "median_os_months": float(mos)})
        return sorted(cts, key=lambda x: x["median_os_months"], reverse=True)[:n]

    def empirical_finding(self, query):
        """Find empirical findings by keyword."""
        results = []
        for emp_id, emp in self.empirical.items():
            text = f"{emp.get('name', '')} {emp.get('claim', '')}".lower()
            if query.lower() in text:
                results.append(emp)
        return results

    def subgraph(self, gene_symbol, depth=1):
        """Extract local subgraph around a gene (for LLM context)."""
        gene = self.genes.get(gene_symbol)
        if not gene:
            return None

        result = {
            "center": gene_symbol,
            "channel": gene["channel"],
            "position": gene["position"],
            "degree": gene["string_ppi_degree"],
            "mutations": gene["mutations"],
            "neighbors": [],
        }

        for edge in gene.get("cross_edges", []):
            partner = edge.get("to", "")
            partner_info = self.genes.get(partner, {})
            result["neighbors"].append({
                "gene": partner,
                "relation": edge.get("relation"),
                "hr": edge.get("hr"),
                "p": edge.get("p"),
                "channel": partner_info.get("channel", "unknown"),
            })

        return result

    def stats(self):
        """Graph statistics."""
        total_edges = sum(len(g.get("cross_edges", [])) for g in self.genes.values())
        total_mutations = sum(len(g.get("mutations", [])) for g in self.genes.values())
        return {
            "total_nodes": len(self.nodes),
            "channels": len(self.channels),
            "genes": len(self.genes),
            "cancer_types": len(self.cancer_types),
            "empirical_findings": len(self.empirical),
            "gene_cross_edges": total_edges,
            "gene_mutations": total_mutations,
        }


def parse_question(question, graph):
    """Route a natural language question to the right graph query."""
    q = question.lower().strip()

    # "What channel is X in?"
    m = re.search(r"what channel is (\w+) in", q)
    if m:
        return graph.gene_channel(m.group(1).upper())

    # "What are [SL/BUFFERS/...] partners of X?"
    m = re.search(r"(?:what are|list|show)(?: the)? (\w+)? ?partners (?:of|for) (\w+)", q)
    if m:
        rel = m.group(1).upper() if m.group(1) else None
        return graph.gene_partners(m.group(2).upper(), relation=rel)

    # "What genes are in [channel]?"
    m = re.search(r"(?:what |which )?genes (?:are )?in (\w+)", q)
    if m:
        return graph.channel_genes(f"CHAN-{m.group(1)}")

    # "What are hub genes in X?"
    m = re.search(r"hub(?:s| genes) (?:in|of|for) (\w+)", q)
    if m:
        return graph.channel_hubs(f"CHAN-{m.group(1)}")

    # "mutations of/in/for X"
    m = re.search(r"mutations? (?:of|in|for) (\w+)", q)
    if m:
        return graph.gene_mutations(m.group(1).upper())

    # "worst/best survival"
    if "worst" in q and "survival" in q:
        return graph.worst_survival()
    if "best" in q and "survival" in q:
        return graph.best_survival()

    # Cancer type lookup
    m = re.search(r"(?:about|info on|tell me about) (.+?)(?:\?|$)", q)
    if m:
        return graph.cancer_type_info(m.group(1).strip())

    # TMB / paradox / compensation
    for keyword in ["tmb", "paradox", "compensation", "compensat", "escalat", "synthetic lethal", "sl "]:
        if keyword in q:
            return graph.empirical_finding(keyword.strip())

    # Subgraph extraction
    m = re.search(r"subgraph (?:of|for|around) (\w+)", q)
    if m:
        return graph.subgraph(m.group(1).upper())

    # Stats
    if "stats" in q or "statistics" in q or "how many" in q:
        return graph.stats()

    return {"error": "Could not parse question. Try: 'What channel is BRCA1 in?' or 'stats'"}


def main():
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 graph_query.py \"<question>\"")
        print("\nExamples:")
        print("  python3 graph_query.py \"What channel is BRCA1 in?\"")
        print("  python3 graph_query.py \"What are BUFFERS partners of TP53?\"")
        print("  python3 graph_query.py \"Which genes are in DDR?\"")
        print("  python3 graph_query.py \"worst survival\"")
        print("  python3 graph_query.py \"stats\"")
        sys.exit(1)

    graph = KnowledgeGraph()
    question = " ".join(sys.argv[1:])
    result = parse_question(question, graph)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
