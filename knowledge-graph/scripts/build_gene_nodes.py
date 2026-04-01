#!/usr/bin/env python3
"""Build YAML channel nodes with embedded genes for the knowledge graph.

One file per channel. Genes, mutations, and cross-gene edges live inside
the channel node. Scales to 20,000 genes without file explosion.

Reads:
  - data/channel_gene_map.csv
  - analysis/mutation_survival_atlas.csv
  - analysis/results/string_degree/string_degree_grounding.json
  - analysis/results/comutation_shift/graph_edges.json
  - analysis/results/comutation_screen/compensates_edges.json
  - ../open-knowledge-graph-data/cancer/cache/ncbi_gene/ncbi_gene_summaries.json
  - ../open-knowledge-graph-data/cancer/cache/ncbi_gene/uniprot_annotations.json
  - ../open-knowledge-graph-data/cancer/cache/ncbi_gene/kegg_pathways.json

Writes:
  - knowledge-graph/nodes/biology/channels/CHAN-{name}.yaml (one per channel)
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHAN_DIR = ROOT / "knowledge-graph" / "nodes" / "biology" / "channels"
DATA_REPO = ROOT.parent / "open-knowledge-graph-data" / "cancer" / "cache" / "ncbi_gene"

CHANNEL_FULL_NAMES = {
    "DDR": "DNA Damage Response",
    "CellCycle": "Cell Cycle / Apoptosis",
    "PI3K_Growth": "PI3K / Growth Signaling",
    "Endocrine": "Endocrine / Hormone",
    "Immune": "Immune Surveillance",
    "TissueArchitecture": "Tissue Architecture",
    "ChromatinRemodel": "Chromatin Remodeling",
    "DNAMethylation": "DNA Methylation",
}


def load_channel_map():
    path = ROOT / "data" / "channel_gene_map.csv"
    gene_to_channel = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gene_to_channel[row["gene"]] = row["channel"]
    return gene_to_channel


def load_string_degrees():
    path = ROOT / "analysis" / "results" / "string_degree" / "string_degree_grounding.json"
    with open(path) as f:
        data = json.load(f)
    degrees = data.get("string_degrees", {})
    hub_comparison = data.get("hub_comparison", {})
    hubs = set()
    for channel_info in hub_comparison.values():
        for gene in channel_info.get("string_hubs", []):
            hubs.add(gene)
    return degrees, hubs


def load_mutation_atlas():
    path = ROOT / "analysis" / "mutation_survival_atlas.csv"
    gene_mutations = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gene = row["gene"]
            gene_mutations.setdefault(gene, []).append({
                "protein_change": row["protein_change"],
                "cancer_type": row["cancer_type"],
                "hr": float(row["hr"]),
                "p_value": float(row["p_value"]),
                "direction": row["direction"],
                "confident": row["confident"] == "True",
                "n_mutated": int(row["n_mutated"]),
            })
    return gene_mutations


def load_graph_edges():
    path = ROOT / "analysis" / "results" / "comutation_shift" / "graph_edges.json"
    with open(path) as f:
        edges = json.load(f)
    gene_edges = {}
    for e in edges:
        src, tgt = e["source"], e["target"]
        rec = {
            "type": e["type"],
            "interaction_hr": e["interaction_HR"],
            "interaction_p": e["interaction_p"],
            "n_both": e["n_both"],
        }
        gene_edges.setdefault(src, []).append({"partner": tgt, **rec})
        gene_edges.setdefault(tgt, []).append({"partner": src, **rec})
    return gene_edges


def load_compensatory_edges():
    path = ROOT / "analysis" / "results" / "comutation_screen" / "compensates_edges.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    edges = data.get("edges", data) if isinstance(data, dict) else data
    gene_comp = {}
    for e in edges:
        src = e.get("from", e.get("source", ""))
        tgt = e.get("to", e.get("target", ""))
        rec = {
            "hr": e.get("HR", e.get("hr", 0)),
            "p_value": e.get("p_value", 0),
            "n_both": e.get("n_patients", e.get("n_both", 0)),
        }
        gene_comp.setdefault(src, []).append({"partner": tgt, **rec})
        gene_comp.setdefault(tgt, []).append({"partner": src, **rec})
    return gene_comp


def load_ncbi_summaries():
    path = DATA_REPO / "ncbi_gene_summaries.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_uniprot_annotations():
    path = DATA_REPO / "uniprot_annotations.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_kegg_pathways():
    path = DATA_REPO / "kegg_pathways.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(v):
    if abs(v) < 0.001:
        return f"{v:.2e}"
    return f"{v:.4f}"


def esc(s):
    s = str(s)
    if any(c in s for c in ":{}\n[]&*#?|-<>=!%@`'\""):
        # Escape internal double quotes, then wrap
        s_escaped = s.replace('"', '\\"')
        return f'"{s_escaped}"'
    return s


def write_channel(channel, genes, degrees, hubs, mutation_atlas, graph_edges, comp_edges,
                   ncbi=None, uniprot=None, kegg=None):
    full_name = CHANNEL_FULL_NAMES.get(channel, channel)
    member_genes = sorted(genes)
    hub_genes = [g for g in member_genes if g in hubs]

    L = []  # lines
    L.append(f"id: CHAN-{channel}")
    L.append("type: knowledge")
    L.append("domain: cancer_biology")
    L.append(f"name: {esc(full_name)}")
    L.append(f"gene_count: {len(member_genes)}")
    L.append(f"hub_genes: [{', '.join(hub_genes)}]")
    L.append("")
    L.append("provenance:")
    L.append("  attribution:")
    L.append('    author: "Patrick D. McCarthy"')
    L.append('    source: "Genome as Projection (Paper 5)"')
    L.append('    date: "2026"')
    L.append('    doi: "10.5281/zenodo.18923066"')
    L.append("  evidence:")
    L.append("    type: empirical")
    L.append('    description: "Channel defined by organizational function; validated across MSK-IMPACT 2017, MSK-MET, MSK-IMPACT 50K (n=73,593)"')
    L.append("")

    # Genes with their data inline
    L.append("genes:")
    for gene in member_genes:
        deg = degrees.get(gene, 0)
        pos = "hub" if gene in hubs else "leaf"
        L.append(f"  - id: {gene}")
        L.append(f"    string_ppi_degree: {deg}")
        L.append(f"    position: {pos}")

        # NCBI Gene metadata
        ncbi_info = (ncbi or {}).get(gene)
        if ncbi_info:
            L.append(f"    full_name: {esc(ncbi_info.get('full_name', ''))}")
            L.append(f"    chromosome: {esc(ncbi_info.get('chromosome', ''))}")
            summary = ncbi_info.get('summary', '')
            if summary:
                # Truncate to first 2 sentences to control size
                sentences = summary.split('. ')
                short = '. '.join(sentences[:2])
                if not short.endswith('.'):
                    short += '.'
                L.append(f"    ncbi_summary: {esc(short)}")

        # UniProt function + GO terms
        uni_info = (uniprot or {}).get(gene)
        if uni_info:
            protein = uni_info.get('protein_name', '')
            if protein:
                L.append(f"    protein_name: {esc(protein)}")
            funcs = uni_info.get('function', [])
            if funcs:
                short_func = funcs[0][:200]
                if len(funcs[0]) > 200:
                    short_func += '...'
                L.append(f"    uniprot_function: {esc(short_func)}")
            locations = list(dict.fromkeys(uni_info.get('subcellular_location', [])))  # dedupe
            if locations:
                L.append(f"    subcellular_location: [{', '.join(esc(l) for l in locations[:3])}]")
            diseases = uni_info.get('diseases', [])
            if diseases:
                L.append("    diseases:")
                for d in diseases[:5]:
                    name = d.get('name', '')
                    if name:
                        L.append(f"      - {esc(name)}")
            go_terms = uni_info.get('go_terms', [])
            # Filter to biological process terms (most useful for benchmarks)
            bp = [g for g in go_terms if g.get('term', '').startswith('P:')]
            if bp:
                L.append("    go_biological_process:")
                for g in bp[:8]:
                    term = g['term'].replace('P:', '')
                    L.append(f"      - {esc(term)}")

        # KEGG pathways
        kegg_info = (kegg or {}).get(gene)
        if kegg_info:
            pathways = kegg_info.get('pathways', [])
            if pathways:
                L.append("    kegg_pathways:")
                for p in pathways[:8]:
                    L.append(f"      - {esc(p['name'])}")

        # Top confident mutations (max 5 per gene to control size)
        muts = mutation_atlas.get(gene, [])
        confident = [m for m in muts if m["confident"]]
        top = sorted(confident, key=lambda m: m["n_mutated"], reverse=True)[:5]
        if top:
            L.append("    mutations:")
            for m in top:
                L.append(f"      - change: {esc(m['protein_change'])}")
                L.append(f"        cancer_type: {esc(m['cancer_type'])}")
                L.append(f"        hr: {fmt(m['hr'])}")
                L.append(f"        p: {fmt(m['p_value'])}")
                L.append(f"        direction: {m['direction']}")
                L.append(f"        n: {m['n_mutated']}")

        # Cross-gene edges (max 5 per gene — most significant)
        ge = graph_edges.get(gene, [])
        ce = comp_edges.get(gene, [])
        gene_edge_list = []
        if ge:
            seen = set()
            for e in sorted(ge, key=lambda x: x["interaction_p"])[:5]:
                if e["partner"] not in seen:
                    seen.add(e["partner"])
                    gene_edge_list.append({
                        "to": e["partner"],
                        "relation": e["type"],
                        "hr": e["interaction_hr"],
                        "p": e["interaction_p"],
                        "n": e["n_both"],
                    })
        if ce:
            seen_c = set()
            for e in sorted(ce, key=lambda x: x["hr"])[:3]:
                if e["partner"] not in seen_c:
                    seen_c.add(e["partner"])
                    gene_edge_list.append({
                        "to": e["partner"],
                        "relation": "COMPENSATES",
                        "hr": e["hr"],
                        "p": e["p_value"],
                        "n": e["n_both"],
                    })

        if gene_edge_list:
            L.append("    cross_edges:")
            for e in gene_edge_list:
                L.append(f"      - to: {e['to']}")
                L.append(f"        relation: {e['relation']}")
                L.append(f"        hr: {fmt(e['hr'])}")
                L.append(f"        p: {fmt(e['p'])}")
                L.append(f"        n: {e['n']}")

    L.append("")

    # Channel-level edges
    L.append("edges:")
    L.append("  - to: NV01-graph-necessity")
    L.append("    relation: instantiated_in")
    L.append("    provenance:")
    L.append("      attribution:")
    L.append('        author: "Patrick D. McCarthy"')
    L.append('        source: "Paper 5 — Genome as Projection"')
    L.append('        date: "2026"')
    L.append("      evidence:")
    L.append("        type: formal-derivation")
    L.append('        description: "Coupling channels instantiate the escalation hierarchy predicted by bounded context theory"')

    path = CHAN_DIR / f"CHAN-{channel}.yaml"
    path.write_text("\n".join(L) + "\n")
    return len(member_genes)


def main():
    CHAN_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    gene_to_channel = load_channel_map()
    degrees, hubs = load_string_degrees()
    mutation_atlas = load_mutation_atlas()
    graph_edges = load_graph_edges()
    comp_edges = load_compensatory_edges()

    print("Loading external sources...")
    ncbi = load_ncbi_summaries()
    uniprot = load_uniprot_annotations()
    kegg = load_kegg_pathways()
    print(f"  NCBI: {len(ncbi)} genes, UniProt: {len(uniprot)} genes, KEGG: {len(kegg)} genes")

    channel_genes = {}
    for gene, channel in gene_to_channel.items():
        channel_genes.setdefault(channel, []).append(gene)

    total_genes = 0
    for channel, genes in channel_genes.items():
        n = write_channel(channel, genes, degrees, hubs, mutation_atlas, graph_edges, comp_edges,
                          ncbi=ncbi, uniprot=uniprot, kegg=kegg)
        total_genes += n
        print(f"  CHAN-{channel}: {n} genes")

    print(f"\nDone. {len(channel_genes)} channel files, {total_genes} genes total.")


if __name__ == "__main__":
    main()
