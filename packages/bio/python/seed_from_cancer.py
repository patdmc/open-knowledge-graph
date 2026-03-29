#!/usr/bin/env python3
"""
Seed the biology knowledge graph from the cancer graph's cached data.

Reads the cancer graph (509 genes, 8 channels, PPI, SL partners) and
emits YAML files that describe FUNCTION, not damage. The cancer graph
is the error log; this produces the source code.

Dimension changes:
  - PPI → INTERACTS_WITH (physical binding, gains context annotation)
  - SL_PARTNER → COMPENSATES (functional redundancy, read forward)
  - BELONGS_TO → multi-channel function profile
  - COOCCURS, HAS_MUTATION → dropped (cancer-only observations)

Usage:
    python seed_from_cancer.py
    # Emits BIO-CH01-DDR.yaml ... BIO-CH08-DNAMethylation.yaml + BIO-TIERS.yaml
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
BIOLOGY_DIR = SCRIPT_DIR.parent
KG_ROOT = BIOLOGY_DIR.parent.parent  # knowledge-graph/
PROJECT_ROOT = KG_ROOT.parent        # open-knowledge-graph/
GNN_CACHE = PROJECT_ROOT / "gnn" / "data" / "cache"

# ── DATA LOADING ─────────────────────────────────────────

def load_graph_config():
    """Load the 509-gene graph config (channels, hub/leaf, GOF/LOF)."""
    path = GNN_CACHE / "graph_derived_config.json"
    with open(path) as f:
        return json.load(f)


def load_ppi_edges():
    """Load STRING PPI edges: [[gene_a, gene_b, score], ...]."""
    path = GNN_CACHE / "string_ppi_edges_503.json"
    with open(path) as f:
        return json.load(f)


def load_sl_partners():
    """Load synthetic lethality pairs (both in 509 gene set)."""
    path = GNN_CACHE / "synthetic_lethality" / "synthetic_lethality_both_in_set.csv"
    pairs = []
    if not path.exists():
        return pairs
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "gene_a": row["gene_a"],
                "gene_b": row["gene_b"],
                "source": row.get("source", ""),
                "context": row.get("cancer_context", ""),
                "evidence": row.get("evidence_type", ""),
            })
    return pairs


# ── TIER STRUCTURE ───────────────────────────────────────

TIER_MAP = {
    "PI3K_Growth": 0, "CellCycle": 0,          # cell_intrinsic
    "DDR": 1, "TissueArch": 1,                  # tissue_level
    "Endocrine": 2, "Immune": 2,                # organism_level
    "ChromatinRemodel": 3, "DNAMethylation": 3,  # meta_regulatory
}

TIER_NAMES = {
    0: "cell_intrinsic",
    1: "tissue_level",
    2: "organism_level",
    3: "meta_regulatory",
}

CHANNEL_ORDER = [
    ("DDR", "CH01", "DNA Damage Response"),
    ("CellCycle", "CH02", "Cell Cycle Regulation"),
    ("PI3K_Growth", "CH03", "PI3K/Growth Signaling"),
    ("Endocrine", "CH04", "Endocrine Signaling"),
    ("Immune", "CH05", "Immune Regulation"),
    ("TissueArch", "CH06", "Tissue Architecture"),
    ("ChromatinRemodel", "CH07", "Chromatin Remodeling"),
    ("DNAMethylation", "CH08", "DNA Methylation"),
]

# ── FUNCTION DESCRIPTIONS (what channels DO, not what breaks) ──

CHANNEL_FUNCTIONS = {
    "DDR": {
        "summary": "Detects and repairs DNA damage (double-strand breaks, single-strand breaks, base mismatches, replication fork stalls). Maintains genomic integrity across cell divisions.",
        "healthy_state": "Damage sensing (ATM/ATR), signal transduction (CHK1/CHK2), repair execution (HR via BRCA1/RAD51, NHEJ via 53BP1, BER via PARP1, MMR via MSH2/MLH1)",
        "failure_mode": "Unrepaired DNA accumulates, genomic instability, structural variants, microsatellite instability",
    },
    "CellCycle": {
        "summary": "Controls cell division timing through cyclin-CDK complexes and checkpoint surveillance. Ensures DNA replication and chromosome segregation fidelity.",
        "healthy_state": "G1/S checkpoint (RB1/E2F/CDK4), S-phase progression (CDK2/CCNE1), G2/M checkpoint (CDK1/CCNB1), mitotic surveillance (AURKA/B), senescence enforcement (TP53/CDKN2A)",
        "failure_mode": "Uncontrolled proliferation, checkpoint bypass, aneuploidy, replicative immortality",
    },
    "PI3K_Growth": {
        "summary": "Transduces extracellular growth signals into intracellular metabolic and survival responses. Central node for receptor tyrosine kinase signaling.",
        "healthy_state": "Growth factor sensing (RTKs: EGFR, ERBB2, FGFR), PI3K/AKT/mTOR axis (nutrient/energy integration), RAS/MAPK cascade (proliferative signals), negative feedback (PTEN, NF1)",
        "failure_mode": "Constitutive growth signaling, metabolic reprogramming, apoptosis resistance",
    },
    "Endocrine": {
        "summary": "Mediates hormone-dependent gene regulation through nuclear receptors. Controls tissue-specific differentiation and homeostasis programs.",
        "healthy_state": "Hormone reception (AR, ESR1/2), transcriptional programs (FOXA1, GATA1), tissue identity maintenance",
        "failure_mode": "Hormone-independent growth, lineage plasticity, treatment resistance",
    },
    "Immune": {
        "summary": "Coordinates immune surveillance, antigen presentation, and inflammatory signaling. Interfaces between cell-autonomous defense and systemic immunity.",
        "healthy_state": "Antigen presentation (HLA-A/B/C, B2M), immune checkpoint balance (PD-1/PD-L1, CTLA4), cytokine signaling (JAK/STAT), NF-kB inflammatory control",
        "failure_mode": "Immune evasion, checkpoint dysregulation, chronic inflammation, loss of antigen presentation",
    },
    "TissueArch": {
        "summary": "Maintains tissue organization through cell-cell adhesion, polarity, and developmental signaling pathways (Wnt, Notch, Hedgehog, TGF-beta).",
        "healthy_state": "Cell adhesion (CDH1), Wnt pathway (APC/CTNNB1), TGF-beta signaling (SMAD2/3/4), Notch signaling, Hippo pathway (YAP1/WWTR1), tissue polarity",
        "failure_mode": "Loss of tissue organization, EMT, invasion, metastatic potential",
    },
    "ChromatinRemodel": {
        "summary": "Controls physical access to DNA through ATP-dependent nucleosome repositioning and histone modification. The 'read/write head' of epigenetic regulation.",
        "healthy_state": "SWI/SNF complex (SMARCA4, ARID1A/B, SMARCB1), histone methylation (KMT2A-D, SETD2, EZH2), histone acetylation (CREBBP, EP300), chromatin reading (BRD4)",
        "failure_mode": "Aberrant gene accessibility, silencing of tumor suppressors, activation of oncogenic programs",
    },
    "DNAMethylation": {
        "summary": "Maintains heritable epigenetic state through CpG methylation. Controls gene silencing, genomic imprinting, and transposon suppression across cell divisions.",
        "healthy_state": "Methylation writers (DNMT3A/B), methylation erasers (TET1/2), metabolic coupling (IDH1/2 produce alpha-KG for TET), splicing regulation (SRSF2, U2AF1)",
        "failure_mode": "Global hypomethylation, focal hypermethylation of tumor suppressors, loss of imprinting, CIMP phenotype",
    },
}

# ── GOF/LOF → FUNCTIONAL DESCRIPTION ────────────────────

def gene_function_description(gene, gof_lof, channel, is_hub):
    """Describe what a gene does when working, given its cancer annotation."""
    role = "hub" if is_hub else "member"
    if gof_lof == "GOF":
        return f"Gain-of-function oncogene: normally a regulated activator in {channel}. When working correctly, provides controlled positive signaling."
    elif gof_lof == "LOF":
        return f"Tumor suppressor: provides essential restraint/repair in {channel}. When working correctly, maintains pathway integrity and prevents aberrant activation."
    else:
        return f"Functional {role} of {channel} pathway."


def loss_consequence(gene, gof_lof, channel):
    """What happens when this gene breaks — the bridge to cancer graph."""
    if gof_lof == "GOF":
        return f"Constitutive activation of {channel} signaling regardless of upstream input"
    elif gof_lof == "LOF":
        return f"Loss of restraint/repair in {channel}, pathway proceeds unchecked"
    else:
        return f"Altered {channel} function"


# ── YAML EMISSION ────────────────────────────────────────

def yaml_str(val, indent=0):
    """Simple YAML value formatter."""
    prefix = "  " * indent
    if isinstance(val, str):
        if "\n" in val or ":" in val or '"' in val or len(val) > 80:
            return f'"{val}"'
        return val
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, list):
        if not val:
            return "[]"
        if all(isinstance(v, str) and len(v) < 40 for v in val):
            return "[" + ", ".join(v for v in val) + "]"
        lines = []
        for v in val:
            lines.append(f"{prefix}- {yaml_str(v, indent + 1)}")
        return "\n" + "\n".join(lines)
    return str(val)


def emit_channel_yaml(channel_key, channel_code, channel_name, genes_in_channel,
                      gene_function_map, hub_genes_set, ppi_by_gene, sl_by_gene):
    """Emit one channel YAML file."""
    tier = TIER_MAP[channel_key]
    tier_name = TIER_NAMES[tier]
    func = CHANNEL_FUNCTIONS[channel_key]

    lines = []
    lines.append(f"id: BIO-{channel_code}-{channel_key}")
    lines.append(f"type: biology_channel")
    lines.append(f"domain: molecular_biology")
    lines.append(f'name: "{channel_name}"')
    lines.append(f"channel: {channel_key}")
    lines.append(f"tier: {tier}")
    lines.append(f"tier_name: {tier_name}")
    lines.append(f"n_genes: {len(genes_in_channel)}")
    lines.append(f"n_hubs: {len([g for g in genes_in_channel if g in hub_genes_set])}")
    lines.append("")

    # Provenance
    lines.append("provenance:")
    lines.append("  attribution:")
    lines.append('    source: "cancer_graph_v6_seed"')
    lines.append('    method: "Seeded from cancer graph — damage framing inverted to function framing"')
    lines.append("    date: 2026-03-28")
    lines.append("  data_sources:")
    lines.append("    - STRING-DB PPI (confidence >= 700)")
    lines.append("    - CIViC GOF/LOF annotations")
    lines.append("    - BioGRID/SynLethDB synthetic lethality")
    lines.append("    - MSK-IMPACT gene panel (509 genes)")
    lines.append("")

    # Function block — what this channel DOES
    lines.append("function:")
    lines.append(f'  summary: "{func["summary"]}"')
    lines.append(f'  healthy_state: "{func["healthy_state"]}"')
    lines.append(f'  failure_mode: "{func["failure_mode"]}"')
    lines.append("")

    # Genes
    lines.append("genes:")
    for gene in sorted(genes_in_channel):
        gof_lof = gene_function_map.get(gene, "")
        is_hub = gene in hub_genes_set
        func_desc = gene_function_description(gene, gof_lof, channel_key, is_hub)
        loss_desc = loss_consequence(gene, gof_lof, channel_key)

        lines.append(f"  - id: BIO-{gene}")
        lines.append(f"    name: {gene}")
        lines.append(f"    position: {'hub' if is_hub else 'leaf'}")
        if gof_lof:
            lines.append(f"    cancer_function: {gof_lof}")
        lines.append(f'    healthy_role: "{func_desc}"')
        lines.append(f'    loss_consequence: "{loss_desc}"')

        # PPI edges for this gene (within-channel only here)
        gene_ppis = ppi_by_gene.get(gene, [])
        within_channel_ppis = [p for p in gene_ppis if p["partner"] in genes_in_channel]
        if within_channel_ppis:
            lines.append(f"    interacts_with:")
            for ppi in sorted(within_channel_ppis, key=lambda p: -p["score"])[:10]:
                lines.append(f"      - target: BIO-{ppi['partner']}")
                lines.append(f"        type: INTERACTS_WITH")
                lines.append(f"        score: {ppi['score']}")
                lines.append(f'        source: "STRING-DB"')

        # SL partners (= COMPENSATES edges)
        gene_sls = sl_by_gene.get(gene, [])
        if gene_sls:
            lines.append(f"    compensates:")
            for sl in gene_sls:
                lines.append(f"      - target: BIO-{sl['partner']}")
                lines.append(f"        type: COMPENSATES")
                if sl.get("context"):
                    lines.append(f'        context: "{sl["context"]}"')
                lines.append(f'        source: "{sl["source"]}"')
                lines.append(f'        note: "Synthetic lethality implies functional redundancy — loss of both is lethal because neither can compensate"')

        lines.append("")

    # Cross-channel PPI edges (genes in THIS channel interacting with genes in OTHER channels)
    cross_channel_edges = []
    for gene in genes_in_channel:
        for ppi in ppi_by_gene.get(gene, []):
            if ppi["partner"] not in genes_in_channel:
                cross_channel_edges.append({
                    "from": gene,
                    "to": ppi["partner"],
                    "score": ppi["score"],
                    "to_channel": ppi.get("partner_channel", "unknown"),
                })

    if cross_channel_edges:
        lines.append("cross_channel_edges:")
        # Deduplicate and sort by score
        seen = set()
        for edge in sorted(cross_channel_edges, key=lambda e: -e["score"]):
            pair = tuple(sorted([edge["from"], edge["to"]]))
            if pair in seen:
                continue
            seen.add(pair)
            if len(seen) > 30:
                break  # Cap at 30 cross-channel edges per file
            lines.append(f"  - from: BIO-{edge['from']}")
            lines.append(f"    to: BIO-{edge['to']}")
            lines.append(f"    to_channel: {edge['to_channel']}")
            lines.append(f"    type: INTERACTS_WITH")
            lines.append(f"    score: {edge['score']}")
            lines.append(f'    source: "STRING-DB"')
            lines.append(f'    note: "Cross-channel physical interaction — potential escalation bridge"')
        lines.append("")

    # Concepts for edge discovery compatibility
    lines.append("concepts:")
    lines.append(f"  - id: BIO-{channel_code}-{channel_key}")
    lines.append(f'    name: "{channel_name}"')
    for gene in sorted(genes_in_channel)[:20]:  # Top genes as concepts
        lines.append(f"  - id: BIO-{gene}")
        lines.append(f"    name: {gene}")
    lines.append("")

    # Edge discovery compatible edges list
    lines.append("edges:")
    for gene in sorted(genes_in_channel):
        lines.append(f"  - from: BIO-{channel_code}-{channel_key}")
        lines.append(f"    to: BIO-{gene}")
        lines.append(f"    type: CONTAINS")

    return "\n".join(lines)


def emit_tiers_yaml():
    """Emit the tier hierarchy definition."""
    lines = []
    lines.append("id: BIO-TIERS")
    lines.append("type: biology_hierarchy")
    lines.append("domain: molecular_biology")
    lines.append('name: "Cellular Function Hierarchy"')
    lines.append("")
    lines.append("provenance:")
    lines.append("  attribution:")
    lines.append('    source: "cancer_graph_v6_seed"')
    lines.append('    method: "Tier structure from coupling-channel cancer model, reframed as functional hierarchy"')
    lines.append("    date: 2026-03-28")
    lines.append("")
    lines.append('description: |')
    lines.append('  Four-tier hierarchy of cellular function. Lower tiers are cell-intrinsic')
    lines.append('  (growth, division). Higher tiers are organism-level (immune, endocrine)')
    lines.append('  and meta-regulatory (chromatin, methylation). Meta-regulatory tiers')
    lines.append('  control ACCESS to the genome — they are the read/write permissions')
    lines.append('  layer that governs which lower-tier programs can execute.')
    lines.append("")
    lines.append("tiers:")
    lines.append("  - tier: 0")
    lines.append("    name: cell_intrinsic")
    lines.append('    description: "Core cellular machinery — growth signaling and cell division"')
    lines.append("    channels: [PI3K_Growth, CellCycle]")
    lines.append('    analogy: "The CPU and clock — computation and timing"')
    lines.append("")
    lines.append("  - tier: 1")
    lines.append("    name: tissue_level")
    lines.append('    description: "Tissue integrity — DNA repair and structural organization"')
    lines.append("    channels: [DDR, TissueArch]")
    lines.append('    analogy: "Error correction and filesystem — data integrity and structure"')
    lines.append("")
    lines.append("  - tier: 2")
    lines.append("    name: organism_level")
    lines.append('    description: "Organism-wide coordination — hormonal and immune signaling"')
    lines.append("    channels: [Endocrine, Immune]")
    lines.append('    analogy: "Network and security — inter-process communication and access control"')
    lines.append("")
    lines.append("  - tier: 3")
    lines.append("    name: meta_regulatory")
    lines.append('    description: "Epigenetic access control — which genes can be read at all"')
    lines.append("    channels: [ChromatinRemodel, DNAMethylation]")
    lines.append('    analogy: "Kernel and permissions — controls what userspace programs can access"')
    lines.append("")

    # Cross-domain edges to existing knowledge graph
    lines.append("cross_domain_edges:")
    lines.append("  - from: BIO-TIERS")
    lines.append("    to: D08-encoding-hierarchy")
    lines.append("    type: INSTANTIATES")
    lines.append('    note: "The cellular tier hierarchy is a biological instantiation of the encoding hierarchy (D08). Tier 0 = L0 genetic, Tier 3 = meta-regulatory."')
    lines.append("  - from: BIO-TIERS")
    lines.append("    to: EC01-bounded-context")
    lines.append("    type: SUPPORTS")
    lines.append('    note: "Each tier has bounded context — channels within a tier share information, cross-tier communication is expensive (escalation)."')
    lines.append("")

    lines.append("concepts:")
    lines.append("  - id: BIO-TIERS")
    lines.append('    name: "Cellular Function Hierarchy"')
    lines.append("  - id: D08-encoding-hierarchy")
    lines.append('    name: "Encoding Hierarchy"')
    lines.append("  - id: EC01-bounded-context")
    lines.append('    name: "Bounded Context"')
    lines.append("")

    lines.append("edges:")
    lines.append("  - from: BIO-TIERS")
    lines.append("    to: D08-encoding-hierarchy")
    lines.append("    type: INSTANTIATES")
    lines.append("  - from: BIO-TIERS")
    lines.append("    to: EC01-bounded-context")
    lines.append("    type: SUPPORTS")

    return "\n".join(lines)


# ── MAIN ─────────────────────────────────────────────────

def main():
    print("Loading cancer graph data...")
    config = load_graph_config()
    ppi_raw = load_ppi_edges()
    sl_pairs = load_sl_partners()

    channel_map = config["channel_map"]
    gene_function_map = config.get("gene_function", {})
    hub_genes_raw = config.get("hub_genes", {})

    # Build hub set
    hub_genes_set = set()
    for ch, genes in hub_genes_raw.items():
        hub_genes_set.update(genes)

    print(f"  {len(channel_map)} genes, {len(ppi_raw)} PPI edges, {len(sl_pairs)} SL pairs")

    # Index genes by channel
    genes_by_channel = defaultdict(list)
    for gene, ch in channel_map.items():
        genes_by_channel[ch].append(gene)

    # Index PPI edges by gene
    ppi_by_gene = defaultdict(list)
    for edge in ppi_raw:
        gene_a, gene_b, score = edge[0], edge[1], edge[2]
        ch_b = channel_map.get(gene_b, "unknown")
        ch_a = channel_map.get(gene_a, "unknown")
        ppi_by_gene[gene_a].append({"partner": gene_b, "score": score, "partner_channel": ch_b})
        ppi_by_gene[gene_b].append({"partner": gene_a, "score": score, "partner_channel": ch_a})

    # Index SL pairs by gene
    sl_by_gene = defaultdict(list)
    for sl in sl_pairs:
        sl_by_gene[sl["gene_a"]].append({
            "partner": sl["gene_b"],
            "source": sl["source"],
            "context": sl.get("context", ""),
        })
        sl_by_gene[sl["gene_b"]].append({
            "partner": sl["gene_a"],
            "source": sl["source"],
            "context": sl.get("context", ""),
        })

    # Emit channel YAML files
    print("\nEmitting biology YAML files...")
    for channel_key, channel_code, channel_name in CHANNEL_ORDER:
        genes = genes_by_channel.get(channel_key, [])
        content = emit_channel_yaml(
            channel_key, channel_code, channel_name, genes,
            gene_function_map, hub_genes_set, ppi_by_gene, sl_by_gene
        )

        filename = f"BIO-{channel_code}-{channel_key}.yaml"
        filepath = BIOLOGY_DIR / filename
        with open(filepath, "w") as f:
            f.write(content + "\n")

        n_hubs = len([g for g in genes if g in hub_genes_set])
        n_ppi = sum(len(ppi_by_gene.get(g, [])) for g in genes)
        n_sl = sum(len(sl_by_gene.get(g, [])) for g in genes)
        print(f"  {filename}: {len(genes)} genes ({n_hubs} hubs), ~{n_ppi} PPI refs, {n_sl} SL refs")

    # Emit tiers YAML
    tiers_content = emit_tiers_yaml()
    tiers_path = BIOLOGY_DIR / "BIO-TIERS.yaml"
    with open(tiers_path, "w") as f:
        f.write(tiers_content + "\n")
    print(f"  BIO-TIERS.yaml: 4 tiers, 8 channels")

    # Summary
    total_genes = sum(len(genes_by_channel[ch]) for ch, _, _ in CHANNEL_ORDER)
    print(f"\nDone. {total_genes} genes across 8 channels, 4 tiers.")
    print(f"Files written to: {BIOLOGY_DIR}")


if __name__ == "__main__":
    main()
