#!/usr/bin/env python3
"""Build top-down biology knowledge nodes from external sources.

Hierarchy:
  Organ Systems → Cellular Processes → Pathways → (Genes already exist)

Classification edges point DOWN (CONTAINS, REGULATES).
Causal edges point UP (ENCODED_BY, PARTICIPATES_IN).
Where they don't meet = open questions worth investigating.

Reads:
  - ../open-knowledge-graph-data/biology/mesh/mesh_biology_terms.json
  - ../open-knowledge-graph-data/biology/gene_ontology/go_biological_processes.json
  - ../open-knowledge-graph-data/biology/reactome/reactome_pathways.json (if available)

Writes:
  - knowledge-graph/nodes/biology/organ_systems/
  - knowledge-graph/nodes/biology/processes/
  - knowledge-graph/nodes/biology/pathways/
  - knowledge-graph/nodes/endocrinology/
  - knowledge-graph/nodes/neurology/
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_REPO = ROOT.parent / "open-knowledge-graph-data" / "biology"
NODES = ROOT / "knowledge-graph" / "nodes"


def esc(s):
    s = str(s)
    if any(c in s for c in ":{}\n[]&*#?|-<>=!%@`'\""):
        s_escaped = s.replace('"', '\\"')
        return f'"{s_escaped}"'
    return s


def write_yaml(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def provenance_block(source, source_url=None):
    """Standard provenance for external sources."""
    L = []
    L.append("provenance:")
    L.append("  attribution:")
    L.append(f'    source: "{source}"')
    if source_url:
        L.append(f'    url: "{source_url}"')
    L.append('    date: "2026"')
    L.append("  evidence:")
    L.append("    type: cited")
    L.append(f'    description: "Curated definition from {source}"')
    return L


# ─── Organ Systems (from MeSH) ───────────────────────────────────────────

ORGAN_SYSTEMS = {
    "nervous_system": {
        "id": "BIO-NERVOUS",
        "name": "Nervous System",
        "domain_dir": "neurology",
        "related_channel": None,
        "mesh_key": "nervous_system",
        "children_keys": ["neurotransmitters", "synaptic_transmission", "neuroplasticity",
                          "glia", "blood_brain_barrier"],
    },
    "endocrine_system": {
        "id": "BIO-ENDOCRINE",
        "name": "Endocrine System",
        "domain_dir": "endocrinology",
        "related_channel": "CHAN-Endocrine",
        "mesh_key": "endocrine_system",
        "children_keys": ["hormones", "estrogen_receptors", "androgen_receptors",
                          "thyroid", "pituitary", "adrenal", "insulin_signaling"],
    },
    "immune_system": {
        "id": "BIO-IMMUNE",
        "name": "Immune System",
        "domain_dir": "biology/organ_systems",
        "related_channel": "CHAN-Immune",
        "mesh_key": "immune_system",
        "children_keys": ["t_cells", "b_cells", "cytokines", "immune_checkpoint",
                          "inflammation"],
    },
}


def build_organ_systems(mesh):
    count = 0
    for key, spec in ORGAN_SYSTEMS.items():
        mesh_data = mesh.get(spec["mesh_key"], {}).get("records", [{}])
        main_rec = mesh_data[0] if mesh_data else {}
        scope = main_rec.get("scope_note", "")

        L = []
        L.append(f"id: {spec['id']}")
        L.append("type: knowledge")
        L.append("domain: biology")
        L.append(f"name: {esc(spec['name'])}")
        L.append("")
        if scope:
            # Truncate to 3 sentences
            sentences = scope.split('. ')
            short = '. '.join(sentences[:3])
            if not short.endswith('.'):
                short += '.'
            L.append(f"description: {esc(short)}")
            L.append("")
        L.extend(provenance_block("MeSH (NLM)", "https://meshb.nlm.nih.gov"))
        L.append("")

        # Sub-components from MeSH children
        child_records = []
        for child_key in spec["children_keys"]:
            child_mesh = mesh.get(child_key, {}).get("records", [])
            for rec in child_mesh[:1]:
                child_records.append({
                    "key": child_key,
                    "name": rec.get("name", child_key),
                    "scope": rec.get("scope_note", "")[:200],
                })

        if child_records:
            L.append("components:")
            for cr in child_records:
                L.append(f"  - name: {esc(cr['name'])}")
                if cr['scope']:
                    L.append(f"    description: {esc(cr['scope'])}")

        L.append("")
        L.append("edges:")

        # Link to related channel if exists
        if spec.get("related_channel"):
            L.append(f"  - to: {spec['related_channel']}")
            L.append("    relation: grounds")
            L.append("    provenance:")
            L.append("      attribution:")
            L.append('        author: "Patrick D. McCarthy"')
            L.append('        source: "Paper 5 — Genome as Projection"')
            L.append('        date: "2026"')
            L.append("      evidence:")
            L.append("        type: empirical")
            desc = f"{spec['name']} biology grounds the {spec['related_channel']} coupling channel"
            L.append(f'        description: "{desc}"')

        # Cross-links to other organ systems
        for other_key, other_spec in ORGAN_SYSTEMS.items():
            if other_key != key:
                L.append(f"  - to: {other_spec['id']}")
                L.append("    relation: interacts_with")

        out_dir = NODES / spec["domain_dir"]
        write_yaml(out_dir / f"{spec['id']}.yaml", L)
        count += 1
        print(f"  {spec['id']}: {spec['name']} ({len(child_records)} components)")

    return count


# ─── Cellular Processes (from GO) ────────────────────────────────────────

PROCESS_MAP = {
    "GO:0006915": {"id": "PROC-APOPTOSIS", "channel": "CHAN-CellCycle"},
    "GO:0007049": {"id": "PROC-CELLCYCLE", "channel": "CHAN-CellCycle"},
    "GO:0006281": {"id": "PROC-DNAREPAIR", "channel": "CHAN-DDR"},
    "GO:0007165": {"id": "PROC-SIGNALTRANSDUCTION", "channel": "CHAN-PI3K_Growth"},
    "GO:0006955": {"id": "PROC-IMMUNERESPONSE", "channel": "CHAN-Immune"},
    "GO:0008283": {"id": "PROC-PROLIFERATION", "channel": "CHAN-CellCycle"},
    "GO:0030154": {"id": "PROC-DIFFERENTIATION", "channel": "CHAN-TissueArchitecture"},
    "GO:0007155": {"id": "PROC-CELLADHESION", "channel": "CHAN-TissueArchitecture"},
    "GO:0006914": {"id": "PROC-AUTOPHAGY", "channel": "CHAN-CellCycle"},
    "GO:0000278": {"id": "PROC-MITOSIS", "channel": "CHAN-CellCycle"},
    "GO:0006974": {"id": "PROC-DNADAMAGERESPONSE", "channel": "CHAN-DDR"},
    "GO:0016055": {"id": "PROC-WNT", "channel": "CHAN-TissueArchitecture"},
    "GO:0007219": {"id": "PROC-NOTCH", "channel": "CHAN-TissueArchitecture"},
    "GO:0007173": {"id": "PROC-EGFR", "channel": "CHAN-PI3K_Growth"},
    "GO:0048015": {"id": "PROC-PI3K", "channel": "CHAN-PI3K_Growth"},
    "GO:0006302": {"id": "PROC-DSBREPAIR", "channel": "CHAN-DDR"},
    "GO:0006298": {"id": "PROC-MISMATCHREPAIR", "channel": "CHAN-DDR"},
    "GO:0001525": {"id": "PROC-ANGIOGENESIS", "channel": "CHAN-TissueArchitecture"},
    "GO:0007399": {"id": "PROC-NEURDEV", "channel": None},
    "GO:0007268": {"id": "PROC-SYNAPTICTRANSMISSION", "channel": None},
    "GO:0006260": {"id": "PROC-DNAREPLICATION", "channel": "CHAN-DDR"},
    "GO:0006351": {"id": "PROC-TRANSCRIPTION", "channel": None},
    "GO:0006412": {"id": "PROC-TRANSLATION", "channel": None},
    "GO:0006468": {"id": "PROC-PHOSPHORYLATION", "channel": "CHAN-PI3K_Growth"},
    "GO:0006954": {"id": "PROC-INFLAMMATION", "channel": "CHAN-Immune"},
}


def build_processes(go_data):
    proc_dir = NODES / "biology" / "processes"
    count = 0

    for go_id, spec in PROCESS_MAP.items():
        term = go_data.get(go_id)
        if not term:
            continue

        L = []
        L.append(f"id: {spec['id']}")
        L.append("type: knowledge")
        L.append("domain: biology")
        L.append(f"name: {esc(term['name'])}")
        L.append(f"go_id: {go_id}")
        L.append("")
        if term.get("definition"):
            defn = term["definition"]
            # Truncate to 3 sentences
            sentences = defn.split('. ')
            short = '. '.join(sentences[:3])
            if not short.endswith('.'):
                short += '.'
            L.append(f"definition: {esc(short)}")
            L.append("")

        L.extend(provenance_block("Gene Ontology", "https://geneontology.org"))
        L.append("")

        # Child processes
        children = term.get("children", [])
        if children:
            L.append("sub_processes:")
            for child in children[:15]:
                L.append(f"  - id: {child['id']}")
                L.append(f"    name: {esc(child['name'])}")
                L.append(f"    relation: {child.get('relation', 'is_a')}")

        L.append("")
        L.append("edges:")

        # Link to channel (bottom-up: process ENCODED_BY channel genes)
        if spec.get("channel"):
            L.append(f"  - to: {spec['channel']}")
            L.append("    relation: encoded_by")
            L.append("    provenance:")
            L.append("      attribution:")
            L.append('        author: "Patrick D. McCarthy"')
            L.append('        source: "Paper 5 — Genome as Projection"')
            L.append('        date: "2026"')
            L.append("      evidence:")
            L.append("        type: empirical")
            desc = f"Genes in {spec['channel']} encode components of {term['name']}"
            L.append(f'        description: "{desc}"')

        # Link neurology/endocrine processes to organ systems
        if "neur" in term["name"].lower() or "synap" in term["name"].lower():
            L.append("  - to: BIO-NERVOUS")
            L.append("    relation: part_of")
        if "immun" in term["name"].lower() or "inflam" in term["name"].lower():
            L.append("  - to: BIO-IMMUNE")
            L.append("    relation: part_of")

        write_yaml(proc_dir / f"{spec['id']}.yaml", L)
        count += 1
        print(f"  {spec['id']}: {term['name']} ({len(children)} children)")

    return count


# ─── Pathways (from Reactome) ────────────────────────────────────────────

# Map Reactome top-level categories to our channel structure
REACTOME_TO_CHANNEL = {
    "DNA Repair": "CHAN-DDR",
    "DNA Replication": "CHAN-DDR",
    "Cell Cycle": "CHAN-CellCycle",
    "Programmed Cell Death": "CHAN-CellCycle",
    "Signal Transduction": "CHAN-PI3K_Growth",
    "Immune System": "CHAN-Immune",
    "Extracellular matrix organization": "CHAN-TissueArchitecture",
    "Cell-Cell communication": "CHAN-TissueArchitecture",
}


def build_pathways(reactome_data):
    if not reactome_data:
        print("  No Reactome data available, skipping pathways")
        return 0

    pw_dir = NODES / "biology" / "pathways"
    count = 0

    for st_id, pw in reactome_data.items():
        name = pw.get("name", "")
        summary = pw.get("summary", "")
        children = pw.get("children", [])
        n_children = pw.get("n_children", 0)

        # Generate a clean ID
        clean = name.replace(" ", "").replace("/", "").replace("-", "").replace("(", "").replace(")", "")[:30]
        node_id = f"PW-{clean}"

        L = []
        L.append(f"id: {node_id}")
        L.append("type: knowledge")
        L.append("domain: biology")
        L.append(f"name: {esc(name)}")
        L.append(f"reactome_id: {st_id}")
        L.append("")
        if summary:
            # Strip HTML tags
            import re
            clean_summary = re.sub(r'<[^>]+>', '', summary)
            sentences = clean_summary.split('. ')
            short = '. '.join(sentences[:3])
            if not short.endswith('.'):
                short += '.'
            L.append(f"description: {esc(short)}")
            L.append("")

        L.extend(provenance_block("Reactome", "https://reactome.org"))
        L.append("")

        if children:
            L.append("sub_pathways:")
            for child in children[:20]:
                L.append(f"  - name: {esc(child['name'])}")
                L.append(f"    reactome_id: {child['stId']}")
                L.append(f"    type: {child.get('type', 'Pathway')}")

        L.append("")
        L.append("edges:")

        # Link to channel
        channel = REACTOME_TO_CHANNEL.get(name)
        if channel:
            L.append(f"  - to: {channel}")
            L.append("    relation: encoded_by")
            L.append("    provenance:")
            L.append("      attribution:")
            L.append('        author: "Patrick D. McCarthy"')
            L.append('        source: "Paper 5 — Genome as Projection"')
            L.append('        date: "2026"')
            L.append("      evidence:")
            L.append("        type: empirical")
            L.append(f'        description: "{name} pathway components are encoded by {channel} genes"')

        write_yaml(pw_dir / f"{node_id}.yaml", L)
        count += 1

    print(f"  {count} pathway nodes from Reactome")
    return count


# ─── Cancer Biology (from MeSH) ──────────────────────────────────────────

CANCER_CONCEPTS = {
    "neoplasms": "BIO-NEOPLASMS",
    "tumor_suppressor": "BIO-TUMORSUPPRESSORS",
    "oncogenes": "BIO-ONCOGENES",
    "metastasis": "BIO-METASTASIS",
    "tumor_microenvironment": "BIO-TME",
}


def build_cancer_concepts(mesh):
    bio_dir = NODES / "biology" / "concepts"
    count = 0

    for key, node_id in CANCER_CONCEPTS.items():
        mesh_data = mesh.get(key, {}).get("records", [{}])
        main_rec = mesh_data[0] if mesh_data else {}
        name = main_rec.get("name", key.replace("_", " ").title())
        scope = main_rec.get("scope_note", "")

        L = []
        L.append(f"id: {node_id}")
        L.append("type: knowledge")
        L.append("domain: cancer_biology")
        L.append(f"name: {esc(name)}")
        L.append("")
        if scope:
            sentences = scope.split('. ')
            short = '. '.join(sentences[:3])
            if not short.endswith('.'):
                short += '.'
            L.append(f"description: {esc(short)}")
            L.append("")
        L.extend(provenance_block("MeSH (NLM)", "https://meshb.nlm.nih.gov"))
        L.append("")

        L.append("edges:")
        L.append("  - to: NV01-graph-necessity")
        L.append("    relation: instantiated_in")

        write_yaml(bio_dir / f"{node_id}.yaml", L)
        count += 1
        print(f"  {node_id}: {name}")

    return count


# ─── Molecular Biology (from MeSH) ──────────────────────────────────────

MOLECULAR_CONCEPTS = {
    "chromatin": "BIO-CHROMATIN",
    "epigenetics": "BIO-EPIGENETICS",
    "transcription": "BIO-TRANSCRIPTION",
    "translation": "BIO-TRANSLATION",
    "protein_folding": "BIO-PROTEINFOLDING",
    "ubiquitin": "BIO-UBIQUITIN",
    "kinases": "BIO-KINASES",
    "phosphatases": "BIO-PHOSPHATASES",
}


def build_molecular_concepts(mesh):
    mol_dir = NODES / "biology" / "molecular"
    count = 0

    for key, node_id in MOLECULAR_CONCEPTS.items():
        mesh_data = mesh.get(key, {}).get("records", [{}])
        main_rec = mesh_data[0] if mesh_data else {}
        name = main_rec.get("name", key.replace("_", " ").title())
        scope = main_rec.get("scope_note", "")

        L = []
        L.append(f"id: {node_id}")
        L.append("type: knowledge")
        L.append("domain: biology")
        L.append(f"name: {esc(name)}")
        L.append("")
        if scope:
            sentences = scope.split('. ')
            short = '. '.join(sentences[:3])
            if not short.endswith('.'):
                short += '.'
            L.append(f"description: {esc(short)}")
            L.append("")
        L.extend(provenance_block("MeSH (NLM)", "https://meshb.nlm.nih.gov"))

        write_yaml(mol_dir / f"{node_id}.yaml", L)
        count += 1
        print(f"  {node_id}: {name}")

    return count


def main():
    print("Loading external sources...")
    mesh = json.load(open(DATA_REPO / "mesh" / "mesh_biology_terms.json"))
    go = json.load(open(DATA_REPO / "gene_ontology" / "go_biological_processes.json"))

    reactome_path = DATA_REPO / "reactome" / "reactome_pathways.json"
    reactome = json.load(open(reactome_path)) if reactome_path.exists() else {}
    print(f"  MeSH: {len(mesh)} concepts, GO: {len(go)} terms, Reactome: {len(reactome)} pathways")

    print("\n--- Organ Systems (top-down entry) ---")
    n1 = build_organ_systems(mesh)

    print("\n--- Cellular Processes (middle layer) ---")
    n2 = build_processes(go)

    print("\n--- Signaling Pathways (Reactome) ---")
    n3 = build_pathways(reactome)

    print("\n--- Cancer Biology Concepts ---")
    n4 = build_cancer_concepts(mesh)

    print("\n--- Molecular Biology ---")
    n5 = build_molecular_concepts(mesh)

    total = n1 + n2 + n3 + n4 + n5
    print(f"\nDone. {total} biology nodes created.")
    print(f"  Organ systems: {n1}")
    print(f"  Processes: {n2}")
    print(f"  Pathways: {n3}")
    print(f"  Cancer concepts: {n4}")
    print(f"  Molecular: {n5}")


if __name__ == "__main__":
    main()
