#!/usr/bin/env python3
"""
Normalize COOCCURS.cancer_type display names to stable identifiers.

- Queries all distinct COOCCURS.cancer_type values and CancerType node names
- Builds mapping from MSK display names → TCGA codes
- Adds display_name to CancerType nodes that lack one
- Adds cancer_type_id to COOCCURS edges (does NOT delete original cancer_type)
"""

import re
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "openknowledgegraph")


# ── Step 1: Query existing data ──────────────────────────────────────────────

driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    # All distinct cancer_type values on COOCCURS edges
    result = session.run(
        "MATCH ()-[r:COOCCURS]->() "
        "RETURN DISTINCT r.cancer_type AS ct"
    )
    cooccurs_types = sorted([rec["ct"] for rec in result if rec["ct"] is not None])

    # All CancerType nodes
    result = session.run(
        "MATCH (c:CancerType) "
        "RETURN c.name AS name, c.display_name AS display_name"
    )
    cancer_nodes = {rec["name"]: rec["display_name"] for rec in result}

print("=" * 70)
print("COOCCURS.cancer_type distinct values:")
for ct in cooccurs_types:
    print(f"  {ct}")
print(f"\nTotal distinct COOCCURS cancer_type values: {len(cooccurs_types)}")

print("\nCancerType nodes (name → display_name):")
for name, dn in sorted(cancer_nodes.items()):
    print(f"  {name} → {dn}")
print(f"\nTotal CancerType nodes: {len(cancer_nodes)}")

# ── Step 2: Build mapping ────────────────────────────────────────────────────

# Known MSK display name → TCGA/standard abbreviation
KNOWN_MAP = {
    "Breast Cancer":                "BRCA",
    "Colorectal Cancer":            "COAD",
    "Non-Small Cell Lung Cancer":   "NSCLC",
    "Lung Cancer":                  "LUAD",
    "Small Cell Lung Cancer":       "SCLC",
    "Melanoma":                     "SKCM",
    "Glioma":                       "LGG",
    "Glioblastoma":                 "GBM",
    "Glioblastoma Multiforme":      "GBM",
    "Bladder Cancer":               "BLCA",
    "Prostate Cancer":              "PRAD",
    "Ovarian Cancer":               "OV",
    "Pancreatic Cancer":            "PAAD",
    "Hepatobiliary Cancer":         "LIHC",
    "Liver Cancer":                 "LIHC",
    "Head and Neck Cancer":         "HNSC",
    "Renal Cell Carcinoma":         "KIRC",
    "Kidney Cancer":                "KIRC",
    "Esophagogastric Cancer":       "ESCA",
    "Gastric Cancer":               "STAD",
    "Stomach Cancer":               "STAD",
    "Endometrial Cancer":           "UCEC",
    "Uterine Cancer":               "UCEC",
    "Cervical Cancer":              "CESC",
    "Thyroid Cancer":               "THCA",
    "Adrenocortical Carcinoma":     "ACC",
    "Mesothelioma":                 "MESO",
    "Soft Tissue Sarcoma":          "SARC",
    "Bone Cancer":                  "SARC",
    "Leukemia":                     "LAML",
    "Acute Myeloid Leukemia":       "LAML",
    "Chronic Lymphocytic Leukemia": "CLL",
    "Lymphoma":                     "DLBC",
    "Diffuse Large B-Cell Lymphoma":"DLBC",
    "Multiple Myeloma":             "MM",
    "Myeloproliferative Neoplasms": "MPN",
    "Myelodysplastic Syndromes":    "MDS",
    "Cholangiocarcinoma":           "CHOL",
    "Pheochromocytoma":             "PCPG",
    "Testicular Cancer":            "TGCT",
    "Penile Cancer":                "penile_cancer",
    "Vulvar Cancer":                "vulvar_cancer",
    "Thymoma":                      "THYM",
    "Uveal Melanoma":               "UVM",
    "Skin Cancer, Non-Melanoma":    "skin_non_melanoma",
    "Cancer of Unknown Primary":    "CUP",
    "Gastrointestinal Stromal Tumor": "GIST",
    "Neuroblastoma":                "NB",
    "Wilms Tumor":                  "WT",
    "Retinoblastoma":               "RB",
    "Embryonal Tumor":              "embryonal_tumor",
    "CNS Cancer":                   "cns_cancer",
    "Nerve Sheath Tumor":           "nerve_sheath_tumor",
    "Salivary Gland Cancer":        "salivary_gland_cancer",
    "Appendiceal Cancer":           "appendiceal_cancer",
    "Ampullary Cancer":             "ampullary_cancer",
    "Small Bowel Cancer":           "small_bowel_cancer",
    "Anal Cancer":                  "anal_cancer",
    "Vaginal Cancer":               "vaginal_cancer",
    "Gestational Trophoblastic Disease": "GTD",
    "Sex Cord Stromal Tumor":       "sex_cord_stromal_tumor",
    "Mature B-Cell Neoplasms":      "mature_b_cell",
    "Mature T and NK Neoplasms":    "mature_t_nk",
    "B-Lymphoblastic Leukemia/Lymphoma": "b_lymphoblastic",
    "T-Lymphoblastic Leukemia/Lymphoma": "t_lymphoblastic",
    "Histiocytosis":                "histiocytosis",
    "Mastocytosis":                 "mastocytosis",
    "Sellar Tumor":                 "sellar_tumor",
    "Gastrointestinal Neuroendocrine Tumor": "gi_neuroendocrine_tumor",
    "Germ Cell Tumor":              "TGCT",
    "Uterine Sarcoma":              "UCS",
}

# Also map any CancerType node name that is itself a TCGA code
# (identity mapping — these are already stable)
cancer_node_names_upper = {n.upper(): n for n in cancer_nodes}


def to_snake_case(s: str) -> str:
    """Convert display name to snake_case fallback identifier."""
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = s.strip("_").lower()
    return s


# Build the final mapping: display_name → cancer_type_id
mapping = {}
unmapped_fallbacks = []

for display_name in cooccurs_types:
    if display_name in KNOWN_MAP:
        mapping[display_name] = KNOWN_MAP[display_name]
    elif display_name.upper() in cancer_node_names_upper:
        # Already a TCGA code used as display name
        mapping[display_name] = cancer_node_names_upper[display_name.upper()]
    elif display_name in cancer_nodes:
        # Exact match to a CancerType node name
        mapping[display_name] = display_name
    else:
        fallback = to_snake_case(display_name)
        mapping[display_name] = fallback
        unmapped_fallbacks.append((display_name, fallback))

print("\n" + "=" * 70)
print("FINAL MAPPING (display_name → cancer_type_id):")
for dn, cid in sorted(mapping.items()):
    tag = " [fallback]" if (dn, cid) in [(u[0], u[1]) for u in unmapped_fallbacks] else ""
    print(f"  {dn:45s} → {cid}{tag}")

if unmapped_fallbacks:
    print(f"\n  ⚠ {len(unmapped_fallbacks)} unmapped (snake_case fallback):")
    for dn, fb in unmapped_fallbacks:
        print(f"    {dn} → {fb}")

# Reverse lookup: cancer_type_id → display_name (for CancerType node updates)
id_to_display = {}
for dn, cid in mapping.items():
    if cid not in id_to_display:
        id_to_display[cid] = dn

# ── Step 3: Add display_name to CancerType nodes ─────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Adding display_name to CancerType nodes that lack one...")

nodes_updated = 0
with driver.session() as session:
    for node_name, existing_display in cancer_nodes.items():
        if existing_display is None and node_name in id_to_display:
            display = id_to_display[node_name]
            session.run(
                "MATCH (c:CancerType {name: $name}) "
                "SET c.display_name = $display_name",
                name=node_name,
                display_name=display,
            )
            print(f"  SET display_name on CancerType({node_name}) = '{display}'")
            nodes_updated += 1

print(f"  CancerType nodes updated with display_name: {nodes_updated}")

# ── Step 4: Add cancer_type_id to COOCCURS edges ─────────────────────────────

print("\n" + "=" * 70)
print("STEP 4: Adding cancer_type_id to COOCCURS edges...")

total_edges_updated = 0
with driver.session() as session:
    for display_name, cancer_type_id in mapping.items():
        result = session.run(
            "MATCH ()-[r:COOCCURS {cancer_type: $ct}]->() "
            "SET r.cancer_type_id = $cid "
            "RETURN count(r) AS cnt",
            ct=display_name,
            cid=cancer_type_id,
        )
        cnt = result.single()["cnt"]
        if cnt > 0:
            print(f"  {display_name:45s} → {cancer_type_id:20s} ({cnt:,} edges)")
            total_edges_updated += cnt

print(f"\n  Total COOCCURS edges updated: {total_edges_updated:,}")

# ── Step 5: Verify ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("VERIFICATION:")
with driver.session() as session:
    result = session.run(
        "MATCH ()-[r:COOCCURS]->() "
        "WHERE r.cancer_type_id IS NULL "
        "RETURN count(r) AS missing"
    )
    missing = result.single()["missing"]
    print(f"  COOCCURS edges still missing cancer_type_id: {missing}")

    result = session.run(
        "MATCH ()-[r:COOCCURS]->() "
        "WHERE r.cancer_type IS NULL "
        "RETURN count(r) AS no_ct"
    )
    no_ct = result.single()["no_ct"]
    print(f"  COOCCURS edges with no cancer_type (original): {no_ct}")

    result = session.run(
        "MATCH ()-[r:COOCCURS]->() "
        "RETURN count(r) AS total"
    )
    total = result.single()["total"]
    print(f"  Total COOCCURS edges: {total:,}")

driver.close()
print("\nDone. Original cancer_type property preserved on all edges.")
