#!/usr/bin/env python3
"""
Atlas Imputation — infer HR estimates for missing (cancer_type, gene) pairs
by walking Neo4j graph edges.

The survival atlas covers 364 of 2,036 (cancer_type, gene) pairs seen in patients.
4,057 patients have mutations with ZERO atlas coverage. This script fills the gap
by generating tier 4 (graph-inferred) entries.

Imputation strategy (weighted ensemble of graph signals):

  1. PPI neighbor transfer: If BRCA1 has HR in breast cancer and BRCA2 is a PPI
     neighbor, transfer a shrunk HR. Weight by PPI confidence.

  2. Same-channel transfer: Genes in the same channel (e.g., DDR) tend to have
     similar prognostic effects. Use channel median HR with shrinkage.

  3. Co-occurrence signal: If gene A always co-occurs with gene B that has an
     atlas entry, the co-occurrence pattern carries information.

  4. Expression context: Genes over-expressed in a cancer type (EXPRESSION_IN)
     are more likely oncogenic (HR > 1), under-expressed more likely TSG (HR < 1).

  5. DepMap essentiality: If a gene is essential in the cancer's lineage
     (ESSENTIAL_IN), loss-of-function mutations should be deleterious (HR > 1).

All imputed entries are stored as PROGNOSTIC_IN edges with tier=4 and
source='graph_imputed' for clear provenance.

Usage:
    python3 -m gnn.scripts.atlas_imputer              # impute + write to Neo4j
    python3 -m gnn.scripts.atlas_imputer --dry-run     # preview without writing
    python3 -m gnn.scripts.atlas_imputer --export atlas_imputed.csv  # also save CSV
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase
from gnn.config import CHANNEL_MAP, ALL_GENES, CHANNEL_NAMES

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")

# Shrinkage toward neutral (HR=1.0): imputed values are pulled toward 1.0
# Higher = more conservative. 0.6 means imputed HR = 0.6 * source_HR + 0.4 * 1.0
SHRINKAGE = 0.5

# Weights for combining signals
WEIGHTS = {
    "ppi_neighbor": 0.35,
    "same_channel": 0.25,
    "cooccurrence": 0.15,
    "expression": 0.15,
    "depmap": 0.10,
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def load_existing_atlas(driver):
    """Load all existing PROGNOSTIC_IN edges (tier 1-3) from Neo4j."""
    atlas = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)-[r:PROGNOSTIC_IN]->(ct:CancerType)
            WHERE r.tier <= 3
            RETURN g.name AS gene, ct.name AS cancer_type,
                   r.hr AS hr, r.tier AS tier, r.ci_width AS ci_width,
                   r.n_with AS n_with, r.channel AS channel
        """)
        for r in result:
            key = (r["cancer_type"], r["gene"])
            entry = {
                "hr": r["hr"],
                "tier": r["tier"],
                "ci_width": r["ci_width"] or 1.0,
                "n_with": r["n_with"] or 0,
                "channel": r["channel"] or CHANNEL_MAP.get(r["gene"], ""),
            }
            # Keep best tier per (ct, gene)
            if key not in atlas or entry["tier"] < atlas[key]["tier"]:
                atlas[key] = entry
    return atlas


def load_ppi_neighbors(driver):
    """Load PPI edges: gene -> {neighbor: weight}."""
    neighbors = defaultdict(dict)
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[r:PPI]-(g2:Gene)
            RETURN g1.name AS gene1, g2.name AS gene2,
                   r.score AS score
        """)
        for r in result:
            score = r["score"] or 0.4  # already 0-1 scale
            neighbors[r["gene1"]][r["gene2"]] = score
            neighbors[r["gene2"]][r["gene1"]] = score
    return dict(neighbors)


def load_cooccurrence(driver):
    """Load co-occurrence edges: (gene1, gene2) -> count."""
    cooc = defaultdict(dict)
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene)
            RETURN g1.name AS gene1, g2.name AS gene2, r.count AS count
        """)
        for r in result:
            cooc[r["gene1"]][r["gene2"]] = r["count"] or 0
    return dict(cooc)


def load_expression_context(driver):
    """Load expression edges: (gene, cancer_type) -> z_score."""
    expr = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)-[r:EXPRESSION_IN]->(ct:CancerType)
            RETURN g.name AS gene, ct.name AS cancer_type, r.z_score AS z
        """)
        for r in result:
            expr[(r["cancer_type"], r["gene"])] = r["z"] or 0.0
    return expr


def load_depmap_essentiality(driver):
    """Load DepMap: gene -> {lineage: dependency_score}."""
    depmap = defaultdict(dict)
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)-[r:ESSENTIAL_IN]->(l:Lineage)
            RETURN g.name AS gene, l.name AS lineage, r.dependency_score AS effect
        """)
        for r in result:
            depmap[r["gene"]][r["lineage"]] = r["effect"] or -0.5
    return dict(depmap)


def load_tcga_to_depmap_lineage():
    """Map atlas cancer type names to DepMap lineages."""
    from gnn.data.treatment_dataset import TCGA_TO_DEPMAP_LINEAGE
    # Atlas uses full names, we need to map them
    # Build reverse: TCGA abbreviation -> atlas name -> lineage
    # But we also need atlas_name -> lineage directly
    return {
        "Breast Cancer": "Breast",
        "Non-Small Cell Lung Cancer": "Lung",
        "Colorectal Cancer": "Bowel",
        "Ovarian Cancer": "Ovary/Fallopian Tube",
        "Glioblastoma": "CNS/Brain",
        "Lower Grade Glioma": "CNS/Brain",
        "Melanoma": "Skin",
        "Stomach Cancer": "Esophagus/Stomach",
        "Esophageal Cancer": "Esophagus/Stomach",
        "Bladder Urothelial Carcinoma": "Bladder/Urinary Tract",
        "Head and Neck Squamous Cell Carcinoma": "Head and Neck",
        "Liver Hepatocellular Carcinoma": "Liver",
        "Kidney Renal Clear Cell Carcinoma": "Kidney",
        "Kidney Renal Papillary Cell Carcinoma": "Kidney",
        "Kidney Chromophobe": "Kidney",
        "Pancreatic Cancer": "Pancreas",
        "Prostate Cancer": "Prostate",
        "Endometrial Cancer": "Uterus",
        "Uterine Carcinosarcoma": "Uterus",
        "Cervical Cancer": "Cervix",
        "Thyroid Cancer": "Thyroid",
        "Sarcoma": "Soft Tissue",
        "Acute Myeloid Leukemia": "Myeloid",
        "Adrenocortical Carcinoma": "Adrenal Gland",
        "Cholangiocarcinoma": "Biliary Tract",
        "Diffuse Large B-Cell Lymphoma": "Lymphoid",
        "Mesothelioma": "Pleura",
        "Uveal Melanoma": "Eye",
        "Pheochromocytoma and Paraganglioma": "Peripheral Nervous System",
        "Thymoma": "Thyroid",
        "Testicular Cancer": "Testis",
    }


def find_missing_pairs(driver, atlas):
    """Find (cancer_type, gene) pairs that patients need but atlas doesn't have."""
    missing = set()
    with driver.session() as s:
        result = s.run("""
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)
            MATCH (p)-[:MEMBER_OF]->(mg:MutationGroup)
            WITH g.name AS gene, p.cancer_type AS ct_raw, count(DISTINCT p) AS n_patients
            WHERE n_patients >= 3
            RETURN gene, ct_raw, n_patients
        """)
        for r in result:
            gene = r["gene"]
            ct_raw = r["ct_raw"]
            n = r["n_patients"]
            if gene in CHANNEL_MAP and ct_raw:
                # Try to match to atlas cancer type names
                # The atlas uses full names; patient nodes use TCGA abbreviations
                # We need both directions
                if (ct_raw, gene) not in atlas:
                    missing.add((ct_raw, gene, n))
    return missing


def impute_hr(gene, cancer_type, atlas, ppi, cooc, expr, depmap,
              ct_to_lineage, channel_map):
    """Impute HR for a missing (cancer_type, gene) pair by walking graph edges.

    Returns (imputed_hr, confidence, evidence_sources).
    """
    channel = channel_map.get(gene, "")
    signals = []
    sources = []

    # 1. PPI neighbor transfer
    if gene in ppi:
        neighbor_hrs = []
        for neighbor, ppi_score in ppi[gene].items():
            key = (cancer_type, neighbor)
            if key in atlas:
                # Weight by PPI confidence and atlas entry quality
                entry = atlas[key]
                weight = ppi_score * (1.0 / max(entry["ci_width"], 0.1))
                neighbor_hrs.append((entry["hr"], weight))

        if neighbor_hrs:
            total_weight = sum(w for _, w in neighbor_hrs)
            weighted_hr = sum(hr * w for hr, w in neighbor_hrs) / total_weight
            signals.append(("ppi_neighbor", weighted_hr, WEIGHTS["ppi_neighbor"]))
            sources.append(f"ppi({len(neighbor_hrs)} neighbors)")

    # 2. Same-channel transfer
    if channel:
        channel_hrs = []
        for (ct, g), entry in atlas.items():
            if ct == cancer_type and channel_map.get(g) == channel and g != gene:
                channel_hrs.append(entry["hr"])
        if channel_hrs:
            median_hr = float(np.median(channel_hrs))
            signals.append(("same_channel", median_hr, WEIGHTS["same_channel"]))
            sources.append(f"channel({channel},{len(channel_hrs)})")

    # 3. Co-occurrence signal
    if gene in cooc:
        cooc_hrs = []
        for partner, count in cooc[gene].items():
            key = (cancer_type, partner)
            if key in atlas and count >= 10:
                cooc_hrs.append((atlas[key]["hr"], count))
        if cooc_hrs:
            total_w = sum(c for _, c in cooc_hrs)
            weighted_hr = sum(hr * c for hr, c in cooc_hrs) / total_w
            signals.append(("cooccurrence", weighted_hr, WEIGHTS["cooccurrence"]))
            sources.append(f"cooc({len(cooc_hrs)} partners)")

    # 4. Expression context
    expr_key = (cancer_type, gene)
    if expr_key in expr:
        z = expr[expr_key]
        # Overexpressed → likely oncogene → HR > 1
        # Underexpressed → likely TSG → depends on context
        # Map z-score to HR estimate: z > 0 → HR slightly > 1
        expr_hr = np.exp(0.1 * z)  # mild effect
        signals.append(("expression", float(expr_hr), WEIGHTS["expression"]))
        sources.append(f"expr(z={z:.1f})")

    # 5. DepMap essentiality
    lineage = ct_to_lineage.get(cancer_type)
    if lineage and gene in depmap and lineage in depmap[gene]:
        dep_score = depmap[gene][lineage]
        # More negative = more essential = loss is more harmful
        # Map: dep_score -1.0 → HR ~1.5, dep_score 0.0 → HR 1.0
        dep_hr = np.exp(-0.4 * dep_score)  # negative dep → higher HR
        signals.append(("depmap", float(dep_hr), WEIGHTS["depmap"]))
        sources.append(f"depmap({dep_score:.2f})")

    if not signals:
        return None, 0.0, []

    # Weighted combination
    total_weight = sum(w for _, _, w in signals)
    raw_hr = sum(hr * w for _, hr, w in signals) / total_weight

    # Shrink toward neutral (HR = 1.0)
    imputed_hr = SHRINKAGE * raw_hr + (1 - SHRINKAGE) * 1.0

    # Confidence: more signals + higher weights = more confident
    confidence = min(1.0, total_weight / sum(WEIGHTS.values()) * len(signals) / 3)

    return imputed_hr, confidence, sources


def run_imputation(dry_run=False, export_path=None):
    """Run the full imputation pipeline."""
    driver = get_driver()

    print("Loading existing atlas from Neo4j...", flush=True)
    atlas = load_existing_atlas(driver)
    print(f"  {len(atlas)} existing entries (tier 1-3)", flush=True)

    print("Loading graph signals from Neo4j...", flush=True)
    ppi = load_ppi_neighbors(driver)
    print(f"  PPI: {len(ppi)} genes with neighbors", flush=True)

    cooc = load_cooccurrence(driver)
    print(f"  Co-occurrence: {len(cooc)} genes", flush=True)

    expr = load_expression_context(driver)
    print(f"  Expression: {len(expr)} (gene, cancer_type) pairs", flush=True)

    depmap = load_depmap_essentiality(driver)
    print(f"  DepMap: {len(depmap)} genes", flush=True)

    ct_to_lineage = load_tcga_to_depmap_lineage()

    # Find missing pairs from patient data
    print("\nFinding missing (cancer_type, gene) pairs...", flush=True)
    missing = find_missing_pairs(driver, atlas)
    print(f"  {len(missing)} missing pairs (>=3 patients each)", flush=True)

    # Also find missing pairs from TCGA mutation data
    # (patients may not be in Neo4j yet if they're wild-type in the graph)
    tcga_mut_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cache", "tcga", "tcga_mutations.csv",
    )
    if os.path.exists(tcga_mut_path):
        tcga_mut = pd.read_csv(tcga_mut_path)
        # Map TCGA abbreviations to atlas cancer type names
        from gnn.data.treatment_dataset import TreatmentDataset
        ds = TreatmentDataset.__new__(TreatmentDataset)
        # Use the cancer type mapping
        tcga_to_atlas = {
            'ACC': 'Adrenocortical Carcinoma', 'BLCA': 'Bladder Urothelial Carcinoma',
            'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer',
            'CHOL': 'Cholangiocarcinoma', 'COADREAD': 'Colorectal Cancer',
            'DLBC': 'Diffuse Large B-Cell Lymphoma', 'ESCA': 'Esophageal Cancer',
            'GBM': 'Glioblastoma', 'HNSC': 'Head and Neck Squamous Cell Carcinoma',
            'KICH': 'Kidney Chromophobe', 'KIRC': 'Kidney Renal Clear Cell Carcinoma',
            'KIRP': 'Kidney Renal Papillary Cell Carcinoma',
            'LAML': 'Acute Myeloid Leukemia', 'LGG': 'Lower Grade Glioma',
            'LIHC': 'Liver Hepatocellular Carcinoma',
            'LUAD': 'Non-Small Cell Lung Cancer', 'LUSC': 'Non-Small Cell Lung Cancer',
            'MESO': 'Mesothelioma', 'OV': 'Ovarian Cancer',
            'PAAD': 'Pancreatic Cancer', 'PCPG': 'Pheochromocytoma and Paraganglioma',
            'PRAD': 'Prostate Cancer', 'SARC': 'Sarcoma',
            'SKCM': 'Melanoma', 'STAD': 'Stomach Cancer',
            'TGCT': 'Testicular Cancer', 'THCA': 'Thyroid Cancer',
            'THYM': 'Thymoma', 'UCEC': 'Endometrial Cancer',
            'UCS': 'Uterine Carcinosarcoma', 'UVM': 'Uveal Melanoma',
        }
        for ct_abbrev, ct_full in tcga_to_atlas.items():
            ct_muts = tcga_mut[tcga_mut["cancer_type"] == ct_abbrev]
            for gene in ct_muts["gene"].unique():
                if gene in CHANNEL_MAP and (ct_full, gene) not in atlas:
                    n = len(ct_muts[ct_muts["gene"] == gene]["patient_id"].unique())
                    if n >= 3:
                        missing.add((ct_full, gene, n))

    print(f"  {len(missing)} total missing pairs after adding TCGA", flush=True)

    # Impute
    print("\nImputing...", flush=True)
    imputed = []
    n_success = 0
    n_fail = 0

    for cancer_type, gene, n_patients in sorted(missing, key=lambda x: -x[2]):
        hr, confidence, sources = impute_hr(
            gene, cancer_type, atlas, ppi, cooc, expr, depmap,
            ct_to_lineage, CHANNEL_MAP,
        )
        if hr is not None and confidence > 0.1:
            imputed.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "hr": round(hr, 4),
                "confidence": round(confidence, 3),
                "n_patients": n_patients,
                "sources": "; ".join(sources),
                "channel": CHANNEL_MAP.get(gene, ""),
            })
            n_success += 1
        else:
            n_fail += 1

    print(f"  Imputed: {n_success}, no signal: {n_fail}", flush=True)

    # Show top imputed entries
    imputed.sort(key=lambda x: -x["n_patients"])
    print(f"\nTop 20 imputed entries by patient count:")
    print(f"  {'Cancer Type':40s} {'Gene':10s} {'HR':>6s} {'Conf':>5s} {'#Pts':>6s}  Sources")
    print(f"  {'-'*40} {'-'*10} {'-'*6} {'-'*5} {'-'*6}  {'-'*30}")
    for entry in imputed[:20]:
        print(f"  {entry['cancer_type']:40s} {entry['gene']:10s} "
              f"{entry['hr']:6.3f} {entry['confidence']:5.2f} {entry['n_patients']:6d}  "
              f"{entry['sources']}")

    # Coverage stats
    new_atlas = dict(atlas)
    for entry in imputed:
        new_atlas[(entry["cancer_type"], entry["gene"])] = entry
    print(f"\nCoverage: {len(atlas)} -> {len(new_atlas)} (cancer_type, gene) pairs")

    # Write to Neo4j
    if not dry_run and imputed:
        print(f"\nWriting {len(imputed)} tier-4 entries to Neo4j...", flush=True)
        with driver.session() as s:
            for entry in imputed:
                s.run("""
                    MATCH (g:Gene {name: $gene})
                    MERGE (ct:CancerType {name: $cancer_type})
                    MERGE (g)-[r:PROGNOSTIC_IN {tier: 4}]->(ct)
                    SET r.hr = $hr, r.confidence = $confidence,
                        r.n_patients = $n_patients, r.sources = $sources,
                        r.channel = $channel, r.source = 'graph_imputed',
                        r.ci_width = $ci_width
                """, gene=entry["gene"], cancer_type=entry["cancer_type"],
                     hr=entry["hr"], confidence=entry["confidence"],
                     n_patients=entry["n_patients"], sources=entry["sources"],
                     channel=entry["channel"],
                     ci_width=2.0 / max(entry["confidence"], 0.01))  # wider CI for low confidence
            print("  Done.", flush=True)
    elif dry_run:
        print(f"\n[DRY RUN] Would write {len(imputed)} tier-4 entries to Neo4j", flush=True)

    # Export CSV
    if export_path and imputed:
        df = pd.DataFrame(imputed)
        df.to_csv(export_path, index=False)
        print(f"\nExported to {export_path}", flush=True)

    # Final summary
    print(f"\n{'='*60}")
    print(f"ATLAS IMPUTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Existing atlas entries (tier 1-3): {len(atlas)}")
    print(f"  New imputed entries (tier 4):      {len(imputed)}")
    print(f"  Total coverage:                    {len(new_atlas)}")
    print(f"  Missing pairs with no signal:      {n_fail}")
    total_patients = sum(e["n_patients"] for e in imputed)
    print(f"  Patients now covered by tier 4:    ~{total_patients}")

    driver.close()
    return imputed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--export", type=str, default=None, help="Export imputed entries to CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_imputation(dry_run=args.dry_run, export_path=args.export)
