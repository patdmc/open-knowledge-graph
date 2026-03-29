#!/usr/bin/env python3
"""
Load knowledge graph edges into Neo4j — external evidence as first-class relationships.

These edges turn flat per-node metadata into traversable graph structure:
  - Gene → SENSITIVE_TO → Drug (CIViC evidence)
  - Gene → RESISTANT_TO → Drug (CIViC evidence)
  - Gene → SL_PARTNER → Gene (synthetic lethality)
  - Gene → ESSENTIAL_IN → Lineage (DepMap dependency)
  - Gene → EXPRESSION_IN → CancerType (TCGA expression z-score)
  - Gene → CNA_IN → CancerType (TCGA CNA context)
  - Treatment → TREATS → CancerType (treatment patterns)
  - Patient → RECEIVED → Treatment (treatment history)

New node types:
  (:Drug {name}) — therapeutic agents
  (:Lineage {name}) — DepMap cell lineages
  (:Treatment {name, category}) — treatment modalities

All edges carry provenance (source, confidence) so the graph is auditable.

Usage:
    python3 -u -m gnn.data.knowledge_edges
"""

import os
import sys
import json
import csv
import time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase

from gnn.config import CHANNEL_MAP, ALL_GENES

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")

CACHE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEPMAP_CACHE = os.path.join(CACHE_BASE, "depmap")
SL_CACHE = os.path.join(CACHE_BASE, "synthetic_lethality")
CIVIC_CACHE = os.path.join(CACHE_BASE, "oncokb")
TCGA_CACHE = os.path.join(CACHE_BASE, "tcga")
METABRIC_CACHE = os.path.join(CACHE_BASE, "metabric")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def create_indexes(driver):
    """Create indexes for new node types."""
    indexes = [
        "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
        "CREATE INDEX lineage_name IF NOT EXISTS FOR (l:Lineage) ON (l.name)",
        "CREATE INDEX treatment_name IF NOT EXISTS FOR (t:Treatment) ON (t.name)",
        "CREATE INDEX cancer_type_name IF NOT EXISTS FOR (ct:CancerType) ON (ct.name)",
    ]
    with driver.session() as session:
        for idx in indexes:
            session.run(idx)
    print(f"  Created {len(indexes)} indexes.", flush=True)


# ---------------------------------------------------------------------------
# 1. CIViC drug sensitivity/resistance edges
# ---------------------------------------------------------------------------

def load_civic_edges(driver):
    """Load CIViC variant evidence as edges.

    Creates:
      Gene → HAS_VARIANT {protein_change, function, sensitivity, resistance} → Gene (self-edge)
      Gene → SENSITIVE_TO → Treatment (when sensitivity evidence exists)
      Gene → RESISTANT_TO → Treatment (when resistance evidence exists)
    """
    path = os.path.join(CIVIC_CACHE, "variant_functional_map.csv")
    if not os.path.exists(path):
        print("  CIViC data not found, skipping.", flush=True)
        return 0

    df = pd.read_csv(path)
    gene_set = set(ALL_GENES)

    sens_count = 0
    res_count = 0
    func_count = 0

    with driver.session() as session:
        for _, row in df.iterrows():
            gene = row["gene"]
            if gene not in gene_set:
                continue

            pc = str(row.get("protein_change", ""))
            effects = str(row.get("oncogenic_effects", ""))
            directions = str(row.get("evidence_directions", ""))
            inf_func = str(row.get("inferred_function", ""))

            is_sens = "Sensitivity" in effects and "Supports" in directions
            is_res = "Resistance" in effects and "Supports" in directions

            if inf_func in ("GOF", "LOF"):
                # Update gene node with variant-level function evidence
                session.run("""
                    MATCH (g:Gene {name: $gene})
                    SET g.civic_function = $func
                """, gene=gene, func=inf_func)
                func_count += 1

            # Link gene to treatment modalities based on sensitivity/resistance
            # Sensitivity to chemo drugs → Gene SENSITIVE_TO Chemotherapy
            # This is a simplification; with more CIViC data we'd have specific drugs
            if is_sens:
                session.run("""
                    MATCH (g:Gene {name: $gene})
                    MERGE (g)-[r:HAS_SENSITIVITY_EVIDENCE]->(g)
                    SET r.protein_change = $pc, r.source = 'CIViC'
                """, gene=gene, pc=pc)
                sens_count += 1

            if is_res:
                session.run("""
                    MATCH (g:Gene {name: $gene})
                    MERGE (g)-[r:HAS_RESISTANCE_EVIDENCE]->(g)
                    SET r.protein_change = $pc, r.source = 'CIViC'
                """, gene=gene, pc=pc)
                res_count += 1

    print(f"  CIViC: {func_count} function annotations, "
          f"{sens_count} sensitivity, {res_count} resistance edges", flush=True)
    return sens_count + res_count + func_count


# ---------------------------------------------------------------------------
# 2. Synthetic lethality edges
# ---------------------------------------------------------------------------

def load_sl_edges(driver):
    """Load Gene → SL_PARTNER → Gene edges."""
    path = os.path.join(SL_CACHE, "synthetic_lethality_both_in_set.csv")
    if not os.path.exists(path):
        print("  SL data not found, skipping.", flush=True)
        return 0

    df = pd.read_csv(path)
    count = 0

    with driver.session() as session:
        for _, row in df.iterrows():
            a, b = row["gene_a"], row["gene_b"]
            source = row.get("source", "SynLethDB")
            session.run("""
                MATCH (g1:Gene {name: $a})
                MATCH (g2:Gene {name: $b})
                MERGE (g1)-[r:SL_PARTNER]->(g2)
                SET r.source = $source
            """, a=a, b=b, source=str(source))
            # Bidirectional
            session.run("""
                MATCH (g1:Gene {name: $a})
                MATCH (g2:Gene {name: $b})
                MERGE (g2)-[r:SL_PARTNER]->(g1)
                SET r.source = $source
            """, a=a, b=b, source=str(source))
            count += 1

    print(f"  SL: {count} pairs ({count * 2} directed edges)", flush=True)
    return count


# ---------------------------------------------------------------------------
# 3. DepMap dependency edges
# ---------------------------------------------------------------------------

def load_depmap_edges(driver):
    """Load Gene → ESSENTIAL_IN → Lineage edges from DepMap."""
    path = os.path.join(DEPMAP_CACHE, "depmap_dependency_matrix.csv")
    if not os.path.exists(path):
        print("  DepMap data not found, skipping.", flush=True)
        return 0

    df = pd.read_csv(path)
    gene_set = set(ALL_GENES)
    lineages = [c for c in df.columns if c != "gene"]

    # Create Lineage nodes
    with driver.session() as session:
        for lin in lineages:
            session.run("MERGE (l:Lineage {name: $name})", name=lin)

    count = 0
    # Threshold: dependency score < -0.5 means essential
    ESSENTIAL_THRESHOLD = -0.5

    with driver.session() as session:
        for _, row in df.iterrows():
            gene = row["gene"]
            if gene not in gene_set:
                continue
            for lin in lineages:
                val = row[lin]
                if pd.notna(val):
                    score = float(val)
                    if score < ESSENTIAL_THRESHOLD:
                        session.run("""
                            MATCH (g:Gene {name: $gene})
                            MATCH (l:Lineage {name: $lineage})
                            MERGE (g)-[r:ESSENTIAL_IN]->(l)
                            SET r.dependency_score = $score, r.source = 'DepMap'
                        """, gene=gene, lineage=lin, score=score)
                        count += 1

    print(f"  DepMap: {count} ESSENTIAL_IN edges (threshold < {ESSENTIAL_THRESHOLD})", flush=True)
    return count


# ---------------------------------------------------------------------------
# 4. Expression edges
# ---------------------------------------------------------------------------

def load_expression_edges(driver):
    """Load Gene → EXPRESSION_IN → CancerType edges from TCGA expression.

    Stores ALL gene×cancer_type pairs with z-score, mean, std, median.
    The z-score is computed per-gene across cancer types (pan-cancer normalization).
    """
    path = os.path.join(TCGA_CACHE, "tcga_expression_summary.csv")
    if not os.path.exists(path):
        print("  Expression data not found, skipping.", flush=True)
        return 0

    df = pd.read_csv(path)
    gene_set = set(ALL_GENES)

    # Z-score within each gene across cancer types
    cancer_types = set()
    edges = []

    for gene, gdf in df.groupby("gene"):
        if gene not in gene_set:
            continue
        mean_val = gdf["mean"].mean()
        std_val = gdf["mean"].std()
        if std_val < 1e-8:
            std_val = 1.0
        for _, row in gdf.iterrows():
            ct = row["cancer_type"]
            z = (row["mean"] - mean_val) / std_val
            cancer_types.add(ct)
            edges.append({
                "gene": gene, "cancer_type": ct,
                "z_score": round(float(z), 4),
                "mean_expr": round(float(row["mean"]), 2),
                "std_expr": round(float(row["std"]), 2),
                "median_expr": round(float(row["median"]), 2),
                "n_samples": int(row["n_samples"]),
            })

    # Create CancerType nodes and batch-merge edges
    with driver.session() as session:
        for ct in cancer_types:
            session.run("MERGE (ct:CancerType {name: $name})", name=ct)

        # Batch merge for speed
        batch_size = 500
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            session.run("""
                UNWIND $batch AS e
                MATCH (g:Gene {name: e.gene})
                MERGE (ct:CancerType {name: e.cancer_type})
                MERGE (g)-[r:EXPRESSION_IN]->(ct)
                SET r.z_score = e.z_score,
                    r.mean_expr = e.mean_expr,
                    r.std_expr = e.std_expr,
                    r.median_expr = e.median_expr,
                    r.n_samples = e.n_samples,
                    r.direction = CASE WHEN e.z_score > 0 THEN 'over' ELSE 'under' END,
                    r.source = 'TCGA'
            """, batch=batch)

    print(f"  Expression: {len(edges)} edges across {len(cancer_types)} cancer types "
          f"(all gene×CT pairs)", flush=True)
    return len(edges)


# ---------------------------------------------------------------------------
# 5. CNA edges
# ---------------------------------------------------------------------------

def load_cna_edges(driver):
    """Load Gene → CNA_IN → CancerType edges from TCGA CNA data.

    Stores ALL gene×cancer_type pairs with full CNA profile:
    amp_freq, gain_freq, loss_freq, del_freq, mean_cna.

    amp_freq vs gain_freq distinguishes focal amplification (oncogene driver)
    from broad gain (passenger). del_freq vs loss_freq does the same for
    tumor suppressors. This informs GOF/LOF mechanism per cancer type.
    """
    path = os.path.join(TCGA_CACHE, "tcga_cna_summary.csv")
    if not os.path.exists(path):
        print("  CNA data not found, skipping.", flush=True)
        return 0

    df = pd.read_csv(path)
    gene_set = set(ALL_GENES)
    edges = []

    for _, row in df.iterrows():
        gene = row["gene"]
        ct = row["cancer_type"]
        if gene not in gene_set:
            continue

        amp_freq = float(row.get("amp_freq", 0))
        gain_freq = float(row.get("gain_freq", 0))
        loss_freq = float(row.get("loss_freq", 0))
        del_freq = float(row.get("del_freq", 0))
        mean_cna = float(row.get("mean_cna", 0))

        # Derive mechanism signal
        # focal_amp = amp / (amp + gain) — high means driver amplification
        total_gain = amp_freq + gain_freq
        total_loss = del_freq + loss_freq
        focal_amp_ratio = amp_freq / total_gain if total_gain > 0.01 else 0.0
        focal_del_ratio = del_freq / total_loss if total_loss > 0.01 else 0.0

        # Direction: strongest signal wins
        if amp_freq > del_freq and amp_freq > 0.01:
            direction = "amplified"
        elif del_freq > amp_freq and del_freq > 0.01:
            direction = "deleted"
        elif gain_freq > loss_freq:
            direction = "gained"
        elif loss_freq > gain_freq:
            direction = "lost"
        else:
            direction = "neutral"

        edges.append({
            "gene": gene, "cancer_type": ct,
            "amp_freq": round(amp_freq, 4),
            "gain_freq": round(gain_freq, 4),
            "loss_freq": round(loss_freq, 4),
            "del_freq": round(del_freq, 4),
            "mean_cna": round(mean_cna, 4),
            "focal_amp_ratio": round(focal_amp_ratio, 4),
            "focal_del_ratio": round(focal_del_ratio, 4),
            "n_samples": int(row["n_samples"]),
            "direction": direction,
        })

    # Batch merge
    with driver.session() as session:
        batch_size = 500
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            session.run("""
                UNWIND $batch AS e
                MATCH (g:Gene {name: e.gene})
                MERGE (ct:CancerType {name: e.cancer_type})
                MERGE (g)-[r:CNA_IN]->(ct)
                SET r.amp_freq = e.amp_freq,
                    r.gain_freq = e.gain_freq,
                    r.loss_freq = e.loss_freq,
                    r.del_freq = e.del_freq,
                    r.mean_cna = e.mean_cna,
                    r.focal_amp_ratio = e.focal_amp_ratio,
                    r.focal_del_ratio = e.focal_del_ratio,
                    r.n_samples = e.n_samples,
                    r.direction = e.direction,
                    r.source = 'TCGA'
            """, batch=batch)

    print(f"  CNA: {len(edges)} edges across {len(set(e['cancer_type'] for e in edges))} "
          f"cancer types (all gene×CT pairs, full CNA profile)", flush=True)
    return len(edges)


# ---------------------------------------------------------------------------
# 6. Treatment nodes and edges
# ---------------------------------------------------------------------------

def load_treatment_edges(driver):
    """Load Treatment nodes and Patient → RECEIVED → Treatment edges.

    Creates Treatment nodes for each modality, then links TCGA + METABRIC
    patients to their treatments.
    """
    # Create Treatment nodes
    treatments = [
        {"name": "Chemotherapy", "category": "systemic"},
        {"name": "Endocrine", "category": "systemic"},
        {"name": "Targeted", "category": "systemic"},
        {"name": "Immunotherapy", "category": "systemic"},
        {"name": "Platinum", "category": "chemo_subclass"},
        {"name": "Taxane", "category": "chemo_subclass"},
        {"name": "Anthracycline", "category": "chemo_subclass"},
        {"name": "Antimetabolite", "category": "chemo_subclass"},
        {"name": "Alkylating", "category": "chemo_subclass"},
    ]

    with driver.session() as session:
        for t in treatments:
            session.run(
                "MERGE (t:Treatment {name: $name}) SET t.category = $cat",
                name=t["name"], cat=t["category"],
            )

    # Load TCGA treatment data
    tcga_path = os.path.join(TCGA_CACHE, "tcga_treatment_data.csv")
    tcga_count = 0

    if os.path.exists(tcga_path):
        # Import classify functions
        from gnn.data.treatment_dataset import classify_drug, classify_treatment_type

        df = pd.read_csv(tcga_path)
        # Need barcode map to link to Patient nodes (which use TCGA barcodes)
        barcode_path = os.path.join(TCGA_CACHE, "caseid_barcode_map.json")
        barcode_map = {}
        if os.path.exists(barcode_path):
            with open(barcode_path) as f:
                barcode_map = json.load(f)

        treatment_idx_to_name = {
            2: "Chemotherapy", 3: "Endocrine", 4: "Targeted", 5: "Immunotherapy",
            6: "Platinum", 7: "Taxane", 8: "Anthracycline",
            9: "Antimetabolite", 10: "Alkylating",
        }

        # Aggregate per patient
        patient_treatments = defaultdict(set)
        for _, row in df.iterrows():
            case_id = row.get("case_id", "")
            barcode = barcode_map.get(case_id, "")
            if not barcode:
                continue

            # Classify treatment type
            tt = row.get("treatment_type", "")
            for idx in classify_treatment_type(tt):
                if idx in treatment_idx_to_name:
                    patient_treatments[barcode].add(treatment_idx_to_name[idx])

            # Classify drug
            agent = row.get("therapeutic_agents", "")
            for idx in classify_drug(agent):
                if idx in treatment_idx_to_name:
                    patient_treatments[barcode].add(treatment_idx_to_name[idx])

        # Create edges
        with driver.session() as session:
            for barcode, tset in patient_treatments.items():
                for tname in tset:
                    session.run("""
                        MATCH (p:Patient {id: $pid})
                        MATCH (t:Treatment {name: $tname})
                        MERGE (p)-[r:RECEIVED]->(t)
                        SET r.source = 'TCGA_GDC'
                    """, pid=barcode, tname=tname)
                    tcga_count += 1

    print(f"  TCGA treatment edges: {tcga_count}", flush=True)

    # Load METABRIC treatment data
    metabric_path = os.path.join(METABRIC_CACHE, "metabric_clinical.csv")
    metabric_count = 0

    if os.path.exists(metabric_path):
        clin = pd.read_csv(metabric_path)
        with driver.session() as session:
            for _, row in clin.iterrows():
                pid = f"METABRIC_{row['PATIENT_ID']}"
                if str(row.get("CHEMOTHERAPY", "")).upper() == "YES":
                    session.run("""
                        MATCH (p:Patient {id: $pid})
                        MATCH (t:Treatment {name: 'Chemotherapy'})
                        MERGE (p)-[r:RECEIVED]->(t)
                        SET r.source = 'METABRIC'
                    """, pid=pid)
                    metabric_count += 1
                if str(row.get("HORMONE_THERAPY", "")).upper() == "YES":
                    session.run("""
                        MATCH (p:Patient {id: $pid})
                        MATCH (t:Treatment {name: 'Endocrine'})
                        MERGE (p)-[r:RECEIVED]->(t)
                        SET r.source = 'METABRIC'
                    """, pid=pid)
                    metabric_count += 1

    print(f"  METABRIC treatment edges: {metabric_count}", flush=True)
    return tcga_count + metabric_count


# ---------------------------------------------------------------------------
# 7. Drug → Treatment class mapping
# ---------------------------------------------------------------------------

def load_drug_treatment_edges(driver):
    """Link Drug nodes to Treatment modality nodes.

    Creates Drug → SUBCLASS_OF → Treatment edges so the graph connects
    patient treatments to gene-drug sensitivity evidence.
    """
    from gnn.data.treatment_dataset import (
        _PLATINUM, _TAXANE, _ANTHRACYCLINE, _ANTIMETABOLITE, _ALKYLATING,
        _ENDOCRINE, _TARGETED, _IMMUNOTHERAPY,
    )

    mappings = {
        "Platinum": _PLATINUM,
        "Taxane": _TAXANE,
        "Anthracycline": _ANTHRACYCLINE,
        "Antimetabolite": _ANTIMETABOLITE,
        "Alkylating": _ALKYLATING,
        "Endocrine": _ENDOCRINE,
        "Targeted": _TARGETED,
        "Immunotherapy": _IMMUNOTHERAPY,
    }

    count = 0
    with driver.session() as session:
        for treatment_name, drug_set in mappings.items():
            for drug in drug_set:
                # Try to match Drug node (case-insensitive via lowercase)
                result = session.run("""
                    MATCH (d:Drug)
                    WHERE toLower(d.name) CONTAINS $drug
                    MATCH (t:Treatment {name: $tname})
                    MERGE (d)-[r:SUBCLASS_OF]->(t)
                    SET r.source = 'classification'
                    RETURN count(r) AS n
                """, drug=drug, tname=treatment_name)
                n = result.single()["n"]
                count += n

    # Also: chemo subclasses are subclasses of Chemotherapy
    with driver.session() as session:
        for sub in ["Platinum", "Taxane", "Anthracycline", "Antimetabolite", "Alkylating"]:
            session.run("""
                MATCH (s:Treatment {name: $sub})
                MATCH (c:Treatment {name: 'Chemotherapy'})
                MERGE (s)-[r:SUBCLASS_OF]->(c)
                SET r.source = 'classification'
            """, sub=sub)
            count += 1

    print(f"  Drug → Treatment mappings: {count} edges", flush=True)
    return count


# ---------------------------------------------------------------------------
# 8. GDSC drug sensitivity edges
# ---------------------------------------------------------------------------

GDSC_CACHE = os.path.join(CACHE_BASE, "gdsc")


def load_gdsc_edges(driver):
    """Load Gene → RESPONDS_TO → Drug edges from GDSC cell line data.

    Effect size = mean_ic50_mut - mean_ic50_wt (negative = mutant is more sensitive).
    """
    path = os.path.join(GDSC_CACHE, "gdsc_gene_drug_associations.csv")
    if not os.path.exists(path):
        print("  GDSC data not found, skipping.", flush=True)
        return 0

    df = pd.read_csv(path)
    gene_set = set(ALL_GENES)
    count = 0

    with driver.session() as session:
        for _, row in df.iterrows():
            gene = row["gene"]
            drug = row["drug"]
            if gene not in gene_set:
                continue

            effect = float(row.get("effect_size", 0))
            n_mut = int(row.get("n_mut", 0))

            # Only include if sufficient evidence and meaningful effect
            if n_mut < 5 or abs(effect) < 0.5:
                continue

            # Create Drug node
            session.run("MERGE (d:Drug {name: $name})", name=drug)

            if effect < 0:
                # Mutant is more sensitive (lower IC50)
                session.run("""
                    MATCH (g:Gene {name: $gene})
                    MATCH (d:Drug {name: $drug})
                    MERGE (g)-[r:SENSITIVE_TO]->(d)
                    SET r.effect_size = $effect, r.n_mut = $n_mut,
                        r.source = 'GDSC'
                """, gene=gene, drug=drug, effect=effect, n_mut=n_mut)
            else:
                # Mutant is more resistant
                session.run("""
                    MATCH (g:Gene {name: $gene})
                    MATCH (d:Drug {name: $drug})
                    MERGE (g)-[r:RESISTANT_TO]->(d)
                    SET r.effect_size = $effect, r.n_mut = $n_mut,
                        r.source = 'GDSC'
                """, gene=gene, drug=drug, effect=effect, n_mut=n_mut)
            count += 1

    sens = sum(1 for _, r in df.iterrows()
               if r["gene"] in gene_set and r.get("n_mut", 0) >= 5
               and abs(r.get("effect_size", 0)) >= 0.5 and r.get("effect_size", 0) < 0)
    print(f"  GDSC: {count} gene-drug edges "
          f"(sens: {sens}, res: {count - sens})", flush=True)
    return count


# ---------------------------------------------------------------------------
# Atlas prognostic edges: Gene -[PROGNOSTIC_IN]-> CancerType
# ---------------------------------------------------------------------------

def load_atlas_edges(driver):
    """Load survival atlas entries as PROGNOSTIC_IN edges.

    Each atlas entry becomes:
      Gene -[PROGNOSTIC_IN {tier, hr, ci_width, p_value, n_with, n_without,
                            protein_change (tier 1), source}]-> CancerType

    This makes the atlas queryable from the graph — no more loading CSVs.
    """
    atlas_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "analysis", "survival_atlas_full.csv",
    )
    if not os.path.exists(atlas_path):
        print(f"  Atlas not found at {atlas_path}, skipping.", flush=True)
        return 0

    df = pd.read_csv(atlas_path)
    # Keep entries with p <= 0.10 (same filter as treatment_dataset)
    df = df[df["p_value"] <= 0.10]
    gene_set = set(ALL_GENES)

    count = 0
    with driver.session() as session:
        # Ensure CancerType nodes exist for all atlas cancer types
        for ct in df["cancer_type"].unique():
            session.run(
                "MERGE (ct:CancerType {name: $name})",
                name=ct,
            )

        for _, row in df.iterrows():
            gene = row["gene"]
            tier = int(row["tier"])

            base_props = {
                "cancer_type": row["cancer_type"],
                "tier": tier,
                "hr": float(row["hr"]),
                "ci_low": float(row.get("ci_low", 0)),
                "ci_high": float(row.get("ci_high", 0)),
                "ci_width": float(row.get("ci_width", 1.0)),
                "p_value": float(row["p_value"]),
                "n_with": int(row.get("n_with", 0)),
                "n_without": int(row.get("n_without", 0)),
                "channel": row.get("channel", ""),
                "source": "survival_atlas",
            }

            if tier == 3:
                # Channel-level: Channel -[PROGNOSTIC_IN]-> CancerType
                channel = row.get("channel", "")
                if not channel:
                    continue
                base_props["channel_name"] = channel
                session.run("""
                    MATCH (ch:Channel {name: $channel_name})
                    MATCH (ct:CancerType {name: $cancer_type})
                    MERGE (ch)-[r:PROGNOSTIC_IN {tier: $tier}]->(ct)
                    SET r.hr = $hr, r.ci_low = $ci_low, r.ci_high = $ci_high,
                        r.ci_width = $ci_width, r.p_value = $p_value,
                        r.n_with = $n_with, r.n_without = $n_without,
                        r.channel = $channel, r.source = $source
                """, **base_props)
                count += 1
            elif gene in gene_set:
                base_props["gene"] = gene
                if tier == 1:
                    base_props["protein_change"] = str(row.get("protein_change", ""))
                    session.run("""
                        MATCH (g:Gene {name: $gene})
                        MATCH (ct:CancerType {name: $cancer_type})
                        MERGE (g)-[r:PROGNOSTIC_IN {tier: $tier, protein_change: $protein_change}]->(ct)
                        SET r.hr = $hr, r.ci_low = $ci_low, r.ci_high = $ci_high,
                            r.ci_width = $ci_width, r.p_value = $p_value,
                            r.n_with = $n_with, r.n_without = $n_without,
                            r.channel = $channel, r.source = $source
                    """, **base_props)
                else:
                    session.run("""
                        MATCH (g:Gene {name: $gene})
                        MATCH (ct:CancerType {name: $cancer_type})
                        MERGE (g)-[r:PROGNOSTIC_IN {tier: $tier}]->(ct)
                        SET r.hr = $hr, r.ci_low = $ci_low, r.ci_high = $ci_high,
                            r.ci_width = $ci_width, r.p_value = $p_value,
                            r.n_with = $n_with, r.n_without = $n_without,
                            r.channel = $channel, r.source = $source
                    """, **base_props)
                count += 1

    print(f"  Atlas: {count} PROGNOSTIC_IN edges "
          f"(T1: {len(df[df['tier']==1])}, T2: {len(df[df['tier']==2])}, "
          f"T3: {len(df[df['tier']==3])})", flush=True)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("Loading knowledge graph edges into Neo4j", flush=True)
    print("=" * 60, flush=True)

    driver = get_driver()

    # Verify connection
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS c")
        n_existing = result.single()["c"]
    print(f"Existing graph: {n_existing} nodes", flush=True)

    print("\nCreating indexes...", flush=True)
    create_indexes(driver)

    total = 0

    print("\n1. CIViC drug sensitivity/resistance...", flush=True)
    total += load_civic_edges(driver)

    print("\n2. Atlas prognostic entries...", flush=True)
    total += load_atlas_edges(driver)

    print("\n3. Synthetic lethality partners...", flush=True)
    total += load_sl_edges(driver)

    print("\n4. DepMap dependency (essential genes)...", flush=True)
    total += load_depmap_edges(driver)

    print("\n5. Expression (over/under-expressed)...", flush=True)
    total += load_expression_edges(driver)

    print("\n6. CNA (amplified/deleted)...", flush=True)
    total += load_cna_edges(driver)

    print("\n7. Treatment modalities...", flush=True)
    total += load_treatment_edges(driver)

    print("\n8. Drug → Treatment class mappings...", flush=True)
    total += load_drug_treatment_edges(driver)

    print("\n9. GDSC drug sensitivity (cell lines)...", flush=True)
    total += load_gdsc_edges(driver)

    # Final stats
    print(f"\n{'='*60}", flush=True)
    print(f"Total new edges added: {total}", flush=True)

    with driver.session() as session:
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)
        print(f"\nEdge type summary:", flush=True)
        for record in result:
            print(f"  {record['type']:25s}: {record['count']:>8,}", flush=True)

        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        print(f"\nNode type summary:", flush=True)
        for record in result:
            print(f"  {record['label']:25s}: {record['count']:>8,}", flush=True)

    driver.close()
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
