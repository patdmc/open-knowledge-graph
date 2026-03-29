"""
Shared Neo4j graph snapshot — single source of truth for all atlas and enrichment data.

Every script that needs atlas tiers, expression, CNA, DepMap, SL, or CIViC lookups
should import from here instead of loading CSVs directly.

Usage:
    from gnn.data.graph_snapshot import load_atlas, load_enrichment, get_driver

    # Atlas only (fast)
    t1, t2, t3, t4, cancer_types = load_atlas()

    # Full enrichment (atlas + expression + CNA + DepMap + SL + CIViC)
    snapshot = load_enrichment()
    snapshot['t1'], snapshot['expr_lookup'], etc.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CIVIC_CACHE = os.path.join(CACHE_DIR, "oncokb")

# Module-level cache so we don't re-query within the same process
_snapshot_cache = {}
_snapshot_ts = 0
_CACHE_TTL = 300  # 5 minutes


def get_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def load_atlas(driver=None):
    """Load atlas tiers 1-4 from Neo4j PROGNOSTIC_IN edges.

    Returns (t1, t2, t3, t4, atlas_cancer_types).

    t1: {(cancer_type, gene, protein_change): entry}
    t2: {(cancer_type, gene): entry}
    t3: {(cancer_type, channel): entry}
    t4: {(cancer_type, gene): entry}  # graph-imputed
    atlas_cancer_types: set of cancer type names
    """
    close_driver = driver is None
    if driver is None:
        driver = get_driver()

    t1, t2, t3, t4 = {}, {}, {}, {}
    atlas_cancer_types = set()

    with driver.session() as s:
        # Gene -> CancerType (tiers 1, 2, 4)
        result = s.run("""
            MATCH (g:Gene)-[r:PROGNOSTIC_IN]->(ct:CancerType)
            RETURN g.name AS gene, ct.name AS cancer_type,
                   r.tier AS tier, r.hr AS hr,
                   r.ci_width AS ci_width, r.n_with AS n_with,
                   r.channel AS channel, r.protein_change AS protein_change,
                   r.confidence AS confidence
        """)
        for rec in result:
            entry = {
                "hr": rec["hr"],
                "ci_width": rec["ci_width"] or 1.0,
                "tier": rec["tier"],
                "n_with": rec["n_with"] or 0,
            }
            tier = rec["tier"]
            ct = rec["cancer_type"]
            gene = rec["gene"]
            atlas_cancer_types.add(ct)

            if tier == 1:
                t1[(ct, gene, rec["protein_change"] or "")] = entry
            elif tier == 2:
                t2[(ct, gene)] = entry
            elif tier == 4:
                entry["confidence"] = rec["confidence"] or 0.0
                t4[(ct, gene)] = entry

        # Channel -> CancerType (tier 3)
        result = s.run("""
            MATCH (ch:Channel)-[r:PROGNOSTIC_IN]->(ct:CancerType)
            WHERE r.tier = 3
            RETURN ch.name AS channel, ct.name AS cancer_type,
                   r.hr AS hr, r.ci_width AS ci_width, r.n_with AS n_with
        """)
        for rec in result:
            t3[(rec["cancer_type"], rec["channel"])] = {
                "hr": rec["hr"],
                "ci_width": rec["ci_width"] or 1.0,
                "tier": 3,
                "n_with": rec["n_with"] or 0,
            }
            atlas_cancer_types.add(rec["cancer_type"])

    if close_driver:
        driver.close()

    return t1, t2, t3, t4, atlas_cancer_types


def load_enrichment(driver=None):
    """Load full graph snapshot: atlas + all enrichment data.

    Returns dict with keys:
        t1, t2, t3, t4, atlas_cancer_types,
        expr_lookup, cna_lookup, depmap_lookup, sl_lookup,
        civic_sens, civic_res, civic_func,
        ppi_neighbors, cooccurrence
    """
    global _snapshot_cache, _snapshot_ts

    # Check cache
    if _snapshot_cache and (time.time() - _snapshot_ts) < _CACHE_TTL:
        return _snapshot_cache

    close_driver = driver is None
    if driver is None:
        driver = get_driver()

    snapshot = {}

    # Atlas
    t1, t2, t3, t4, cts = load_atlas(driver)
    snapshot.update({
        "t1": t1, "t2": t2, "t3": t3, "t4": t4,
        "atlas_cancer_types": cts,
    })

    with driver.session() as s:
        # Expression
        expr = {}
        result = s.run("""
            MATCH (g:Gene)-[r:EXPRESSION_IN]->(ct:CancerType)
            RETURN g.name AS gene, ct.name AS ct, r.z_score AS z
        """)
        for rec in result:
            if rec["z"] is not None:
                expr[(rec["gene"], rec["ct"])] = float(rec["z"])
        snapshot["expr_lookup"] = expr

        # CNA
        cna = {}
        result = s.run("""
            MATCH (g:Gene)-[r:CNA_IN]->(ct:CancerType)
            RETURN g.name AS gene, ct.name AS ct,
                   r.amp_freq AS amp, r.del_freq AS del, r.mean_cna AS mean_cna
        """)
        for rec in result:
            cna[(rec["gene"], rec["ct"])] = {
                "mean_cna": float(rec["mean_cna"] or 0),
                "amp_freq": float(rec["amp"] or 0),
                "del_freq": float(rec["del"] or 0),
            }
        snapshot["cna_lookup"] = cna

        # DepMap
        depmap = {}
        result = s.run("""
            MATCH (g:Gene)-[r:ESSENTIAL_IN]->(l:Lineage)
            RETURN g.name AS gene, l.name AS lineage, r.dependency_score AS score
        """)
        for rec in result:
            if rec["score"] is not None:
                depmap[(rec["gene"], rec["lineage"])] = float(rec["score"])
        snapshot["depmap_lookup"] = depmap

        # SL partners
        sl = {}
        result = s.run("""
            MATCH (g1:Gene)-[:SL_PARTNER]-(g2:Gene)
            RETURN g1.name AS a, g2.name AS b
        """)
        for rec in result:
            sl.setdefault(rec["a"], set()).add(rec["b"])
        snapshot["sl_lookup"] = sl

        # CIViC sensitivity/resistance
        civic_sens, civic_res = {}, {}
        result = s.run("""
            MATCH (g:Gene)-[r:HAS_SENSITIVITY_EVIDENCE]->(t)
            RETURN g.name AS gene, r.protein_change AS pc
        """)
        for rec in result:
            civic_sens[(rec["gene"], rec["pc"] or "")] = True
        result = s.run("""
            MATCH (g:Gene)-[r:HAS_RESISTANCE_EVIDENCE]->(t)
            RETURN g.name AS gene, r.protein_change AS pc
        """)
        for rec in result:
            civic_res[(rec["gene"], rec["pc"] or "")] = True
        snapshot["civic_sens"] = civic_sens
        snapshot["civic_res"] = civic_res

        # PPI neighbors
        ppi = {}
        result = s.run("""
            MATCH (g1:Gene)-[r:PPI]-(g2:Gene)
            RETURN g1.name AS a, g2.name AS b, r.score AS score
        """)
        for rec in result:
            score = rec["score"] or 0.4
            ppi.setdefault(rec["a"], {})[rec["b"]] = score
        snapshot["ppi_neighbors"] = ppi

        # Co-occurrence
        cooc = {}
        result = s.run("""
            MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene)
            RETURN g1.name AS a, g2.name AS b, r.count AS count
        """)
        for rec in result:
            cooc.setdefault(rec["a"], {})[rec["b"]] = rec["count"] or 0
        snapshot["cooccurrence"] = cooc

    if close_driver:
        driver.close()

    # CIViC GOF/LOF from CSV (not yet edges)
    civic_func = {}
    path = os.path.join(CIVIC_CACHE, "variant_functional_map.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            gene, pc = row["gene"], row["protein_change"]
            effects = str(row.get("oncogenic_effects", ""))
            directions = str(row.get("evidence_directions", ""))
            if "Sensitivity" in effects and "Supports" in directions:
                civic_sens[(gene, pc)] = True
            if "Resistance" in effects and "Supports" in directions:
                civic_res[(gene, pc)] = True
            inf = row.get("inferred_function", "")
            if isinstance(inf, str) and inf in ("GOF", "LOF"):
                civic_func[(gene, pc)] = inf
    snapshot["civic_func"] = civic_func

    # Cache
    _snapshot_cache = snapshot
    _snapshot_ts = time.time()

    return snapshot


# Convenience: TCGA abbreviation -> atlas cancer type name mapping
TCGA_TO_ATLAS_CT = {
    "ACC": "Adrenocortical Carcinoma",
    "BLCA": "Bladder Cancer",
    "BRCA": "Breast Cancer",
    "CESC": "Cervical Cancer",
    "CHOL": "Cholangiocarcinoma",
    "COADREAD": "Colorectal Cancer",
    "DLBC": "Diffuse Large B-Cell Lymphoma",
    "ESCA": "Esophagogastric Cancer",
    "GBM": "Glioblastoma",
    "HNSC": "Head and Neck Cancer",
    "KICH": "Renal Cell Carcinoma",
    "KIRC": "Renal Cell Carcinoma",
    "KIRP": "Renal Cell Carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG": "Low-Grade Glioma",
    "LIHC": "Hepatobiliary Cancer",
    "LUAD": "Non-Small Cell Lung Cancer",
    "LUSC": "Non-Small Cell Lung Cancer",
    "MESO": "Mesothelioma",
    "OV": "Ovarian Cancer",
    "PAAD": "Pancreatic Cancer",
    "PCPG": "Pheochromocytoma",
    "PRAD": "Prostate Cancer",
    "SARC": "Soft Tissue Sarcoma",
    "SKCM": "Melanoma",
    "STAD": "Esophagogastric Cancer",
    "TGCT": "Germ Cell Tumor",
    "THCA": "Thyroid Cancer",
    "THYM": "Thymic Epithelial Tumor",
    "UCEC": "Endometrial Cancer",
    "UCS": "Uterine Sarcoma",
    "UVM": "Uveal Melanoma",
}


def get_atlas_cancer_type(tcga_abbrev, atlas_cancer_types=None):
    """Map TCGA abbreviation to atlas cancer type name.

    If atlas_cancer_types is provided, validates against actual atlas entries
    with fuzzy matching fallback.
    """
    name = TCGA_TO_ATLAS_CT.get(tcga_abbrev)
    if not name:
        return None
    if atlas_cancer_types is None:
        return name
    if name in atlas_cancer_types:
        return name
    # Fuzzy match
    for act in atlas_cancer_types:
        if name.split()[0].lower() in act.lower():
            return act
    return None


def atlas_lookup(atlas_ct, gene, protein_change, channel, t1, t2, t3, t4):
    """Look up atlas entry for a mutation, falling through tiers 1 -> 2 -> 3 -> 4.

    Returns (hr, ci_width, tier, n_with) or (1.0, 0.0, 0, 0) if no match.
    """
    if atlas_ct:
        entry = None
        if (atlas_ct, gene, protein_change) in t1:
            entry = t1[(atlas_ct, gene, protein_change)]
        elif (atlas_ct, gene) in t2:
            entry = t2[(atlas_ct, gene)]
        elif channel and (atlas_ct, channel) in t3:
            entry = t3[(atlas_ct, channel)]
        elif (atlas_ct, gene) in t4:
            entry = t4[(atlas_ct, gene)]

        if entry is not None:
            return (
                entry["hr"],
                entry.get("ci_width", 1.0),
                entry["tier"],
                entry.get("n_with", 0),
            )

    return (1.0, 0.0, 0, 0)
