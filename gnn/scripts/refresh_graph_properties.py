"""
Refresh all computed/derived properties on the graph from current state.

Any time edges or nodes change, denormalized values become stale.
This module recomputes everything from scratch:

  1. Gene channel_profile — from current PPI + BELONGS_TO
  2. Gene confidence — from current edge coverage
  3. Gene function — from CIViC + biallelic evidence
  4. HAS_MUTATION log_hr — from current atlas entries
  5. Patient tissue_delta — from refreshed log_hr
  6. MutationGroup stats — median_os, n_events, n_patients
  7. Invalidate disk caches — so next read picks up fresh state

Usage:
    python3 -u -m gnn.scripts.refresh_graph_properties [--dry-run]

Called automatically by the feedback loop after any graph mutation step.
"""

import os
import sys
import time
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import ALL_GENES, CHANNEL_MAP, CHANNEL_NAMES, GNN_CACHE


def _driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver("bolt://localhost:7687",
                                auth=("neo4j", "openknowledgegraph"))


def _step(name, actual=None, elapsed=None):
    status = f"{actual:,}" if actual is not None else ""
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"    {name:.<50s} {status:>12s}  {t}", flush=True)


# =========================================================================
# 0. Sync Gene.channel = Gene.primary_channel
# =========================================================================

def sync_channel_property(session):
    """Ensure Gene.channel always matches Gene.primary_channel.

    primary_channel is derived from BELONGS_TO weights (data-driven).
    channel is the property read by config.py and all downstream code.
    If they diverge, everything downstream (CHANNEL_MAP, block assignments,
    feature encoding, ridge features) uses the wrong channel.
    """
    t0 = time.time()
    result = session.run("""
        MATCH (g:Gene) WHERE g.channel <> g.primary_channel
        SET g.channel = g.primary_channel
        RETURN count(g) AS fixed
    """)
    n_fixed = result.single()["fixed"]
    _step("Gene.channel synced to primary_channel", actual=n_fixed, elapsed=time.time() - t0)
    return n_fixed


# =========================================================================
# 1. Gene channel_profile — recompute from PPI + BELONGS_TO
# =========================================================================

def refresh_channel_profiles(session):
    """Recompute each gene's 8-dim channel_profile from current graph."""
    t0 = time.time()

    # Get current BELONGS_TO weights
    result = session.run("""
        MATCH (g:Gene)-[b:BELONGS_TO]->(c:Channel)
        RETURN g.name AS gene, c.name AS channel, b.weight AS weight
    """)
    gene_channel = defaultdict(dict)
    for r in result:
        gene_channel[r["gene"]][r["channel"]] = r["weight"]

    # Get PPI neighbors
    result = session.run("""
        MATCH (g1:Gene)-[r:PPI]-(g2:Gene)
        RETURN g1.name AS gene, g2.name AS neighbor, r.score AS score
    """)
    ppi = defaultdict(list)
    for r in result:
        ppi[r["gene"]].append((r["neighbor"], r["score"] or 0.4))

    # Compute profile: own channels + neighbor influence
    n_updated = 0
    batch = []
    for gene in gene_channel:
        profile = np.zeros(len(CHANNEL_NAMES))
        # Direct membership
        for ch, w in gene_channel[gene].items():
            if ch in CHANNEL_NAMES:
                idx = CHANNEL_NAMES.index(ch)
                profile[idx] += w

        # PPI neighbor influence (20% bleed)
        for neighbor, score in ppi.get(gene, []):
            if neighbor in gene_channel:
                for ch, w in gene_channel[neighbor].items():
                    if ch in CHANNEL_NAMES:
                        idx = CHANNEL_NAMES.index(ch)
                        profile[idx] += 0.2 * score * w

        # Normalize
        total = profile.sum()
        if total > 0:
            profile = profile / total

        batch.append({
            "name": gene,
            "profile": list(np.round(profile, 4)),
        })

    # Write back
    session.run("""
        UNWIND $batch AS b
        MATCH (g:Gene {name: b.name})
        SET g.channel_profile = b.profile
    """, batch=batch)
    n_updated = len(batch)

    _step("Gene channel_profile", actual=n_updated, elapsed=time.time() - t0)
    return n_updated


# =========================================================================
# 2. Gene confidence — based on current edge coverage
# =========================================================================

def refresh_gene_confidence(session):
    """Recompute gene confidence levels from edge coverage."""
    t0 = time.time()

    result = session.run("""
        MATCH (g:Gene)
        OPTIONAL MATCH (g)-[:PPI]-()
        WITH g, count(*) AS n_ppi
        OPTIONAL MATCH (g)-[:BELONGS_TO]->()
        WITH g, n_ppi, count(*) AS n_channel
        OPTIONAL MATCH (g)<-[:PROGNOSTIC_IN]-()
        WITH g, n_ppi, n_channel, count(*) AS n_prognostic
        OPTIONAL MATCH (g)<-[:EXPRESSION_IN]-()
        WITH g, n_ppi, n_channel, n_prognostic, count(*) AS n_expr
        OPTIONAL MATCH (g)<-[:CNA_IN]-()
        WITH g, n_ppi, n_channel, n_prognostic, n_expr, count(*) AS n_cna
        OPTIONAL MATCH (g)<-[:BIALLELIC_IN]-()
        WITH g, n_ppi, n_channel, n_prognostic, n_expr, n_cna, count(*) AS n_biallelic
        OPTIONAL MATCH ()-[:HAS_MUTATION]->(g)
        WITH g, n_ppi, n_channel, n_prognostic, n_expr, n_cna, n_biallelic, count(*) AS n_patients
        RETURN g.name AS gene, n_ppi, n_channel, n_prognostic, n_expr, n_cna, n_biallelic, n_patients
    """)

    batch = []
    for r in result:
        score = 0
        if r["n_ppi"] > 0: score += 1
        if r["n_ppi"] > 5: score += 1
        if r["n_channel"] > 0: score += 1
        if r["n_prognostic"] > 0: score += 1
        if r["n_prognostic"] > 5: score += 1
        if r["n_expr"] > 0: score += 1
        if r["n_cna"] > 0: score += 1
        if r["n_biallelic"] > 0: score += 1
        if r["n_patients"] > 100: score += 1
        if r["n_patients"] > 500: score += 1

        if score >= 8:
            level = "very_high"
        elif score >= 6:
            level = "high"
        elif score >= 4:
            level = "medium"
        elif score >= 2:
            level = "low"
        else:
            level = "minimal"

        batch.append({
            "name": r["gene"],
            "confidence": level,
            "confidence_score": score,
        })

    session.run("""
        UNWIND $batch AS b
        MATCH (g:Gene {name: b.name})
        SET g.confidence = b.confidence,
            g.confidence_score = b.confidence_score
    """, batch=batch)

    _step("Gene confidence", actual=len(batch), elapsed=time.time() - t0)
    return len(batch)


# =========================================================================
# 3. Gene function — from CIViC evidence + biallelic patterns
# =========================================================================

def refresh_gene_function(session):
    """Recompute gene function (TSG/oncogene/dual) from current evidence."""
    t0 = time.time()

    # Known classifications from CIViC / literature
    KNOWN_TSG = {
        "TP53", "RB1", "CDKN2A", "PTEN", "APC", "VHL", "BRCA1", "BRCA2",
        "NF1", "NF2", "SMAD4", "BAP1", "WT1", "STK11", "CDH1", "ARID1A",
        "SMARCA4", "ATM", "RAD51C", "RAD51D", "PALB2", "CHEK2", "MLH1",
        "MSH2", "MSH6", "PMS2", "MUTYH",
    }
    KNOWN_ONCO = {
        "KRAS", "BRAF", "PIK3CA", "EGFR", "ERBB2", "MET", "ALK", "RET",
        "FGFR1", "FGFR2", "FGFR3", "IDH1", "IDH2", "JAK2", "KIT", "PDGFRA",
        "NRAS", "HRAS", "CTNNB1", "NOTCH1", "MYC", "CCND1", "CDK4", "MDM2",
    }

    # Check biallelic evidence — genes with high biallelic rate are likely TSGs
    result = session.run("""
        MATCH (g:Gene)<-[b:BIALLELIC_IN]-()
        WITH g.name AS gene, avg(b.biallelic_freq) AS avg_rate,
             sum(b.n_biallelic) AS total_biallelic
        WHERE total_biallelic >= 5
        RETURN gene, avg_rate, total_biallelic
    """)
    biallelic_genes = {}
    for r in result:
        biallelic_genes[r["gene"]] = {
            "rate": r["avg_rate"],
            "count": r["total_biallelic"],
        }

    # Get all genes
    result = session.run("MATCH (g:Gene) RETURN g.name AS name")
    all_genes = [r["name"] for r in result]

    batch = []
    for gene in all_genes:
        if gene in KNOWN_TSG and gene in KNOWN_ONCO:
            func = "dual"
        elif gene in KNOWN_TSG:
            func = "TSG"
        elif gene in KNOWN_ONCO:
            func = "oncogene"
        elif gene in biallelic_genes and biallelic_genes[gene]["rate"] > 0.15:
            func = "likely_TSG"
        else:
            func = "unknown"

        batch.append({"name": gene, "function": func})

    session.run("""
        UNWIND $batch AS b
        MATCH (g:Gene {name: b.name})
        SET g.function = b.function
    """, batch=batch)

    _step("Gene function", actual=len(batch), elapsed=time.time() - t0)
    return len(batch)


# =========================================================================
# 4. HAS_MUTATION log_hr — from current PROGNOSTIC_IN atlas
# =========================================================================

def refresh_log_hr(session):
    """Recompute log_hr on every HAS_MUTATION edge from current atlas.

    Instead of pulling all 345K edges, we push the join into Cypher:
    for each atlas entry (gene, ct, log_hr), update all matching HAS_MUTATION edges.
    """
    t0 = time.time()

    # Load atlas: gene × cancer_type → hr, compute log_hr
    result = session.run("""
        MATCH (ct:CancerType)<-[p:PROGNOSTIC_IN]-(g:Gene)
        WHERE p.hr IS NOT NULL AND p.hr > 0
        RETURN g.name AS gene, ct.name AS ct, p.hr AS hr, p.tier AS tier
    """)
    atlas = {}
    for r in result:
        key = (r["gene"], r["ct"])
        tier = r["tier"] or 99
        if key not in atlas or tier < atlas[key]["tier"]:
            atlas[key] = {"hr": r["hr"], "log_hr": float(np.log(r["hr"])), "tier": tier}

    # Push updates: batch by gene×ct so Cypher does the fan-out
    entries = [{"gene": g, "ct": ct, "log_hr": round(float(v["log_hr"]), 4)}
               for (g, ct), v in atlas.items() if v["log_hr"] is not None]

    n_updated = 0
    batch_size = 100
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        result = session.run("""
            UNWIND $batch AS b
            MATCH (p:Patient {cancer_type: b.ct})-[r:HAS_MUTATION]->(g:Gene {name: b.gene})
            SET r.log_hr = b.log_hr
            RETURN count(r) AS updated
        """, batch=batch)
        rec = result.single()
        if rec:
            n_updated += rec["updated"]

    _step("HAS_MUTATION log_hr", actual=n_updated, elapsed=time.time() - t0)
    return n_updated


# =========================================================================
# 5. Patient tissue_delta — from refreshed log_hr
# =========================================================================

def refresh_tissue_delta(session):
    """Recompute per-patient tissue_delta from current HAS_MUTATION log_hr values.

    Pushes aggregation into Cypher to avoid pulling all edges into Python.
    Processes by cancer type to keep memory bounded.
    """
    t0 = time.time()

    # Get cancer types
    result = session.run("MATCH (ct:CancerType) RETURN ct.name AS name")
    cancer_types = [r["name"] for r in result]

    n_total = 0
    for ct in cancer_types:
        result = session.run("""
            MATCH (p:Patient {cancer_type: $ct})-[r:HAS_MUTATION]->(g:Gene)
            WHERE r.log_hr IS NOT NULL
            WITH p, avg(r.log_hr) AS mean_hr, count(r) AS n_scored
            SET p.tissue_delta = round(mean_hr * 10000) / 10000.0,
                p.n_scored_mutations = n_scored
            RETURN count(p) AS updated
        """, ct=ct)
        rec = result.single()
        if rec:
            n_total += rec["updated"]

    _step("Patient tissue_delta", actual=n_total, elapsed=time.time() - t0)
    return n_total


# =========================================================================
# 6. MutationGroup stats — median_os, n_events, n_patients
# =========================================================================

def refresh_mutation_group_stats(session):
    """Recompute MutationGroup aggregate stats from current patient data.

    Strategy: build gene→patient stats in Python (fast), then aggregate
    per MutationGroup from its gene set. Avoids expensive multi-hop Cypher.
    """
    t0 = time.time()

    # Step 1: Get per-gene patient stats (single efficient query)
    result = session.run("""
        MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)
        WITH g.name AS gene,
             count(DISTINCT p) AS n_patients,
             sum(CASE WHEN p.event = 1 THEN 1 ELSE 0 END) AS n_events,
             percentileDisc(p.os_months, 0.5) AS median_os
        RETURN gene, n_patients, n_events, median_os
    """)
    gene_stats = {}
    for r in result:
        gene_stats[r["gene"]] = {
            "n_patients": r["n_patients"],
            "n_events": r["n_events"],
            "median_os": r["median_os"],
        }

    # Step 2: Get MutationGroup → gene mapping
    result = session.run("""
        MATCH (mg:MutationGroup)-[:HAS_GENE]->(g:Gene)
        RETURN mg.mutation_key AS key, collect(g.name) AS genes
    """)

    batch = []
    for r in result:
        genes = r["genes"]
        # Aggregate: union of patients across genes (approximate with max)
        if not genes:
            continue
        stats = [gene_stats.get(g) for g in genes if g in gene_stats]
        if not stats:
            continue

        # For multi-gene groups: sum patients (approximate), sum events
        n_patients = max(s["n_patients"] for s in stats)
        n_events = max(s["n_events"] for s in stats)
        # Median OS: use the primary gene (first)
        median_os = stats[0]["median_os"]

        batch.append({
            "key": r["key"],
            "n_patients": n_patients,
            "n_events": n_events,
            "median_os": round(float(median_os), 2) if median_os is not None else None,
        })

    # Step 3: Write back in batches
    n_total = 0
    for i in range(0, len(batch), 500):
        chunk = batch[i:i + 500]
        session.run("""
            UNWIND $batch AS b
            MATCH (mg:MutationGroup {mutation_key: b.key})
            SET mg.n_patients = b.n_patients,
                mg.n_events = b.n_events,
                mg.median_os = b.median_os
        """, batch=chunk)
        n_total += len(chunk)

    _step("MutationGroup stats", actual=n_total, elapsed=time.time() - t0)
    return n_total


# =========================================================================
# 7. Invalidate disk caches
# =========================================================================

def invalidate_caches():
    """Remove ALL stale cache files so they get rebuilt fresh.

    Covers:
      - precomputed_*.json (gene properties, centrality, communities, etc.)
      - graph_derived_config.json (CHANNEL_MAP — must reload from Neo4j)
      - block_assignments.json (sub-pathway communities)
      - graph_schema.json (edge types, n_blocks, n_channels)
      - channel_v6*.pt, hublf_*.pt (feature tensors)
      - *.npz (gene_node_dynamic, pairwise_edges_dynamic, patient_connectivity)
      - mutation_graph.pkl, hetero_graph.pt
      - pyg_msk_impact_50k/processed/ (PyG dataset)
    """
    t0 = time.time()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "cache")
    n_removed = 0

    if not os.path.isdir(cache_dir):
        return n_removed

    # Files to remove by pattern
    remove_patterns = [
        lambda f: f.startswith("precomputed_") and f.endswith(".json"),
        lambda f: f == "graph_derived_config.json",
        lambda f: f == "block_assignments.json",
        lambda f: f == "graph_schema.json",
        lambda f: f.startswith("channel_v6") and f.endswith(".pt"),
        lambda f: f.startswith("channel_features") and f.endswith(".pt"),
        lambda f: f.startswith("hublf_") and f.endswith(".pt"),
        lambda f: f.endswith(".npz"),
        lambda f: f == "mutation_graph.pkl",
        lambda f: f == "hetero_graph.pt",
    ]

    for fname in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, fname)
        if os.path.isfile(fpath) and any(p(fname) for p in remove_patterns):
            os.remove(fpath)
            n_removed += 1

    # PyG processed directory
    pyg_processed = os.path.join(cache_dir, "pyg_msk_impact_50k", "processed")
    if os.path.isdir(pyg_processed):
        import shutil
        shutil.rmtree(pyg_processed)
        n_removed += 1

    _step("Cache files invalidated", actual=n_removed, elapsed=time.time() - t0)
    return n_removed


def rebuild_caches():
    """Rebuild critical precomputed caches from current graph state.

    Called after invalidate_caches() to ensure downstream consumers
    (training scripts, scorers) get fresh data on next run.
    """
    t0 = time.time()

    # 1. Force config reload (CHANNEL_MAP, HUB_GENES)
    import importlib
    import gnn.config
    importlib.reload(gnn.config)

    # 2. Rebuild block assignments
    from gnn.data.block_assignments import load_block_assignments
    gb, nb, nc = load_block_assignments(force_refresh=True)
    print(f"    Rebuilt block_assignments: {len(gb)} genes → {nb} blocks, {nc} channels")

    # 3. Rebuild precomputed graph caches
    from gnn.data.graph_precompute import (
        precompute_gene_properties,
        precompute_gene_centrality,
        precompute_communities,
        precompute_pairwise_edges,
        precompute_cross_channel_matrix,
    )
    precompute_gene_properties()
    precompute_gene_centrality()
    precompute_communities()
    precompute_cross_channel_matrix()
    precompute_pairwise_edges()

    elapsed = round(time.time() - t0, 1)
    _step("Caches rebuilt", actual=5, elapsed=elapsed)
    return elapsed


# =========================================================================
# Main entry point
# =========================================================================

def refresh_all(dry_run=False):
    """Refresh all computed properties from current graph state.

    Returns a summary dict with counts of what was updated.
    """
    print(f"\n  {'='*60}")
    print(f"  REFRESHING COMPUTED PROPERTIES")
    print(f"  {'='*60}")

    t0 = time.time()

    if dry_run:
        print("    [DRY RUN] Would refresh all computed properties")
        return {"dry_run": True, "elapsed_s": 0}

    driver = _driver()
    summary = {}

    with driver.session() as session:
        summary["channel_synced"] = sync_channel_property(session)
        summary["channel_profiles"] = refresh_channel_profiles(session)
        summary["gene_confidence"] = refresh_gene_confidence(session)
        summary["gene_function"] = refresh_gene_function(session)
        summary["log_hr"] = refresh_log_hr(session)
        summary["tissue_delta"] = refresh_tissue_delta(session)
        summary["mutation_group_stats"] = refresh_mutation_group_stats(session)

    driver.close()

    summary["caches_invalidated"] = invalidate_caches()
    summary["caches_rebuild_s"] = rebuild_caches()
    summary["elapsed_s"] = round(time.time() - t0, 1)

    print(f"\n    Total refresh time: {summary['elapsed_s']}s")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = refresh_all(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
