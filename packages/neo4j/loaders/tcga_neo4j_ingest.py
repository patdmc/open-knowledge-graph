#!/usr/bin/env python3
"""
Ingest TCGA data into Neo4j — adds alongside existing MSK data.

Creates:
  (:Patient {id, cancer_type, dataset, os_months, event, tissue_delta,
             age, sex, stage, msi_score, fga, aneuploidy, tmb,
             mutation_count, neoadjuvant, radiation,
             hypoxia_buffa, hypoxia_ragnum})
  (:Patient)-[:HAS_MUTATION {direction, protein_change, mutation_type}]->(:Gene)
  (:Gene)-[:EXPRESSION_IN {mean_expr, z_vs_pancancer, is_overexpressed,
                            is_underexpressed, cancer_type}]->(:Channel)
  (:Gene)-[:CNA_IN {amp_freq, del_freq, mean_cna, cancer_type}]->(:Channel)

Does NOT clear existing data. Uses MERGE on Patient(id) to avoid duplicates.

Usage:
    python3 -u -m gnn.scripts.tcga_neo4j_ingest
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase
from gnn.config import CHANNEL_MAP, GENE_FUNCTION, GNN_CACHE
from gnn.data.neo4j_loader import NEO4J_URI, NEO4J_AUTH

TCGA_DIR = os.path.join(GNN_CACHE, "tcga")
TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"}


def _step(msg, n=None, elapsed=None):
    parts = [f"  {msg}"]
    if n is not None:
        parts.append(f"({n:,})")
    if elapsed is not None:
        parts.append(f"[{elapsed:.1f}s]")
    print(" ".join(parts))


def load_tcga_patients(driver):
    """Create TCGA Patient nodes with rich clinical properties."""
    t0 = time.time()
    clin = pd.read_csv(os.path.join(TCGA_DIR, "tcga_clinical_detail.csv"), low_memory=False)

    # Filter to valid survival
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["os_months"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["os_months"] > 0]

    # Tissue deltas
    deceased = clin[clin["event"] == 1]
    global_median = deceased["os_months"].median()
    ct_medians = deceased.groupby("cancer_type")["os_months"].median()
    tissue_deltas = (ct_medians - global_median).to_dict()

    # Parse stage to numeric
    stage_map = {"STAGE I": 1, "STAGE II": 2, "STAGE III": 3, "STAGE IV": 4}
    clin["stage_num"] = clin["stage"].map(
        lambda s: next((v for k, v in stage_map.items() if str(s).upper().startswith(k)), 0)
    )

    # Neoadjuvant / radiation as binary
    clin["neoadj"] = (clin["HISTORY_NEOADJUVANT_TRTYN"].str.upper() == "YES").astype(int)
    clin["rad"] = (clin["RADIATION_THERAPY"].str.upper() == "YES").astype(int)

    n_patients = 0
    with driver.session() as session:
        # Create index for dataset property
        session.run("CREATE INDEX patient_dataset IF NOT EXISTS FOR (p:Patient) ON (p.dataset)")

        batch = []
        for _, row in clin.iterrows():
            ct = row["cancer_type"]
            batch.append({
                "id": row["patient_id"],
                "cancer_type": ct,
                "dataset": "TCGA",
                "os_months": float(row["os_months"]),
                "event": int(row["event"]),
                "tissue_delta": tissue_deltas.get(ct, 0.0),
                "age": float(row["age"]) if pd.notna(row.get("age")) else -1.0,
                "sex": str(row.get("sex", "")),
                "stage": int(row["stage_num"]),
                "msi_score": float(row["MSI_SCORE_MANTIS"]) if pd.notna(row.get("MSI_SCORE_MANTIS")) else -1.0,
                "fga": float(row["FRACTION_GENOME_ALTERED"]) if pd.notna(row.get("FRACTION_GENOME_ALTERED")) else -1.0,
                "aneuploidy": float(row["ANEUPLOIDY_SCORE"]) if pd.notna(row.get("ANEUPLOIDY_SCORE")) else -1.0,
                "tmb": float(row["TMB_NONSYNONYMOUS"]) if pd.notna(row.get("TMB_NONSYNONYMOUS")) else -1.0,
                "mutation_count": int(row["MUTATION_COUNT"]) if pd.notna(row.get("MUTATION_COUNT")) else -1,
                "neoadjuvant": int(row["neoadj"]),
                "radiation": int(row["rad"]),
                "hypoxia_buffa": float(row["BUFFA_HYPOXIA_SCORE"]) if pd.notna(row.get("BUFFA_HYPOXIA_SCORE")) else -1.0,
                "hypoxia_ragnum": float(row["RAGNUM_HYPOXIA_SCORE"]) if pd.notna(row.get("RAGNUM_HYPOXIA_SCORE")) else -1.0,
            })
            if len(batch) >= 500:
                _create_patients(session, batch)
                n_patients += len(batch)
                batch = []
        if batch:
            _create_patients(session, batch)
            n_patients += len(batch)

    _step("TCGA Patient nodes created", n=n_patients, elapsed=time.time() - t0)
    return set(clin["patient_id"]), clin


def _create_patients(session, batch):
    session.run(
        """UNWIND $batch AS row
           MERGE (p:Patient {id: row.id})
           SET p.cancer_type = row.cancer_type,
               p.dataset = row.dataset,
               p.os_months = row.os_months,
               p.event = row.event,
               p.tissue_delta = row.tissue_delta,
               p.age = row.age,
               p.sex = row.sex,
               p.stage = row.stage,
               p.msi_score = row.msi_score,
               p.fga = row.fga,
               p.aneuploidy = row.aneuploidy,
               p.tmb = row.tmb,
               p.mutation_count = row.mutation_count,
               p.neoadjuvant = row.neoadjuvant,
               p.radiation = row.radiation,
               p.hypoxia_buffa = row.hypoxia_buffa,
               p.hypoxia_ragnum = row.hypoxia_ragnum""",
        batch=batch,
    )


def load_tcga_mutations(driver, valid_pids):
    """Create HAS_MUTATION edges for TCGA patients."""
    t0 = time.time()
    mut = pd.read_csv(os.path.join(TCGA_DIR, "tcga_mutations.csv"), low_memory=False)

    # Filter to known genes and valid patients
    known_genes = set(CHANNEL_MAP.keys())
    mut = mut[mut["gene"].isin(known_genes) & mut["patient_id"].isin(valid_pids)]

    # Deduplicate per (patient, gene)
    mut_dedup = mut.drop_duplicates(subset=["patient_id", "gene"])

    n_edges = 0
    with driver.session() as session:
        batch = []
        for _, row in mut_dedup.iterrows():
            gene = row["gene"]
            func = GENE_FUNCTION.get(gene, "context")
            if func == "context":
                direction = "LOF" if row.get("mutation_type") in TRUNCATING else "GOF"
            else:
                direction = func

            batch.append({
                "pid": row["patient_id"],
                "gene": gene,
                "direction": direction,
                "protein_change": str(row.get("protein_change", "")),
                "mutation_type": str(row.get("mutation_type", "")),
            })
            if len(batch) >= 2000:
                _create_mutations(session, batch)
                n_edges += len(batch)
                if n_edges % 10000 == 0:
                    print(f"    {n_edges:,} mutation edges...")
                batch = []
        if batch:
            _create_mutations(session, batch)
            n_edges += len(batch)

    _step("HAS_MUTATION edges created", n=n_edges, elapsed=time.time() - t0)


def _create_mutations(session, batch):
    session.run(
        """UNWIND $batch AS row
           MATCH (p:Patient {id: row.pid}), (g:Gene {name: row.gene})
           CREATE (p)-[:HAS_MUTATION {
               direction: row.direction,
               protein_change: row.protein_change,
               mutation_type: row.mutation_type
           }]->(g)""",
        batch=batch,
    )


def load_expression_edges(driver):
    """Create EXPRESSION_IN edges: Gene -[EXPRESSION_IN]-> Channel per cancer type."""
    t0 = time.time()
    expr = pd.read_csv(os.path.join(TCGA_DIR, "tcga_expression_edges.csv"), low_memory=False)

    # Map gene to channel
    known_genes = set(CHANNEL_MAP.keys())
    expr = expr[expr["from"].isin(known_genes)]

    n_edges = 0
    with driver.session() as session:
        batch = []
        for _, row in expr.iterrows():
            gene = row["from"]
            ct = row["to"]  # cancer type acronym
            ch = CHANNEL_MAP.get(gene)
            if not ch:
                continue
            batch.append({
                "gene": gene,
                "channel": ch,
                "cancer_type": ct,
                "mean_expr": float(row["mean_expr"]) if pd.notna(row.get("mean_expr")) else 0.0,
                "z_vs_pancancer": float(row["z_vs_pancancer"]) if pd.notna(row.get("z_vs_pancancer")) else 0.0,
                "is_overexpressed": bool(row.get("is_overexpressed", False)),
                "is_underexpressed": bool(row.get("is_underexpressed", False)),
            })
            if len(batch) >= 500:
                _create_expression(session, batch)
                n_edges += len(batch)
                batch = []
        if batch:
            _create_expression(session, batch)
            n_edges += len(batch)

    _step("EXPRESSION_IN edges created", n=n_edges, elapsed=time.time() - t0)


def _create_expression(session, batch):
    session.run(
        """UNWIND $batch AS row
           MATCH (g:Gene {name: row.gene})
           MERGE (g)-[e:EXPRESSION_IN {cancer_type: row.cancer_type}]->(c:Channel {name: row.channel})
           SET e.mean_expr = row.mean_expr,
               e.z_vs_pancancer = row.z_vs_pancancer,
               e.is_overexpressed = row.is_overexpressed,
               e.is_underexpressed = row.is_underexpressed""",
        batch=batch,
    )


def load_cna_edges(driver):
    """Create CNA_IN edges: Gene -[CNA_IN]-> Channel per cancer type."""
    t0 = time.time()
    cna = pd.read_csv(os.path.join(TCGA_DIR, "tcga_cna_summary.csv"), low_memory=False)

    known_genes = set(CHANNEL_MAP.keys())
    cna = cna[cna["gene"].isin(known_genes)]

    n_edges = 0
    with driver.session() as session:
        batch = []
        for _, row in cna.iterrows():
            gene = row["gene"]
            ch = row.get("channel") or CHANNEL_MAP.get(gene)
            if not ch:
                continue
            batch.append({
                "gene": gene,
                "channel": ch,
                "cancer_type": row["cancer_type"],
                "amp_freq": float(row["amp_freq"]) if pd.notna(row.get("amp_freq")) else 0.0,
                "del_freq": float(row["del_freq"]) if pd.notna(row.get("del_freq")) else 0.0,
                "mean_cna": float(row["mean_cna"]) if pd.notna(row.get("mean_cna")) else 0.0,
            })
            if len(batch) >= 500:
                _create_cna(session, batch)
                n_edges += len(batch)
                batch = []
        if batch:
            _create_cna(session, batch)
            n_edges += len(batch)

    _step("CNA_IN edges created", n=n_edges, elapsed=time.time() - t0)


def _create_cna(session, batch):
    session.run(
        """UNWIND $batch AS row
           MATCH (g:Gene {name: row.gene})
           MERGE (g)-[e:CNA_IN {cancer_type: row.cancer_type}]->(c:Channel {name: row.channel})
           SET e.amp_freq = row.amp_freq,
               e.del_freq = row.del_freq,
               e.mean_cna = row.mean_cna""",
        batch=batch,
    )


def load_tcga_cooccurrence(driver, clin_df):
    """Build TCGA-specific COOCCURS edges."""
    t0 = time.time()
    mut = pd.read_csv(os.path.join(TCGA_DIR, "tcga_mutations.csv"), low_memory=False)
    known_genes = set(CHANNEL_MAP.keys())
    valid_pids = set(clin_df["patient_id"])
    mut = mut[mut["gene"].isin(known_genes) & mut["patient_id"].isin(valid_pids)]

    # Build patient → gene sets, patient → CT
    from collections import defaultdict
    from itertools import combinations

    patient_genes = defaultdict(set)
    patient_ct = {}
    for _, row in mut.iterrows():
        patient_genes[row["patient_id"]].add(row["gene"])
        patient_ct[row["patient_id"]] = row["cancer_type"]

    # Count co-occurrences per CT
    cooccur = defaultdict(lambda: defaultdict(int))
    for pid, genes in patient_genes.items():
        ct = patient_ct.get(pid)
        if not ct or len(genes) < 2:
            continue
        for g1, g2 in combinations(sorted(genes), 2):
            cooccur[(g1, g2)][ct] += 1

    # Filter to min_count >= 5 (TCGA is smaller)
    min_count = 5
    n_edges = 0
    with driver.session() as session:
        batch = []
        for (g1, g2), ct_counts in cooccur.items():
            for ct, count in ct_counts.items():
                if count < min_count:
                    continue
                batch.append({
                    "g1": g1, "g2": g2,
                    "ct": f"TCGA_{ct}",  # prefix to distinguish from MSK
                    "count": count,
                    "dataset": "TCGA",
                })
                if len(batch) >= 2000:
                    _create_cooccurs(session, batch)
                    n_edges += len(batch)
                    if n_edges % 10000 == 0:
                        print(f"    {n_edges:,} co-occurrence edges...")
                    batch = []
        if batch:
            _create_cooccurs(session, batch)
            n_edges += len(batch)

    _step("TCGA COOCCURS edges created", n=n_edges, elapsed=time.time() - t0)


def _create_cooccurs(session, batch):
    session.run(
        """UNWIND $batch AS row
           MATCH (a:Gene {name: row.g1}), (b:Gene {name: row.g2})
           CREATE (a)-[:COOCCURS {cancer_type: row.ct, count: row.count, dataset: row.dataset}]->(b)""",
        batch=batch,
    )


def main():
    t0 = time.time()
    print("=" * 70)
    print("  TCGA NEO4J INGEST")
    print("  Adding TCGA patients, mutations, expression, CNA to graph")
    print("=" * 70)

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    # Check existing state
    with driver.session() as session:
        result = session.run("MATCH (p:Patient) RETURN count(p) AS n")
        n_existing = result.single()["n"]
        print(f"\n  Existing patients in graph: {n_existing:,}")

        result = session.run("MATCH (p:Patient {dataset: 'TCGA'}) RETURN count(p) AS n")
        n_tcga = result.single()["n"]
        if n_tcga > 0:
            print(f"  Found {n_tcga:,} existing TCGA patients — will MERGE (update, not duplicate)")

    # 1. Patient nodes
    print("\n  Loading TCGA Patient nodes...")
    valid_pids, clin_df = load_tcga_patients(driver)

    # 2. Mutations
    print("\n  Loading TCGA HAS_MUTATION edges...")
    load_tcga_mutations(driver, valid_pids)

    # 3. Expression edges
    print("\n  Loading EXPRESSION_IN edges...")
    load_expression_edges(driver)

    # 4. CNA edges
    print("\n  Loading CNA_IN edges...")
    load_cna_edges(driver)

    # 5. Co-occurrence
    print("\n  Building TCGA co-occurrence edges...")
    load_tcga_cooccurrence(driver, clin_df)

    # Summary
    print("\n" + "=" * 70)
    with driver.session() as session:
        stats = {}
        for label in ["Patient", "Gene", "Channel"]:
            r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n")
            stats[label] = r.single()["n"]
        r = session.run("MATCH (p:Patient {dataset: 'TCGA'}) RETURN count(p) AS n")
        stats["TCGA_patients"] = r.single()["n"]
        for rel in ["HAS_MUTATION", "EXPRESSION_IN", "CNA_IN", "COOCCURS"]:
            r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")
            stats[rel] = r.single()["n"]

    print(f"  Total Patients: {stats['Patient']:,} ({stats['TCGA_patients']:,} TCGA)")
    print(f"  Genes: {stats['Gene']:,}, Channels: {stats['Channel']:,}")
    print(f"  HAS_MUTATION: {stats['HAS_MUTATION']:,}")
    print(f"  EXPRESSION_IN: {stats['EXPRESSION_IN']:,}")
    print(f"  CNA_IN: {stats['CNA_IN']:,}")
    print(f"  COOCCURS: {stats['COOCCURS']:,}")
    print(f"\n  Total time: {time.time() - t0:.0f}s")
    print("=" * 70)

    driver.close()


if __name__ == "__main__":
    main()
