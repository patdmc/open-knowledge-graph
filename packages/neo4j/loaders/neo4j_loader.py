#!/usr/bin/env python3
"""
Neo4j graph loader — migrates the mutation graph into Neo4j.

Schema:
  Nodes:
    (:Gene {name, function, position, is_hub,
            channel_profile: [8-dim], profile_entropy, primary_channel})
    (:Channel {name, idx, tier, tier_name})
    (:Patient {id, cancer_type, os_months, event, tissue_delta})

  Relationships:
    (:Gene)-[:BELONGS_TO {weight}]->(:Channel)
        Multi-channel: one edge per channel with weight from profile vector.
    (:Gene)-[:PPI {score}]->(:Gene)
        STRING PPI edges with confidence score.
    (:Gene)-[:COOCCURS {count, cancer_type}]->(:Gene)
        Per-cancer-type co-occurrence counts.
    (:Patient)-[:HAS_MUTATION {direction, log_hr, protein_change}]->(:Gene)
    (:Patient)-[:SIMILAR_TO {jaccard, shared_genes}]->(:Patient)
        Patient affinity edges (top-k per patient).

Indexes:
    Gene(name), Channel(name), Patient(id)

Usage:
    python3 -u -m gnn.data.neo4j_loader
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase

from gnn.config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    HUB_GENES, GENE_FUNCTION, GENE_POSITION,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
    GNN_CACHE, GNN_RESULTS, ANALYSIS_CACHE,
)
from gnn.data.channel_dataset_v6 import V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def clear_database(driver):
    """Drop all nodes and relationships."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  Database cleared.")


def create_indexes(driver):
    """Create indexes for fast lookups."""
    indexes = [
        "CREATE INDEX gene_name IF NOT EXISTS FOR (g:Gene) ON (g.name)",
        "CREATE INDEX channel_name IF NOT EXISTS FOR (c:Channel) ON (c.name)",
        "CREATE INDEX patient_id IF NOT EXISTS FOR (p:Patient) ON (p.id)",
        "CREATE INDEX gene_primary_channel IF NOT EXISTS FOR (g:Gene) ON (g.primary_channel)",
        "CREATE INDEX patient_cancer_type IF NOT EXISTS FOR (p:Patient) ON (p.cancer_type)",
    ]
    with driver.session() as session:
        for idx in indexes:
            session.run(idx)
    print(f"  Created {len(indexes)} indexes.")


def load_channel_nodes(driver):
    """Create 8 Channel nodes."""
    with driver.session() as session:
        for ch_name in V6_CHANNEL_NAMES:
            tier = V6_TIER_MAP.get(ch_name, -1)
            tier_names = ["cell_intrinsic", "tissue_level", "organism_level", "meta_regulatory"]
            tier_name = tier_names[tier] if 0 <= tier < len(tier_names) else "unknown"
            session.run(
                "CREATE (c:Channel {name: $name, idx: $idx, tier: $tier, tier_name: $tier_name})",
                name=ch_name, idx=V6_CHANNEL_NAMES.index(ch_name),
                tier=tier, tier_name=tier_name,
            )
    print(f"  Created {len(V6_CHANNEL_NAMES)} Channel nodes.")


def load_gene_nodes(driver, channel_profiles):
    """Create Gene nodes with channel profile vectors."""
    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set.update(hubs)

    genes = set(V6_CHANNEL_MAP.keys())
    # Also include expanded genes
    expanded_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    if os.path.exists(expanded_path):
        with open(expanded_path) as f:
            ecm = json.load(f)
        for gene in ecm:
            genes.add(gene)

    n_created = 0
    with driver.session() as session:
        for gene in sorted(genes):
            func = GENE_FUNCTION.get(gene, "context")
            pos = GENE_POSITION.get(gene, "unclassified")
            is_hub = gene in hub_set

            profile = channel_profiles.get(gene, np.zeros(len(V6_CHANNEL_NAMES)))
            primary_ch = V6_CHANNEL_NAMES[int(np.argmax(profile))] if profile.sum() > 0 else "unknown"
            entropy = _profile_entropy(profile)

            session.run(
                """CREATE (g:Gene {
                    name: $name, function: $func, position: $pos, is_hub: $is_hub,
                    channel_profile: $profile, profile_entropy: $entropy,
                    primary_channel: $primary_ch
                })""",
                name=gene, func=func, pos=pos, is_hub=is_hub,
                profile=profile.tolist(), entropy=entropy,
                primary_ch=primary_ch,
            )
            n_created += 1

    print(f"  Created {n_created} Gene nodes.")
    return genes


def load_belongs_to_edges(driver, channel_profiles):
    """Create BELONGS_TO edges — multi-channel, weighted by profile."""
    n_edges = 0
    with driver.session() as session:
        for gene, profile in channel_profiles.items():
            for ci, ch_name in enumerate(V6_CHANNEL_NAMES):
                if profile[ci] > 0.01:
                    session.run(
                        """MATCH (g:Gene {name: $gene}), (c:Channel {name: $ch})
                           CREATE (g)-[:BELONGS_TO {weight: $weight}]->(c)""",
                        gene=gene, ch=ch_name, weight=float(profile[ci]),
                    )
                    n_edges += 1
    print(f"  Created {n_edges} BELONGS_TO edges (multi-channel weighted).")


def load_ppi_edges(driver):
    """Load STRING PPI edges."""
    ppi_path = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    if not os.path.exists(ppi_path):
        print("  No PPI cache found, skipping PPI edges.")
        return

    with open(ppi_path) as f:
        ppi_edges = json.load(f)

    n_edges = 0
    with driver.session() as session:
        # Batch for performance
        batch = []
        for g1, g2, score in ppi_edges:
            batch.append({"g1": g1, "g2": g2, "score": score})
            if len(batch) >= 500:
                session.run(
                    """UNWIND $batch AS row
                       MATCH (a:Gene {name: row.g1}), (b:Gene {name: row.g2})
                       CREATE (a)-[:PPI {score: row.score}]->(b)""",
                    batch=batch,
                )
                n_edges += len(batch)
                batch = []
        if batch:
            session.run(
                """UNWIND $batch AS row
                   MATCH (a:Gene {name: row.g1}), (b:Gene {name: row.g2})
                   CREATE (a)-[:PPI {score: row.score}]->(b)""",
                batch=batch,
            )
            n_edges += len(batch)

    print(f"  Created {n_edges} PPI edges.")


def load_patient_nodes_and_mutations(driver, dataset_name="msk_impact_50k"):
    """Create Patient nodes and HAS_MUTATION edges."""
    # Load clinical + sample clinical (for CANCER_TYPE)
    clin = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, f"{dataset_name}_2026_clinical.csv"),
        low_memory=False,
    )
    sample_clin = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, f"{dataset_name}_2026_sample_clinical.csv"),
        low_memory=False,
    )
    ct_map = sample_clin[["patientId", "CANCER_TYPE"]].drop_duplicates("patientId")
    clin = clin.merge(ct_map, on="patientId", how="left")
    clin["CANCER_TYPE"] = clin["CANCER_TYPE"].fillna("Unknown")

    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["time"] > 0]

    # Tissue intercepts
    deceased = clin[clin["event"] == 1]
    global_median = deceased["time"].median()
    ct_medians = deceased.groupby("CANCER_TYPE")["time"].median()
    tissue_deltas = (ct_medians - global_median).to_dict()

    # Create patient nodes in batches
    n_patients = 0
    with driver.session() as session:
        batch = []
        for _, row in clin.iterrows():
            ct = row.get("CANCER_TYPE", "Unknown")
            batch.append({
                "id": row["patientId"],
                "cancer_type": ct,
                "os_months": float(row["time"]),
                "event": int(row["event"]),
                "tissue_delta": tissue_deltas.get(ct, 0.0),
            })
            if len(batch) >= 1000:
                session.run(
                    """UNWIND $batch AS row
                       CREATE (p:Patient {
                           id: row.id, cancer_type: row.cancer_type,
                           os_months: row.os_months, event: row.event,
                           tissue_delta: row.tissue_delta
                       })""",
                    batch=batch,
                )
                n_patients += len(batch)
                batch = []
        if batch:
            session.run(
                """UNWIND $batch AS row
                   CREATE (p:Patient {
                       id: row.id, cancer_type: row.cancer_type,
                       os_months: row.os_months, event: row.event,
                       tissue_delta: row.tissue_delta
                   })""",
                batch=batch,
            )
            n_patients += len(batch)

    print(f"  Created {n_patients} Patient nodes.")

    # Load mutations
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, f"{dataset_name}_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange", "mutationType"],
    )
    # Filter to non-silent, known genes
    expanded_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    known_genes = set(V6_CHANNEL_MAP.keys())
    if os.path.exists(expanded_path):
        with open(expanded_path) as f:
            ecm = json.load(f)
        known_genes |= set(ecm.keys())

    mut = mut[mut["gene.hugoGeneSymbol"].isin(known_genes)]
    valid_pids = set(clin["patientId"])
    mut = mut[mut["patientId"].isin(valid_pids)]

    # Deduplicate per (patient, gene)
    mut_dedup = mut.drop_duplicates(subset=["patientId", "gene.hugoGeneSymbol"])

    n_edges = 0
    with driver.session() as session:
        batch = []
        for _, row in mut_dedup.iterrows():
            gene = row["gene.hugoGeneSymbol"]
            func = GENE_FUNCTION.get(gene, "context")
            if func == "context":
                direction = "LOF" if row.get("mutationType") in TRUNCATING else "GOF"
            else:
                direction = func

            batch.append({
                "pid": row["patientId"],
                "gene": gene,
                "direction": direction,
                "protein_change": str(row.get("proteinChange", "")),
            })
            if len(batch) >= 2000:
                session.run(
                    """UNWIND $batch AS row
                       MATCH (p:Patient {id: row.pid}), (g:Gene {name: row.gene})
                       CREATE (p)-[:HAS_MUTATION {
                           direction: row.direction,
                           protein_change: row.protein_change
                       }]->(g)""",
                    batch=batch,
                )
                n_edges += len(batch)
                if n_edges % 20000 == 0:
                    print(f"    {n_edges} mutation edges...")
                batch = []
        if batch:
            session.run(
                """UNWIND $batch AS row
                   MATCH (p:Patient {id: row.pid}), (g:Gene {name: row.gene})
                   CREATE (p)-[:HAS_MUTATION {
                       direction: row.direction,
                       protein_change: row.protein_change
                   }]->(g)""",
                batch=batch,
            )
            n_edges += len(batch)

    print(f"  Created {n_edges} HAS_MUTATION edges.")
    return valid_pids


def load_cooccurrence_edges(driver, patient_genes, ct_per_patient, min_count=10):
    """Create COOCCURS edges for gene pairs with significant co-occurrence."""
    from gnn.scripts.expanded_graph_scorer import compute_cooccurrence
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=min_count)

    n_edges = 0
    with driver.session() as session:
        batch = []
        for (g1, g2), ct_counts in cooccurrence.items():
            for ct, count in ct_counts.items():
                batch.append({"g1": g1, "g2": g2, "ct": ct, "count": count})
                if len(batch) >= 2000:
                    session.run(
                        """UNWIND $batch AS row
                           MATCH (a:Gene {name: row.g1}), (b:Gene {name: row.g2})
                           CREATE (a)-[:COOCCURS {cancer_type: row.ct, count: row.count}]->(b)""",
                        batch=batch,
                    )
                    n_edges += len(batch)
                    if n_edges % 50000 == 0:
                        print(f"    {n_edges} co-occurrence edges...")
                    batch = []
        if batch:
            session.run(
                """UNWIND $batch AS row
                   MATCH (a:Gene {name: row.g1}), (b:Gene {name: row.g2})
                   CREATE (a)-[:COOCCURS {cancer_type: row.ct, count: row.count}]->(b)""",
                batch=batch,
            )
            n_edges += len(batch)

    print(f"  Created {n_edges} COOCCURS edges.")


def _profile_entropy(vec):
    """Shannon entropy in bits."""
    v = vec[vec > 0]
    if len(v) <= 1:
        return 0.0
    return float(-np.sum(v * np.log2(v)))


def compute_channel_profiles_for_loader(expanded_cm):
    """Compute curated-anchor channel profiles (same as focused_multichannel_scorer)."""
    import networkx as nx
    from gnn.scripts.expanded_graph_scorer import fetch_string_expanded, build_expanded_graph

    ppi_path = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    if os.path.exists(ppi_path):
        with open(ppi_path) as f:
            ppi_edges = json.load(f)
    else:
        ppi_edges = fetch_string_expanded(set(expanded_cm.keys()))

    # Build PPI-only graph
    G_ppi = nx.Graph()
    for gene, ch in expanded_cm.items():
        G_ppi.add_node(gene)
    for g1, g2, score in ppi_edges:
        if g1 in expanded_cm and g2 in expanded_cm:
            G_ppi.add_edge(g1, g2, weight=score)

    curated_set = set(V6_CHANNEL_MAP.keys())
    CH_TO_IDX = {ch: i for i, ch in enumerate(V6_CHANNEL_NAMES)}
    N_CH = len(V6_CHANNEL_NAMES)

    profiles = {}
    for gene in set(G_ppi.nodes()) | set(expanded_cm.keys()):
        vec = np.zeros(N_CH)
        if gene in G_ppi:
            for nb in G_ppi.neighbors(gene):
                if nb not in curated_set:
                    continue
                nb_ch = V6_CHANNEL_MAP.get(nb)
                if nb_ch and nb_ch in CH_TO_IDX:
                    w = G_ppi[gene][nb].get("weight", 0.7)
                    vec[CH_TO_IDX[nb_ch]] += w

        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            ch = expanded_cm.get(gene) or V6_CHANNEL_MAP.get(gene)
            if ch and ch in CH_TO_IDX:
                vec[CH_TO_IDX[ch]] = 1.0

        profiles[gene] = vec

    return profiles


def main():
    t0 = time.time()
    print("=" * 70)
    print("  NEO4J GRAPH LOADER")
    print("  Migrating mutation graph + new dimensions to Neo4j")
    print("=" * 70)

    driver = get_driver()

    # --- Clear and setup ---
    print("\n  Setting up database...")
    clear_database(driver)
    create_indexes(driver)

    # --- Compute channel profiles ---
    print("\n  Computing channel profiles...")
    expanded_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    with open(expanded_path) as f:
        ecm = json.load(f)
    expanded_cm = {gene: info["channel"] for gene, info in ecm.items() if gene and gene != "nan"}

    channel_profiles = compute_channel_profiles_for_loader(expanded_cm)
    n_multi = sum(1 for p in channel_profiles.values() if np.sum(p > 0.1) >= 2)
    print(f"  {len(channel_profiles)} gene profiles ({n_multi} multi-channel)")

    # --- Load nodes ---
    print("\n  Loading Channel nodes...")
    load_channel_nodes(driver)

    print("\n  Loading Gene nodes...")
    gene_set = load_gene_nodes(driver, channel_profiles)

    print("\n  Loading BELONGS_TO edges (multi-channel)...")
    load_belongs_to_edges(driver, channel_profiles)

    print("\n  Loading PPI edges...")
    load_ppi_edges(driver)

    print("\n  Loading Patient nodes + HAS_MUTATION edges...")
    valid_pids = load_patient_nodes_and_mutations(driver)

    # --- Co-occurrence (needs patient gene sets) ---
    print("\n  Building patient gene sets for co-occurrence...")
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol"],
    )
    clin2 = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"),
        low_memory=False,
    )
    sc2 = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_sample_clinical.csv"),
        low_memory=False,
    )
    ct_map2 = sc2[["patientId", "CANCER_TYPE"]].drop_duplicates("patientId")
    clin2 = clin2.merge(ct_map2, on="patientId", how="left")
    clin2["CANCER_TYPE"] = clin2["CANCER_TYPE"].fillna("Unknown")
    clin2 = clin2[clin2["OS_MONTHS"].notna() & clin2["OS_STATUS"].notna()].copy()
    clin2 = clin2[clin2["OS_MONTHS"].astype(float) > 0]
    patient_ids = clin2["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    expanded_genes = set(expanded_cm.keys())
    mut_exp = mut[mut["gene.hugoGeneSymbol"].isin(expanded_genes)].copy()
    mut_exp["patient_idx"] = mut_exp["patientId"].map(pid_to_idx)
    mut_exp = mut_exp[mut_exp["patient_idx"].notna()]

    patient_genes = defaultdict(set)
    for _, row in mut_exp.iterrows():
        patient_genes[int(row["patient_idx"])].add(row["gene.hugoGeneSymbol"])

    ct_dict2 = dict(zip(clin2["patientId"], clin2["CANCER_TYPE"]))
    ct_per_patient = {}
    for pid, idx in pid_to_idx.items():
        ct_per_patient[idx] = ct_dict2.get(pid, "Unknown")

    print("\n  Loading COOCCURS edges...")
    load_cooccurrence_edges(driver, patient_genes, ct_per_patient, min_count=10)

    # --- Verify ---
    print(f"\n{'='*70}")
    print("  VERIFICATION")
    print(f"{'='*70}")
    with driver.session() as session:
        for label in ["Gene", "Channel", "Patient"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            print(f"  {label} nodes: {result.single()['c']:,}")

        for rel in ["BELONGS_TO", "PPI", "HAS_MUTATION", "COOCCURS"]:
            result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
            print(f"  {rel} edges: {result.single()['c']:,}")

        # Sample query: TP53's channel profile
        result = session.run(
            "MATCH (g:Gene {name: 'TP53'})-[r:BELONGS_TO]->(c:Channel) "
            "RETURN c.name AS channel, r.weight AS weight ORDER BY r.weight DESC"
        )
        print(f"\n  TP53 channel membership (from BELONGS_TO edges):")
        for record in result:
            print(f"    {record['channel']}: {record['weight']:.3f}")

        # Sample: TP53's PPI neighbors
        result = session.run(
            "MATCH (g:Gene {name: 'TP53'})-[:PPI]-(n:Gene) "
            "RETURN count(n) AS neighbors"
        )
        print(f"  TP53 PPI neighbors: {result.single()['neighbors']}")

        # Sample: most mutated genes
        result = session.run(
            "MATCH (:Patient)-[:HAS_MUTATION]->(g:Gene) "
            "RETURN g.name AS gene, count(*) AS mutations "
            "ORDER BY mutations DESC LIMIT 10"
        )
        print(f"\n  Top 10 most mutated genes:")
        for record in result:
            print(f"    {record['gene']}: {record['mutations']:,} patients")

    elapsed = time.time() - t0
    print(f"\n  Loaded in {elapsed:.1f}s")
    print(f"  Neo4j browser: http://localhost:7474")
    driver.close()


if __name__ == "__main__":
    main()
