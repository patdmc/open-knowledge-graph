#!/usr/bin/env python3
"""
Add Organ nodes and MET_COOCCURS edges to Neo4j from MSK-MET 2021 data.

Organ adjacency is derived from metastatic site co-occurrence patterns:
if two organs co-occur as metastatic sites more than expected by chance,
they share a MET_COOCCURS edge weighted by log-odds ratio.

Also adds ORGAN_TROPISM edges from CancerType nodes to Organ nodes,
weighted by the rate at which each cancer type metastasizes to each organ.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from itertools import combinations

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "openknowledgegraph")

DMETS_COLS = [
    "DMETS_DX_ADRENAL_GLAND",
    "DMETS_DX_BILIARY_TRACT",
    "DMETS_DX_BLADDER_UT",
    "DMETS_DX_BONE",
    "DMETS_DX_BOWEL",
    "DMETS_DX_BREAST",
    "DMETS_DX_CNS_BRAIN",
    "DMETS_DX_DIST_LN",
    "DMETS_DX_FEMALE_GENITAL",
    "DMETS_DX_HEAD_NECK",
    "DMETS_DX_INTRA_ABDOMINAL",
    "DMETS_DX_KIDNEY",
    "DMETS_DX_LIVER",
    "DMETS_DX_LUNG",
    "DMETS_DX_MALE_GENITAL",
    "DMETS_DX_MEDIASTINUM",
    "DMETS_DX_OVARY",
    "DMETS_DX_PLEURA",
    "DMETS_DX_PNS",
    "DMETS_DX_SKIN",
]

# Map DMETS columns to clean organ names
ORGAN_NAMES = {
    "DMETS_DX_ADRENAL_GLAND": "Adrenal",
    "DMETS_DX_BILIARY_TRACT": "Biliary",
    "DMETS_DX_BLADDER_UT": "Bladder",
    "DMETS_DX_BONE": "Bone",
    "DMETS_DX_BOWEL": "Bowel",
    "DMETS_DX_BREAST": "Breast",
    "DMETS_DX_CNS_BRAIN": "Brain",
    "DMETS_DX_DIST_LN": "DistantLN",
    "DMETS_DX_FEMALE_GENITAL": "FemaleGenital",
    "DMETS_DX_HEAD_NECK": "HeadNeck",
    "DMETS_DX_INTRA_ABDOMINAL": "IntraAbdominal",
    "DMETS_DX_KIDNEY": "Kidney",
    "DMETS_DX_LIVER": "Liver",
    "DMETS_DX_LUNG": "Lung",
    "DMETS_DX_MALE_GENITAL": "MaleGenital",
    "DMETS_DX_MEDIASTINUM": "Mediastinum",
    "DMETS_DX_OVARY": "Ovary",
    "DMETS_DX_PLEURA": "Pleura",
    "DMETS_DX_PNS": "PNS",
    "DMETS_DX_SKIN": "Skin",
}

# Body cavity groupings (for node properties)
BODY_CAVITY = {
    "Adrenal": "retroperitoneal",
    "Biliary": "abdominal",
    "Bladder": "pelvic",
    "Bone": "systemic",
    "Bowel": "abdominal",
    "Breast": "thoracic_wall",
    "Brain": "cranial",
    "DistantLN": "systemic",
    "FemaleGenital": "pelvic",
    "HeadNeck": "cranial",
    "IntraAbdominal": "abdominal",
    "Kidney": "retroperitoneal",
    "Liver": "abdominal",
    "Lung": "thoracic",
    "MaleGenital": "pelvic",
    "Mediastinum": "thoracic",
    "Ovary": "pelvic",
    "Pleura": "thoracic",
    "PNS": "systemic",
    "Skin": "systemic",
}

CLINICAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis", "cache", "msk_met_2021_full_clinical.csv",
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gnn", "results", "organ_adjacency",
)


def load_data():
    df = pd.read_csv(CLINICAL_PATH)
    for col in DMETS_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    return df


def compute_cooccurrence(df):
    """Compute log-odds ratio for every organ pair."""
    n = len(df)
    edges = []

    for col_a, col_b in combinations(DMETS_COLS, 2):
        organ_a = ORGAN_NAMES[col_a]
        organ_b = ORGAN_NAMES[col_b]

        both = ((df[col_a] == 1) & (df[col_b] == 1)).sum()
        a_only = ((df[col_a] == 1) & (df[col_b] == 0)).sum()
        b_only = ((df[col_a] == 0) & (df[col_b] == 1)).sum()
        neither = ((df[col_a] == 0) & (df[col_b] == 0)).sum()

        pa = (both + a_only) / n
        pb = (both + b_only) / n
        expected = pa * pb * n

        if a_only > 0 and b_only > 0 and neither > 0 and both > 0:
            odds_ratio = (both * neither) / (a_only * b_only)
            log_or = float(np.log(odds_ratio))
        else:
            odds_ratio = None
            log_or = None

        table = np.array([[both, a_only], [b_only, neither]])
        if min(both, a_only, b_only, neither) < 5:
            _, p = fisher_exact(table)
        else:
            _, p, _, _ = chi2_contingency(table)

        edges.append({
            "organ_a": organ_a,
            "organ_b": organ_b,
            "both": int(both),
            "a_only": int(a_only),
            "b_only": int(b_only),
            "neither": int(neither),
            "prevalence_a": float(pa),
            "prevalence_b": float(pb),
            "expected": float(expected),
            "observed": int(both),
            "obs_exp_ratio": float(both / expected) if expected > 0 else None,
            "odds_ratio": float(odds_ratio) if odds_ratio else None,
            "log_odds_ratio": log_or,
            "p_value": float(p),
        })

    return edges


def compute_tropism(df):
    """Compute cancer-type → organ tropism rates."""
    tropism = []
    ct_counts = df["CANCER_TYPE"].value_counts()

    for ct in ct_counts.index:
        if ct_counts[ct] < 50:
            continue
        ct_sub = df[df["CANCER_TYPE"] == ct]
        n_ct = len(ct_sub)

        for col in DMETS_COLS:
            organ = ORGAN_NAMES[col]
            n_met = int(ct_sub[col].sum())
            rate = n_met / n_ct
            # Background rate
            bg_rate = df[col].mean()

            if n_met >= 5:
                tropism.append({
                    "cancer_type": ct,
                    "organ": organ,
                    "n_patients": n_ct,
                    "n_met": n_met,
                    "rate": float(rate),
                    "background_rate": float(bg_rate),
                    "enrichment": float(rate / bg_rate) if bg_rate > 0 else None,
                })

    return tropism


def write_to_neo4j(driver, organs, edges, tropism, df):
    """Create Organ nodes and MET_COOCCURS edges in Neo4j."""

    # 1. Create Organ nodes
    print("\n=== Creating Organ nodes ===")
    with driver.session() as session:
        for col in DMETS_COLS:
            organ = ORGAN_NAMES[col]
            cavity = BODY_CAVITY[organ]
            prevalence = float(df[col].mean())
            n_patients = int(df[col].sum())

            session.run("""
                MERGE (o:Organ {name: $name})
                SET o.body_cavity = $cavity,
                    o.met_prevalence = $prevalence,
                    o.n_patients = $n_patients,
                    o.source = 'MSK-MET-2021',
                    o.provenance = 'encode_organ_adjacency.py'
            """, name=organ, cavity=cavity, prevalence=prevalence,
                n_patients=n_patients)

        print(f"  Created {len(DMETS_COLS)} Organ nodes")

    # 2. Create MET_COOCCURS edges (only significant ones)
    print("\n=== Creating MET_COOCCURS edges ===")
    n_created = 0
    with driver.session() as session:
        for edge in edges:
            if edge["p_value"] > 0.001:
                continue
            if edge["log_odds_ratio"] is None:
                continue

            session.run("""
                MATCH (a:Organ {name: $organ_a})
                MATCH (b:Organ {name: $organ_b})
                MERGE (a)-[r:MET_COOCCURS]-(b)
                SET r.log_odds_ratio = $lor,
                    r.odds_ratio = $or_val,
                    r.obs_exp_ratio = $oer,
                    r.p_value = $pval,
                    r.observed = $observed,
                    r.expected = $expected,
                    r.source = 'MSK-MET-2021',
                    r.provenance = 'encode_organ_adjacency.py'
            """,
                organ_a=edge["organ_a"],
                organ_b=edge["organ_b"],
                lor=edge["log_odds_ratio"],
                or_val=edge["odds_ratio"],
                oer=edge["obs_exp_ratio"],
                pval=edge["p_value"],
                observed=edge["observed"],
                expected=edge["expected"],
            )
            n_created += 1

    print(f"  Created {n_created} MET_COOCCURS edges (p < 0.001)")

    # 3. Create ORGAN_TROPISM edges from CancerType to Organ
    print("\n=== Creating ORGAN_TROPISM edges ===")
    # First ensure CancerType nodes exist
    ct_set = set()
    n_tropism = 0
    with driver.session() as session:
        for t in tropism:
            ct = t["cancer_type"]
            if ct not in ct_set:
                session.run("""
                    MERGE (c:CancerType {name: $name})
                    SET c.provenance = 'encode_organ_adjacency.py'
                """, name=ct)
                ct_set.add(ct)

            session.run("""
                MATCH (c:CancerType {name: $ct})
                MATCH (o:Organ {name: $organ})
                MERGE (c)-[r:ORGAN_TROPISM]->(o)
                SET r.rate = $rate,
                    r.enrichment = $enrichment,
                    r.n_met = $n_met,
                    r.n_patients = $n_patients,
                    r.background_rate = $bg_rate,
                    r.source = 'MSK-MET-2021',
                    r.provenance = 'encode_organ_adjacency.py'
            """,
                ct=t["cancer_type"],
                organ=t["organ"],
                rate=t["rate"],
                enrichment=t["enrichment"],
                n_met=t["n_met"],
                n_patients=t["n_patients"],
                bg_rate=t["background_rate"],
            )
            n_tropism += 1

    print(f"  Created {len(ct_set)} CancerType nodes")
    print(f"  Created {n_tropism} ORGAN_TROPISM edges")


def main():
    print("Loading MSK-MET 2021 data...")
    df = load_data()
    print(f"  {len(df)} patients")

    print("\nComputing organ co-occurrence...")
    edges = compute_cooccurrence(df)
    sig_edges = [e for e in edges if e["p_value"] < 0.001 and e["log_odds_ratio"] is not None]
    pos_edges = [e for e in sig_edges if e["log_odds_ratio"] > 0]
    neg_edges = [e for e in sig_edges if e["log_odds_ratio"] < 0]
    print(f"  {len(edges)} total pairs, {len(sig_edges)} significant (p<0.001)")
    print(f"  {len(pos_edges)} positive (co-occur), {len(neg_edges)} negative (anti-co-occur)")

    print("\nComputing cancer type tropism...")
    tropism = compute_tropism(df)
    print(f"  {len(tropism)} cancer-type → organ edges")

    # Save JSON for bilinear model
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "organ_cooccurrence.json"), "w") as f:
        json.dump(edges, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "organ_tropism.json"), "w") as f:
        json.dump(tropism, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/")

    # Write to Neo4j
    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("  Connected")

    write_to_neo4j(driver, ORGAN_NAMES, edges, tropism, df)

    driver.close()
    print("\nDone. Organ adjacency graph created.")


if __name__ == "__main__":
    main()
