"""
Commit channel-organ affinity edges to Neo4j.

Encodes the empirical relationship between mutation channel profiles and
metastatic site preferences. This is structural prior knowledge derived from
MSK-IMPACT + msk_met_2021 DMETS data (2,019 patients).

Key insight: the prognostic meaning of a metastatic site depends on which
channel is driving the cancer. DDR + bone = home = favorable (DFS pattern).
DDR + liver = away = unfavorable (BFS pattern). This interaction is invisible
if met site and channel are encoded independently.

By committing this as graph structure rather than as a model parameter:
  - It applies wherever the graph connects to it (any model, any query)
  - Sparse data (3.9K patients) becomes durable knowledge, not a noisy weight
  - The model learns the residual on top of the prior

Creates:
  - :Organ nodes (one per metastatic site with sufficient data)
  - (Channel)-[:ORGAN_AFFINITY]->(Organ) edges with enrichment scores
  - (CancerType)-[:PRIMARY_SITE]->(Organ) edges mapping cancer to tissue

Usage:
    python3 -u -m gnn.scripts.commit_channel_organ_affinity
    python3 -u -m gnn.scripts.commit_channel_organ_affinity --dry-run
"""

import os, sys, json, argparse, ssl, urllib.request
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, MSK_DATASETS, CBIOPORTAL_BASE, GNN_RESULTS
from gnn.data.graph_changelog import GraphGateway

# Map DMETS site names to organ names (cleaned up for graph)
SITE_TO_ORGAN = {
    "BONE": "Bone",
    "LIVER": "Liver",
    "LUNG": "Lung",
    "CNS_BRAIN": "Brain",
    "DIST_LN": "LymphNode",
    "INTRA_ABDOMINAL": "Peritoneum",
    "PLEURA": "Pleura",
    "ADRENAL_GLAND": "Adrenal",
    "KIDNEY": "Kidney",
    "SKIN": "Skin",
    "BOWEL": "Bowel",
    "BILIARY_TRACT": "BiliaryTract",
    "FEMALE_GENITAL": "FemaleGenital",
    "MALE_GENITAL": "MaleGenital",
    "BLADDER_UT": "Bladder",
    "MEDIASTINUM": "Mediastinum",
    "OVARY": "Ovary",
    "HEAD_NECK": "HeadNeck",
    "BREAST": "Breast",
}

# Cancer type → primary organ mapping
CANCER_TO_ORGAN = {
    "Non-Small Cell Lung Cancer": "Lung",
    "Breast Cancer": "Breast",
    "Colorectal Cancer": "Bowel",
    "Prostate Cancer": "MaleGenital",
    "Glioma": "Brain",
    "Bladder Cancer": "Bladder",
    "Melanoma": "Skin",
    "Renal Cell Carcinoma": "Kidney",
    "Ovarian Cancer": "Ovary",
    "Hepatobiliary Cancer": "Liver",
    "Esophagogastric Cancer": "Bowel",
    "Pancreatic Cancer": "Peritoneum",
    "Endometrial Cancer": "FemaleGenital",
    "Thyroid Cancer": "HeadNeck",
    "Head and Neck Cancer": "HeadNeck",
    "Soft Tissue Sarcoma": "Bone",
    "Small Cell Lung Cancer": "Lung",
}

MIN_PATIENTS_PER_SITE = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fetch_dmets():
    """Fetch DMETS data from cBioPortal."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"{CBIOPORTAL_BASE}/studies/msk_met_2021/clinical-data?clinicalDataType=SAMPLE&pageSize=100000"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    patient_mets = defaultdict(set)
    for item in data:
        attr = item.get("clinicalAttributeId", "")
        if attr.startswith("DMETS_DX_") and attr != "DMETS_DX_UNSPECIFIED":
            if item.get("value") == "Yes":
                site = attr.replace("DMETS_DX_", "")
                patient_mets[item["patientId"]].add(site)

    return patient_mets


def compute_affinity_matrix(patient_mets, mutations_df, clinical_df):
    """Compute channel-organ affinity enrichment matrix.

    For each (channel, met_site) pair, measures whether patients who
    metastasize to that site have more mutations in that channel than
    the baseline population.

    Returns:
        edges: list of {from, to, enrichment, p_site, p_baseline, n_patients, is_home}
        organ_nodes: list of {name, n_patients, top_channels}
    """
    train_pids = set(clinical_df["patientId"])
    overlap = train_pids & set(patient_mets.keys())

    # Channel profiles per patient
    profiles = {}
    for pid, group in mutations_df[mutations_df["patientId"].isin(overlap)].groupby("patientId"):
        genes = group["gene.hugoGeneSymbol"].values
        channels = [CHANNEL_MAP.get(g) for g in genes if CHANNEL_MAP.get(g)]
        if not channels:
            continue
        ch_counts = defaultdict(int)
        for c in channels:
            ch_counts[c] += 1
        total = len(channels)
        profiles[pid] = {ch: n / total for ch, n in ch_counts.items()}

    print(f"  Patients with channel profiles + DMETS: {len(profiles)}")

    # Baseline channel fracs
    baseline = defaultdict(float)
    for pid, ch_fracs in profiles.items():
        for ch, f in ch_fracs.items():
            baseline[ch] += f
    for ch in baseline:
        baseline[ch] /= len(profiles)

    # Compute per-site enrichment
    edges = []
    organ_nodes = []

    for site_raw, organ_name in SITE_TO_ORGAN.items():
        site_pids = [pid for pid in profiles if site_raw in patient_mets.get(pid, set())]
        if len(site_pids) < MIN_PATIENTS_PER_SITE:
            continue

        site_ch = defaultdict(float)
        for pid in site_pids:
            for ch, f in profiles[pid].items():
                site_ch[ch] += f
        for ch in site_ch:
            site_ch[ch] /= len(site_pids)

        # Find enriched channels for this organ
        enrichments = {}
        for ch in CHANNEL_NAMES:
            base = baseline.get(ch, 0.001)
            enrich = site_ch.get(ch, 0) / base
            enrichments[ch] = enrich

        top_channels = sorted(enrichments.items(), key=lambda x: -x[1])[:3]
        organ_nodes.append({
            "name": organ_name,
            "dmets_key": site_raw,
            "n_patients": len(site_pids),
            "top_channels": [{"channel": ch, "enrichment": round(e, 3)} for ch, e in top_channels],
        })

        for ch in CHANNEL_NAMES:
            enrich = enrichments[ch]
            # Only commit meaningful affinities (>5% enrichment or depletion)
            if abs(enrich - 1.0) < 0.05:
                continue

            is_home = enrich > 1.1
            edges.append({
                "from": ch,
                "to": organ_name,
                "enrichment": round(enrich, 3),
                "site_frac": round(site_ch.get(ch, 0), 4),
                "baseline_frac": round(baseline.get(ch, 0), 4),
                "n_patients": len(site_pids),
                "is_home": is_home,
                "weight": round(abs(enrich - 1.0), 3),
            })

    return edges, organ_nodes


def main():
    args = parse_args()

    print("=" * 60)
    print("  COMMIT CHANNEL-ORGAN AFFINITY TO KNOWLEDGE GRAPH")
    print("=" * 60)

    # Fetch DMETS data
    print("\n  Fetching metastasis site data...")
    patient_mets = fetch_dmets()
    print(f"  Patients with DMETS data: {len(patient_mets)}")

    # Load training data
    paths = MSK_DATASETS["msk_impact_50k"]
    mutations = pd.read_csv(paths["mutations"])
    clinical = pd.read_csv(paths["clinical"])
    sample_clinical = pd.read_csv(paths["sample_clinical"])

    # Compute affinity
    print("\n  Computing channel-organ affinity matrix...")
    affinity_edges, organ_nodes = compute_affinity_matrix(patient_mets, mutations, clinical)

    print(f"\n  Organ nodes to create: {len(organ_nodes)}")
    for o in organ_nodes:
        top = ", ".join(f"{t['channel']}={t['enrichment']:.2f}x" for t in o["top_channels"])
        print(f"    {o['name']:<15s} (n={o['n_patients']:>5d}): {top}")

    home_edges = [e for e in affinity_edges if e["is_home"]]
    away_edges = [e for e in affinity_edges if not e["is_home"]]
    print(f"\n  Affinity edges: {len(affinity_edges)} total")
    print(f"    Home (enriched): {len(home_edges)}")
    print(f"    Away (depleted): {len(away_edges)}")

    # Cancer type → organ edges
    ct_organ_edges = []
    ct_lookup = dict(zip(sample_clinical["patientId"], sample_clinical["CANCER_TYPE"]))
    organ_names = {o["name"] for o in organ_nodes}
    for ct, organ in CANCER_TO_ORGAN.items():
        if organ in organ_names:
            ct_organ_edges.append({"from": ct, "to": organ, "weight": 1.0})

    print(f"\n  Cancer type → organ edges: {len(ct_organ_edges)}")

    # Commit to graph
    gw = GraphGateway(dry_run=args.dry_run)

    # Create Organ nodes
    print("\n  Creating Organ nodes...")
    organ_updates = []
    for o in organ_nodes:
        organ_updates.append({
            "name": o["name"],
            "dmets_key": o["dmets_key"],
            "n_patients": o["n_patients"],
        })

    if not args.dry_run:
        with gw.driver.session() as s:
            for o in organ_updates:
                s.run("""
                    MERGE (org:Organ {name: $name})
                    ON CREATE SET org.dmets_key = $dmets_key,
                                  org.n_patients = $n_patients,
                                  org.source = 'msk_met_2021_dmets',
                                  org.created_at = datetime()
                    ON MATCH SET  org.n_patients = $n_patients,
                                  org.updated_at = datetime()
                """, **o)
        print(f"  Created/updated {len(organ_updates)} Organ nodes")
    else:
        print(f"  [DRY RUN] Would create {len(organ_updates)} Organ nodes")

    # Commit Channel → Organ affinity edges
    print("\n  Committing ORGAN_AFFINITY edges...")
    n = gw.merge_edges(
        "ORGAN_AFFINITY",
        affinity_edges,
        source="msk_met_2021_channel_organ_analysis",
        source_detail=f"n_patients={len(patient_mets)}, min_site={MIN_PATIENTS_PER_SITE}",
        match_from=("Channel", "name"),
        match_to=("Organ", "name"),
    )

    # Commit CancerType → Organ edges
    if ct_organ_edges:
        print("\n  Committing PRIMARY_SITE edges...")
        n2 = gw.merge_edges(
            "PRIMARY_SITE",
            ct_organ_edges,
            source="cancer_type_organ_mapping",
            match_from=("CancerType", "name"),
            match_to=("Organ", "name"),
        )

    gw.close()

    # Save results
    results = {
        "organ_nodes": organ_nodes,
        "affinity_edges": affinity_edges,
        "ct_organ_edges": ct_organ_edges,
        "n_patients": len(patient_mets),
    }
    out_path = os.path.join(GNN_RESULTS, "channel_organ_affinity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
