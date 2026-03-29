#!/usr/bin/env python3
"""
Pure graph walk scorer using learned bilinear W matrices.

No transformer. No neural network at inference time. Just:
  1. Load per-CT W matrices from bilinear training
  2. For each patient's mutations, compute pairwise edge scores via W
  3. Propagate damage through the graph using W-weighted adjacency
  4. Add clinical covariates as linear term
  5. Evaluate on holdback

The W matrix tells us how edge types interact per cancer type.
The walk propagates damage: if gene A is mutated and connected to gene B
via edges weighted by W, gene B accumulates proportional hazard.

Usage:
    python3 -u -m gnn.scripts.bilinear_walk_scorer
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy import sparse as sp_sparse
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "bilinear_walk_scorer",
)
BILINEAR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "bilinear_edge",
)
CLINICAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "analysis", "cache", "msk_met_2021_full_clinical.csv",
)
MUTATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "analysis", "cache", "msk_met_2021_mutations.csv",
)

NON_SILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
}

CHANNELS = list(CHANNEL_NAMES) if isinstance(CHANNEL_NAMES, list) else list(CHANNEL_NAMES.keys())

# Edge types matching bilinear model
EDGE_TYPES = [
    'PPI', 'COUPLES', 'SYNTHETIC_LETHAL', 'CO_ESSENTIAL',
    'CO_EXPRESSED', 'CO_CNA', 'ATTENDS_TO',
    'HAS_SENSITIVITY_EVIDENCE', 'HAS_RESISTANCE_EVIDENCE',
    'CO_SENSITIVE', 'CO_RESISTANT', 'DRUG_CONFLICT',
    'CO_TISSUE_EXPR', 'CO_BIALLELIC',
    'ANALOGOUS', 'CONVERGES', 'TRANSPOSES',
    'SAME_STRAND',
]
N_EDGE_TYPES = len(EDGE_TYPES)

CLINICAL_COVARIATES = [
    'age', 'sex', 'fga', 'msi_score',
    'met_site_count', 'tumor_purity', 'sample_type', 'tmb',
]

# DMETS columns for organ topology features
DMETS_COLS = [
    "DMETS_DX_ADRENAL_GLAND", "DMETS_DX_BILIARY_TRACT", "DMETS_DX_BLADDER_UT",
    "DMETS_DX_BONE", "DMETS_DX_BOWEL", "DMETS_DX_BREAST",
    "DMETS_DX_CNS_BRAIN", "DMETS_DX_DIST_LN", "DMETS_DX_FEMALE_GENITAL",
    "DMETS_DX_HEAD_NECK", "DMETS_DX_INTRA_ABDOMINAL", "DMETS_DX_KIDNEY",
    "DMETS_DX_LIVER", "DMETS_DX_LUNG", "DMETS_DX_MALE_GENITAL",
    "DMETS_DX_MEDIASTINUM", "DMETS_DX_OVARY", "DMETS_DX_PLEURA",
    "DMETS_DX_PNS", "DMETS_DX_SKIN",
]

ORGAN_COOCCURRENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "organ_adjacency", "organ_cooccurrence.json",
)


def load_bilinear_artifacts():
    """Load W matrices, edge matrix, gene vocab, and clinical betas."""
    with open(os.path.join(BILINEAR_DIR, "W_matrices.json")) as f:
        W_all = json.load(f)

    edge_matrix = np.load(os.path.join(BILINEAR_DIR, "raw_edge_matrix.npy"))

    with open(os.path.join(BILINEAR_DIR, "gene_vocab.json")) as f:
        gene_vocab = json.load(f)

    model_state = torch.load(
        os.path.join(BILINEAR_DIR, "best_model.pt"),
        map_location="cpu", weights_only=True,
    )

    # Extract clinical betas if present
    beta_clinical = None
    if "beta_clinical" in model_state:
        beta_clinical = model_state["beta_clinical"].numpy()

    # Extract bias
    bias = model_state["bias"].numpy()

    return W_all, edge_matrix, gene_vocab, beta_clinical, bias


# CT index → property key suffix (must match write_w_confidence.py)
CT_IDX_TO_PROP = {
    0: "nsclc", 1: "colorectal", 2: "breast", 3: "prostate",
    4: "pancreatic", 5: "endometrial", 6: "ovarian", 7: "bladder",
    8: "melanoma", 9: "hepatobiliary", 10: "esophagogastric",
    11: "sarcoma", 12: "thyroid", 13: "renal", 14: "head_neck",
    15: "gist", 16: "germ_cell", 17: "sclc", 18: "mesothelioma",
    19: "appendiceal", 20: "uterine_sarcoma", 21: "salivary",
    22: "gi_neuroendocrine", 23: "skin_nonmelanoma", 24: "cervical",
    25: "small_bowel", 26: "anal",
}
# Reverse: cancer type full name → property suffix
CT_FULLNAME_TO_PROP = {
    "Non-Small Cell Lung Cancer": "nsclc",
    "Colorectal Cancer": "colorectal",
    "Breast Cancer": "breast",
    "Prostate Cancer": "prostate",
    "Pancreatic Cancer": "pancreatic",
    "Endometrial Cancer": "endometrial",
    "Ovarian Cancer": "ovarian",
    "Bladder Cancer": "bladder",
    "Melanoma": "melanoma",
    "Hepatobiliary Cancer": "hepatobiliary",
    "Esophagogastric Cancer": "esophagogastric",
    "Soft Tissue Sarcoma": "sarcoma",
    "Thyroid Cancer": "thyroid",
    "Renal Cell Carcinoma": "renal",
    "Head and Neck Cancer": "head_neck",
    "Gastrointestinal Stromal Tumor": "gist",
    "Germ Cell Tumor": "germ_cell",
    "Small Cell Lung Cancer": "sclc",
    "Mesothelioma": "mesothelioma",
    "Appendiceal Cancer": "appendiceal",
    "Uterine Sarcoma": "uterine_sarcoma",
    "Salivary Gland Cancer": "salivary",
    "Gastrointestinal Neuroendocrine Tumor": "gi_neuroendocrine",
    "Skin Cancer, Non-Melanoma": "skin_nonmelanoma",
    "Cervical Cancer": "cervical",
    "Small Bowel Cancer": "small_bowel",
    "Anal Cancer": "anal",
}


def load_graph_calibrated_adjacency(ct_prop_key, gene_vocab):
    """Load W-calibrated edge weights directly from Neo4j.

    Reads the w_<ct> property that was written back to the graph
    by write_w_confidence.py. The graph IS the model.

    Only uses structural edge types for walk topology.
    """
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    G = len(gene_vocab)
    prop_name = f"w_{ct_prop_key}"

    rows, cols, vals = [], [], []
    n_edges = 0

    with driver.session() as session:
        for etype in STRUCTURAL_EDGE_TYPES:
            result = session.run(f"""
                MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
                  AND (r.deprecated IS NULL OR r.deprecated = false)
                  AND r.{prop_name} IS NOT NULL
                RETURN a.name AS g1, b.name AS g2, r.{prop_name} AS w
            """)
            for r in result:
                g1, g2, w = r["g1"], r["g2"], r["w"]
                i = gene_vocab.get(g1)
                j = gene_vocab.get(g2)
                if i is None or j is None:
                    continue
                if abs(w) < 1e-6:
                    continue
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w, w])
                n_edges += 1

    driver.close()

    if not rows:
        return sp_sparse.csr_matrix((G, G)), 0

    A = sp_sparse.csr_matrix((vals, (rows, cols)), shape=(G, G))
    return A, n_edges


def load_patients():
    """Load clinical + mutation data, return patient-level structures."""
    clin = pd.read_csv(CLINICAL_PATH)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(
        lambda x: 1 if "DECEASED" in str(x).upper() or "1:" in str(x) else 0
    )
    clin = clin[clin["time"] > 0].copy()

    mut = pd.read_csv(MUTATIONS_PATH)
    mut = mut[mut["mutationType"].isin(NON_SILENT)]
    channel_genes = set(CHANNEL_MAP.keys())
    mut = mut[mut["gene.hugoGeneSymbol"].isin(channel_genes)]

    # Per-patient mutation sets
    patient_genes = defaultdict(set)
    for _, row in mut.iterrows():
        pid = row["patientId"]
        gene = row["gene.hugoGeneSymbol"]
        patient_genes[pid].add(gene)

    return clin, patient_genes


def build_clinical_features(clin):
    """Extract clinical covariates, z-score normalize.

    Returns the 8 base clinical features that match the bilinear model's
    beta_clinical weights.
    """
    n = len(clin)
    covs = np.zeros((n, len(CLINICAL_COVARIATES)), dtype=np.float32)

    for i, (_, row) in enumerate(clin.iterrows()):
        age = pd.to_numeric(row.get("AGE_AT_SEQUENCING"), errors="coerce")
        covs[i, 0] = age if pd.notna(age) else 60.0
        covs[i, 1] = 1.0 if row.get("SEX") == "Male" else 0.0
        fga = pd.to_numeric(row.get("FRACTION_GENOME_ALTERED"), errors="coerce")
        covs[i, 2] = fga if pd.notna(fga) else 0.0
        msi = pd.to_numeric(row.get("MSI_SCORE"), errors="coerce")
        covs[i, 3] = msi if pd.notna(msi) else 0.0
        msc = pd.to_numeric(row.get("MET_SITE_COUNT"), errors="coerce")
        covs[i, 4] = msc if pd.notna(msc) else 0.0
        tp = pd.to_numeric(row.get("TUMOR_PURITY"), errors="coerce")
        covs[i, 5] = tp / 100.0 if pd.notna(tp) else 0.4
        covs[i, 6] = 1.0 if row.get("SAMPLE_TYPE") == "Metastasis" else 0.0
        tmb = pd.to_numeric(row.get("TMB_NONSYNONYMOUS"), errors="coerce")
        covs[i, 7] = tmb if pd.notna(tmb) else 0.0

    for col_idx in [0, 2, 3, 4, 5, 7]:
        vals = covs[:, col_idx]
        mu, std = vals.mean(), vals.std()
        if std > 1e-8:
            covs[:, col_idx] = (vals - mu) / std

    return covs


def build_organ_features(clin):
    """Extract per-patient met site binary vector + organ topology score.

    For each patient, encode:
    1. Binary met site vector (20 organs)
    2. Organ topology score: sum of log-odds ratios for all co-occurring
       met site pairs. High score = met sites that co-occur more than
       expected (anatomically clustered). Low score = met sites that
       are anti-correlated (spread across body cavities = worse).
    """
    n = len(clin)
    n_organs = len(DMETS_COLS)

    # Binary met site matrix
    met_binary = np.zeros((n, n_organs), dtype=np.float32)
    for j, col in enumerate(DMETS_COLS):
        if col in clin.columns:
            met_binary[:, j] = (clin[col] == "Yes").astype(float).values

    # Organ co-occurrence topology score
    topo_scores = np.zeros(n, dtype=np.float32)

    if os.path.exists(ORGAN_COOCCURRENCE_PATH):
        with open(ORGAN_COOCCURRENCE_PATH) as f:
            cooc = json.load(f)

        # Map DMETS columns to organ names used in cooccurrence data
        dmets_to_organ = {
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

        # Build log-odds lookup
        lor_lookup = {}
        for edge in cooc:
            a, b = edge["organ_a"], edge["organ_b"]
            lor = edge.get("log_odds_ratio")
            if lor is not None:
                lor_lookup[(a, b)] = lor
                lor_lookup[(b, a)] = lor

        # For each patient, sum LOR across all co-occurring met site pairs
        organ_names = [dmets_to_organ.get(col, col) for col in DMETS_COLS]
        for i in range(n):
            active = [j for j in range(n_organs) if met_binary[i, j] > 0]
            if len(active) < 2:
                continue
            score = 0.0
            n_pairs = 0
            for a_idx in range(len(active)):
                for b_idx in range(a_idx + 1, len(active)):
                    pair = (organ_names[active[a_idx]], organ_names[active[b_idx]])
                    lor = lor_lookup.get(pair, 0.0)
                    score += lor
                    n_pairs += 1
            if n_pairs > 0:
                topo_scores[i] = score / n_pairs  # average LOR

        # z-score normalize
        mu, std = topo_scores.mean(), topo_scores.std()
        if std > 1e-8:
            topo_scores = (topo_scores - mu) / std

        print(f"  Organ topology: {(met_binary.sum(axis=1) > 0).sum()} patients "
              f"with met site data")
    else:
        print("  WARNING: organ co-occurrence data not found")

    return met_binary, topo_scores


# Structural edge types: real biological relationships that define graph topology.
# These create walkable paths. Attribute edge types (SAME_STRAND, CO_CNA, etc.)
# modulate weights on structural edges but don't create new paths.
STRUCTURAL_EDGE_TYPES = {
    'PPI', 'COUPLES', 'SYNTHETIC_LETHAL', 'CO_ESSENTIAL',
    'ATTENDS_TO', 'HAS_SENSITIVITY_EVIDENCE', 'HAS_RESISTANCE_EVIDENCE',
    'ANALOGOUS', 'CONVERGES', 'TRANSPOSES',
}
STRUCTURAL_INDICES = [i for i, et in enumerate(EDGE_TYPES) if et in STRUCTURAL_EDGE_TYPES]


def compute_w_weighted_adjacency(W_ct, edge_matrix, gene_vocab):
    """Build W-weighted adjacency using structural edges only.

    Walk topology comes from structural edges (PPI, COUPLES, SL, etc.).
    Walk weights come from the full 18-dim bilinear score e^T W e.

    Attribute edge types (SAME_STRAND, CO_CNA, CO_EXPRESSED, etc.) modulate
    the weight of existing structural edges but don't create new paths.
    This prevents the walk from spreading through non-biological connections.

    Returns sparse (G, G) adjacency matrix.
    """
    G = len(gene_vocab)
    W = np.array(W_ct)
    W = (W + W.T) / 2

    rows, cols, vals = [], [], []
    for i in range(G):
        for j in range(i + 1, G):
            e = edge_matrix[i, j, :]  # (18,)

            # Only create an edge if at least one STRUCTURAL edge type is nonzero
            has_structural = any(e[k] > 0 for k in STRUCTURAL_INDICES)
            if not has_structural:
                continue

            # Weight uses the FULL edge vector (all 18 dims including attributes)
            score = float(e @ W @ e)
            if abs(score) < 1e-6:
                continue

            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([score, score])

    if not rows:
        return sp_sparse.csr_matrix((G, G))

    A = sp_sparse.csr_matrix((vals, (rows, cols)), shape=(G, G))
    return A


def propagate(A, init_hazard, alpha=0.1, n_steps=3):
    """Propagate hazard through W-weighted adjacency.

    M = (I + alpha * D^-1 * A)^K
    h_final = h_init @ M^T
    """
    G = A.shape[0]

    degree = np.abs(np.array(A.sum(axis=1)).flatten())
    degree[degree == 0] = 1.0
    D_inv = sp_sparse.diags(1.0 / degree)
    DA = D_inv @ A

    step = sp_sparse.eye(G) + alpha * DA
    step_dense = step.toarray()

    M = step_dense.copy()
    for _ in range(n_steps - 1):
        M = M @ step_dense

    return init_hazard @ M.T


def score_patient_graph(mutated_genes, gene_vocab, A_ct):
    """Score a patient by reading THEIR edges from the graph.

    The patient's mutations define WHICH edges to read (personal).
    The cancer type selects WHICH weight to read on those edges (context).
    The edge itself is universal (graph knowledge).

    1. Edges BETWEEN mutated genes (co-mutation context)
    2. Edges FROM mutated genes to their neighbors (damage reach)

    Returns:
        inter_score: sum of edge weights between mutated gene pairs
        reach_score: sum of edge weights from mutated genes to non-mutated neighbors
        n_inter: number of edges between mutated genes
        n_reach: number of reach edges
    """
    if not mutated_genes or A_ct is None:
        return 0.0, 0.0, 0, 0

    mut_idx = set()
    for gene in mutated_genes:
        idx = gene_vocab.get(gene)
        if idx is not None:
            mut_idx.add(idx)

    if not mut_idx:
        return 0.0, 0.0, 0, 0

    # Edges BETWEEN mutated genes — the co-mutation context
    inter_score = 0.0
    n_inter = 0
    mut_list = sorted(mut_idx)
    for a in range(len(mut_list)):
        for b in range(a + 1, len(mut_list)):
            w = A_ct[mut_list[a], mut_list[b]]
            if abs(w) > 1e-6:
                inter_score += w
                n_inter += 1

    # Edges FROM mutated genes to non-mutated neighbors — damage reach
    reach_score = 0.0
    n_reach = 0
    for idx in mut_idx:
        row = A_ct.getrow(idx)
        for neighbor, w in zip(row.indices, row.data):
            if neighbor not in mut_idx and abs(w) > 1e-6:
                reach_score += w
                n_reach += 1

    return inter_score, reach_score, n_inter, n_reach


def compute_node_hazards(patient_genes_dict, times, events, valid_pids,
                         gene_vocab, ct_patients):
    """Per-CT node hazards: event rate difference vs CT baseline."""
    G = len(gene_vocab)
    idx_to_gene = {v: k for k, v in gene_vocab.items()}

    # Global gene→patient sets
    gene_patients = defaultdict(set)
    for pid in valid_pids:
        for g in patient_genes_dict.get(pid, set()):
            if g in gene_vocab:
                gene_patients[g].add(pid)

    ct_hazards = {}
    for ct_name, ct_pids in ct_patients.items():
        ct_pids_set = set(ct_pids)
        ct_valid = ct_pids_set & set(valid_pids)
        if len(ct_valid) < 30:
            ct_hazards[ct_name] = np.zeros(G)
            continue

        ct_arr = np.array(sorted(ct_valid))
        ct_er = events[ct_arr].mean()

        h = np.zeros(G)
        ct_gene_patients = defaultdict(set)
        for pid in ct_valid:
            for g in patient_genes_dict.get(pid, set()):
                if g in gene_vocab:
                    ct_gene_patients[g].add(pid)

        for gene, gidx in gene_vocab.items():
            pts = ct_gene_patients.get(gene, set())
            if len(pts) < 10:
                # Fall back to global
                g_pts = gene_patients.get(gene, set())
                if len(g_pts) < 20:
                    continue
                g_arr = np.array(sorted(g_pts))
                g_er = events[g_arr].mean()
                global_er = events.mean()
                shrink = len(g_pts) / (len(g_pts) + 20.0)
                h[gidx] = (g_er - global_er) * shrink
                continue

            pts_arr = np.array(sorted(pts))
            gene_er = events[pts_arr].mean()
            shrink = len(pts) / (len(pts) + 15.0)
            h[gidx] = (gene_er - ct_er) * shrink

        ct_hazards[ct_name] = h

    return ct_hazards


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  BILINEAR WALK SCORER — Pure graph, no transformer")
    print("  W matrices from bilinear training + clinical covariates")
    print("=" * 70)

    # Load bilinear artifacts
    print("\nLoading bilinear model artifacts...")
    W_all, edge_matrix, gene_vocab, beta_clinical, bias = load_bilinear_artifacts()
    G = len(gene_vocab)
    n_cts = len(W_all)
    print(f"  {n_cts} cancer types, {G} genes, {N_EDGE_TYPES} edge types")
    if beta_clinical is not None:
        print(f"  Clinical betas: {beta_clinical.shape}")

    # Load patient data
    print("\nLoading patient data...")
    clin, patient_genes = load_patients()
    N = len(clin)
    print(f"  {N} patients with valid OS data")

    # Build index arrays
    pids = clin["patientId"].values
    times_arr = clin["time"].values
    events_arr = clin["event"].values
    ct_arr = clin["CANCER_TYPE"].values

    pid_to_row = {pid: i for i, pid in enumerate(pids)}

    # Clinical features
    print("\nBuilding clinical features...")
    clinical_feats = build_clinical_features(clin)

    # Organ topology features
    print("\nBuilding organ topology features...")
    met_binary, topo_scores = build_organ_features(clin)

    # CT mapping (must match bilinear model)
    # The bilinear model used CT_0, CT_1, ... — we need the same mapping
    with open(os.path.join(BILINEAR_DIR, "results.json")) as f:
        bilinear_results = json.load(f)
    # Reconstruct CT mapping from the atlas dataset ordering
    ct_unique = []
    ct_name_to_idx = {}
    for ct in ct_arr:
        if ct not in ct_name_to_idx:
            ct_name_to_idx[ct] = len(ct_unique)
            ct_unique.append(ct)
    ct_idx_arr = np.array([ct_name_to_idx[ct] for ct in ct_arr])

    # Map CT names to W matrix keys
    ct_to_w_key = {}
    for ct_name, ct_idx in ct_name_to_idx.items():
        w_key = f"CT_{ct_idx}"
        if w_key in W_all:
            ct_to_w_key[ct_name] = w_key
    print(f"  {len(ct_to_w_key)}/{len(ct_name_to_idx)} cancer types have W matrices")

    # Holdback split (same seed as bilinear training)
    np.random.seed(args.seed)
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_holdback = int(N * 0.15)
    holdback_idx = all_idx[:n_holdback]
    train_idx = all_idx[n_holdback:]
    print(f"\n  Holdback: {n_holdback}, Train: {len(train_idx)}")

    # Patient gene dict indexed by row number
    patient_genes_by_row = {}
    for pid, genes in patient_genes.items():
        row = pid_to_row.get(pid)
        if row is not None:
            patient_genes_by_row[row] = genes

    # CT patient groups
    ct_patients = defaultdict(list)
    for i in train_idx:
        ct_patients[ct_arr[i]].append(i)

    # Compute per-CT node hazards from training data
    print("\nComputing per-CT node hazards...")
    ct_node_hazards = compute_node_hazards(
        patient_genes_by_row, times_arr, events_arr,
        set(train_idx), gene_vocab, ct_patients,
    )

    # Load per-CT adjacency from graph — but the PATIENT selects which edges
    # to read. Cancer type picks which weight column. Mutations pick which rows.
    print("\nLoading per-CT graph-calibrated adjacency...")
    ct_adjacency = {}
    for ct_name, w_key in ct_to_w_key.items():
        ct_idx = int(w_key.split("_")[1])
        ct_prop_key = CT_IDX_TO_PROP.get(ct_idx)
        if ct_prop_key is None:
            continue
        A, n_edges = load_graph_calibrated_adjacency(ct_prop_key, gene_vocab)
        if n_edges > 0:
            ct_adjacency[ct_name] = A
            print(f"  {ct_name[:30]:30s}: {n_edges:6d} edges")

    # Calibrate organ topology weight from training data
    # Simple: correlate topo_score with event rate on training set
    print("\nCalibrating organ topology weight...")
    train_topo = topo_scores[train_idx]
    train_events = events_arr[train_idx].astype(float)
    train_times = times_arr[train_idx]
    # Hazard proxy: event / time
    hazard_proxy = train_events / np.maximum(train_times, 1.0)
    from scipy.stats import pearsonr
    if train_topo.std() > 1e-8:
        r_topo, p_topo = pearsonr(train_topo, hazard_proxy)
        # Negative topo = spread across cavities = worse
        # If r is negative, that means low topo → high hazard → use negative weight
        topo_weight = r_topo * 0.5  # moderate scaling
        print(f"  Topo correlation: r={r_topo:.4f}, p={p_topo:.2e}")
        print(f"  Topo weight: {topo_weight:.4f}")
    else:
        topo_weight = 0.0
        print("  No topology variance, weight=0")

    # Compute per-patient channel count (the "why" feature)
    print("\nComputing per-patient channel counts...")
    channel_counts = np.zeros(N, dtype=np.float32)
    mutation_counts = np.zeros(N, dtype=np.float32)
    per_channel = np.zeros((N, len(CHANNELS)), dtype=np.float32)

    for row_idx in range(N):
        genes = patient_genes_by_row.get(row_idx, set())
        mutation_counts[row_idx] = len(genes)
        hit_channels = set()
        for gene in genes:
            ch = CHANNEL_MAP.get(gene)
            if ch:
                hit_channels.add(ch)
        channel_counts[row_idx] = len(hit_channels)
        for ch in hit_channels:
            if ch in CHANNELS:
                per_channel[row_idx, CHANNELS.index(ch)] = 1.0

    print(f"  Mean channel count: {channel_counts.mean():.2f}")
    print(f"  Mean mutation count: {mutation_counts.mean():.2f}")

    # Step 1: Score ALL patients for graph, clinical, and channel features
    print(f"\nScoring all {N} patients...")
    all_graph_inter = np.zeros(N)
    all_graph_reach = np.zeros(N)
    all_clinical = np.zeros(N)
    all_organ = np.zeros(N)
    all_bias = np.zeros(N)
    total_inter_edges = 0
    total_reach_edges = 0

    for row_idx in range(N):
        ct = ct_arr[row_idx]
        genes = patient_genes_by_row.get(row_idx, set())

        # Graph score — THIS patient's mutations, CT-specific weights
        A_ct = ct_adjacency.get(ct)
        inter, reach, n_inter, n_reach = score_patient_graph(
            genes, gene_vocab, A_ct,
        )
        all_graph_inter[row_idx] = inter
        all_graph_reach[row_idx] = reach
        total_inter_edges += n_inter
        total_reach_edges += n_reach

        # Clinical score
        if beta_clinical is not None:
            ct_idx = ct_name_to_idx.get(ct, 0)
            if ct_idx < beta_clinical.shape[0]:
                all_clinical[row_idx] = float(
                    beta_clinical[ct_idx] @ clinical_feats[row_idx]
                )

        # Organ topology
        all_organ[row_idx] = topo_weight * topo_scores[row_idx]

        # Bias
        ct_idx = ct_name_to_idx.get(ct, 0)
        all_bias[row_idx] = float(bias[ct_idx]) if ct_idx < len(bias) else 0.0

    avg_inter = total_inter_edges / max(N, 1)
    avg_reach = total_reach_edges / max(N, 1)
    print(f"  Avg inter-mutation edges per patient: {avg_inter:.1f}")
    print(f"  Avg reach edges per patient: {avg_reach:.1f}")

    # Step 2: Calibrate graph delta weight on TRAINING data
    # Clinical is Newton. Graph is the relativistic correction.
    # Fit: optimal_score = clinical + w_inter * inter + w_reach * reach
    # where w_inter, w_reach are calibrated so graph is a DELTA, not competing.
    print("\nCalibrating graph delta weights on training data...")
    from scipy.optimize import minimize

    train_clinical = all_clinical[train_idx] + all_organ[train_idx] + all_bias[train_idx]
    train_inter = all_graph_inter[train_idx]
    train_reach = all_graph_reach[train_idx]
    train_channel = channel_counts[train_idx]
    train_e = events_arr[train_idx].astype(bool)
    train_t = times_arr[train_idx]
    train_valid = train_t > 0

    # Subsample training set for fast grid search
    np.random.seed(42)
    cal_size = min(5000, len(train_idx))
    cal_mask = np.random.choice(len(train_clinical), cal_size, replace=False)
    cal_clin = train_clinical[cal_mask]
    cal_inter = train_inter[cal_mask]
    cal_reach = train_reach[cal_mask]
    cal_ch = train_channel[cal_mask]
    cal_e = train_e[cal_mask]
    cal_t = train_t[cal_mask]
    cal_valid = cal_t > 0

    # Grid search: clinical + w_ch * channel_count + w_inter * inter + w_reach * reach
    best_c = -1
    best_wch, best_wi, best_wr = 0.0, 0.0, 0.0
    for wch in np.linspace(-1.0, 1.0, 21):
        for wi in np.linspace(-0.5, 0.5, 11):
            for wr in np.linspace(-0.1, 0.1, 11):
                combined = cal_clin + wch * cal_ch + wi * cal_inter + wr * cal_reach
                try:
                    c = concordance_index_censored(
                        cal_e[cal_valid], cal_t[cal_valid], combined[cal_valid]
                    )[0]
                except Exception:
                    c = 0.0
                if c > best_c:
                    best_c = c
                    best_wch, best_wi, best_wr = wch, wi, wr

    print(f"  Optimal weights: channel={best_wch:.3f}, inter={best_wi:.3f}, reach={best_wr:.3f}")
    print(f"  Training C with all: {best_c:.4f}")

    # Component analysis on training data
    c_train_clin = concordance_index_censored(
        train_e[train_valid], train_t[train_valid], train_clinical[train_valid]
    )[0]
    # Channel only (on top of clinical)
    best_wch_only = 0.0
    best_c_ch = c_train_clin
    for wch in np.linspace(-1.0, 1.0, 41):
        combined = cal_clin + wch * cal_ch
        try:
            c = concordance_index_censored(
                cal_e[cal_valid], cal_t[cal_valid], combined[cal_valid]
            )[0]
        except Exception:
            c = 0.0
        if c > best_c_ch:
            best_c_ch = c
            best_wch_only = wch

    print(f"  Training C clinical-only:        {c_train_clin:.4f}")
    print(f"  Training C clinical+channel:     {best_c_ch:.4f}  (w_ch={best_wch_only:.3f}, delta: {best_c_ch - c_train_clin:+.4f})")
    print(f"  Training C clinical+ch+graph:    {best_c:.4f}  (delta: {best_c - c_train_clin:+.4f})")

    # Step 3: Score holdback
    print(f"\nScoring {n_holdback} holdback patients...")
    hb_clinical = all_clinical[holdback_idx] + all_organ[holdback_idx] + all_bias[holdback_idx]
    hb_inter = all_graph_inter[holdback_idx]
    hb_reach = all_graph_reach[holdback_idx]
    hb_channel = channel_counts[holdback_idx]
    scores = hb_clinical + best_wch * hb_channel + best_wi * hb_inter + best_wr * hb_reach
    inter_scores = hb_inter
    reach_scores = hb_reach

    # Evaluate
    e_hb = events_arr[holdback_idx].astype(bool)
    t_hb = times_arr[holdback_idx]
    valid = t_hb > 0

    c_global = concordance_index_censored(e_hb[valid], t_hb[valid], scores[valid])[0]

    print(f"\n{'='*70}")
    print(f"  BILINEAR WALK SCORER RESULTS")
    print(f"  Holdback C-index: {c_global:.4f}")
    print(f"{'='*70}")

    # Per-CT holdback
    print(f"\n  Per-CT holdback C-index:")
    ct_results = {}
    ct_hb = ct_arr[holdback_idx]
    for ct_name in sorted(set(ct_hb)):
        mask = ct_hb == ct_name
        e_ct = e_hb[mask]
        t_ct = t_hb[mask]
        s_ct = scores[mask]
        valid_ct = t_ct > 0
        n_pts = mask.sum()
        n_events = e_ct.sum()

        if n_pts >= 20 and n_events >= 5:
            try:
                c_ct = concordance_index_censored(
                    e_ct[valid_ct], t_ct[valid_ct], s_ct[valid_ct]
                )[0]
                print(f"    {ct_name:40s}: C={c_ct:.4f} (n={n_pts}, events={n_events})")
                ct_results[ct_name] = {
                    "c_index": float(c_ct), "n": int(n_pts), "events": int(n_events)
                }
            except Exception:
                pass

    # Component analysis: walk-only vs clinical-only vs organ-only
    print(f"\n  Component analysis:")

    # Component analysis on holdback
    c_clin_only = concordance_index_censored(e_hb[valid], t_hb[valid], hb_clinical[valid])[0]
    c_clin_ch = concordance_index_censored(
        e_hb[valid], t_hb[valid], (hb_clinical + best_wch_only * hb_channel)[valid]
    )[0]
    c_ch_alone = concordance_index_censored(e_hb[valid], t_hb[valid], hb_channel[valid])[0]
    c_inter = concordance_index_censored(e_hb[valid], t_hb[valid], inter_scores[valid])[0]
    c_reach = concordance_index_censored(e_hb[valid], t_hb[valid], reach_scores[valid])[0]

    print(f"    Clinical baseline (Newton):     C={c_clin_only:.4f}")
    print(f"    Channel count alone:            C={c_ch_alone:.4f}")
    print(f"    Clinical + channel (why):       C={c_clin_ch:.4f}  (delta: {c_clin_ch - c_clin_only:+.4f})")
    print(f"    Clinical + channel + graph:     C={c_global:.4f}  (delta: {c_global - c_clin_only:+.4f})")
    print(f"    ---")
    print(f"    Inter-mutation edges alone:     C={c_inter:.4f}")
    print(f"    Reach edges alone:              C={c_reach:.4f}")

    # Save results
    results = {
        "holdback_c_index": float(c_global),
        "clinical_baseline_c_index": float(c_clin_only),
        "clinical_plus_channel_c_index": float(c_clin_ch),
        "channel_delta": float(c_clin_ch - c_clin_only),
        "full_delta": float(c_global - c_clin_only),
        "channel_alone_c_index": float(c_ch_alone),
        "inter_mutation_c_index": float(c_inter),
        "reach_c_index": float(c_reach),
        "delta_weights": {
            "channel": float(best_wch), "inter": float(best_wi), "reach": float(best_wr)
        },
        "topo_weight": float(topo_weight),
        "n_holdback": n_holdback,
        "n_genes": G,
        "n_cancer_types_with_adjacency": len(ct_adjacency),
        "avg_inter_edges_per_patient": float(avg_inter),
        "avg_reach_edges_per_patient": float(avg_reach),
        "per_ct_holdback": ct_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    print(f"  Total time: {time.time() - t0:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Walk damping factor")
    parser.add_argument("--n-steps", type=int, default=3,
                        help="Walk propagation steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
