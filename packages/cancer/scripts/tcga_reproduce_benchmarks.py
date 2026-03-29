#!/usr/bin/env python3
"""
Reproduce benchmark methodology exactly, then compare with our features.

Runs CoxEN with:
  1. Gene expression (reproducing their results)
  2. Channel features only (our contribution)
  3. Expression + channels (combined)

All using their exact methodology:
  - 80/20 fixed split (random_state=0, stratify=vital_status)
  - 5-fold CV within train to select best model
  - Best fold selected by VAL c-index (we fix their test-leakage bug)
  - Predict on held-out 20% test set

Usage:
    python3 -u -m gnn.scripts.tcga_reproduce_benchmarks
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Optimization terminated early.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX,
    N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6, N_TIERS_V6,
    V6_TIER_MAP, V6_GENE_FUNCTION,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
import torch

# V6 mutation-level transformer imports
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, NON_SILENT, MSK_DATASETS
from gnn.data.atlas_dataset import make_node_features, MAX_NODES, NODE_FEAT_DIM
from gnn.models.atlas_transformer_v6 import AtlasTransformerV6
from gnn.scripts.train_atlas_transformer_v6 import (
    load_graph_data, compute_gene_pair_matrix,
    compute_patient_edge_features, gather_edge_features,
)
from gnn.data.block_assignments import load_block_assignments

KAGGLE_DATA = os.path.expanduser(
    "~/.cache/kagglehub/datasets/ridgiemo/processed-gene-and-clinical-data/versions/1/data")
TCGA_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "tcga_benchmark")
SAVE_BASE = os.path.join(GNN_RESULTS, "tcga_reproduce_benchmarks")

DATASETS = ["COAD", "ESCA", "HNSC", "LIHC", "LUAD", "LUSC", "STAD"]

TCGA_TO_CT_IDX = {
    "COAD": 11, "ESCA": 14, "HNSC": 19, "LIHC": 20,
    "LUAD": 24, "LUSC": 24, "STAD": 14,
}

# TCGA dataset name → MSK-IMPACT cancer type (for atlas lookup + CT embedding)
TCGA_TO_MSK_CT = {
    "COAD": "Colorectal Cancer",
    "ESCA": "Esophagogastric Cancer",
    "HNSC": "Head and Neck Cancer",
    "LIHC": "Hepatobiliary Cancer",
    "LUAD": "Non-Small Cell Lung Cancer",
    "LUSC": "Non-Small Cell Lung Cancer",
    "STAD": "Esophagogastric Cancer",
}

RESULTS_DIR_V6 = os.path.join(GNN_RESULTS, "atlas_transformer_v6")


def load_expression_and_clinical(ds):
    """Load and align expression + clinical exactly as benchmarks do."""
    expr = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "gene_expression.csv"), low_memory=False)
    clin = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "clinical.csv"))

    expr = expr.sort_values("patient_id").reset_index(drop=True)
    clin = clin.sort_values("patient_id").reset_index(drop=True)

    patient_ids = expr["patient_id"].values
    X_expr = expr.drop(columns=["patient_id"]).values.astype(float)
    X_expr = np.log2(X_expr + 1)

    times = clin["real_survival_time"].values.astype(float)
    events = clin["vital_status"].values.astype(int)

    return patient_ids, X_expr, times, events


def build_channel_matrix(ds, patient_ids, clin):
    """Build channel feature matrix aligned with patient_ids."""
    mut_path = os.path.join(TCGA_DATA, f"{ds}_mutations.csv")
    if not os.path.exists(mut_path):
        return None
    mut = pd.read_csv(mut_path)

    # Map cBioPortal barcodes to short patient_ids
    barcode_to_short = {b: b.split("-")[-1] for b in mut["patientId"].unique()}

    # Also try tissue_source_site matching
    if "tissue_source_site" in clin.columns:
        clin_copy = clin.copy()
        clin_copy["tcga_barcode"] = (
            "TCGA-" + clin_copy["tissue_source_site"].astype(str) +
            "-" + clin_copy["patient_id"].astype(str)
        )
        pid_to_barcode = dict(zip(
            clin_copy["patient_id"].astype(str), clin_copy["tcga_barcode"]
        ))
    else:
        short_to_barcode = {v: k for k, v in barcode_to_short.items()}
        pid_to_barcode = {str(k): v for k, v in short_to_barcode.items()}

    # Channel genes
    channel_genes = set(V6_CHANNEL_MAP.keys())
    valid_types = {
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
        "Nonstop_Mutation", "Translation_Start_Site",
    }

    mut = mut[
        mut["hugoGeneSymbol"].isin(channel_genes) &
        mut["mutationType"].isin(valid_types)
    ].copy()
    mut["channel"] = mut["hugoGeneSymbol"].map(V6_CHANNEL_MAP)
    mut["is_truncating"] = mut["mutationType"].isin({
        "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
        "Splice_Site", "Nonstop_Mutation",
    })
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")

    if "tumorAltCount" in mut.columns and "tumorRefCount" in mut.columns:
        alt = pd.to_numeric(mut["tumorAltCount"], errors="coerce")
        ref = pd.to_numeric(mut["tumorRefCount"], errors="coerce")
        mut["vaf"] = (alt / (alt + ref)).fillna(0.0)
    else:
        mut["vaf"] = 0.0

    mut["is_gof"] = mut["hugoGeneSymbol"].map(lambda g: V6_GENE_FUNCTION.get(g) == "GOF")
    mut["is_lof"] = mut["hugoGeneSymbol"].map(lambda g: V6_GENE_FUNCTION.get(g) == "LOF")

    grouped = mut.groupby(["patientId", "channel"]).agg(
        n_mutations=("hugoGeneSymbol", "size"),
        n_genes=("hugoGeneSymbol", "nunique"),
        frac_truncating=("is_truncating", "mean"),
        frac_missense=("is_missense", "mean"),
        mean_vaf=("vaf", "mean"),
        max_vaf=("vaf", "max"),
        gof_count=("is_gof", "sum"),
        lof_count=("is_lof", "sum"),
    ).reset_index()

    # Build per-patient feature vectors: 8 channels × 9 features = 72
    # + 4 tiers × 5 = 20 → 92 features total
    N = len(patient_ids)
    ch_array = np.zeros((N, N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6))

    for _, row in grouped.iterrows():
        barcode = row["patientId"]
        short_pid = barcode.split("-")[-1]
        # Find index in patient_ids
        matches = np.where(patient_ids.astype(str) == short_pid)[0]
        if len(matches) == 0:
            # Try with barcode
            for i, pid in enumerate(patient_ids):
                if pid_to_barcode.get(str(pid)) == barcode:
                    matches = [i]
                    break
        if len(matches) == 0:
            continue

        p_idx = matches[0]
        c_idx = V6_CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        ch_array[p_idx, c_idx] = [
            1.0,
            np.log1p(row["n_mutations"]),
            float(row["n_genes"]),
            float(row["frac_truncating"]),
            float(row["frac_missense"]),
            float(row["mean_vaf"]),
            float(row["max_vaf"]),
            float(row["gof_count"]),
            float(row["lof_count"]),
        ]

    # Tier features
    tier_array = np.zeros((N, N_TIERS_V6, 5))
    for t_idx in range(N_TIERS_V6):
        tier_channels = [V6_CHANNEL_TO_IDX[ch] for ch, t in V6_TIER_MAP.items() if t == t_idx]
        if not tier_channels:
            continue
        tier_ch = ch_array[:, tier_channels, :]
        tier_array[:, t_idx, 0] = tier_ch[:, :, 0].sum(axis=1)
        tier_array[:, t_idx, 1] = tier_ch[:, :, 1].sum(axis=1)
        tier_array[:, t_idx, 2] = tier_ch[:, :, 2].sum(axis=1)
        tier_array[:, t_idx, 3] = tier_ch[:, :, 5].max(axis=1)
        tier_array[:, t_idx, 4] = tier_ch[:, :, 6].max(axis=1)

    X_ch = np.hstack([ch_array.reshape(N, -1), tier_array.reshape(N, -1)])
    return X_ch


def run_coxen_benchmark(X, times, events, train_idx, test_idx, seed=0, scale=True):
    """Run CoxEN exactly as the benchmark does, but with correct val-based selection."""
    y = np.array([(bool(e), float(t)) for e, t in zip(events, times)],
                 dtype=[("event", bool), ("time", float)])

    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    if scale:
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[train_idx] = scaler.fit_transform(X[train_idx])
        X_scaled[test_idx] = scaler.transform(X[test_idx])
    else:
        X_scaled = X

    best_val_ci = 0
    best_test_ci = 0

    train_events = events[train_idx]

    for cv_fold, (cv_train_rel, cv_val_rel) in enumerate(
        skf.split(np.arange(len(train_idx)), train_events)
    ):
        cv_train = [train_idx[i] for i in cv_train_rel]
        cv_val = [train_idx[i] for i in cv_val_rel]

        try:
            model = CoxnetSurvivalAnalysis(
                alpha_min_ratio="auto",
                l1_ratio=0.5,
                tol=1e-5,
            )
            model.fit(X_scaled[cv_train], y[cv_train])

            # Find best alpha on val
            best_alpha_val_ci = 0
            best_alpha_test_ci = 0
            for col in range(model.coef_.shape[1]):
                pred_val = X_scaled[cv_val] @ model.coef_[:, col]
                val_ci = concordance_index_censored(
                    y[cv_val]["event"], y[cv_val]["time"], pred_val
                )[0]

                if val_ci > best_alpha_val_ci:
                    best_alpha_val_ci = val_ci
                    pred_test = X_scaled[test_idx] @ model.coef_[:, col]
                    best_alpha_test_ci = concordance_index_censored(
                        y[test_idx]["event"], y[test_idx]["time"], pred_test
                    )[0]

            if best_alpha_val_ci > best_val_ci:
                best_val_ci = best_alpha_val_ci
                best_test_ci = best_alpha_test_ci

        except Exception as e:
            pass

    return best_val_ci, best_test_ci


def run_v6c_zero_shot(ds, patient_ids, clin, times, events, test_idx):
    """V6c zero-shot on test patients."""
    from gnn.scripts.tcga_fair_comparison import build_channel_features_tcga

    data = build_channel_features_tcga(ds)
    if data is None:
        return None

    # Map test patient_ids to our data indices
    our_pids = [str(p) for p in data["patients"]]
    test_pids = set(str(patient_ids[i]) for i in test_idx)
    our_test_idx = [i for i, p in enumerate(our_pids) if p in test_pids]

    if len(our_test_idx) < 10:
        return None

    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    msk_config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
        "hidden_dim": 128,
        "cross_channel_heads": 4,
        "cross_channel_layers": 2,
        "dropout": 0.3,
        "n_cancer_types": 42,
    }

    all_hazards = []
    for fold_idx in range(5):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        model = ChannelNetV6c(msk_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        batch = {k: v[our_test_idx] for k, v in data.items()
                 if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            hazard = model(batch)
        all_hazards.append(hazard)

    if not all_hazards:
        return None
    avg = torch.stack(all_hazards).mean(dim=0)
    ci = concordance_index(avg, data["times"][our_test_idx], data["events"][our_test_idx])
    return ci


# =========================================================================
# V6 mutation-level transformer zero-shot
# =========================================================================

_v6_infra = None


def _get_v6_infrastructure():
    """Load V6 model infrastructure (atlas, graph data, etc.). Cached."""
    global _v6_infra
    if _v6_infra is not None:
        return _v6_infra

    results_path = os.path.join(RESULTS_DIR_V6, "results.json")
    if not os.path.exists(results_path):
        print("  V6: No trained model found, skipping")
        return None

    has_model = any(
        os.path.exists(os.path.join(RESULTS_DIR_V6, f"fold_{i}", "best_model.pt"))
        for i in range(5)
    )
    if not has_model:
        print("  V6: No fold checkpoints found, skipping")
        return None

    with open(results_path) as f:
        v6_results = json.load(f)

    print("\n  Loading V6 infrastructure...", flush=True)

    # Atlas from Neo4j
    from gnn.data.graph_snapshot import load_atlas
    t1, t2, t3, t4, _ = load_atlas()

    # Graph data for pairwise features
    graph_data = load_graph_data()

    # Block/channel assignments
    gene_block, n_blocks, n_channels = load_block_assignments()

    # Reconstruct MSK cancer type map (matches training iteration order)
    paths = MSK_DATASETS["msk_impact_50k"]
    clin = pd.read_csv(paths["clinical"])
    s_clin = pd.read_csv(paths["sample_clinical"])
    if 'CANCER_TYPE' in clin.columns:
        clin = clin.drop(columns=['CANCER_TYPE'])
    clin = clin.merge(s_clin[['patientId', 'CANCER_TYPE']].drop_duplicates(),
                      on='patientId', how='left')
    clin = clin.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    ct_map = {}
    idx = 0
    for ct_val in clin['CANCER_TYPE']:
        if ct_val not in ct_map:
            ct_map[ct_val] = idx
            idx += 1

    # GOF/LOF per gene
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    gof_lof = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene) WHERE g.function IS NOT NULL
            RETURN g.name AS gene, g.function AS func
        """)
        for r in result:
            f = r["func"]
            if f in ("oncogene",):
                gof_lof[r["gene"]] = 1
            elif f in ("TSG", "likely_TSG"):
                gof_lof[r["gene"]] = -1
    driver.close()

    _v6_infra = {
        'atlas': (t1, t2, t3, t4),
        'graph_data': graph_data,
        'gene_block': gene_block,
        'n_blocks': n_blocks,
        'n_channels': n_channels,
        'ct_map': ct_map,
        'config': v6_results['config'],
        'gof_lof': gof_lof,
    }
    print("  V6 infrastructure loaded", flush=True)
    return _v6_infra


def run_v6_zero_shot(ds, test_idx, patient_ids, times, events):
    """V6 mutation-level transformer, MSK-trained, zero-shot on TCGA test set.

    Encodes TCGA patients through the same atlas → mutation-level pipeline
    used in training, then runs inference with the trained V6 model ensemble.
    Uses Cox-SAGE's exact 80/20 test split for apples-to-apples comparison.
    """
    infra = _get_v6_infrastructure()
    if infra is None:
        return None

    t1, t2, t3, t4 = infra['atlas']
    graph_data = infra['graph_data']
    gene_block = infra['gene_block']
    n_blocks, n_channels = infra['n_blocks'], infra['n_channels']
    ct_map = infra['ct_map']
    model_config = infra['config']
    gof_lof = infra['gof_lof']

    # Map TCGA dataset → MSK cancer type
    msk_ct = TCGA_TO_MSK_CT.get(ds)
    if msk_ct is None or msk_ct not in ct_map:
        print(f"    V6: No CT mapping for {ds}")
        return None
    ct_idx = ct_map[msk_ct]

    # Load TCGA mutations, filter to channel genes + non-silent
    mut_path = os.path.join(TCGA_DATA, f"{ds}_mutations.csv")
    if not os.path.exists(mut_path):
        return None
    mut = pd.read_csv(mut_path)
    channel_genes = set(CHANNEL_MAP.keys())
    mut = mut[
        mut["hugoGeneSymbol"].isin(channel_genes) &
        mut["mutationType"].isin(NON_SILENT)
    ]

    # Patient ID alignment: barcode "TCGA-AA-3488" → short "3488"
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}

    # Collect atlas-matched mutations per patient
    N = len(patient_ids)
    patient_mutations = {}

    for barcode, group in mut.groupby("patientId"):
        short = barcode.split("-")[-1]
        p_idx = pid_to_idx.get(short)
        if p_idx is None:
            continue

        mutations = []
        for _, row in group.iterrows():
            gene = row["hugoGeneSymbol"]
            pc = row.get("proteinChange", "")
            if not isinstance(pc, str):
                pc = ""
            ch = CHANNEL_MAP.get(gene)

            entry = None
            if (msk_ct, gene, pc) in t1:
                entry = t1[(msk_ct, gene, pc)]
            elif (msk_ct, gene) in t2:
                entry = t2[(msk_ct, gene)]
            elif ch and (msk_ct, ch) in t3:
                entry = t3[(msk_ct, ch)]
            elif (msk_ct, gene) in t4:
                entry = t4[(msk_ct, gene)]

            if entry is not None:
                feat = make_node_features(
                    gene, pc, entry['hr'], entry.get('ci_width', 1.0),
                    entry['tier'], entry.get('n_with', 50),
                    gof_lof=gof_lof.get(gene, 0),
                )
                mutations.append((feat, gene, np.log(entry['hr'])))

        if mutations:
            patient_mutations[p_idx] = mutations

    n_with_muts = len(patient_mutations)
    if n_with_muts == 0:
        return None
    print(f"    V6: {n_with_muts}/{N} patients with atlas-matched mutations")

    # Build per-patient tensors
    node_features = np.zeros((N, MAX_NODES, NODE_FEAT_DIM), dtype=np.float32)
    node_masks = np.zeros((N, MAX_NODES), dtype=np.float32)
    gene_names_all = [[''] * MAX_NODES for _ in range(N)]
    atlas_sums = np.zeros(N, dtype=np.float32)

    for p_idx in range(N):
        muts = patient_mutations.get(p_idx)
        if muts:
            if len(muts) > MAX_NODES:
                muts.sort(key=lambda x: abs(x[2]), reverse=True)
                muts = muts[:MAX_NODES]
            for slot, (feat, gene, log_hr) in enumerate(muts):
                node_features[p_idx, slot] = feat
                node_masks[p_idx, slot] = 1.0
                gene_names_all[p_idx][slot] = gene
                atlas_sums[p_idx] += log_hr
        else:
            # WT default node
            node_masks[p_idx, 0] = 1.0
            gene_names_all[p_idx][0] = 'WT'

    # Gene vocabulary (for this TCGA dataset)
    gene_vocab = {}
    for patient_genes in gene_names_all:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)

    if not gene_vocab:
        return None

    # Gene indices
    gene_indices = np.zeros((N, MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(gene_names_all):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]

    # Gene-pair matrix (graph-level features for this gene vocab)
    gene_pair_matrix, _ = compute_gene_pair_matrix(graph_data, gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32)

    # Patient-level edge features
    patient_edge_feats, _ = compute_patient_edge_features(node_features)
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32)

    # Block/channel assignments
    ch_idx_map = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
    block_ids = np.full((N, MAX_NODES), n_blocks, dtype=np.int64)
    channel_ids_arr = np.full((N, MAX_NODES), n_channels, dtype=np.int64)

    for b, patient_genes in enumerate(gene_names_all):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT':
                info = gene_block.get(g)
                if info:
                    block_ids[b, s] = info['block_id']
                    channel_ids_arr[b, s] = info['channel_id']
                else:
                    ch = CHANNEL_MAP.get(g)
                    if ch and ch in ch_idx_map:
                        channel_ids_arr[b, s] = ch_idx_map[ch]

    # Clinical features
    clin = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "clinical.csv"))
    clin = clin.sort_values("patient_id").reset_index(drop=True)
    age_raw = pd.to_numeric(clin["age"], errors="coerce").fillna(0.5)
    ages_t = torch.tensor((age_raw.values * 90 - 60) / 15, dtype=torch.float32)
    sex_map = {"MALE": 1.0, "FEMALE": 0.0}
    sexes_t = torch.tensor(
        clin["gender"].map(sex_map).fillna(0.0).values, dtype=torch.float32
    )
    # Temporal: TCGA data ~2013, norm_year=(2013-2019.5)/3=-2.17, mature=1
    temporal_t = torch.tensor(
        np.tile([(-6.5 / 3.0), 1.0, 1.0], (N, 1)), dtype=torch.float32
    )
    clinical = torch.cat([
        ages_t.unsqueeze(-1), sexes_t.unsqueeze(-1), temporal_t,
    ], dim=-1)  # (N, 5)

    # Assemble tensors
    nf = torch.tensor(node_features, dtype=torch.float32)
    nm = torch.tensor(node_masks, dtype=torch.float32)
    ct_tensor = torch.full((N,), ct_idx, dtype=torch.long)
    atlas_sums_t = torch.tensor(atlas_sums, dtype=torch.float32).unsqueeze(1)
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)
    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    channel_ids_t = torch.tensor(channel_ids_arr, dtype=torch.long)

    # Ensemble: load each fold's best model, average predictions
    all_hazards = []
    for fold in range(5):
        model_path = os.path.join(RESULTS_DIR_V6, f"fold_{fold}", "best_model.pt")
        if not os.path.exists(model_path):
            continue

        model = AtlasTransformerV6(model_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu",
                                          weights_only=True))
        model.eval()

        test_t = torch.tensor(test_idx, dtype=torch.long)
        with torch.no_grad():
            batch_edge = gather_edge_features(
                gene_pair_matrix_t, patient_edge_feats_t[test_t],
                gene_indices_t[test_t], nm[test_t],
            )
            h = model(
                nf[test_t], nm[test_t], ct_tensor[test_t],
                clinical[test_t], atlas_sums_t[test_t],
                batch_edge, block_ids_t[test_t], channel_ids_t[test_t],
            )
        all_hazards.append(h.cpu())

    if not all_hazards:
        return None

    avg_hazard = torch.stack(all_hazards).mean(dim=0).numpy().flatten()

    # C-index on the Cox-SAGE test split
    e_test = events[test_idx].astype(bool)
    t_test = times[test_idx].astype(float)
    valid = t_test > 0
    try:
        ci = concordance_index_censored(e_test[valid], t_test[valid],
                                        avg_hazard[valid])[0]
    except Exception:
        ci = 0.5

    # --- Signal-stratified C-index ---
    # Per-patient signal strength = sum(|log_hr|) for test patients
    test_signal = np.abs(atlas_sums[test_idx])
    n_muts_test = node_masks[test_idx].sum(axis=1)

    # Bin by signal strength: low (bottom third), mid, high (top third)
    strat = {}
    nonzero_signal = test_signal[test_signal > 0]
    if len(nonzero_signal) >= 15:
        t_lo = np.percentile(nonzero_signal, 33)
        t_hi = np.percentile(nonzero_signal, 67)

        bins = {
            'zero': test_signal == 0,
            'low': (test_signal > 0) & (test_signal <= t_lo),
            'mid': (test_signal > t_lo) & (test_signal <= t_hi),
            'high': test_signal > t_hi,
        }

        for bin_name, mask in bins.items():
            n_bin = mask.sum()
            if n_bin < 10:
                continue
            e_bin = e_test[mask]
            t_bin = t_test[mask]
            h_bin = avg_hazard[mask]
            v_bin = t_bin > 0
            if e_bin[v_bin].sum() < 3:
                continue
            try:
                c_bin = concordance_index_censored(
                    e_bin[v_bin], t_bin[v_bin], h_bin[v_bin]
                )[0]
                strat[bin_name] = {
                    'c_index': float(c_bin), 'n': int(n_bin),
                    'n_events': int(e_bin.sum()),
                    'mean_signal': float(test_signal[mask].mean()),
                    'mean_n_muts': float(n_muts_test[mask].mean()),
                }
            except Exception:
                pass

    if strat:
        print(f"    V6 signal-stratified (test set):")
        for bin_name in ['zero', 'low', 'mid', 'high']:
            if bin_name in strat:
                s = strat[bin_name]
                c = s['c_index']
                disc = abs(c - 0.5)
                direction = "+" if c >= 0.5 else "-"
                print(f"      {bin_name:>5}: C={c:.4f} |C-0.5|={disc:.4f}{direction}  "
                      f"n={s['n']:>3}  events={s['n_events']:>2}  "
                      f"signal={s['mean_signal']:.2f}  "
                      f"muts={s['mean_n_muts']:.1f}")

    # Weighted average |C - 0.5| across bins (doesn't cancel like raw C)
    if strat:
        total_n = sum(s['n'] for s in strat.values())
        weighted_disc = sum(abs(s['c_index'] - 0.5) * s['n']
                           for s in strat.values()) / max(total_n, 1)
    else:
        weighted_disc = abs(ci - 0.5)

    return {'c_index': ci, 'disc_power': weighted_disc,
            'stratified': strat, 'n_with_muts': n_with_muts, 'n_total': N}


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 110)
    print("  REPRODUCE BENCHMARK METHODOLOGY + COMPARE")
    print("  Same split, same CoxEN, same alpha selection — different features")
    print("=" * 110)

    all_results = []

    for ds in DATASETS:
        print(f"\n{'='*90}")
        print(f"  {ds}")
        print(f"{'='*90}")

        # Load data
        patient_ids, X_expr, times, events = load_expression_and_clinical(ds)
        N = len(patient_ids)
        clin = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "clinical.csv"))
        clin = clin.sort_values("patient_id").reset_index(drop=True)

        # Fixed 80/20 split (same as benchmarks)
        indices = list(range(N))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=0, stratify=events
        )
        n_test_events = events[test_idx].sum()
        print(f"  {N} patients, {len(train_idx)} train / {len(test_idx)} test "
              f"({n_test_events} test events)")

        if n_test_events < 3:
            print(f"  Skipped (too few test events)")
            continue

        # Build channel features
        X_ch = build_channel_matrix(ds, patient_ids, clin)
        has_channels = X_ch is not None
        n_ch_patients = int((X_ch[:, 0] > 0).sum()) if has_channels else 0
        if has_channels:
            print(f"  {n_ch_patients}/{N} patients have channel mutations")

        result = {"dataset": ds, "n_patients": N, "n_test": len(test_idx),
                  "n_test_events": int(n_test_events)}

        # Run across 5 seeds (matching benchmark protocol)
        for feature_name, X, do_scale in [
            ("CoxEN_expr", X_expr, False),  # benchmarks don't scale
            ("CoxEN_ch", X_ch, True),
            ("CoxEN_expr+ch", np.hstack([X_expr, X_ch]) if has_channels else None, False),
        ]:
            if X is None:
                continue

            seed_cis = []
            for seed in range(5):
                val_ci, test_ci = run_coxen_benchmark(
                    X, times, events, train_idx, test_idx, seed=seed, scale=do_scale
                )
                seed_cis.append(test_ci)

            mean_ci = np.mean(seed_cis)
            std_ci = np.std(seed_cis)
            best_ci = np.max(seed_cis)
            print(f"  {feature_name:<16} mean={mean_ci:.4f} ± {std_ci:.4f}  "
                  f"best={best_ci:.4f}  seeds={[f'{c:.3f}' for c in seed_cis]}")
            result[f"{feature_name}_mean"] = mean_ci
            result[f"{feature_name}_std"] = std_ci
            result[f"{feature_name}_best"] = best_ci
            result[f"{feature_name}_seeds"] = seed_cis

        # V6c zero-shot (channel-level)
        v6c_ci = run_v6c_zero_shot(ds, patient_ids, clin, times, events, test_idx)
        if v6c_ci is not None:
            print(f"  {'V6c_zero_shot':<16} {v6c_ci:.4f} (MSK-trained, never saw TCGA)")
            result["V6c_zero_shot"] = v6c_ci

        # V6 zero-shot (mutation-level transformer)
        v6_result = run_v6_zero_shot(ds, test_idx, patient_ids, times, events)
        if v6_result is not None:
            v6_ci = v6_result['c_index']
            v6_disc = v6_result['disc_power']
            print(f"  {'V6_zero_shot':<16} C={v6_ci:.4f}  |C-0.5|={v6_disc:.4f}  "
                  f"(MSK-trained, mutation-level)")
            result["V6_zero_shot"] = v6_ci
            result["V6_disc_power"] = v6_disc
            result["V6_stratified"] = v6_result['stratified']
            result["V6_n_with_muts"] = v6_result['n_with_muts']

        all_results.append(result)

    # Summary table
    print(f"\n\n{'='*110}")
    print(f"  RESULTS: SAME METHODOLOGY, DIFFERENT FEATURES")
    print(f"{'='*110}")
    print(f"  {'DS':<6} {'N':>4} {'CoxEN(expr)':>12} {'CoxEN(ch)':>12} "
          f"{'CoxEN(e+c)':>12} {'V6c-ZS':>10} {'V6-ZS':>10}   notes")
    print(f"  {'-'*6} {'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}   {'-'*20}")

    for r in all_results:
        def fmt(key):
            v = r.get(key)
            return f"{v:.4f}" if v is not None else "    ---"

        # Compare to expression baseline
        expr_ci = r.get("CoxEN_expr_mean", 0)
        v6_ci = r.get("V6_zero_shot", 0)
        v6c_ci = r.get("V6c_zero_shot", 0)
        notes = []
        if v6_ci and expr_ci and v6_ci > expr_ci:
            notes.append("V6 > expr!")
        if v6c_ci and expr_ci and v6c_ci > expr_ci:
            notes.append("V6c > expr!")

        print(f"  {r['dataset']:<6} {r['n_patients']:>4} "
              f"{fmt('CoxEN_expr_mean'):>12} {fmt('CoxEN_ch_mean'):>12} "
              f"{fmt('CoxEN_expr+ch_mean'):>12} {fmt('V6c_zero_shot'):>10}"
              f" {fmt('V6_zero_shot'):>10}"
              f"   {', '.join(notes)}")

    print(f"  {'-'*6} {'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  CoxEN(expr)  = Gene expression features (benchmarks use this)")
    print(f"  CoxEN(ch)    = Channel features only (92 features)")
    print(f"  CoxEN(e+c)   = Expression + channels combined")
    print(f"  V6c-ZS       = MSK-trained V6c channel-level, zero-shot")
    print(f"  V6-ZS        = MSK-trained V6 mutation-level transformer, zero-shot")
    print(f"  All use Cox-SAGE's exact 80/20 split (random_state=0)")
    print(f"  NOTE: Cox-SAGE early-stops on test C-index (line 254), inflating results")
    print(f"{'='*110}")

    # Signal-stratified summary
    has_strat = any(r.get("V6_stratified") for r in all_results)
    if has_strat:
        print(f"\n{'='*90}")
        print(f"  V6 DISCRIMINATIVE POWER BY SIGNAL STRENGTH  |C - 0.5|")
        print(f"  0.0 = noise, >0 = signal (anti-correlated C<0.5 is also signal, just flip)")
        print(f"{'='*90}")
        print(f"  {'DS':<6} {'zero':>12} {'low':>12} {'mid':>12} {'high':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for r in all_results:
            s = r.get("V6_stratified", {})
            if not s:
                continue
            vals = {}
            for b in ['zero', 'low', 'mid', 'high']:
                if b in s:
                    c = s[b]['c_index']
                    disc = abs(c - 0.5)
                    sign = "+" if c >= 0.5 else "-"
                    vals[b] = f"{disc:.3f}{sign} (n={s[b]['n']})"
                else:
                    vals[b] = "     ---    "
            print(f"  {r['dataset']:<6} {vals['zero']:>12} {vals['low']:>12} "
                  f"{vals['mid']:>12} {vals['high']:>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  + = concordant (C>0.5), - = anti-concordant (C<0.5, flip sign to use)")
        print(f"  Expect: zero ≈ 0.000, high >> 0.000 (sigmoid curve)")
        print(f"{'='*90}")

    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
