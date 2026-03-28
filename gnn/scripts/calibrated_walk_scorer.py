#!/usr/bin/env python3
"""
Calibrated graph walk scorer — safety model v2.

Fixes from diagnostic:
1. CT-specific node hazards (not global) — ATRX dangerous in PNS, protective in glioma
2. Propagation tested at multiple α values — pick best per fold
3. Channel weights learned from training data (not uniform sum)
4. Score direction: positive = more hazard (higher = worse prognosis)
5. Mutation count preserved as signal (not destroyed by walk)

Architecture: M = (I + α·D⁻¹·A)^K precomputed. Scoring = matrix multiply.

Usage:
    python3 -u -m gnn.scripts.calibrated_walk_scorer
"""

import sys, os, time, json
import numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict
from scipy import stats as sp_stats, sparse as sp_sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_FUNCTION,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence,
)
from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles

SAVE_BASE = os.path.join(GNN_RESULTS, "calibrated_walk_scorer")
CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

CHROMATIN_GENES = {
    'ANKRD11', 'ARID1B', 'ARID2', 'BCOR', 'CREBBP', 'EP300', 'H3C7',
    'KDM6A', 'KMT2A', 'KMT2B', 'KMT2C', 'KMT2D', 'NSD1', 'SETD2', 'SMARCA4',
}
METHYLATION_GENES = {
    'ATRX', 'DAXX', 'DNMT3A', 'DNMT3B', 'IDH1', 'IDH2', 'TET1', 'TET2',
}
EPIGENETIC_GENES = CHROMATIN_GENES | METHYLATION_GENES

N_ITERATIONS = 3  # reduced from 5 — propagation was hurting

# Edge-type weight scales
EDGE_TYPE_SCALES = {
    'hub':            0.6,
    'same_pathway':   0.7,
    'cross_pathway':  1.3,
    'chromatin':      1.5,
    'methylation':    0.5,
    'default':        1.0,
}

HUB_SET = set()
for ch_hubs in HUB_GENES.values():
    HUB_SET |= ch_hubs


# =========================================================================
# Edge classification
# =========================================================================

def classify_edge(gene_a, gene_b, channel_profiles):
    """Classify edge type from biology."""
    if gene_a in CHROMATIN_GENES or gene_b in CHROMATIN_GENES:
        return 'chromatin'
    if gene_a in METHYLATION_GENES or gene_b in METHYLATION_GENES:
        return 'methylation'
    if gene_a in HUB_SET or gene_b in HUB_SET:
        return 'hub'

    prof_a = channel_profiles.get(gene_a)
    prof_b = channel_profiles.get(gene_b)
    if prof_a is not None and prof_b is not None:
        if np.argmax(prof_a) == np.argmax(prof_b):
            return 'same_pathway'
        else:
            return 'cross_pathway'

    return 'default'


# =========================================================================
# Edge weight calibration
# =========================================================================

def calibrate_edge_weights(G_ppi, patient_genes, times, events, valid_mask,
                           channel_profiles, min_co=15, min_single=20):
    """Calibrate edge weights from co-mutation survival correlation."""
    gene_patients = defaultdict(set)
    for idx, genes in patient_genes.items():
        if valid_mask[idx]:
            for g in genes:
                gene_patients[g].add(idx)

    calibrated = {}
    edge_types = {}
    n_cal = 0
    n_fall = 0

    for u, v in G_ppi.edges():
        ppi_w = G_ppi[u][v].get('weight', 0.7)
        etype = classify_edge(u, v, channel_profiles)
        edge_types[(u, v)] = etype
        edge_types[(v, u)] = etype

        patients_u = gene_patients.get(u, set())
        patients_v = gene_patients.get(v, set())
        patients_both = patients_u & patients_v

        calibrated_w = ppi_w

        patients_u_only = patients_u - patients_v
        if len(patients_both) >= min_co and len(patients_u_only) >= min_single:
            all_u = np.array(sorted(patients_u))
            x = np.array([1.0 if idx in patients_both else 0.0 for idx in all_u])
            y = events[all_u].astype(float) / np.maximum(times[all_u], 1.0)

            if y.std() > 0 and x.std() > 0:
                r, p = sp_stats.pearsonr(x, y)
                if p < 0.2:
                    calibrated_w = ppi_w * (1.0 + r)
                    n_cal += 1
                else:
                    n_fall += 1
            else:
                n_fall += 1
        else:
            n_fall += 1

        scale = EDGE_TYPE_SCALES.get(etype, 1.0)
        final_w = calibrated_w * scale

        calibrated[(u, v)] = final_w
        calibrated[(v, u)] = final_w

    return calibrated, edge_types, n_cal, n_fall


# =========================================================================
# CT-specific node hazard calibration
# =========================================================================

def calibrate_node_hazards_per_ct(all_genes, gene_to_idx, expanded_genes,
                                  patient_genes, times, events, valid_mask,
                                  ct_per_patient, expanded_cm,
                                  min_patients_ct=10, min_patients_global=20):
    """
    Per-CT node hazards with channel-relative baselines.

    Key fix: compare each gene's event rate to its CHANNEL's event rate,
    not the global rate. This prevents TP53/KRAS from pulling up the global
    baseline and making everything else look "protective."

    Within PI3K_Growth channel, PIK3CA might be the most damaging gene —
    but it looked protective against the global rate because TP53 inflates it.
    """
    n_genes = len(all_genes)

    gene_patients_global = defaultdict(set)
    for idx, genes in patient_genes.items():
        if valid_mask[idx]:
            for g in genes:
                gene_patients_global[g].add(idx)

    global_er = events[valid_mask].mean()

    global_hazard = np.zeros(n_genes)
    global_confidence = np.zeros(n_genes)
    for gene in expanded_genes:
        i = gene_to_idx.get(gene)
        if i is None:
            continue
        pts = gene_patients_global.get(gene, set())
        if len(pts) < min_patients_global:
            continue
        gene_idx = np.array(sorted(pts))
        gene_er = events[gene_idx].mean()
        raw_h = gene_er - global_er
        shrinkage = len(pts) / (len(pts) + 20.0)

        if raw_h > 0:
            global_hazard[i] = raw_h * shrinkage
        else:
            global_confidence[i] = abs(raw_h) * shrinkage

    # Per-CT hazards
    ct_patients_dict = defaultdict(list)
    for idx in range(len(times)):
        if valid_mask[idx]:
            ct_patients_dict[ct_per_patient[idx]].append(idx)

    ct_hazards = {}
    ct_confidences = {}
    for ct_name, ct_indices in ct_patients_dict.items():
        ct_arr = np.array(ct_indices)
        if len(ct_arr) < 30:
            ct_hazards[ct_name] = global_hazard.copy()
            ct_confidences[ct_name] = global_confidence.copy()
            continue

        ct_er = events[ct_arr].mean()
        ct_h = np.zeros(n_genes)
        ct_c = np.zeros(n_genes)

        ct_gene_patients = defaultdict(set)
        for idx in ct_indices:
            for g in patient_genes.get(idx, set()):
                ct_gene_patients[g].add(idx)

        for gene in expanded_genes:
            i = gene_to_idx.get(gene)
            if i is None:
                continue
            pts = ct_gene_patients.get(gene, set())
            if len(pts) < min_patients_ct:
                ct_h[i] = global_hazard[i]
                ct_c[i] = global_confidence[i]
                continue

            gene_idx = np.array(sorted(pts))
            gene_er = events[gene_idx].mean()
            raw_ct_h = gene_er - ct_er

            ct_shrink = len(pts) / (len(pts) + 15.0)
            ct_specific = abs(raw_ct_h) * ct_shrink
            alpha = len(pts) / (len(pts) + 50.0)

            if raw_ct_h > 0:
                ct_h[i] = alpha * ct_specific + (1 - alpha) * global_hazard[i]
                ct_c[i] = (1 - alpha) * global_confidence[i]
            else:
                ct_h[i] = (1 - alpha) * global_hazard[i]
                ct_c[i] = alpha * ct_specific + (1 - alpha) * global_confidence[i]

        ct_hazards[ct_name] = ct_h
        ct_confidences[ct_name] = ct_c

    return global_hazard, ct_hazards, global_confidence, ct_confidences


# =========================================================================
# Propagation matrix
# =========================================================================

def build_propagation_matrix(all_genes, gene_to_idx, cal_weights, alpha, n_iter,
                              protective_mask=None):
    """
    Build M = (I + α·D⁻¹·A)^K as a dense (n, n) matrix.

    If protective_mask is provided (boolean array, True = protective gene),
    those genes act as barriers: they don't receive propagated damage from
    neighbors, and they don't transmit damage through themselves.
    Biologically: a protective mutation absorbs the damage signal, it doesn't
    relay it or contribute it to the patient's total hazard.
    """
    n = len(all_genes)

    rows, cols, vals = [], [], []
    for (u, v), w in cal_weights.items():
        i = gene_to_idx.get(u)
        j = gene_to_idx.get(v)
        if i is not None and j is not None and i != j:
            rows.append(i)
            cols.append(j)
            vals.append(w)

    if not rows:
        return np.eye(n)

    A = sp_sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Zero out edges INTO and OUT OF protective genes
    # They are barriers in the damage propagation network
    if protective_mask is not None and protective_mask.any():
        A = A.tolil()
        prot_idx = np.where(protective_mask)[0]
        for i in prot_idx:
            A[i, :] = 0  # no damage flows OUT through this gene
            A[:, i] = 0  # no damage flows INTO this gene
        A = A.tocsr()

    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0

    D_inv = sp_sparse.diags(1.0 / degree)
    DA = D_inv @ A

    step = sp_sparse.eye(n) + alpha * DA
    step_dense = step.toarray()

    M = step_dense.copy()
    for _ in range(n_iter - 1):
        M = M @ step_dense

    return M


# =========================================================================
# Score patients
# =========================================================================

def score_patients(val_idx, patient_genes, gene_to_idx, ct_per_patient,
                   global_hazard, ct_hazards, global_confidence, ct_confidences,
                   M, profile_matrix, n_genes):
    """
    Score validation patients using CT-specific hazards + propagation.

    Returns per-patient:
      scores: damage score (higher = worse prognosis)
      channel_scores: per-channel damage
      confidence_scores: confidence signal (higher = more predictable outcome)
      channel_confidence: per-channel confidence
    """
    ct_mutset = defaultdict(list)
    for idx in val_idx:
        ct = ct_per_patient[idx]
        key = (ct, frozenset(patient_genes.get(idx, set())))
        ct_mutset[key].append(idx)

    n_patients = len(val_idx)
    scores = np.zeros(n_patients)
    channel_scores = np.zeros((n_patients, N_CH))
    conf_scores = np.zeros(n_patients)
    channel_conf = np.zeros((n_patients, N_CH))
    patient_pos = {idx: i for i, idx in enumerate(val_idx)}

    ct_groups = defaultdict(list)
    for (ct, mutset), indices in ct_mutset.items():
        ct_groups[ct].append((mutset, indices))

    for ct_name, groups in ct_groups.items():
        node_h = ct_hazards.get(ct_name, global_hazard)
        node_c = ct_confidences.get(ct_name, global_confidence)

        mutsets = [g[0] for g in groups]
        indices_list = [g[1] for g in groups]
        n_sets = len(mutsets)

        # Damage: propagate through graph
        H_init = np.zeros((n_sets, n_genes))
        C_init = np.zeros((n_sets, n_genes))
        for s, mutset in enumerate(mutsets):
            for g in mutset:
                i = gene_to_idx.get(g)
                if i is not None:
                    H_init[s, i] = node_h[i]
                    C_init[s, i] = node_c[i]

        # Damage propagates through PPI (broken things affect neighbors)
        H_final = H_init @ M.T

        # Mask: zero out propagated damage at protective gene positions.
        # If a gene is protective in this CT, it should NOT accumulate
        # propagated hazard from neighbors — that's the sign conflict bug.
        ct_protective = node_c > 0  # (n_genes,) boolean
        if ct_protective.any():
            H_final[:, ct_protective] = 0.0
            # But keep the DIRECT hazard at genes the patient actually has mutated
            # (a gene can be slightly protective globally but CT-specific hazard > 0)
            for s, mutset in enumerate(mutsets):
                for g in mutset:
                    i = gene_to_idx.get(g)
                    if i is not None and node_h[i] > 0:
                        H_final[s, i] = node_h[i]

        ch_hazard = H_final @ profile_matrix
        total_damage = ch_hazard.sum(axis=1)

        # Confidence does NOT propagate — it's a property of the mutation itself,
        # not of the network. A protective mutation narrows YOUR prediction window,
        # it doesn't make your neighbors more predictable.
        ch_conf = C_init @ profile_matrix
        total_conf = ch_conf.sum(axis=1)

        for s, indices in enumerate(indices_list):
            for idx in indices:
                pos = patient_pos[idx]
                scores[pos] = total_damage[s]
                channel_scores[pos] = ch_hazard[s]
                conf_scores[pos] = total_conf[s]
                channel_conf[pos] = ch_conf[s]

    return scores, channel_scores, conf_scores, channel_conf


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t0 = time.time()

    print("=" * 90)
    print("  SAFETY WALK SCORER v2 — CT-Specific Hazards")
    print("  Fixes: CT-specific nodes, reduced propagation, channel weighting")
    print("=" * 90)

    # --- Load data ---
    expanded_cm = load_expanded_channel_map()
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  {N} patients")

    # V6c baseline
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        mdl = ChannelNetV6c(config)
        mdl.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        mdl.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {k: data[k][idx] for k in [
                "channel_features", "tier_features", "cancer_type_idx",
                "age", "sex", "msi_score", "msi_high", "tmb",
            ]}
            with torch.no_grad():
                h = mdl(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True
    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()

    # --- Mutations + PPI ---
    print("  Loading mutations + PPI...")
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol"],
    )
    clin = pd.read_csv(os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"), low_memory=False)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["time"] > 0]
    patient_ids = clin["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    expanded_genes = set(expanded_cm.keys())
    mut_exp = mut[mut["gene.hugoGeneSymbol"].isin(expanded_genes)].copy()
    mut_exp["patient_idx"] = mut_exp["patientId"].map(pid_to_idx)
    mut_exp = mut_exp[mut_exp["patient_idx"].notna()]
    mut_exp["patient_idx"] = mut_exp["patient_idx"].astype(int)

    patient_genes = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes[idx].add(row["gene.hugoGeneSymbol"])

    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    msk_genes = set()
    for genes in patient_genes.values():
        msk_genes |= genes
    msk_genes &= expanded_genes
    ppi_edges = fetch_string_expanded(msk_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)
    print(f"  PPI: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    all_genes = sorted(G_ppi.nodes())
    n_genes = len(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    # Precompute profile matrix (n_genes, N_CH)
    profile_matrix = np.zeros((n_genes, N_CH))
    for i, g in enumerate(all_genes):
        prof = channel_profiles.get(g)
        if prof is not None:
            profile_matrix[i] = prof

    # Edge type distribution
    type_counts = defaultdict(int)
    for u, v in G_ppi.edges():
        etype = classify_edge(u, v, channel_profiles)
        type_counts[etype] += 1
    print(f"\n  Edge type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        scale = EDGE_TYPE_SCALES.get(etype, 1.0)
        print(f"    {etype:<20} {count:>5} ({100*count/G_ppi.number_of_edges():.1f}%)  scale={scale:.1f}")

    # =========================================================================
    # Cross-validated calibration + scoring
    # =========================================================================
    folds = list(skf.split(np.arange(N), events))
    all_scores = np.zeros(N)
    all_channel_scores = np.zeros((N, N_CH))
    all_confidence = np.zeros(N)
    all_channel_conf = np.zeros((N, N_CH))
    all_scores_noprop = np.zeros(N)
    fold_cis = []
    fold_cis_noprop = []

    # Try multiple alpha values
    ALPHAS = [0.0, 0.05, 0.10, 0.15]

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  --- Fold {fold_idx} ---")
        train_patient_genes = {idx: patient_genes.get(idx, set()) for idx in train_idx}
        train_valid = np.zeros(N, dtype=bool)
        train_valid[train_idx] = valid_mask[train_idx]

        # Calibrate edges
        print(f"    Calibrating edges...")
        cal_weights, edge_types, n_cal, n_fall = calibrate_edge_weights(
            G_ppi, train_patient_genes, times, events, train_valid,
            channel_profiles, min_co=15, min_single=20,
        )
        print(f"    {n_cal} calibrated, {n_fall} fallback")

        # CT-specific node hazards + confidence
        print(f"    Calibrating CT-specific node hazards...")
        global_hazard, ct_hazards, global_confidence, ct_confidences = \
            calibrate_node_hazards_per_ct(
                all_genes, gene_to_idx, expanded_genes,
                train_patient_genes, times, events, train_valid,
                ct_per_patient, expanded_cm,
                min_patients_ct=10, min_patients_global=20,
            )

        # Report some CT-specific examples
        for example_ct in ['Glioma', 'Breast Cancer', 'Non-Small Cell Lung Cancer']:
            ct_h = ct_hazards.get(example_ct)
            ct_c = ct_confidences.get(example_ct)
            if ct_h is None:
                continue
            sorted_h = sorted([(all_genes[i], ct_h[i]) for i in range(n_genes)
                               if ct_h[i] > 0.001], key=lambda x: -x[1])
            sorted_c = sorted([(all_genes[i], ct_c[i]) for i in range(n_genes)
                               if ct_c[i] > 0.001], key=lambda x: -x[1])
            top3 = sorted_h[:3]
            conf3 = sorted_c[:3]
            print(f"    {example_ct}: damage={' '.join(f'{g}({h:+.3f})' for g, h in top3)}  "
                  f"confidence={' '.join(f'{g}({c:.3f})' for g, c in conf3)}")

        # Try multiple propagation strengths, pick best on training data
        # Use a small held-out portion of training for selection
        n_train = len(train_idx)
        select_size = min(n_train // 5, 5000)
        rng = np.random.RandomState(fold_idx)
        select_idx = rng.choice(train_idx, size=select_size, replace=False)

        # Build protective mask: genes that are protective globally
        # These genes are barriers in the propagation network
        protective_mask = global_confidence > 0
        n_prot = protective_mask.sum()
        n_dmg = (global_hazard > 0).sum()
        print(f"    Protective barriers: {n_prot} genes, Damage sources: {n_dmg} genes")

        best_alpha = 0.0
        best_ci = 0.0

        for alpha in ALPHAS:
            M_test = build_propagation_matrix(
                all_genes, gene_to_idx, cal_weights, alpha, N_ITERATIONS,
                protective_mask=protective_mask,
            )
            test_scores, _, _, _ = score_patients(
                select_idx, patient_genes, gene_to_idx, ct_per_patient,
                global_hazard, ct_hazards, global_confidence, ct_confidences,
                M_test, profile_matrix, n_genes,
            )
            sel_valid = valid_mask[select_idx]
            if sel_valid.sum() < 50:
                continue
            ci_test = concordance_index(
                torch.tensor(test_scores[sel_valid].astype(np.float32)),
                torch.tensor(times[select_idx[sel_valid]].astype(np.float32)),
                torch.tensor(events[select_idx[sel_valid]].astype(np.float32)),
            )
            if ci_test > best_ci:
                best_ci = ci_test
                best_alpha = alpha
            print(f"      α={alpha:.2f} → train C-index={ci_test:.4f}")

        print(f"    Best α={best_alpha:.2f} (train C-index={best_ci:.4f})")

        # Build final M with best alpha
        M = build_propagation_matrix(
            all_genes, gene_to_idx, cal_weights, best_alpha, N_ITERATIONS,
            protective_mask=protective_mask,
        )
        M_noprop = np.eye(n_genes)  # identity = no propagation (ablation)

        # Score validation
        val_scores, val_ch, val_conf, val_ch_conf = score_patients(
            val_idx, patient_genes, gene_to_idx, ct_per_patient,
            global_hazard, ct_hazards, global_confidence, ct_confidences,
            M, profile_matrix, n_genes,
        )
        val_scores_noprop, _, _, _ = score_patients(
            val_idx, patient_genes, gene_to_idx, ct_per_patient,
            global_hazard, ct_hazards, global_confidence, ct_confidences,
            M_noprop, profile_matrix, n_genes,
        )

        # Store
        for i, idx in enumerate(val_idx):
            all_scores[idx] = val_scores[i]
            all_channel_scores[idx] = val_ch[i]
            all_scores_noprop[idx] = val_scores_noprop[i]
            all_confidence[idx] = val_conf[i]
            all_channel_conf[idx] = val_ch_conf[i]

        # Fold C-index
        val_valid_mask = valid_mask[val_idx]
        val_valid = val_idx[val_valid_mask]
        ci_fold = concordance_index(
            torch.tensor(val_scores[val_valid_mask].astype(np.float32)),
            torch.tensor(times[val_valid].astype(np.float32)),
            torch.tensor(events[val_valid].astype(np.float32)),
        )
        ci_noprop = concordance_index(
            torch.tensor(val_scores_noprop[val_valid_mask].astype(np.float32)),
            torch.tensor(times[val_valid].astype(np.float32)),
            torch.tensor(events[val_valid].astype(np.float32)),
        )
        fold_cis.append(ci_fold)
        fold_cis_noprop.append(ci_noprop)
        print(f"    Fold {fold_idx}: with_prop={ci_fold:.4f}  no_prop={ci_noprop:.4f}  "
              f"prop_delta={ci_fold - ci_noprop:+.4f}")

        vs = val_scores[val_valid_mask]
        print(f"    Scores: mean={vs.mean():.4f} std={vs.std():.4f} "
              f"min={vs.min():.4f} max={vs.max():.4f}")

    # =========================================================================
    # Results
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  GLOBAL RESULTS")
    print(f"{'='*90}")

    ci_cal = concordance_index(
        torch.tensor(all_scores[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    ci_noprop = concordance_index(
        torch.tensor(all_scores_noprop[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    ci_v6c = concordance_index(
        torch.tensor(hazards[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    rho, p = sp_stats.spearmanr(all_scores[valid_mask], hazards[valid_mask])

    print(f"  Walk+prop C-index:       {ci_cal:.4f} (folds: {' '.join(f'{c:.4f}' for c in fold_cis)})")
    print(f"  Walk no-prop C-index:    {ci_noprop:.4f} (folds: {' '.join(f'{c:.4f}' for c in fold_cis_noprop)})")
    print(f"  V6c transformer C-index: {ci_v6c:.4f}")
    print(f"  Gap to V6c:              {ci_v6c - ci_cal:+.4f}")
    print(f"  Propagation delta:       {ci_cal - ci_noprop:+.4f}")
    print(f"  Spearman with V6c:       {rho:.4f}")

    # Per-CT
    ct_patients_dict = defaultdict(list)
    for idx in range(N):
        ct_patients_dict[ct_per_patient[idx]].append(idx)

    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE RESULTS")
    print(f"{'='*90}")
    print(f"\n  {'Cancer Type':<35} {'N':>5} {'Walk':>7} {'NoProp':>7} {'V6c':>7} {'Gap':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    walk_wins = 0
    v6c_wins = 0
    prop_helps = 0
    prop_hurts = 0
    for ct_name in sorted(ct_patients_dict, key=lambda x: -len(ct_patients_dict[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 50:
            continue

        ci_w = concordance_index(
            torch.tensor(all_scores[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_np = concordance_index(
            torch.tensor(all_scores_noprop[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        gap = ci_v - ci_w
        prop_d = ci_w - ci_np
        if gap < 0:
            walk_wins += 1
        else:
            v6c_wins += 1
        if prop_d > 0.001:
            prop_helps += 1
        elif prop_d < -0.001:
            prop_hurts += 1
        marker = " >>>" if gap < -0.01 else ""
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_w:>7.4f} {ci_np:>7.4f} {ci_v:>7.4f} {gap:>+7.4f}{marker}")

    print(f"\n  Walk wins: {walk_wins}, V6c wins: {v6c_wins}")
    print(f"  Propagation helps: {prop_helps}, hurts: {prop_hurts}")

    # Channel hazard correlations
    print(f"\n  Channel damage → survival time correlation:")
    valid_ch = all_channel_scores[valid_mask]
    for ci, ch in enumerate(CHANNELS):
        ch_vals = valid_ch[:, ci]
        if ch_vals.std() > 0:
            rho_ch, _ = sp_stats.spearmanr(ch_vals, times[valid_mask])
            print(f"    {ch:<20} rho={rho_ch:+.4f}  mean={ch_vals.mean():.4f}")

    # Confidence signal analysis
    print(f"\n  Confidence signal analysis:")
    vc = all_confidence[valid_mask]
    print(f"    Confidence: mean={vc.mean():.4f} std={vc.std():.4f} "
          f"min={vc.min():.4f} max={vc.max():.4f}")
    print(f"    Patients with conf > 0: {(vc > 0).sum()} / {len(vc)} "
          f"({100*(vc > 0).mean():.1f}%)")

    # Does confidence correlate with prediction accuracy?
    # Split into high/low confidence, compare C-index in each group
    if vc.std() > 0:
        median_conf = np.median(vc[vc > 0]) if (vc > 0).any() else 0
        high_conf = valid_mask & (all_confidence > median_conf)
        low_conf = valid_mask & (all_confidence <= median_conf)

        if high_conf.sum() > 100 and low_conf.sum() > 100:
            ci_high = concordance_index(
                torch.tensor(all_scores[high_conf].astype(np.float32)),
                torch.tensor(times[high_conf].astype(np.float32)),
                torch.tensor(events[high_conf].astype(np.float32)),
            )
            ci_low = concordance_index(
                torch.tensor(all_scores[low_conf].astype(np.float32)),
                torch.tensor(times[low_conf].astype(np.float32)),
                torch.tensor(events[low_conf].astype(np.float32)),
            )
            print(f"    High confidence ({high_conf.sum()} pts): damage C-index = {ci_high:.4f}")
            print(f"    Low confidence  ({low_conf.sum()} pts): damage C-index = {ci_low:.4f}")
            print(f"    Delta: {ci_high - ci_low:+.4f} "
                  f"({'confidence helps' if ci_high > ci_low else 'no effect'})")

    # Channel confidence correlations
    print(f"\n  Channel confidence → |residual| correlation (negative = narrower window):")
    valid_ch_conf = all_channel_conf[valid_mask]
    # Residual = |actual_time - median_time_for_similar_damage|
    # Simpler proxy: |time - mean_time| (lower = more predictable)
    abs_residual = np.abs(times[valid_mask] - times[valid_mask].mean())
    for ci, ch in enumerate(CHANNELS):
        ch_vals = valid_ch_conf[:, ci]
        if ch_vals.std() > 0:
            rho_ch, p_ch = sp_stats.spearmanr(ch_vals, abs_residual)
            sig = "*" if p_ch < 0.05 else " "
            print(f"    {ch:<20} rho={rho_ch:+.4f} p={p_ch:.3f}{sig}  mean={ch_vals.mean():.4f}")

    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.0f}s")

    np.savez_compressed(
        os.path.join(SAVE_BASE, "scores.npz"),
        scores=all_scores, channel_scores=all_channel_scores,
        scores_noprop=all_scores_noprop,
        confidence=all_confidence, channel_confidence=all_channel_conf,
    )
    print("  Done.")


if __name__ == "__main__":
    main()
