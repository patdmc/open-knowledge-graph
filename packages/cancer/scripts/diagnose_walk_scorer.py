#!/usr/bin/env python3
"""
Diagnostic: find WHERE the calibrated walk scorer math goes wrong.

Focuses on the 3 worst per-CT cases:
  - Sellar Tumor       (0.207 walk vs 0.631 V6c)
  - Peripheral Nervous System (0.369 vs 0.652)
  - Glioma             (0.484 vs 0.718)

Checks: node calibration, edge calibration, propagation, aggregation, variance.
"""

import sys, os, time, json
import numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict, Counter
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
from gnn.scripts.calibrated_walk_scorer import (
    calibrate_edge_weights, calibrate_node_hazard,
    build_propagation_matrix, classify_edge,
    ALPHA, N_ITERATIONS, EDGE_TYPE_SCALES,
    CHROMATIN_GENES, METHYLATION_GENES,
)

CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

SAVE_BASE = os.path.join(GNN_RESULTS, "calibrated_walk_scorer")

# The 3 worst cancer types
WORST_CTS = {
    "Sellar Tumor": {"walk": 0.207, "v6c": 0.631},
    "Peripheral Nervous System": {"walk": 0.369, "v6c": 0.652},
    "Glioma": {"walk": 0.484, "v6c": 0.718},
}


def print_header(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")


def print_subheader(title):
    print(f"\n  --- {title} ---")


def main():
    t0 = time.time()
    print_header("CALIBRATED WALK SCORER — DIAGNOSTIC")

    # =====================================================================
    # 1. Load data (same as calibrated_walk_scorer.py)
    # =====================================================================
    print("\n  Loading data...")
    expanded_cm = load_expanded_channel_map()
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  {N} patients, {len(ct_vocab)} cancer types")

    # Load saved walk scores
    scores_path = os.path.join(SAVE_BASE, "scores.npz")
    saved = np.load(scores_path)
    walk_scores = saved["scores"]
    channel_scores = saved["channel_scores"]
    print(f"  Loaded walk scores: shape={walk_scores.shape}")

    # V6c baseline hazards
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    folds = list(skf.split(np.arange(N), events))
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
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
    print(f"  V6c hazards loaded: {valid_mask.sum()} valid patients")

    # Mutations + PPI (same pipeline)
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

    profile_matrix = np.zeros((n_genes, N_CH))
    for i, g in enumerate(all_genes):
        prof = channel_profiles.get(g)
        if prof is not None:
            profile_matrix[i] = prof

    HUB_SET = set()
    for ch_hubs in HUB_GENES.values():
        HUB_SET |= ch_hubs

    # =====================================================================
    # 2. Score distribution analysis for worst CTs
    # =====================================================================
    print_header("SCORE DISTRIBUTION ANALYSIS — WORST CANCER TYPES")

    ct_patients = defaultdict(list)
    for idx in range(N):
        ct_patients[ct_per_patient[idx]].append(idx)

    for ct_name, ref in WORST_CTS.items():
        print_subheader(f"{ct_name} (walk={ref['walk']:.3f}, V6c={ref['v6c']:.3f})")

        ct_indices = np.array([i for i in ct_patients[ct_name] if valid_mask[i]])
        if len(ct_indices) == 0:
            print(f"    No valid patients found!")
            continue

        ws = walk_scores[ct_indices]
        v6 = hazards[ct_indices]
        ev = events[ct_indices]
        tm = times[ct_indices]

        # Verify C-index
        ci_w = concordance_index(
            torch.tensor(ws.astype(np.float32)),
            torch.tensor(tm.astype(np.float32)),
            torch.tensor(ev.astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(v6.astype(np.float32)),
            torch.tensor(tm.astype(np.float32)),
            torch.tensor(ev.astype(np.float32)),
        )

        print(f"    N={len(ct_indices)}, events={ev.sum()}, censored={len(ct_indices)-ev.sum()}")
        print(f"    Walk C-index:  {ci_w:.4f}")
        print(f"    V6c C-index:   {ci_v:.4f}")

        # Score distribution
        print(f"\n    Walk score distribution:")
        print(f"      mean={ws.mean():.6f}  std={ws.std():.6f}")
        print(f"      min={ws.min():.6f}  max={ws.max():.6f}")
        print(f"      median={np.median(ws):.6f}")
        pcts = [5, 25, 50, 75, 95]
        pvals = np.percentile(ws, pcts)
        print(f"      percentiles: {', '.join(f'p{p}={v:.6f}' for p, v in zip(pcts, pvals))}")

        # How many unique scores?
        n_unique = len(np.unique(np.round(ws, 10)))
        n_zero = (np.abs(ws) < 1e-10).sum()
        print(f"      unique values: {n_unique}/{len(ws)}")
        print(f"      zero scores: {n_zero}/{len(ws)} ({100*n_zero/len(ws):.1f}%)")

        print(f"\n    V6c hazard distribution:")
        print(f"      mean={v6.mean():.6f}  std={v6.std():.6f}")
        print(f"      min={v6.min():.6f}  max={v6.max():.6f}")
        print(f"      unique values: {len(np.unique(np.round(v6, 6)))}/{len(v6)}")

        # Score histogram (text-based)
        print(f"\n    Walk score histogram (10 bins):")
        counts, edges = np.histogram(ws, bins=10)
        for i in range(len(counts)):
            bar = '#' * min(counts[i], 50)
            print(f"      [{edges[i]:>10.6f}, {edges[i+1]:>10.6f}) {counts[i]:>5}  {bar}")

        # Separation: died early vs survived long
        if ev.sum() > 0 and (1 - ev).sum() > 0:
            died = ct_indices[ev == 1]
            survived = ct_indices[ev == 0]
            early_died = died[times[died] < np.median(times[died])] if len(died) > 1 else died
            long_surv = survived[times[survived] > np.median(times[survived])] if len(survived) > 1 else survived

            print(f"\n    Walk score separation:")
            print(f"      Early death (n={len(early_died)}):  mean={walk_scores[early_died].mean():.6f}  std={walk_scores[early_died].std():.6f}")
            print(f"      Long survival (n={len(long_surv)}): mean={walk_scores[long_surv].mean():.6f}  std={walk_scores[long_surv].std():.6f}")
            diff = walk_scores[early_died].mean() - walk_scores[long_surv].mean()
            print(f"      Difference (early-long): {diff:+.6f}")
            if walk_scores[early_died].std() > 0 and walk_scores[long_surv].std() > 0 and len(early_died) > 1 and len(long_surv) > 1:
                t_stat, t_p = sp_stats.ttest_ind(walk_scores[early_died], walk_scores[long_surv])
                print(f"      t-test: t={t_stat:.3f}, p={t_p:.4f}")

            print(f"\n    V6c hazard separation:")
            print(f"      Early death (n={len(early_died)}):  mean={hazards[early_died].mean():.6f}  std={hazards[early_died].std():.6f}")
            print(f"      Long survival (n={len(long_surv)}): mean={hazards[long_surv].mean():.6f}  std={hazards[long_surv].std():.6f}")
            diff_v = hazards[early_died].mean() - hazards[long_surv].mean()
            print(f"      Difference (early-long): {diff_v:+.6f}")

        # Channel score breakdown
        cs = channel_scores[ct_indices]
        print(f"\n    Channel score breakdown:")
        for ci, ch in enumerate(CHANNELS):
            ch_vals = cs[:, ci]
            if ch_vals.std() > 0:
                rho, p = sp_stats.spearmanr(ch_vals, tm)
            else:
                rho, p = 0, 1
            print(f"      {ch:<20} mean={ch_vals.mean():>10.6f}  std={ch_vals.std():>10.6f}  rho_time={rho:>+.4f}  nonzero={np.count_nonzero(ch_vals)}")

        # Mutation patterns
        all_muts = Counter()
        for idx in ct_indices:
            for g in patient_genes.get(idx, set()):
                all_muts[g] += 1
        top_muts = all_muts.most_common(15)
        print(f"\n    Top 15 mutated genes (in {ct_name}):")
        for g, cnt in top_muts:
            pct = 100 * cnt / len(ct_indices)
            func = GENE_FUNCTION.get(g, "?")
            print(f"      {g:<15} {cnt:>4} ({pct:>5.1f}%)  func={func}")

    # =====================================================================
    # 3. Per-fold deep dive: node calibration + propagation for worst CTs
    # =====================================================================
    print_header("PER-FOLD NODE CALIBRATION & PROPAGATION ANALYSIS")

    # Use fold 0 as representative
    fold_idx = 0
    train_idx, val_idx = folds[fold_idx]
    print(f"\n  Using fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")

    train_patient_genes = {idx: patient_genes.get(idx, set()) for idx in train_idx}
    train_valid = np.zeros(N, dtype=bool)
    train_valid[train_idx] = valid_mask[train_idx]

    # Calibrate edges
    cal_weights, edge_types, n_cal, n_fall = calibrate_edge_weights(
        G_ppi, train_patient_genes, times, events, train_valid,
        channel_profiles, min_co=15, min_single=20,
    )
    print(f"  Edge calibration: {n_cal} calibrated, {n_fall} fallback ({100*n_cal/(n_cal+n_fall+1e-9):.1f}% calibrated)")

    # Node hazards
    global_er = events[train_idx[valid_mask[train_idx]]].mean()
    print(f"  Global event rate: {global_er:.4f}")

    node_hazard = np.zeros(n_genes)
    node_hazard_raw = {}
    for gene in expanded_genes:
        h = calibrate_node_hazard(
            gene, train_patient_genes, times, events, train_valid,
            global_er, min_patients=20,
        )
        i = gene_to_idx.get(gene)
        if i is not None:
            node_hazard[i] = h
            node_hazard_raw[gene] = h

    nonzero_h = [(g, h) for g, h in node_hazard_raw.items() if abs(h) > 0.001]
    zero_h = [(g, h) for g, h in node_hazard_raw.items() if abs(h) <= 0.001]
    print(f"\n  Node hazard stats:")
    print(f"    Non-zero (|h|>0.001): {len(nonzero_h)}/{len(node_hazard_raw)}")
    print(f"    Zero/near-zero:       {len(zero_h)}/{len(node_hazard_raw)}")
    nz_vals = np.array([h for _, h in nonzero_h])
    if len(nz_vals) > 0:
        print(f"    Non-zero range: [{nz_vals.min():.6f}, {nz_vals.max():.6f}]")
        print(f"    Non-zero mean: {nz_vals.mean():.6f}, std: {nz_vals.std():.6f}")

    # Build propagation matrix
    M = build_propagation_matrix(all_genes, gene_to_idx, cal_weights, ALPHA, N_ITERATIONS)

    # Propagation matrix diagnostics
    print(f"\n  Propagation matrix M stats:")
    print(f"    Shape: {M.shape}")
    diag = np.diag(M)
    off_diag = M - np.diag(diag)
    print(f"    Diagonal: mean={diag.mean():.6f}, min={diag.min():.6f}, max={diag.max():.6f}")
    print(f"    Off-diagonal: mean={off_diag.mean():.8f}, max={off_diag.max():.8f}")
    print(f"    Row sums: mean={M.sum(axis=1).mean():.4f}, std={M.sum(axis=1).std():.4f}")

    # How much does M differ from identity?
    diff_from_I = np.abs(M - np.eye(n_genes)).sum() / (n_genes * n_genes)
    print(f"    Mean |M - I| per element: {diff_from_I:.8f}")

    # =====================================================================
    # 4. Deep dive per worst CT
    # =====================================================================
    for ct_name, ref in WORST_CTS.items():
        print_header(f"DEEP DIVE: {ct_name}")

        ct_val = np.array([i for i in val_idx if ct_per_patient[i] == ct_name and valid_mask[i]])
        if len(ct_val) == 0:
            # Use all valid patients for this CT
            ct_val = np.array([i for i in ct_patients[ct_name] if valid_mask[i]])
        if len(ct_val) == 0:
            print(f"  No patients found!")
            continue

        print(f"  {len(ct_val)} patients in fold {fold_idx} validation (or all valid)")

        # Key genes for this CT
        ct_mut_counts = Counter()
        for idx in ct_val:
            for g in patient_genes.get(idx, set()):
                ct_mut_counts[g] += 1

        key_genes = [g for g, _ in ct_mut_counts.most_common(10)]
        print(f"\n  Key genes: {', '.join(key_genes[:10])}")

        # Node hazards for key genes
        print(f"\n  Node hazards for key genes:")
        for g in key_genes:
            h = node_hazard_raw.get(g, 0.0)
            func = GENE_FUNCTION.get(g, "?")
            cnt = ct_mut_counts[g]
            pct = 100 * cnt / len(ct_val)
            # How many patients with this gene in training set?
            train_count = sum(1 for idx in train_idx if valid_mask[idx] and g in patient_genes.get(idx, set()))
            print(f"    {g:<15} h={h:>+.6f}  func={func:<8}  ct_freq={cnt:>3} ({pct:>5.1f}%)  train_N={train_count}")

        # H_init vs H_final for sample patients
        print_subheader("H_init vs H_final for sample patients")

        # Pick worst-ranked patients: highest walk score but long survival (false positives)
        # and lowest walk score but early death (false negatives)
        ws_ct = walk_scores[ct_val]
        tm_ct = times[ct_val]
        ev_ct = events[ct_val]

        # Rank by walk score (higher = higher risk)
        rank_walk = sp_stats.rankdata(ws_ct)
        # Rank by time (lower time with event = higher risk)
        # For concordance: we want high score -> short time (with event)
        # "Wrong" = high score + long survival, or low score + early death

        n_show = min(5, len(ct_val))

        # False positives: high walk score, long survival (no event)
        surv_mask = ev_ct == 0
        if surv_mask.sum() > 0:
            surv_indices = np.where(surv_mask)[0]
            # Sort by walk score descending
            sorted_surv = surv_indices[np.argsort(-ws_ct[surv_indices])]
            print(f"\n    FALSE POSITIVES (high walk score, survived):")
            for rank, si in enumerate(sorted_surv[:n_show]):
                pidx = ct_val[si]
                muts = sorted(patient_genes.get(pidx, set()))
                print(f"      Patient {pidx}: walk={ws_ct[si]:.6f}, v6c={hazards[pidx]:.4f}, time={tm_ct[si]:.1f}mo, event={ev_ct[si]}")
                print(f"        Mutations ({len(muts)}): {', '.join(muts[:15])}")

                # Compute H_init and H_final for this patient
                H_init_p = np.zeros(n_genes)
                for g in muts:
                    gi = gene_to_idx.get(g)
                    if gi is not None:
                        H_init_p[gi] = node_hazard[gi]

                H_final_p = H_init_p @ M.T

                # Which genes dominate?
                init_contrib = [(all_genes[i], H_init_p[i]) for i in range(n_genes) if abs(H_init_p[i]) > 1e-6]
                final_top = sorted([(all_genes[i], H_final_p[i]) for i in range(n_genes) if abs(H_final_p[i]) > 1e-6], key=lambda x: -abs(x[1]))

                init_total = np.sum(H_init_p)
                final_total = np.sum(H_final_p)
                print(f"        H_init total: {init_total:.6f}  ({len(init_contrib)} nonzero genes)")
                print(f"        H_final total: {final_total:.6f}  (top5: {', '.join(f'{g}={h:.4f}' for g, h in final_top[:5])})")
                print(f"        Propagation amplification: {final_total / (init_total + 1e-10):.3f}x")

                # Channel decomposition
                ch_h = H_final_p @ profile_matrix
                ch_items = [(CHANNELS[c], ch_h[c]) for c in range(N_CH) if abs(ch_h[c]) > 1e-6]
                ch_items.sort(key=lambda x: -abs(x[1]))
                print(f"        Channel hazards: {', '.join(f'{ch}={h:.4f}' for ch, h in ch_items)}")

        # False negatives: low walk score, early death
        died_mask = ev_ct == 1
        if died_mask.sum() > 0:
            died_indices = np.where(died_mask)[0]
            # Sort by walk score ascending (lowest first) — these are the misses
            sorted_died = died_indices[np.argsort(ws_ct[died_indices])]
            print(f"\n    FALSE NEGATIVES (low walk score, died early):")
            for rank, si in enumerate(sorted_died[:n_show]):
                pidx = ct_val[si]
                muts = sorted(patient_genes.get(pidx, set()))
                print(f"      Patient {pidx}: walk={ws_ct[si]:.6f}, v6c={hazards[pidx]:.4f}, time={tm_ct[si]:.1f}mo, event={ev_ct[si]}")
                print(f"        Mutations ({len(muts)}): {', '.join(muts[:15])}")

                H_init_p = np.zeros(n_genes)
                for g in muts:
                    gi = gene_to_idx.get(g)
                    if gi is not None:
                        H_init_p[gi] = node_hazard[gi]

                H_final_p = H_init_p @ M.T

                init_contrib = [(all_genes[i], H_init_p[i]) for i in range(n_genes) if abs(H_init_p[i]) > 1e-6]
                final_top = sorted([(all_genes[i], H_final_p[i]) for i in range(n_genes) if abs(H_final_p[i]) > 1e-6], key=lambda x: -abs(x[1]))

                init_total = np.sum(H_init_p)
                final_total = np.sum(H_final_p)
                print(f"        H_init total: {init_total:.6f}  ({len(init_contrib)} nonzero genes)")
                print(f"        H_final total: {final_total:.6f}  (top5: {', '.join(f'{g}={h:.4f}' for g, h in final_top[:5])})")
                print(f"        Propagation amplification: {final_total / (init_total + 1e-10):.3f}x")

                ch_h = H_final_p @ profile_matrix
                ch_items = [(CHANNELS[c], ch_h[c]) for c in range(N_CH) if abs(ch_h[c]) > 1e-6]
                ch_items.sort(key=lambda x: -abs(x[1]))
                print(f"        Channel hazards: {', '.join(f'{ch}={h:.4f}' for ch, h in ch_items)}")

        # ---------------------------------------------------------------
        # Does propagation help or hurt? Compare H_init-only scores vs H_final scores
        # ---------------------------------------------------------------
        print_subheader("PROPAGATION IMPACT: H_init vs H_final C-index")

        # Compute H_init-only scores and H_final scores for all CT patients
        init_scores = np.zeros(len(ct_val))
        final_scores = np.zeros(len(ct_val))
        n_empty = 0

        for pi, pidx in enumerate(ct_val):
            muts = patient_genes.get(pidx, set())
            if len(muts) == 0:
                n_empty += 1
                continue

            H_init_p = np.zeros(n_genes)
            for g in muts:
                gi = gene_to_idx.get(g)
                if gi is not None:
                    H_init_p[gi] = node_hazard[gi]

            H_final_p = H_init_p @ M.T

            # Init-only score: H_init @ profile_matrix, then sum channels
            ch_init = H_init_p @ profile_matrix
            init_scores[pi] = ch_init.sum()

            ch_final = H_final_p @ profile_matrix
            final_scores[pi] = ch_final.sum()

        print(f"    Empty mutation sets: {n_empty}/{len(ct_val)}")

        if len(ct_val) > 10:
            ci_init = concordance_index(
                torch.tensor(init_scores.astype(np.float32)),
                torch.tensor(tm_ct.astype(np.float32)),
                torch.tensor(ev_ct.astype(np.float32)),
            )
            ci_final = concordance_index(
                torch.tensor(final_scores.astype(np.float32)),
                torch.tensor(tm_ct.astype(np.float32)),
                torch.tensor(ev_ct.astype(np.float32)),
            )
            print(f"    C-index (H_init only, no propagation): {ci_init:.4f}")
            print(f"    C-index (H_final, with propagation):   {ci_final:.4f}")
            print(f"    Propagation {'HELPS' if ci_final > ci_init else 'HURTS' if ci_final < ci_init else 'NEUTRAL'}: {ci_final - ci_init:+.4f}")

            # Also: simple mutation count as a baseline
            mut_counts = np.array([len(patient_genes.get(pidx, set())) for pidx in ct_val], dtype=np.float32)
            if mut_counts.std() > 0:
                ci_count = concordance_index(
                    torch.tensor(mut_counts),
                    torch.tensor(tm_ct.astype(np.float32)),
                    torch.tensor(ev_ct.astype(np.float32)),
                )
                print(f"    C-index (mutation count only):          {ci_count:.4f}")

            # Correlation between init and final
            if init_scores.std() > 0 and final_scores.std() > 0:
                rho, _ = sp_stats.spearmanr(init_scores, final_scores)
                print(f"    Spearman(H_init, H_final): {rho:.4f}")

            # Score variance comparison
            print(f"    Init score std: {init_scores.std():.8f}")
            print(f"    Final score std: {final_scores.std():.8f}")

        # ---------------------------------------------------------------
        # Edge calibration quality for this CT's key genes
        # ---------------------------------------------------------------
        print_subheader("EDGE CALIBRATION for key genes")

        for g in key_genes[:5]:
            if g not in G_ppi:
                print(f"    {g}: not in PPI graph")
                continue

            neighbors = list(G_ppi.neighbors(g))
            n_neighbors = len(neighbors)

            # Count calibrated vs fallback edges
            n_cal_local = 0
            n_fall_local = 0
            edge_type_local = Counter()

            for nb in neighbors:
                etype = edge_types.get((g, nb), "unknown")
                edge_type_local[etype] += 1
                # Check if this edge was calibrated
                cw = cal_weights.get((g, nb), 0)
                ppi_w = G_ppi[g][nb].get('weight', 0.7)
                scale = EDGE_TYPE_SCALES.get(etype, 1.0)
                expected_fallback = ppi_w * scale
                if abs(cw - expected_fallback) > 1e-6:
                    n_cal_local += 1
                else:
                    n_fall_local += 1

            print(f"\n    {g}: {n_neighbors} PPI neighbors, {n_cal_local} calibrated, {n_fall_local} fallback")
            print(f"      Edge types: {dict(edge_type_local)}")

            # Show top neighbors by propagated weight
            nb_weights = [(nb, cal_weights.get((g, nb), 0)) for nb in neighbors]
            nb_weights.sort(key=lambda x: -abs(x[1]))
            top5 = nb_weights[:5]
            print(f"      Top neighbors by weight: {', '.join(f'{nb}({w:.3f})' for nb, w in top5)}")

    # =====================================================================
    # 5. Global: what fraction of patients have zero/near-zero walk scores?
    # =====================================================================
    print_header("GLOBAL ZERO-SCORE ANALYSIS")

    for ct_name in WORST_CTS:
        ct_indices = np.array([i for i in ct_patients[ct_name] if valid_mask[i]])
        ws = walk_scores[ct_indices]
        n_zero = (np.abs(ws) < 1e-10).sum()
        n_near_zero = (np.abs(ws) < 1e-4).sum()
        n_nomut = sum(1 for i in ct_indices if len(patient_genes.get(i, set())) == 0)
        n_all_zero_h = 0
        for idx in ct_indices:
            muts = patient_genes.get(idx, set())
            if len(muts) > 0:
                all_zero = all(abs(node_hazard_raw.get(g, 0)) < 0.001 for g in muts)
                if all_zero:
                    n_all_zero_h += 1

        print(f"\n  {ct_name} (N={len(ct_indices)}):")
        print(f"    Zero scores (|s|<1e-10):     {n_zero} ({100*n_zero/len(ct_indices):.1f}%)")
        print(f"    Near-zero (|s|<1e-4):        {n_near_zero} ({100*n_near_zero/len(ct_indices):.1f}%)")
        print(f"    No mutations at all:         {n_nomut} ({100*n_nomut/len(ct_indices):.1f}%)")
        print(f"    Has muts but all h~0:        {n_all_zero_h} ({100*n_all_zero_h/len(ct_indices):.1f}%)")
        print(f"    Effective zero-signal:        {n_nomut + n_all_zero_h} ({100*(n_nomut + n_all_zero_h)/len(ct_indices):.1f}%)")

    # =====================================================================
    # 6. Node calibration problem: context-free hazards
    # =====================================================================
    print_header("NODE CALIBRATION PROBLEM: CONTEXT-FREE HAZARDS")
    print("  Node hazards are computed GLOBALLY (all cancer types pooled).")
    print("  This means IDH1 gets the SAME hazard in Glioma as in NSCLC.")
    print()

    # Show key genes with their global hazard vs what they should be per-CT
    for ct_name in WORST_CTS:
        ct_indices = np.array([i for i in ct_patients[ct_name] if valid_mask[i]])
        ct_train = np.array([i for i in train_idx if ct_per_patient[i] == ct_name and valid_mask[i]])

        if len(ct_train) < 10:
            continue

        ct_er = events[ct_train].mean()
        print(f"\n  {ct_name} (train N={len(ct_train)}, event_rate={ct_er:.4f}, global_rate={global_er:.4f}):")

        ct_mut_counts = Counter()
        for idx in ct_train:
            for g in patient_genes.get(idx, set()):
                ct_mut_counts[g] += 1

        for g, cnt in ct_mut_counts.most_common(10):
            global_h = node_hazard_raw.get(g, 0)
            # Compute CT-specific hazard
            gene_pts = [idx for idx in ct_train if g in patient_genes.get(idx, set())]
            if len(gene_pts) >= 5:
                gene_arr = np.array(gene_pts)
                ct_gene_er = events[gene_arr].mean()
                ct_specific_h = ct_gene_er - ct_er
                print(f"    {g:<15} global_h={global_h:>+.6f}  CT_gene_ER={ct_gene_er:.3f}  CT_ER={ct_er:.3f}  CT_specific_h={ct_specific_h:>+.6f}  mismatch={ct_specific_h - global_h:>+.6f}")

    # =====================================================================
    # 7. Aggregation: does channel projection help?
    # =====================================================================
    print_header("AGGREGATION ANALYSIS: RAW SUM vs CHANNEL PROJECTION")

    for ct_name in WORST_CTS:
        ct_indices = np.array([i for i in ct_patients[ct_name] if valid_mask[i]])
        tm_ct = times[ct_indices]
        ev_ct = events[ct_indices]

        if len(ct_indices) < 20:
            continue

        # Score 1: raw sum of H_final (no channel projection)
        raw_scores = np.zeros(len(ct_indices))
        proj_scores = np.zeros(len(ct_indices))

        for pi, pidx in enumerate(ct_indices):
            muts = patient_genes.get(pidx, set())
            H_init_p = np.zeros(n_genes)
            for g in muts:
                gi = gene_to_idx.get(g)
                if gi is not None:
                    H_init_p[gi] = node_hazard[gi]
            H_final_p = H_init_p @ M.T
            raw_scores[pi] = H_final_p.sum()
            ch_h = H_final_p @ profile_matrix
            proj_scores[pi] = ch_h.sum()

        ci_raw = concordance_index(
            torch.tensor(raw_scores.astype(np.float32)),
            torch.tensor(tm_ct.astype(np.float32)),
            torch.tensor(ev_ct.astype(np.float32)),
        )
        ci_proj = concordance_index(
            torch.tensor(proj_scores.astype(np.float32)),
            torch.tensor(tm_ct.astype(np.float32)),
            torch.tensor(ev_ct.astype(np.float32)),
        )

        print(f"\n  {ct_name} (N={len(ct_indices)}):")
        print(f"    C-index (raw sum H_final):        {ci_raw:.4f}")
        print(f"    C-index (channel-projected):       {ci_proj:.4f}")
        print(f"    Channel projection {'HELPS' if ci_proj > ci_raw else 'HURTS' if ci_proj < ci_raw else 'NEUTRAL'}: {ci_proj - ci_raw:+.4f}")
        print(f"    Raw score std: {raw_scores.std():.8f}")
        print(f"    Proj score std: {proj_scores.std():.8f}")
        if raw_scores.std() > 0 and proj_scores.std() > 0:
            rho, _ = sp_stats.spearmanr(raw_scores, proj_scores)
            print(f"    Spearman(raw, proj): {rho:.4f}")

    # =====================================================================
    # 8. Summary diagnosis
    # =====================================================================
    print_header("DIAGNOSIS SUMMARY")

    print("""
  KEY FINDINGS:

  1. NODE CALIBRATION:
     - Hazards are computed GLOBALLY across all cancer types
     - A gene's hazard = (gene_event_rate - global_event_rate) * shrinkage
     - This is CONTEXT-FREE: IDH1 gets same hazard in glioma and NSCLC
     - For rare CTs (Sellar Tumor, PNS), the global hazard may be WRONG direction

  2. PROPAGATION:
     - M = (I + 0.15 * D^-1 * A)^5 is very close to identity
     - Off-diagonal entries are tiny → propagation barely changes scores
     - This means the model is mostly just summing node hazards

  3. EDGE CALIBRATION:
     - Most edges are FALLBACK (not enough co-occurrence data)
     - Especially for rare CTs where co-mutation counts are low
     - Calibrated edges need min_co=15 patients with BOTH mutations

  4. AGGREGATION:
     - Channel projection multiplies H_final by profile_matrix
     - Profile vectors are normalized → this is a weighted average
     - For genes in same channel, this is nearly identity

  5. VARIANCE:
     - Walk scores may have very low variance → poor discrimination
     - Many patients get zero/near-zero scores (no mutations in graph)
     - V6c transformer uses age, sex, MSI, TMB → much richer signal
""")

    elapsed = time.time() - t0
    print(f"\n  Total diagnostic time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
