#!/usr/bin/env python3
"""
Per-cancer-type C-index histogram: how many patients are in cancer types
where the graph scorer achieves various C-index thresholds.
"""

import sys, os, json, numpy as np, torch, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

# Reuse the expanded graph scorer machinery
from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence, expanded_graph_walk_batch, fit_per_ct_ridge,
)


def main():
    print("=" * 90)
    print("  PER-CANCER-TYPE C-INDEX HISTOGRAM")
    print("=" * 90)

    expanded_cm = load_expanded_channel_map()

    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    channel_features = data["channel_features"].numpy()
    tier_features = data["tier_features"].numpy()
    age = data["age"].numpy()
    sex = data["sex"].numpy()
    msi = data["msi_score"].numpy()
    msi_high = data["msi_high"].numpy()
    tmb = data["tmb"].numpy()

    # V6c predictions
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
        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {k: data[k][idx] for k in [
                "channel_features", "tier_features", "cancer_type_idx",
                "age", "sex", "msi_score", "msi_high", "tmb",
            ]}
            with torch.no_grad():
                h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline = hazards[valid_mask].mean()

    # Mutations
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange", "mutationType"],
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
    patient_channels = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            g = row["gene.hugoGeneSymbol"]
            patient_genes[idx].add(g)
            ch = expanded_cm.get(g)
            if ch:
                patient_channels[idx].add(ch)

    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    ppi_edges = fetch_string_expanded(expanded_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)

    # Compute shifts
    gene_patients_map = defaultdict(set)
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)

    global_gene_shift = {}
    for gene, pts in gene_patients_map.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            global_gene_shift[gene] = float(hazards[pts_valid].mean() - baseline)

    ct_patients = defaultdict(set)
    ct_ch_patients = defaultdict(lambda: defaultdict(set))
    ct_gene_patients_map = defaultdict(lambda: defaultdict(set))
    for idx in range(N):
        ct_name = ct_per_patient[idx]
        ct_patients[ct_name].add(idx)
        for ch in patient_channels.get(idx, set()):
            ct_ch_patients[ct_name][ch].add(idx)
        for g in patient_genes.get(idx, set()):
            ct_gene_patients_map[ct_name][g].add(idx)

    ct_baseline_map = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 50:
            ct_baseline_map[ct_name] = float(hazards[pts_valid].mean())

    ct_ch_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for ch_name in set(expanded_cm.values()):
            pts = ct_ch_patients[ct_name].get(ch_name, set())
            pts_arr = np.array(sorted(pts)) if pts else np.array([], dtype=int)
            if len(pts_arr) == 0:
                continue
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 20:
                ct_ch_shift[(ct_name, ch_name)] = float(hazards[pts_valid].mean() - bl)

    ct_gene_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for gene, pts in ct_gene_patients_map[ct_name].items():
            pts_arr = np.array(sorted(pts))
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 15:
                ct_gene_shift[(ct_name, gene)] = float(hazards[pts_valid].mean() - bl)

    # Graph walk
    print("\n  Running graph walk...")
    X_all = expanded_graph_walk_batch(
        G_ppi, patient_genes, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, N,
    )
    print("  Done.")

    # Per-CT ridge
    folds = list(skf.split(np.arange(N), events))
    score_perct = fit_per_ct_ridge(X_all, hazards, valid_mask, events, times,
                                    ct_per_patient, folds, ct_min_patients=200)

    # =========================================================================
    # Per-cancer-type C-index
    # =========================================================================
    ct_ci = {}
    ct_ci_v6c = {}
    ct_n = {}
    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 50 or events[ct_indices].sum() < 5:
            continue

        ci = concordance_index(
            torch.tensor(score_perct[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v6c = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ct_ci[ct_name] = ci
        ct_ci_v6c[ct_name] = ci_v6c
        ct_n[ct_name] = len(ct_indices)

    # =========================================================================
    # Histogram
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE C-INDEX DISTRIBUTION (Graph Walk + Per-CT Ridge)")
    print(f"{'='*90}")

    # Sort by graph C-index
    sorted_cts = sorted(ct_ci, key=lambda x: ct_ci[x], reverse=True)

    print(f"\n  {'Cancer Type':<35} {'N':>5} {'Graph':>6} {'V6c':>6} {'Δ':>6}  Bar")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6}  {'-'*40}")

    for ct in sorted_cts:
        ci = ct_ci[ct]
        ci_v = ct_ci_v6c[ct]
        n = ct_n[ct]
        delta = ci - ci_v

        # Bar: scale 0.5-0.85 to 0-35 chars
        bar_len = max(0, int((ci - 0.50) * 100))
        bar = "█" * bar_len

        # Mark 0.75 threshold
        marker = " ★" if ci >= 0.75 else ""
        beat = " ▲" if delta > 0.005 else ""

        print(f"  {ct:<35} {n:>5} {ci:>6.3f} {ci_v:>6.3f} {delta:>+6.3f}  {bar}{marker}{beat}")

    # Thresholds
    print(f"\n{'='*90}")
    print(f"  PATIENT COVERAGE BY C-INDEX THRESHOLD")
    print(f"{'='*90}")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print(f"\n  {'Threshold':>10} {'CTs':>5} {'Patients':>9} {'% of Total':>10}  Cancer Types")
    print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*10}  {'-'*50}")

    for thresh in thresholds:
        cts_above = [ct for ct in ct_ci if ct_ci[ct] >= thresh]
        n_patients = sum(ct_n[ct] for ct in cts_above)
        pct = n_patients / sum(ct_n.values()) * 100

        if len(cts_above) <= 5:
            ct_list = ", ".join(cts_above)
        else:
            ct_list = f"{len(cts_above)} cancer types"

        print(f"  {thresh:>10.2f} {len(cts_above):>5} {n_patients:>9,} {pct:>9.1f}%  {ct_list}")

    # V6c comparison at same thresholds
    print(f"\n  V6c transformer at same thresholds:")
    print(f"  {'Threshold':>10} {'CTs':>5} {'Patients':>9} {'% of Total':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*10}")
    for thresh in thresholds:
        cts_above = [ct for ct in ct_ci_v6c if ct_ci_v6c[ct] >= thresh]
        n_patients = sum(ct_n[ct] for ct in cts_above)
        pct = n_patients / sum(ct_n.values()) * 100
        print(f"  {thresh:>10.2f} {len(cts_above):>5} {n_patients:>9,} {pct:>9.1f}%")

    # Head to head
    print(f"\n{'='*90}")
    print(f"  HEAD TO HEAD: GRAPH vs V6c")
    print(f"{'='*90}")

    graph_wins = [(ct, ct_ci[ct] - ct_ci_v6c[ct]) for ct in ct_ci if ct_ci[ct] > ct_ci_v6c[ct]]
    v6c_wins = [(ct, ct_ci_v6c[ct] - ct_ci[ct]) for ct in ct_ci if ct_ci_v6c[ct] > ct_ci[ct]]
    ties = [ct for ct in ct_ci if abs(ct_ci[ct] - ct_ci_v6c[ct]) < 0.001]

    graph_win_patients = sum(ct_n[ct] for ct, _ in graph_wins)
    v6c_win_patients = sum(ct_n[ct] for ct, _ in v6c_wins)
    tie_patients = sum(ct_n[ct] for ct in ties)

    print(f"\n  Graph wins: {len(graph_wins)} cancer types, {graph_win_patients:,} patients")
    for ct, delta in sorted(graph_wins, key=lambda x: -x[1]):
        print(f"    {ct:<35} +{delta:.4f}  ({ct_n[ct]:,} patients)")

    print(f"\n  V6c wins:   {len(v6c_wins)} cancer types, {v6c_win_patients:,} patients")
    for ct, delta in sorted(v6c_wins, key=lambda x: -x[1])[:10]:
        print(f"    {ct:<35} +{delta:.4f}  ({ct_n[ct]:,} patients)")
    if len(v6c_wins) > 10:
        print(f"    ... and {len(v6c_wins) - 10} more")

    print(f"\n  Ties (<0.001): {len(ties)} cancer types, {tie_patients:,} patients")


if __name__ == "__main__":
    main()
