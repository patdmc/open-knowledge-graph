#!/usr/bin/env python3
"""
Patient-level confidence analysis: how well does the graph personalize
survival projections beyond cancer-type averages?

For each patient:
  1. Cancer-type baseline = mean survival for their cancer type
  2. Graph deviation = how far the graph walk moves them from baseline
  3. Bin patients by deviation magnitude
  4. Show C-index per bin — patients the graph moves most should be predicted best

This is the fair comparison: not arbitrary cancer-type buckets, but
patient-level personalization across the whole corpus.

Usage:
    python3 -u -m gnn.scripts.patient_confidence
"""

import sys, os, json, numpy as np, torch, pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence, expanded_graph_walk_batch, fit_per_ct_ridge,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "patient_confidence")


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  PATIENT-LEVEL CONFIDENCE ANALYSIS")
    print("  Graph personalization vs cancer-type baseline")
    print("=" * 90)

    expanded_cm = load_expanded_channel_map()

    # --- Load data ---
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))

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
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
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
    print(f"\n  {all_in_val.sum().item()} patients with V6c predictions")

    # --- Load mutations ---
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

    # --- Build graph ---
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    ppi_edges = fetch_string_expanded(expanded_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)

    # --- Compute shifts ---
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

    # --- Graph walk ---
    print("\n  Running expanded graph walk...")
    X_all = expanded_graph_walk_batch(
        G_ppi, patient_genes, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, N,
    )
    print("  Done.")

    # Save X_all for future use
    np.save(os.path.join(SAVE_BASE, "X_all.npy"), X_all)

    # --- Per-CT ridge scores ---
    print("\n  Fitting per-CT ridge...")
    graph_scores = fit_per_ct_ridge(X_all, hazards, valid_mask, events, times,
                                     ct_per_patient, folds, ct_min_patients=200)

    # Global C-index
    valid_idx = np.where(valid_mask)[0]
    ci_graph_global = concordance_index(
        torch.tensor(graph_scores[valid_idx].astype(np.float32)),
        torch.tensor(times[valid_idx].astype(np.float32)),
        torch.tensor(events[valid_idx].astype(np.float32)),
    )
    ci_v6c_global = concordance_index(
        torch.tensor(hazards[valid_idx].astype(np.float32)),
        torch.tensor(times[valid_idx].astype(np.float32)),
        torch.tensor(events[valid_idx].astype(np.float32)),
    )
    print(f"\n  Global C-index:  Graph={ci_graph_global:.4f}  V6c={ci_v6c_global:.4f}")

    # =========================================================================
    # Cancer-type baseline per patient
    # =========================================================================
    ct_mean_time = {}
    ct_event_rate = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        if len(pts_arr) >= 20:
            ct_mean_time[ct_name] = float(times[pts_arr].mean())
            ct_event_rate[ct_name] = float(events[pts_arr].mean())

    # Per-patient: deviation from cancer-type mean score
    ct_mean_score_graph = {}
    ct_mean_score_v6c = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 50:
            ct_mean_score_graph[ct_name] = float(graph_scores[pts_valid].mean())
            ct_mean_score_v6c[ct_name] = float(hazards[pts_valid].mean())

    graph_deviation = np.zeros(N)
    v6c_deviation = np.zeros(N)
    for idx in valid_idx:
        ct = ct_per_patient[idx]
        if ct in ct_mean_score_graph:
            graph_deviation[idx] = abs(graph_scores[idx] - ct_mean_score_graph[ct])
            v6c_deviation[idx] = abs(hazards[idx] - ct_mean_score_v6c[ct])

    # Number of mutations per patient
    n_muts = np.zeros(N, dtype=int)
    for idx, genes in patient_genes.items():
        if idx < N:
            n_muts[idx] = len(genes)

    # =========================================================================
    # ANALYSIS 1: Bin by graph deviation (personalization magnitude)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  HOW FAR DOES THE GRAPH MOVE PATIENTS FROM THEIR CANCER-TYPE AVERAGE?")
    print(f"{'='*90}")

    dev_valid = graph_deviation[valid_idx]
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    pct_vals = np.percentile(dev_valid, percentiles)
    print(f"\n  Graph deviation distribution (|score - CT mean|):")
    for p, v in zip(percentiles, pct_vals):
        print(f"    P{p:>3}: {v:.4f}")

    # Quintile bins by deviation
    print(f"\n  PATIENTS BINNED BY PERSONALIZATION (graph deviation quintiles)")
    print(f"  More deviation = graph moves them further from cancer-type average")
    print(f"\n  {'Bin':<25} {'N':>6} {'Dev Range':>14} {'Graph CI':>9} {'V6c CI':>9} {'Delta':>7} {'Muts':>5} {'Event%':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*14} {'-'*9} {'-'*9} {'-'*7} {'-'*5} {'-'*7}")

    n_bins = 5
    bin_edges = np.percentile(dev_valid, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 0.001  # include max

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (graph_deviation[valid_idx] >= lo) & (graph_deviation[valid_idx] < hi)
        idx_bin = valid_idx[mask]

        if len(idx_bin) < 50 or events[idx_bin].sum() < 5:
            continue

        ci_g = concordance_index(
            torch.tensor(graph_scores[idx_bin].astype(np.float32)),
            torch.tensor(times[idx_bin].astype(np.float32)),
            torch.tensor(events[idx_bin].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[idx_bin].astype(np.float32)),
            torch.tensor(times[idx_bin].astype(np.float32)),
            torch.tensor(events[idx_bin].astype(np.float32)),
        )
        delta = ci_g - ci_v
        avg_muts = n_muts[idx_bin].mean()
        event_rate = events[idx_bin].mean()

        label = f"Q{b+1} ({'low' if b==0 else 'high' if b==n_bins-1 else 'mid'} personalization)"
        print(f"  {label:<25} {len(idx_bin):>6} {lo:>6.4f}-{hi:.4f} {ci_g:>9.4f} {ci_v:>9.4f} {delta:>+7.4f} {avg_muts:>5.1f} {event_rate:>6.1%}")

    # =========================================================================
    # ANALYSIS 2: Bin by number of mutations (graph complexity)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  C-INDEX BY MUTATION COUNT (graph complexity)")
    print(f"{'='*90}")

    print(f"\n  {'Mutations':>10} {'N':>6} {'Graph CI':>9} {'V6c CI':>9} {'Delta':>7} {'Event%':>7}")
    print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")

    mut_bins = [(0, 0, "0 (wild type)"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"),
                (4, 5, "4-5"), (6, 8, "6-8"), (9, 15, "9-15"), (16, 999, "16+")]

    for lo, hi, label in mut_bins:
        mask = (n_muts[valid_idx] >= lo) & (n_muts[valid_idx] <= hi)
        idx_bin = valid_idx[mask]
        if len(idx_bin) < 50 or events[idx_bin].sum() < 5:
            continue

        ci_g = concordance_index(
            torch.tensor(graph_scores[idx_bin].astype(np.float32)),
            torch.tensor(times[idx_bin].astype(np.float32)),
            torch.tensor(events[idx_bin].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[idx_bin].astype(np.float32)),
            torch.tensor(times[idx_bin].astype(np.float32)),
            torch.tensor(events[idx_bin].astype(np.float32)),
        )
        delta = ci_g - ci_v
        event_rate = events[idx_bin].mean()
        print(f"  {label:>10} {len(idx_bin):>6} {ci_g:>9.4f} {ci_v:>9.4f} {delta:>+7.4f} {event_rate:>6.1%}")

    # =========================================================================
    # ANALYSIS 3: Co-occurring mutations — the graph's sweet spot
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  CO-OCCURRING MUTATIONS: WHERE THE GRAPH SHOULD DOMINATE")
    print(f"{'='*90}")

    n_channels_hit = np.zeros(N, dtype=int)
    for idx in range(N):
        n_channels_hit[idx] = len(patient_channels.get(idx, set()))

    print(f"\n  {'Channels Hit':>13} {'N':>6} {'Graph CI':>9} {'V6c CI':>9} {'Delta':>7} {'Muts':>5}")
    print(f"  {'-'*13} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*5}")

    for n_ch in range(7):
        if n_ch < 5:
            mask = n_channels_hit[valid_idx] == n_ch
            label = f"{n_ch}"
        elif n_ch == 5:
            mask = (n_channels_hit[valid_idx] >= 5) & (n_channels_hit[valid_idx] <= 6)
            label = "5-6"
        else:
            mask = n_channels_hit[valid_idx] >= 7
            label = "7-8"

        idx_bin = valid_idx[mask]
        if len(idx_bin) < 50 or events[idx_bin].sum() < 5:
            continue

        ci_g = concordance_index(
            torch.tensor(graph_scores[idx_bin].astype(np.float32)),
            torch.tensor(times[idx_bin].astype(np.float32)),
            torch.tensor(events[idx_bin].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[idx_bin].astype(np.float32)),
            torch.tensor(times[idx_bin].astype(np.float32)),
            torch.tensor(events[idx_bin].astype(np.float32)),
        )
        delta = ci_g - ci_v
        avg_muts = n_muts[idx_bin].mean()
        print(f"  {label:>13} {len(idx_bin):>6} {ci_g:>9.4f} {ci_v:>9.4f} {delta:>+7.4f} {avg_muts:>5.1f}")

    # =========================================================================
    # ANALYSIS 4: Cross-cancer-type ranking (the global test)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  CROSS-CANCER-TYPE RANKING (the real test)")
    print(f"{'='*90}")
    print(f"\n  The graph should rank a glioma patient with IDH1+TP53+ATRX")
    print(f"  worse than a thyroid patient with BRAF alone — across types.")
    print(f"\n  Can it rank patients from DIFFERENT cancer types correctly?")

    # Pick random pairs from different cancer types
    np.random.seed(42)
    n_pairs = 100000
    idx_a = np.random.choice(valid_idx, n_pairs)
    idx_b = np.random.choice(valid_idx, n_pairs)

    # Only keep pairs from different cancer types with different outcomes
    diff_ct = np.array([ct_per_patient[a] != ct_per_patient[b] for a, b in zip(idx_a, idx_b)])
    concordant_mask = (times[idx_a] != times[idx_b])  # different times
    keep = diff_ct & concordant_mask
    idx_a = idx_a[keep]
    idx_b = idx_b[keep]

    # Concordance: does the model correctly rank the longer-surviving patient lower risk?
    def cross_concordance(scores, t, e, ia, ib):
        """Manual concordance for pre-selected pairs."""
        concordant = 0
        discordant = 0
        for i in range(len(ia)):
            a, b = ia[i], ib[i]
            # Only count if one has an event and shorter time
            if t[a] < t[b] and e[a] == 1:
                if scores[a] > scores[b]:
                    concordant += 1
                elif scores[a] < scores[b]:
                    discordant += 1
            elif t[b] < t[a] and e[b] == 1:
                if scores[b] > scores[a]:
                    concordant += 1
                elif scores[b] < scores[a]:
                    discordant += 1
        total = concordant + discordant
        return concordant / total if total > 0 else 0.5, total

    # Subsample for speed
    n_sub = min(50000, len(idx_a))
    idx_a = idx_a[:n_sub]
    idx_b = idx_b[:n_sub]

    ci_cross_graph, n_cross = cross_concordance(graph_scores, times, events, idx_a, idx_b)
    ci_cross_v6c, _ = cross_concordance(hazards, times, events, idx_a, idx_b)

    # CT-baseline only: just use the cancer-type mean hazard
    ct_baseline_scores = np.zeros(N)
    for idx in range(N):
        ct = ct_per_patient[idx]
        ct_baseline_scores[idx] = ct_mean_score_v6c.get(ct, baseline)
    ci_cross_ctbase, _ = cross_concordance(ct_baseline_scores, times, events, idx_a, idx_b)

    print(f"\n  Cross-cancer-type concordance ({n_cross:,} comparable pairs):")
    print(f"    CT baseline only:  {ci_cross_ctbase:.4f}")
    print(f"    V6c transformer:   {ci_cross_v6c:.4f}  (+ {ci_cross_v6c - ci_cross_ctbase:+.4f} over baseline)")
    print(f"    Graph walk:        {ci_cross_graph:.4f}  (+ {ci_cross_graph - ci_cross_ctbase:+.4f} over baseline)")
    print(f"    Graph vs V6c:      {ci_cross_graph - ci_cross_v6c:+.4f}")

    # =========================================================================
    # ANALYSIS 5: Patient-level summary
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  SUMMARY: PATIENT-LEVEL PROJECTION QUALITY")
    print(f"{'='*90}")

    # How many patients get a meaningfully different projection from CT average?
    # "Meaningful" = deviation > 1 std of the CT's score distribution
    n_meaningful = 0
    for idx in valid_idx:
        ct = ct_per_patient[idx]
        ct_pts = np.array(sorted(ct_patients.get(ct, set())))
        ct_valid = ct_pts[valid_mask[ct_pts]]
        if len(ct_valid) < 20:
            continue
        ct_std = graph_scores[ct_valid].std()
        if ct_std > 0 and abs(graph_scores[idx] - graph_scores[ct_valid].mean()) > ct_std:
            n_meaningful += 1

    print(f"\n  Total patients:                      {len(valid_idx):,}")
    print(f"  Meaningfully personalized (>1 SD):   {n_meaningful:,} ({n_meaningful/len(valid_idx)*100:.1f}%)")
    print(f"\n  Global C-index:")
    print(f"    CT baseline only:  {ci_cross_ctbase:.4f}")
    print(f"    V6c transformer:   {ci_v6c_global:.4f}")
    print(f"    Graph walk:        {ci_graph_global:.4f}")

    # Save
    results = {
        "global_ci_graph": float(ci_graph_global),
        "global_ci_v6c": float(ci_v6c_global),
        "cross_ct_ci_graph": float(ci_cross_graph),
        "cross_ct_ci_v6c": float(ci_cross_v6c),
        "cross_ct_ci_baseline": float(ci_cross_ctbase),
        "n_meaningful": n_meaningful,
        "n_total": len(valid_idx),
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
