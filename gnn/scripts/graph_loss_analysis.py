#!/usr/bin/env python3
"""
Analyze patients where the graph scorer loses to V6c.
Look for patterns: what do these patients have in common?

A "loss" = graph ranks a pair wrong where V6c gets it right.
Focus on discordant pairs: graph wrong AND v6c right.

Usage:
    python3 -u -m gnn.scripts.graph_loss_analysis
"""

import sys, os, json, numpy as np, torch, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_CHANNEL_MAP, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence, expanded_graph_walk_batch, fit_per_ct_ridge,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "graph_loss_analysis")


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  GRAPH LOSS ANALYSIS")
    print("  Where does the graph get it wrong and V6c gets it right?")
    print("=" * 90)

    expanded_cm = load_expanded_channel_map()

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
        if not os.path.exists(model_path): continue
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

    # Shifts
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
            if len(pts_arr) == 0: continue
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

    # Graph walk + per-CT ridge
    print("\n  Running graph walk...")
    X_all = expanded_graph_walk_batch(
        G_ppi, patient_genes, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, N,
    )
    print("  Done.\n  Fitting per-CT ridge...")
    graph_scores = fit_per_ct_ridge(X_all, hazards, valid_mask, events, times,
                                     ct_per_patient, folds, ct_min_patients=200)

    valid_idx = np.where(valid_mask)[0]

    # =========================================================================
    # Per-patient error: residual from actual outcome
    # =========================================================================
    # For each patient with an event, compute rank-based error
    # Higher score should mean shorter survival (higher hazard)
    # Error = cases where the model assigns wrong relative rank

    print(f"\n{'='*90}")
    print(f"  PER-PATIENT RANKING ANALYSIS")
    print(f"{'='*90}")

    # For each patient, compute their percentile rank in graph vs v6c
    graph_rank = np.zeros(N)
    v6c_rank = np.zeros(N)
    graph_rank[valid_idx] = pd.Series(graph_scores[valid_idx]).rank(pct=True).values
    v6c_rank[valid_idx] = pd.Series(hazards[valid_idx]).rank(pct=True).values

    # Actual survival rank (lower time with event = worse)
    # Use negative time for events, large positive for censored
    actual_rank_val = np.where(events == 1, -times, times + 10000)
    actual_rank = np.zeros(N)
    actual_rank[valid_idx] = pd.Series(actual_rank_val[valid_idx]).rank(pct=True).values

    # Error: |predicted_rank - actual_rank|
    graph_error = np.abs(graph_rank - actual_rank)
    v6c_error = np.abs(v6c_rank - actual_rank)

    # Where does the graph do WORSE than V6c?
    graph_worse = graph_error > v6c_error  # per patient
    graph_better = graph_error < v6c_error

    n_worse = graph_worse[valid_idx].sum()
    n_better = graph_better[valid_idx].sum()
    n_tie = len(valid_idx) - n_worse - n_better

    print(f"\n  Per-patient rank error (|predicted_rank - actual_rank|):")
    print(f"    Graph better:  {n_better:>6,} ({n_better/len(valid_idx)*100:.1f}%)")
    print(f"    V6c better:    {n_worse:>6,} ({n_worse/len(valid_idx)*100:.1f}%)")
    print(f"    Tie:           {n_tie:>6,} ({n_tie/len(valid_idx)*100:.1f}%)")

    # =========================================================================
    # PATTERNS in graph losses
    # =========================================================================
    loss_idx = valid_idx[graph_worse[valid_idx]]
    win_idx = valid_idx[graph_better[valid_idx]]

    print(f"\n{'='*90}")
    print(f"  PATTERNS: GRAPH LOSSES vs GRAPH WINS")
    print(f"{'='*90}")

    def profile(idx_set, label):
        """Profile a set of patients."""
        print(f"\n  --- {label} ({len(idx_set):,} patients) ---")

        # Cancer type distribution
        ct_counts = Counter(ct_per_patient[i] for i in idx_set)
        ct_total = Counter(ct_per_patient[i] for i in valid_idx)
        print(f"\n  Cancer type enrichment (loss/win rate vs base rate):")
        print(f"  {'Cancer Type':<35} {'In Set':>6} {'Base%':>6} {'Set%':>6} {'Ratio':>6}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
        for ct in sorted(ct_counts, key=lambda x: -ct_counts[x])[:15]:
            base_pct = ct_total[ct] / len(valid_idx) * 100
            set_pct = ct_counts[ct] / len(idx_set) * 100
            ratio = set_pct / base_pct if base_pct > 0 else 0
            flag = " **" if abs(ratio - 1) > 0.15 else ""
            print(f"  {ct:<35} {ct_counts[ct]:>6} {base_pct:>5.1f}% {set_pct:>5.1f}% {ratio:>5.2f}x{flag}")

        # Mutation count
        muts = np.array([len(patient_genes.get(i, set())) for i in idx_set])
        print(f"\n  Mutations: mean={muts.mean():.1f}, median={np.median(muts):.0f}, "
              f"P90={np.percentile(muts, 90):.0f}")

        # Channel hit count
        chs = np.array([len(patient_channels.get(i, set())) for i in idx_set])
        print(f"  Channels hit: mean={chs.mean():.1f}, median={np.median(chs):.0f}")

        # Age
        ages = age[idx_set]
        print(f"  Age: mean={ages.mean():.1f}, median={np.median(ages):.0f}")

        # Event rate
        ev = events[idx_set].mean()
        print(f"  Event rate: {ev:.1%}")

        # Survival time
        surv = times[idx_set]
        print(f"  Survival: mean={surv.mean():.1f}mo, median={np.median(surv):.1f}mo")

        # Most common genes
        gene_counts = Counter()
        for i in idx_set:
            for g in patient_genes.get(i, set()):
                gene_counts[g] += 1
        gene_total = Counter()
        for i in valid_idx:
            for g in patient_genes.get(i, set()):
                gene_total[g] += 1

        print(f"\n  Gene enrichment (top mutations):")
        print(f"  {'Gene':<15} {'In Set':>7} {'Base%':>6} {'Set%':>6} {'Ratio':>6}")
        print(f"  {'-'*15} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
        # Sort by enrichment ratio
        gene_enrichment = {}
        for g, cnt in gene_counts.items():
            if cnt < 50:
                continue
            base_pct = gene_total[g] / len(valid_idx) * 100
            set_pct = cnt / len(idx_set) * 100
            ratio = set_pct / base_pct if base_pct > 0 else 0
            gene_enrichment[g] = (cnt, base_pct, set_pct, ratio)

        for g in sorted(gene_enrichment, key=lambda x: -gene_enrichment[x][3])[:15]:
            cnt, base_pct, set_pct, ratio = gene_enrichment[g]
            flag = " **" if ratio > 1.2 else ""
            print(f"  {g:<15} {cnt:>7} {base_pct:>5.1f}% {set_pct:>5.1f}% {ratio:>5.2f}x{flag}")

        # Channel enrichment
        ch_counts = Counter()
        for i in idx_set:
            for ch in patient_channels.get(i, set()):
                ch_counts[ch] += 1
        ch_total = Counter()
        for i in valid_idx:
            for ch in patient_channels.get(i, set()):
                ch_total[ch] += 1

        print(f"\n  Channel enrichment:")
        print(f"  {'Channel':<20} {'In Set':>7} {'Base%':>6} {'Set%':>6} {'Ratio':>6}")
        print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
        for ch in sorted(ch_counts, key=lambda x: -ch_counts[x]):
            base_pct = ch_total[ch] / len(valid_idx) * 100
            set_pct = ch_counts[ch] / len(idx_set) * 100
            ratio = set_pct / base_pct if base_pct > 0 else 0
            flag = " **" if abs(ratio - 1) > 0.1 else ""
            print(f"  {ch:<20} {ch_counts[ch]:>7} {base_pct:>5.1f}% {set_pct:>5.1f}% {ratio:>5.2f}x{flag}")

        # Graph feature stats (from X_all)
        X_set = X_all[idx_set]
        feat_names = [
            "ct_baseline", "ct_ch_defense", "ct_gene_shift", "global_gene_shift",
            "n_channels_hit", "total_isolation", "total_hub_damage", "channels_severed",
            "tier_conn", "n_mutated", "n_cooccur_pairs", "total_cooccur_wt",
            "max_cooccur_wt", "cross_ch_cooccur",
        ]
        X_base = X_all[valid_idx]
        print(f"\n  Graph feature means (set vs baseline):")
        print(f"  {'Feature':<20} {'Set Mean':>9} {'Base Mean':>10} {'Ratio':>6}")
        print(f"  {'-'*20} {'-'*9} {'-'*10} {'-'*6}")
        for fi, fname in enumerate(feat_names):
            s_mean = X_set[:, fi].mean()
            b_mean = X_base[:, fi].mean()
            ratio = s_mean / b_mean if abs(b_mean) > 1e-6 else 0
            flag = " **" if abs(ratio - 1) > 0.15 else ""
            print(f"  {fname:<20} {s_mean:>9.3f} {b_mean:>10.3f} {ratio:>5.2f}x{flag}")

        return ct_counts

    loss_cts = profile(loss_idx, "GRAPH LOSSES (V6c better)")
    win_cts = profile(win_idx, "GRAPH WINS (Graph better)")

    # =========================================================================
    # DIRECT COMPARISON: What's different about losses vs wins?
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  LOSS vs WIN: KEY DIFFERENCES")
    print(f"{'='*90}")

    loss_muts = np.array([len(patient_genes.get(i, set())) for i in loss_idx])
    win_muts = np.array([len(patient_genes.get(i, set())) for i in win_idx])
    loss_chs = np.array([len(patient_channels.get(i, set())) for i in loss_idx])
    win_chs = np.array([len(patient_channels.get(i, set())) for i in win_idx])

    print(f"\n  {'Metric':<30} {'Losses':>10} {'Wins':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Mutations (mean)':<30} {loss_muts.mean():>10.1f} {win_muts.mean():>10.1f}")
    print(f"  {'Channels hit (mean)':<30} {loss_chs.mean():>10.1f} {win_chs.mean():>10.1f}")
    print(f"  {'Age (mean)':<30} {age[loss_idx].mean():>10.1f} {age[win_idx].mean():>10.1f}")
    print(f"  {'Event rate':<30} {events[loss_idx].mean():>10.1%} {events[win_idx].mean():>10.1%}")
    print(f"  {'Survival (median mo)':<30} {np.median(times[loss_idx]):>10.1f} {np.median(times[win_idx]):>10.1f}")
    print(f"  {'TMB (mean)':<30} {tmb[loss_idx].mean():>10.1f} {tmb[win_idx].mean():>10.1f}")
    print(f"  {'MSI high (%)':<30} {msi_high[loss_idx].mean():>10.1%} {msi_high[win_idx].mean():>10.1%}")

    # Graph features
    feat_names = [
        "ct_baseline", "ct_ch_defense", "ct_gene_shift", "global_gene_shift",
        "n_channels_hit", "total_isolation", "total_hub_damage", "channels_severed",
        "tier_conn", "n_mutated", "n_cooccur_pairs", "total_cooccur_wt",
        "max_cooccur_wt", "cross_ch_cooccur",
    ]
    print(f"\n  {'Graph Feature':<25} {'Losses':>10} {'Wins':>10} {'L/W Ratio':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for fi, fname in enumerate(feat_names):
        l_mean = X_all[loss_idx, fi].mean()
        w_mean = X_all[win_idx, fi].mean()
        ratio = l_mean / w_mean if abs(w_mean) > 1e-6 else 0
        flag = " **" if abs(ratio - 1) > 0.15 else ""
        print(f"  {fname:<25} {l_mean:>10.3f} {w_mean:>10.3f} {ratio:>9.2f}x{flag}")

    # =========================================================================
    # Error magnitude: worst graph misses
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  WORST GRAPH MISSES (largest rank error where V6c was right)")
    print(f"{'='*90}")

    # Patients where graph error >> v6c error
    error_diff = graph_error - v6c_error  # positive = graph worse
    worst_idx = valid_idx[np.argsort(-error_diff[valid_idx])[:30]]

    print(f"\n  {'Idx':>6} {'CT':<25} {'Muts':>4} {'Chs':>3} {'Age':>4} {'Time':>6} {'Ev':>3} "
          f"{'GrRnk':>6} {'V6Rnk':>6} {'ActRnk':>6} {'GrErr':>6} {'V6Err':>6}")
    print(f"  {'-'*6} {'-'*25} {'-'*4} {'-'*3} {'-'*4} {'-'*6} {'-'*3} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for idx in worst_idx:
        ct = ct_per_patient[idx]
        nm = len(patient_genes.get(idx, set()))
        nc = len(patient_channels.get(idx, set()))
        a = age[idx]
        t = times[idx]
        e = events[idx]
        gr = graph_rank[idx]
        vr = v6c_rank[idx]
        ar = actual_rank[idx]
        ge = graph_error[idx]
        ve = v6c_error[idx]
        genes = sorted(patient_genes.get(idx, set()))[:5]
        gene_str = ",".join(genes)
        if len(patient_genes.get(idx, set())) > 5:
            gene_str += f"+{len(patient_genes.get(idx, set()))-5}"
        print(f"  {idx:>6} {ct:<25} {nm:>4} {nc:>3} {a:>4.0f} {t:>6.1f} {e:>3} "
              f"{gr:>6.3f} {vr:>6.3f} {ar:>6.3f} {ge:>6.3f} {ve:>6.3f}  {gene_str}")

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
