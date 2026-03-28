#!/usr/bin/env python3
"""
Cancer-type-specific graph traversal scorer.

Instead of using global gene shifts, compute per-cancer-type gene shifts
and per-cancer-type channel defense weights. Then feed those into the
ridge traversal scorer.

This is the cancer_type_modulates_channel edge from EV03, encoded as
a scoring function.

Usage:
    python3 -u -m gnn.scripts.cancer_type_traversal
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_POSITION, GENE_FUNCTION, TRUNCATING,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

SAVE_BASE = os.path.join(GNN_RESULTS, "cancer_type_traversal")
MIN_CT_PATIENTS = 50  # minimum patients per cancer type for specific weights


def build_graph():
    with open(os.path.join(GNN_CACHE, "string_ppi_edges.json")) as f:
        ppi = json.load(f)
    G = nx.Graph()
    for gene in V6_CHANNEL_MAP:
        G.add_node(gene, channel=V6_CHANNEL_MAP[gene],
                   tier=V6_TIER_MAP.get(V6_CHANNEL_MAP[gene], -1))
    for ch_name in ppi:
        for src, tgt, score in ppi[ch_name]:
            if src in V6_CHANNEL_MAP and tgt in V6_CHANNEL_MAP:
                G.add_edge(src, tgt, weight=score / 1000.0)
    return G


def cascade_features(G, mutated_genes):
    """Compute traversal features for a set of mutated genes."""
    mutated = set(mutated_genes) & set(V6_CHANNEL_MAP.keys())
    if not mutated:
        return {
            "total_isolation": 0, "total_hub_damage": 0,
            "channels_severed": 0, "tier_connectivity": 1.0,
            "lcc_fraction": 1.0, "n_mutated": 0,
        }

    total_isolation = 0
    total_hub_damage = 0
    channels_severed = 0

    for ch_name in V6_CHANNEL_NAMES:
        ch_genes = [g for g, c in V6_CHANNEL_MAP.items() if c == ch_name]
        ch_hubs = HUB_GENES.get(ch_name, set())
        ch_mutated = mutated & set(ch_genes)
        if not ch_mutated:
            continue

        G_ch = G.subgraph(ch_genes).copy()
        G_d = G_ch.copy()
        G_d.remove_nodes_from(ch_mutated)

        surviving_hubs = ch_hubs - ch_mutated
        reachable = set()
        for hub in surviving_hubs:
            if hub in G_d:
                reachable |= set(nx.descendants(G_d, hub)) | {hub}

        surviving = set(ch_genes) - ch_mutated
        iso_frac = len(surviving - reachable) / max(len(surviving), 1)
        total_isolation += iso_frac

        hub_hit = len(ch_mutated & ch_hubs)
        total_hub_damage += hub_hit / max(len(ch_hubs), 1)
        if len(ch_mutated) >= 2 or hub_hit > 0:
            channels_severed += 1

    G_d = G.copy()
    G_d.remove_nodes_from(mutated)

    tier_genes = defaultdict(set)
    for g, ch in V6_CHANNEL_MAP.items():
        if g not in mutated:
            tier_genes[V6_TIER_MAP.get(ch, -1)].add(g)

    tier_connected = 0
    tier_tested = 0
    for t1 in range(4):
        for t2 in range(t1 + 1, 4):
            found = False
            for g1 in sorted(tier_genes.get(t1, set()))[:5]:
                for g2 in sorted(tier_genes.get(t2, set()))[:5]:
                    if g1 in G_d and g2 in G_d and nx.has_path(G_d, g1, g2):
                        found = True
                        break
                if found:
                    break
            tier_tested += 1
            if found:
                tier_connected += 1

    tier_conn = tier_connected / max(tier_tested, 1)

    if G_d.number_of_nodes() > 0:
        lcc = max(nx.connected_components(G_d), key=len)
        lcc_frac = len(lcc) / G.number_of_nodes()
    else:
        lcc_frac = 0

    return {
        "total_isolation": total_isolation,
        "total_hub_damage": total_hub_damage,
        "channels_severed": channels_severed,
        "tier_connectivity": tier_conn,
        "lcc_fraction": lcc_frac,
        "n_mutated": len(mutated),
    }


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  CANCER-TYPE-SPECIFIC GRAPH TRAVERSAL SCORER")
    print("  Apply per-cancer-type channel weights to close the gap")
    print("=" * 90)

    G = build_graph()

    # Load data
    print("\n  Loading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    print(f"  {all_in_val.sum().item()} patients")

    # Load mutations
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

    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mut[mut["gene.hugoGeneSymbol"].isin(channel_genes)].copy()
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)
    mut["is_truncating"] = mut["mutationType"].isin(TRUNCATING).astype(int)

    # Per-patient gene sets + channel sets
    patient_genes = defaultdict(set)
    patient_channels = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            g = row["gene.hugoGeneSymbol"]
            patient_genes[idx].add(g)
            patient_channels[idx].add(V6_CHANNEL_MAP.get(g, "?"))

    # =========================================================================
    # Compute per-cancer-type, per-channel hazard shifts
    # =========================================================================
    print("\n  Computing cancer-type x channel defense matrix...")

    # For each (cancer_type, channel): mean hazard of patients with that
    # cancer type who have a mutation in that channel
    ct_ch_patients = defaultdict(lambda: defaultdict(set))
    ct_patients = defaultdict(set)

    for idx in range(N):
        ct_name = ct_vocab[int(ct_idx[idx])] if int(ct_idx[idx]) < len(ct_vocab) else "Other"
        ct_patients[ct_name].add(idx)
        for ch in patient_channels.get(idx, set()):
            ct_ch_patients[ct_name][ch].add(idx)

    # Per-cancer-type baseline
    ct_baseline = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= MIN_CT_PATIENTS:
            ct_baseline[ct_name] = float(hazards[pts_valid].mean())

    # Per-cancer-type, per-channel shift
    ct_ch_shift = {}
    for ct_name in ct_baseline:
        bl = ct_baseline[ct_name]
        for ch_name in V6_CHANNEL_NAMES:
            pts = ct_ch_patients[ct_name].get(ch_name, set())
            pts_arr = np.array(sorted(pts))
            if len(pts_arr) == 0:
                continue
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 20:
                shift = float(hazards[pts_valid].mean() - bl)
                ct_ch_shift[(ct_name, ch_name)] = shift

    # Also compute per-cancer-type gene shifts
    ct_gene_patients = defaultdict(lambda: defaultdict(set))
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            g = row["gene.hugoGeneSymbol"]
            ct_name = ct_vocab[int(ct_idx[idx])] if int(ct_idx[idx]) < len(ct_vocab) else "Other"
            ct_gene_patients[ct_name][g].add(idx)

    ct_gene_shift = {}
    for ct_name in ct_baseline:
        bl = ct_baseline[ct_name]
        for gene, pts in ct_gene_patients[ct_name].items():
            pts_arr = np.array(sorted(pts))
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 15:
                ct_gene_shift[(ct_name, gene)] = float(hazards[pts_valid].mean() - bl)

    # Global gene shifts (fallback)
    gene_patients_map = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)
    global_gene_shift = {}
    for gene, pts in gene_patients_map.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            global_gene_shift[gene] = float(hazards[pts_valid].mean() - baseline)

    # Print the matrix for top cancer types
    print(f"\n  Cancer Type x Channel Defense Shifts:")
    print(f"  {'Cancer Type':<30} " + "  ".join(f"{ch:>10}" for ch in V6_CHANNEL_NAMES))
    print(f"  {'-'*30} " + "  ".join("-" * 10 for _ in V6_CHANNEL_NAMES))

    for ct_name in sorted(ct_baseline, key=lambda x: -len(ct_patients[x])):
        if len(ct_patients[ct_name]) < 500:
            continue
        vals = []
        for ch in V6_CHANNEL_NAMES:
            s = ct_ch_shift.get((ct_name, ch), None)
            vals.append(f"{s:>+10.3f}" if s is not None else f"{'n/a':>10}")
        print(f"  {ct_name:<30} " + "  ".join(vals))

    # Show some dramatic cancer-type-specific gene shifts
    print(f"\n  Cancer-type-specific gene shifts (biggest divergences from global):")
    print(f"  {'Cancer Type':<25} {'Gene':<12} {'CT shift':>8} {'Global':>8} {'Delta':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

    divergences = []
    for (ct, gene), ct_s in ct_gene_shift.items():
        gl_s = global_gene_shift.get(gene, None)
        if gl_s is not None:
            delta = ct_s - gl_s
            divergences.append((ct, gene, ct_s, gl_s, delta))
    divergences.sort(key=lambda x: -abs(x[4]))

    for ct, gene, ct_s, gl_s, delta in divergences[:25]:
        print(f"  {ct:<25} {gene:<12} {ct_s:>+8.3f} {gl_s:>+8.3f} {delta:>+8.3f}")

    # =========================================================================
    # Build features: global + cancer-type-specific
    # =========================================================================
    print(f"\n  Computing per-patient features...")

    trav_feat_names = [
        "total_isolation", "total_hub_damage", "channels_severed",
        "tier_connectivity", "lcc_fraction", "n_mutated",
    ]
    trav_features = np.zeros((N, len(trav_feat_names)))

    for idx in range(N):
        genes = patient_genes.get(idx, set())
        if not genes:
            trav_features[idx, 3] = 1.0  # tier_connectivity
            trav_features[idx, 4] = 1.0  # lcc_fraction
            continue
        feats = cascade_features(G, genes)
        for fi, fn in enumerate(trav_feat_names):
            trav_features[idx, fi] = feats[fn]
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

    print(f"    Done.")

    # Gene shift features: global and cancer-type-specific
    global_gene_score = np.zeros(N)
    ct_gene_score = np.zeros(N)
    ct_channel_score = np.zeros(N)

    for idx in range(N):
        ct_name = ct_vocab[int(ct_idx[idx])] if int(ct_idx[idx]) < len(ct_vocab) else "Other"
        genes = patient_genes.get(idx, set())

        for g in genes:
            # Global
            if g in global_gene_shift:
                global_gene_score[idx] += global_gene_shift[g]

            # Cancer-type-specific gene shift (use if available, else global)
            ct_s = ct_gene_shift.get((ct_name, g), None)
            if ct_s is not None:
                ct_gene_score[idx] += ct_s
            elif g in global_gene_shift:
                ct_gene_score[idx] += global_gene_shift[g]

        # Cancer-type-specific channel score
        for ch in patient_channels.get(idx, set()):
            ct_s = ct_ch_shift.get((ct_name, ch), None)
            if ct_s is not None:
                ct_channel_score[idx] += ct_s

    # =========================================================================
    # Scoring functions with 5-fold CV
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  SCORING FUNCTIONS")
    print(f"{'='*90}")

    folds = list(skf.split(np.arange(N), events))
    scores = {}

    # A: global gene shift (baseline)
    scores["A_global_gene"] = global_gene_score

    # B: cancer-type-specific gene shift
    scores["B_ct_gene"] = ct_gene_score

    # C: cancer-type channel shift
    scores["C_ct_channel"] = ct_channel_score

    # D: ridge on global gene + traversal (previous best)
    X_global = np.column_stack([global_gene_score, trav_features])
    score_d = np.zeros(N)
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        valid_train = valid_mask[train_idx]
        reg = Ridge(alpha=1.0)
        reg.fit(X_global[train_idx][valid_train], hazards[train_idx][valid_train])
        score_d[val_idx] = reg.predict(X_global[val_idx])
    scores["D_ridge_global"] = score_d

    # E: ridge on CT gene + traversal
    X_ct_gene = np.column_stack([ct_gene_score, trav_features])
    score_e = np.zeros(N)
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        valid_train = valid_mask[train_idx]
        reg = Ridge(alpha=1.0)
        reg.fit(X_ct_gene[train_idx][valid_train], hazards[train_idx][valid_train])
        score_e[val_idx] = reg.predict(X_ct_gene[val_idx])
    scores["E_ridge_ct_gene"] = score_e

    # F: ridge on CT gene + CT channel + traversal
    X_ct_full = np.column_stack([ct_gene_score, ct_channel_score, trav_features])
    score_f = np.zeros(N)
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        valid_train = valid_mask[train_idx]
        reg = Ridge(alpha=1.0)
        reg.fit(X_ct_full[train_idx][valid_train], hazards[train_idx][valid_train])
        score_f[val_idx] = reg.predict(X_ct_full[val_idx])
    scores["F_ridge_ct_full"] = score_f

    # G: ridge on everything: global gene + CT gene + CT channel + traversal
    X_all = np.column_stack([
        global_gene_score, ct_gene_score, ct_channel_score, trav_features,
    ])
    score_g = np.zeros(N)
    last_reg = None
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        valid_train = valid_mask[train_idx]
        reg = Ridge(alpha=1.0)
        reg.fit(X_all[train_idx][valid_train], hazards[train_idx][valid_train])
        score_g[val_idx] = reg.predict(X_all[val_idx])
        last_reg = reg
    scores["G_ridge_all"] = score_g

    # Evaluate
    print(f"\n  {'Scorer':<30} {'Mean CI':>8} {'Std':>7}  Fold CIs")
    print(f"  {'-'*30} {'-'*8} {'-'*7}  {'-'*40}")

    results = {}
    for name in ["A_global_gene", "B_ct_gene", "C_ct_channel",
                  "D_ridge_global", "E_ridge_ct_gene", "F_ridge_ct_full",
                  "G_ridge_all"]:
        s = scores[name]
        fold_cis = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ci = concordance_index(
                torch.tensor(s[val_idx].astype(np.float32)),
                torch.tensor(times[val_idx].astype(np.float32)),
                torch.tensor(events[val_idx].astype(np.float32)),
            )
            fold_cis.append(ci)
        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        results[name] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis}
        folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
        print(f"  {name:<30} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # Print ridge coefficients
    feat_labels = ["global_gene", "ct_gene", "ct_channel"] + trav_feat_names
    if last_reg is not None:
        print(f"\n  Ridge coefficients (G_ridge_all, last fold):")
        for fi, fn in enumerate(feat_labels):
            print(f"    {fn:<25} {last_reg.coef_[fi]:>+8.4f}")

    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    if os.path.exists(ua_path):
        with open(ua_path) as f:
            ua = json.load(f)
        v6c_mean = ua["means"].get("V6c", 0)
        best = max(r["mean"] for r in results.values())
        best_name = max(results, key=lambda k: results[k]["mean"])
        print(f"\n  V6c transformer:     {v6c_mean:.4f}")
        print(f"  Best graph scorer:   {best:.4f} ({best_name})")
        print(f"  Gap:                 {v6c_mean - best:+.4f}")
        print(f"  Gap closed:          {(best - 0.5381) / (v6c_mean - 0.5381) * 100:.1f}%")

    # =========================================================================
    # Per-cancer-type comparison
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE C-INDEX: G_ridge_all vs V6c")
    print(f"{'='*90}")

    print(f"\n  {'Cancer Type':<30} {'N':>6} {'Graph':>7} {'V6c':>7} {'Delta':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([
            (int(ct_idx[i]) < len(ct_vocab) and ct_vocab[int(ct_idx[i])] == ct_name)
            for i in range(N)
        ])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 100 or events[ct_indices].sum() < 10:
            continue

        ci_graph = concordance_index(
            torch.tensor(score_g[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v6c = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        delta = ci_graph - ci_v6c
        marker = " <-- BEATS" if delta > 0.005 else ""
        print(f"  {ct_name:<30} {len(ct_indices):>6} {ci_graph:>7.4f} {ci_v6c:>7.4f} "
              f"{delta:>+7.4f}{marker}")

    # Save
    # Convert ct_ch_shift to serializable format
    ct_ch_json = {f"{ct}|{ch}": float(v) for (ct, ch), v in ct_ch_shift.items()}
    with open(os.path.join(SAVE_BASE, "ct_channel_matrix.json"), "w") as f:
        json.dump(ct_ch_json, f, indent=2)

    with open(os.path.join(SAVE_BASE, "scoring_results.json"), "w") as f:
        json.dump({k: {"mean": float(v["mean"]), "std": float(v["std"])}
                   for k, v in results.items()}, f, indent=2)

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
