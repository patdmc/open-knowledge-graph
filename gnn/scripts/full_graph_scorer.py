#!/usr/bin/env python3
"""
Full graph scorer: encode ALL information as graph edge/node attributes
and score via a single traversal.

Node attributes:
  - CancerType: baseline_hazard
  - Channel: is_severed, n_mutations, n_genes, frac_trunc, frac_miss, mean_vaf, max_vaf, gof, lof
  - Gene: shift (cancer-type-specific), hub/leaf, GOF/LOF, betweenness
  - Patient: age, sex, msi, tmb

Edge attributes:
  - cancer_type -> channel: defense_shift (per CT)
  - channel -> gene: containment within channel subgraph
  - gene -> gene (PPI): interaction weight
  - hotspot -> hotspot: directional synergy

Traversal:
  1. Start at cancer_type node, read baseline
  2. Walk to each mutated channel, read CT-specific defense_shift
  3. Within channel: compute isolation, hub damage from PPI subgraph
  4. Check tier connectivity
  5. Add clinical features
  6. Sum with learned (ridge) weights

Usage:
    python3 -u -m gnn.scripts.full_graph_scorer
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict

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

SAVE_BASE = os.path.join(GNN_RESULTS, "full_graph_scorer")


def build_ppi_graph():
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


def graph_walk(G, patient_genes, channel_features, tier_features,
               ct_name, ct_ch_shift, ct_gene_shift, global_gene_shift,
               age, sex, msi, msi_high, tmb, ct_baseline):
    """
    Single graph traversal for one patient. Returns a feature vector.

    Walk:
      1. Cancer type node -> baseline hazard
      2. For each mutated channel:
         a. CT x channel defense shift (edge weight)
         b. Channel node features: is_severed, n_mut, frac_trunc, vaf, gof, lof
         c. Per-gene shifts (CT-specific with global fallback)
         d. Isolation: remove mutated genes, measure hub reachability
      3. Tier connectivity: can signals still escalate?
      4. Clinical: age, sex, msi, tmb
    """
    mutated = patient_genes & set(V6_CHANNEL_MAP.keys())

    features = {}

    # --- 1. Cancer type baseline ---
    features["ct_baseline"] = ct_baseline

    # --- 2. Per-channel walk ---
    ct_ch_defense_sum = 0.0
    ct_gene_shift_sum = 0.0
    global_gene_shift_sum = 0.0
    channel_feat_sum = np.zeros(9)  # 9 features per channel
    n_channels_hit = 0
    total_isolation = 0.0
    total_hub_damage = 0.0
    channels_severed = 0

    for ci, ch_name in enumerate(V6_CHANNEL_NAMES):
        ch_genes = [g for g, c in V6_CHANNEL_MAP.items() if c == ch_name]
        ch_mutated = mutated & set(ch_genes)

        if not ch_mutated:
            continue

        n_channels_hit += 1

        # 2a. CT x channel edge weight
        ct_shift = ct_ch_shift.get((ct_name, ch_name), 0.0)
        ct_ch_defense_sum += ct_shift

        # 2b. Channel node features (from the V6c feature tensor)
        channel_feat_sum += channel_features[ci]

        # 2c. Per-gene shifts
        for g in ch_mutated:
            ct_s = ct_gene_shift.get((ct_name, g), None)
            if ct_s is not None:
                ct_gene_shift_sum += ct_s
            gl_s = global_gene_shift.get(g, None)
            if gl_s is not None:
                global_gene_shift_sum += gl_s

        # 2d. Isolation from PPI subgraph
        ch_hubs = HUB_GENES.get(ch_name, set())
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

    # --- 3. Tier connectivity ---
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

    # Tier features from V6c
    tier_feat_flat = tier_features.flatten()  # 4 tiers x 5 features = 20

    # --- 4. Assemble feature vector ---
    # Graph-derived features
    graph_feats = np.array([
        ct_baseline,
        ct_ch_defense_sum,
        ct_gene_shift_sum,
        global_gene_shift_sum,
        n_channels_hit,
        total_isolation,
        total_hub_damage,
        channels_severed,
        tier_conn,
        len(mutated),
    ])

    # Channel features (aggregated across mutated channels)
    # Also include the raw per-channel features for all 8 channels
    all_ch_feats = channel_features.flatten()  # 8 x 9 = 72

    # Clinical
    clinical = np.array([age, sex, msi, msi_high, tmb])

    return np.concatenate([graph_feats, all_ch_feats, tier_feat_flat, clinical])


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  FULL GRAPH SCORER")
    print("  All information encoded as graph attributes, scored via traversal")
    print("=" * 90)

    G = build_ppi_graph()

    # Load data
    print("\n  Loading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    channel_features = data["channel_features"].numpy()  # N x 8 x 9
    tier_features = data["tier_features"].numpy()  # N x 4 x 5
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

    patient_genes = defaultdict(set)
    patient_channels = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            g = row["gene.hugoGeneSymbol"]
            patient_genes[idx].add(g)
            patient_channels[idx].add(V6_CHANNEL_MAP.get(g, "?"))

    # Compute shifts
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

    ct_patients = defaultdict(set)
    ct_ch_patients = defaultdict(lambda: defaultdict(set))
    ct_gene_patients_map = defaultdict(lambda: defaultdict(set))
    for idx in range(N):
        ct_name = ct_vocab[int(ct_idx_arr[idx])] if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other"
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
        for ch_name in set(V6_CHANNEL_MAP.values()):
            pts = ct_ch_patients[ct_name].get(ch_name, set())
            pts_arr = np.array(sorted(pts))
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

    # =========================================================================
    # Graph walk for all patients
    # =========================================================================
    print("\n  Running graph walk for all patients...")

    # First pass: determine feature vector size
    sample = graph_walk(
        G, patient_genes.get(0, set()), channel_features[0], tier_features[0],
        ct_vocab[0], ct_ch_shift, ct_gene_shift, global_gene_shift,
        age[0], sex[0], msi[0], msi_high[0], tmb[0],
        ct_baseline_map.get(ct_vocab[0], baseline),
    )
    feat_dim = len(sample)
    print(f"  Feature vector dimension: {feat_dim}")

    X_all = np.zeros((N, feat_dim))
    for idx in range(N):
        ct_name = ct_vocab[int(ct_idx_arr[idx])] if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other"
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        X_all[idx] = graph_walk(
            G, patient_genes.get(idx, set()),
            channel_features[idx], tier_features[idx],
            ct_name, ct_ch_shift, ct_gene_shift, global_gene_shift,
            age[idx], sex[idx], msi[idx], msi_high[idx], tmb[idx],
            ct_bl,
        )
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

    print(f"    Done.")

    # =========================================================================
    # Feature groups for ablation
    # =========================================================================
    folds = list(skf.split(np.arange(N), events))

    # Feature indices
    # graph_feats: 0-9 (10)
    # channel_feats: 10-81 (72)
    # tier_feats: 82-101 (20)
    # clinical: 102-106 (5)
    graph_idx = list(range(0, 10))
    ch_idx = list(range(10, 82))
    tier_idx = list(range(82, 102))
    clin_idx = list(range(102, 107))

    configs = {
        "A graph_walk only":            graph_idx,
        "B graph + clinical":           graph_idx + clin_idx,
        "C graph + channel_feats":      graph_idx + ch_idx,
        "D graph + ch + tier":          graph_idx + ch_idx + tier_idx,
        "E graph + ch + tier + clin":   graph_idx + ch_idx + tier_idx + clin_idx,
        "F ch + tier + clin (no graph)": ch_idx + tier_idx + clin_idx,
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION: WHAT DOES THE GRAPH WALK ADD?")
    print(f"{'='*90}")

    print(f"\n  {'Scorer':<40} {'Mean CI':>8} {'Std':>7}  Folds")
    print(f"  {'-'*40} {'-'*8} {'-'*7}  {'-'*35}")

    results = {}
    for name, feat_indices in configs.items():
        X = X_all[:, feat_indices]
        score = np.zeros(N)
        last_reg = None
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            vt = valid_mask[train_idx]
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx][vt], hazards[train_idx][vt])
            score[val_idx] = reg.predict(X[val_idx])
            last_reg = reg

        fold_cis = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ci = concordance_index(
                torch.tensor(score[val_idx].astype(np.float32)),
                torch.tensor(times[val_idx].astype(np.float32)),
                torch.tensor(events[val_idx].astype(np.float32)),
            )
            fold_cis.append(ci)

        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        results[name] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis, "score": score}
        folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
        print(f"  {name:<40} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    best = max(r["mean"] for r in results.values())
    best_name = max(results, key=lambda k: results[k]["mean"])
    no_graph = results["F ch + tier + clin (no graph)"]["mean"]
    with_graph = results["E graph + ch + tier + clin"]["mean"]

    print(f"\n  V6c transformer:                           {v6c:.4f}")
    print(f"  Best graph scorer:                         {best:.4f} ({best_name})")
    print(f"  Gap to V6c:                                {v6c - best:+.4f}")
    print(f"  Gap closed (from 0.538 baseline):          {(best - 0.5381) / (v6c - 0.5381) * 100:.1f}%")
    print(f"\n  Without graph walk (F):                    {no_graph:.4f}")
    print(f"  With graph walk (E):                       {with_graph:.4f}")
    print(f"  Graph walk adds:                           {with_graph - no_graph:+.4f}")

    # =========================================================================
    # Per-cancer-type for the best scorer
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE: {best_name}")
    print(f"{'='*90}")

    best_score = results[best_name]["score"]

    print(f"\n  {'Cancer Type':<30} {'N':>6} {'Graph':>7} {'V6c':>7} {'Delta':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([
            (int(ct_idx_arr[i]) < len(ct_vocab) and ct_vocab[int(ct_idx_arr[i])] == ct_name)
            for i in range(N)
        ])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 100 or events[ct_indices].sum() < 10:
            continue

        ci_graph = concordance_index(
            torch.tensor(best_score[ct_indices].astype(np.float32)),
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
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump({k: {"mean": float(v["mean"]), "std": float(v["std"])}
                   for k, v in results.items()}, f, indent=2)

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
