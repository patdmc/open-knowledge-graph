#!/usr/bin/env python3
"""
Graph distance between mutation pairs predicts synergy direction.

Hypothesis: for two mutations in the same patient:
  1. Graph distance between them predicts whether they compound or rescue
  2. Relative position (which is closer to hub = "older") predicts direction
  3. Close pairs in same subgraph = interact; distant pairs = independent

Usage:
    python3 -u -m gnn.scripts.graph_distance_pairs
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES, GENE_POSITION, TRUNCATING,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "graph_distance_pairs")
MIN_HOTSPOT_PATIENTS = 30
MIN_CO_OCCUR = 15


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    # =========================================================================
    # Build PPI graph
    # =========================================================================
    print("=" * 90)
    print("  GRAPH DISTANCE BETWEEN MUTATION PAIRS")
    print("=" * 90)

    with open(os.path.join(GNN_CACHE, "string_ppi_edges.json")) as f:
        ppi = json.load(f)

    G_full = nx.Graph()
    for gene in V6_CHANNEL_MAP:
        G_full.add_node(gene)
    for ch_name in ppi:
        for src, tgt, score in ppi[ch_name]:
            if src in V6_CHANNEL_MAP and tgt in V6_CHANNEL_MAP:
                G_full.add_edge(src, tgt, weight=score / 1000.0)

    # Precompute all-pairs shortest path
    print("\n  Computing all-pairs shortest paths...")
    all_paths = dict(nx.all_pairs_shortest_path_length(G_full))

    def graph_dist(g1, g2):
        if g1 == g2:
            return 0
        if g1 in all_paths and g2 in all_paths[g1]:
            return all_paths[g1][g2]
        return -1  # disconnected

    # Distance to nearest hub
    all_hubs = set()
    for ch, hubs in HUB_GENES.items():
        all_hubs |= hubs

    gene_to_hub_dist = {}
    for gene in V6_CHANNEL_MAP:
        ch = V6_CHANNEL_MAP[gene]
        ch_hubs = HUB_GENES.get(ch, set())
        min_d = float("inf")
        for hub in ch_hubs:
            d = graph_dist(gene, hub)
            if d >= 0:
                min_d = min(min_d, d)
        gene_to_hub_dist[gene] = min_d if min_d < float("inf") else -1

    # =========================================================================
    # Load data + predictions
    # =========================================================================
    print("  Loading data + predictions...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(data["cancer_type_vocab"]),
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
                "age", "sex", "msi_score", "msi_high", "tmb"
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
    mut = mut[
        mut["gene.hugoGeneSymbol"].isin(channel_genes)
        & mut["proteinChange"].notna()
        & (mut["proteinChange"] != "")
    ].copy()
    mut["hotspot"] = mut["gene.hugoGeneSymbol"] + " " + mut["proteinChange"].astype(str)
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)

    hotspot_freq = mut.groupby("hotspot")["patient_idx"].nunique()
    valid_hotspots = hotspot_freq[hotspot_freq >= MIN_HOTSPOT_PATIENTS].index
    mut_valid = mut[mut["hotspot"].isin(valid_hotspots)]

    # Per-hotspot data
    hotspot_patients = {}
    hotspot_shift = {}
    hotspot_gene = {}
    for hs, group in mut_valid.groupby("hotspot"):
        pts = set(group["patient_idx"].unique())
        hotspot_patients[hs] = pts
        hotspot_gene[hs] = group.iloc[0]["gene.hugoGeneSymbol"]
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 10:
            hotspot_shift[hs] = float(hazards[pts_valid].mean() - baseline)

    hotspot_list = sorted(hotspot_shift.keys())
    freq_lookup = hotspot_freq.to_dict()

    # =========================================================================
    # Build pairs with graph distance
    # =========================================================================
    print("\n  Building mutation pairs with graph distances...")

    pairs = []
    for i in range(len(hotspot_list)):
        ha = hotspot_list[i]
        ga = hotspot_gene[ha]
        pts_a = hotspot_patients[ha]
        fa = freq_lookup.get(ha, 0)

        for j in range(i + 1, len(hotspot_list)):
            hb = hotspot_list[j]
            gb = hotspot_gene[hb]
            pts_b = hotspot_patients[hb]
            fb = freq_lookup.get(hb, 0)

            co = pts_a & pts_b
            co_valid = np.array(sorted(co))
            if len(co_valid) == 0:
                continue
            co_valid = co_valid[valid_mask[co_valid]]
            if len(co_valid) < MIN_CO_OCCUR:
                continue

            sa = hotspot_shift[ha]
            sb = hotspot_shift[hb]
            actual = float(hazards[co_valid].mean() - baseline)
            expected = sa + sb
            synergy = actual - expected

            # Graph distance between the two genes
            dist = graph_dist(ga, gb)

            # Hub distances
            hub_dist_a = gene_to_hub_dist.get(ga, -1)
            hub_dist_b = gene_to_hub_dist.get(gb, -1)

            # Which is "older" (closer to hub)?
            if hub_dist_a >= 0 and hub_dist_b >= 0:
                if hub_dist_a < hub_dist_b:
                    older, newer = ha, hb
                    older_gene, newer_gene = ga, gb
                    older_shift, newer_shift = sa, sb
                    older_hub_dist, newer_hub_dist = hub_dist_a, hub_dist_b
                elif hub_dist_b < hub_dist_a:
                    older, newer = hb, ha
                    older_gene, newer_gene = gb, ga
                    older_shift, newer_shift = sb, sa
                    older_hub_dist, newer_hub_dist = hub_dist_b, hub_dist_a
                else:
                    # Same hub distance — use frequency as tiebreaker
                    if fa >= fb:
                        older, newer = ha, hb
                        older_gene, newer_gene = ga, gb
                        older_shift, newer_shift = sa, sb
                        older_hub_dist, newer_hub_dist = hub_dist_a, hub_dist_b
                    else:
                        older, newer = hb, ha
                        older_gene, newer_gene = gb, ga
                        older_shift, newer_shift = sb, sa
                        older_hub_dist, newer_hub_dist = hub_dist_b, hub_dist_a
            else:
                # Can't determine — use frequency
                if fa >= fb:
                    older, newer = ha, hb
                    older_gene, newer_gene = ga, gb
                    older_shift, newer_shift = sa, sb
                    older_hub_dist, newer_hub_dist = hub_dist_a, hub_dist_b
                else:
                    older, newer = hb, ha
                    older_gene, newer_gene = gb, ga
                    older_shift, newer_shift = sb, sa
                    older_hub_dist, newer_hub_dist = hub_dist_b, hub_dist_a

            hub_dist_diff = newer_hub_dist - older_hub_dist if (older_hub_dist >= 0 and newer_hub_dist >= 0) else None

            ch_a = V6_CHANNEL_MAP.get(ga, "?")
            ch_b = V6_CHANNEL_MAP.get(gb, "?")
            tier_a = V6_TIER_MAP.get(ch_a, -1)
            tier_b = V6_TIER_MAP.get(ch_b, -1)

            pairs.append({
                "older": older, "newer": newer,
                "older_gene": older_gene, "newer_gene": newer_gene,
                "older_shift": older_shift, "newer_shift": newer_shift,
                "older_hub_dist": older_hub_dist, "newer_hub_dist": newer_hub_dist,
                "hub_dist_diff": hub_dist_diff,
                "graph_dist": dist,
                "expected": expected, "actual": actual, "synergy": synergy,
                "n_co_occur": len(co_valid),
                "same_gene": ga == gb,
                "same_channel": ch_a == ch_b,
                "tier_older": tier_a if older_gene == ga else tier_b,
                "tier_newer": tier_b if older_gene == ga else tier_a,
            })

    print(f"  {len(pairs)} pairs with graph distance data")

    # =========================================================================
    # Analysis 1: Graph distance vs synergy
    # =========================================================================
    print("\n" + "=" * 90)
    print("  GRAPH DISTANCE vs SYNERGY")
    print("  Does distance between mutations predict interaction type?")
    print("=" * 90)

    # Only pairs with valid graph distance
    connected = [p for p in pairs if p["graph_dist"] >= 0]
    disconnected = [p for p in pairs if p["graph_dist"] < 0]

    print(f"\n  Connected pairs: {len(connected)}")
    print(f"  Disconnected pairs: {len(disconnected)}")

    if connected:
        dists_arr = np.array([p["graph_dist"] for p in connected])
        syns_arr = np.array([p["synergy"] for p in connected])
        r, p_val = stats.pearsonr(dists_arr, syns_arr)
        print(f"\n  Graph distance vs synergy: r={r:+.4f}, p={p_val:.2e}")
        print(f"  (Positive r = farther apart = more multiplicative)")

        print(f"\n  Dist    N  Mean_syn   Pct_mult  Pct_rescue")
        print(f"  ----  ---  --------  ---------  ----------")
        for d in sorted(set(dists_arr)):
            mask = dists_arr == d
            d_syns = syns_arr[mask]
            pct_mult = 100 * np.mean(d_syns > 0.05)
            pct_resc = 100 * np.mean(d_syns < -0.05)
            print(f"  {int(d):>4}  {int(mask.sum()):>3}  {d_syns.mean():>+8.4f}  {pct_mult:>8.1f}%  {pct_resc:>9.1f}%")

    if disconnected:
        disc_syns = np.array([p["synergy"] for p in disconnected])
        pct_mult_disc = 100 * np.mean(disc_syns > 0.05)
        pct_resc_disc = 100 * np.mean(disc_syns < -0.05)
        print(f"  disc  {len(disconnected):>3}  {disc_syns.mean():>+8.4f}  {pct_mult_disc:>8.1f}%  {pct_resc_disc:>9.1f}%")

    # =========================================================================
    # Analysis 2: Relative hub distance (older vs newer) vs synergy direction
    # =========================================================================
    print("\n" + "=" * 90)
    print("  RELATIVE HUB DISTANCE vs SYNERGY")
    print("  When one mutation is 'older' (closer to hub), how does the pair behave?")
    print("=" * 90)

    pairs_with_diff = [p for p in pairs if p["hub_dist_diff"] is not None and p["hub_dist_diff"] > 0]
    print(f"\n  Pairs where mutations have DIFFERENT hub distances: {len(pairs_with_diff)}")

    if pairs_with_diff:
        diffs = np.array([p["hub_dist_diff"] for p in pairs_with_diff])
        syns = np.array([p["synergy"] for p in pairs_with_diff])
        older_shifts = np.array([p["older_shift"] for p in pairs_with_diff])
        newer_shifts = np.array([p["newer_shift"] for p in pairs_with_diff])

        r, p_val = stats.pearsonr(diffs, syns)
        print(f"  Hub distance difference vs synergy: r={r:+.4f}, p={p_val:.2e}")

        # Key test: is the older mutation more harmful as solo?
        older_more_harmful = np.sum(older_shifts > newer_shifts)
        newer_more_harmful = np.sum(newer_shifts > older_shifts)
        total_comp = older_more_harmful + newer_more_harmful
        print(f"\n  Older (closer to hub) more harmful solo: {older_more_harmful}/{total_comp} "
              f"({100*older_more_harmful/total_comp:.1f}%)")
        print(f"  Newer (farther from hub) more harmful solo: {newer_more_harmful}/{total_comp} "
              f"({100*newer_more_harmful/total_comp:.1f}%)")

        from scipy.stats import binomtest
        bt = binomtest(older_more_harmful, total_comp, 0.5)
        print(f"  Binomial test: p = {bt.pvalue:.4f}")

        # Correlation: older shift vs synergy
        r2, p2 = stats.pearsonr(older_shifts, syns)
        print(f"\n  Older mutation shift vs synergy: r={r2:+.4f}, p={p2:.2e}")
        print(f"  (Positive r = more harmful older mutation = more multiplicative pair)")

        # Correlation: newer shift vs synergy
        r3, p3 = stats.pearsonr(newer_shifts, syns)
        print(f"  Newer mutation shift vs synergy: r={r3:+.4f}, p={p3:.2e}")

        # Combined: does the DIRECTION matter?
        # If older is harmful and newer is protective, is that worse than the reverse?
        older_harm_newer_prot = [p for p in pairs_with_diff
                                 if p["older_shift"] > 0 and p["newer_shift"] < 0]
        older_prot_newer_harm = [p for p in pairs_with_diff
                                 if p["older_shift"] < 0 and p["newer_shift"] > 0]

        if older_harm_newer_prot and older_prot_newer_harm:
            mean_syn_oh = np.mean([p["synergy"] for p in older_harm_newer_prot])
            mean_syn_op = np.mean([p["synergy"] for p in older_prot_newer_harm])
            print(f"\n  Older harmful + Newer protective: N={len(older_harm_newer_prot)}, "
                  f"mean synergy={mean_syn_oh:+.4f}")
            print(f"  Older protective + Newer harmful: N={len(older_prot_newer_harm)}, "
                  f"mean synergy={mean_syn_op:+.4f}")
            print(f"  --> Direction matters: delta = {mean_syn_oh - mean_syn_op:+.4f}")

    # =========================================================================
    # Analysis 3: Same gene vs same channel vs cross-channel
    # =========================================================================
    print("\n" + "=" * 90)
    print("  GRAPH LOCALITY: SAME GENE vs SAME CHANNEL vs CROSS-CHANNEL")
    print("=" * 90)

    same_gene = [p for p in pairs if p["same_gene"]]
    same_ch = [p for p in pairs if p["same_channel"] and not p["same_gene"]]
    cross_ch = [p for p in pairs if not p["same_channel"]]

    for label, group in [("Same gene", same_gene), ("Same channel (diff gene)", same_ch),
                          ("Cross-channel", cross_ch)]:
        if not group:
            continue
        syns = np.array([p["synergy"] for p in group])
        pct_mult = 100 * np.mean(syns > 0.05)
        pct_resc = 100 * np.mean(syns < -0.05)
        print(f"\n  {label}: N={len(group)}")
        print(f"    Mean synergy: {syns.mean():+.4f}")
        print(f"    Multiplicative: {pct_mult:.1f}%, Rescue: {pct_resc:.1f}%")

        if group == cross_ch:
            # Break down cross-channel by tier direction
            up = [p for p in group if p["tier_newer"] > p["tier_older"]]
            down = [p for p in group if p["tier_newer"] < p["tier_older"]]
            same_tier = [p for p in group if p["tier_newer"] == p["tier_older"]]
            for tlabel, tgroup in [("  Up (newer higher tier)", up),
                                    ("  Down (newer lower tier)", down),
                                    ("  Same tier", same_tier)]:
                if tgroup:
                    ts = np.array([p["synergy"] for p in tgroup])
                    print(f"    {tlabel}: N={len(tgroup)}, mean syn={ts.mean():+.4f}, "
                          f"{100*np.mean(ts>0.05):.1f}% mult")

    # =========================================================================
    # Analysis 4: Build scoring function with graph distance
    # =========================================================================
    print("\n" + "=" * 90)
    print("  GRAPH-DISTANCE SCORING FUNCTION")
    print("=" * 90)

    # Per-patient hotspot sets
    patient_hotspots = defaultdict(set)
    for _, row in mut_valid.iterrows():
        patient_hotspots[int(row["patient_idx"])].add(row["hotspot"])

    # Per-patient gene sets
    patient_genes = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes[idx].add(row["gene.hugoGeneSymbol"])

    # Gene-level shifts
    gene_patients_map = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)

    gene_shift = {}
    for gene, pts in gene_patients_map.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            gene_shift[gene] = float(hazards[pts_valid].mean() - baseline)

    # Edge lookups
    pair_lookup = {}
    for p in pairs:
        key = tuple(sorted([p["older"], p["newer"]]))
        pair_lookup[key] = p

    dir_edges_path = os.path.join(GNN_RESULTS, "directional_edges", "directional_edges.json")
    hotspot_edge_lookup = {}
    if os.path.exists(dir_edges_path):
        with open(dir_edges_path) as f:
            dir_edges = json.load(f)
        for e in dir_edges:
            hotspot_edge_lookup[(e["primary"], e["secondary"])] = e["synergy"]

    folds = list(skf.split(np.arange(N), events))
    scores = {}

    # Baseline: hotspot shift sum
    score_base = np.zeros(N)
    for idx in range(N):
        for hs in patient_hotspots.get(idx, set()):
            if hs in hotspot_shift:
                score_base[idx] += hotspot_shift[hs]
    scores["A_hotspot_sum"] = score_base

    # Directional (from previous work)
    score_dir = score_base.copy()
    for idx in range(N):
        hs_list = sorted(patient_hotspots.get(idx, set()),
                         key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) < 2:
            continue
        for i in range(len(hs_list)):
            for j in range(i + 1, len(hs_list)):
                key = (hs_list[i], hs_list[j])
                if key in hotspot_edge_lookup:
                    score_dir[idx] += hotspot_edge_lookup[key]
    scores["B_directional"] = score_dir

    # Graph-distance-weighted synergy
    # Weight synergy corrections by inverse graph distance (closer = stronger interaction)
    score_gdist = score_base.copy()
    n_applied = 0
    for idx in range(N):
        hs_list = sorted(patient_hotspots.get(idx, set()),
                         key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) < 2:
            continue
        for i in range(len(hs_list)):
            for j in range(i + 1, len(hs_list)):
                key = (hs_list[i], hs_list[j])
                syn = hotspot_edge_lookup.get(key, None)
                if syn is None:
                    continue

                ga = hotspot_gene.get(hs_list[i], "?")
                gb = hotspot_gene.get(hs_list[j], "?")
                dist = graph_dist(ga, gb)

                if dist >= 0:
                    # Closer = stronger interaction
                    # dist=1: weight=1.0, dist=2: weight=0.5, dist=3: weight=0.33
                    weight = 1.0 / max(dist, 1)
                else:
                    # Disconnected: weak interaction
                    weight = 0.1

                score_gdist[idx] += syn * weight
                n_applied += 1
    scores["C_graph_dist_weighted"] = score_gdist
    print(f"  Applied {n_applied} graph-distance-weighted corrections")

    # Hub-relative scoring: weight by relative hub position
    score_hub = score_base.copy()
    for idx in range(N):
        hs_list = sorted(patient_hotspots.get(idx, set()),
                         key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) < 2:
            continue
        for i in range(len(hs_list)):
            for j in range(i + 1, len(hs_list)):
                key = (hs_list[i], hs_list[j])
                syn = hotspot_edge_lookup.get(key, None)
                if syn is None:
                    continue

                ga = hotspot_gene.get(hs_list[i], "?")
                gb = hotspot_gene.get(hs_list[j], "?")
                ha = gene_to_hub_dist.get(ga, -1)
                hb = gene_to_hub_dist.get(gb, -1)

                # Hub distance difference as weight
                if ha >= 0 and hb >= 0:
                    hub_diff = abs(ha - hb)
                    # Larger difference = more asymmetric = stronger directional effect
                    weight = 1.0 + hub_diff * 0.3
                else:
                    weight = 1.0

                score_hub[idx] += syn * weight
    scores["D_hub_relative"] = score_hub

    # Combined: graph distance + hub relative + gene topology
    score_combined = np.zeros(N)
    for idx in range(N):
        # Gene-level base with hub distance weighting
        for g in patient_genes.get(idx, set()):
            if g not in gene_shift:
                continue
            hd = gene_to_hub_dist.get(g, -1)
            if hd >= 0:
                # Closer to hub = less containable = amplify positive (harmful) shifts
                # Farther from hub = more containable = amplify negative (protective) shifts
                if gene_shift[g] > 0:
                    # Harmful: amplify if close to hub
                    weight = 1.0 + max(0, 2 - hd) * 0.3
                else:
                    # Protective: amplify if far from hub (containable)
                    weight = 1.0 + hd * 0.1
            else:
                weight = 1.0
            score_combined[idx] += gene_shift[g] * weight

        # Pair corrections with graph distance weighting
        hs_list = sorted(patient_hotspots.get(idx, set()),
                         key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) >= 2:
            for i in range(len(hs_list)):
                for j in range(i + 1, len(hs_list)):
                    key = (hs_list[i], hs_list[j])
                    syn = hotspot_edge_lookup.get(key, None)
                    if syn is None:
                        continue
                    ga = hotspot_gene.get(hs_list[i], "?")
                    gb = hotspot_gene.get(hs_list[j], "?")
                    dist = graph_dist(ga, gb)
                    weight = 1.0 / max(dist, 1) if dist >= 0 else 0.1
                    score_combined[idx] += syn * weight * 0.5
    scores["E_combined"] = score_combined

    # Evaluate
    print(f"\n  {'Scorer':<30} {'Mean CI':>8} {'Std':>7}  Fold CIs")
    print(f"  {'-'*30} {'-'*8} {'-'*7}  {'-'*40}")

    results = {}
    for name in ["A_hotspot_sum", "B_directional", "C_graph_dist_weighted",
                  "D_hub_relative", "E_combined"]:
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
        results[name] = {"mean": mean_ci, "std": std_ci}
        folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis)
        print(f"  {name:<30} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

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

    # Save
    with open(os.path.join(SAVE_BASE, "pair_analysis.json"), "w") as f:
        # Save summary stats, not all pairs
        summary = {
            "n_pairs": len(pairs),
            "n_connected": len(connected),
            "n_disconnected": len(disconnected),
            "scoring_results": {k: v["mean"] for k, v in results.items()},
        }
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
