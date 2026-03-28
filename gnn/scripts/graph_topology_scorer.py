#!/usr/bin/env python3
"""
Graph topology scoring: use the actual PPI graph shape to predict cascade response.

The insight: if we know WHERE in the graph a mutation occurs, we can predict
how the escalation cascade responds — because the response depends on:
  1. How many paths the mutation cuts (betweenness centrality)
  2. How many alternative paths exist (local redundancy)
  3. Distance to tier boundary (escalation distance)
  4. Whether the gene is a bridge (cut vertex) or redundant node

We use STRING PPI within-channel subgraphs + cross-channel edges to build
the actual graph, then compute topology features per gene and per mutation.
These features become the scoring function.

Usage:
    python3 -u -m gnn.scripts.graph_topology_scorer
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    CHANNEL_MAP, CHANNEL_NAMES, ALL_GENES,
    HUB_GENES, LEAF_GENES, GENE_POSITION, GENE_FUNCTION,
    GENES_PER_CHANNEL, CHANNEL_TO_IDX, TRUNCATING,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, CHANNEL_FEAT_DIM_V6, V6_TIER_MAP,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

import networkx as nx

SAVE_BASE = os.path.join(GNN_RESULTS, "graph_topology")
MIN_HOTSPOT_PATIENTS = 30


def build_graphs():
    """Build NetworkX graphs from STRING PPI edges."""
    ppi_path = os.path.join(GNN_CACHE, "string_ppi_edges.json")
    with open(ppi_path) as f:
        ppi = json.load(f)

    # Full graph with all edges
    G_full = nx.Graph()
    for gene in V6_CHANNEL_MAP.keys():
        ch = V6_CHANNEL_MAP[gene]
        tier = V6_TIER_MAP.get(ch, -1)
        pos = GENE_POSITION.get(gene, "unclassified")
        func = GENE_FUNCTION.get(gene, "unknown")
        G_full.add_node(gene, channel=ch, tier=tier, position=pos, function=func)

    # Within-channel edges
    channel_graphs = {}
    for ch_name in V6_CHANNEL_NAMES:
        G_ch = nx.Graph()
        ch_genes = [g for g, c in V6_CHANNEL_MAP.items() if c == ch_name]
        for g in ch_genes:
            G_ch.add_node(g, position=GENE_POSITION.get(g, "unclassified"),
                         function=GENE_FUNCTION.get(g, "unknown"))

        if ch_name in ppi:
            for src, tgt, score in ppi[ch_name]:
                if src in V6_CHANNEL_MAP and tgt in V6_CHANNEL_MAP:
                    G_ch.add_edge(src, tgt, weight=score / 1000.0)
                    G_full.add_edge(src, tgt, weight=score / 1000.0, edge_type="within_channel")

        channel_graphs[ch_name] = G_ch

    # Cross-channel edges
    for src, tgt, score in ppi.get("cross_channel", []):
        if src in V6_CHANNEL_MAP and tgt in V6_CHANNEL_MAP:
            G_full.add_edge(src, tgt, weight=score / 1000.0, edge_type="cross_channel")

    return G_full, channel_graphs


def compute_gene_topology(G_full, channel_graphs):
    """Compute topology features for each gene."""
    gene_features = {}

    # Global metrics
    degree_cent = nx.degree_centrality(G_full)
    try:
        betweenness = nx.betweenness_centrality(G_full)
    except Exception:
        betweenness = {g: 0 for g in G_full.nodes()}

    try:
        closeness = nx.closeness_centrality(G_full)
    except Exception:
        closeness = {g: 0 for g in G_full.nodes()}

    # Find cut vertices (bridge nodes — removing them disconnects the graph)
    cut_vertices = set()
    for ch_name, G_ch in channel_graphs.items():
        if G_ch.number_of_edges() > 0:
            try:
                cv = set(nx.articulation_points(G_ch))
                cut_vertices |= cv
            except Exception:
                pass

    # Per-gene features
    for gene in G_full.nodes():
        ch = V6_CHANNEL_MAP.get(gene, "?")
        tier = V6_TIER_MAP.get(ch, -1)
        G_ch = channel_graphs.get(ch, nx.Graph())

        # Within-channel degree
        ch_degree = G_ch.degree(gene) if gene in G_ch else 0
        ch_max_degree = max(dict(G_ch.degree()).values()) if G_ch.number_of_edges() > 0 else 1

        # Cross-channel degree (edges to other channels)
        cross_degree = sum(1 for n in G_full.neighbors(gene)
                         if V6_CHANNEL_MAP.get(n, "") != ch)

        # Is this gene a bridge in its channel subgraph?
        is_bridge = 1 if gene in cut_vertices else 0

        # Local clustering coefficient
        clustering = nx.clustering(G_full, gene)

        # Average shortest path to other genes in same channel
        avg_path_in_channel = 0
        if gene in G_ch and G_ch.number_of_nodes() > 1:
            paths = []
            for other in G_ch.nodes():
                if other != gene and nx.has_path(G_ch, gene, other):
                    paths.append(nx.shortest_path_length(G_ch, gene, other))
            if paths:
                avg_path_in_channel = np.mean(paths)

        # Number of connected components in channel if this gene is removed
        components_after_removal = 0
        if gene in G_ch and G_ch.number_of_edges() > 0:
            G_tmp = G_ch.copy()
            G_tmp.remove_node(gene)
            components_after_removal = nx.number_connected_components(G_tmp)

        # Components before removal
        components_before = nx.number_connected_components(G_ch) if G_ch.number_of_nodes() > 0 else 1
        components_delta = components_after_removal - components_before

        gene_features[gene] = {
            "channel": ch,
            "tier": tier,
            "position": GENE_POSITION.get(gene, "unclassified"),
            "function": GENE_FUNCTION.get(gene, "unknown"),
            "degree_centrality": round(degree_cent.get(gene, 0), 4),
            "betweenness_centrality": round(betweenness.get(gene, 0), 4),
            "closeness_centrality": round(closeness.get(gene, 0), 4),
            "within_channel_degree": ch_degree,
            "within_channel_degree_norm": round(ch_degree / max(ch_max_degree, 1), 4),
            "cross_channel_degree": cross_degree,
            "total_degree": ch_degree + cross_degree,
            "is_bridge": is_bridge,
            "clustering_coefficient": round(clustering, 4),
            "avg_path_in_channel": round(avg_path_in_channel, 3),
            "components_delta_on_removal": components_delta,
        }

    return gene_features


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 110)
    print("  GRAPH TOPOLOGY SCORER")
    print("  Using actual PPI graph shape to predict cascade response")
    print("=" * 110)

    # =========================================================================
    # 1. Build graphs
    # =========================================================================
    print("\n[1] Building graphs from STRING PPI...")
    G_full, channel_graphs = build_graphs()
    print(f"    Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    for ch_name, G_ch in channel_graphs.items():
        n_comp = nx.number_connected_components(G_ch) if G_ch.number_of_nodes() > 0 else 0
        print(f"    {ch_name:>18}: {G_ch.number_of_nodes():>3} nodes, {G_ch.number_of_edges():>3} edges, "
              f"{n_comp} components")

    # =========================================================================
    # 2. Compute gene topology features
    # =========================================================================
    print("\n[2] Computing gene topology features...")
    gene_features = compute_gene_topology(G_full, channel_graphs)

    # Print topology summary
    print(f"\n    {'Gene':<12} {'Ch':<14} {'Tier':>4} {'Pos':<6} {'Func':<7} "
          f"{'Deg':>3} {'XDeg':>4} {'Betw':>6} {'Bridge':>6} {'Clust':>6} "
          f"{'AvgPath':>7} {'CompΔ':>5}")
    print(f"    {'-'*12} {'-'*14} {'-'*4} {'-'*6} {'-'*7} "
          f"{'-'*3} {'-'*4} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*7} {'-'*5}")

    # Sort by betweenness to show most critical nodes first
    sorted_genes = sorted(gene_features.items(), key=lambda x: -x[1]["betweenness_centrality"])
    for gene, gf in sorted_genes[:40]:
        print(f"    {gene:<12} {gf['channel']:<14} {gf['tier']:>4} {gf['position']:<6} "
              f"{gf['function']:<7} {gf['within_channel_degree']:>3} {gf['cross_channel_degree']:>4} "
              f"{gf['betweenness_centrality']:>6.4f} {gf['is_bridge']:>6} "
              f"{gf['clustering_coefficient']:>6.3f} "
              f"{gf['avg_path_in_channel']:>7.2f} {gf['components_delta_on_removal']:>5}")

    # =========================================================================
    # 3. Hub vs leaf validation: do topology metrics match our classification?
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  HUB vs LEAF: TOPOLOGY VALIDATION")
    print(f"{'='*110}")

    hub_metrics = defaultdict(list)
    leaf_metrics = defaultdict(list)
    for gene, gf in gene_features.items():
        target = hub_metrics if gf["position"] == "hub" else (
            leaf_metrics if gf["position"] == "leaf" else None)
        if target is not None:
            for k in ["degree_centrality", "betweenness_centrality", "within_channel_degree",
                      "cross_channel_degree", "is_bridge", "clustering_coefficient",
                      "components_delta_on_removal"]:
                target[k].append(gf[k])

    print(f"\n    {'Metric':<30} {'Hub mean':>10} {'Leaf mean':>10} {'Ratio':>8}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    for k in ["degree_centrality", "betweenness_centrality", "within_channel_degree",
              "cross_channel_degree", "is_bridge", "clustering_coefficient",
              "components_delta_on_removal"]:
        hm = np.mean(hub_metrics[k]) if hub_metrics[k] else 0
        lm = np.mean(leaf_metrics[k]) if leaf_metrics[k] else 0
        ratio = hm / lm if lm > 0 else float("inf")
        print(f"    {k:<30} {hm:>10.4f} {lm:>10.4f} {ratio:>8.2f}x")

    # =========================================================================
    # 4. Load hazard data and test topology → defense correlation
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  TOPOLOGY → DEFENSE STRENGTH CORRELATION")
    print(f"  Does graph position predict cascade response?")
    print(f"{'='*110}")

    print("\n    Loading data + predictions...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)

    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
        "hidden_dim": 128,
        "cross_channel_heads": 4,
        "cross_channel_layers": 2,
        "dropout": 0.3,
        "n_cancer_types": n_ct,
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
            batch = {
                "channel_features": data["channel_features"][idx],
                "tier_features": data["tier_features"][idx],
                "cancer_type_idx": data["cancer_type_idx"][idx],
                "age": data["age"][idx],
                "sex": data["sex"][idx],
                "msi_score": data["msi_score"][idx],
                "msi_high": data["msi_high"][idx],
                "tmb": data["tmb"][idx],
            }
            with torch.no_grad():
                h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline_hazard = hazards[valid_mask].mean()
    print(f"    {all_in_val.sum().item()} patients with predictions")

    # Load mutations
    import pandas as pd
    mut_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    mutations = pd.read_csv(mut_path, low_memory=False,
                            usecols=["patientId", "gene.hugoGeneSymbol",
                                     "proteinChange", "mutationType"])

    clin_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv")
    clin = pd.read_csv(clin_path, low_memory=False)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(
        lambda x: 1 if "DECEASED" in str(x).upper() else 0
    )
    clin = clin[clin["time"] > 0]
    patient_ids = clin["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mutations[
        mutations["gene.hugoGeneSymbol"].isin(channel_genes) &
        mutations["proteinChange"].notna() &
        (mutations["proteinChange"] != "")
    ].copy()
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)
    mut["is_truncating"] = mut["mutationType"].isin(TRUNCATING).astype(int)

    # Per-gene hazard shift
    gene_shifts = {}
    gene_patient_sets = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patient_sets[row["gene.hugoGeneSymbol"]].add(idx)

    for gene, pts in gene_patient_sets.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            gene_shifts[gene] = float(hazards[pts_valid].mean() - baseline_hazard)

    # Correlate topology features with defense strength
    from scipy import stats

    print(f"\n    Topology feature correlations with gene hazard shift:")
    print(f"    (Negative shift = protective, positive = harmful)")
    print(f"\n    {'Feature':<35} {'r':>8} {'p':>12} {'N':>5}")
    print(f"    {'-'*35} {'-'*8} {'-'*12} {'-'*5}")

    topology_features_to_test = [
        "degree_centrality", "betweenness_centrality", "closeness_centrality",
        "within_channel_degree", "within_channel_degree_norm",
        "cross_channel_degree", "total_degree",
        "is_bridge", "clustering_coefficient",
        "avg_path_in_channel", "components_delta_on_removal",
    ]

    genes_with_both = [g for g in gene_shifts if g in gene_features]
    shifts_arr = np.array([gene_shifts[g] for g in genes_with_both])

    topology_correlations = {}
    for feat in topology_features_to_test:
        feat_arr = np.array([gene_features[g][feat] for g in genes_with_both])
        if np.std(feat_arr) > 0:
            r, p = stats.pearsonr(feat_arr, shifts_arr)
            topology_correlations[feat] = {"r": r, "p": p, "n": len(genes_with_both)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {feat:<35} {r:>+8.4f} {p:>12.2e} {len(genes_with_both):>5} {sig}")

    # =========================================================================
    # 5. Per-gene detail: topology + defense shift + graph position
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  GENE-LEVEL: GRAPH POSITION → DEFENSE SHIFT")
    print(f"{'='*110}")

    print(f"\n    {'Gene':<12} {'Channel':<14} {'Tier':>4} {'HubLf':<5} {'Func':<4} "
          f"{'Deg':>3} {'Betw':>6} {'Bridge':>6} {'Shift':>7} {'Interp'}")
    print(f"    {'-'*12} {'-'*14} {'-'*4} {'-'*5} {'-'*4} "
          f"{'-'*3} {'-'*6} {'-'*6} {'-'*7} {'-'*30}")

    gene_rows = []
    for gene in sorted(gene_shifts.keys(), key=lambda g: gene_shifts[g]):
        gf = gene_features.get(gene)
        if gf is None:
            continue
        shift = gene_shifts[gene]
        # Interpret: bridge + protective = strong cascade defense at critical node
        if gf["is_bridge"] and shift < -0.3:
            interp = "CRITICAL DEFENDER: bridge + strong defense"
        elif gf["is_bridge"] and shift > 0:
            interp = "CRITICAL VULNERABILITY: bridge + harmful"
        elif gf["betweenness_centrality"] > 0.05 and shift < -0.3:
            interp = "Central + strong defense"
        elif gf["within_channel_degree"] == 0 and shift < -0.3:
            interp = "Isolated + strong defense (local fail-safe)"
        elif gf["cross_channel_degree"] > 2 and shift < -0.3:
            interp = "Cross-channel hub + defense"
        else:
            interp = ""

        gene_rows.append((gene, gf, shift, interp))
        print(f"    {gene:<12} {gf['channel']:<14} {gf['tier']:>4} {gf['position']:<5} "
              f"{gf['function']:<4} {gf['within_channel_degree']:>3} "
              f"{gf['betweenness_centrality']:>6.4f} {gf['is_bridge']:>6} "
              f"{shift:>+7.3f} {interp}")

    # =========================================================================
    # 6. Build graph-based scoring function
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  GRAPH-BASED SCORING FUNCTION")
    print(f"  Using topology features to weight mutation contributions")
    print(f"{'='*110}")

    # Patient-level scoring with graph topology weights
    # For each patient:
    #   score = sum over mutated genes of:
    #     gene_shift * topology_weight + directional_corrections

    # Load directional edges from previous analysis
    dir_edges_path = os.path.join(GNN_RESULTS, "directional_edges", "directional_edges.json")
    dir_edges = []
    if os.path.exists(dir_edges_path):
        with open(dir_edges_path) as f:
            dir_edges = json.load(f)

    hotspot_edge_lookup = {}
    for e in dir_edges:
        hotspot_edge_lookup[(e["primary"], e["secondary"])] = e["synergy"]

    # Per-hotspot shift
    mut["hotspot"] = mut["gene.hugoGeneSymbol"] + " " + mut["proteinChange"].astype(str)
    hotspot_freq = mut.groupby("hotspot")["patient_idx"].nunique()
    valid_hotspots = hotspot_freq[hotspot_freq >= MIN_HOTSPOT_PATIENTS].index

    mut_valid = mut[mut["hotspot"].isin(valid_hotspots)]
    hotspot_shift = {}
    hotspot_patients = {}
    for hs, group in mut_valid.groupby("hotspot"):
        pts = set(group["patient_idx"].unique())
        hotspot_patients[hs] = pts
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 10:
            hotspot_shift[hs] = float(hazards[pts_valid].mean() - baseline_hazard)

    patient_hotspots = defaultdict(set)
    for _, row in mut_valid.iterrows():
        patient_hotspots[int(row["patient_idx"])].add(row["hotspot"])

    freq_lookup = hotspot_freq.to_dict()

    # Patient gene sets for scoring
    patient_genes = defaultdict(set)
    patient_gene_trunc = defaultdict(lambda: defaultdict(int))
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene = row["gene.hugoGeneSymbol"]
            patient_genes[idx].add(gene)
            if row["is_truncating"]:
                patient_gene_trunc[idx][gene] = 1

    # Cancer type for each patient
    cancer_type_idx = data["cancer_type_idx"].numpy()
    ct_vocab = data["cancer_type_vocab"]

    # Load cancer type × channel defense matrix if available
    ct_channel_path = os.path.join(GNN_RESULTS, "directional_edges", "cancer_channel_matrix.json")
    ct_channel_matrix = {}
    if os.path.exists(ct_channel_path):
        with open(ct_channel_path) as f:
            ct_channel_matrix = json.load(f)

    # -------------------------------------------------------------------------
    # Scoring functions
    # -------------------------------------------------------------------------
    scores = {}
    folds = list(skf.split(np.arange(N), events))

    # Score A: Baseline — gene-level shift sum
    score_gene = np.zeros(N)
    for idx, genes in patient_genes.items():
        for g in genes:
            if g in gene_shifts:
                score_gene[idx] += gene_shifts[g]
    scores["A_gene_shift"] = score_gene

    # Score B: Hotspot-level shift sum
    score_hotspot = np.zeros(N)
    for idx in range(N):
        for hs in patient_hotspots.get(idx, set()):
            if hs in hotspot_shift:
                score_hotspot[idx] += hotspot_shift[hs]
    scores["B_hotspot_shift"] = score_hotspot

    # Score C: Hotspot + directional edges
    score_dir = score_hotspot.copy()
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
    scores["C_directional"] = score_dir

    # Score D: Graph-topology-weighted gene shifts
    # Weight each gene's contribution by its graph centrality
    # High centrality → mutation has more impact → amplify the shift
    score_topo = np.zeros(N)
    for idx, genes in patient_genes.items():
        for g in genes:
            if g not in gene_shifts or g not in gene_features:
                continue
            shift = gene_shifts[g]
            gf = gene_features[g]

            # Topology weight: genes with higher betweenness carry more signal
            # Bridge genes: their shift matters more (they cut paths)
            topo_weight = 1.0
            topo_weight += gf["betweenness_centrality"] * 2.0  # more central = more impactful
            topo_weight += gf["is_bridge"] * 0.5  # bridge = critical
            topo_weight += gf["cross_channel_degree"] * 0.1  # cross-channel = escalation path

            # Mutation type adjustment: truncating at bridge = catastrophic
            if patient_gene_trunc[idx].get(g, 0) and gf["is_bridge"]:
                topo_weight *= 1.3  # clean break at bridge = amplified

            score_topo[idx] += shift * topo_weight
    scores["D_topology_weighted"] = score_topo

    # Score E: Full graph scorer — topology + directional edges + tier hierarchy
    score_full = np.zeros(N)
    for idx in range(N):
        # Gene-level with topology weights
        for g in patient_genes.get(idx, set()):
            if g not in gene_shifts or g not in gene_features:
                continue
            shift = gene_shifts[g]
            gf = gene_features[g]

            topo_weight = 1.0
            topo_weight += gf["betweenness_centrality"] * 2.0
            topo_weight += gf["is_bridge"] * 0.5
            topo_weight += gf["cross_channel_degree"] * 0.1

            if patient_gene_trunc[idx].get(g, 0) and gf["is_bridge"]:
                topo_weight *= 1.3

            score_full[idx] += shift * topo_weight

        # Hotspot-level directional corrections
        hs_list = sorted(patient_hotspots.get(idx, set()),
                        key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) >= 2:
            for i in range(len(hs_list)):
                for j in range(i + 1, len(hs_list)):
                    key = (hs_list[i], hs_list[j])
                    if key in hotspot_edge_lookup:
                        score_full[idx] += hotspot_edge_lookup[key] * 0.5  # dampen synergy contrib

        # Tier hierarchy: count how many tiers have mutations
        mutated_tiers = set()
        for g in patient_genes.get(idx, set()):
            ch = V6_CHANNEL_MAP.get(g, "?")
            t = V6_TIER_MAP.get(ch, -1)
            if t >= 0:
                mutated_tiers.add(t)

        # Multi-tier mutations: escalation cascade engaged across levels
        if len(mutated_tiers) >= 3:
            # 3+ tiers = cascade likely overwhelmed OR comprehensive defense
            # Use the Mr. Burns heuristic: extreme damage can be protective
            n_genes_mutated = len(patient_genes.get(idx, set()))
            if n_genes_mutated >= 6:
                score_full[idx] -= 0.15  # saturation protective bonus
            else:
                score_full[idx] += 0.1  # moderate multi-tier = cascading failure

    scores["E_full_graph"] = score_full

    # Score F: Graph + cancer type modulation
    # Use per-cancer-type per-channel defense weights when available
    score_ct = np.zeros(N)
    for idx in range(N):
        ct_idx = int(cancer_type_idx[idx])
        ct_name = ct_vocab[ct_idx] if ct_idx < len(ct_vocab) else "Other"

        # Channel-level: cancer type modulates the base defense
        mutated_channels = defaultdict(list)
        for g in patient_genes.get(idx, set()):
            ch = V6_CHANNEL_MAP.get(g, "?")
            mutated_channels[ch].append(g)

        for ch, ch_genes in mutated_channels.items():
            # Base: sum of gene shifts with topology weight
            ch_score = 0
            for g in ch_genes:
                if g not in gene_shifts or g not in gene_features:
                    continue
                shift = gene_shifts[g]
                gf = gene_features[g]
                topo_weight = 1.0 + gf["betweenness_centrality"] * 2.0 + gf["is_bridge"] * 0.5
                ch_score += shift * topo_weight

            # Cancer type modulation: look up ct×channel weight
            ct_ch_key = f"{ct_name}_{ch}"
            if ct_ch_key in ct_channel_matrix:
                ct_mod = ct_channel_matrix[ct_ch_key]
                # Blend: 70% gene-level, 30% cancer-type modulation
                ch_score = 0.7 * ch_score + 0.3 * ct_mod * len(ch_genes)

            score_ct[idx] += ch_score

        # Directional hotspot corrections
        hs_list = sorted(patient_hotspots.get(idx, set()),
                        key=lambda x: -freq_lookup.get(x, 0))
        if len(hs_list) >= 2:
            for i in range(len(hs_list)):
                for j in range(i + 1, len(hs_list)):
                    key = (hs_list[i], hs_list[j])
                    if key in hotspot_edge_lookup:
                        score_ct[idx] += hotspot_edge_lookup[key] * 0.5

    scores["F_graph+cancer_type"] = score_ct

    # =========================================================================
    # 7. Evaluate all scoring functions
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  C-INDEX COMPARISON")
    print(f"{'='*110}")

    level_order = ["A_gene_shift", "B_hotspot_shift", "C_directional",
                   "D_topology_weighted", "E_full_graph", "F_graph+cancer_type"]

    results = {}
    for level in level_order:
        s = scores[level]
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
        results[level] = {"mean": mean_ci, "std": std_ci, "folds": fold_cis}

    print(f"\n    {'Scorer':<25} {'Mean CI':>8} {'Std':>7}  Fold CIs")
    print(f"    {'-'*25} {'-'*8} {'-'*7}  {'-'*40}")
    for level in level_order:
        r = results[level]
        folds_str = "  ".join(f"{ci:.4f}" for ci in r["folds"])
        print(f"    {level:<25} {r['mean']:>8.4f} {r['std']:>7.4f}  {folds_str}")

    # V6c comparison
    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    if os.path.exists(ua_path):
        with open(ua_path) as f:
            ua = json.load(f)
        v6c_mean = ua["means"].get("V6c", 0)
        best_simple = max(r["mean"] for r in results.values())
        best_name = max(results, key=lambda k: results[k]["mean"])
        print(f"\n    V6c transformer:     {v6c_mean:.4f}")
        print(f"    Best graph scorer:   {best_simple:.4f} ({best_name})")
        print(f"    Gap:                 {v6c_mean - best_simple:+.4f}")

    # =========================================================================
    # 8. Graph shape visualization data (for knowledge graph)
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"  GRAPH SHAPE SUMMARY — ESCALATION CHAIN POSITION")
    print(f"{'='*110}")

    # For each channel, show the graph shape
    for ch_name in V6_CHANNEL_NAMES:
        G_ch = channel_graphs[ch_name]
        tier = V6_TIER_MAP.get(ch_name, -1)
        ch_genes_with_shift = [(g, gene_shifts.get(g, None), gene_features.get(g, {}))
                               for g in G_ch.nodes() if g in gene_features]

        print(f"\n    {ch_name} (Tier {tier}):")
        print(f"    Nodes: {G_ch.number_of_nodes()}, Edges: {G_ch.number_of_edges()}")

        # Show adjacency for this channel
        if G_ch.number_of_edges() > 0:
            # Sort by degree
            for gene, shift, gf in sorted(ch_genes_with_shift,
                                          key=lambda x: -(x[2].get("within_channel_degree", 0))):
                neighbors = sorted(G_ch.neighbors(gene))
                shift_str = f"{shift:+.3f}" if shift is not None else "  n/a "
                bridge_str = " [BRIDGE]" if gf.get("is_bridge") else ""
                cross = gf.get("cross_channel_degree", 0)
                cross_str = f" (+{cross} cross-ch)" if cross > 0 else ""
                print(f"      {gene:<12} shift={shift_str} deg={gf.get('within_channel_degree',0):>2}"
                      f"{bridge_str}{cross_str}")
                if neighbors:
                    print(f"        → {', '.join(neighbors)}")

    # =========================================================================
    # 9. Save
    # =========================================================================
    with open(os.path.join(SAVE_BASE, "gene_topology.json"), "w") as f:
        json.dump(gene_features, f, indent=2)

    with open(os.path.join(SAVE_BASE, "topology_correlations.json"), "w") as f:
        json.dump({k: {"r": v["r"], "p": v["p"], "n": v["n"]}
                  for k, v in topology_correlations.items()}, f, indent=2)

    with open(os.path.join(SAVE_BASE, "scoring_results.json"), "w") as f:
        json.dump({k: {"mean": v["mean"], "std": v["std"]} for k, v in results.items()}, f, indent=2)

    # Save graph structure for knowledge graph
    graph_structure = {}
    for ch_name, G_ch in channel_graphs.items():
        graph_structure[ch_name] = {
            "nodes": list(G_ch.nodes()),
            "edges": [(u, v, d.get("weight", 1.0)) for u, v, d in G_ch.edges(data=True)],
            "n_components": nx.number_connected_components(G_ch),
            "bridges": [],
            "cut_vertices": [],
        }
        if G_ch.number_of_edges() > 0:
            try:
                graph_structure[ch_name]["cut_vertices"] = list(nx.articulation_points(G_ch))
            except Exception:
                pass
            try:
                graph_structure[ch_name]["bridges"] = list(nx.bridges(G_ch))
            except Exception:
                pass

    with open(os.path.join(SAVE_BASE, "graph_structure.json"), "w") as f:
        json.dump(graph_structure, f, indent=2, default=str)

    print(f"\n    Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
