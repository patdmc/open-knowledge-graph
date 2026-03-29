#!/usr/bin/env python3
"""
Gap analysis: where does the graph lose most to V6c, and why?

For each cancer type with a large graph-vs-V6c gap, profiles the "gap patients"
(where graph rank is much worse than V6c rank) to find systematic patterns —
specific genes, channels, mutation counts, GOF/LOF ratios — that point to
missing graph dimensions.

Queries Neo4j for patient mutation data.

Usage:
    python3 -u -m gnn.scripts.gap_analysis
"""

import sys, os, json, numpy as np, torch, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence, fit_per_ct_ridge,
)
from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles, profile_entropy
from gnn.scripts.pairwise_graph_scorer import (
    precompute_ppi_distances, cosine_sim, pairwise_graph_walk_batch,
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")


def query_patient_mutations(driver, cancer_type):
    """Pull patient mutation details from Neo4j for a specific cancer type."""
    with driver.session() as session:
        result = session.run(
            """MATCH (p:Patient {cancer_type: $ct})
               OPTIONAL MATCH (p)-[m:HAS_MUTATION]->(g:Gene)
               WITH p, collect({
                   gene: g.name, channel: g.channel,
                   primary_channel: g.primary_channel,
                   is_hub: g.is_hub, function: g.function,
                   direction: m.direction,
                   profile_entropy: g.profile_entropy,
                   confidence: g.confidence
               }) AS mutations
               RETURN p.id AS pid, p.os_months AS os_months, p.event AS event,
                      mutations, size(mutations) AS n_mut""",
            ct=cancer_type,
        )
        records = []
        for r in result:
            muts = [m for m in r["mutations"] if m["gene"] is not None]
            records.append({
                "pid": r["pid"],
                "os_months": r["os_months"],
                "event": r["event"],
                "n_mut": len(muts),
                "mutations": muts,
                "genes": [m["gene"] for m in muts],
                "channels": [m["channel"] for m in muts],
                "n_hub": sum(1 for m in muts if m["is_hub"]),
                "n_gof": sum(1 for m in muts if m["direction"] == "GOF"),
                "n_lof": sum(1 for m in muts if m["direction"] == "LOF"),
                "n_channels": len(set(m["channel"] for m in muts)),
                "mean_entropy": np.mean([m["profile_entropy"] for m in muts]) if muts else 0,
                "n_low_conf": sum(1 for m in muts if m["confidence"] != "high"),
            })
        return pd.DataFrame(records)


def main():
    print("=" * 90)
    print("  GAP ANALYSIS: WHERE AND WHY DOES THE GRAPH LOSE?")
    print("=" * 90)

    # --- Load data and compute scores (same as pairwise scorer) ---
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

    # Graph scorer — reuse pairwise feature extraction
    print("\n  Computing graph features...")
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
    for _, row in mut_exp.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes[idx].add(row["gene.hugoGeneSymbol"])

    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    ppi_edges = fetch_string_expanded(set().union(*patient_genes.values()) & expanded_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)
    ppi_dists = precompute_ppi_distances(G_ppi, set().union(*patient_genes.values()) & expanded_genes)

    hub_gene_set = set()
    for ch_hubs in HUB_GENES.values():
        hub_gene_set |= ch_hubs

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
    for idx in range(N):
        ct_name = ct_per_patient[idx]
        ct_patients[ct_name].add(idx)
        for g in patient_genes.get(idx, set()):
            prof = channel_profiles.get(g)
            if prof is not None:
                for ci, ch_name in enumerate(V6_CHANNEL_NAMES):
                    if prof[ci] > 0.05:
                        ct_ch_patients[ct_name][ch_name].add(idx)

    ct_baseline_map = {}
    for ct_name, pts in ct_patients.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 50:
            ct_baseline_map[ct_name] = float(hazards[pts_valid].mean())

    ct_ch_shift = {}
    for ct_name in ct_baseline_map:
        bl = ct_baseline_map[ct_name]
        for ch_name in V6_CHANNEL_NAMES:
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
        ct_gene_map = defaultdict(set)
        for idx in ct_patients[ct_name]:
            for g in patient_genes.get(idx, set()):
                ct_gene_map[g].add(idx)
        for gene, pts in ct_gene_map.items():
            pts_arr = np.array(sorted(pts))
            pts_valid = pts_arr[valid_mask[pts_arr]]
            if len(pts_valid) >= 15:
                ct_gene_shift[(ct_name, gene)] = float(hazards[pts_valid].mean() - bl)

    # Graph walk (cached to avoid slow recomputation)
    cache_path = os.path.join(ANALYSIS_CACHE, "gap_analysis_features.npz")
    if os.path.exists(cache_path):
        print("  Loading cached pairwise features...")
        cached = np.load(cache_path)
        X_all = cached["X_all"]
    else:
        print("  Running pairwise graph walk...")
        X_all = pairwise_graph_walk_batch(
            G_ppi, patient_genes, channel_profiles, ppi_dists,
            channel_features, tier_features, ct_per_patient,
            ct_ch_shift, ct_gene_shift, global_gene_shift,
            age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
            expanded_cm, cooccurrence, hub_gene_set, N,
        )
        np.savez_compressed(cache_path, X_all=X_all)
        print(f"  Cached features to {cache_path}")

    # Fit per-CT ridge to get graph scores
    folds = list(skf.split(np.arange(N), events))
    # Use all features for best graph score
    score_graph = fit_per_ct_ridge(X_all, hazards, valid_mask, events, times,
                                    ct_per_patient, folds, ct_min_patients=200)

    # =========================================================================
    # Per-CT gap analysis
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE GAP RANKING")
    print(f"{'='*90}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    ct_gaps = {}
    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 100 or events[ct_indices].sum() < 10:
            continue

        ci_graph = concordance_index(
            torch.tensor(score_graph[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v6c = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ct_gaps[ct_name] = {
            "n": len(ct_indices), "graph": ci_graph, "v6c": ci_v6c,
            "gap": ci_v6c - ci_graph, "indices": ct_indices,
        }

    # Sort by gap (worst first)
    sorted_cts = sorted(ct_gaps, key=lambda x: -ct_gaps[x]["gap"])

    print(f"\n  {'Cancer Type':<35} {'N':>5} {'Graph':>7} {'V6c':>7} {'Gap':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    for ct in sorted_cts:
        g = ct_gaps[ct]
        marker = " <<<" if g["gap"] > 0.02 else ""
        print(f"  {ct:<35} {g['n']:>5} {g['graph']:>7.4f} {g['v6c']:>7.4f} {g['gap']:>+7.4f}{marker}")

    # =========================================================================
    # Deep dive on worst-gap CTs
    # =========================================================================
    worst_cts = [ct for ct in sorted_cts if ct_gaps[ct]["gap"] > 0.015 and ct_gaps[ct]["n"] >= 200]

    for ct_name in worst_cts[:6]:
        gap_info = ct_gaps[ct_name]
        ct_indices = gap_info["indices"]

        print(f"\n{'='*90}")
        print(f"  DEEP DIVE: {ct_name} (N={gap_info['n']}, gap={gap_info['gap']:+.4f})")
        print(f"{'='*90}")

        # Per-patient rank comparison
        graph_scores = score_graph[ct_indices]
        v6c_scores = hazards[ct_indices]
        t = times[ct_indices]
        e = events[ct_indices]

        # Rank within CT (higher score = higher predicted risk)
        graph_ranks = np.argsort(np.argsort(-graph_scores))  # 0 = highest risk
        v6c_ranks = np.argsort(np.argsort(-v6c_scores))
        rank_diff = graph_ranks.astype(float) - v6c_ranks.astype(float)  # positive = graph ranks higher risk

        # Gap patients: where graph rank is much worse (further from truth)
        # For deceased patients: graph should rank them high risk. If graph ranks them lower than V6c, that's a miss.
        # For alive patients: graph should rank them low risk. If graph ranks them higher than V6c, that's a miss.
        # Simple: look at large |rank_diff| patients
        abs_rank_diff = np.abs(rank_diff)
        p75 = np.percentile(abs_rank_diff, 75)
        gap_mask = abs_rank_diff > p75
        nogap_mask = abs_rank_diff <= np.percentile(abs_rank_diff, 25)

        # Query Neo4j for mutation details
        neo4j_df = query_patient_mutations(driver, ct_name)
        if len(neo4j_df) == 0 or "pid" not in neo4j_df.columns:
            print("  No Neo4j data for this cancer type — skipping.")
            continue

        # Map ct_indices to patient IDs
        idx_to_pid = {v: k for k, v in pid_to_idx.items()}
        gap_pids = set(idx_to_pid.get(ct_indices[i]) for i in range(len(ct_indices)) if gap_mask[i])
        nogap_pids = set(idx_to_pid.get(ct_indices[i]) for i in range(len(ct_indices)) if nogap_mask[i])

        gap_df = neo4j_df[neo4j_df["pid"].isin(gap_pids)]
        nogap_df = neo4j_df[neo4j_df["pid"].isin(nogap_pids)]

        if len(gap_df) == 0 or len(nogap_df) == 0:
            print("  Insufficient Neo4j data for comparison.")
            continue

        # --- Compare gap vs no-gap patients ---
        print(f"\n  Gap patients (top 25% rank disagreement): {len(gap_df)}")
        print(f"  No-gap patients (bottom 25%):              {len(nogap_df)}")

        print(f"\n  {'Metric':<35} {'Gap':>10} {'No-Gap':>10} {'Ratio':>8}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")

        metrics = {
            "Mean mutations": ("n_mut", "mean"),
            "Mean hub mutations": ("n_hub", "mean"),
            "Mean GOF count": ("n_gof", "mean"),
            "Mean LOF count": ("n_lof", "mean"),
            "Mean channels hit": ("n_channels", "mean"),
            "Mean gene entropy": ("mean_entropy", "mean"),
            "Mean low-conf genes": ("n_low_conf", "mean"),
            "Frac with 0 mutations": ("n_mut", lambda x: (x == 0).mean()),
            "Frac with 1 mutation": ("n_mut", lambda x: (x == 1).mean()),
            "Frac with 5+ mutations": ("n_mut", lambda x: (x >= 5).mean()),
            "Frac with 10+ mutations": ("n_mut", lambda x: (x >= 10).mean()),
            "Frac hub-only": ("n_mut", lambda x: None),  # placeholder
        }

        for label, (col, agg) in metrics.items():
            if label == "Frac hub-only":
                gap_v = (gap_df["n_hub"] == gap_df["n_mut"]).mean() if len(gap_df) > 0 else 0
                nogap_v = (nogap_df["n_hub"] == nogap_df["n_mut"]).mean() if len(nogap_df) > 0 else 0
            elif callable(agg):
                gap_v = agg(gap_df[col]) if len(gap_df) > 0 else 0
                nogap_v = agg(nogap_df[col]) if len(nogap_df) > 0 else 0
            else:
                gap_v = gap_df[col].agg(agg) if len(gap_df) > 0 else 0
                nogap_v = nogap_df[col].agg(agg) if len(nogap_df) > 0 else 0

            ratio = gap_v / nogap_v if nogap_v > 0 else float('inf')
            print(f"  {label:<35} {gap_v:>10.3f} {nogap_v:>10.3f} {ratio:>8.2f}x")

        # --- Gene enrichment in gap patients ---
        print(f"\n  Gene enrichment (gap vs no-gap):")
        gap_gene_counts = Counter()
        nogap_gene_counts = Counter()
        for _, row in gap_df.iterrows():
            for g in row["genes"]:
                gap_gene_counts[g] += 1
        for _, row in nogap_df.iterrows():
            for g in row["genes"]:
                nogap_gene_counts[g] += 1

        # Enrichment ratio
        n_gap = len(gap_df)
        n_nogap = len(nogap_df)
        gene_enrichment = {}
        all_genes = set(gap_gene_counts.keys()) | set(nogap_gene_counts.keys())
        for g in all_genes:
            gap_frac = gap_gene_counts.get(g, 0) / max(n_gap, 1)
            nogap_frac = nogap_gene_counts.get(g, 0) / max(n_nogap, 1)
            if gap_frac + nogap_frac > 0.05:  # at least 5% prevalence
                enrichment = gap_frac / max(nogap_frac, 0.001)
                gene_enrichment[g] = {
                    "gap_frac": gap_frac, "nogap_frac": nogap_frac,
                    "enrichment": enrichment,
                    "gap_n": gap_gene_counts.get(g, 0),
                    "nogap_n": nogap_gene_counts.get(g, 0),
                }

        # Top enriched in gap (graph misses these)
        top_gap = sorted(gene_enrichment, key=lambda x: -gene_enrichment[x]["enrichment"])
        print(f"\n  {'Gene':<12} {'Gap%':>7} {'NoGap%':>7} {'Enrich':>7} {'Channel':<20} {'Hub':>4}")
        print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*20} {'-'*4}")
        for g in top_gap[:15]:
            e = gene_enrichment[g]
            ch = expanded_cm.get(g, "?")
            hub = "Y" if g in hub_gene_set else ""
            print(f"  {g:<12} {e['gap_frac']:>7.1%} {e['nogap_frac']:>7.1%} "
                  f"{e['enrichment']:>7.2f}x {ch:<20} {hub:>4}")

        # Top enriched in no-gap (graph handles these well)
        top_nogap = sorted(gene_enrichment, key=lambda x: gene_enrichment[x]["enrichment"])
        print(f"\n  Genes graph handles well (enriched in no-gap):")
        for g in top_nogap[:10]:
            e = gene_enrichment[g]
            ch = expanded_cm.get(g, "?")
            hub = "Y" if g in hub_gene_set else ""
            print(f"  {g:<12} {e['gap_frac']:>7.1%} {e['nogap_frac']:>7.1%} "
                  f"{e['enrichment']:>7.2f}x {ch:<20} {hub:>4}")

        # --- Channel enrichment ---
        print(f"\n  Channel distribution:")
        gap_ch = Counter()
        nogap_ch = Counter()
        for _, row in gap_df.iterrows():
            for ch in row["channels"]:
                gap_ch[ch] += 1
        for _, row in nogap_df.iterrows():
            for ch in row["channels"]:
                nogap_ch[ch] += 1

        print(f"  {'Channel':<20} {'Gap%':>7} {'NoGap%':>7} {'Ratio':>7}")
        print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7}")
        for ch in V6_CHANNEL_NAMES:
            gf = gap_ch.get(ch, 0) / max(sum(gap_ch.values()), 1)
            nf = nogap_ch.get(ch, 0) / max(sum(nogap_ch.values()), 1)
            ratio = gf / max(nf, 0.001)
            print(f"  {ch:<20} {gf:>7.1%} {nf:>7.1%} {ratio:>7.2f}x")

        # --- MSI/TMB analysis ---
        gap_indices = [ct_indices[i] for i in range(len(ct_indices)) if gap_mask[i]]
        nogap_indices = [ct_indices[i] for i in range(len(ct_indices)) if nogap_mask[i]]

        if len(gap_indices) > 0 and len(nogap_indices) > 0:
            gap_msi = msi[gap_indices].mean()
            nogap_msi = msi[nogap_indices].mean()
            gap_tmb = tmb[gap_indices].mean()
            nogap_tmb = tmb[nogap_indices].mean()
            gap_msi_high = msi_high[gap_indices].mean()
            nogap_msi_high = msi_high[nogap_indices].mean()

            print(f"\n  Clinical features:")
            print(f"  {'Feature':<20} {'Gap':>10} {'NoGap':>10}")
            print(f"  {'-'*20} {'-'*10} {'-'*10}")
            print(f"  {'MSI score':<20} {gap_msi:>10.3f} {nogap_msi:>10.3f}")
            print(f"  {'MSI-high frac':<20} {gap_msi_high:>10.3f} {nogap_msi_high:>10.3f}")
            print(f"  {'TMB (log)':<20} {gap_tmb:>10.3f} {nogap_tmb:>10.3f}")
            print(f"  {'Mean age':<20} {age[gap_indices].mean():>10.1f} {age[nogap_indices].mean():>10.1f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  SUMMARY: LARGEST GAPS")
    print(f"{'='*90}")
    print(f"\n  CTs where graph loses by >1.5%:")
    for ct in worst_cts[:6]:
        g = ct_gaps[ct]
        print(f"    {ct:<35} gap={g['gap']:+.4f} (N={g['n']})")

    print(f"\n  CTs where graph wins:")
    winners = [ct for ct in sorted_cts if ct_gaps[ct]["gap"] < -0.001]
    for ct in winners:
        g = ct_gaps[ct]
        print(f"    {ct:<35} gap={g['gap']:+.4f} (N={g['n']})")

    driver.close()
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
