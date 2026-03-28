#!/usr/bin/env python3
"""
Multi-channel graph scorer: genes belong to multiple channels weighted by PPI neighborhood.

Key insight: ancient genes like TP53/BRCA1 predate channel specialization.
Their PPI neighborhoods span multiple channels. A BRCA1 mutation damages
DDR (0.25), CellCycle (0.40), and PI3K_Growth (0.16) simultaneously.

Instead of assigning each gene to one channel and computing binary
channel damage, we:
  1. Derive per-gene channel weight vectors from PPI neighborhood
  2. Distribute mutation damage across channels proportionally
  3. Compute continuous channel damage scores (not binary severed/intact)

This directly addresses why subchannels hurt: finer single-assignment bins
fragment signal. The correct dimension is weighted multi-channel membership.

Usage:
    python3 -u -m gnn.scripts.multichannel_graph_scorer
"""

import sys, os, json, time, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_POSITION, GENE_FUNCTION, TRUNCATING,
    STRING_SPECIES, STRING_SCORE_THRESHOLD,
)
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
    compute_cooccurrence, fit_per_ct_ridge,
)

SAVE_BASE = os.path.join(GNN_RESULTS, "multichannel_graph_scorer")
CHANNELS = V6_CHANNEL_NAMES  # 8 channels
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}


# =========================================================================
# 1. PPI-derived multi-channel membership vectors
# =========================================================================
def compute_channel_profiles(G_ppi, channel_map):
    """For each gene, compute channel weight vector from PPI neighborhood.

    A gene's channel profile = fraction of its PPI neighbors in each channel.
    Genes with no PPI edges fall back to their assigned channel (one-hot).

    Returns: dict of gene -> np.array(N_CH,) normalized channel weights.
    """
    profiles = {}
    for gene in G_ppi.nodes():
        neighbors = list(G_ppi.neighbors(gene))
        if not neighbors:
            # Fallback: one-hot from assigned channel
            ch = channel_map.get(gene)
            vec = np.zeros(N_CH)
            if ch and ch in CH_TO_IDX:
                vec[CH_TO_IDX[ch]] = 1.0
            profiles[gene] = vec
            continue

        # Count neighbor channels, weighted by PPI edge score
        vec = np.zeros(N_CH)
        for nb in neighbors:
            nb_ch = channel_map.get(nb)
            if nb_ch and nb_ch in CH_TO_IDX:
                w = G_ppi[gene][nb].get("weight", 0.7)
                vec[CH_TO_IDX[nb_ch]] += w

        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            # No neighbors in known channels — fallback
            ch = channel_map.get(gene)
            if ch and ch in CH_TO_IDX:
                vec[CH_TO_IDX[ch]] = 1.0

        profiles[gene] = vec

    # Genes in channel_map but not in PPI graph
    for gene, ch in channel_map.items():
        if gene not in profiles:
            vec = np.zeros(N_CH)
            if ch in CH_TO_IDX:
                vec[CH_TO_IDX[ch]] = 1.0
            profiles[gene] = vec

    return profiles


def profile_entropy(vec):
    """Shannon entropy of a channel weight vector (in bits). Higher = more multi-channel."""
    v = vec[vec > 0]
    if len(v) <= 1:
        return 0.0
    return float(-np.sum(v * np.log2(v)))


# =========================================================================
# 2. Multi-channel graph walk
# =========================================================================
def multichannel_graph_walk_batch(
    G_ppi, patient_genes_map, channel_profiles, channel_features_all,
    tier_features_all, ct_per_patient,
    ct_ch_shift, ct_gene_shift, global_gene_shift,
    age_all, sex_all, msi_all, msi_high_all, tmb_all,
    ct_baseline_map, baseline, channel_map, cooccurrence, N,
):
    """
    Multi-channel graph walk: mutation damage distributed across channels
    by PPI-derived channel weight vectors.

    Feature vector (per patient):
      [0]     ct_baseline
      [1]     weighted_ch_shift_sum (CT-specific channel damage, distributed)
      [2]     ct_gene_shift_sum
      [3]     global_gene_shift_sum
      [4-11]  channel_damage (8-dim continuous: sum of channel weights for mutated genes)
      [12-19] channel_damage_entropy (8-dim: per-channel, entropy of contributing genes)
      [20]    total_damage_entropy (how spread across channels is total damage)
      [21]    n_channels_damaged (continuous: channels with damage > threshold)
      [22]    max_channel_damage
      [23]    damage_concentration (max / total — Herfindahl-like)
      [24]    total_isolation (weighted across channels)
      [25]    total_hub_damage (weighted)
      [26]    tier_conn
      [27]    n_mutated_genes
      [28-31] tier_damage (4-dim: aggregated from channel damage)
      [32-35] co-occurrence features (same as expanded)
      [36-107] channel_features (72)
      [108-127] tier_features (20)
      [128]   age
      [129]   sex
      [130]   msi
      [131]   msi_high
      [132]   tmb
    Total: 133 features
    """
    all_expanded_genes = set(channel_map.keys())
    FEAT_DIM = 133
    X = np.zeros((N, FEAT_DIM))

    # Precompute channel -> gene sets and tier -> channel mapping
    ch_gene_sets = defaultdict(set)
    for g, ch in channel_map.items():
        ch_gene_sets[ch].add(g)

    # Tier damage aggregation: which channels belong to which tier
    tier_channels = defaultdict(list)
    for ch, tier in V6_TIER_MAP.items():
        if ch in CH_TO_IDX:
            tier_channels[tier].append(CH_TO_IDX[ch])

    gene_tier = {}
    for g, ch in channel_map.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    for idx in range(N):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

        patient_genes = patient_genes_map.get(idx, set())
        mutated = patient_genes & all_expanded_genes
        ct_name = ct_per_patient[idx]
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        # --- 1. Channel damage vector (continuous, 8-dim) ---
        channel_damage = np.zeros(N_CH)
        ct_gene_shift_sum = 0.0
        global_gene_shift_sum = 0.0

        for g in mutated:
            profile = channel_profiles.get(g)
            if profile is not None:
                channel_damage += profile

            ct_s = ct_gene_shift.get((ct_name, g), None)
            if ct_s is not None:
                ct_gene_shift_sum += ct_s
            gl_s = global_gene_shift.get(g, None)
            if gl_s is not None:
                global_gene_shift_sum += gl_s

        # Weighted channel shift: distribute CT channel shifts by damage weight
        weighted_ch_shift = 0.0
        total_damage = channel_damage.sum()
        if total_damage > 0:
            damage_frac = channel_damage / total_damage
            for ci, ch_name in enumerate(CHANNELS):
                if damage_frac[ci] > 0:
                    weighted_ch_shift += damage_frac[ci] * ct_ch_shift.get((ct_name, ch_name), 0.0)

        # Entropy of total damage distribution
        if total_damage > 0:
            damage_norm = channel_damage / total_damage
            total_entropy = profile_entropy(damage_norm)
        else:
            total_entropy = 0.0

        n_channels_damaged = np.sum(channel_damage > 0.1)
        max_ch_damage = channel_damage.max() if total_damage > 0 else 0.0
        damage_concentration = (max_ch_damage / total_damage) if total_damage > 0 else 0.0

        # --- 2. Weighted isolation and hub damage ---
        total_isolation = 0.0
        total_hub_damage = 0.0

        for ci, ch_name in enumerate(CHANNELS):
            if channel_damage[ci] < 0.05:
                continue

            ch_genes = ch_gene_sets.get(ch_name, set())
            # Which mutated genes contribute to this channel?
            ch_mutated = set()
            for g in mutated:
                p = channel_profiles.get(g)
                if p is not None and p[ci] > 0.05:
                    ch_mutated.add(g)

            ch_hubs = HUB_GENES.get(ch_name, set())
            ch_in_ppi = ch_genes & set(G_ppi.nodes())
            if ch_in_ppi:
                G_ch = G_ppi.subgraph(ch_in_ppi - ch_mutated)
                components = list(nx.connected_components(G_ch))
                surviving_hubs = ch_hubs - ch_mutated
                hub_comps = set()
                for comp_i, comp in enumerate(components):
                    if comp & surviving_hubs:
                        hub_comps.add(comp_i)
                reachable = set()
                for comp_i in hub_comps:
                    reachable |= components[comp_i]
                surviving = ch_in_ppi - ch_mutated
                iso_frac = len(surviving - reachable) / max(len(surviving), 1)
                # Weight by channel damage
                total_isolation += iso_frac * channel_damage[ci]

            hub_hit = len(ch_mutated & ch_hubs)
            total_hub_damage += (hub_hit / max(len(ch_hubs), 1)) * channel_damage[ci]

        # Normalize by total damage
        if total_damage > 0:
            total_isolation /= total_damage
            total_hub_damage /= total_damage

        # --- 3. Tier connectivity ---
        ppi_nodes = set(G_ppi.nodes()) - mutated
        if ppi_nodes:
            G_d = G_ppi.subgraph(ppi_nodes)
            components = list(nx.connected_components(G_d))
            node_comp = {}
            for comp_i, comp in enumerate(components):
                for g in comp:
                    node_comp[g] = comp_i

            tier_comp_sets = defaultdict(set)
            for g in ppi_nodes:
                t = gene_tier.get(g, -1)
                if t >= 0 and g in node_comp:
                    tier_comp_sets[t].add(node_comp[g])

            tier_connected = 0
            tier_tested = 0
            for t1 in range(4):
                for t2 in range(t1 + 1, 4):
                    tier_tested += 1
                    if tier_comp_sets[t1] & tier_comp_sets[t2]:
                        tier_connected += 1
            tier_conn = tier_connected / max(tier_tested, 1)
        else:
            tier_conn = 0.0

        # --- 4. Co-occurrence features ---
        n_cooccur_pairs = 0
        total_cooccur_weight = 0.0
        max_cooccur_weight = 0.0
        cross_channel_cooccur = 0

        if len(mutated) >= 2:
            sorted_mutated = sorted(mutated)
            for i in range(len(sorted_mutated)):
                for j in range(i + 1, len(sorted_mutated)):
                    key = (sorted_mutated[i], sorted_mutated[j])
                    ct_counts = cooccurrence.get(key, {})
                    w = ct_counts.get(ct_name, 0)
                    if w > 0:
                        n_cooccur_pairs += 1
                        total_cooccur_weight += w
                        if w > max_cooccur_weight:
                            max_cooccur_weight = w
                        # Cross-channel: use primary channel (max weight)
                        p1 = channel_profiles.get(key[0])
                        p2 = channel_profiles.get(key[1])
                        if p1 is not None and p2 is not None:
                            if np.argmax(p1) != np.argmax(p2):
                                cross_channel_cooccur += 1

        # --- 5. Tier damage (aggregated from channel damage) ---
        tier_damage = np.zeros(4)
        for tier_idx, ch_indices in tier_channels.items():
            tier_damage[tier_idx] = sum(channel_damage[ci] for ci in ch_indices)

        # --- 6. Assemble feature vector ---
        X[idx, 0] = ct_bl
        X[idx, 1] = weighted_ch_shift
        X[idx, 2] = ct_gene_shift_sum
        X[idx, 3] = global_gene_shift_sum
        X[idx, 4:12] = channel_damage
        # channel_damage_entropy: skip for now (12-19 reserved, set to 0)
        X[idx, 20] = total_entropy
        X[idx, 21] = n_channels_damaged
        X[idx, 22] = max_ch_damage
        X[idx, 23] = damage_concentration
        X[idx, 24] = total_isolation
        X[idx, 25] = total_hub_damage
        X[idx, 26] = tier_conn
        X[idx, 27] = len(mutated)
        X[idx, 28:32] = tier_damage
        X[idx, 32] = n_cooccur_pairs
        X[idx, 33] = total_cooccur_weight
        X[idx, 34] = max_cooccur_weight
        X[idx, 35] = cross_channel_cooccur
        X[idx, 36:108] = channel_features_all[idx].flatten()
        X[idx, 108:128] = tier_features_all[idx].flatten()
        X[idx, 128] = age_all[idx]
        X[idx, 129] = sex_all[idx]
        X[idx, 130] = msi_all[idx]
        X[idx, 131] = msi_high_all[idx]
        X[idx, 132] = tmb_all[idx]

    return X


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  MULTI-CHANNEL GRAPH SCORER")
    print("  PPI-derived channel membership vectors — genes span multiple channels")
    print("=" * 90)

    # --- Load expanded channel map ---
    expanded_cm = load_expanded_channel_map()
    print(f"\n  Expanded channel map: {len(expanded_cm)} genes")

    # --- Load data ---
    print("  Loading data...")
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
    print(f"  {all_in_val.sum().item()} patients")

    # --- Load mutations ---
    print("  Loading mutations...")
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

    # --- Co-occurrence ---
    print("  Computing co-occurrence...")
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    print(f"  Co-occurring pairs: {len(cooccurrence)}")

    # --- STRING PPI ---
    msk_genes_in_data = set()
    for genes in patient_genes.values():
        msk_genes_in_data |= genes
    msk_genes_in_data &= expanded_genes
    ppi_edges = fetch_string_expanded(msk_genes_in_data)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    print(f"  PPI graph: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # --- Compute multi-channel profiles ---
    print("\n  Computing PPI-derived channel profiles...")
    channel_profiles = compute_channel_profiles(G_ppi, expanded_cm)

    # Report profile statistics
    entropies = []
    n_multi = 0
    for gene, prof in channel_profiles.items():
        e = profile_entropy(prof)
        entropies.append(e)
        if np.sum(prof > 0.1) >= 2:
            n_multi += 1

    print(f"  Genes with profiles: {len(channel_profiles)}")
    print(f"  Multi-channel genes (>1 channel above 10%): {n_multi} ({n_multi/len(channel_profiles)*100:.1f}%)")
    print(f"  Mean entropy: {np.mean(entropies):.3f} bits (max possible: {np.log2(N_CH):.3f})")

    # Show top multi-channel genes
    top_multi = sorted(channel_profiles.items(),
                       key=lambda x: profile_entropy(x[1]), reverse=True)[:15]
    print(f"\n  Top 15 most multi-channel genes (by entropy):")
    header = "  Gene".ljust(12) + "".join(f"{ch[:6]:>8}" for ch in CHANNELS) + "  Entropy"
    print(header)
    print(f"  {'-'*10}" + "".join(f"{'------':>8}" for _ in CHANNELS) + "  -------")
    for gene, prof in top_multi:
        vals = "".join(f"{v:>8.3f}" for v in prof)
        print(f"  {gene:<10}{vals}  {profile_entropy(prof):.3f}")

    # Key genes
    print(f"\n  Key gene profiles:")
    for g in ["TP53", "BRCA1", "KRAS", "APC", "EP300", "BRAF", "PIK3CA", "IDH1"]:
        if g in channel_profiles:
            prof = channel_profiles[g]
            n_ch = np.sum(prof > 0.05)
            top_chs = sorted(range(N_CH), key=lambda i: -prof[i])[:3]
            top_str = ", ".join(f"{CHANNELS[i]}:{prof[i]:.2f}" for i in top_chs if prof[i] > 0.01)
            print(f"    {g:<10} {n_ch} channels: {top_str}  (entropy={profile_entropy(prof):.3f})")

    # --- Compute shifts ---
    print("\n  Computing gene/channel/CT shifts...")
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
        # Use multi-channel profiles for channel membership
        for g in patient_genes.get(idx, set()):
            prof = channel_profiles.get(g)
            if prof is not None:
                for ci, ch_name in enumerate(CHANNELS):
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
        for ch_name in CHANNELS:
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

    print(f"  Global gene shifts: {len(global_gene_shift)}")
    print(f"  CT gene shifts: {len(ct_gene_shift)}")
    print(f"  CT channel shifts: {len(ct_ch_shift)}")

    # =========================================================================
    # Multi-channel graph walk
    # =========================================================================
    print(f"\n  Running multi-channel graph walk...")
    print(f"  Feature vector dimension: 133")

    X_all = multichannel_graph_walk_batch(
        G_ppi, patient_genes, channel_profiles, channel_features, tier_features,
        ct_per_patient, ct_ch_shift, ct_gene_shift, global_gene_shift,
        age, sex, msi, msi_high, tmb, ct_baseline_map, baseline,
        expanded_cm, cooccurrence, N,
    )
    print(f"  Done.")

    # =========================================================================
    # Ablation
    # =========================================================================
    folds = list(skf.split(np.arange(N), events))

    # Feature groups
    mc_graph_idx = list(range(0, 36))        # multi-channel graph features (36)
    ch_idx = list(range(36, 108))            # original channel features (72)
    tier_idx = list(range(108, 128))         # tier features (20)
    clin_idx = list(range(128, 133))         # clinical (5)

    # Subgroups within mc_graph
    mc_damage_idx = list(range(4, 12))       # 8-dim channel damage vector
    mc_meta_idx = [0, 1, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27]  # shifts + topology + meta
    mc_tier_dmg_idx = list(range(28, 32))    # 4-dim tier damage
    mc_cooccur_idx = list(range(32, 36))     # co-occurrence

    configs = {
        "A multichannel graph only":            mc_graph_idx,
        "B mc_graph + original_ch":             mc_graph_idx + ch_idx,
        "C mc_graph + original_ch + tier":      mc_graph_idx + ch_idx + tier_idx,
        "D mc_graph + ch + tier + clin (full)": mc_graph_idx + ch_idx + tier_idx + clin_idx,
        "E mc_damage_vec only (8-dim)":         mc_damage_idx,
        "F mc_damage + meta":                   mc_damage_idx + mc_meta_idx,
        "G mc_damage + meta + cooccur":         mc_damage_idx + mc_meta_idx + mc_cooccur_idx,
        "H ch + tier + clin (no graph)":        ch_idx + tier_idx + clin_idx,
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION: MULTI-CHANNEL GRAPH SCORER")
    print(f"{'='*90}")

    print(f"\n  {'Scorer':<45} {'Mean CI':>8} {'Std':>7}  Folds")
    print(f"  {'-'*45} {'-'*8} {'-'*7}  {'-'*35}")

    results = {}
    for name, feat_indices in configs.items():
        X = X_all[:, feat_indices]
        score = np.zeros(N)
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            vt = valid_mask[train_idx]
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx][vt], hazards[train_idx][vt])
            score[val_idx] = reg.predict(X[val_idx])

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
        print(f"  {name:<45} {mean_ci:>8.4f} {std_ci:>7.4f}  {folds_str}")

    # =========================================================================
    # Per-CT ridge
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE RIDGE FITTING")
    print(f"{'='*90}")

    best_config = "D mc_graph + ch + tier + clin (full)"
    best_indices = configs[best_config]
    X_best = X_all[:, best_indices]

    score_perct = fit_per_ct_ridge(X_best, hazards, valid_mask, events, times,
                                    ct_per_patient, folds, ct_min_patients=200)

    fold_cis_perct = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        ci = concordance_index(
            torch.tensor(score_perct[val_idx].astype(np.float32)),
            torch.tensor(times[val_idx].astype(np.float32)),
            torch.tensor(events[val_idx].astype(np.float32)),
        )
        fold_cis_perct.append(ci)

    mean_perct = np.mean(fold_cis_perct)
    std_perct = np.std(fold_cis_perct)
    folds_str = "  ".join(f"{ci:.4f}" for ci in fold_cis_perct)
    print(f"\n  Per-CT ridge (full multichannel):       {mean_perct:.4f} +/- {std_perct:.4f}")
    print(f"  Folds: {folds_str}")
    results["I per-CT ridge (full multichannel)"] = {
        "mean": mean_perct, "std": std_perct, "folds": fold_cis_perct, "score": score_perct
    }

    # =========================================================================
    # Ridge coefficient analysis
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  RIDGE COEFFICIENT ANALYSIS (full model)")
    print(f"{'='*90}")

    # Fit on all valid data
    valid_idx = np.where(valid_mask)[0]
    reg_full = Ridge(alpha=1.0)
    reg_full.fit(X_best[valid_idx], hazards[valid_idx])

    feat_names = (
        ["ct_baseline", "weighted_ch_shift", "ct_gene_shift", "global_gene_shift"]
        + [f"ch_damage_{ch}" for ch in CHANNELS]
        + [f"ch_dmg_entropy_{ch}" for ch in CHANNELS]  # 12-19 (zeroed)
        + ["total_entropy", "n_ch_damaged", "max_ch_damage", "damage_concentration",
           "isolation", "hub_damage", "tier_conn", "n_mutated"]
        + [f"tier_damage_{i}" for i in range(4)]
        + ["n_cooccur_pairs", "total_cooccur_wt", "max_cooccur_wt", "cross_ch_cooccur"]
        + [f"v6ch{c}_f{f}" for c in range(8) for f in range(9)]
        + [f"tier{t}_f{f}" for t in range(4) for f in range(5)]
        + ["age", "sex", "msi", "msi_high", "tmb"]
    )
    # Only show graph features (first 36 in best_indices)
    graph_feat_names = feat_names[:36]
    coefs = reg_full.coef_[:36]

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    print(f"\n  {'Feature':<30} {'Coeff':>10} {'|Coeff|':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for i in sorted_idx:
        if abs(coefs[i]) > 0.001:
            print(f"  {graph_feat_names[i]:<30} {coefs[i]:>10.4f} {abs(coefs[i]):>10.4f}")

    # =========================================================================
    # Compare to expanded_graph_scorer
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  COMPARISON")
    print(f"{'='*90}")

    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    prev_expanded = 0.6881  # from expanded_graph_scorer per-CT ridge
    prev_single_ch = 0.6673  # from full_graph_scorer (243 genes, single channel)

    print(f"\n  V6c transformer:                       {v6c:.4f}")
    print(f"  Single-channel (243 gene, global):     {prev_single_ch:.4f}")
    print(f"  Single-channel (503 gene, per-CT):     {prev_expanded:.4f}")
    print(f"  Multi-channel (503 gene, per-CT):      {mean_perct:.4f}")
    print(f"  Delta vs single-channel per-CT:        {mean_perct - prev_expanded:+.4f}")
    print(f"  Delta vs V6c:                          {mean_perct - v6c:+.4f}")

    # Per-CT comparison
    print(f"\n  Per-cancer-type comparison (multi-channel vs V6c):")
    print(f"  {'Cancer Type':<35} {'N':>5} {'Multi':>7} {'V6c':>7} {'Delta':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    best_score = score_perct
    ct_results = {}
    graph_wins = 0
    v6c_wins = 0
    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
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
        marker = " ▲" if delta > 0.005 else ""
        ct_results[ct_name] = {"n": len(ct_indices), "multi": float(ci_graph),
                                "v6c": float(ci_v6c), "delta": float(delta)}
        if delta > 0.001:
            graph_wins += 1
        elif delta < -0.001:
            v6c_wins += 1
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_graph:>7.4f} {ci_v6c:>7.4f} "
              f"{delta:>+7.4f}{marker}")

    print(f"\n  Multi-channel wins: {graph_wins}  V6c wins: {v6c_wins}")

    # Save
    save_results = {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                    for k, v in results.items()}
    save_results["per_cancer_type"] = ct_results
    save_results["channel_profiles"] = {
        g: prof.tolist() for g, prof in sorted(channel_profiles.items())
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
