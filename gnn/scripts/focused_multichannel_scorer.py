#!/usr/bin/env python3
"""
Focused multi-channel scorer: curated-anchor profiles + entropy weighting.

Two key changes from multichannel_graph_scorer.py:
  1. Channel profiles computed using ONLY curated V6 genes (243) as anchors.
     A gene's profile = fraction of PPI neighbors in each channel, but only
     counting neighbors that are in V6_CHANNEL_MAP (reliable labels).
     This prevents noisy comutation-inferred genes from diluting profiles.

  2. Entropy-weighted damage: channel damage is weighted by concentration.
     Focused damage (low entropy, hitting 1-2 channels) is more lethal
     than diffuse damage (high entropy, spread across all channels).
     concentration_weight = 1 + (max_entropy - patient_entropy) / max_entropy

Usage:
    python3 -u -m gnn.scripts.focused_multichannel_scorer
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_FUNCTION, TRUNCATING,
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

SAVE_BASE = os.path.join(GNN_RESULTS, "focused_multichannel_scorer")
CHANNELS = V6_CHANNEL_NAMES  # 8 channels (including epigenetic)
N_CH = len(CHANNELS)
CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS)}
MAX_ENTROPY = np.log2(N_CH)  # 3.0 bits


# =========================================================================
# 1. Curated-anchor channel profiles
# =========================================================================
def compute_curated_profiles(G_ppi, expanded_cm):
    """Channel profiles using ONLY V6 curated genes as label anchors.

    For each gene g in the graph:
      profile[g] = normalized sum of PPI-weighted channel labels
                   from neighbors that are in V6_CHANNEL_MAP.

    Genes with no curated neighbors fall back to their own channel assignment.
    All 8 channels (including ChromatinRemodel, DNAMethylation) are represented
    because V6_CHANNEL_MAP includes those.
    """
    curated_set = set(V6_CHANNEL_MAP.keys())
    profiles = {}

    for gene in set(G_ppi.nodes()) | set(expanded_cm.keys()):
        vec = np.zeros(N_CH)

        if gene in G_ppi:
            for nb in G_ppi.neighbors(gene):
                if nb not in curated_set:
                    continue
                nb_ch = V6_CHANNEL_MAP.get(nb)
                if nb_ch and nb_ch in CH_TO_IDX:
                    w = G_ppi[gene][nb].get("weight", 0.7)
                    vec[CH_TO_IDX[nb_ch]] += w

        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            # No curated neighbors — use own channel assignment
            ch = expanded_cm.get(gene)
            if ch and ch in CH_TO_IDX:
                vec[CH_TO_IDX[ch]] = 1.0
            elif gene in V6_CHANNEL_MAP:
                ch = V6_CHANNEL_MAP[gene]
                vec[CH_TO_IDX[ch]] = 1.0

        profiles[gene] = vec

    return profiles


def profile_entropy(vec):
    """Shannon entropy in bits."""
    v = vec[vec > 0]
    if len(v) <= 1:
        return 0.0
    return float(-np.sum(v * np.log2(v)))


def concentration_weight(entropy):
    """Weight factor: higher for concentrated (low-entropy) damage."""
    # Maps entropy 0 -> weight 2.0, entropy MAX -> weight 1.0
    return 1.0 + (MAX_ENTROPY - entropy) / MAX_ENTROPY


# =========================================================================
# 2. Focused multi-channel graph walk with entropy weighting
# =========================================================================
def focused_graph_walk_batch(
    G_ppi, patient_genes_map, channel_profiles, channel_features_all,
    tier_features_all, ct_per_patient,
    ct_ch_shift, ct_gene_shift, global_gene_shift,
    age_all, sex_all, msi_all, msi_high_all, tmb_all,
    ct_baseline_map, baseline, expanded_cm, cooccurrence, N,
):
    """
    Feature vector (per patient):
      [0]     ct_baseline
      [1]     weighted_ch_shift (CT channel shifts weighted by damage profile)
      [2]     ct_gene_shift_sum
      [3]     global_gene_shift_sum
      [4-11]  channel_damage (8-dim: sum of curated-anchor profiles)
      [12-19] entropy_weighted_channel_damage (channel_damage * concentration_weight)
      [20]    total_entropy
      [21]    concentration_weight
      [22]    n_channels_damaged (count with damage > 0.1)
      [23]    max_channel_damage
      [24]    damage_hhi (Herfindahl: sum of squared damage fracs — higher = more concentrated)
      [25]    isolation (weighted by channel damage)
      [26]    hub_damage (weighted)
      [27]    tier_conn
      [28]    n_mutated_genes
      [29-32] tier_damage (4-dim)
      [33-36] entropy_weighted_tier_damage (4-dim)
      [37-40] co-occurrence features
      [41]    mean_gene_entropy (avg entropy of mutated genes' profiles)
      [42]    max_gene_entropy
      [43]    min_gene_entropy (0 if no mutations)
      [44]    n_curated_anchor_genes (mutated genes that ARE in V6 curated set)
      [45]    frac_curated (fraction of mutated genes in curated set)
      [46-117] channel_features (72)
      [118-137] tier_features (20)
      [138]   age
      [139]   sex
      [140]   msi
      [141]   msi_high
      [142]   tmb
    Total: 143 features
    """
    all_expanded_genes = set(expanded_cm.keys())
    curated_set = set(V6_CHANNEL_MAP.keys())
    FEAT_DIM = 143
    X = np.zeros((N, FEAT_DIM))

    ch_gene_sets = defaultdict(set)
    for g, ch in V6_CHANNEL_MAP.items():
        ch_gene_sets[ch].add(g)
    # Also add expanded genes to their primary channel for isolation calc
    for g, ch in expanded_cm.items():
        if ch in CH_TO_IDX:
            ch_gene_sets[ch].add(g)

    tier_channels = defaultdict(list)
    for ch, tier in V6_TIER_MAP.items():
        if ch in CH_TO_IDX:
            tier_channels[tier].append(CH_TO_IDX[ch])

    gene_tier = {}
    for g, ch in expanded_cm.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)
    for g, ch in V6_CHANNEL_MAP.items():
        gene_tier[g] = V6_TIER_MAP.get(ch, -1)

    for idx in range(N):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx}/{N}...")

        patient_genes = patient_genes_map.get(idx, set())
        mutated = patient_genes & all_expanded_genes
        ct_name = ct_per_patient[idx]
        ct_bl = ct_baseline_map.get(ct_name, baseline)

        # --- 1. Channel damage from curated-anchor profiles ---
        channel_damage = np.zeros(N_CH)
        ct_gene_shift_sum = 0.0
        global_gene_shift_sum = 0.0
        gene_entropies = []
        n_curated = 0

        for g in mutated:
            profile = channel_profiles.get(g)
            if profile is not None:
                channel_damage += profile
                gene_entropies.append(profile_entropy(profile))

            if g in curated_set:
                n_curated += 1

            ct_s = ct_gene_shift.get((ct_name, g), None)
            if ct_s is not None:
                ct_gene_shift_sum += ct_s
            gl_s = global_gene_shift.get(g, None)
            if gl_s is not None:
                global_gene_shift_sum += gl_s

        # Damage distribution metrics
        total_damage = channel_damage.sum()
        if total_damage > 0:
            damage_frac = channel_damage / total_damage
            entropy = profile_entropy(damage_frac)
            conc_w = concentration_weight(entropy)
            hhi = float(np.sum(damage_frac ** 2))
        else:
            damage_frac = np.zeros(N_CH)
            entropy = 0.0
            conc_w = concentration_weight(0.0)
            hhi = 0.0

        # Entropy-weighted channel damage
        ew_channel_damage = channel_damage * conc_w

        # Weighted channel shift
        weighted_ch_shift = 0.0
        if total_damage > 0:
            for ci, ch_name in enumerate(CHANNELS):
                if damage_frac[ci] > 0:
                    weighted_ch_shift += damage_frac[ci] * ct_ch_shift.get((ct_name, ch_name), 0.0)

        n_channels_damaged = np.sum(channel_damage > 0.1)
        max_ch_damage = channel_damage.max() if total_damage > 0 else 0.0

        # Tier damage
        tier_damage = np.zeros(4)
        ew_tier_damage = np.zeros(4)
        for tier_idx, ch_indices in tier_channels.items():
            tier_damage[tier_idx] = sum(channel_damage[ci] for ci in ch_indices)
            ew_tier_damage[tier_idx] = sum(ew_channel_damage[ci] for ci in ch_indices)

        # --- 2. Weighted isolation and hub damage ---
        total_isolation = 0.0
        total_hub_damage = 0.0

        for ci, ch_name in enumerate(CHANNELS):
            if channel_damage[ci] < 0.05:
                continue

            ch_genes = ch_gene_sets.get(ch_name, set())
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
                total_isolation += iso_frac * channel_damage[ci]

            hub_hit = len(ch_mutated & ch_hubs)
            total_hub_damage += (hub_hit / max(len(ch_hubs), 1)) * channel_damage[ci]

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

        # --- 4. Co-occurrence ---
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
                        p1 = channel_profiles.get(key[0])
                        p2 = channel_profiles.get(key[1])
                        if p1 is not None and p2 is not None:
                            if np.argmax(p1) != np.argmax(p2):
                                cross_channel_cooccur += 1

        # --- 5. Gene entropy stats ---
        mean_gene_ent = np.mean(gene_entropies) if gene_entropies else 0.0
        max_gene_ent = max(gene_entropies) if gene_entropies else 0.0
        min_gene_ent = min(gene_entropies) if gene_entropies else 0.0
        frac_curated = n_curated / len(mutated) if mutated else 0.0

        # --- 6. Assemble ---
        X[idx, 0] = ct_bl
        X[idx, 1] = weighted_ch_shift
        X[idx, 2] = ct_gene_shift_sum
        X[idx, 3] = global_gene_shift_sum
        X[idx, 4:12] = channel_damage
        X[idx, 12:20] = ew_channel_damage
        X[idx, 20] = entropy
        X[idx, 21] = conc_w
        X[idx, 22] = n_channels_damaged
        X[idx, 23] = max_ch_damage
        X[idx, 24] = hhi
        X[idx, 25] = total_isolation
        X[idx, 26] = total_hub_damage
        X[idx, 27] = tier_conn
        X[idx, 28] = len(mutated)
        X[idx, 29:33] = tier_damage
        X[idx, 33:37] = ew_tier_damage
        X[idx, 37] = n_cooccur_pairs
        X[idx, 38] = total_cooccur_weight
        X[idx, 39] = max_cooccur_weight
        X[idx, 40] = cross_channel_cooccur
        X[idx, 41] = mean_gene_ent
        X[idx, 42] = max_gene_ent
        X[idx, 43] = min_gene_ent
        X[idx, 44] = n_curated
        X[idx, 45] = frac_curated
        X[idx, 46:118] = channel_features_all[idx].flatten()
        X[idx, 118:138] = tier_features_all[idx].flatten()
        X[idx, 138] = age_all[idx]
        X[idx, 139] = sex_all[idx]
        X[idx, 140] = msi_all[idx]
        X[idx, 141] = msi_high_all[idx]
        X[idx, 142] = tmb_all[idx]

    return X


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  FOCUSED MULTI-CHANNEL SCORER")
    print("  Curated-anchor profiles + entropy-weighted damage")
    print("=" * 90)

    # --- Load ---
    expanded_cm = load_expanded_channel_map()
    print(f"\n  Expanded channel map: {len(expanded_cm)} genes")
    print(f"  V6 curated anchors: {len(V6_CHANNEL_MAP)} genes across {N_CH} channels")

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

    # --- Mutations ---
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

    # --- Co-occurrence + PPI ---
    print("  Computing co-occurrence...")
    cooccurrence = compute_cooccurrence(patient_genes, ct_per_patient, min_count=10)
    msk_genes_in_data = set()
    for genes in patient_genes.values():
        msk_genes_in_data |= genes
    msk_genes_in_data &= expanded_genes
    ppi_edges = fetch_string_expanded(msk_genes_in_data)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    print(f"  PPI graph: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

    # --- Curated-anchor profiles ---
    print("\n  Computing curated-anchor channel profiles...")
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)

    # Profile stats
    entropies = []
    n_multi = 0
    n_epi = 0  # genes with epigenetic channel weight > 0.1
    for gene, prof in channel_profiles.items():
        e = profile_entropy(prof)
        entropies.append(e)
        if np.sum(prof > 0.1) >= 2:
            n_multi += 1
        if prof[CH_TO_IDX["ChromatinRemodel"]] > 0.1 or prof[CH_TO_IDX["DNAMethylation"]] > 0.1:
            n_epi += 1

    print(f"  Genes with profiles: {len(channel_profiles)}")
    print(f"  Multi-channel (>1 ch above 10%): {n_multi} ({n_multi/len(channel_profiles)*100:.1f}%)")
    print(f"  Epigenetic channel signal: {n_epi} genes ({n_epi/len(channel_profiles)*100:.1f}%)")
    print(f"  Mean entropy: {np.mean(entropies):.3f} bits")

    # Key genes: compare curated-anchor vs all-neighbor profiles
    print(f"\n  Key gene curated-anchor profiles:")
    header = "  Gene".ljust(12) + "".join(f"{ch[:6]:>8}" for ch in CHANNELS) + "  Entropy"
    print(header)
    print(f"  {'-'*10}" + "".join(f"{'------':>8}" for _ in CHANNELS) + "  -------")
    for g in ["TP53", "BRCA1", "KRAS", "APC", "EP300", "BRAF", "PIK3CA", "IDH1",
              "CREBBP", "KMT2D", "DNMT3A", "TET2", "ARID1A", "SMARCA4"]:
        if g in channel_profiles:
            prof = channel_profiles[g]
            vals = "".join(f"{v:>8.3f}" for v in prof)
            print(f"  {g:<10}{vals}  {profile_entropy(prof):.3f}")

    # --- Compute shifts ---
    print("\n  Computing shifts...")
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
    print(f"  CT channel shifts: {len(ct_ch_shift)}")
    print(f"  CT gene shifts: {len(ct_gene_shift)}")

    # =========================================================================
    # Graph walk
    # =========================================================================
    print(f"\n  Running focused multi-channel graph walk...")
    print(f"  Feature dimension: 143")

    X_all = focused_graph_walk_batch(
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
    graph_base_idx = [0, 1, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28]  # 13 scalar graph feats
    ch_damage_idx = list(range(4, 12))        # 8-dim raw channel damage
    ew_damage_idx = list(range(12, 20))       # 8-dim entropy-weighted channel damage
    tier_dmg_idx = list(range(29, 33))        # 4-dim raw tier damage
    ew_tier_idx = list(range(33, 37))         # 4-dim entropy-weighted tier damage
    cooccur_idx = list(range(37, 41))         # co-occurrence
    gene_ent_idx = list(range(41, 46))        # gene entropy stats + curated fraction
    v6ch_idx = list(range(46, 118))           # original channel features (72)
    v6tier_idx = list(range(118, 138))        # tier features (20)
    clin_idx = list(range(138, 143))          # clinical (5)

    mc_graph_all = graph_base_idx + ch_damage_idx + ew_damage_idx + tier_dmg_idx + ew_tier_idx + cooccur_idx + gene_ent_idx
    mc_graph_no_ew = graph_base_idx + ch_damage_idx + tier_dmg_idx + cooccur_idx + gene_ent_idx

    configs = {
        "A mc_graph (all new features)":        mc_graph_all,
        "B mc_graph (no entropy weighting)":    mc_graph_no_ew,
        "C mc_graph + v6ch + tier + clin":      mc_graph_all + v6ch_idx + v6tier_idx + clin_idx,
        "D mc_graph_no_ew + v6ch+tier+clin":    mc_graph_no_ew + v6ch_idx + v6tier_idx + clin_idx,
        "E ew_damage only (8-dim)":             ew_damage_idx,
        "F ew_damage + graph_base":             ew_damage_idx + graph_base_idx,
        "G v6ch + tier + clin (no graph)":      v6ch_idx + v6tier_idx + clin_idx,
    }

    print(f"\n{'='*90}")
    print(f"  ABLATION: FOCUSED MULTI-CHANNEL (curated anchors + entropy weighting)")
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
    # Per-CT ridge (best config)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE RIDGE")
    print(f"{'='*90}")

    # Try both full and no-ew
    for label, feat_idx in [
        ("C (with entropy weighting)", mc_graph_all + v6ch_idx + v6tier_idx + clin_idx),
        ("D (without entropy weighting)", mc_graph_no_ew + v6ch_idx + v6tier_idx + clin_idx),
    ]:
        X_best = X_all[:, feat_idx]
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
        print(f"\n  Per-CT ridge {label}:")
        print(f"    {mean_perct:.4f} +/- {std_perct:.4f}  Folds: {folds_str}")
        results[f"perCT_{label}"] = {
            "mean": mean_perct, "std": std_perct, "folds": fold_cis_perct, "score": score_perct,
        }

    # =========================================================================
    # Ridge coefficient analysis (full model with entropy weighting)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  RIDGE COEFFICIENTS (graph features, full model C)")
    print(f"{'='*90}")

    full_idx = mc_graph_all + v6ch_idx + v6tier_idx + clin_idx
    valid_idx = np.where(valid_mask)[0]
    reg_full = Ridge(alpha=1.0)
    reg_full.fit(X_all[valid_idx][:, full_idx], hazards[valid_idx])

    graph_feat_names = (
        ["ct_baseline", "weighted_ch_shift", "ct_gene_shift", "global_gene_shift",
         "entropy", "concentration_wt", "n_ch_damaged", "max_ch_damage",
         "hhi", "isolation", "hub_damage", "tier_conn", "n_mutated"]
        + [f"ch_dmg_{ch}" for ch in CHANNELS]
        + [f"ew_ch_{ch}" for ch in CHANNELS]
        + [f"tier_dmg_{i}" for i in range(4)]
        + [f"ew_tier_{i}" for i in range(4)]
        + ["n_cooccur", "cooccur_wt", "max_cooccur", "cross_ch_cooccur"]
        + ["mean_gene_ent", "max_gene_ent", "min_gene_ent", "n_curated", "frac_curated"]
    )
    n_graph = len(graph_feat_names)
    coefs = reg_full.coef_[:n_graph]

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    print(f"\n  {'Feature':<25} {'Coeff':>10} {'|Coeff|':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for i in sorted_idx:
        if abs(coefs[i]) > 0.001:
            print(f"  {graph_feat_names[i]:<25} {coefs[i]:>10.4f} {abs(coefs[i]):>10.4f}")

    # =========================================================================
    # Comparison
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  COMPARISON")
    print(f"{'='*90}")

    ua_path = os.path.join(GNN_RESULTS, "unified_ablation", "results.json")
    with open(ua_path) as f:
        ua = json.load(f)
    v6c = ua["means"]["V6c"]

    best_name = max(results, key=lambda k: results[k]["mean"])
    best_ci = results[best_name]["mean"]

    print(f"\n  V6c transformer:                       {v6c:.4f}")
    print(f"  Single-channel per-CT (expanded):      0.6881")
    print(f"  Multi-channel per-CT (all-neighbor):   0.6880")
    print(f"  Focused multi-ch (best: {best_name}):")
    print(f"                                         {best_ci:.4f}")
    print(f"  Delta vs V6c:                          {best_ci - v6c:+.4f}")

    # Per-CT breakdown for best scorer
    best_score = results[best_name]["score"]
    print(f"\n  Per-cancer-type (best scorer vs V6c):")
    print(f"  {'Cancer Type':<35} {'N':>5} {'Graph':>7} {'V6c':>7} {'Delta':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    ct_results = {}
    for ct_name in sorted(ct_patients, key=lambda x: -len(ct_patients[x])):
        ct_mask = np.array([ct_per_patient[i] == ct_name for i in range(N)])
        ct_valid = ct_mask & valid_mask
        ct_indices = np.where(ct_valid)[0]
        if len(ct_indices) < 100 or events[ct_indices].sum() < 10:
            continue
        ci_g = concordance_index(
            torch.tensor(best_score[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        delta = ci_g - ci_v
        marker = " ▲" if delta > 0.005 else ""
        ct_results[ct_name] = {"n": len(ct_indices), "graph": float(ci_g),
                                "v6c": float(ci_v), "delta": float(delta)}
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_g:>7.4f} {ci_v:>7.4f} "
              f"{delta:>+7.4f}{marker}")

    # Save
    save_results = {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                    for k, v in results.items()}
    save_results["per_cancer_type"] = ct_results
    save_results["channel_profiles"] = {
        g: channel_profiles[g].tolist() for g in sorted(channel_profiles.keys())
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
