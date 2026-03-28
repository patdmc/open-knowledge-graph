#!/usr/bin/env python3
"""
Diagnose what signal the walk scorer is counteracting.

Hypotheses:
1. Per-gene hazard weighting is worse than equal weighting (mutation count wins)
2. Channel projection destroys unprojected signal
3. CT-specific shrinkage toward wrong-sign global
4. Dropping protective mutations loses real discriminative power
5. Absolute event-rate delta is the wrong hazard metric
"""

import sys, os, time
import numpy as np, pandas as pd, torch
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES, GENE_FUNCTION
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_CHANNEL_NAMES, CHANNEL_FEAT_DIM_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold
from gnn.scripts.expanded_graph_scorer import (
    load_expanded_channel_map, fetch_string_expanded, build_expanded_graph,
    compute_cooccurrence,
)
from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles

CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)
WALK_RESULTS = os.path.join(GNN_RESULTS, "calibrated_walk_scorer")


def main():
    print("=" * 90)
    print("  COUNTERACTION DIAGNOSIS")
    print("=" * 90)

    # Load everything
    walk = np.load(os.path.join(WALK_RESULTS, "scores.npz"))
    damage = walk["scores"]
    channel_damage = walk["channel_scores"]
    damage_noprop = walk["scores_noprop"]
    confidence = walk["confidence"]
    channel_conf = walk["channel_confidence"]

    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    ct_vocab = data["cancer_type_vocab"]
    ct_idx_arr = data["cancer_type_idx"].numpy()

    ct_per_patient = {}
    for idx in range(N):
        ct_per_patient[idx] = (ct_vocab[int(ct_idx_arr[idx])]
                               if int(ct_idx_arr[idx]) < len(ct_vocab) else "Other")

    # Load V6c hazards
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), events))
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        mdl = ChannelNetV6c(config)
        mdl.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        mdl.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {k: data[k][idx] for k in [
                "channel_features", "tier_features", "cancer_type_idx",
                "age", "sex", "msi_score", "msi_high", "tmb",
            ]}
            with torch.no_grad():
                h = mdl(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True
    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()

    # Load mutations
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False, usecols=["patientId", "gene.hugoGeneSymbol"],
    )
    clin = pd.read_csv(os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"), low_memory=False)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["time"] > 0]
    patient_ids = clin["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    expanded_cm = load_expanded_channel_map()
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

    # =========================================================================
    # Test 1: Mutation count vs walk score
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 1: Mutation count as baseline")
    print(f"{'='*90}")

    mut_count = np.zeros(N)
    for idx, genes in patient_genes.items():
        if idx < N:
            mut_count[idx] = len(genes)

    ci_mutcount = concordance_index(
        torch.tensor(mut_count[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    ci_walk = concordance_index(
        torch.tensor(damage[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    ci_noprop = concordance_index(
        torch.tensor(damage_noprop[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    print(f"  Mutation count C-index:    {ci_mutcount:.4f}")
    print(f"  Walk damage C-index:       {ci_walk:.4f}")
    print(f"  Walk no-prop C-index:      {ci_noprop:.4f}")
    print(f"  Walk vs mut count:         {ci_walk - ci_mutcount:+.4f}")

    # =========================================================================
    # Test 2: Equal-weighted mutations vs hazard-weighted
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 2: Equal weight (binary) vs hazard weight per gene")
    print(f"{'='*90}")

    # Equal weight = sum of binary indicators (same as mut count for expanded genes)
    # But let's also check: sum(1) vs sum(hazard_weight) per channel
    # Using channel_profiles, equal weight = sum of profile vectors

    msk_genes = set()
    for genes in patient_genes.values():
        msk_genes |= genes
    msk_genes &= expanded_genes

    ppi_edges = fetch_string_expanded(msk_genes)
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges,
                                     compute_cooccurrence(patient_genes, ct_per_patient, min_count=10))
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)

    all_genes = sorted(G_ppi.nodes())
    n_genes = len(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    profile_matrix = np.zeros((n_genes, N_CH))
    for i, g in enumerate(all_genes):
        prof = channel_profiles.get(g)
        if prof is not None:
            profile_matrix[i] = prof

    # Equal-weight channel scores: binary mutation vector × profile_matrix
    equal_channel = np.zeros((N, N_CH))
    for idx, genes in patient_genes.items():
        if idx < N:
            for g in genes:
                i = gene_to_idx.get(g)
                if i is not None:
                    equal_channel[idx] += profile_matrix[i]

    equal_total = equal_channel.sum(axis=1)

    ci_equal = concordance_index(
        torch.tensor(equal_total[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    print(f"  Equal-weight channel sum:  {ci_equal:.4f}")
    print(f"  Hazard-weight (walk):      {ci_walk:.4f}")
    print(f"  Equal vs hazard:           {ci_equal - ci_walk:+.4f}")

    # Per-channel comparison
    print(f"\n  {'Channel':<20} {'Equal':>8} {'Hazard':>8} {'Diff':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for ci_ch, ch in enumerate(CHANNELS):
        eq_v = equal_channel[valid_mask, ci_ch]
        hz_v = channel_damage[valid_mask, ci_ch]
        if eq_v.std() > 0:
            ci_eq = concordance_index(
                torch.tensor(eq_v.astype(np.float32)),
                torch.tensor(times[valid_mask].astype(np.float32)),
                torch.tensor(events[valid_mask].astype(np.float32)),
            )
        else:
            ci_eq = 0.5
        if hz_v.std() > 0:
            ci_hz = concordance_index(
                torch.tensor(hz_v.astype(np.float32)),
                torch.tensor(times[valid_mask].astype(np.float32)),
                torch.tensor(events[valid_mask].astype(np.float32)),
            )
        else:
            ci_hz = 0.5
        diff = ci_hz - ci_eq
        marker = " <<<" if diff < -0.005 else (" >>>" if diff > 0.005 else "")
        print(f"  {ch:<20} {ci_eq:>8.4f} {ci_hz:>8.4f} {diff:>+8.4f}{marker}")

    # =========================================================================
    # Test 3: Sign confusion — which genes have wrong-sign hazards?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 3: Per-gene hazard sign check")
    print(f"{'='*90}")

    # For each gene: compare (gene mutated vs not) survival using V6c residuals
    # If our hazard says "dangerous" but V6c-residual says "protective", we're counteracting
    global_er = events[valid_mask].mean()
    gene_patients = defaultdict(set)
    for idx, genes in patient_genes.items():
        if valid_mask[idx]:
            for g in genes:
                gene_patients[g].add(idx)

    # Load walk scorer's per-gene hazards from the damage array
    # Reconstruct: for each gene, the hazard = damage contribution
    # Approximate: patients with ONLY that gene → damage ≈ node hazard for that gene
    # Better: compute correlation of gene-presence with damage score direction

    sign_conflicts = []
    for gene in sorted(expanded_genes):
        pts = gene_patients.get(gene, set())
        if len(pts) < 30:
            continue
        gene_idx = np.array(sorted(pts))
        non_gene_mask = valid_mask.copy()
        non_gene_mask[gene_idx] = False
        non_gene_idx = np.where(non_gene_mask)[0]

        # Gene event rate vs population
        gene_er = events[gene_idx].mean()
        pop_er = events[non_gene_idx].mean()
        actual_direction = gene_er - pop_er  # positive = truly dangerous

        # Walk scorer's assessment: mean damage for gene patients vs non-gene
        walk_gene = damage[gene_idx].mean()
        walk_nongene = damage[non_gene_idx].mean()
        walk_direction = walk_gene - walk_nongene  # positive = scorer thinks dangerous

        # V6c's assessment
        v6c_gene = hazards[gene_idx].mean()
        v6c_nongene = hazards[non_gene_idx].mean()
        v6c_direction = v6c_gene - v6c_nongene

        # Check if walk scorer contradicts reality
        if actual_direction != 0 and walk_direction != 0:
            if (actual_direction > 0) != (walk_direction > 0):
                sign_conflicts.append((
                    gene, len(pts), actual_direction, walk_direction,
                    v6c_direction, gene_er, pop_er
                ))

    print(f"  Genes with sign conflicts (walk says opposite of event rate): {len(sign_conflicts)}")
    if sign_conflicts:
        sign_conflicts.sort(key=lambda x: -abs(x[2]) * x[1])
        print(f"\n  {'Gene':<15} {'N':>5} {'Actual':>8} {'Walk':>8} {'V6c':>8} {'GeneER':>7} {'PopER':>7}")
        print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for gene, n, actual, walk_d, v6c_d, g_er, p_er in sign_conflicts[:30]:
            print(f"  {gene:<15} {n:>5} {actual:>+8.4f} {walk_d:>+8.4f} {v6c_d:>+8.4f} {g_er:>7.3f} {p_er:>7.3f}")

    # =========================================================================
    # Test 4: Protective signal magnitude — are we throwing away too much?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 4: What signal does confidence carry?")
    print(f"{'='*90}")

    # Try: damage - confidence (use the full signed signal)
    signed_score = damage - confidence
    ci_signed = concordance_index(
        torch.tensor(signed_score[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    # Try: damage + confidence (both as positive hazard)
    sum_score = damage + confidence
    ci_sum = concordance_index(
        torch.tensor(sum_score[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    # Try: confidence alone (negated — protective = better survival)
    ci_conf_neg = concordance_index(
        torch.tensor((-confidence[valid_mask]).astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    ci_conf_pos = concordance_index(
        torch.tensor(confidence[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )

    print(f"  Damage only:               {ci_walk:.4f}")
    print(f"  Confidence (positive):     {ci_conf_pos:.4f}  (higher conf → worse outcome?)")
    print(f"  Confidence (negated):      {ci_conf_neg:.4f}  (higher conf → better outcome?)")
    print(f"  Damage + confidence:       {ci_sum:.4f}")
    print(f"  Damage - confidence:       {ci_signed:.4f}  (restore original signed hazard)")
    print(f"  Mutation count:            {ci_mutcount:.4f}")

    # =========================================================================
    # Test 5: Correlation structure — what's redundant with mutation count?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 5: Correlation with mutation count")
    print(f"{'='*90}")

    rho_dmg, _ = sp_stats.spearmanr(damage[valid_mask], mut_count[valid_mask])
    rho_conf, _ = sp_stats.spearmanr(confidence[valid_mask], mut_count[valid_mask])
    rho_sum, _ = sp_stats.spearmanr(sum_score[valid_mask], mut_count[valid_mask])
    rho_signed, _ = sp_stats.spearmanr(signed_score[valid_mask], mut_count[valid_mask])
    rho_equal, _ = sp_stats.spearmanr(equal_total[valid_mask], mut_count[valid_mask])

    print(f"  Damage vs mut_count:       rho={rho_dmg:+.4f}")
    print(f"  Confidence vs mut_count:   rho={rho_conf:+.4f}")
    print(f"  Damage+conf vs mut_count:  rho={rho_sum:+.4f}")
    print(f"  Damage-conf vs mut_count:  rho={rho_signed:+.4f}")
    print(f"  Equal-weight vs mut_count: rho={rho_equal:+.4f}")

    # =========================================================================
    # Test 6: Per-CT — where does hazard weighting HURT vs equal weighting?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 6: Per-CT — hazard-weighted vs equal-weighted vs mut count")
    print(f"{'='*90}")

    ct_patients_dict = defaultdict(list)
    for idx in range(N):
        if valid_mask[idx]:
            ct_patients_dict[ct_per_patient[idx]].append(idx)

    print(f"\n  {'Cancer Type':<35} {'N':>5} {'MutCnt':>7} {'Equal':>7} {'Walk':>7} {'V6c':>7} {'Walk-Eq':>8}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    hazard_helps = 0
    hazard_hurts = 0
    big_hurt_cts = []
    for ct_name in sorted(ct_patients_dict, key=lambda x: -len(ct_patients_dict[x])):
        ct_indices = np.array(ct_patients_dict[ct_name])
        if len(ct_indices) < 50:
            continue

        ci_mc = concordance_index(
            torch.tensor(mut_count[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_eq = concordance_index(
            torch.tensor(equal_total[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_wk = concordance_index(
            torch.tensor(damage[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        diff = ci_wk - ci_eq
        if diff < -0.005:
            hazard_hurts += 1
            big_hurt_cts.append((ct_name, len(ct_indices), ci_mc, ci_eq, ci_wk, ci_v, diff))
        elif diff > 0.005:
            hazard_helps += 1
        marker = " <<<" if diff < -0.01 else (" >>>" if diff > 0.01 else "")
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_mc:>7.4f} {ci_eq:>7.4f} {ci_wk:>7.4f} {ci_v:>7.4f} {diff:>+8.4f}{marker}")

    print(f"\n  Hazard weighting helps: {hazard_helps} CTs, hurts: {hazard_hurts} CTs")

    # =========================================================================
    # Test 7: Damage score decomposition — is it just mutation count in disguise?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 7: Residual signal after removing mutation count")
    print(f"{'='*90}")

    # Regress out mutation count from damage, check if residual has signal
    from sklearn.linear_model import LinearRegression
    mc_valid = mut_count[valid_mask].reshape(-1, 1)
    dmg_valid = damage[valid_mask]

    lr = LinearRegression().fit(mc_valid, dmg_valid)
    predicted = lr.predict(mc_valid)
    residual = dmg_valid - predicted

    ci_residual = concordance_index(
        torch.tensor(residual.astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    print(f"  Damage = {lr.coef_[0]:.4f} × mut_count + {lr.intercept_:.4f}")
    print(f"  R² = {lr.score(mc_valid, dmg_valid):.4f}")
    print(f"  Residual (damage - predicted) C-index: {ci_residual:.4f}")
    print(f"  {'Graph adds info beyond mut count' if ci_residual > 0.51 else 'Graph adds NO info beyond mut count'}")

    # Same for equal-weight
    eq_valid = equal_total[valid_mask]
    lr2 = LinearRegression().fit(mc_valid, eq_valid)
    residual2 = eq_valid - lr2.predict(mc_valid)
    ci_residual2 = concordance_index(
        torch.tensor(residual2.astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    print(f"\n  Equal-weight = {lr2.coef_[0]:.4f} × mut_count + {lr2.intercept_:.4f}")
    print(f"  R² = {lr2.score(mc_valid, eq_valid):.4f}")
    print(f"  Equal-weight residual C-index: {ci_residual2:.4f}")
    print(f"  {'Channel profiles add info' if ci_residual2 > 0.51 else 'Channel profiles add NO info beyond mut count'}")

    # =========================================================================
    # Test 8: Check if WHICH genes matter, not how many
    # =========================================================================
    print(f"\n{'='*90}")
    print("  TEST 8: Gene identity signal — top 20 most-mutated genes as binary features")
    print(f"{'='*90}")

    gene_counts = defaultdict(int)
    for idx, genes in patient_genes.items():
        if valid_mask[idx]:
            for g in genes:
                gene_counts[g] += 1

    top_genes = sorted(gene_counts, key=gene_counts.get, reverse=True)[:20]
    print(f"  Top genes: {', '.join(f'{g}({gene_counts[g]})' for g in top_genes[:10])}")

    # Binary features for top genes
    gene_features = np.zeros((N, len(top_genes)))
    for idx, genes in patient_genes.items():
        if idx < N:
            for gi, g in enumerate(top_genes):
                if g in genes:
                    gene_features[idx, gi] = 1.0

    # Each gene alone
    print(f"\n  {'Gene':<15} {'N_mut':>6} {'CI(binary)':>10} {'EventRate':>10} {'PopER':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*7}")
    for gi, g in enumerate(top_genes):
        gf = gene_features[valid_mask, gi]
        if gf.std() > 0:
            ci_g = concordance_index(
                torch.tensor(gf.astype(np.float32)),
                torch.tensor(times[valid_mask].astype(np.float32)),
                torch.tensor(events[valid_mask].astype(np.float32)),
            )
            er = events[valid_mask][gf > 0].mean() if (gf > 0).any() else 0
            print(f"  {g:<15} {gene_counts[g]:>6} {ci_g:>10.4f} {er:>10.3f} {events[valid_mask].mean():>7.3f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
