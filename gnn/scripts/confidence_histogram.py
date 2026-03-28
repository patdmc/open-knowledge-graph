#!/usr/bin/env python3
"""
Confidence-stratified analysis of walk scorer.

Key question: for which patients does the graph-based scorer have strong
discriminative power? Even if the overall C-index is 0.59, it might be
0.70+ for a subset — and knowing which subset makes it clinically useful.

Stratify by:
1. Damage score (do patients with more damage signal have better discrimination?)
2. Confidence score (does low confidence = better discrimination?)
3. Damage AND confidence (2D bins)
4. Number of damaging mutations (not just any mutation — damage-source genes)

Usage:
    python3 -u -m gnn.scripts.confidence_histogram
"""

import sys, os, time
import numpy as np, pandas as pd, torch
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, ANALYSIS_CACHE
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_CHANNEL_NAMES, CHANNEL_FEAT_DIM_V6
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

CHANNELS = V6_CHANNEL_NAMES
WALK_RESULTS = os.path.join(GNN_RESULTS, "calibrated_walk_scorer")


def ci_safe(scores, times, events, min_n=30):
    """C-index with minimum patient count."""
    if len(scores) < min_n or events.sum() < 5 or (1 - events).sum() < 5:
        return float('nan')
    return concordance_index(
        torch.tensor(scores.astype(np.float32)),
        torch.tensor(times.astype(np.float32)),
        torch.tensor(events.astype(np.float32)),
    )


def main():
    print("=" * 90)
    print("  CONFIDENCE-STRATIFIED ANALYSIS")
    print("  Question: where does the graph scorer have real signal?")
    print("=" * 90)

    # Load
    walk = np.load(os.path.join(WALK_RESULTS, "scores.npz"))
    damage = walk["scores"]
    confidence = walk["confidence"]
    channel_damage = walk["channel_scores"]
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

    # V6c hazards
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events)):
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

    # Load mutations for mutation count
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
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)
    patient_genes = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_genes[idx].add(row["gene.hugoGeneSymbol"])
    mut_count = np.zeros(N)
    for idx, genes in patient_genes.items():
        if idx < N:
            mut_count[idx] = len(genes)

    d = damage[valid_mask]
    c = confidence[valid_mask]
    t = times[valid_mask]
    e = events[valid_mask]
    h = hazards[valid_mask]
    mc = mut_count[valid_mask]

    # =========================================================================
    # 1. Histogram by DAMAGE score
    # =========================================================================
    print(f"\n{'='*90}")
    print("  1. STRATIFY BY DAMAGE SCORE")
    print(f"{'='*90}")

    # Bin patients by damage score
    # Zero damage = no damaging mutations found
    has_damage = d > 0
    print(f"  Patients with damage > 0: {has_damage.sum()} / {len(d)} ({100*has_damage.mean():.1f}%)")

    ci_no_dmg = ci_safe(d[~has_damage], t[~has_damage], e[~has_damage])
    ci_has_dmg = ci_safe(d[has_damage], t[has_damage], e[has_damage])
    ci_v6c_no_dmg = ci_safe(h[~has_damage], t[~has_damage], e[~has_damage])
    ci_v6c_has_dmg = ci_safe(h[has_damage], t[has_damage], e[has_damage])
    print(f"  No damage (n={int((~has_damage).sum())}): walk={ci_no_dmg:.4f}, V6c={ci_v6c_no_dmg:.4f}")
    print(f"  Has damage (n={int(has_damage.sum())}): walk={ci_has_dmg:.4f}, V6c={ci_v6c_has_dmg:.4f}")

    # Finer bins among patients WITH damage
    d_pos = d[has_damage]
    t_pos = t[has_damage]
    e_pos = e[has_damage]
    h_pos = h[has_damage]

    percentiles = [0, 25, 50, 75, 90, 95, 100]
    thresholds = np.percentile(d_pos, percentiles)

    print(f"\n  {'Damage Bin':<25} {'N':>6} {'Walk CI':>8} {'V6c CI':>8} {'Gap':>7} {'ER':>6} {'MeanDmg':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")

    for i in range(len(percentiles) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == len(percentiles) - 2:
            mask = (d_pos >= lo)
        else:
            mask = (d_pos >= lo) & (d_pos < hi)
        if mask.sum() < 30:
            continue
        ci_w = ci_safe(d_pos[mask], t_pos[mask], e_pos[mask])
        ci_v = ci_safe(h_pos[mask], t_pos[mask], e_pos[mask])
        gap = ci_v - ci_w if not (np.isnan(ci_w) or np.isnan(ci_v)) else float('nan')
        er = e_pos[mask].mean()
        label = f"p{percentiles[i]}-p{percentiles[i+1]} [{lo:.3f},{hi:.3f})"
        print(f"  {label:<25} {mask.sum():>6} {ci_w:>8.4f} {ci_v:>8.4f} {gap:>+7.4f} {er:>6.3f} {d_pos[mask].mean():>8.4f}")

    # =========================================================================
    # 2. Histogram by CONFIDENCE score
    # =========================================================================
    print(f"\n{'='*90}")
    print("  2. STRATIFY BY CONFIDENCE SCORE")
    print(f"{'='*90}")

    has_conf = c > 0
    no_conf = ~has_conf

    print(f"  Patients with confidence > 0: {has_conf.sum()} ({100*has_conf.mean():.1f}%)")
    print(f"  Patients with NO confidence signal: {no_conf.sum()} ({100*no_conf.mean():.1f}%)")

    # Among patients with confidence, bin by decile
    c_pos = c[has_conf]
    d_conf = d[has_conf]
    t_conf = t[has_conf]
    e_conf = e[has_conf]
    h_conf = h[has_conf]

    # Also show no-confidence group
    ci_w0 = ci_safe(d[no_conf], t[no_conf], e[no_conf])
    ci_v0 = ci_safe(h[no_conf], t[no_conf], e[no_conf])
    print(f"\n  No confidence signal (n={int(no_conf.sum())}): walk={ci_w0:.4f}, V6c={ci_v0:.4f}, gap={ci_v0-ci_w0:+.4f}")

    decile_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    conf_thresholds = np.percentile(c_pos, decile_pcts)

    print(f"\n  {'Confidence Decile':<30} {'N':>6} {'Walk CI':>8} {'V6c CI':>8} {'Gap':>7} {'ER':>6} {'MeanConf':>9}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*9}")

    for i in range(len(decile_pcts) - 1):
        lo, hi = conf_thresholds[i], conf_thresholds[i + 1]
        if i == len(decile_pcts) - 2:
            mask = (c_pos >= lo)
        else:
            mask = (c_pos >= lo) & (c_pos < hi)
        if mask.sum() < 30:
            continue
        ci_w = ci_safe(d_conf[mask], t_conf[mask], e_conf[mask])
        ci_v = ci_safe(h_conf[mask], t_conf[mask], e_conf[mask])
        gap = ci_v - ci_w if not (np.isnan(ci_w) or np.isnan(ci_v)) else float('nan')
        er = e_conf[mask].mean()
        label = f"d{decile_pcts[i]}-d{decile_pcts[i+1]} [{lo:.2f},{hi:.2f})"
        print(f"  {label:<30} {mask.sum():>6} {ci_w:>8.4f} {ci_v:>8.4f} {gap:>+7.4f} {er:>6.3f} {c_pos[mask].mean():>9.4f}")

    # =========================================================================
    # 3. 2D: Damage × Confidence bins
    # =========================================================================
    print(f"\n{'='*90}")
    print("  3. 2D STRATIFICATION: Damage × Confidence")
    print(f"{'='*90}")

    # Damage: zero / low / mid / high
    # Confidence: zero / low / high
    d_zero = d == 0
    d_lo = (d > 0) & (d <= np.percentile(d[d > 0], 50))
    d_hi = d > np.percentile(d[d > 0], 50)

    c_zero = c == 0
    c_lo = (c > 0) & (c <= np.percentile(c[c > 0], 50))
    c_hi = c > np.percentile(c[c > 0], 50)

    print(f"\n  {'Damage':<10} {'Confidence':<12} {'N':>6} {'Walk CI':>8} {'V6c CI':>8} {'Gap':>7} {'ER':>6}")
    print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")

    for d_label, d_mask in [("zero", d_zero), ("low", d_lo), ("high", d_hi)]:
        for c_label, c_mask in [("zero", c_zero), ("low", c_lo), ("high", c_hi)]:
            mask = d_mask & c_mask
            if mask.sum() < 30:
                continue
            ci_w = ci_safe(d[mask], t[mask], e[mask])
            ci_v = ci_safe(h[mask], t[mask], e[mask])
            gap = ci_v - ci_w if not (np.isnan(ci_w) or np.isnan(ci_v)) else float('nan')
            er = e[mask].mean()
            print(f"  {d_label:<10} {c_label:<12} {mask.sum():>6} {ci_w:>8.4f} {ci_v:>8.4f} {gap:>+7.4f} {er:>6.3f}")

    # =========================================================================
    # 4. Focus on HIGH-DAMAGE patients — this is where the graph SHOULD shine
    # =========================================================================
    print(f"\n{'='*90}")
    print("  4. HIGH-DAMAGE PATIENTS — Graph's best case")
    print(f"{'='*90}")

    # Top 20% by damage score
    d_thresh_80 = np.percentile(d[d > 0], 80)
    high_dmg = d >= d_thresh_80
    n_high = high_dmg.sum()

    ci_walk_high = ci_safe(d[high_dmg], t[high_dmg], e[high_dmg])
    ci_v6c_high = ci_safe(h[high_dmg], t[high_dmg], e[high_dmg])
    print(f"  Top 20% damage (n={n_high}, threshold={d_thresh_80:.4f}):")
    print(f"    Walk C-index: {ci_walk_high:.4f}")
    print(f"    V6c C-index:  {ci_v6c_high:.4f}")
    print(f"    Gap:          {ci_v6c_high - ci_walk_high:+.4f}")

    # Per-CT among high-damage patients
    ct_arr = np.array([ct_per_patient[i] for i in range(N)])[valid_mask]
    ct_high = ct_arr[high_dmg]
    unique_cts = sorted(set(ct_high))

    print(f"\n  Per-CT among high-damage patients:")
    print(f"  {'Cancer Type':<35} {'N':>5} {'Walk':>7} {'V6c':>7} {'Gap':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    walk_wins_high = 0
    for ct_name in unique_cts:
        ct_mask = high_dmg & (ct_arr == ct_name)
        if ct_mask.sum() < 30:
            continue
        ci_w = ci_safe(d[ct_mask], t[ct_mask], e[ct_mask])
        ci_v = ci_safe(h[ct_mask], t[ct_mask], e[ct_mask])
        if np.isnan(ci_w) or np.isnan(ci_v):
            continue
        gap = ci_v - ci_w
        marker = " >>>" if gap < -0.01 else ""
        if gap < 0:
            walk_wins_high += 1
        print(f"  {ct_name:<35} {ct_mask.sum():>5} {ci_w:>7.4f} {ci_v:>7.4f} {gap:>+7.4f}{marker}")

    print(f"\n  Walk wins among high-damage: {walk_wins_high}")

    # =========================================================================
    # 5. Spearman correlation by damage bin
    # =========================================================================
    print(f"\n{'='*90}")
    print("  5. WALK vs V6c CORRELATION BY DAMAGE BIN")
    print(f"{'='*90}")

    print(f"\n  {'Bin':<20} {'N':>6} {'Spearman':>9} {'p-value':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*10}")

    for label, mask in [("no damage", d == 0),
                         ("low damage", (d > 0) & (d <= np.percentile(d[d>0], 33))),
                         ("mid damage", (d > np.percentile(d[d>0], 33)) & (d <= np.percentile(d[d>0], 66))),
                         ("high damage", d > np.percentile(d[d>0], 66))]:
        if mask.sum() < 30:
            continue
        if d[mask].std() > 0 and h[mask].std() > 0:
            rho, p = sp_stats.spearmanr(d[mask], h[mask])
            print(f"  {label:<20} {mask.sum():>6} {rho:>+9.4f} {p:>10.2e}")

    # =========================================================================
    # 6. What percentage of patients have "actionable" graph signal?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  6. ACTIONABLE SIGNAL — patients with damage in known-mechanism genes")
    print(f"{'='*90}")

    # Patients with at least one damaging mutation (damage > 0)
    # + patients with high damage (top quartile)
    # + patients where walk score rank differs from V6c rank by > 1000
    walk_rank = sp_stats.rankdata(d)
    v6c_rank = sp_stats.rankdata(h)
    rank_diff = np.abs(walk_rank - v6c_rank)

    print(f"\n  Total valid patients: {len(d)}")
    print(f"  Damage = 0 (no signal):      {(d == 0).sum():>6} ({100*(d == 0).mean():.1f}%)")
    print(f"  Damage > 0:                  {(d > 0).sum():>6} ({100*(d > 0).mean():.1f}%)")
    print(f"  Damage > median:             {(d > np.median(d[d>0])).sum():>6} "
          f"({100*(d > np.median(d[d>0])).mean():.1f}%)")
    print(f"  Rank diff > 5000 with V6c:   {(rank_diff > 5000).sum():>6} "
          f"({100*(rank_diff > 5000).mean():.1f}%)")

    # Where walk and V6c AGREE vs DISAGREE
    # Agreement = similar ranking direction
    walk_above_median = d > np.median(d)
    v6c_above_median = h > np.median(h)
    agree = walk_above_median == v6c_above_median
    disagree = ~agree

    ci_agree = ci_safe(d[agree], t[agree], e[agree])
    ci_disagree = ci_safe(d[disagree], t[disagree], e[disagree])
    ci_v_agree = ci_safe(h[agree], t[agree], e[agree])
    ci_v_disagree = ci_safe(h[disagree], t[disagree], e[disagree])

    print(f"\n  Walk & V6c agree (n={agree.sum()}):    walk={ci_agree:.4f}  V6c={ci_v_agree:.4f}")
    print(f"  Walk & V6c disagree (n={disagree.sum()}): walk={ci_disagree:.4f}  V6c={ci_v_disagree:.4f}")

    # =========================================================================
    # 7. Clinical value proposition — how many patients does the graph help?
    # =========================================================================
    print(f"\n{'='*90}")
    print("  7. CLINICAL VALUE — where graph + V6c > V6c alone")
    print(f"{'='*90}")

    # Ensemble: normalize both, average
    d_norm = (d - d.mean()) / (d.std() + 1e-8)
    h_norm = (h - h.mean()) / (h.std() + 1e-8)

    for w_walk in [0.1, 0.2, 0.3, 0.5]:
        ensemble = (1 - w_walk) * h_norm + w_walk * d_norm
        ci_ens = ci_safe(ensemble, t, e)
        ci_v_all = ci_safe(h, t, e)
        print(f"  Ensemble (walk={w_walk:.1f}): C-index={ci_ens:.4f}  "
              f"vs V6c alone={ci_v_all:.4f}  delta={ci_ens-ci_v_all:+.4f}")

    # Per-CT ensemble
    print(f"\n  Per-CT ensemble (walk=0.2):")
    ensemble_02 = 0.8 * h_norm + 0.2 * d_norm
    ct_arr = np.array([ct_per_patient[i] for i in range(N)])[valid_mask]

    print(f"  {'Cancer Type':<35} {'N':>5} {'V6c':>7} {'Ens':>7} {'Delta':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    ens_wins = 0
    for ct_name in sorted(set(ct_arr)):
        ct_mask = ct_arr == ct_name
        if ct_mask.sum() < 50:
            continue
        ci_v = ci_safe(h[ct_mask], t[ct_mask], e[ct_mask])
        ci_e = ci_safe(ensemble_02[ct_mask], t[ct_mask], e[ct_mask])
        if np.isnan(ci_v) or np.isnan(ci_e):
            continue
        delta = ci_e - ci_v
        marker = " >>>" if delta > 0.005 else ""
        if delta > 0:
            ens_wins += 1
        print(f"  {ct_name:<35} {ct_mask.sum():>5} {ci_v:>7.4f} {ci_e:>7.4f} {delta:>+7.4f}{marker}")
    print(f"\n  Ensemble improves: {ens_wins} CTs")

    print("\n  Done.")


if __name__ == "__main__":
    main()
