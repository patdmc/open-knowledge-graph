#!/usr/bin/env python3
"""
Cox regression on walk scorer outputs (damage + confidence).

Damage = CT-specific hazard propagated through PPI graph.
Confidence = outcome compression signal from protective mutations.
  - High confidence = narrow survival window = less discriminative power.
  - Enters cox as variance indicator, not second hazard axis.

Models tested:
  1. Damage only (baseline)
  2. Damage + confidence (additive)
  3. Damage + damage×confidence interaction (multiplicative dampening)
  4. Per-channel (8 damage + 8 confidence = 16 features)
  5. Per-channel + interactions (8 damage + 8 conf + 8 interactions = 24 features)

Usage:
    python3 -u -m gnn.scripts.cox_walk_scorer
"""

import sys, os, time, json
import numpy as np, pandas as pd, torch
from collections import defaultdict
from scipy import stats as sp_stats
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import V6_CHANNEL_NAMES
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "cox_walk_scorer")
CHANNELS = V6_CHANNEL_NAMES
N_CH = len(CHANNELS)

WALK_RESULTS = os.path.join(GNN_RESULTS, "calibrated_walk_scorer")


def load_walk_scores():
    """Load precomputed damage + confidence from walk scorer."""
    d = np.load(os.path.join(WALK_RESULTS, "scores.npz"))
    return {
        "damage": d["scores"],
        "channel_damage": d["channel_scores"],
        "damage_noprop": d["scores_noprop"],
        "confidence": d["confidence"],
        "channel_confidence": d["channel_confidence"],
    }


def build_features(walk, indices, mode="damage_only"):
    """
    Build feature matrix for cox regression.

    Modes:
      damage_only: [damage]
      additive: [damage, confidence]
      interaction: [damage, confidence, damage×confidence]
      channel: [8 damage channels, 8 confidence channels]
      channel_interaction: [8 damage, 8 conf, 8 damage×conf]
    """
    if mode == "damage_only":
        return walk["damage"][indices].reshape(-1, 1)

    elif mode == "additive":
        d = walk["damage"][indices]
        c = walk["confidence"][indices]
        return np.column_stack([d, c])

    elif mode == "interaction":
        d = walk["damage"][indices]
        c = walk["confidence"][indices]
        return np.column_stack([d, c, d * c])

    elif mode == "channel":
        cd = walk["channel_damage"][indices]   # (n, 8)
        cc = walk["channel_confidence"][indices]  # (n, 8)
        return np.column_stack([cd, cc])

    elif mode == "channel_interaction":
        cd = walk["channel_damage"][indices]
        cc = walk["channel_confidence"][indices]
        return np.column_stack([cd, cc, cd * cc])

    else:
        raise ValueError(f"Unknown mode: {mode}")


def fit_cox_elastic(X_train, y_train, X_val, alpha_min_ratio=0.01):
    """
    Fit CoxNet (elastic net regularized Cox) with alpha selection.
    Returns predicted risk scores for validation set.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)

    # Remove zero-variance columns
    var = X_tr.var(axis=0)
    keep = var > 1e-10
    if keep.sum() == 0:
        return np.zeros(len(X_val))
    X_tr = X_tr[:, keep]
    X_va = X_va[:, keep]

    # CoxNet with path of alphas
    try:
        cox = CoxnetSurvivalAnalysis(
            l1_ratio=0.5,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=1000,
            n_alphas=20,
        )
        cox.fit(X_tr, y_train)

        # Pick best alpha via internal deviance (use last = most regularized that converged)
        # Actually, pick middle of path for reasonable regularization
        n_alphas = len(cox.alphas_)
        best_idx = max(0, n_alphas // 3)  # moderate regularization
        coef = cox.coef_[:, best_idx]

        # Predict: risk = X @ coef
        risk = X_va @ coef
        return risk
    except Exception as e:
        # Fallback: simple CoxPH
        try:
            cox = CoxPHSurvivalAnalysis(alpha=0.1)
            cox.fit(X_tr, y_train)
            return cox.predict(X_va)
        except Exception:
            return np.zeros(len(X_val))


def make_survival_y(times, events, indices):
    """Create structured array for sksurv."""
    dt = np.dtype([("event", bool), ("time", float)])
    y = np.empty(len(indices), dtype=dt)
    y["event"] = events[indices].astype(bool)
    y["time"] = times[indices]
    return y


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t0 = time.time()

    print("=" * 90)
    print("  COX REGRESSION ON WALK SCORER — Damage + Confidence")
    print("=" * 90)

    # Load data
    walk = load_walk_scores()
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

    # V6c baseline
    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    from gnn.data.channel_dataset_v6 import CHANNEL_FEAT_DIM_V6
    from gnn.models.channel_net_v6c import ChannelNetV6c
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

    print(f"  {N} patients, {valid_mask.sum()} valid")
    print(f"  Walk damage: mean={walk['damage'][valid_mask].mean():.4f}, "
          f"std={walk['damage'][valid_mask].std():.4f}")
    print(f"  Walk confidence: mean={walk['confidence'][valid_mask].mean():.4f}, "
          f"std={walk['confidence'][valid_mask].std():.4f}")

    # =========================================================================
    # Cross-validated Cox regression with multiple feature sets
    # =========================================================================

    MODES = [
        "damage_only",
        "additive",
        "interaction",
        "channel",
        "channel_interaction",
    ]

    results = {mode: {"scores": np.zeros(N), "fold_cis": []} for mode in MODES}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  --- Fold {fold_idx} ---")

        # Filter to valid patients
        train_valid = train_idx[valid_mask[train_idx]]
        val_valid = val_idx[valid_mask[val_idx]]

        if len(train_valid) < 100 or len(val_valid) < 50:
            print(f"    Skipping fold {fold_idx}: too few valid patients")
            continue

        y_train = make_survival_y(times, events, train_valid)
        y_val = make_survival_y(times, events, val_valid)

        for mode in MODES:
            X_train = build_features(walk, train_valid, mode)
            X_val = build_features(walk, val_valid, mode)

            risk = fit_cox_elastic(X_train, y_train, X_val)
            results[mode]["scores"][val_valid] = risk

            ci = concordance_index(
                torch.tensor(risk.astype(np.float32)),
                torch.tensor(times[val_valid].astype(np.float32)),
                torch.tensor(events[val_valid].astype(np.float32)),
            )
            results[mode]["fold_cis"].append(ci)

        fold_str = "  ".join(f"{m}={results[m]['fold_cis'][-1]:.4f}" for m in MODES)
        print(f"    {fold_str}")

    # =========================================================================
    # Global results
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  GLOBAL RESULTS")
    print(f"{'='*90}")

    # Raw walk damage baseline (no cox)
    ci_raw_global = concordance_index(
        torch.tensor(walk["damage"][valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )
    ci_v6c = concordance_index(
        torch.tensor(hazards[valid_mask].astype(np.float32)),
        torch.tensor(times[valid_mask].astype(np.float32)),
        torch.tensor(events[valid_mask].astype(np.float32)),
    )

    print(f"\n  {'Model':<30} {'C-index':>8} {'Folds':>40} {'vs Raw':>8} {'vs V6c':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*40} {'-'*8} {'-'*8}")
    print(f"  {'Raw walk damage':<30} {ci_raw_global:>8.4f} {'':>40} {'':>8} {ci_raw_global-ci_v6c:>+8.4f}")
    print(f"  {'V6c transformer':<30} {ci_v6c:>8.4f} {'':>40} {'':>8} {'':>8}")

    for mode in MODES:
        ci_global = concordance_index(
            torch.tensor(results[mode]["scores"][valid_mask].astype(np.float32)),
            torch.tensor(times[valid_mask].astype(np.float32)),
            torch.tensor(events[valid_mask].astype(np.float32)),
        )
        fold_str = " ".join(f"{c:.4f}" for c in results[mode]["fold_cis"])
        print(f"  {('Cox: ' + mode):<30} {ci_global:>8.4f} {fold_str:>40} "
              f"{ci_global-ci_raw_global:>+8.4f} {ci_global-ci_v6c:>+8.4f}")

    # =========================================================================
    # Per-CT analysis for best model
    # =========================================================================

    # Find best mode
    best_mode = max(MODES, key=lambda m: np.mean(results[m]["fold_cis"]))
    print(f"\n  Best model: Cox: {best_mode}")

    best_scores = results[best_mode]["scores"]
    ct_patients_dict = defaultdict(list)
    for idx in range(N):
        if valid_mask[idx]:
            ct_patients_dict[ct_per_patient[idx]].append(idx)

    print(f"\n{'='*90}")
    print(f"  PER-CANCER-TYPE — Cox: {best_mode}")
    print(f"{'='*90}")
    print(f"\n  {'Cancer Type':<35} {'N':>5} {'CoxWalk':>8} {'RawWalk':>8} {'V6c':>8} {'CoxGap':>8}")
    print(f"  {'-'*35} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    walk_wins = 0
    v6c_wins = 0
    ct_results = {}
    for ct_name in sorted(ct_patients_dict, key=lambda x: -len(ct_patients_dict[x])):
        ct_indices = np.array(ct_patients_dict[ct_name])
        if len(ct_indices) < 50:
            continue

        ci_cox = concordance_index(
            torch.tensor(best_scores[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_raw = concordance_index(
            torch.tensor(walk["damage"][ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        ci_v = concordance_index(
            torch.tensor(hazards[ct_indices].astype(np.float32)),
            torch.tensor(times[ct_indices].astype(np.float32)),
            torch.tensor(events[ct_indices].astype(np.float32)),
        )
        gap = ci_v - ci_cox
        if gap < 0:
            walk_wins += 1
        else:
            v6c_wins += 1
        marker = " >>>" if gap < -0.01 else ""
        print(f"  {ct_name:<35} {len(ct_indices):>5} {ci_cox:>8.4f} {ci_raw:>8.4f} "
              f"{ci_v:>8.4f} {gap:>+8.4f}{marker}")
        ct_results[ct_name] = {"cox": ci_cox, "raw": ci_raw, "v6c": ci_v, "n": len(ct_indices)}

    print(f"\n  Cox walk wins: {walk_wins}, V6c wins: {v6c_wins}")

    # =========================================================================
    # Confidence as variance indicator analysis
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  CONFIDENCE AS VARIANCE INDICATOR")
    print(f"{'='*90}")

    # Bin patients by confidence level, check damage C-index in each bin
    conf = walk["confidence"][valid_mask]
    damage = walk["damage"][valid_mask]
    t_valid = times[valid_mask]
    e_valid = events[valid_mask]

    # Quartile bins among patients with confidence > 0
    has_conf = conf > 0
    if has_conf.sum() > 200:
        conf_positive = conf[has_conf]
        q25, q50, q75 = np.percentile(conf_positive, [25, 50, 75])

        bins = [
            ("no confidence", conf == 0),
            (f"low (0, {q25:.2f}]", (conf > 0) & (conf <= q25)),
            (f"mid ({q25:.2f}, {q50:.2f}]", (conf > q25) & (conf <= q50)),
            (f"high ({q50:.2f}, {q75:.2f}]", (conf > q50) & (conf <= q75)),
            (f"very high (>{q75:.2f})", conf > q75),
        ]

        print(f"\n  {'Confidence Bin':<30} {'N':>6} {'Dmg C-idx':>10} {'Time Std':>10} {'Dmg Std':>10}")
        print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

        for label, mask in bins:
            if mask.sum() < 30:
                continue
            ci_bin = concordance_index(
                torch.tensor(damage[mask].astype(np.float32)),
                torch.tensor(t_valid[mask].astype(np.float32)),
                torch.tensor(e_valid[mask].astype(np.float32)),
            )
            t_std = t_valid[mask].std()
            d_std = damage[mask].std()
            print(f"  {label:<30} {mask.sum():>6} {ci_bin:>10.4f} {t_std:>10.1f} {d_std:>10.4f}")

    # Spearman: confidence vs |residual from median|
    abs_resid = np.abs(t_valid - np.median(t_valid))
    if conf.std() > 0:
        rho, p = sp_stats.spearmanr(conf, abs_resid)
        print(f"\n  Confidence vs |residual from median|: rho={rho:+.4f} p={p:.2e}")
        print(f"    {'Confirms variance narrowing' if rho < 0 else 'Does NOT confirm narrowing'}")

    # Spearman: confidence vs time variance within damage deciles
    damage_deciles = pd.qcut(damage, 10, labels=False, duplicates="drop")
    decile_df = pd.DataFrame({
        "damage_decile": damage_deciles,
        "confidence": conf,
        "time": t_valid,
    })
    decile_stats = decile_df.groupby("damage_decile").agg(
        mean_conf=("confidence", "mean"),
        time_std=("time", "std"),
        n=("time", "count"),
    )
    rho_d, p_d = sp_stats.spearmanr(decile_stats["mean_conf"], decile_stats["time_std"])
    print(f"\n  Within damage deciles — mean_conf vs time_std:")
    print(f"    rho={rho_d:+.4f} p={p_d:.3f}")
    print(f"    {'Higher confidence → lower time variance' if rho_d < 0 else 'No clear pattern'}")

    # =========================================================================
    # Save
    # =========================================================================
    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.0f}s")

    save_data = {
        "best_mode": best_mode,
        "raw_damage_ci": float(ci_raw_global),
        "v6c_ci": float(ci_v6c),
    }
    for mode in MODES:
        ci_g = concordance_index(
            torch.tensor(results[mode]["scores"][valid_mask].astype(np.float32)),
            torch.tensor(times[valid_mask].astype(np.float32)),
            torch.tensor(events[valid_mask].astype(np.float32)),
        )
        save_data[f"cox_{mode}_ci"] = float(ci_g)
        save_data[f"cox_{mode}_fold_cis"] = [float(c) for c in results[mode]["fold_cis"]]

    save_data["ct_results"] = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                    for kk, vv in v.items()}
                                for k, v in ct_results.items()}

    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    # Save best scores
    np.savez_compressed(
        os.path.join(SAVE_BASE, "scores.npz"),
        **{f"cox_{mode}": results[mode]["scores"] for mode in MODES},
    )
    print("  Done.")


if __name__ == "__main__":
    main()
