#!/usr/bin/env python3
"""
Hotspot-level synergy analysis: mutation-specific interactions.

Gene-level synergy showed 0 multiplicative pairs because adaptive and
maladaptive hotspots within the same gene cancel out. This analysis
operates at the (gene, proteinChange) level to find specific mutation
combinations that are truly multiplicative (maladaptive) or counteractive.

For each hotspot pair with sufficient co-occurrence:
  expected = shift_a + shift_b
  actual   = mean hazard among patients with both
  synergy  = actual - expected

Classification:
  0 = counteractive  (actual < 0 despite both individually harmful)
  1 = redundant      (synergy < -threshold, overlapping damage)
  2 = additive       (|synergy| < threshold)
  3 = multiplicative (synergy > +threshold, compounding damage)

Usage:
    python3 -u -m gnn.scripts.hotspot_synergy
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS, ANALYSIS_CACHE
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX,
    CHANNEL_FEAT_DIM_V6, V6_TIER_MAP,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "hotspot_synergy")
MIN_HOTSPOT_PATIENTS = 30   # minimum patients with a hotspot to include
MIN_CO_OCCUR = 15           # minimum co-occurrence for pair analysis
SYNERGY_THRESHOLD = 0.05    # threshold for additive vs synergistic/redundant


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 100)
    print("  HOTSPOT-LEVEL SYNERGY ANALYSIS")
    print("  Mutation-specific interactions (gene + proteinChange)")
    print("=" * 100)

    # Load data and model predictions (same as per_mutation_impact)
    print("\nLoading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

    # Out-of-fold predictions from V6c
    print("Computing out-of-fold predictions...")
    events = data["events"].numpy().astype(int)
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
            print(f"  WARNING: missing fold {fold_idx} model")
            continue

        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        batch_size = 2048
        for start in range(0, len(val_idx), batch_size):
            end = min(start + batch_size, len(val_idx))
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

    print(f"  {all_in_val.sum().item()} patients with predictions")

    # Load mutation data with proteinChange
    print("\nLoading mutation data with protein changes...")
    import pandas as pd

    mut_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    mutations = pd.read_csv(mut_path, low_memory=False,
                            usecols=["patientId", "gene.hugoGeneSymbol",
                                     "proteinChange", "mutationType"])

    # Get patient ordering
    patient_ids = data.get("patient_ids")
    if patient_ids is None:
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

    # Filter to channel genes with valid protein changes
    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mutations[
        mutations["gene.hugoGeneSymbol"].isin(channel_genes) &
        mutations["proteinChange"].notna() &
        (mutations["proteinChange"] != "")
    ].copy()

    # Build hotspot key: "GENE proteinChange"
    mut["hotspot"] = mut["gene.hugoGeneSymbol"] + " " + mut["proteinChange"].astype(str)

    # Map patients to indices
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)

    # Find hotspots with enough patients
    hotspot_counts = mut.groupby("hotspot")["patient_idx"].nunique()
    valid_hotspots = hotspot_counts[hotspot_counts >= MIN_HOTSPOT_PATIENTS].index
    print(f"  {len(valid_hotspots)} hotspots with >= {MIN_HOTSPOT_PATIENTS} patients")

    mut_valid = mut[mut["hotspot"].isin(valid_hotspots)]

    # Build per-hotspot patient sets
    hotspot_patients = {}
    for hotspot, group in mut_valid.groupby("hotspot"):
        hotspot_patients[hotspot] = set(group["patient_idx"].unique())

    # Get hotspot metadata
    hotspot_meta = {}
    for hotspot, group in mut_valid.groupby("hotspot"):
        row = group.iloc[0]
        gene = row["gene.hugoGeneSymbol"]
        hotspot_meta[hotspot] = {
            "gene": gene,
            "proteinChange": row["proteinChange"],
            "mutationType": row["mutationType"],
            "channel": V6_CHANNEL_MAP.get(gene, "unknown"),
            "n_patients": len(hotspot_patients[hotspot]),
        }

    # Compute per-hotspot hazard shifts
    hazards = all_hazards.numpy()
    times_np = data["times"].numpy()
    events_np = data["events"].numpy()
    valid_mask = all_in_val.numpy()

    baseline_hazard = hazards[valid_mask].mean()

    print("\nComputing per-hotspot hazard shifts...")
    hotspot_shifts = {}
    for hotspot, patients in hotspot_patients.items():
        pts = np.array(sorted(patients))
        pts_valid = pts[valid_mask[pts]]
        if len(pts_valid) < 10:
            continue
        hotspot_shifts[hotspot] = float(hazards[pts_valid].mean() - baseline_hazard)

    print(f"  {len(hotspot_shifts)} hotspots with valid shifts")

    # =========================================================================
    # Per-hotspot results table
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"  TOP HOTSPOTS BY HAZARD SHIFT")
    print(f"{'='*120}")

    sorted_hotspots = sorted(hotspot_shifts.items(), key=lambda x: -abs(x[1]))

    print(f"  {'Hotspot':<25} {'Channel':<18} {'Type':<15} {'N':>5} "
          f"{'Shift':>8} {'MedSurv':>8} {'EvRate':>7}")
    print(f"  {'-'*25} {'-'*18} {'-'*15} {'-'*5} {'-'*8} {'-'*8} {'-'*7}")

    hotspot_results = []
    for hotspot, shift in sorted_hotspots[:80]:
        meta = hotspot_meta[hotspot]
        pts = np.array(sorted(hotspot_patients[hotspot]))
        pts_valid = pts[valid_mask[pts]]

        med_surv = np.median(times_np[pts_valid])
        ev_rate = events_np[pts_valid].mean()

        print(f"  {hotspot:<25} {meta['channel']:<18} {meta['mutationType']:<15} "
              f"{len(pts_valid):>5} {shift:>+8.4f} {med_surv:>8.1f} {ev_rate:>7.3f}")

        hotspot_results.append({
            "hotspot": hotspot,
            "gene": meta["gene"],
            "proteinChange": meta["proteinChange"],
            "channel": meta["channel"],
            "mutationType": meta["mutationType"],
            "n_patients": len(pts_valid),
            "hazard_shift": shift,
            "median_survival": float(med_surv),
            "event_rate": float(ev_rate),
        })

    # =========================================================================
    # Pairwise hotspot synergy
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"  PAIRWISE HOTSPOT SYNERGY (min co-occur = {MIN_CO_OCCUR})")
    print(f"{'='*120}")

    # Find all pairs with sufficient co-occurrence
    hotspot_list = sorted(hotspot_shifts.keys())
    pair_results = []
    n_checked = 0

    print("  Scanning pairs...")
    for i in range(len(hotspot_list)):
        ha = hotspot_list[i]
        pts_a = hotspot_patients[ha]
        shift_a = hotspot_shifts[ha]

        for j in range(i + 1, len(hotspot_list)):
            hb = hotspot_list[j]
            pts_b = hotspot_patients[hb]

            co_occur = pts_a & pts_b
            co_valid = np.array(sorted(co_occur))
            if len(co_valid) == 0:
                continue
            co_valid = co_valid[valid_mask[co_valid]]
            if len(co_valid) < MIN_CO_OCCUR:
                continue

            n_checked += 1
            shift_b = hotspot_shifts[hb]
            expected = shift_a + shift_b
            actual = float(hazards[co_valid].mean() - baseline_hazard)
            synergy = actual - expected

            # Classification
            if actual < 0 and shift_a > 0 and shift_b > 0:
                interaction = 0  # counteractive
            elif synergy < -SYNERGY_THRESHOLD:
                interaction = 1  # redundant
            elif synergy > SYNERGY_THRESHOLD:
                interaction = 3  # multiplicative
            else:
                interaction = 2  # additive

            # Median survival for co-occurring patients
            med_surv = float(np.median(times_np[co_valid]))
            ev_rate = float(events_np[co_valid].mean())

            meta_a = hotspot_meta[ha]
            meta_b = hotspot_meta[hb]

            pair_results.append({
                "hotspot_a": ha,
                "hotspot_b": hb,
                "gene_a": meta_a["gene"],
                "gene_b": meta_b["gene"],
                "channel_a": meta_a["channel"],
                "channel_b": meta_b["channel"],
                "type_a": meta_a["mutationType"],
                "type_b": meta_b["mutationType"],
                "n_co_occur": len(co_valid),
                "shift_a": shift_a,
                "shift_b": shift_b,
                "expected": expected,
                "actual": actual,
                "synergy": synergy,
                "interaction": interaction,
                "cross_channel": meta_a["channel"] != meta_b["channel"],
                "cross_gene": meta_a["gene"] != meta_b["gene"],
                "median_survival": med_surv,
                "event_rate": ev_rate,
            })

    print(f"  Checked {n_checked} pairs, {len(pair_results)} with >= {MIN_CO_OCCUR} co-occurrence")

    # Interaction classification summary
    interaction_names = {0: "counteractive", 1: "redundant", 2: "additive", 3: "multiplicative"}
    interaction_counts = defaultdict(int)
    for p in pair_results:
        interaction_counts[p["interaction"]] += 1

    print(f"\n  Interaction classification:")
    for code in [0, 1, 2, 3]:
        n = interaction_counts[code]
        pct = 100 * n / max(len(pair_results), 1)
        print(f"    {code} = {interaction_names[code]:<15} {n:>6} ({pct:.1f}%)")

    # =========================================================================
    # Top multiplicative pairs (MALADAPTIVE — what we're looking for)
    # =========================================================================
    mult_pairs = [p for p in pair_results if p["interaction"] == 3]
    mult_pairs.sort(key=lambda x: -x["synergy"])

    print(f"\n{'='*120}")
    print(f"  TOP MULTIPLICATIVE (MALADAPTIVE) PAIRS — synergy > 0")
    print(f"  These mutation combos are WORSE together than sum of individual effects")
    print(f"{'='*120}")

    if mult_pairs:
        print(f"  {'Hotspot A':<22} {'Hotspot B':<22} {'N':>4} "
              f"{'Sh_A':>7} {'Sh_B':>7} {'Exp':>7} {'Act':>7} {'Syn':>7} {'MedS':>6} {'EvR':>5}")
        print(f"  {'-'*22} {'-'*22} {'-'*4} "
              f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")

        for p in mult_pairs[:50]:
            print(f"  {p['hotspot_a']:<22} {p['hotspot_b']:<22} {p['n_co_occur']:>4} "
                  f"{p['shift_a']:>+7.3f} {p['shift_b']:>+7.3f} "
                  f"{p['expected']:>+7.3f} {p['actual']:>+7.3f} {p['synergy']:>+7.3f} "
                  f"{p['median_survival']:>6.1f} {p['event_rate']:>5.2f}")
    else:
        print("  No multiplicative pairs found!")

    # =========================================================================
    # Top counteractive pairs (Mr. Burns syndrome at mutation level)
    # =========================================================================
    counter_pairs = [p for p in pair_results if p["interaction"] == 0]
    counter_pairs.sort(key=lambda x: x["actual"])  # most protective

    print(f"\n{'='*120}")
    print(f"  TOP COUNTERACTIVE PAIRS — both harmful alone, protective together")
    print(f"{'='*120}")

    if counter_pairs:
        print(f"  {'Hotspot A':<22} {'Hotspot B':<22} {'N':>4} "
              f"{'Sh_A':>7} {'Sh_B':>7} {'Exp':>7} {'Act':>7} {'Syn':>7} {'MedS':>6} {'EvR':>5}")
        print(f"  {'-'*22} {'-'*22} {'-'*4} "
              f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")

        for p in counter_pairs[:30]:
            print(f"  {p['hotspot_a']:<22} {p['hotspot_b']:<22} {p['n_co_occur']:>4} "
                  f"{p['shift_a']:>+7.3f} {p['shift_b']:>+7.3f} "
                  f"{p['expected']:>+7.3f} {p['actual']:>+7.3f} {p['synergy']:>+7.3f} "
                  f"{p['median_survival']:>6.1f} {p['event_rate']:>5.2f}")
    else:
        print("  No counteractive pairs found!")

    # =========================================================================
    # Same-gene different-hotspot analysis
    # =========================================================================
    same_gene_pairs = [p for p in pair_results if not p["cross_gene"]]
    same_gene_pairs.sort(key=lambda x: -abs(x["synergy"]))

    print(f"\n{'='*120}")
    print(f"  SAME-GENE DIFFERENT-HOTSPOT PAIRS ({len(same_gene_pairs)} pairs)")
    print(f"  Do different mutations in the same gene interact?")
    print(f"{'='*120}")

    if same_gene_pairs:
        print(f"  {'Hotspot A':<22} {'Hotspot B':<22} {'N':>4} "
              f"{'Exp':>7} {'Act':>7} {'Syn':>7} {'Type':>14}")
        print(f"  {'-'*22} {'-'*22} {'-'*4} "
              f"{'-'*7} {'-'*7} {'-'*7} {'-'*14}")

        for p in same_gene_pairs[:30]:
            itype = interaction_names[p["interaction"]]
            print(f"  {p['hotspot_a']:<22} {p['hotspot_b']:<22} {p['n_co_occur']:>4} "
                  f"{p['expected']:>+7.3f} {p['actual']:>+7.3f} {p['synergy']:>+7.3f} "
                  f"{itype:>14}")

    # =========================================================================
    # Missense × Missense vs Truncating × Truncating vs Mixed
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"  SYNERGY BY MUTATION TYPE COMBINATION")
    print(f"{'='*120}")

    type_combos = defaultdict(lambda: {"synergies": [], "n": 0})
    for p in pair_results:
        ta = "missense" if "Missense" in str(p["type_a"]) else "truncating"
        tb = "missense" if "Missense" in str(p["type_b"]) else "truncating"
        combo = " × ".join(sorted([ta, tb]))
        type_combos[combo]["synergies"].append(p["synergy"])
        type_combos[combo]["n"] += 1

    print(f"  {'Type Combo':<30} {'N':>6} {'Mean Syn':>9} {'Std':>7} "
          f"{'%Multi':>7} {'%Counter':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*9} {'-'*7} {'-'*7} {'-'*8}")

    for combo, stats in sorted(type_combos.items()):
        syns = np.array(stats["synergies"])
        n_multi = (syns > SYNERGY_THRESHOLD).sum()
        n_counter = sum(1 for p in pair_results
                       if p["interaction"] == 0 and
                       " × ".join(sorted([
                           "missense" if "Missense" in str(p["type_a"]) else "truncating",
                           "missense" if "Missense" in str(p["type_b"]) else "truncating",
                       ])) == combo)
        pct_multi = 100 * n_multi / max(len(syns), 1)
        # Count counteractive properly
        combo_pairs = [p for p in pair_results if
                      " × ".join(sorted([
                          "missense" if "Missense" in str(p["type_a"]) else "truncating",
                          "missense" if "Missense" in str(p["type_b"]) else "truncating",
                      ])) == combo]
        n_counter_actual = sum(1 for p in combo_pairs if p["interaction"] == 0)
        pct_counter = 100 * n_counter_actual / max(len(combo_pairs), 1)

        print(f"  {combo:<30} {stats['n']:>6} {np.mean(syns):>+9.4f} "
              f"{np.std(syns):>7.4f} {pct_multi:>6.1f}% {pct_counter:>7.1f}%")

    # =========================================================================
    # Cross-channel vs same-channel
    # =========================================================================
    print(f"\n  Cross-channel pairs:")
    cross = [p for p in pair_results if p["cross_channel"]]
    same = [p for p in pair_results if not p["cross_channel"]]
    if cross:
        cross_syn = [p["synergy"] for p in cross]
        print(f"    N={len(cross)}, mean synergy={np.mean(cross_syn):+.4f}, "
              f"std={np.std(cross_syn):.4f}")
        print(f"    Multiplicative: {sum(1 for p in cross if p['interaction']==3)} "
              f"({100*sum(1 for p in cross if p['interaction']==3)/len(cross):.1f}%)")
    if same:
        same_syn = [p["synergy"] for p in same]
        print(f"  Same-channel pairs:")
        print(f"    N={len(same)}, mean synergy={np.mean(same_syn):+.4f}, "
              f"std={np.std(same_syn):.4f}")
        print(f"    Multiplicative: {sum(1 for p in same if p['interaction']==3)} "
              f"({100*sum(1 for p in same if p['interaction']==3)/len(same):.1f}%)")

    print(f"\n{'='*120}")

    # Save results
    with open(os.path.join(SAVE_BASE, "hotspot_results.json"), "w") as f:
        json.dump(hotspot_results, f, indent=2, default=str)

    with open(os.path.join(SAVE_BASE, "pair_results.json"), "w") as f:
        json.dump(pair_results, f, indent=2, default=str)

    summary = {
        "n_hotspots": len(hotspot_shifts),
        "n_pairs_checked": n_checked,
        "n_pairs_valid": len(pair_results),
        "interaction_counts": {interaction_names[k]: v for k, v in interaction_counts.items()},
        "min_hotspot_patients": MIN_HOTSPOT_PATIENTS,
        "min_co_occur": MIN_CO_OCCUR,
        "synergy_threshold": SYNERGY_THRESHOLD,
    }
    with open(os.path.join(SAVE_BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
