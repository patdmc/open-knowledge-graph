#!/usr/bin/env python3
"""
Mutation synergy analysis (Experiment 8).

For every patient, classify their mutation profile by channel pattern,
compute the expected hazard from sum of individual gene shifts, and
compare to the model's actual prediction.

The gap = interaction effect:
  > 0: synergistic (worse together than expected from individual genes)
  < 0: redundant (overlapping damage, not additive)

Groups patients by channel combination pattern to find which multi-channel
disruptions produce the strongest synergies.

Also analyzes gene pairs for pairwise synergy.

Usage:
    python3 -u -m gnn.scripts.mutation_synergy
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.data.channel_dataset_v6 import CHANNEL_FEAT_DIM_V6
from gnn.training.metrics import concordance_index
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "mutation_synergy")

TIER_NAMES = {0: "Cell Intrinsic", 1: "Tissue Level", 2: "Organism Level", 3: "Meta Regulatory"}

# Minimum patients to analyze a group
MIN_GROUP = 30
MIN_CO_OCCUR = 30


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 100)
    print("  MUTATION SYNERGY ANALYSIS")
    print("  Per-patient interaction effects: actual hazard vs sum of individual gene contributions")
    print("=" * 100)

    # Load data and compute out-of-fold hazard predictions
    print("\nLoading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

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

    print("Computing out-of-fold predictions...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue

        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
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

    # Load raw mutation data
    print("\nLoading per-gene mutation data...")
    import pandas as pd
    from gnn.config import ANALYSIS_CACHE

    mut_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    mutations = pd.read_csv(mut_path, low_memory=False,
                            usecols=["patientId", "gene.hugoGeneSymbol"])

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

    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mutations[mutations["gene.hugoGeneSymbol"].isin(channel_genes)].copy()

    # Build per-patient mutation set and channel set
    print("Building per-patient mutation profiles...")
    patient_mutations = defaultdict(set)
    patient_channels = defaultdict(set)
    gene_patient_sets = defaultdict(set)

    for _, row in mut.iterrows():
        pid = row["patientId"]
        if pid in pid_to_idx:
            idx = pid_to_idx[pid]
            gene = row["gene.hugoGeneSymbol"]
            patient_mutations[idx].add(gene)
            patient_channels[idx].add(V6_CHANNEL_MAP[gene])
            gene_patient_sets[gene].add(idx)

    hazards = all_hazards.numpy()
    times_np = data["times"].numpy()
    events_np = data["events"].numpy()
    valid = all_in_val.numpy()
    valid_idx = set(np.where(valid)[0])

    # Compute baseline (no channel mutation) mean hazard
    no_mut_idx = np.array([i for i in range(N) if valid[i] and i not in patient_mutations])
    baseline_hazard = hazards[no_mut_idx].mean() if len(no_mut_idx) > 0 else hazards[valid].mean()
    print(f"  Baseline hazard (no channel mutations): {baseline_hazard:.4f} ({len(no_mut_idx)} patients)")

    # Per-gene individual hazard shifts
    print("Computing individual gene hazard shifts...")
    gene_shift = {}
    gene_n = {}
    for gene in sorted(gene_patient_sets.keys()):
        mutated = np.array(sorted(gene_patient_sets[gene]))
        mutated_valid = mutated[valid[mutated]]
        if len(mutated_valid) < 20:
            continue
        gene_shift[gene] = hazards[mutated_valid].mean() - baseline_hazard
        gene_n[gene] = len(mutated_valid)

    print(f"  {len(gene_shift)} genes with individual shifts")

    # =========================================================================
    # PART 1: PER-PATIENT INTERACTION ANALYSIS
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"  PART 1: PER-PATIENT INTERACTION EFFECTS")
    print(f"{'='*100}")

    patient_interactions = []
    for i in range(N):
        if not valid[i]:
            continue
        genes = patient_mutations.get(i, set())
        if not genes:
            continue

        # Expected hazard = baseline + sum of individual gene shifts
        known_genes = [g for g in genes if g in gene_shift]
        if not known_genes:
            continue

        expected_shift = sum(gene_shift[g] for g in known_genes)
        actual_shift = hazards[i] - baseline_hazard
        interaction = actual_shift - expected_shift

        channels = patient_channels.get(i, set())
        channel_pattern = tuple(sorted(channels))
        n_channels = len(channels)

        patient_interactions.append({
            "patient_idx": i,
            "n_genes": len(known_genes),
            "n_channels": n_channels,
            "channel_pattern": channel_pattern,
            "expected_shift": float(expected_shift),
            "actual_shift": float(actual_shift),
            "interaction": float(interaction),
            "time": float(times_np[i]),
            "event": int(events_np[i]),
        })

    print(f"  {len(patient_interactions)} patients with interaction scores")

    # Group by number of channels severed
    print(f"\n  Interaction by number of channels severed:")
    by_n_ch = defaultdict(list)
    for p in patient_interactions:
        by_n_ch[p["n_channels"]].append(p)

    print(f"  {'N_ch':>4} {'N_pts':>7} {'Mean_Exp':>9} {'Mean_Act':>9} "
          f"{'Mean_Int':>9} {'%_Syn':>6} {'Med_Surv':>8} {'EvRate':>6}")
    print(f"  {'-'*4} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*8} {'-'*6}")

    for n_ch in sorted(by_n_ch.keys()):
        pts = by_n_ch[n_ch]
        exps = [p["expected_shift"] for p in pts]
        acts = [p["actual_shift"] for p in pts]
        ints = [p["interaction"] for p in pts]
        survs = [p["time"] for p in pts]
        evts = [p["event"] for p in pts]
        pct_syn = 100 * np.mean(np.array(ints) > 0)
        print(f"  {n_ch:>4} {len(pts):>7} {np.mean(exps):>+9.4f} {np.mean(acts):>+9.4f} "
              f"{np.mean(ints):>+9.4f} {pct_syn:>5.1f}% {np.median(survs):>8.1f} "
              f"{np.mean(evts):>6.3f}")

    # Group by channel pattern (which specific channels are severed)
    print(f"\n  Top channel patterns by synergy (min {MIN_GROUP} patients):")
    by_pattern = defaultdict(list)
    for p in patient_interactions:
        by_pattern[p["channel_pattern"]].append(p)

    pattern_summaries = []
    for pattern, pts in by_pattern.items():
        if len(pts) < MIN_GROUP:
            continue
        ints = [p["interaction"] for p in pts]
        exps = [p["expected_shift"] for p in pts]
        acts = [p["actual_shift"] for p in pts]
        survs = [p["time"] for p in pts]
        evts = [p["event"] for p in pts]

        # C-index for this group
        idx_arr = np.array([p["patient_idx"] for p in pts])
        ci = None
        if np.sum(events_np[idx_arr]) >= 5:
            ci = concordance_index(
                torch.tensor(hazards[idx_arr]),
                torch.tensor(times_np[idx_arr]),
                torch.tensor(events_np[idx_arr]),
            )

        pattern_summaries.append({
            "pattern": list(pattern),
            "n_channels": len(pattern),
            "n_patients": len(pts),
            "mean_interaction": float(np.mean(ints)),
            "mean_expected": float(np.mean(exps)),
            "mean_actual": float(np.mean(acts)),
            "pct_synergistic": float(np.mean(np.array(ints) > 0)),
            "median_survival": float(np.median(survs)),
            "event_rate": float(np.mean(evts)),
            "c_index": ci,
        })

    pattern_summaries.sort(key=lambda x: -x["mean_interaction"])

    print(f"\n  {'Pattern':<50} {'N':>5} {'Exp':>8} {'Act':>8} "
          f"{'Inter':>8} {'%Syn':>5} {'CI':>6} {'MedSurv':>7}")
    print(f"  {'-'*50} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*6} {'-'*7}")

    for s in pattern_summaries[:25]:
        pat_str = "+".join(s["pattern"])
        if len(pat_str) > 48:
            pat_str = pat_str[:45] + "..."
        ci_str = f"{s['c_index']:.3f}" if s["c_index"] else " ---"
        print(f"  {pat_str:<50} {s['n_patients']:>5} {s['mean_expected']:>+8.4f} "
              f"{s['mean_actual']:>+8.4f} {s['mean_interaction']:>+8.4f} "
              f"{100*s['pct_synergistic']:>4.0f}% {ci_str:>6} {s['median_survival']:>7.1f}")

    # Most redundant patterns
    print(f"\n  Most REDUNDANT channel patterns (actual < expected):")
    for s in pattern_summaries[-15:]:
        pat_str = "+".join(s["pattern"])
        if len(pat_str) > 48:
            pat_str = pat_str[:45] + "..."
        ci_str = f"{s['c_index']:.3f}" if s["c_index"] else " ---"
        print(f"  {pat_str:<50} {s['n_patients']:>5} {s['mean_expected']:>+8.4f} "
              f"{s['mean_actual']:>+8.4f} {s['mean_interaction']:>+8.4f} "
              f"{100*s['pct_synergistic']:>4.0f}% {ci_str:>6} {s['median_survival']:>7.1f}")

    # Find all gene pairs with sufficient co-occurrence
    print(f"\nFinding gene pairs with >= {MIN_CO_OCCUR} co-occurring patients...")
    genes_list = sorted(gene_shift.keys())

    pair_results = []
    n_tested = 0

    for i, gene_a in enumerate(genes_list):
        set_a = gene_patient_sets[gene_a] & valid_idx
        if len(set_a) < MIN_CO_OCCUR:
            continue

        for gene_b in genes_list[i+1:]:
            set_b = gene_patient_sets[gene_b] & valid_idx
            co_occur = set_a & set_b

            if len(co_occur) < MIN_CO_OCCUR:
                continue

            n_tested += 1
            co_idx = np.array(sorted(co_occur))

            # Actual hazard when both mutated
            actual_shift = hazards[co_idx].mean() - baseline_hazard

            # Expected if additive
            expected_shift = gene_shift[gene_a] + gene_shift[gene_b]

            # Synergy
            synergy = actual_shift - expected_shift

            # C-index among co-occurring patients
            co_events = data["events"].numpy()[co_idx]
            ci = None
            if co_events.sum() >= 5:
                ci = concordance_index(
                    torch.tensor(hazards[co_idx]),
                    torch.tensor(data["times"].numpy()[co_idx]),
                    torch.tensor(co_events),
                )

            ch_a = V6_CHANNEL_MAP.get(gene_a, "unknown")
            ch_b = V6_CHANNEL_MAP.get(gene_b, "unknown")
            tier_a = V6_TIER_MAP.get(ch_a, -1)
            tier_b = V6_TIER_MAP.get(ch_b, -1)
            cross_channel = ch_a != ch_b
            cross_tier = tier_a != tier_b

            pair_results.append({
                "gene_a": gene_a,
                "gene_b": gene_b,
                "channel_a": ch_a,
                "channel_b": ch_b,
                "tier_a": TIER_NAMES.get(tier_a, "?"),
                "tier_b": TIER_NAMES.get(tier_b, "?"),
                "cross_channel": cross_channel,
                "cross_tier": cross_tier,
                "n_co_occur": len(co_occur),
                "shift_a": float(gene_shift[gene_a]),
                "shift_b": float(gene_shift[gene_b]),
                "expected_shift": float(expected_shift),
                "actual_shift": float(actual_shift),
                "synergy": float(synergy),
                "ci_co_occur": ci,
            })

    print(f"  {n_tested} pairs tested, {len(pair_results)} with results")

    # Sort by synergy (most synergistic first)
    pair_results.sort(key=lambda x: -x["synergy"])

    # Top synergistic pairs (worse together than expected)
    print(f"\n{'='*110}")
    print(f"  TOP SYNERGISTIC PAIRS (worse together than sum of individual effects)")
    print(f"{'='*110}")
    print(f"  {'Gene_A':<10} {'Gene_B':<10} {'Ch_A':<15} {'Ch_B':<15} "
          f"{'N':>5} {'Shift_A':>8} {'Shift_B':>8} {'Expected':>8} {'Actual':>8} {'Synergy':>8} {'CI':>6}")
    print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*15} "
          f"{'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in pair_results[:30]:
        ci_str = f"{r['ci_co_occur']:.3f}" if r['ci_co_occur'] else "  ---"
        xch = "*" if r["cross_channel"] else " "
        print(f"  {r['gene_a']:<10} {r['gene_b']:<10} {r['channel_a']:<15} {r['channel_b']:<15} "
              f"{r['n_co_occur']:>5} {r['shift_a']:>+8.4f} {r['shift_b']:>+8.4f} "
              f"{r['expected_shift']:>+8.4f} {r['actual_shift']:>+8.4f} "
              f"{r['synergy']:>+8.4f} {ci_str:>6}{xch}")

    # Top redundant pairs (less bad together than expected)
    pair_results_rev = sorted(pair_results, key=lambda x: x["synergy"])
    print(f"\n{'='*110}")
    print(f"  TOP REDUNDANT PAIRS (less bad together than sum — overlapping damage)")
    print(f"{'='*110}")
    print(f"  {'Gene_A':<10} {'Gene_B':<10} {'Ch_A':<15} {'Ch_B':<15} "
          f"{'N':>5} {'Shift_A':>8} {'Shift_B':>8} {'Expected':>8} {'Actual':>8} {'Synergy':>8} {'CI':>6}")
    print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*15} "
          f"{'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in pair_results_rev[:30]:
        ci_str = f"{r['ci_co_occur']:.3f}" if r['ci_co_occur'] else "  ---"
        xch = "*" if r["cross_channel"] else " "
        print(f"  {r['gene_a']:<10} {r['gene_b']:<10} {r['channel_a']:<15} {r['channel_b']:<15} "
              f"{r['n_co_occur']:>5} {r['shift_a']:>+8.4f} {r['shift_b']:>+8.4f} "
              f"{r['expected_shift']:>+8.4f} {r['actual_shift']:>+8.4f} "
              f"{r['synergy']:>+8.4f} {ci_str:>6}{xch}")

    # Channel-pair synergy summary
    print(f"\n{'='*110}")
    print(f"  CHANNEL-PAIR SYNERGY SUMMARY")
    print(f"{'='*110}")

    ch_pair_stats = defaultdict(lambda: {"synergies": [], "n_pairs": 0, "n_patients": 0})
    for r in pair_results:
        key = tuple(sorted([r["channel_a"], r["channel_b"]]))
        ch_pair_stats[key]["synergies"].append(r["synergy"])
        ch_pair_stats[key]["n_pairs"] += 1
        ch_pair_stats[key]["n_patients"] += r["n_co_occur"]

    print(f"  {'Channel_A':<18} {'Channel_B':<18} {'Pairs':>5} {'Patients':>8} "
          f"{'Mean_Syn':>9} {'Max_Syn':>8} {'%_Pos':>6}")
    print(f"  {'-'*18} {'-'*18} {'-'*5} {'-'*8} {'-'*9} {'-'*8} {'-'*6}")

    ch_pair_sorted = sorted(ch_pair_stats.items(),
                            key=lambda x: -np.mean(x[1]["synergies"]))
    for (ch_a, ch_b), s in ch_pair_sorted:
        syns = np.array(s["synergies"])
        pct_pos = (syns > 0).mean() * 100
        print(f"  {ch_a:<18} {ch_b:<18} {s['n_pairs']:>5} {s['n_patients']:>8} "
              f"{np.mean(syns):>+9.4f} {np.max(syns):>+8.4f} {pct_pos:>5.1f}%")

    # Tier-pair synergy summary
    print(f"\n{'='*110}")
    print(f"  TIER-PAIR SYNERGY SUMMARY")
    print(f"{'='*110}")

    tier_pair_stats = defaultdict(lambda: {"synergies": [], "n_pairs": 0})
    for r in pair_results:
        key = tuple(sorted([r["tier_a"], r["tier_b"]]))
        tier_pair_stats[key]["synergies"].append(r["synergy"])
        tier_pair_stats[key]["n_pairs"] += 1

    print(f"  {'Tier_A':<18} {'Tier_B':<18} {'Pairs':>5} {'Mean_Syn':>9} {'%_Pos':>6}")
    print(f"  {'-'*18} {'-'*18} {'-'*5} {'-'*9} {'-'*6}")

    for (t_a, t_b), s in sorted(tier_pair_stats.items(),
                                  key=lambda x: -np.mean(x[1]["synergies"])):
        syns = np.array(s["synergies"])
        pct_pos = (syns > 0).mean() * 100
        print(f"  {t_a:<18} {t_b:<18} {s['n_pairs']:>5} {np.mean(syns):>+9.4f} {pct_pos:>5.1f}%")

    # Cross-channel vs within-channel
    print(f"\n{'='*110}")
    print(f"  CROSS-CHANNEL vs WITHIN-CHANNEL SYNERGY")
    print(f"{'='*110}")

    cross_syns = [r["synergy"] for r in pair_results if r["cross_channel"]]
    within_syns = [r["synergy"] for r in pair_results if not r["cross_channel"]]

    if cross_syns:
        print(f"  Cross-channel: {len(cross_syns):>5} pairs  "
              f"mean synergy={np.mean(cross_syns):+.4f}  "
              f"% positive={100*np.mean(np.array(cross_syns) > 0):.1f}%")
    if within_syns:
        print(f"  Within-channel: {len(within_syns):>5} pairs  "
              f"mean synergy={np.mean(within_syns):+.4f}  "
              f"% positive={100*np.mean(np.array(within_syns) > 0):.1f}%")

    # Cross-tier vs within-tier
    cross_tier_syns = [r["synergy"] for r in pair_results if r["cross_tier"]]
    within_tier_syns = [r["synergy"] for r in pair_results if not r["cross_tier"]]

    if cross_tier_syns:
        print(f"  Cross-tier:    {len(cross_tier_syns):>5} pairs  "
              f"mean synergy={np.mean(cross_tier_syns):+.4f}  "
              f"% positive={100*np.mean(np.array(cross_tier_syns) > 0):.1f}%")
    if within_tier_syns:
        print(f"  Within-tier:   {len(within_tier_syns):>5} pairs  "
              f"mean synergy={np.mean(within_tier_syns):+.4f}  "
              f"% positive={100*np.mean(np.array(within_tier_syns) > 0):.1f}%")

    # Overall stats
    all_syns = np.array([r["synergy"] for r in pair_results])
    print(f"\n  Overall: {len(all_syns)} pairs, "
          f"mean synergy={np.mean(all_syns):+.4f}, "
          f"median={np.median(all_syns):+.4f}, "
          f"% synergistic={100*np.mean(all_syns > 0):.1f}%")

    print(f"\n{'='*110}")

    # Save
    with open(os.path.join(SAVE_BASE, "pair_results.json"), "w") as f:
        json.dump(pair_results, f, indent=2, default=str)

    # Save channel-pair summary
    ch_summary = {}
    for (ch_a, ch_b), s in ch_pair_stats.items():
        key = f"{ch_a}___{ch_b}"
        syns = np.array(s["synergies"])
        ch_summary[key] = {
            "n_pairs": s["n_pairs"],
            "n_patients": s["n_patients"],
            "mean_synergy": float(np.mean(syns)),
            "max_synergy": float(np.max(syns)),
            "pct_synergistic": float((syns > 0).mean()),
        }
    with open(os.path.join(SAVE_BASE, "channel_pair_summary.json"), "w") as f:
        json.dump(ch_summary, f, indent=2)

    print(f"  Saved to {SAVE_BASE}")
    print(f"  {len(pair_results)} pairs analyzed")


if __name__ == "__main__":
    main()
