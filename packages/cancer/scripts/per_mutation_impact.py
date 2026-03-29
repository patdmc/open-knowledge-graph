#!/usr/bin/env python3
"""
Per-mutation predictive impact analysis.

For each gene in the channel framework, measure how much the model's
hazard prediction changes when that gene is mutated vs not.

This decomposes the aggregate C-index into per-gene contributions,
showing that predictive power is concentrated in specific channel-relevant
mutations rather than uniformly distributed.

Output:
  - Per-gene hazard shift (mean hazard for patients with mutation - without)
  - Per-gene C-index contribution (C-index among patients with vs without)
  - Grouped by channel and tier

Usage:
    python3 -u -m gnn.scripts.per_mutation_impact
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS
from gnn.data.channel_dataset_v6c import build_channel_features_v6c, ChannelDatasetV6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX,
    N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6, V6_TIER_MAP,
    V6_GENE_FUNCTION,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

SAVE_BASE = os.path.join(GNN_RESULTS, "per_mutation_impact")

TIER_NAMES = {0: "Cell Intrinsic", 1: "Tissue Level", 2: "Organism Level", 3: "Meta Regulatory"}


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 100)
    print("  PER-MUTATION PREDICTIVE IMPACT ANALYSIS")
    print("  How much does each gene's mutation change the model's survival prediction?")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    n_ct = len(data["cancer_type_vocab"])

    # Load all 5 fold models and get out-of-fold predictions
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
            continue

        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # Batch inference on val set
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

    # Now load the raw mutation data to get per-gene mutation status
    print("\nLoading per-gene mutation data...")
    import pandas as pd
    from gnn.config import ANALYSIS_CACHE

    mut_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    mutations = pd.read_csv(mut_path, low_memory=False,
                            usecols=["patientId", "gene.hugoGeneSymbol"])

    # Reconstruct patient ordering (same as build_channel_features_v6c)
    # The data dict has patients in order of clin["patientId"].unique()
    # We need to get that same ordering
    patient_ids = data.get("patient_ids")
    if patient_ids is None:
        # Rebuild from clinical data
        clin_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv")
        clin = pd.read_csv(clin_path, low_memory=False)
        # Same filtering as build_channel_features_v6c
        clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
        clin["time"] = clin["OS_MONTHS"].astype(float)
        clin["event"] = clin["OS_STATUS"].apply(
            lambda x: 1 if "DECEASED" in str(x).upper() else 0
        )
        clin = clin[clin["time"] > 0]
        patient_ids = clin["patientId"].unique()

    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    # Get channel genes
    channel_genes = set(V6_CHANNEL_MAP.keys())

    # Filter mutations to channel genes
    mut = mutations[mutations["gene.hugoGeneSymbol"].isin(channel_genes)].copy()

    # Build per-patient-per-gene mutation indicator
    print("Building per-gene mutation matrix...")
    gene_patient_sets = defaultdict(set)
    for _, row in mut.iterrows():
        pid = row["patientId"]
        if pid in pid_to_idx:
            gene_patient_sets[row["gene.hugoGeneSymbol"]].add(pid_to_idx[pid])

    # Compute per-gene impact
    print(f"\nAnalyzing {len(gene_patient_sets)} genes...\n")

    hazards = all_hazards.numpy()
    times_np = data["times"].numpy()
    events_np = data["events"].numpy()
    valid = all_in_val.numpy()

    gene_results = []

    for gene in sorted(gene_patient_sets.keys()):
        mutated_idx = np.array(sorted(gene_patient_sets[gene]))
        # Filter to valid (in validation set) patients
        mutated_valid = mutated_idx[valid[mutated_idx]]
        n_mutated = len(mutated_valid)

        if n_mutated < 20:
            continue

        # All valid patients without this mutation
        all_valid_idx = np.where(valid)[0]
        unmutated_valid = np.setdiff1d(all_valid_idx, mutated_valid)

        # Hazard shift
        mean_h_mut = hazards[mutated_valid].mean()
        mean_h_unmut = hazards[unmutated_valid].mean()
        hazard_shift = mean_h_mut - mean_h_unmut

        # C-index among mutated patients only
        if events_np[mutated_valid].sum() >= 5:
            ci_mut = concordance_index(
                torch.tensor(hazards[mutated_valid]),
                torch.tensor(times_np[mutated_valid]),
                torch.tensor(events_np[mutated_valid]),
            )
        else:
            ci_mut = None

        # Median survival difference
        med_surv_mut = np.median(times_np[mutated_valid])
        med_surv_unmut = np.median(times_np[unmutated_valid])

        # Event rate
        event_rate_mut = events_np[mutated_valid].mean()
        event_rate_unmut = events_np[unmutated_valid].mean()

        channel = V6_CHANNEL_MAP.get(gene, "unknown")
        tier_idx = V6_TIER_MAP.get(channel, -1)
        tier_name = TIER_NAMES.get(tier_idx, "unknown")
        func = V6_GENE_FUNCTION.get(gene, "---")

        gene_results.append({
            "gene": gene,
            "channel": channel,
            "tier": tier_name,
            "function": func,
            "n_mutated": n_mutated,
            "hazard_shift": float(hazard_shift),
            "ci_among_mutated": ci_mut,
            "median_surv_mutated": float(med_surv_mut),
            "median_surv_unmutated": float(med_surv_unmut),
            "event_rate_mutated": float(event_rate_mut),
            "event_rate_unmutated": float(event_rate_unmut),
        })

    # Sort by absolute hazard shift
    gene_results.sort(key=lambda x: -abs(x["hazard_shift"]))

    # Print top genes by hazard shift (most predictive)
    print(f"{'='*110}")
    print(f"  TOP GENES BY HAZARD SHIFT (model prediction change when gene is mutated)")
    print(f"{'='*110}")
    print(f"  {'Gene':<12} {'Channel':<20} {'Tier':<16} {'Func':>4} {'N_mut':>6} "
          f"{'H_shift':>8} {'CI_mut':>7} {'Med_mut':>8} {'Med_wt':>8} {'EvR_mut':>7} {'EvR_wt':>7}")
    print(f"  {'-'*12} {'-'*20} {'-'*16} {'-'*4} {'-'*6} "
          f"{'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

    for r in gene_results[:50]:
        ci_str = f"{r['ci_among_mutated']:.3f}" if r['ci_among_mutated'] else "  ---"
        print(f"  {r['gene']:<12} {r['channel']:<20} {r['tier']:<16} {r['function']:>4} "
              f"{r['n_mutated']:>6} {r['hazard_shift']:>+8.4f} {ci_str:>7} "
              f"{r['median_surv_mutated']:>8.1f} {r['median_surv_unmutated']:>8.1f} "
              f"{r['event_rate_mutated']:>7.3f} {r['event_rate_unmutated']:>7.3f}")

    # Channel-level summary
    print(f"\n{'='*110}")
    print(f"  PER-CHANNEL SUMMARY")
    print(f"{'='*110}")

    channel_stats = defaultdict(lambda: {"shifts": [], "genes": 0, "n_patients": 0})
    for r in gene_results:
        ch = r["channel"]
        channel_stats[ch]["shifts"].append(r["hazard_shift"])
        channel_stats[ch]["genes"] += 1
        channel_stats[ch]["n_patients"] += r["n_mutated"]

    print(f"  {'Channel':<20} {'Tier':<16} {'Genes':>5} {'Patients':>8} "
          f"{'Mean|shift|':>11} {'Max|shift|':>10} {'Direction':>9}")
    print(f"  {'-'*20} {'-'*16} {'-'*5} {'-'*8} {'-'*11} {'-'*10} {'-'*9}")

    for ch_name in V6_CHANNEL_NAMES:
        if ch_name not in channel_stats:
            continue
        s = channel_stats[ch_name]
        shifts = np.array(s["shifts"])
        tier_idx = V6_TIER_MAP.get(ch_name, -1)
        tier = TIER_NAMES.get(tier_idx, "?")
        mean_abs = np.mean(np.abs(shifts))
        max_abs = np.max(np.abs(shifts))
        direction = "worse" if np.mean(shifts) > 0 else "better"
        print(f"  {ch_name:<20} {tier:<16} {s['genes']:>5} {s['n_patients']:>8} "
              f"{mean_abs:>11.4f} {max_abs:>10.4f} {direction:>9}")

    # Tier-level summary
    print(f"\n{'='*110}")
    print(f"  PER-TIER SUMMARY")
    print(f"{'='*110}")

    tier_stats = defaultdict(lambda: {"shifts": [], "genes": 0})
    for r in gene_results:
        tier_stats[r["tier"]]["shifts"].append(r["hazard_shift"])
        tier_stats[r["tier"]]["genes"] += 1

    for tier_name in ["Cell Intrinsic", "Tissue Level", "Organism Level", "Meta Regulatory"]:
        if tier_name not in tier_stats:
            continue
        s = tier_stats[tier_name]
        shifts = np.array(s["shifts"])
        print(f"  {tier_name:<20} {s['genes']:>3} genes  "
              f"mean|shift|={np.mean(np.abs(shifts)):.4f}  "
              f"max|shift|={np.max(np.abs(shifts)):.4f}")

    print(f"\n{'='*110}")

    # Save
    with open(os.path.join(SAVE_BASE, "gene_results.json"), "w") as f:
        json.dump(gene_results, f, indent=2, default=str)

    # Also save a clean TSV for plotting
    with open(os.path.join(SAVE_BASE, "gene_impact.tsv"), "w") as f:
        f.write("gene\tchannel\ttier\tfunction\tn_mutated\thazard_shift\t"
                "ci_among_mutated\tmedian_surv_mut\tmedian_surv_wt\t"
                "event_rate_mut\tevent_rate_wt\n")
        for r in gene_results:
            ci = r['ci_among_mutated'] if r['ci_among_mutated'] else ""
            f.write(f"{r['gene']}\t{r['channel']}\t{r['tier']}\t{r['function']}\t"
                    f"{r['n_mutated']}\t{r['hazard_shift']:.4f}\t{ci}\t"
                    f"{r['median_surv_mutated']:.1f}\t{r['median_surv_unmutated']:.1f}\t"
                    f"{r['event_rate_mutated']:.3f}\t{r['event_rate_unmutated']:.3f}\n")

    print(f"  Saved to {SAVE_BASE}")
    print(f"  {len(gene_results)} genes analyzed")


if __name__ == "__main__":
    main()
