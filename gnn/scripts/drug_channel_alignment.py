#!/usr/bin/env python3
"""
Drug-channel alignment: does the model's attention predict known pharmacology?

Maps FDA-approved targeted therapies to coupling channels, then compares
per-cancer-type attention weights against approved indications.

Usage:
    python3 -u -m gnn.scripts.drug_channel_alignment
"""

import json
import os
import numpy as np

sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import GNN_RESULTS, CHANNEL_NAMES

# ── FDA-approved targeted therapies mapped to coupling channels ──────────
# Each entry: drug name, channel it targets, approved cancer types
DRUG_CHANNEL_MAP = [
    # PI3K_Growth channel
    ("Alpelisib (PI3Kα)", "PI3K_Growth", ["Breast Cancer"]),
    ("Everolimus (mTOR)", "PI3K_Growth", ["Breast Cancer", "Renal Cell Carcinoma"]),
    ("Vemurafenib (BRAF)", "PI3K_Growth", ["Melanoma"]),
    ("Dabrafenib (BRAF)", "PI3K_Growth", ["Melanoma", "Thyroid Cancer", "Non-Small Cell Lung Cancer"]),
    ("Trametinib (MEK)", "PI3K_Growth", ["Melanoma", "Thyroid Cancer", "Non-Small Cell Lung Cancer"]),
    ("Sotorasib (KRAS G12C)", "PI3K_Growth", ["Non-Small Cell Lung Cancer"]),
    ("Adagrasib (KRAS G12C)", "PI3K_Growth", ["Non-Small Cell Lung Cancer"]),

    # CellCycle channel
    ("Palbociclib (CDK4/6)", "CellCycle", ["Breast Cancer"]),
    ("Ribociclib (CDK4/6)", "CellCycle", ["Breast Cancer"]),
    ("Abemaciclib (CDK4/6)", "CellCycle", ["Breast Cancer"]),

    # DDR channel
    ("Olaparib (PARP)", "DDR", ["Breast Cancer", "Ovarian Cancer", "Prostate Cancer", "Pancreatic Cancer"]),
    ("Rucaparib (PARP)", "DDR", ["Ovarian Cancer", "Prostate Cancer"]),
    ("Talazoparib (PARP)", "DDR", ["Breast Cancer"]),
    ("Niraparib (PARP)", "DDR", ["Ovarian Cancer"]),

    # Endocrine channel
    ("Tamoxifen (ER)", "Endocrine", ["Breast Cancer"]),
    ("Letrozole (aromatase)", "Endocrine", ["Breast Cancer"]),
    ("Anastrozole (aromatase)", "Endocrine", ["Breast Cancer"]),
    ("Enzalutamide (AR)", "Endocrine", ["Prostate Cancer"]),
    ("Abiraterone (CYP17)", "Endocrine", ["Prostate Cancer"]),
    ("Apalutamide (AR)", "Endocrine", ["Prostate Cancer"]),

    # Immune channel
    ("Pembrolizumab (PD-1)", "Immune", ["Non-Small Cell Lung Cancer", "Melanoma", "Bladder Cancer",
                                         "Renal Cell Carcinoma", "Head and Neck Cancer"]),
    ("Nivolumab (PD-1)", "Immune", ["Non-Small Cell Lung Cancer", "Melanoma", "Renal Cell Carcinoma",
                                     "Bladder Cancer"]),
    ("Atezolizumab (PD-L1)", "Immune", ["Non-Small Cell Lung Cancer", "Bladder Cancer",
                                          "Breast Cancer"]),
    ("Ipilimumab (CTLA-4)", "Immune", ["Melanoma", "Renal Cell Carcinoma"]),

    # TissueArch channel
    ("Bevacizumab (VEGF)", "TissueArch", ["Colorectal Cancer", "Non-Small Cell Lung Cancer",
                                            "Renal Cell Carcinoma", "Ovarian Cancer", "Glioma"]),
    ("Ramucirumab (VEGFR2)", "TissueArch", ["Gastric Cancer", "Non-Small Cell Lung Cancer",
                                              "Colorectal Cancer"]),
]

# Cancer types we have attention data for
TARGET_TYPES = [
    "Non-Small Cell Lung Cancer",
    "Breast Cancer",
    "Prostate Cancer",
    "Thyroid Cancer",
    "Glioma",
]


def main():
    # Load attention data
    attn_path = os.path.join(GNN_RESULTS, "channelnet_v2", "attention_per_type.json")
    with open(attn_path) as f:
        type_summaries = json.load(f)

    print("=" * 78)
    print("  DRUG-CHANNEL ALIGNMENT ANALYSIS")
    print("  Does the model's attention predict known pharmacology?")
    print("=" * 78)

    # ── Per cancer type: which channels does the model emphasize vs which drugs are approved? ──
    for ct_name in TARGET_TYPES:
        if ct_name not in type_summaries:
            continue

        importance = type_summaries[ct_name]["channel_importance"]

        # Rank channels by attention
        ranked = sorted(importance.items(), key=lambda x: -x[1])

        # Find approved drugs for this cancer type
        approved_drugs = {}
        for drug, channel, indications in DRUG_CHANNEL_MAP:
            if ct_name in indications:
                if channel not in approved_drugs:
                    approved_drugs[channel] = []
                approved_drugs[channel].append(drug)

        print(f"\n  {'─' * 72}")
        print(f"  {ct_name}")
        print(f"  {'─' * 72}")
        print(f"  {'Channel':<16} {'Attention':>10} {'Rank':>6}  FDA-Approved Drugs Targeting This Channel")
        print(f"  {'─'*16} {'─'*10} {'─'*6}  {'─'*42}")

        for rank, (ch_name, attn_val) in enumerate(ranked, 1):
            drugs = approved_drugs.get(ch_name, [])
            drug_str = ", ".join(drugs) if drugs else "—"
            match = "✓" if drugs else " "
            # Highlight if high attention AND has approved drugs
            marker = ""
            if rank <= 3 and drugs:
                marker = " ← MODEL AGREES"
            elif rank <= 2 and not drugs:
                marker = " ← no drug yet"
            print(f"  {ch_name:<16} {attn_val:>10.4f} {rank:>5}   {match} {drug_str}{marker}")

        # Score: what fraction of approved-drug channels are in top 3 attention?
        approved_channels = set(approved_drugs.keys())
        top3 = set(ch for ch, _ in ranked[:3])
        if approved_channels:
            overlap = approved_channels & top3
            score = len(overlap) / len(approved_channels)
            print(f"\n  Alignment: {len(overlap)}/{len(approved_channels)} "
                  f"drug-targeted channels in model's top 3 "
                  f"({score:.0%})")

    # ── Summary table: attention vs drug approval ──
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY: Channel Attention vs Drug Approval Matrix")
    print(f"{'=' * 78}")
    print(f"\n  Attention weight for channels WITH vs WITHOUT approved drugs:\n")

    all_with_drug = []
    all_without_drug = []

    print(f"  {'Cancer Type':<28} {'With Drug':>10} {'No Drug':>10} {'Delta':>8}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*8}")

    for ct_name in TARGET_TYPES:
        if ct_name not in type_summaries:
            continue

        importance = type_summaries[ct_name]["channel_importance"]

        # Which channels have approved drugs for this type?
        drug_channels = set()
        for drug, channel, indications in DRUG_CHANNEL_MAP:
            if ct_name in indications:
                drug_channels.add(channel)

        with_drug = [v for ch, v in importance.items() if ch in drug_channels]
        without_drug = [v for ch, v in importance.items() if ch not in drug_channels]

        if with_drug and without_drug:
            mean_with = np.mean(with_drug)
            mean_without = np.mean(without_drug)
            delta = mean_with - mean_without
            all_with_drug.extend(with_drug)
            all_without_drug.extend(without_drug)
            print(f"  {ct_name:<28} {mean_with:>10.4f} {mean_without:>10.4f} {delta:>+8.4f}")

    if all_with_drug and all_without_drug:
        overall_with = np.mean(all_with_drug)
        overall_without = np.mean(all_without_drug)
        overall_delta = overall_with - overall_without
        print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*8}")
        print(f"  {'OVERALL':<28} {overall_with:>10.4f} {overall_without:>10.4f} {overall_delta:>+8.4f}")

        # Effect size
        pooled_std = np.sqrt((np.var(all_with_drug) + np.var(all_without_drug)) / 2)
        if pooled_std > 0:
            cohens_d = overall_delta / pooled_std
            print(f"\n  Cohen's d = {cohens_d:.2f} (effect size of attention difference)")

    # ── The pharma prediction: channels with high attention but no approved drug ──
    print(f"\n{'=' * 78}")
    print(f"  PREDICTION: High-Attention Channels Without Approved Drugs")
    print(f"  (potential therapeutic targets the model identifies)")
    print(f"{'=' * 78}")

    for ct_name in TARGET_TYPES:
        if ct_name not in type_summaries:
            continue

        importance = type_summaries[ct_name]["channel_importance"]
        ranked = sorted(importance.items(), key=lambda x: -x[1])

        drug_channels = set()
        for drug, channel, indications in DRUG_CHANNEL_MAP:
            if ct_name in indications:
                drug_channels.add(channel)

        # Channels with above-average attention but no drug
        mean_attn = np.mean(list(importance.values()))
        gaps = [(ch, v) for ch, v in ranked if v > mean_attn and ch not in drug_channels]

        if gaps:
            for ch, v in gaps:
                excess = (v - mean_attn) / mean_attn * 100
                print(f"  {ct_name:<28} → {ch:<16} (attn={v:.4f}, "
                      f"+{excess:.0f}% above mean)")

    print(f"\n{'=' * 78}")

    # Save
    out_path = os.path.join(GNN_RESULTS, "channelnet_v2", "drug_channel_alignment.json")
    results = {
        "analysis": "drug_channel_alignment",
        "target_types": TARGET_TYPES,
        "drug_map_size": len(DRUG_CHANNEL_MAP),
    }
    for ct_name in TARGET_TYPES:
        if ct_name in type_summaries:
            results[ct_name] = type_summaries[ct_name]["channel_importance"]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
