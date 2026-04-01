#!/usr/bin/env python3
"""Build YAML cancer-type nodes for the knowledge graph.

Reads:
  - analysis/escalation_entropy.csv

Writes:
  - knowledge-graph/nodes/biology/cancer_types/CT-{abbrev}.yaml (one per cancer type)
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CT_DIR = ROOT / "knowledge-graph" / "nodes" / "biology" / "cancer_types"

# Abbreviation map for cancer types → short IDs
ABBREV = {
    "Thyroid Cancer": "THYR",
    "Pancreatic Cancer": "PANC",
    "Germ Cell Tumor": "GCT",
    "Appendiceal Cancer": "APPC",
    "Esophagogastric Cancer": "ESGC",
    "Bladder Cancer": "BLCA",
    "Non-Small Cell Lung Cancer": "NSCLC",
    "Hepatobiliary Cancer": "HBC",
    "Renal Cell Carcinoma": "RCC",
    "Colorectal Cancer": "CRC",
    "Head and Neck Cancer": "HNSC",
    "Breast Cancer": "BRCA",
    "Cancer of Unknown Primary": "CUP",
    "Ovarian Cancer": "OV",
    "Gastrointestinal Stromal Tumor": "GIST",
    "Endometrial Cancer": "UCEC",
    "Soft Tissue Sarcoma": "STS",
    "Prostate Cancer": "PRAD",
    "Cervical Cancer": "CESC",
    "Melanoma": "MEL",
    "Glioma": "GBM",
    "Non-Melanoma Skin Cancer": "NMSC",
    "Bone Cancer": "BONE",
    "Mesothelioma": "MESO",
    "Small Cell Lung Cancer": "SCLC",
    "Peripheral Nervous System": "PNS",
    "CNS Cancer": "CNS",
    "Mature B-Cell Neoplasms": "BCELL",
    "Uterine Sarcoma": "USARC",
    "Adrenocortical Carcinoma": "ACC",
    "Wilms Tumor": "WILMS",
    "Mastocytosis": "MAST",
    "Embryonal Tumor": "EMBT",
    "Pheochromocytoma": "PHEO",
    "Sex Cord Stromal Tumor": "SCST",
    "Nerve Sheath Tumor": "NST",
    "Myeloproliferative Neoplasms": "MPN",
    "Sellar Tumor": "SELL",
}

CHANNEL_MAP = {
    "PI3K_Growth": "CHAN-PI3K_Growth",
    "CellCycle": "CHAN-CellCycle",
    "DDR": "CHAN-DDR",
    "Immune": "CHAN-Immune",
    "Endocrine": "CHAN-Endocrine",
    "TissueArch": "CHAN-TissueArchitecture",
    "TissueArchitecture": "CHAN-TissueArchitecture",
}


def esc(s):
    if any(c in str(s) for c in ":{}\\n[]&*#?|-<>=!%@`"):
        return f'"{s}"'
    return str(s)


def fmt(v):
    v = float(v)
    if abs(v) < 0.001:
        return f"{v:.2e}"
    return f"{v:.4f}"


def main():
    CT_DIR.mkdir(parents=True, exist_ok=True)

    path = ROOT / "analysis" / "escalation_entropy.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))

    count = 0
    for row in rows:
        ct_name = row["cancer_type"]
        abbrev = ABBREV.get(ct_name, ct_name.replace(" ", "")[:6].upper())

        n_patients = int(row["n_patients"])
        n_mutated = int(row["n_mutated"])
        channel_entropy = float(row["channel_entropy"])
        hub_fraction = float(row["hub_fraction"])
        median_os = float(row["median_os_months"])
        event_rate = float(row["event_rate"])
        survival_36m = float(row["survival_36m"])
        top1_gene = row["top1_gene"]
        top1_share = float(row["top1_gene_share"])
        mean_channels = float(row["mean_channels_per_patient"])
        n_harmful = int(row["n_harmful_atlas"])
        n_protective = int(row["n_protective_atlas"])

        # Parse dominant channels
        dom_raw = row["dominant_channels"].strip().strip('"')
        dom_channels = [CHANNEL_MAP.get(c.strip(), f"CHAN-{c.strip()}") for c in dom_raw.split(",")]

        L = []
        L.append(f"id: CT-{abbrev}")
        L.append("type: knowledge")
        L.append("domain: cancer_biology")
        L.append(f"name: {esc(ct_name)}")
        L.append("")
        L.append(f"n_patients: {n_patients}")
        L.append(f"n_mutated: {n_mutated}")
        L.append(f"mutation_rate: {fmt(float(row['mutation_rate']))}")
        L.append(f"channel_entropy: {fmt(channel_entropy)}")
        L.append(f"hub_fraction: {fmt(hub_fraction)}")
        L.append(f"median_os_months: {fmt(median_os)}")
        L.append(f"event_rate: {fmt(event_rate)}")
        L.append(f"survival_36m: {fmt(survival_36m)}")
        L.append(f"mean_channels_per_patient: {fmt(mean_channels)}")
        L.append(f"top_gene: {top1_gene}")
        L.append(f"top_gene_frequency: {fmt(top1_share)}")
        L.append(f"n_harmful_mutations: {n_harmful}")
        L.append(f"n_protective_mutations: {n_protective}")
        L.append(f"dominant_channels: [{', '.join(dom_channels)}]")
        L.append("")
        L.append("provenance:")
        L.append("  attribution:")
        L.append('    author: "Patrick D. McCarthy"')
        L.append('    source: "Genome as Projection (Paper 5)"')
        L.append('    date: "2026"')
        L.append('    doi: "10.5281/zenodo.18923066"')
        L.append("  evidence:")
        L.append("    type: empirical")
        L.append(f'    description: "Statistics computed from MSK-IMPACT cohort (n={n_patients})"')
        L.append("")
        L.append("edges:")
        for chan_id in dom_channels:
            L.append(f"  - to: {chan_id}")
            L.append("    relation: primarily_disrupts")
            L.append("    provenance:")
            L.append("      attribution:")
            L.append('        author: "Patrick D. McCarthy"')
            L.append('        source: "Paper 5 — Genome as Projection"')
            L.append('        date: "2026"')
            L.append("      evidence:")
            L.append("        type: empirical")
            L.append(f'        description: "Dominant disruption channel in {ct_name}"')
        L.append(f"  - to: GENE-{top1_gene}")
        L.append("    relation: frequently_mutated_in")
        L.append("    provenance:")
        L.append("      attribution:")
        L.append('        source: "MSK-IMPACT mutation frequencies"')
        L.append('        date: "2026"')
        L.append("      evidence:")
        L.append("        type: empirical")
        L.append(f'        description: "{top1_gene} mutated in {fmt(top1_share)} of {ct_name} patients"')
        L.append("")

        out = CT_DIR / f"CT-{abbrev}.yaml"
        out.write_text("\n".join(L) + "\n")
        count += 1
        print(f"  CT-{abbrev}: {ct_name} (n={n_patients})")

    print(f"\nDone. {count} cancer type files.")


if __name__ == "__main__":
    main()
