"""Test 4: disease spectrum by gene class.

The encapsulation inversion hypothesis predicts:
  - First-order (implementation) genes: disease failures are developmentally
    lethal, produce pediatric Mendelian syndromes, or are absorbed by error
    handling. They do NOT typically produce adult-onset cancer, because
    their mutations are either fatal early or compensable.
  - Higher-order (M-layer) genes: disease failures are silent-until-
    catastrophic adult-onset conditions. They DO produce cancer, because
    they break the error handling layer itself and the failure accumulates
    over years before manifesting.

So: channel genes (higher-order) should be enriched in CANCER associations
vs developmental/pediatric; implementation controls should show the opposite
skew.

Data source: Jensen Lab DISEASES (text-mining + curated gene-disease
associations, knowledge-tier subset), plus NCBI mim2gene_medgen for OMIM
phenotype associations as a cross-check.

Usage:
  python test_disease_spectrum.py
"""

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# Keyword classification — crude but reproducible
CANCER_KEYWORDS = [
    "cancer", "carcinoma", "sarcoma", "leukemia", "lymphoma", "tumor",
    "neoplasm", "myeloma", "melanoma", "glioma", "blastoma", "adenoma",
    "malignancy", "malignant",
]
DEVELOPMENTAL_KEYWORDS = [
    "syndrome", "dysplasia", "malformation", "congenital", "hereditary",
    "inherited", "dystrophy", "atrophy", "agenesis", "microcephaly",
    "macrocephaly", "dwarfism", "intellectual disability",
]
NEURODEGEN_KEYWORDS = [
    "alzheimer", "parkinson", "amyotrophic", "neurodegen", "huntington",
    "frontotemporal", "lewy body",
]
METABOLIC_KEYWORDS = [
    "metabolic disease", "metabolism disease", "diabetes", "hyperlipidemia",
    "glycogen storage", "mitochondrial",
]

IMPLEMENTATION_CONTROLS = [
    # Chosen as non-catalytic A-role or minimally targetable primitives
    "POLA1", "POLB", "POLD1", "POLE", "POLG", "POLH", "POLK",
    "LIG1", "LIG3", "LIG4", "FEN1",
    "RPS3", "RPS6", "RPL7", "RPL11", "RPL22", "RPL23",
    "ACTB", "GAPDH", "EEF1A1", "EEF1A2", "TUBB", "TUBA1A",
    "LARS", "VARS", "MARS1",
    "HK1", "HK2", "PKM", "LDHA",
    "RAD51", "DMC1",
    "TOP1", "TOP2A", "TOP2B", "TOP3A", "TOP3B",
    # Structural additions for the genuinely-untreatable cell
    "COL1A1", "COL3A1", "COL4A1", "COL6A1",
    "KRT5", "KRT14", "KRT18",
    "LMNA", "LMNB1",
    "FN1", "LAMA1", "LAMB1",
]


def classify_disease(name: str) -> str:
    n = name.lower()
    if any(k in n for k in CANCER_KEYWORDS):
        return "cancer"
    if any(k in n for k in NEURODEGEN_KEYWORDS):
        return "neurodegen"
    if any(k in n for k in METABOLIC_KEYWORDS):
        return "metabolic"
    if any(k in n for k in DEVELOPMENTAL_KEYWORDS):
        return "developmental"
    return "other"


def load_diseases(path: Path) -> dict[str, list[tuple[str, str, float]]]:
    """Return {gene_symbol: [(doid, disease_name, score), ...]}."""
    print(f"[disease] loading {path}", file=sys.stderr)
    out: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            _prot, gene, doid, name, _source, _type, score_str = parts[:7]
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            out[gene].append((doid, name, score))
    print(f"[disease] {len(out)} genes with ≥1 disease", file=sys.stderr)
    return out


def gene_class_summary(gene_diseases: list[tuple[str, str, float]]) -> dict:
    """Per-gene classification summary."""
    categories = {"cancer": 0, "developmental": 0, "neurodegen": 0,
                  "metabolic": 0, "other": 0}
    max_score_by_cat = {k: 0.0 for k in categories}
    for doid, name, score in gene_diseases:
        cat = classify_disease(name)
        categories[cat] += 1
        if score > max_score_by_cat[cat]:
            max_score_by_cat[cat] = score
    return {
        "n_diseases": len(gene_diseases),
        "n_cancer": categories["cancer"],
        "n_developmental": categories["developmental"],
        "n_neurodegen": categories["neurodegen"],
        "n_metabolic": categories["metabolic"],
        "n_other": categories["other"],
        "max_cancer_score": max_score_by_cat["cancer"],
        "max_dev_score": max_score_by_cat["developmental"],
        "has_cancer": categories["cancer"] > 0,
        "has_developmental": categories["developmental"] > 0,
        "cancer_dominated": categories["cancer"] > categories["developmental"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel-map", type=Path, default=Path("../../data/channel_gene_map.csv"))
    ap.add_argument("--diseases", type=Path, default=Path("data/disease/diseases_gene_full.tsv"))
    ap.add_argument("--background-n", type=int, default=500)
    ap.add_argument("--string-aliases", type=Path,
                    default=Path("data/string/9606.protein.aliases.v12.0.txt.gz"))
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p

    channel_df = pd.read_csv(rel(args.channel_map))
    channel_set = set(channel_df.gene.unique())
    impl_set = set(IMPLEMENTATION_CONTROLS)

    gene_diseases = load_diseases(rel(args.diseases))

    # Sample background from STRING HGNC symbols
    all_syms: list[str] = []
    with gzip.open(rel(args.string_aliases), "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and "HGNC_symbol" in parts[2]:
                sym = parts[1]
                if sym not in channel_set and sym not in impl_set:
                    all_syms.append(sym)
    all_syms = list(set(all_syms))
    rng = np.random.default_rng(42)
    bg = list(rng.choice(all_syms, size=min(args.background_n, len(all_syms)), replace=False))

    # Classify each test gene
    def tag(gene: str) -> str:
        if gene in channel_set:
            return "channel"
        if gene in impl_set:
            return "implementation"
        return "background"

    test_genes = sorted(channel_set) + sorted(impl_set) + sorted(bg)
    rows = []
    for g in test_genes:
        summary = gene_class_summary(gene_diseases.get(g, []))
        summary["gene"] = g
        summary["class"] = tag(g)
        rows.append(summary)
    df = pd.DataFrame(rows)

    # Drop genes that have no disease annotations at all
    df_with_any = df[df.n_diseases > 0].copy()
    print(f"\n[disease] genes with ≥1 disease annotation: {len(df_with_any)}")
    print(f"[disease]   channel:        {(df_with_any['class']=='channel').sum()} of {(df['class']=='channel').sum()}")
    print(f"[disease]   implementation: {(df_with_any['class']=='implementation').sum()} of {(df['class']=='implementation').sum()}")
    print(f"[disease]   background:     {(df_with_any['class']=='background').sum()} of {(df['class']=='background').sum()}")
    print()

    # Summary by class: fraction of genes with cancer / developmental / etc.
    summary = df_with_any.groupby("class").agg(
        n=("gene", "count"),
        frac_has_cancer=("has_cancer", "mean"),
        frac_has_developmental=("has_developmental", "mean"),
        frac_cancer_dominated=("cancer_dominated", "mean"),
        mean_n_cancer=("n_cancer", "mean"),
        mean_n_developmental=("n_developmental", "mean"),
    )
    print("=== Fraction of genes with each disease category ===")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print()

    # Fisher's exact: channel vs implementation
    print("=== Fisher's exact: channel vs implementation (alternative=greater unless noted) ===")
    ch = df_with_any[df_with_any["class"] == "channel"]
    impl = df_with_any[df_with_any["class"] == "implementation"]
    bg_df = df_with_any[df_with_any["class"] == "background"]

    for metric, direction in [
        ("has_cancer", "greater"),
        ("has_developmental", "less"),
        ("cancer_dominated", "greater"),
    ]:
        a = int(ch[metric].sum()); b = len(ch) - a
        c = int(impl[metric].sum()); d = len(impl) - c
        odds, p = fisher_exact([[a, b], [c, d]], alternative=direction)
        arrow = "↑" if direction == "greater" else "↓"
        print(f"  {metric:25s} {arrow} channel {a}/{len(ch)}  impl {c}/{len(impl)}  OR={odds:.2f}  p={p:.3e}")

    print()
    print("=== Fisher's exact: channel vs background ===")
    for metric, direction in [
        ("has_cancer", "greater"),
        ("has_developmental", "less"),
        ("cancer_dominated", "greater"),
    ]:
        a = int(ch[metric].sum()); b = len(ch) - a
        c = int(bg_df[metric].sum()); d = len(bg_df) - c
        odds, p = fisher_exact([[a, b], [c, d]], alternative=direction)
        arrow = "↑" if direction == "greater" else "↓"
        print(f"  {metric:25s} {arrow} channel {a}/{len(ch)}  bg {c}/{len(bg_df)}  OR={odds:.2f}  p={p:.3e}")

    print()
    print("=== Per-gene breakdown (top 15 channel, top 10 impl) ===")
    print("Channel top-15 by cancer association count:")
    chview = df_with_any[df_with_any["class"]=="channel"].sort_values("n_cancer", ascending=False).head(15)
    print(chview[["gene", "n_cancer", "n_developmental", "max_cancer_score", "max_dev_score"]].to_string(index=False))
    print()
    print("Implementation by disease counts:")
    implview = df_with_any[df_with_any["class"]=="implementation"].sort_values("n_developmental", ascending=False)
    print(implview[["gene", "n_cancer", "n_developmental", "max_cancer_score", "max_dev_score"]].to_string(index=False))

    out = rel(Path("data/test_disease_spectrum.tsv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"\n[disease] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
