#!/usr/bin/env python3
"""
Fetch ortholog data for Hallmarks of Cancer genes NOT already in channel_gene_map.csv.
Covers: Apoptosis, Telomere/Immortality, Angiogenesis, Metastasis, additional repair.
"""

import json
import time
import csv
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

ENSEMBL_REST = "https://rest.ensembl.org"
DATA_DIR = Path(__file__).parent

SPECIES = [
    ("saccharomyces_cerevisiae", 1000, "Yeast"),
    ("drosophila_melanogaster", 600, "Fly"),
    ("caenorhabditis_elegans", 600, "Worm"),
    ("danio_rerio", 450, "Zebrafish"),
    ("xenopus_tropicalis", 350, "Frog"),
    ("gallus_gallus", 300, "Chicken"),
    ("monodelphis_domestica", 180, "Opossum"),
    ("mus_musculus", 90, "Mouse"),
    ("canis_lupus_familiaris", 85, "Dog"),
    ("homo_sapiens", 0, "Human"),
]

# Genes NOT already in channel_gene_map.csv
EXPANSION_GENES = {
    # Apoptosis
    "BCL2": "Apoptosis",
    "BCL2L1": "Apoptosis",  # BCL-XL
    "BCL2L11": "Apoptosis",  # BIM
    "BAX": "Apoptosis",
    "BAK1": "Apoptosis",
    "BID": "Apoptosis",
    "BAD": "Apoptosis",
    "CASP3": "Apoptosis",
    "CASP8": "Apoptosis",
    "CASP9": "Apoptosis",
    "XIAP": "Apoptosis",
    "BIRC5": "Apoptosis",  # Survivin
    "CYCS": "Apoptosis",  # Cytochrome c
    "APAF1": "Apoptosis",
    "FAS": "Apoptosis",
    "FASLG": "Apoptosis",

    # Telomere / Replicative Immortality
    "TERT": "Telomere",
    "TERC": "Telomere",
    "POT1": "Telomere",
    "TINF2": "Telomere",  # TIN2
    "TERF1": "Telomere",  # TRF1
    "TERF2": "Telomere",  # TRF2
    "TPP1": "Telomere",  # ACD
    "RAP1": "Telomere",  # TERF2IP

    # Angiogenesis
    "VEGFA": "Angiogenesis",
    "VEGFB": "Angiogenesis",
    "VEGFC": "Angiogenesis",
    "KDR": "Angiogenesis",  # VEGFR2
    "FLT1": "Angiogenesis",  # VEGFR1
    "FLT4": "Angiogenesis",  # VEGFR3
    "HIF1A": "Angiogenesis",
    "EPAS1": "Angiogenesis",  # HIF2A
    "VHL": "Angiogenesis",
    "ANGPT1": "Angiogenesis",
    "ANGPT2": "Angiogenesis",
    "TEK": "Angiogenesis",  # TIE2

    # Invasion / Metastasis (beyond what's in TissueArchitecture)
    "MMP2": "InvasionMetastasis",
    "MMP9": "InvasionMetastasis",
    "MMP14": "InvasionMetastasis",
    "SNAI1": "InvasionMetastasis",  # Snail
    "SNAI2": "InvasionMetastasis",  # Slug
    "TWIST1": "InvasionMetastasis",
    "ZEB1": "InvasionMetastasis",
    "ZEB2": "InvasionMetastasis",
    "VIM": "InvasionMetastasis",  # Vimentin
    "ITGB1": "InvasionMetastasis",

    # Additional repair (from our Fanconi fetch but not in channel_gene_map)
    "RAD51": "DDR_additional",
    "XRCC2": "DDR_additional",
    "XRCC3": "DDR_additional",
    "ERCC4": "DDR_additional",
    "FANCB": "DDR_additional",
    "FANCE": "DDR_additional",
    "FANCF": "DDR_additional",
    "FANCG": "DDR_additional",
    "FANCI": "DDR_additional",
    "FANCL": "DDR_additional",
    "FANCM": "DDR_additional",
    "BRIP1": "DDR_additional",
    "SLX4": "DDR_additional",
    "PARP1": "DDR_additional",
    "PARP2": "DDR_additional",

    # TP53 family (the guardian and its relatives)
    "TP63": "CellCycle_additional",
    "TP73": "CellCycle_additional",
}


def get_orthologs(gene_symbol, target_species, retries=3):
    params = urllib.parse.urlencode({
        "type": "orthologues",
        "target_species": target_species,
        "content-type": "application/json",
    })
    url = f"{ENSEMBL_REST}/homology/symbol/homo_sapiens/{gene_symbol}?{params}"
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                homologies = data.get("data", [{}])[0].get("homologies", [])
                return homologies
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return []
            elif e.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            else:
                if attempt == retries - 1:
                    print(f"  Error {e.code} for {gene_symbol} in {target_species}")
                    return []
                time.sleep(1)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"  Request failed for {gene_symbol} in {target_species}: {e}")
                return []
            time.sleep(1)
    return []


def main():
    gene_list = sorted(EXPANSION_GENES.keys())
    results = []

    # Check for checkpoint
    checkpoint_path = DATA_DIR / "expansion_checkpoint.json"
    done_pairs = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
            done_pairs = {(r["gene"], r["species"]) for r in results}
            print(f"Resuming: {len(done_pairs)} pairs done")

    total = len(gene_list) * len(SPECIES)
    print(f"Fetching {len(gene_list)} expansion genes across {len(SPECIES)} species")
    print(f"Total pairs: {total}, already done: {len(done_pairs)}")

    query_count = 0
    for gene in gene_list:
        channel = EXPANSION_GENES[gene]
        any_new = False

        for species_name, mya, label in SPECIES:
            if (gene, species_name) in done_pairs:
                continue

            if species_name == "homo_sapiens":
                results.append({
                    "gene": gene, "channel": channel, "species": species_name,
                    "divergence_mya": 0, "has_ortholog": True,
                    "ortholog_type": "self", "target_gene": gene,
                    "target_symbol": "", "identity": 100.0, "source_identity": 0,
                })
                done_pairs.add((gene, species_name))
                continue

            query_count += 1
            any_new = True
            homologies = get_orthologs(gene, species_name)

            if homologies:
                best = max(homologies, key=lambda h: h.get("target", {}).get("perc_id", 0))
                target = best.get("target", {})
                source = best.get("source", {})
                results.append({
                    "gene": gene, "channel": channel, "species": species_name,
                    "divergence_mya": mya, "has_ortholog": True,
                    "ortholog_type": best.get("type", "unknown"),
                    "target_gene": target.get("id", ""),
                    "target_symbol": target.get("protein_id", ""),
                    "identity": target.get("perc_id", 0),
                    "source_identity": source.get("perc_id", 0),
                })
            else:
                results.append({
                    "gene": gene, "channel": channel, "species": species_name,
                    "divergence_mya": mya, "has_ortholog": False,
                    "ortholog_type": "none", "target_gene": "",
                    "target_symbol": "", "identity": 0, "source_identity": 0,
                })

            done_pairs.add((gene, species_name))
            time.sleep(0.085)

            if query_count % 100 == 0:
                print(f"  Progress: {query_count} queries, {len(done_pairs)}/{total} pairs")
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f)

        if any_new:
            has = sum(1 for s, m, l in SPECIES if s != "homo_sapiens"
                      and any(r["gene"] == gene and r["species"] == s and r["has_ortholog"]
                              for r in results))
            print(f"{gene} ({channel}): orthologs in {has}/9 species")

    # Write CSV
    csv_path = DATA_DIR / "expansion_orthologs.csv"
    fieldnames = ["gene", "channel", "species", "divergence_mya", "has_ortholog",
                  "ortholog_type", "target_gene", "target_symbol",
                  "identity", "source_identity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["channel"], r["gene"], -r["divergence_mya"])))

    # JSON
    json_path = DATA_DIR / "expansion_orthologs.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Channel summary
    print("\n=== EXPANSION CHANNEL FIRST APPEARANCES ===\n")
    for channel in sorted(set(EXPANSION_GENES.values())):
        ch_genes = [g for g, c in EXPANSION_GENES.items() if c == channel]
        print(f"\n{channel} ({len(ch_genes)} genes):")
        for gene in sorted(ch_genes):
            for species_name, mya, label in SPECIES:
                if species_name == "homo_sapiens":
                    continue
                match = [r for r in results
                         if r["gene"] == gene and r["species"] == species_name and r["has_ortholog"]]
                if match:
                    print(f"  {gene:<12} first in {label:<12} ({mya} Mya) {match[0]['identity']:.0f}%")
                    break
            else:
                print(f"  {gene:<12} Human only")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nResults: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
