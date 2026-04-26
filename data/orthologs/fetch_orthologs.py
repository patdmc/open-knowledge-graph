#!/usr/bin/env python3
"""
Fetch ortholog data for Fanconi anemia / HR repair pathway genes
across species at key evolutionary branch points.

Uses Ensembl REST API: https://rest.ensembl.org
"""

import json
import time
import urllib.request
import urllib.error
import urllib.parse
import csv
from pathlib import Path

ENSEMBL_REST = "https://rest.ensembl.org"

# The genes we care about — Fanconi/HR pathway
GENES = [
    # Ancient repair core (~2-3.5 billion years)
    "RAD51",

    # RAD51 paralogs (~1-1.5 billion years)
    "RAD51B", "RAD51C", "RAD51D", "XRCC2", "XRCC3",

    # Fanconi core complex (~800M-1B years)
    "FANCA", "FANCB", "FANCC", "FANCE", "FANCF", "FANCG", "FANCL", "FANCM",

    # ID2 complex (~800M-1B years)
    "FANCD2", "FANCI",

    # Ancient nuclease
    "ERCC4",  # FANCQ / XPF

    # Middle layer (~500M years)
    "PALB2",  # FANCN
    "BRIP1",  # FANCJ
    "SLX4",   # FANCP

    # Vertebrate management (~400-500M years)
    "BRCA1",  # FANCS
    "BRCA2",  # FANCD1

    # PARP (the synthetic lethal target)
    "PARP1", "PARP2",

    # CDK4/6 (Ibrance target) and estrogen receptor
    "CDK4", "CDK6", "ESR1",
]

# Species at key evolutionary branch points
# Using Ensembl species names
SPECIES = [
    # Yeast (~1B years divergence)
    "saccharomyces_cerevisiae",

    # Fruit fly (~500-600M years)
    "drosophila_melanogaster",

    # Nematode (~500-600M years)
    "caenorhabditis_elegans",

    # Zebrafish (~450M years — ray-finned fish)
    "danio_rerio",

    # Frog (~350M years — amphibian)
    "xenopus_tropicalis",

    # Chicken (~300M years — reptile/bird)
    "gallus_gallus",

    # Opossum (~180M years — marsupial)
    "monodelphis_domestica",

    # Mouse (~90M years — rodent)
    "mus_musculus",

    # Dog (~85M years — carnivore)
    "canis_lupus_familiaris",

    # Human (reference)
    "homo_sapiens",
]

# Approximate divergence times from human (millions of years)
DIVERGENCE_MYA = {
    "saccharomyces_cerevisiae": 1000,
    "drosophila_melanogaster": 600,
    "caenorhabditis_elegans": 600,
    "danio_rerio": 450,
    "xenopus_tropicalis": 350,
    "gallus_gallus": 300,
    "monodelphis_domestica": 180,
    "mus_musculus": 90,
    "canis_lupus_familiaris": 85,
    "homo_sapiens": 0,
}

def get_orthologs(gene_symbol, target_species):
    """Query Ensembl REST API for orthologs of a human gene in a target species."""
    params = urllib.parse.urlencode({
        "type": "orthologues",
        "target_species": target_species,
        "content-type": "application/json",
    })
    url = f"{ENSEMBL_REST}/homology/symbol/homo_sapiens/{gene_symbol}?{params}"

    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            homologies = data.get("data", [{}])[0].get("homologies", [])
            return homologies
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return []
        elif e.code == 429:
            time.sleep(2)
            return get_orthologs(gene_symbol, target_species)
        else:
            print(f"  Error {e.code} for {gene_symbol} in {target_species}")
            return []
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  Request failed for {gene_symbol} in {target_species}: {e}")
        return []


def main():
    output_dir = Path(__file__).parent
    results = []

    total_queries = len(GENES) * len(SPECIES)
    query_count = 0

    print(f"Fetching orthologs for {len(GENES)} genes across {len(SPECIES)} species")
    print(f"Total queries: {total_queries}")
    print()

    for gene in GENES:
        print(f"Gene: {gene}")
        for species in SPECIES:
            query_count += 1
            if species == "homo_sapiens":
                # Skip self-comparison
                results.append({
                    "gene": gene,
                    "species": species,
                    "divergence_mya": 0,
                    "has_ortholog": True,
                    "ortholog_type": "self",
                    "target_gene": gene,
                    "identity": 100.0,
                })
                continue

            homologies = get_orthologs(gene, species)

            if homologies:
                # Take the best hit (highest identity)
                best = max(homologies, key=lambda h: h.get("target", {}).get("perc_id", 0))
                target = best.get("target", {})
                source = best.get("source", {})

                results.append({
                    "gene": gene,
                    "species": species,
                    "divergence_mya": DIVERGENCE_MYA.get(species, -1),
                    "has_ortholog": True,
                    "ortholog_type": best.get("type", "unknown"),
                    "target_gene": target.get("id", ""),
                    "target_symbol": target.get("protein_id", ""),
                    "identity": target.get("perc_id", 0),
                    "source_identity": source.get("perc_id", 0),
                })
                print(f"  {species}: YES ({target.get('perc_id', 0):.1f}% identity)")
            else:
                results.append({
                    "gene": gene,
                    "species": species,
                    "divergence_mya": DIVERGENCE_MYA.get(species, -1),
                    "has_ortholog": False,
                    "ortholog_type": "none",
                    "target_gene": "",
                    "target_symbol": "",
                    "identity": 0,
                    "source_identity": 0,
                })
                print(f"  {species}: NO")

            # Rate limiting — Ensembl allows 15 requests/second
            time.sleep(0.1)

            if query_count % 50 == 0:
                print(f"\n  Progress: {query_count}/{total_queries}\n")

    # Write results to CSV
    csv_path = output_dir / "fanconi_hr_orthologs.csv"
    fieldnames = ["gene", "species", "divergence_mya", "has_ortholog",
                   "ortholog_type", "target_gene", "target_symbol",
                   "identity", "source_identity"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")
    print(f"Total results: {len(results)}")

    # Also write a presence/absence matrix for quick viewing
    matrix_path = output_dir / "fanconi_hr_presence_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: gene, then species sorted by divergence time
        sorted_species = sorted(SPECIES, key=lambda s: DIVERGENCE_MYA.get(s, 0), reverse=True)
        header = ["gene"] + [f"{s} ({DIVERGENCE_MYA[s]}Mya)" for s in sorted_species]
        writer.writerow(header)

        for gene in GENES:
            row = [gene]
            for species in sorted_species:
                match = [r for r in results if r["gene"] == gene and r["species"] == species]
                if match and match[0]["has_ortholog"]:
                    row.append(f"{match[0]['identity']:.0f}%")
                else:
                    row.append("-")
            writer.writerow(row)

    print(f"Presence matrix written to {matrix_path}")

    # Write raw JSON for detailed analysis
    json_path = output_dir / "fanconi_hr_orthologs.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Raw JSON written to {json_path}")


if __name__ == "__main__":
    main()
