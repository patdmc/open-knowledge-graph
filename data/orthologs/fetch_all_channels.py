#!/usr/bin/env python3
"""
Fetch ortholog data for ALL coupling channel genes across species.

Reads gene list from channel_gene_map.csv.
Uses Ensembl REST API.
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

# Species at key evolutionary branch points
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

DIVERGENCE_MYA = {s[0]: s[1] for s in SPECIES}


def load_genes():
    """Load gene list from channel_gene_map.csv."""
    gene_map_path = DATA_DIR.parent / "channel_gene_map.csv"
    genes = {}
    with open(gene_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            genes[row["gene"]] = row["channel"]
    return genes


def get_orthologs(gene_symbol, target_species, retries=3):
    """Query Ensembl REST API for orthologs of a human gene in a target species."""
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
    genes = load_genes()
    gene_list = sorted(genes.keys())
    species_list = [s[0] for s in SPECIES]

    total_queries = len(gene_list) * (len(species_list) - 1)  # skip human self
    query_count = 0
    results = []

    # Check for existing partial results to resume
    checkpoint_path = DATA_DIR / "all_channels_checkpoint.json"
    done_pairs = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            existing = json.load(f)
            results = existing
            done_pairs = {(r["gene"], r["species"]) for r in existing}
            print(f"Resuming from checkpoint: {len(done_pairs)} pairs already done")

    print(f"Fetching orthologs for {len(gene_list)} genes across {len(species_list)} species")
    print(f"Total queries needed: {total_queries}")
    print(f"Already done: {len(done_pairs)}")
    print()

    for gene in gene_list:
        channel = genes[gene]
        any_new = False

        for species_name, mya, label in SPECIES:
            if (gene, species_name) in done_pairs:
                continue

            if species_name == "homo_sapiens":
                results.append({
                    "gene": gene,
                    "channel": channel,
                    "species": species_name,
                    "divergence_mya": 0,
                    "has_ortholog": True,
                    "ortholog_type": "self",
                    "target_gene": gene,
                    "target_symbol": "",
                    "identity": 100.0,
                    "source_identity": 0,
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
                    "gene": gene,
                    "channel": channel,
                    "species": species_name,
                    "divergence_mya": mya,
                    "has_ortholog": True,
                    "ortholog_type": best.get("type", "unknown"),
                    "target_gene": target.get("id", ""),
                    "target_symbol": target.get("protein_id", ""),
                    "identity": target.get("perc_id", 0),
                    "source_identity": source.get("perc_id", 0),
                })
            else:
                results.append({
                    "gene": gene,
                    "channel": channel,
                    "species": species_name,
                    "divergence_mya": mya,
                    "has_ortholog": False,
                    "ortholog_type": "none",
                    "target_gene": "",
                    "target_symbol": "",
                    "identity": 0,
                    "source_identity": 0,
                })

            done_pairs.add((gene, species_name))

            # Rate limiting — 15 req/sec allowed, use 12 to be safe
            time.sleep(0.085)

            if query_count % 100 == 0:
                print(f"  Progress: {query_count} queries done, {len(done_pairs)}/{total_queries + len(gene_list)} total pairs")
                # Save checkpoint
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f)

        if any_new:
            has = sum(1 for s, m, l in SPECIES if s != "homo_sapiens"
                      and any(r["gene"] == gene and r["species"] == s and r["has_ortholog"]
                              for r in results))
            print(f"{gene} ({channel}): orthologs in {has}/9 species")

    # Write final results
    csv_path = DATA_DIR / "all_channels_orthologs.csv"
    fieldnames = ["gene", "channel", "species", "divergence_mya", "has_ortholog",
                  "ortholog_type", "target_gene", "target_symbol",
                  "identity", "source_identity"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["channel"], r["gene"], -r["divergence_mya"])))

    print(f"\nResults written to {csv_path}")

    # Presence matrix by channel
    matrix_path = DATA_DIR / "all_channels_presence_matrix.csv"
    sorted_species = sorted(SPECIES, key=lambda s: s[1], reverse=True)

    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["channel", "gene"] + [f"{l} ({m}Mya)" for _, m, l in sorted_species]
        writer.writerow(header)

        for channel in sorted(set(genes.values())):
            channel_genes = sorted([g for g, c in genes.items() if c == channel])
            for gene in channel_genes:
                row = [channel, gene]
                for species_name, mya, label in sorted_species:
                    match = [r for r in results
                             if r["gene"] == gene and r["species"] == species_name]
                    if match and match[0]["has_ortholog"]:
                        row.append(f"{match[0]['identity']:.0f}%")
                    else:
                        row.append("-")
                writer.writerow(row)

    print(f"Presence matrix written to {matrix_path}")

    # JSON
    json_path = DATA_DIR / "all_channels_orthologs.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw JSON written to {json_path}")

    # Channel summary — first appearance by channel
    print("\n\n=== CHANNEL FIRST APPEARANCES ===\n")
    for channel in sorted(set(genes.values())):
        channel_genes = [g for g, c in genes.items() if c == channel]
        first_myas = []
        for gene in channel_genes:
            for species_name, mya, label in sorted_species:
                if species_name == "homo_sapiens":
                    continue
                match = [r for r in results
                         if r["gene"] == gene and r["species"] == species_name and r["has_ortholog"]]
                if match:
                    first_myas.append((gene, mya, label))
                    break
            else:
                first_myas.append((gene, 0, "Human only"))

        print(f"\n{channel} ({len(channel_genes)} genes):")
        for gene, mya, label in sorted(first_myas, key=lambda x: -x[1]):
            print(f"  {gene:<12} first in {label:<12} ({mya} Mya)")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("\nCheckpoint removed.")


if __name__ == "__main__":
    main()
