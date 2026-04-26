#!/usr/bin/env python3
"""
Analyze Fanconi/HR ortholog data — conservation curves and evolutionary architecture.

Designed to also run as a Google Colab notebook.
To use in Colab, copy cells between the # %% markers.
"""

# %% [markdown]
# # Evolutionary Architecture of the DNA Repair Pathway
#
# Reading the genome like a git log: which genes arrived when,
# and do the newer genes sit between older ones (escalation architecture)
# or at the ends (extension)?

# %% Imports and data loading
import csv
import json
from pathlib import Path

# If running in Colab, upload the CSV or fetch from GitHub
# from google.colab import files
# uploaded = files.upload()

DATA_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

def load_data():
    """Load ortholog data from CSV."""
    results = []
    csv_path = DATA_DIR / "fanconi_hr_orthologs.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["divergence_mya"] = int(row["divergence_mya"])
            row["has_ortholog"] = row["has_ortholog"] == "True"
            row["identity"] = float(row["identity"]) if row["identity"] else 0.0
            row["source_identity"] = float(row["source_identity"]) if row["source_identity"] else 0.0
            results.append(row)
    return results

data = load_data()
print(f"Loaded {len(data)} records")

# %% Gene categories
# Categorize genes by evolutionary role and approximate age

GENE_CATEGORIES = {
    # Ancient core repair (~2-3.5B years)
    "RAD51": {"category": "ancient_worker", "age_by": 3500, "layer": "execution"},
    "ERCC4": {"category": "ancient_worker", "age_by": 2000, "layer": "execution"},

    # RAD51 paralogs — meiotic recombination (~1-1.5B years)
    "RAD51B": {"category": "rad51_paralog", "age_by": 1500, "layer": "execution"},
    "RAD51C": {"category": "rad51_paralog", "age_by": 1500, "layer": "execution_and_alarm"},
    "RAD51D": {"category": "rad51_paralog", "age_by": 1500, "layer": "execution"},
    "XRCC2": {"category": "rad51_paralog", "age_by": 1500, "layer": "execution"},
    "XRCC3": {"category": "rad51_paralog", "age_by": 1500, "layer": "execution"},

    # Fanconi core complex — alarm system
    "FANCA": {"category": "fanc_core", "age_by": 450, "layer": "alarm"},
    "FANCB": {"category": "fanc_core", "age_by": 450, "layer": "alarm"},
    "FANCC": {"category": "fanc_core", "age_by": 450, "layer": "alarm"},
    "FANCE": {"category": "fanc_core", "age_by": 450, "layer": "alarm"},
    "FANCF": {"category": "fanc_core", "age_by": 450, "layer": "alarm"},
    "FANCG": {"category": "fanc_core", "age_by": 450, "layer": "alarm"},
    "FANCL": {"category": "fanc_core", "age_by": 600, "layer": "alarm"},
    "FANCM": {"category": "fanc_core", "age_by": 1000, "layer": "alarm"},

    # ID2 complex — alarm relay
    "FANCD2": {"category": "id2_complex", "age_by": 600, "layer": "alarm_relay"},
    "FANCI": {"category": "id2_complex", "age_by": 600, "layer": "alarm_relay"},

    # Vertebrate coordination — middleware
    "PALB2": {"category": "vertebrate_middleware", "age_by": 450, "layer": "coordination"},
    "BRIP1": {"category": "vertebrate_middleware", "age_by": 600, "layer": "coordination"},
    "SLX4": {"category": "vertebrate_middleware", "age_by": 600, "layer": "coordination"},

    # Vertebrate management — the trait
    "BRCA1": {"category": "vertebrate_management", "age_by": 350, "layer": "management"},
    "BRCA2": {"category": "vertebrate_management", "age_by": 450, "layer": "management"},

    # Drug targets
    "PARP1": {"category": "drug_target", "age_by": 600, "layer": "backup_repair"},
    "PARP2": {"category": "drug_target", "age_by": 600, "layer": "backup_repair"},
    "CDK4": {"category": "drug_target", "age_by": 450, "layer": "cell_cycle"},
    "CDK6": {"category": "drug_target", "age_by": 450, "layer": "cell_cycle"},
    "ESR1": {"category": "drug_target", "age_by": 600, "layer": "signaling"},
}

# %% Build presence/absence matrix
SPECIES_ORDER = [
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

GENES_IN_ORDER = list(GENE_CATEGORIES.keys())

def get_record(gene, species):
    """Get the record for a gene/species pair."""
    matches = [r for r in data if r["gene"] == gene and r["species"] == species]
    return matches[0] if matches else None

# %% Determine first appearance for each gene
print("\n=== FIRST APPEARANCE (earliest species with ortholog) ===\n")
print(f"{'Gene':<10} {'Category':<25} {'Layer':<20} {'First in':<12} {'Mya':<6} {'Identity'}")
print("-" * 90)

first_appearances = {}
for gene in GENES_IN_ORDER:
    first_species = None
    first_mya = 0
    first_identity = 0
    # Check species from most distant to closest
    for species, mya, label in SPECIES_ORDER:
        if species == "homo_sapiens":
            continue
        rec = get_record(gene, species)
        if rec and rec["has_ortholog"]:
            first_species = label
            first_mya = mya
            first_identity = rec["identity"]
            break

    cat = GENE_CATEGORIES[gene]
    first_appearances[gene] = {"species": first_species, "mya": first_mya}
    print(f"{gene:<10} {cat['category']:<25} {cat['layer']:<20} {first_species or 'N/A':<12} {first_mya:<6} {first_identity:.0f}%")

# %% Identify co-occurring appearances (same branch point)
print("\n\n=== CO-OCCURRING APPEARANCES (genes arriving at same branch point) ===\n")

by_branch_point = {}
for gene, info in first_appearances.items():
    mya = info["mya"]
    if mya not in by_branch_point:
        by_branch_point[mya] = []
    cat = GENE_CATEGORIES[gene]
    by_branch_point[mya].append(f"{gene} ({cat['layer']})")

for mya in sorted(by_branch_point.keys(), reverse=True):
    species_at_branch = [s[2] for s in SPECIES_ORDER if s[1] == mya]
    species_label = species_at_branch[0] if species_at_branch else "?"
    genes = by_branch_point[mya]
    print(f"\n{mya} Mya ({species_label}) — {len(genes)} genes:")
    for g in genes:
        print(f"  {g}")

# %% Conservation curves — identity vs divergence time
print("\n\n=== CONSERVATION CURVES (identity at each branch point) ===\n")

# Group genes by category for comparison
categories_to_plot = {
    "Ancient workers": ["RAD51", "ERCC4"],
    "RAD51 paralogs": ["RAD51C", "RAD51D", "XRCC3"],
    "FANC core (vertebrate)": ["FANCA", "FANCC", "FANCG"],
    "ID2 alarm": ["FANCD2", "FANCI"],
    "Management": ["BRCA1", "BRCA2"],
    "Drug targets": ["PARP1", "CDK6", "ESR1"],
}

for cat_name, genes in categories_to_plot.items():
    print(f"\n{cat_name}:")
    print(f"  {'Gene':<10}", end="")
    for _, mya, label in SPECIES_ORDER:
        if mya > 0:
            print(f" {label:>8}", end="")
    print()

    for gene in genes:
        print(f"  {gene:<10}", end="")
        for species, mya, label in SPECIES_ORDER:
            if mya == 0:
                continue
            rec = get_record(gene, species)
            if rec and rec["has_ortholog"]:
                print(f" {rec['identity']:>7.0f}%", end="")
            else:
                print(f" {'—':>8}", end="")
        print()

# %% Classify genes as inserts vs appends
print("\n\n=== INSERT vs APPEND CLASSIFICATION ===\n")
print("Insert = coordination gene that sits between older layers in the signal flow")
print("Append = gene that extends capability at the edge of the pathway")
print()

# Signal flow in the Fanconi pathway:
# Detection (FANCM) → Core complex (FANCA-G) → ID2 activation (FANCD2/I)
#   → BRCA1 coordination → PALB2 bridge → BRCA2 loading → RAD51/RAD51C execution
#
# If a gene sits BETWEEN two older genes in this flow, it's an insert.
# If it sits at the end of a chain, it's an append.

flow_order = [
    ("FANCM", 1000, "detection"),
    ("FANCL", 600, "core_complex"),
    ("FANCA", 450, "core_complex"),
    ("FANCC", 450, "core_complex"),
    ("FANCG", 450, "core_complex"),
    ("FANCD2", 600, "id2_alarm"),
    ("FANCI", 600, "id2_alarm"),
    ("BRCA1", 350, "management"),
    ("PALB2", 450, "bridge"),
    ("BRCA2", 450, "management"),
    ("RAD51C", 600, "execution"),
    ("RAD51", 1000, "execution"),
    ("ERCC4", 1000, "execution"),
]

print(f"{'Position':<4} {'Gene':<10} {'First Mya':<10} {'Role':<15} {'Classification'}")
print("-" * 60)

for i, (gene, first_mya, role) in enumerate(flow_order):
    # Check if neighbors in the flow are older
    older_before = False
    older_after = False
    if i > 0:
        older_before = flow_order[i-1][1] > first_mya or flow_order[i-1][1] == first_mya
    if i < len(flow_order) - 1:
        older_after = flow_order[i+1][1] > first_mya or flow_order[i+1][1] == first_mya

    if older_before and older_after:
        classification = "INSERT (between older layers)"
    elif not older_before and not older_after:
        classification = "ANCIENT (foundational)"
    else:
        classification = "EDGE (append or co-arrival)"

    actual_first = first_appearances.get(gene, {}).get("mya", "?")
    print(f"{i+1:<4} {gene:<10} {actual_first:<10} {role:<15} {classification}")

# %% Summary statistics
print("\n\n=== SUMMARY ===\n")

total_genes = len(GENES_IN_ORDER)
in_yeast = sum(1 for g in GENES_IN_ORDER if first_appearances[g]["mya"] >= 1000)
in_fly = sum(1 for g in GENES_IN_ORDER if first_appearances[g]["mya"] >= 600)
in_fish = sum(1 for g in GENES_IN_ORDER if first_appearances[g]["mya"] >= 450)
in_frog = sum(1 for g in GENES_IN_ORDER if first_appearances[g]["mya"] >= 350)

print(f"Total genes analyzed: {total_genes}")
print(f"Present in yeast (1B+): {in_yeast} ({100*in_yeast/total_genes:.0f}%)")
print(f"Present by fly/worm (600M+): {in_fly} ({100*in_fly/total_genes:.0f}%)")
print(f"Present by fish (450M+): {in_fish} ({100*in_fish/total_genes:.0f}%)")
print(f"Present by frog (350M+): {in_frog} ({100*in_frog/total_genes:.0f}%)")
print(f"\nThe vertebrate fish branch point (450 Mya) is the major refactor:")
print(f"  {in_fish - in_fly} genes arrive between fly/worm and fish")

# The fish explosion
fish_arrivals = [g for g in GENES_IN_ORDER
                 if first_appearances[g]["mya"] == 450]
print(f"\nGenes first appearing at the fish branch point:")
for g in fish_arrivals:
    cat = GENE_CATEGORIES[g]
    print(f"  {g} ({cat['layer']})")

print("\n\nData files in:", DATA_DIR)
print("  fanconi_hr_orthologs.csv — full data")
print("  fanconi_hr_presence_matrix.csv — presence/absence grid")
print("  fanconi_hr_orthologs.json — raw JSON")
