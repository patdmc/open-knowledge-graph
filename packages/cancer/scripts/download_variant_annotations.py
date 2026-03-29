#!/usr/bin/env python3
"""
Download variant-level functional annotations from OncoKB and CIViC.

Sources:
  1. OncoKB public TSVs (may require API token)
  2. CIViC nightly bulk downloads (fully open)

Saves results to gnn/data/cache/oncokb/
"""

import os
import sys
import json
import csv
import urllib.request
import urllib.error
import ssl
import certifi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use certifi for SSL verification, fall back to unverified if needed
try:
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

from gnn.config import GNN_CACHE, GNN_RESULTS

ONCOKB_DIR = os.path.join(GNN_CACHE, "oncokb")
os.makedirs(ONCOKB_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load our gene list (509 genes from expanded_channel_map)
# ---------------------------------------------------------------------------
def load_project_genes():
    """Load all genes from the expanded channel map."""
    ecm_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    with open(ecm_path) as f:
        ecm = json.load(f)
    genes = set()
    for gene, info in ecm.items():
        if gene and gene != "nan":
            genes.add(gene)
    print(f"Loaded {len(genes)} project genes from expanded_channel_map.json")
    return genes


# ---------------------------------------------------------------------------
# 2. Try downloading OncoKB public data
# ---------------------------------------------------------------------------
def try_oncokb_download():
    """
    Try to download OncoKB annotated variants.
    Returns (rows, success_bool).
    """
    # Try GitHub raw URLs for various versions (newest first)
    github_versions = ["v4.25", "v4.24", "v4.23", "v4.22", "v4.20", "v4.18", "v4.15", "v4.10"]
    urls = []
    for ver in github_versions:
        urls.append(f"https://raw.githubusercontent.com/oncokb/oncokb-public/master/data/{ver}/allAnnotatedVariants.tsv")
    # Also try the API endpoints
    urls.extend([
        "https://www.oncokb.org/api/v1/utils/allAnnotatedVariants.tsv",
        "https://oncokb.org/api/v1/utils/allAnnotatedVariants.tsv",
    ])

    for url in urls:
        print(f"Trying OncoKB: {url}")
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (research-download)")
            resp = urllib.request.urlopen(req, timeout=30, context=SSL_CTX)
            data = resp.read().decode("utf-8")
            lines = data.strip().split("\n")
            if len(lines) > 1:
                save_path = os.path.join(ONCOKB_DIR, "allAnnotatedVariants_raw.tsv")
                with open(save_path, "w") as f:
                    f.write(data)
                print(f"  Downloaded {len(lines)-1} rows -> {save_path}")
                return lines, True
            else:
                print(f"  Empty response from {url}")
        except Exception as e:
            print(f"  Failed: {e}")

    # Try OncoKB curated genes
    curated_url = "https://www.oncokb.org/api/v1/utils/allCuratedGenes.tsv"
    print(f"Trying OncoKB curated genes: {curated_url}")
    try:
        req = urllib.request.Request(curated_url)
        req.add_header("User-Agent", "Mozilla/5.0 (research-download)")
        resp = urllib.request.urlopen(req, timeout=30, context=SSL_CTX)
        data = resp.read().decode("utf-8")
        lines = data.strip().split("\n")
        if len(lines) > 1:
            save_path = os.path.join(ONCOKB_DIR, "allCuratedGenes_raw.tsv")
            with open(save_path, "w") as f:
                f.write(data)
            print(f"  Downloaded {len(lines)-1} curated gene rows -> {save_path}")
    except Exception as e:
        print(f"  Curated genes failed: {e}")

    return None, False


# ---------------------------------------------------------------------------
# 3. Download CIViC variant data (fully open)
# ---------------------------------------------------------------------------
def download_civic():
    """
    Download CIViC variant summaries (nightly dump) and variant molecular profiles.
    Returns (lines_list, success_bool).
    """
    urls_to_try = [
        ("nightly-VariantSummaries.tsv",
         "https://civicdb.org/downloads/nightly/nightly-VariantSummaries.tsv"),
        ("nightly-VariantSummaries.tsv",
         "https://civicdb.org/downloads/nightly/nightly-VariantSummaries.tsv"),
    ]

    for fname, url in urls_to_try:
        print(f"Trying CIViC: {url}")
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (research-download)")
            resp = urllib.request.urlopen(req, timeout=60, context=SSL_CTX)
            data = resp.read().decode("utf-8")
            lines = data.strip().split("\n")
            if len(lines) > 1:
                save_path = os.path.join(ONCOKB_DIR, "civic_VariantSummaries_raw.tsv")
                with open(save_path, "w") as f:
                    f.write(data)
                print(f"  Downloaded {len(lines)-1} CIViC variant rows -> {save_path}")
                return lines, True
        except Exception as e:
            print(f"  Failed: {e}")

    # Try the CIViC API for clinical evidence
    print("Trying CIViC GraphQL API for variants...")
    try:
        import json as _json
        api_url = "https://civicdb.org/api/graphql"

        # CIViC GraphQL - get all variants with molecular profiles
        query = """
        {
          variants(first: 5000) {
            totalCount
            nodes {
              id
              name
              feature {
                name
              }
              molecularProfiles {
                nodes {
                  id
                  name
                }
              }
              variantTypes {
                name
              }
            }
          }
        }
        """
        req = urllib.request.Request(api_url, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "Mozilla/5.0 (research-download)")
        body = _json.dumps({"query": query}).encode("utf-8")
        resp = urllib.request.urlopen(req, body, timeout=60, context=SSL_CTX)
        result = _json.loads(resp.read().decode("utf-8"))

        if "data" in result and "variants" in result["data"]:
            variants = result["data"]["variants"]["nodes"]
            total = result["data"]["variants"]["totalCount"]
            save_path = os.path.join(ONCOKB_DIR, "civic_variants_graphql.json")
            with open(save_path, "w") as f:
                _json.dump(result["data"]["variants"], f, indent=2)
            print(f"  Downloaded {len(variants)} / {total} CIViC variants via GraphQL")
            return variants, True
    except Exception as e:
        print(f"  GraphQL failed: {e}")

    return None, False


# ---------------------------------------------------------------------------
# 4. Download CIViC clinical evidence items (has functional significance)
# ---------------------------------------------------------------------------
def download_civic_evidence():
    """Download CIViC clinical evidence to get significance/direction."""
    urls = [
        ("nightly-ClinicalEvidenceSummaries.tsv",
         "https://civicdb.org/downloads/nightly/nightly-ClinicalEvidenceSummaries.tsv"),
    ]
    for fname, url in urls:
        print(f"Trying CIViC evidence: {url}")
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (research-download)")
            resp = urllib.request.urlopen(req, timeout=60, context=SSL_CTX)
            data = resp.read().decode("utf-8")
            lines = data.strip().split("\n")
            if len(lines) > 1:
                save_path = os.path.join(ONCOKB_DIR, "civic_ClinicalEvidenceSummaries_raw.tsv")
                with open(save_path, "w") as f:
                    f.write(data)
                print(f"  Downloaded {len(lines)-1} CIViC evidence rows -> {save_path}")
                return lines, True
        except Exception as e:
            print(f"  Failed: {e}")
    return None, False


# ---------------------------------------------------------------------------
# 5. Parse OncoKB data
# ---------------------------------------------------------------------------
def parse_oncokb(lines, project_genes):
    """Parse OncoKB allAnnotatedVariants TSV. Returns list of dicts."""
    reader = csv.DictReader(lines, delimiter="\t")
    rows = []
    for row in reader:
        gene = row.get("Gene", row.get("gene", "")).strip()
        if gene in project_genes:
            rows.append({
                "gene": gene,
                "protein_change": row.get("Alteration", row.get("alteration", "")),
                "oncogenic_effect": row.get("Oncogenicity", row.get("oncogenicity", "")),
                "mutation_effect": row.get("Mutation Effect", row.get("mutationEffect", "")),
                "evidence_level": row.get("Highest Level", row.get("highestLevel", "")),
                "source": "OncoKB",
            })
    return rows


# ---------------------------------------------------------------------------
# 6. Parse CIViC variant summaries
# ---------------------------------------------------------------------------
def parse_civic_tsv(lines, project_genes):
    """Parse CIViC nightly VariantSummaries TSV."""
    reader = csv.DictReader(lines, delimiter="\t")
    rows = []
    for row in reader:
        # Try multiple column names for gene
        gene = (row.get("gene", "") or row.get("Gene", "") or
                row.get("feature_name", "")).strip()
        # Handle fusions in feature_name (e.g. "BCR::ABL1")
        if "::" in gene:
            for g in gene.split("::"):
                if g.strip() in project_genes:
                    gene = g.strip()
                    break
        if gene not in project_genes:
            continue

        variant_name = row.get("variant", row.get("Variant", ""))
        variant_types = row.get("variant_types", "")

        rows.append({
            "gene": gene,
            "protein_change": variant_name,
            "oncogenic_effect": "unknown",
            "mutation_effect": variant_types if variant_types else "",
            "evidence_level": "",
            "source": "CIViC",
        })
    return rows


def parse_civic_evidence_tsv(lines, project_genes):
    """Parse CIViC clinical evidence summaries for significance data.

    The evidence file has 'molecular_profile' (e.g. 'JAK2 V617F') but no 'gene' column.
    We extract gene from the molecular_profile field.
    """
    reader = csv.DictReader(lines, delimiter="\t")
    rows = []
    seen = set()
    for row in reader:
        mol_profile = row.get("molecular_profile", "").strip()
        # Extract gene: take first token (e.g. "JAK2 V617F" -> "JAK2")
        # Also handle fusions like "BCR::ABL1 ..." -> try both parts
        if not mol_profile:
            continue

        parts = mol_profile.split()
        gene_candidates = []
        if "::" in parts[0]:
            # Fusion
            gene_candidates = parts[0].split("::")
        else:
            gene_candidates = [parts[0]]

        matched_gene = None
        for g in gene_candidates:
            if g in project_genes:
                matched_gene = g
                break
        if not matched_gene:
            continue

        variant = " ".join(parts[1:]) if len(parts) > 1 else mol_profile
        significance = row.get("significance", "")
        evidence_type = row.get("evidence_type", "")
        evidence_level = row.get("evidence_level", "")
        evidence_direction = row.get("evidence_direction", "")

        key = (matched_gene, variant, significance, evidence_type)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "gene": matched_gene,
            "protein_change": variant,
            "oncogenic_effect": significance if significance else "unknown",
            "mutation_effect": evidence_type,
            "evidence_level": evidence_level,
            "evidence_direction": evidence_direction,
            "source": "CIViC_evidence",
        })
    return rows


def parse_civic_graphql(variants_data, project_genes):
    """Parse CIViC GraphQL response."""
    if isinstance(variants_data, dict) and "nodes" in variants_data:
        nodes = variants_data["nodes"]
    else:
        nodes = variants_data

    rows = []
    for v in nodes:
        gene = ""
        if isinstance(v, dict):
            feature = v.get("feature", {})
            if feature:
                gene = feature.get("name", "")
        if gene not in project_genes:
            continue

        variant_name = v.get("name", "")
        variant_types = [vt.get("name", "") for vt in v.get("variantTypes", [])]

        rows.append({
            "gene": gene,
            "protein_change": variant_name,
            "oncogenic_effect": "unknown",
            "mutation_effect": "; ".join(variant_types) if variant_types else "",
            "evidence_level": "",
            "source": "CIViC",
        })
    return rows


# ---------------------------------------------------------------------------
# 7. Enrich with project gene-level GOF/LOF and channel info
# ---------------------------------------------------------------------------
def enrich_with_project_annotations(all_rows, project_genes):
    """
    Add gene-level GOF/LOF from config.py GENE_FUNCTION, and channel from
    expanded_channel_map. Also infer variant-level functional impact where possible.
    """
    from gnn.config import GENE_FUNCTION, CHANNEL_MAP

    # Load expanded channel map for 509 genes
    ecm_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")
    with open(ecm_path) as f:
        ecm = json.load(f)

    for row in all_rows:
        gene = row["gene"]
        # Add channel
        if gene in ecm:
            row["channel"] = ecm[gene]["channel"]
        elif gene in CHANNEL_MAP:
            row["channel"] = CHANNEL_MAP[gene]
        else:
            row["channel"] = ""

        # Add gene-level function annotation
        row["gene_level_function"] = GENE_FUNCTION.get(gene, "")

        # Infer variant-level GOF/LOF from variant type + gene function
        variant_type = row.get("mutation_effect", "")
        gene_func = GENE_FUNCTION.get(gene, "")
        inferred = ""

        # If variant type suggests truncating -> LOF
        truncating_keywords = {"nonsense", "frameshift", "splice_donor", "splice_acceptor",
                               "stop_gained", "Truncating"}
        missense_keywords = {"missense", "missense_variant"}

        if any(kw in variant_type.lower() for kw in truncating_keywords):
            inferred = "LOF"
        elif any(kw in variant_type.lower() for kw in missense_keywords):
            # Missense: depends on gene role
            if gene_func == "GOF":
                inferred = "likely_GOF"
            elif gene_func == "LOF":
                inferred = "likely_LOF"
            elif gene_func == "context":
                inferred = "context_dependent"
        elif "amplification" in variant_type.lower() or "overexpression" in variant_type.lower():
            inferred = "GOF"
        elif "deletion" in variant_type.lower() or "loss" in variant_type.lower():
            inferred = "LOF"

        row["inferred_function"] = inferred

    return all_rows


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Downloading variant-level functional annotations")
    print("=" * 70)

    project_genes = load_project_genes()

    all_rows = []

    # --- OncoKB ---
    print("\n--- OncoKB ---")
    oncokb_lines, oncokb_ok = try_oncokb_download()
    if oncokb_ok and isinstance(oncokb_lines, list) and isinstance(oncokb_lines[0], str):
        parsed = parse_oncokb(oncokb_lines, project_genes)
        print(f"  Filtered to {len(parsed)} variants in project genes")
        all_rows.extend(parsed)

    # --- CIViC variants ---
    print("\n--- CIViC Variant Summaries ---")
    civic_data, civic_ok = download_civic()
    if civic_ok:
        if isinstance(civic_data, list) and civic_data and isinstance(civic_data[0], str):
            parsed = parse_civic_tsv(civic_data, project_genes)
        elif isinstance(civic_data, list) and civic_data and isinstance(civic_data[0], dict):
            parsed = parse_civic_graphql(civic_data, project_genes)
        else:
            parsed = []
        print(f"  Filtered to {len(parsed)} variants in project genes")
        all_rows.extend(parsed)

    # --- CIViC evidence ---
    print("\n--- CIViC Clinical Evidence ---")
    civic_ev_lines, civic_ev_ok = download_civic_evidence()
    if civic_ev_ok:
        parsed = parse_civic_evidence_tsv(civic_ev_lines, project_genes)
        print(f"  Filtered to {len(parsed)} evidence items in project genes")
        all_rows.extend(parsed)

    # --- Save combined ---
    if not all_rows:
        print("\nERROR: No data was downloaded from any source!")
        sys.exit(1)

    # --- Enrich with project annotations ---
    print("\n--- Enriching with project GOF/LOF and channel annotations ---")
    all_rows = enrich_with_project_annotations(all_rows, project_genes)

    # Count unique genes covered
    genes_with_data = set(r["gene"] for r in all_rows)
    print(f"\n--- Summary ---")
    print(f"Total annotation rows: {len(all_rows)}")
    print(f"Genes with annotations: {len(genes_with_data)} / {len(project_genes)}")
    print(f"Genes missing: {len(project_genes - genes_with_data)}")

    # Count inferred functions
    inferred_counts = {}
    for r in all_rows:
        inf = r.get("inferred_function", "")
        if inf:
            inferred_counts[inf] = inferred_counts.get(inf, 0) + 1
    if inferred_counts:
        print(f"\nInferred variant-level functions:")
        for func, count in sorted(inferred_counts.items(), key=lambda x: -x[1]):
            print(f"  {func}: {count}")

    # Save full data
    out_path = os.path.join(ONCOKB_DIR, "variant_annotations_combined.csv")
    fieldnames = ["gene", "protein_change", "oncogenic_effect", "mutation_effect",
                  "evidence_level", "evidence_direction", "source",
                  "channel", "gene_level_function", "inferred_function"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["gene"], r["protein_change"])):
            writer.writerow(row)
    print(f"\nSaved combined annotations -> {out_path}")

    # Save gene coverage summary
    coverage = {}
    for gene in sorted(project_genes):
        gene_rows = [r for r in all_rows if r["gene"] == gene]
        if gene_rows:
            sources = set(r["source"] for r in gene_rows)
            coverage[gene] = {
                "n_variants": len(gene_rows),
                "sources": sorted(sources),
            }
        else:
            coverage[gene] = {"n_variants": 0, "sources": []}

    coverage_path = os.path.join(ONCOKB_DIR, "gene_coverage.json")
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"Saved gene coverage -> {coverage_path}")

    # --- Create deduplicated variant-level mapping ---
    # Merge variant summaries and evidence into one row per (gene, variant)
    variant_map = {}
    for row in all_rows:
        key = (row["gene"], row["protein_change"])
        if key not in variant_map:
            variant_map[key] = {
                "gene": row["gene"],
                "protein_change": row["protein_change"],
                "channel": row.get("channel", ""),
                "gene_level_function": row.get("gene_level_function", ""),
                "variant_types": set(),
                "oncogenic_effects": set(),
                "evidence_levels": set(),
                "evidence_directions": set(),
                "sources": set(),
                "inferred_function": row.get("inferred_function", ""),
            }
        vm = variant_map[key]
        if row.get("mutation_effect"):
            vm["variant_types"].add(row["mutation_effect"])
        if row.get("oncogenic_effect") and row["oncogenic_effect"] != "unknown":
            vm["oncogenic_effects"].add(row["oncogenic_effect"])
        if row.get("evidence_level"):
            vm["evidence_levels"].add(row["evidence_level"])
        if row.get("evidence_direction"):
            vm["evidence_directions"].add(row["evidence_direction"])
        vm["sources"].add(row.get("source", ""))
        # Prefer non-empty inferred function
        if not vm["inferred_function"] and row.get("inferred_function"):
            vm["inferred_function"] = row["inferred_function"]

    # Save deduplicated mapping
    dedup_path = os.path.join(ONCOKB_DIR, "variant_functional_map.csv")
    dedup_fields = ["gene", "protein_change", "channel", "gene_level_function",
                    "inferred_function", "variant_types", "oncogenic_effects",
                    "evidence_levels", "evidence_directions", "sources"]
    with open(dedup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dedup_fields)
        writer.writeheader()
        for key in sorted(variant_map.keys()):
            vm = variant_map[key]
            writer.writerow({
                "gene": vm["gene"],
                "protein_change": vm["protein_change"],
                "channel": vm["channel"],
                "gene_level_function": vm["gene_level_function"],
                "inferred_function": vm["inferred_function"],
                "variant_types": "; ".join(sorted(vm["variant_types"])),
                "oncogenic_effects": "; ".join(sorted(vm["oncogenic_effects"])),
                "evidence_levels": "; ".join(sorted(vm["evidence_levels"])),
                "evidence_directions": "; ".join(sorted(vm["evidence_directions"])),
                "sources": "; ".join(sorted(vm["sources"])),
            })
    print(f"Saved deduplicated variant map ({len(variant_map)} unique variants) -> {dedup_path}")

    # Print top genes
    print("\nTop 20 genes by annotation count:")
    gene_counts = sorted(
        [(g, c["n_variants"]) for g, c in coverage.items()],
        key=lambda x: -x[1]
    )
    for gene, count in gene_counts[:20]:
        print(f"  {gene}: {count} annotations")

    missing = sorted(g for g, c in coverage.items() if c["n_variants"] == 0)
    if missing:
        print(f"\nGenes with no annotations ({len(missing)}):")
        for i in range(0, len(missing), 10):
            print(f"  {', '.join(missing[i:i+10])}")


if __name__ == "__main__":
    main()
