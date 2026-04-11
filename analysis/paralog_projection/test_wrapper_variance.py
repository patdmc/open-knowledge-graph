"""Test 1: does wrapper-gene sequence variance correlate with species phenotype?

The wrapper variance hypothesis (see memory/project_wrapper_variance_hypothesis.md):

  Implementation genes (ancient biochemical core, bacterial origin where
  possible) should show low phylogeny-locked sequence variance across
  species — purifying selection holds them rigid everywhere.

  Wrapper / config-layer genes (eukaryotic-innovation scaffolds + kinases
  + factory proteins like BRCA1) should show HIGH, phenotype-correlated
  variance — they encode the organism's specific policy, which differs
  across species based on life history (lifespan, body size, cancer rate).

Concrete test: for the DDR channel genes + implementation controls, pull
ortholog sequences across a curated species set via the Ensembl REST API,
compute per-gene variance metrics, and correlate with species phenotype
(max lifespan from AnAge) where available.

This is not going to prove the full hypothesis on its own — it's a
directional first pass. If the variance ordering is
  implementation < routing-scaffold < factory-wrapper
on the test genes, we keep pushing. If it's flat, we rethink.

Species set chosen for phenotype spread at moderate phylogenetic distance:
  mouse, rat, human, chimp, dog, cow, horse, pig, cat, rabbit,
  naked_mole_rat (Heterocephalus glaber) — extreme longevity at small body,
  microbat                                 — intermediate longevity, tiny body,
  zebrafish, chicken, frog                 — distant vertebrate controls

Usage:
  python test_wrapper_variance.py
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ENSEMBL_REST = "https://rest.ensembl.org"

# Species chosen to span lifespan / body size gradients while keeping
# reasonably complete Ensembl Compara coverage. Species names must match
# Ensembl conventions (lowercase, underscore).
SPECIES = [
    "homo_sapiens",          # baseline
    "pan_troglodytes",       # chimp — close phylo
    "mus_musculus",          # mouse — short lifespan, small body
    "rattus_norvegicus",     # rat   — short lifespan, small body
    "heterocephalus_glaber_female",  # naked mole rat — extreme longevity
    "canis_lupus_familiaris",        # dog
    "bos_taurus",            # cow
    "sus_scrofa",            # pig
    "oryctolagus_cuniculus", # rabbit
    "myotis_lucifugus",      # little brown bat — long lifespan small body
    "equus_caballus",        # horse
    "danio_rerio",           # zebrafish
    "gallus_gallus",         # chicken
    "xenopus_tropicalis",    # frog
]

# Test genes: DDR channel mix of implementation (RAD51 family catalytic),
# scaffold/config-reader (RAD51 loading complex + BRCA), and a few
# implementation controls from outside DDR for baseline comparison.
TEST_GENES = {
    # Implementation (catalytic recombinase)
    "RAD51":   "implementation",
    "DMC1":    "implementation",
    # Routing / loading scaffolds (RAD51 paralogs that lost catalytic role)
    "RAD51B":  "routing",
    "RAD51C":  "routing",
    "RAD51D":  "routing",
    "XRCC2":   "routing",
    "XRCC3":   "routing",
    # Factory / config-reader wrappers (eukaryotic-innovation scaffolds)
    "BRCA1":   "factory",
    "BRCA2":   "factory",
    "PALB2":   "factory",
    "BARD1":   "factory",
    "MDC1":    "reader",
    "TP53BP1": "reader",
    # Config writers (kinases)
    "ATM":     "writer",
    "ATR":     "writer",
    "CHEK1":   "writer",
    "CHEK2":   "writer",
    # Implementation controls (strictly conserved housekeeping)
    "ACTB":    "implementation-control",
    "GAPDH":   "implementation-control",
    "RPL7":    "implementation-control",
    "EEF1A1":  "implementation-control",
}


def ensembl_get(path: str, cache: Path) -> dict | list | None:
    """GET an Ensembl REST endpoint with disk caching."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    url = f"{ENSEMBL_REST}{path}"
    req = urllib.request.Request(
        url,
        headers={"Content-Type": "application/json", "User-Agent": "paralog-projection/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        cache.write_text(json.dumps(data))
        return data
    except urllib.error.HTTPError as e:
        print(f"[wrapper-var] HTTP {e.code} on {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[wrapper-var] error on {url}: {e}", file=sys.stderr)
        return None


def get_orthologs(gene: str, cache_dir: Path, target_species: list[str]) -> dict[str, list]:
    """Return {species: [ortholog records]} via Ensembl Compara homology."""
    # Ask Ensembl for all homologs of this human gene, then filter to target species
    cache = cache_dir / f"homology_{gene}.json"
    data = ensembl_get(
        f"/homology/symbol/human/{urllib.parse.quote(gene)}"
        f"?type=orthologues;format=full",
        cache,
    )
    if data is None:
        return {}
    species_orthos: dict[str, list] = {s: [] for s in target_species}
    for entry in data.get("data", []):
        for homo in entry.get("homologies", []):
            sp = homo.get("target", {}).get("species")
            if sp in species_orthos:
                species_orthos[sp].append(homo)
    return species_orthos


def identity_metric(hom: dict) -> float | None:
    """Extract pairwise identity % from a homology record."""
    tgt = hom.get("target", {})
    perc_id = tgt.get("perc_id")
    if perc_id is None:
        return None
    try:
        return float(perc_id)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/compara"))
    ap.add_argument("--out", type=Path, default=Path("data/wrapper_variance.tsv"))
    args = ap.parse_args()

    here = Path(__file__).parent
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else here / args.cache_dir
    out_path = args.out if args.out.is_absolute() else here / args.out

    rows = []
    non_human = [s for s in SPECIES if s != "homo_sapiens"]
    for i, (gene, cls) in enumerate(TEST_GENES.items()):
        print(f"[wrapper-var] {i+1}/{len(TEST_GENES)}  {gene} ({cls})", file=sys.stderr)
        orthos = get_orthologs(gene, cache_dir, non_human)
        # Collect per-species identity (max over multiple orthologs per species)
        per_species_id: dict[str, float] = {}
        for sp, records in orthos.items():
            ids = [identity_metric(h) for h in records]
            ids = [x for x in ids if x is not None]
            if ids:
                per_species_id[sp] = max(ids)
        row = {"gene": gene, "class": cls}
        for sp in non_human:
            row[sp] = per_species_id.get(sp)
        rows.append(row)
        time.sleep(0.05)  # rate limit politely

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[wrapper-var] wrote {out_path}", file=sys.stderr)
    print()

    # ============================================================
    # Quick analysis: per-gene variance across species
    # ============================================================
    sp_cols = [c for c in df.columns if c not in ("gene", "class")]
    df["mean_id"] = df[sp_cols].mean(axis=1)
    df["std_id"] = df[sp_cols].std(axis=1)
    df["cv_id"] = df.std_id / df.mean_id
    df["min_id"] = df[sp_cols].min(axis=1)
    df["range_id"] = df[sp_cols].max(axis=1) - df[sp_cols].min(axis=1)
    df["n_species"] = df[sp_cols].notna().sum(axis=1)

    print("=== Per-gene variance across species ===")
    print(df[["gene", "class", "n_species", "mean_id", "std_id", "cv_id", "min_id", "range_id"]]
          .sort_values(["class", "cv_id"])
          .to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notnull(x) else "NaN"))
    print()

    print("=== Variance by class ===")
    grouped = df.groupby("class").agg(
        n_genes=("gene", "count"),
        mean_cv=("cv_id", "mean"),
        median_cv=("cv_id", "median"),
        mean_range=("range_id", "mean"),
    ).sort_values("median_cv")
    print(grouped.to_string())
    print()
    print("Expected ordering (low → high variance): implementation-control ≈ implementation < routing < reader/factory < writer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
