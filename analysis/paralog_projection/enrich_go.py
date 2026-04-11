"""Enrich a paralog pair table with GO semantic similarity.

GO semantic similarity measures how similar two genes' Gene Ontology
annotations are. We compute Lin similarity on the biological_process
and molecular_function sub-ontologies separately — each is a distinct
projection onto "what curators think this gene does" versus "what
molecular activity does it carry out."

Data:
  go-basic.obo           — ontology DAG (go.obolibrary.org)
  gene2go.gz             — NCBI gene-to-GO annotations (all species)

Computation strategy (fast enough for 1.77M pairs without being clever):
  1. Load the ontology once.
  2. Load gene2go filtered to tax_id == 9606 (human).
  3. Map HUGO symbol → NCBI Entrez gene id via the DepMap column names
     if possible, otherwise via mygene or a static cache.
  4. For each gene, collect its GO_BP / GO_MF / GO_CC annotation sets.
  5. For each paralog pair, compute Lin similarity between the two
     annotation sets (pairwise max over term × term, averaged).

Lin similarity uses Information Content (IC) from the term frequencies
in the annotation corpus. Two genes whose most-specific shared ancestor
term is rare (high IC) are more similar than two genes whose most-
specific shared ancestor is a very general term (low IC).

Usage:
  python enrich_go.py \\
      --pair-table data/pair_table_gm12878_primary_heavy_coess.parquet \\
      --obo data/go/go-basic.obo \\
      --gene2go data/go/gene2go.gz \\
      --out data/pair_table_gm12878_primary_heavy_coess_go.parquet
"""

import argparse
import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_ontology(obo_path: Path):
    """Lazy import so --list-features etc. work without goatools installed."""
    from goatools.obo_parser import GODag
    print(f"[go] loading {obo_path}", file=sys.stderr)
    return GODag(str(obo_path), prt=None)


def load_human_annotations(gene2go_gz: Path) -> dict[int, dict[str, set[str]]]:
    """Parse gene2go.gz filtered to human. Returns:
      {entrez_id: {'BP': {GO:...}, 'MF': {GO:...}, 'CC': {GO:...}}}
    """
    print(f"[go] parsing {gene2go_gz}", file=sys.stderr)
    t0 = time.time()
    out: dict[int, dict[str, set[str]]] = defaultdict(lambda: {"BP": set(), "MF": set(), "CC": set()})
    category_map = {
        "Process": "BP", "Function": "MF", "Component": "CC",
        "process": "BP", "function": "MF", "component": "CC",
    }
    with gzip.open(gene2go_gz, "rt") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            if parts[0] != "9606":
                continue
            try:
                entrez = int(parts[1])
            except ValueError:
                continue
            go_id = parts[2]
            category = parts[7]
            cat = category_map.get(category)
            if cat is None:
                continue
            out[entrez][cat].add(go_id)
    print(f"[go] {len(out)} human genes with annotations ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    return dict(out)


def build_sym_to_entrez_from_depmap_header(depmap_csv: Path) -> dict[str, int]:
    """Parse just the header of CRISPRGeneEffect.csv to get SYMBOL → Entrez map.

    DepMap column format is 'SYMBOL (ENTREZ_ID)', which gives us a curated
    symbol-to-entrez mapping without needing an extra fetch.
    """
    import re
    GENE_COL_RE = re.compile(r"^(?P<sym>[^ ]+) \((?P<entrez>\d+)\)$")
    print(f"[go] reading symbol→entrez map from {depmap_csv} header", file=sys.stderr)
    df_head = pd.read_csv(depmap_csv, nrows=0)
    sym_to_entrez: dict[str, int] = {}
    for col in df_head.columns:
        m = GENE_COL_RE.match(col)
        if m:
            sym = m.group("sym")
            if sym not in sym_to_entrez:
                sym_to_entrez[sym] = int(m.group("entrez"))
    print(f"[go] {len(sym_to_entrez)} symbol→entrez mappings", file=sys.stderr)
    return sym_to_entrez


def propagate_ancestors(godag, term_sets: dict[int, dict[str, set[str]]]) -> dict[int, dict[str, set[str]]]:
    """For each gene's annotation set, propagate to all ancestor GO terms.

    GO annotations are leaf-term; for semantic similarity we need the full
    is-a / part-of ancestor closure. GODag provides `.get_all_parents()`.
    """
    print("[go] propagating ancestors", file=sys.stderr)
    t0 = time.time()
    out: dict[int, dict[str, set[str]]] = {}
    cache: dict[str, set[str]] = {}
    def ancestors(go_id):
        if go_id in cache:
            return cache[go_id]
        node = godag.get(go_id)
        if node is None:
            cache[go_id] = set()
            return cache[go_id]
        anc = node.get_all_parents() | {go_id}
        cache[go_id] = anc
        return anc
    for entrez, cats in term_sets.items():
        new_cats = {}
        for cat, terms in cats.items():
            expanded: set[str] = set()
            for t in terms:
                expanded |= ancestors(t)
            new_cats[cat] = expanded
        out[entrez] = new_cats
    print(f"[go] propagation done ({time.time()-t0:.1f}s)", file=sys.stderr)
    return out


def compute_ic(
    annotations: dict[int, dict[str, set[str]]], cat: str
) -> dict[str, float]:
    """Information content per term: -log(freq / total_genes_annotated)."""
    n_annotated = 0
    freq: dict[str, int] = defaultdict(int)
    for genes_terms in annotations.values():
        terms = genes_terms.get(cat, set())
        if terms:
            n_annotated += 1
            for t in terms:
                freq[t] += 1
    ic: dict[str, float] = {}
    if n_annotated == 0:
        return ic
    for t, f in freq.items():
        p = f / n_annotated
        ic[t] = -np.log(p)
    return ic


def lin_similarity(
    a_terms: set[str], b_terms: set[str], ic: dict[str, float]
) -> float:
    """Lin similarity between two annotation sets.

    Classic Lin pairwise term similarity:
      sim_Lin(t1, t2) = 2 * IC(LCA(t1, t2)) / (IC(t1) + IC(t2))
    We approximate LCA by intersection of ancestor-expanded sets (so for
    pairs of already-expanded sets, the LCA is simply the most-informative
    shared term).
    """
    if not a_terms or not b_terms:
        return np.nan
    shared = a_terms & b_terms
    if not shared:
        return 0.0
    # Most-informative shared ancestor
    max_ic = max(ic.get(t, 0.0) for t in shared)
    # Denominator: most-informative term in each set
    ic_a = max((ic.get(t, 0.0) for t in a_terms), default=0.0)
    ic_b = max((ic.get(t, 0.0) for t in b_terms), default=0.0)
    denom = ic_a + ic_b
    if denom <= 0:
        return 0.0
    return float(2.0 * max_ic / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-table", type=Path, required=True)
    ap.add_argument("--obo", type=Path, default=Path("data/go/go-basic.obo"))
    ap.add_argument("--gene2go", type=Path, default=Path("data/go/gene2go.gz"))
    ap.add_argument("--depmap", type=Path, default=Path("data/depmap_CRISPRGeneEffect_26Q1.csv"),
                    help="DepMap CSV, used for its header's symbol→entrez mapping")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p

    godag = load_ontology(rel(args.obo))
    annotations_raw = load_human_annotations(rel(args.gene2go))
    annotations = propagate_ancestors(godag, annotations_raw)
    ic_bp = compute_ic(annotations, "BP")
    ic_mf = compute_ic(annotations, "MF")
    print(f"[go] BP terms with IC: {len(ic_bp)}, MF terms with IC: {len(ic_mf)}",
          file=sys.stderr)

    sym_to_entrez = build_sym_to_entrez_from_depmap_header(rel(args.depmap))

    print(f"[go] loading pair table", file=sys.stderr)
    pairs = pd.read_parquet(rel(args.pair_table))
    n = len(pairs)
    print(f"[go] {n:,} pairs", file=sys.stderr)

    # Per-gene BP/MF term sets
    bp_sets: dict[str, set[str]] = {}
    mf_sets: dict[str, set[str]] = {}
    for sym, entrez in sym_to_entrez.items():
        a = annotations.get(entrez)
        if a is None:
            continue
        if a.get("BP"):
            bp_sets[sym] = a["BP"]
        if a.get("MF"):
            mf_sets[sym] = a["MF"]
    print(f"[go] genes with BP: {len(bp_sets)}, MF: {len(mf_sets)}", file=sys.stderr)

    # Compute similarity for each pair
    print("[go] computing per-pair similarity", file=sys.stderr)
    t0 = time.time()
    bp_out = np.full(n, np.nan, dtype=np.float32)
    mf_out = np.full(n, np.nan, dtype=np.float32)
    gene_a_arr = pairs.gene_a.values
    gene_b_arr = pairs.gene_b.values
    for i in range(n):
        ga = gene_a_arr[i]
        gb = gene_b_arr[i]
        a_bp = bp_sets.get(ga)
        b_bp = bp_sets.get(gb)
        if a_bp is not None and b_bp is not None:
            bp_out[i] = lin_similarity(a_bp, b_bp, ic_bp)
        a_mf = mf_sets.get(ga)
        b_mf = mf_sets.get(gb)
        if a_mf is not None and b_mf is not None:
            mf_out[i] = lin_similarity(a_mf, b_mf, ic_mf)
        if i % 200_000 == 0 and i > 0:
            print(f"[go] {i:,}/{n:,} ({time.time()-t0:.1f}s)", file=sys.stderr)

    pairs = pairs.copy()
    pairs["go_bp_sim"] = bp_out
    pairs["go_mf_sim"] = mf_out

    n_bp = int(np.isfinite(bp_out).sum())
    n_mf = int(np.isfinite(mf_out).sum())
    print(f"[go] pairs with BP sim: {n_bp:,}, with MF sim: {n_mf:,}", file=sys.stderr)
    if n_bp:
        print(f"[go] BP distribution: median={np.nanmedian(bp_out):.3f} "
              f"q75={np.nanquantile(bp_out,0.75):.3f}", file=sys.stderr)
    if n_mf:
        print(f"[go] MF distribution: median={np.nanmedian(mf_out):.3f} "
              f"q75={np.nanquantile(mf_out,0.75):.3f}", file=sys.stderr)

    out_path = rel(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"[go] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
