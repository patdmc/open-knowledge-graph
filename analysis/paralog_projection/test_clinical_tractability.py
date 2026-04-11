"""Test 3: are channel genes enriched for OpenTargets tractability signals?

OpenTargets Platform ships a pre-computed `tractability` vector per target
containing structured binary flags across two modalities (SM = small
molecule, AB = antibody) with labels like:
  Approved Drug, Advanced Clinical, Phase 1 Clinical,
  Structure with Ligand, High-Quality Pocket, Med-Quality Pocket,
  Druggable Family, High-Quality Ligand, ...

Sharper prediction under the higher-order/first-order framework:

  Catalytic A-role (DNMT, HDAC, metabolic enzymes)     → HIGH sm_pocket flags
  Catalytic M-role (kinases: ATM/ATR/CHEK/mTOR/PIK)    → HIGH sm_pocket flags (pocket) AND synthetic-lethality reachable
  Non-catalytic M-role (BRCA1/2, PALB2, 53BP1, MDC1)   → LOW sm_pocket flags, reachable only via synthetic lethality (not in OT)
  Non-catalytic A-role (histones, structural, ribosomal) → LOW sm_pocket flags AND unreachable

So OpenTargets tractability specifically measures "is there a small-molecule
pocket here" — which is orthogonal to "is this gene higher-order." Catalytic
genes of any order should be enriched; non-catalytic M-role genes should NOT
be (because direct tractability can't see synthetic lethality).

Hypothesis check: channel genes are a MIX of catalytic M-role (kinases) and
non-catalytic M-role (scaffolds). We should see MODERATE enrichment in the
channel set vs implementation controls — higher because kinases are present,
but diluted because scaffolds are too. Per-channel breakdown should track
the kinase/scaffold ratio of each channel.

Usage:
  python test_clinical_tractability.py
"""

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"


SEARCH_QUERY = """
query ($sym: String!) {
  search(queryString: $sym, entityNames: ["target"]) {
    hits { id entity name }
  }
}
"""

TARGET_QUERY = """
query ($id: String!) {
  target(ensemblId: $id) {
    id approvedSymbol
    drugAndClinicalCandidates { count }
    tractability { modality label value }
  }
}
"""


IMPLEMENTATION_CONTROLS = [
    "RAD51", "DMC1",
    "POLA1", "POLB", "POLD1", "POLE", "POLG", "POLH", "POLK",
    "LIG1", "LIG3", "LIG4", "FEN1",
    "TOP1", "TOP2A", "TOP2B", "TOP3A", "TOP3B",
    "RPS3", "RPS6", "RPL7", "RPL11", "RPL22", "RPL23",
    "ACTB", "GAPDH", "EEF1A1", "EEF1A2", "TUBB", "TUBA1A",
    "LARS", "VARS", "MARS1",
    "HK1", "HK2", "PKM", "LDHA",
]


def gql_post(query: str, variables: dict) -> dict | None:
    req = urllib.request.Request(
        OPENTARGETS_GQL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "paralog-projection/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            parsed = json.loads(body)
            err = parsed.get("errors", [{}])[0].get("message", body[:200])
            print(f"[clinical] GraphQL error: {err}", file=sys.stderr)
        except Exception:
            print(f"[clinical] HTTP {e.code}: {body[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[clinical] request error: {e}", file=sys.stderr)
        return None


def opentargets_query(gene: str, cache_dir: Path) -> dict | None:
    cache = cache_dir / f"ot_{gene}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    # 1. Resolve symbol → ensembl id
    sd = gql_post(SEARCH_QUERY, {"sym": gene})
    if sd is None:
        return None
    hits = (sd.get("data", {}) or {}).get("search", {}).get("hits") or []
    tgt = None
    for h in hits:
        if h.get("entity") == "target" and (h.get("name", "").upper() == gene.upper()):
            tgt = h; break
    if tgt is None:
        for h in hits:
            if h.get("entity") == "target":
                tgt = h; break
    if tgt is None:
        result = {"gene": gene, "ensembl_id": None, "found": False}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
        return result

    ensembl_id = tgt["id"]

    # 2. Fetch target with tractability + drug count
    td = gql_post(TARGET_QUERY, {"id": ensembl_id})
    if td is None:
        return None
    target = (td.get("data", {}) or {}).get("target") or {}
    drug_count = int(((target.get("drugAndClinicalCandidates") or {}).get("count")) or 0)
    trac_list = target.get("tractability") or []

    # Summarize SM tractability
    sm_flags = {
        f["label"]: bool(f["value"]) for f in trac_list
        if f.get("modality") == "SM"
    }
    ab_flags = {
        f["label"]: bool(f["value"]) for f in trac_list
        if f.get("modality") == "AB"
    }

    result = {
        "gene": gene,
        "ensembl_id": ensembl_id,
        "found": True,
        "drug_count": drug_count,
        "sm_flag_count": sum(1 for v in sm_flags.values() if v),
        "sm_approved_drug": sm_flags.get("Approved Drug", False),
        "sm_advanced_clinical": sm_flags.get("Advanced Clinical", False),
        "sm_phase1_clinical": sm_flags.get("Phase 1 Clinical", False),
        "sm_high_pocket": sm_flags.get("High-Quality Pocket", False),
        "sm_med_pocket": sm_flags.get("Med-Quality Pocket", False),
        "sm_structure_ligand": sm_flags.get("Structure with Ligand", False),
        "sm_druggable_family": sm_flags.get("Druggable Family", False),
        "ab_approved_drug": ab_flags.get("Approved Drug", False),
        "ab_advanced_clinical": ab_flags.get("Advanced Clinical", False),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel-map", type=Path, default=Path("../../data/channel_gene_map.csv"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/opentargets_v2"))
    ap.add_argument("--background-n", type=int, default=120)
    ap.add_argument("--string-aliases", type=Path,
                    default=Path("data/string/9606.protein.aliases.v12.0.txt.gz"))
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p

    channel_df = pd.read_csv(rel(args.channel_map))
    channel_set = set(channel_df.gene.unique())
    impl_set = set(IMPLEMENTATION_CONTROLS)

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

    channel_to_channel = {row.gene: row.channel for _, row in channel_df.iterrows()}

    test_genes: list[tuple[str, str, str | None]] = (
        [(g, "channel", channel_to_channel[g]) for g in sorted(channel_set)]
        + [(g, "implementation", None) for g in sorted(impl_set)]
        + [(g, "background", None) for g in sorted(bg)]
    )
    print(f"[clinical] channel:{len(channel_set)} impl:{len(impl_set)} bg:{len(bg)}", file=sys.stderr)
    print(f"[clinical] total queries: {len(test_genes)}", file=sys.stderr)

    cache_dir = rel(args.cache_dir)
    results = []
    for i, (gene, cls, ch) in enumerate(test_genes):
        if i % 25 == 0:
            print(f"[clinical] {i}/{len(test_genes)}", file=sys.stderr)
        r = opentargets_query(gene, cache_dir)
        if r is None:
            r = {"gene": gene, "found": False}
        r["class"] = cls
        r["channel"] = ch
        results.append(r)
        time.sleep(0.02)

    df = pd.DataFrame(results).fillna({"found": False, "drug_count": 0, "sm_flag_count": 0})
    # Coerce bool columns
    bool_cols = [c for c in df.columns if c.startswith("sm_") or c.startswith("ab_")]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    print(f"\n[clinical] raw results: {len(df)}  found: {df.found.sum() if 'found' in df.columns else 0}")

    print()
    print("=== Summary by class ===")
    summary = df.groupby("class").agg(
        n=("gene", "count"),
        mean_sm_flags=("sm_flag_count", "mean"),
        frac_sm_approved=("sm_approved_drug", "mean"),
        frac_sm_any_pocket=("sm_high_pocket", lambda x: (x | df.loc[x.index, "sm_med_pocket"]).mean()),
        frac_sm_ligand_structure=("sm_structure_ligand", "mean"),
        frac_sm_druggable_family=("sm_druggable_family", "mean"),
        frac_ab_approved=("ab_approved_drug", "mean"),
        mean_drug_count=("drug_count", "mean"),
    )
    print(summary.to_string())
    print()

    print("=== Fisher's exact: channel vs implementation ===")
    ch_df = df[df["class"] == "channel"]
    impl_df = df[df["class"] == "implementation"]
    for metric in ["sm_approved_drug", "sm_high_pocket", "sm_med_pocket",
                   "sm_structure_ligand", "sm_druggable_family", "ab_approved_drug"]:
        if metric not in df.columns:
            continue
        a = int(ch_df[metric].sum()); b = len(ch_df) - a
        c = int(impl_df[metric].sum()); d = len(impl_df) - c
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        print(f"  {metric:25s}  channel {a}/{len(ch_df)}  impl {c}/{len(impl_df)}  OR={odds:.2f}  p={p:.3e}")

    print()
    print("=== Fisher's exact: channel vs background ===")
    bg_df = df[df["class"] == "background"]
    for metric in ["sm_approved_drug", "sm_high_pocket", "sm_med_pocket",
                   "sm_structure_ligand", "sm_druggable_family", "ab_approved_drug"]:
        if metric not in df.columns:
            continue
        a = int(ch_df[metric].sum()); b = len(ch_df) - a
        c = int(bg_df[metric].sum()); d = len(bg_df) - c
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        print(f"  {metric:25s}  channel {a}/{len(ch_df)}  bg {c}/{len(bg_df)}  OR={odds:.2f}  p={p:.3e}")

    print()
    print("=== Per-channel sm_approved_drug and mean drug count ===")
    per_ch = df[df["class"] == "channel"].groupby("channel").agg(
        n=("gene", "count"),
        frac_sm_approved=("sm_approved_drug", "mean"),
        frac_any_pocket=("sm_high_pocket", lambda x: (x | df.loc[x.index, "sm_med_pocket"]).mean()),
        mean_drug_count=("drug_count", "mean"),
    ).sort_values("frac_sm_approved", ascending=False)
    print(per_ch.to_string())

    out = rel(Path("data/test_clinical_tractability.tsv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"\n[clinical] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
