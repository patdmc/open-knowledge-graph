"""Query Open Targets Platform for HBOC gene → developmental phenotype associations.

Tests the framework prediction (project_developmental_phenotype_predicts_cancer.md):
HBOC / FA-pathway genes should show gene-disease associations with VACTERL-spectrum
developmental anomalies, beyond just the bi-allelic Fanconi anemia syndromes.

Source: Open Targets Platform GraphQL API (https://platform.opentargets.org/).
Schema: ~/Downloads/schema.graphql (mirrored to repo if needed).

Output: JSON files per gene in data/opentargets/ + a parsed comparison table.
"""
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "analysis" / "paralog_projection" / "data" / "opentargets"
OUTDIR.mkdir(parents=True, exist_ok=True)

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

GENES = {
    "BRCA1":  "ENSG00000012048",
    "BRCA2":  "ENSG00000139618",
    "PALB2":  "ENSG00000083093",
    "BRIP1":  "ENSG00000136492",
    "RAD51C": "ENSG00000108384",
    "RAD51D": "ENSG00000185379",
}

QUERY = """
query GeneAssoc($id: String!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: 0, size: 100}) {
      count
      rows {
        score
        datatypeScores { id score }
        datasourceScores { id score }
        disease {
          id
          name
          therapeuticAreas { id name }
        }
      }
    }
  }
}
"""


def fetch(symbol: str, ensg: str) -> dict:
    payload = {"query": QUERY, "variables": {"id": ensg}}
    out_path = OUTDIR / f"ot_{symbol}.json"
    cmd = [
        "curl", "-sSL", "--max-time", "30",
        "-X", "POST", OT_GRAPHQL,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]
    raw = subprocess.check_output(cmd)
    data = json.loads(raw)
    if "errors" in data:
        raise RuntimeError(f"{symbol} GraphQL errors: {data['errors']}")
    out_path.write_bytes(raw)
    return data


def parse_developmental_hits(data: dict, gene: str) -> list:
    """Return non-cancer developmental / syndrome hits."""
    rows = data["data"]["target"]["associatedDiseases"]["rows"]
    cancer_re = re.compile(r"cancer|carcin|tumor|tumour|neoplas|leukem|lymphom|melanom|sarcom|adenoma", re.I)
    hits = []
    for r in rows:
        name = r["disease"]["name"]
        if cancer_re.search(name):
            continue
        # Get datasource breakdown
        ds = {d["id"]: d["score"] for d in r["datasourceScores"]}
        dt = {d["id"]: d["score"] for d in r["datatypeScores"]}
        hits.append({
            "gene": gene,
            "disease_id": r["disease"]["id"],
            "disease_name": name,
            "score": r["score"],
            "genetic_association": dt.get("genetic_association", 0),
            "literature": dt.get("literature", 0),
            "eva_score": ds.get("eva", 0),
            "therapeutic_areas": [t["name"] for t in r["disease"]["therapeuticAreas"]],
        })
    return hits


if __name__ == "__main__":
    all_hits = []
    for sym, ensg in GENES.items():
        data = fetch(sym, ensg)
        hits = parse_developmental_hits(data, sym)
        all_hits.extend(hits)
        print(f"\n=== {sym} ===")
        print(f"  {data['data']['target']['associatedDiseases']['count']} total disease associations")
        for h in hits[:10]:
            print(f"  {h['score']:.3f}  ga={h['genetic_association']:.3f}  eva={h['eva_score']:.3f}  {h['disease_name']}")

    out_csv = OUTDIR / "developmental_hits_summary.csv"
    import csv
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["gene", "disease_id", "disease_name", "score", "genetic_association", "literature", "eva_score", "therapeutic_areas"])
        w.writeheader()
        for h in all_hits:
            h["therapeutic_areas"] = "|".join(h["therapeutic_areas"])
            w.writerow(h)
    print(f"\nSaved summary to {out_csv}")
