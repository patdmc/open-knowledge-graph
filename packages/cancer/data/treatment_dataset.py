"""
Treatment-Conditional Survival Dataset — TCGA patients with mutations,
treatment vectors, and enriched per-mutation node features.

Each patient's mutations are looked up in the survival atlas (same as AtlasDataset),
then enriched with 8 additional features from expression, CNA, DepMap, synthetic
lethality, CIViC drug sensitivity/resistance, GOF/LOF, and CNA context.

Node feature layout (26-dim):
  Original 14 from atlas_dataset:
    [0]  log_hr
    [1]  ci_width
    [2]  tier
    [3]  is_hub
    [4-9] channel_onehot (6 channels)
    [10] is_harmful (hr > 1.1)
    [11] is_protective (hr < 0.9)
    [12] log(n_patients) normalized
    [13] normalized protein position

  Enrichment 8:
    [14] expression_z: z-scored mean expression for (gene, cancer_type)
    [15] cna_score: mean CNA for (gene, cancer_type)
    [16] depmap_dependency: DepMap CRISPR dependency score
    [17] has_sl_partner_mutated: 1 if patient has mutation in known SL partner
    [18] civic_sensitivity: 1 if (gene, protein_change) has sensitivity evidence
    [19] civic_resistance: 1 if (gene, protein_change) has resistance evidence
    [20] gof_lof: +1 GOF, -1 LOF, 0 context/unknown
    [21] cna_context: +1 amp_freq > 0.05, -1 del_freq > 0.05, 0 otherwise

  Treatment × channel (per-node, enables channel-specific treatment effects):
    [22] chemotherapy
    [23] endocrine
    [24] targeted
    [25] immunotherapy

Treatment vector (11-dim binary):
    [0] surgery, [1] radiation, [2] chemotherapy, [3] endocrine, [4] targeted,
    [5] immunotherapy, [6] platinum, [7] taxane, [8] anthracycline,
    [9] antimetabolite, [10] alkylating
"""

import os
import sys
import re
import json
import time
import csv
import ssl
import urllib.parse
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES, ALL_GENES,
    GENE_FUNCTION, NON_SILENT, TRUNCATING,
)

MAX_NODES = 32
NODE_FEAT_DIM = 26  # 22 base + 4 systemic treatment flags per node

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "tcga")
DEPMAP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "depmap")
SL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "synthetic_lethality")
CIVIC_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "oncokb")

# Pre-compute hub gene set for fast lookup
_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)

# Protein lengths for position normalization
PROTEIN_LENGTHS = {
    'TP53': 393, 'KRAS': 189, 'BRAF': 766, 'PIK3CA': 1068,
    'PTEN': 403, 'APC': 2843, 'EGFR': 1210, 'BRCA1': 1863,
    'BRCA2': 3418, 'ATM': 3056, 'RB1': 928, 'SMAD4': 552,
    'NF1': 2818, 'CDH1': 882, 'ARID1A': 2285, 'FBXW7': 707,
    'CTNNB1': 781, 'PIK3R1': 724, 'STK11': 433, 'MAP3K1': 1512,
    'ERBB2': 1255, 'FGFR3': 806, 'ESR1': 595, 'AR': 919,
    'GATA3': 443, 'FOXA1': 472, 'MYC': 439, 'CDKN2A': 156,
    'JAK1': 1154, 'JAK2': 1132, 'B2M': 119, 'NOTCH1': 2555,
    'MSH6': 1360, 'MSH2': 934, 'MLH1': 756, 'POLE': 2286,
    'POLD1': 1107, 'NRAS': 189, 'AKT1': 480, 'MTOR': 2549,
    'ERBB3': 1342, 'FGFR2': 821, 'FGFR1': 822, 'MET': 1390,
}

# TCGA pan_can_atlas study abbreviations
TCGA_STUDIES = [
    "acc", "blca", "brca", "cesc", "chol", "coadread", "dlbc", "esca",
    "gbm", "hnsc", "kich", "kirc", "kirp", "laml", "lgg", "lihc",
    "luad", "lusc", "meso", "ov", "paad", "pcpg", "prad", "sarc",
    "skcm", "stad", "tgct", "thca", "thym", "ucec", "ucs", "uvm",
]

# TCGA project_id -> short cancer type (matches expression/CNA summary files)
PROJECT_TO_CANCER = {
    f"TCGA-{abbrev.upper()}": abbrev.upper() for abbrev in TCGA_STUDIES
}

# TCGA cancer type -> DepMap lineage mapping
TCGA_TO_DEPMAP_LINEAGE = {
    "ACC": "Adrenal Gland",
    "BLCA": "Bladder/Urinary Tract",
    "BRCA": "Breast",
    "CESC": "Cervix",
    "CHOL": "Biliary Tract",
    "COADREAD": "Bowel",
    "DLBC": "Lymphoid",
    "ESCA": "Esophagus/Stomach",
    "GBM": "CNS/Brain",
    "HNSC": "Head and Neck",
    "KICH": "Kidney",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "LAML": "Myeloid",
    "LGG": "CNS/Brain",
    "LIHC": "Liver",
    "LUAD": "Lung",
    "LUSC": "Lung",
    "MESO": "Pleura",
    "OV": "Ovary/Fallopian Tube",
    "PAAD": "Pancreas",
    "PCPG": "Peripheral Nervous System",
    "PRAD": "Prostate",
    "SARC": "Soft Tissue",
    "SKCM": "Skin",
    "STAD": "Esophagus/Stomach",
    "TGCT": "Testis",
    "THCA": "Thyroid",
    "THYM": "Thyroid",
    "UCEC": "Uterus",
    "UCS": "Uterus",
    "UVM": "Eye",
}


# ---------------------------------------------------------------------------
# Drug classification
# ---------------------------------------------------------------------------

_PLATINUM = {"cisplatin", "carboplatin", "oxaliplatin"}
_TAXANE = {"paclitaxel", "docetaxel", "nab-paclitaxel", "cabazitaxel"}
_ANTHRACYCLINE = {"doxorubicin", "epirubicin", "daunorubicin", "idarubicin"}
_ANTIMETABOLITE = {
    "fluorouracil", "5-fu", "gemcitabine", "capecitabine",
    "pemetrexed", "methotrexate",
}
_ALKYLATING = {
    "cyclophosphamide", "temozolomide", "lomustine", "carmustine",
    "ifosfamide", "dacarbazine",
}
_ENDOCRINE = {
    "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant",
    "goserelin", "leuprolide", "bicalutamide", "enzalutamide", "abiraterone",
}
_TARGETED = {
    "trastuzumab", "bevacizumab", "erlotinib", "gefitinib", "sorafenib",
    "sunitinib", "imatinib", "lapatinib", "vemurafenib", "olaparib",
    "palbociclib", "everolimus", "cetuximab", "rituximab", "crizotinib",
    "dabrafenib", "trametinib", "osimertinib", "regorafenib", "lenvatinib",
}
_IMMUNOTHERAPY = {
    "nivolumab", "pembrolizumab", "ipilimumab", "atezolizumab",
    "durvalumab", "avelumab", "interferon", "interleukin",
}


def classify_drug(agent_name):
    """Classify a drug name into treatment sub-classes.

    Returns a set of indices into the 11-dim treatment vector:
        0=surgery, 1=radiation, 2=chemotherapy, 3=endocrine, 4=targeted,
        5=immunotherapy, 6=platinum, 7=taxane, 8=anthracycline,
        9=antimetabolite, 10=alkylating
    """
    if not isinstance(agent_name, str) or not agent_name.strip():
        return set()

    name = agent_name.strip().lower()
    # Strip common suffixes: "hydrochloride", "citrate", etc.
    for suffix in ["hydrochloride", "citrate", "acetate", "sulfate", "disodium"]:
        name = name.replace(suffix, "").strip()

    indices = set()

    if name in _PLATINUM:
        indices.update({2, 6})  # chemotherapy + platinum
    elif name in _TAXANE:
        indices.update({2, 7})  # chemotherapy + taxane
    elif name in _ANTHRACYCLINE:
        indices.update({2, 8})  # chemotherapy + anthracycline
    elif name in _ANTIMETABOLITE:
        indices.update({2, 9})  # chemotherapy + antimetabolite
    elif name in _ALKYLATING:
        indices.update({2, 10})  # chemotherapy + alkylating
    elif name in _ENDOCRINE:
        indices.add(3)
    elif name in _TARGETED:
        indices.add(4)
    elif name in _IMMUNOTHERAPY:
        indices.add(5)

    return indices


def classify_treatment_type(treatment_type):
    """Classify a GDC treatment_type string into broad categories.

    Returns a set of indices into the 11-dim treatment vector.
    """
    if not isinstance(treatment_type, str):
        return set()

    tt = treatment_type.lower()
    indices = set()

    if "surgery" in tt or "hysterectomy" in tt or "biopsy" in tt or "ablation" in tt:
        indices.add(0)
    if "radiation" in tt or "brachytherapy" in tt:
        indices.add(1)
    if "chemotherapy" in tt:
        indices.add(2)
    if "hormone" in tt:
        indices.add(3)
    if "targeted" in tt:
        indices.add(4)
    if "immuno" in tt:
        indices.add(5)

    return indices


# ---------------------------------------------------------------------------
# cBioPortal mutation download
# ---------------------------------------------------------------------------

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _api_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60, context=_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def _api_post(url, body, retries=3):
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180, context=_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def load_entrez_map():
    cache_path = os.path.join(CACHE_DIR, "entrez_map.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    raise FileNotFoundError(f"entrez_map.json not found at {cache_path}")


def download_tcga_mutations(output_path=None):
    """Download TCGA mutations for our genes via cBioPortal API.

    Fetches mutations across all 32 pan_can_atlas studies, filtered to our
    gene set using entrez gene IDs. Saves as tcga_mutations.csv.
    """
    if output_path is None:
        output_path = os.path.join(CACHE_DIR, "tcga_mutations.csv")

    if os.path.exists(output_path):
        print(f"Mutations already cached at {output_path}", flush=True)
        return output_path

    print("Downloading TCGA mutations from cBioPortal...", flush=True)
    entrez_map = load_entrez_map()
    entrez_to_hugo = {v: k for k, v in entrez_map.items()}
    entrez_ids = list(entrez_map.values())

    base = "https://www.cbioportal.org/api"
    all_rows = []

    for abbrev in TCGA_STUDIES:
        study_id = f"{abbrev}_tcga_pan_can_atlas_2018"
        cancer = abbrev.upper()
        print(f"  {cancer}: ", end="", flush=True)

        # Find mutation profile
        profiles = _api_get(f"{base}/studies/{study_id}/molecular-profiles")
        if not profiles:
            print("no profiles", flush=True)
            continue

        mut_profile = None
        for p in profiles:
            if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
                mut_profile = p["molecularProfileId"]
                break
        if not mut_profile:
            print("no mutation profile", flush=True)
            continue

        # Fetch mutations filtered to our genes
        url = f"{base}/molecular-profiles/{mut_profile}/mutations/fetch?projection=DETAILED"
        body = {
            "sampleListId": f"{study_id}_all",
            "entrezGeneIds": entrez_ids,
        }
        result = _api_post(url, body)
        if not result:
            print("fetch failed", flush=True)
            continue

        count = 0
        for m in result:
            eid = m.get("entrezGeneId")
            gene = entrez_to_hugo.get(eid)
            if not gene:
                continue
            mut_type = m.get("mutationType", "")
            if mut_type not in NON_SILENT:
                continue

            row = {
                "patient_id": m.get("patientId", ""),
                "sample_id": m.get("sampleId", ""),
                "gene": gene,
                "protein_change": m.get("proteinChange", ""),
                "mutation_type": mut_type,
                "cancer_type": cancer,
                "chromosome": m.get("chr", ""),
                "start_position": m.get("startPosition", ""),
                "end_position": m.get("endPosition", ""),
                "ref_allele": m.get("referenceAllele", ""),
                "var_allele": m.get("variantAllele", ""),
            }
            all_rows.append(row)
            count += 1

        print(f"{count} mutations", flush=True)
        time.sleep(0.3)

    if all_rows:
        with open(output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"Saved {len(all_rows)} mutations to {output_path}", flush=True)
    else:
        print("WARNING: No mutations fetched!", flush=True)

    return output_path


# ---------------------------------------------------------------------------
# Feature lookup builders
# ---------------------------------------------------------------------------

def parse_position(pc):
    if not isinstance(pc, str):
        return None
    m = re.search(r'[A-Z*]?(\d+)', pc)
    return int(m.group(1)) if m else None


def get_channel_pos_id(gene):
    """Return 0-11 index: channel_idx * 2 + (0 if hub, 1 if leaf)."""
    ch = CHANNEL_MAP.get(gene)
    if ch is None:
        return 0
    ch_idx = CHANNEL_NAMES.index(ch) if ch in CHANNEL_NAMES else 0
    is_hub = gene in _HUB_SET
    return ch_idx * 2 + (0 if is_hub else 1)


def _get_neo4j_driver():
    """Get a shared Neo4j driver for enrichment data loading."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "openknowledgegraph"),
    )


def _load_expression_lookup():
    """Build (gene, cancer_type) -> z-scored expression from Neo4j EXPRESSION_IN edges."""
    try:
        driver = _get_neo4j_driver()
        lookup = {}
        with driver.session() as s:
            result = s.run("""
                MATCH (g:Gene)-[r:EXPRESSION_IN]->(ct:CancerType)
                RETURN g.name AS gene, ct.name AS cancer_type, r.z_score AS z
            """)
            for rec in result:
                z = rec["z"]
                if z is not None:
                    lookup[(rec["gene"], rec["cancer_type"])] = float(z)
        driver.close()
        if lookup:
            return lookup
    except Exception:
        pass
    # Fallback to CSV
    path = os.path.join(CACHE_DIR, "tcga_expression_summary.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    lookup = {}
    for gene, gdf in df.groupby("gene"):
        mean_val = gdf["mean"].mean()
        std_val = gdf["mean"].std()
        if std_val < 1e-8:
            std_val = 1.0
        for _, row in gdf.iterrows():
            z = (row["mean"] - mean_val) / std_val
            lookup[(row["gene"], row["cancer_type"])] = float(z)
    return lookup


def _load_cna_lookup():
    """Build (gene, cancer_type) -> CNA info from Neo4j CNA_IN edges."""
    try:
        driver = _get_neo4j_driver()
        lookup = {}
        with driver.session() as s:
            result = s.run("""
                MATCH (g:Gene)-[r:CNA_IN]->(ct:CancerType)
                RETURN g.name AS gene, ct.name AS cancer_type,
                       r.amp_freq AS amp, r.del_freq AS del, r.mean_cna AS mean_cna
            """)
            for rec in result:
                lookup[(rec["gene"], rec["cancer_type"])] = {
                    "mean_cna": float(rec["mean_cna"] or 0),
                    "amp_freq": float(rec["amp"] or 0),
                    "del_freq": float(rec["del"] or 0),
                }
        driver.close()
        if lookup:
            return lookup
    except Exception:
        pass
    # Fallback to CSV
    path = os.path.join(CACHE_DIR, "tcga_cna_summary.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["gene"], row["cancer_type"])] = {
            "mean_cna": float(row.get("mean_cna", 0)),
            "amp_freq": float(row.get("amp_freq", 0)),
            "del_freq": float(row.get("del_freq", 0)),
        }
    return lookup


def _load_depmap_lookup():
    """Build (gene, lineage) -> dependency score from Neo4j ESSENTIAL_IN edges."""
    try:
        driver = _get_neo4j_driver()
        lookup = {}
        with driver.session() as s:
            result = s.run("""
                MATCH (g:Gene)-[r:ESSENTIAL_IN]->(l:Lineage)
                RETURN g.name AS gene, l.name AS lineage,
                       r.dependency_score AS score
            """)
            for rec in result:
                score = rec["score"]
                if score is not None:
                    lookup[(rec["gene"], rec["lineage"])] = float(score)
        driver.close()
        if lookup:
            return lookup
    except Exception:
        pass
    # Fallback to CSV
    path = os.path.join(DEPMAP_CACHE, "depmap_dependency_matrix.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    lookup = {}
    lineages = [c for c in df.columns if c != "gene"]
    for _, row in df.iterrows():
        gene = row["gene"]
        if gene not in set(ALL_GENES):
            continue
        for lin in lineages:
            val = row[lin]
            if pd.notna(val):
                lookup[(gene, lin)] = float(val)
    return lookup


def _load_sl_lookup():
    """Build gene -> set of SL partner genes from Neo4j SL_PARTNER edges."""
    try:
        driver = _get_neo4j_driver()
        lookup = {}
        with driver.session() as s:
            result = s.run("""
                MATCH (g1:Gene)-[:SL_PARTNER]-(g2:Gene)
                RETURN g1.name AS gene1, g2.name AS gene2
            """)
            for rec in result:
                a, b = rec["gene1"], rec["gene2"]
                lookup.setdefault(a, set()).add(b)
                lookup.setdefault(b, set()).add(a)
        driver.close()
        if lookup:
            return lookup
    except Exception:
        pass
    # Fallback to CSV
    path = os.path.join(SL_CACHE, "synthetic_lethality_both_in_set.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        lookup.setdefault(a, set()).add(b)
        lookup.setdefault(b, set()).add(a)
    return lookup


def _load_civic_lookup():
    """Build sensitivity/resistance/function lookups from Neo4j edges + CIViC CSV."""
    sensitivity = {}
    resistance = {}
    func_lookup = {}

    try:
        driver = _get_neo4j_driver()
        with driver.session() as s:
            # Sensitivity edges
            result = s.run("""
                MATCH (g:Gene)-[r:HAS_SENSITIVITY_EVIDENCE]->(t)
                RETURN g.name AS gene, r.protein_change AS pc
            """)
            for rec in result:
                pc = rec["pc"] or ""
                sensitivity[(rec["gene"], pc)] = True

            # Resistance edges
            result = s.run("""
                MATCH (g:Gene)-[r:HAS_RESISTANCE_EVIDENCE]->(t)
                RETURN g.name AS gene, r.protein_change AS pc
            """)
            for rec in result:
                pc = rec["pc"] or ""
                resistance[(rec["gene"], pc)] = True
        driver.close()
    except Exception:
        pass

    # CIViC CSV for functional annotations (GOF/LOF) — not yet as edges
    path = os.path.join(CIVIC_CACHE, "variant_functional_map.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            gene = row["gene"]
            pc = row["protein_change"]
            effects = str(row.get("oncogenic_effects", ""))
            directions = str(row.get("evidence_directions", ""))

            if "Sensitivity" in effects and "Supports" in directions:
                sensitivity[(gene, pc)] = True
            if "Resistance" in effects and "Supports" in directions:
                resistance[(gene, pc)] = True

            inf = row.get("inferred_function", "")
            if isinstance(inf, str) and inf in ("GOF", "LOF"):
                func_lookup[(gene, pc)] = inf

    return sensitivity, resistance, func_lookup


# ---------------------------------------------------------------------------
# Treatment dataset class
# ---------------------------------------------------------------------------

class TreatmentDataset:

    def __init__(self):
        print("=" * 60, flush=True)
        print("TreatmentDataset — loading TCGA data", flush=True)
        print("=" * 60, flush=True)

        # Load treatment/clinical data
        self.treatments_df = pd.read_csv(
            os.path.join(CACHE_DIR, "tcga_treatment_data.csv")
        )

        # Load or download mutations
        mut_path = os.path.join(CACHE_DIR, "tcga_mutations.csv")
        if not os.path.exists(mut_path):
            download_tcga_mutations(mut_path)
        self.mutations = pd.read_csv(mut_path)
        print(f"Loaded {len(self.mutations)} mutations", flush=True)

        # Load everything from Neo4j in a single snapshot
        print("Loading graph snapshot from Neo4j...", flush=True)
        self._load_graph_snapshot()
        print(f"  Atlas: T1={len(self.t1)}, T2={len(self.t2)}, "
              f"T3={len(self.t3)}, T4={len(self.t4)}", flush=True)
        print(f"  Expression: {len(self.expr_lookup)}, CNA: {len(self.cna_lookup)}, "
              f"DepMap: {len(self.depmap_lookup)}, SL pairs: {len(self.sl_lookup)}, "
              f"CIViC sens: {len(self.civic_sens)}, res: {len(self.civic_res)}",
              flush=True)

        # Load METABRIC data
        metabric_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache", "metabric"
        )
        metabric_clin_path = os.path.join(metabric_dir, "metabric_clinical.csv")
        metabric_mut_path = os.path.join(metabric_dir, "metabric_mutations.csv")
        if os.path.exists(metabric_clin_path) and os.path.exists(metabric_mut_path):
            self.metabric_clinical = pd.read_csv(metabric_clin_path)
            self.metabric_mutations = pd.read_csv(metabric_mut_path)
            print(f"  METABRIC: {len(self.metabric_clinical)} patients, "
                  f"{len(self.metabric_mutations)} mutations", flush=True)
        else:
            self.metabric_clinical = None
            self.metabric_mutations = None

    def _build_patient_table(self):
        """Build per-patient clinical table from treatment data.

        Returns DataFrame with one row per patient: case_id, project_id,
        vital_status, survival_time, event, cancer_type.
        """
        df = self.treatments_df.copy()

        # Deduplicate to one row per patient for clinical info
        clinical = df.groupby("case_id").first().reset_index()
        clinical = clinical[["case_id", "project_id", "vital_status",
                             "days_to_death", "days_to_last_follow_up"]].copy()

        # Survival time: days_to_death if dead, else days_to_last_follow_up
        clinical["event"] = clinical["vital_status"].apply(
            lambda x: 1 if isinstance(x, str) and x.strip().lower() == "dead" else 0
        )
        clinical["time"] = clinical.apply(
            lambda r: r["days_to_death"] if pd.notna(r["days_to_death"]) and r["event"] == 1
            else r["days_to_last_follow_up"],
            axis=1,
        )
        clinical["time"] = pd.to_numeric(clinical["time"], errors="coerce")
        clinical = clinical.dropna(subset=["time"])
        clinical = clinical[clinical["time"] > 0]

        # Cancer type from project_id
        clinical["cancer_type"] = clinical["project_id"].map(PROJECT_TO_CANCER)
        clinical = clinical.dropna(subset=["cancer_type"])

        print(f"Patients with valid survival: {len(clinical)}", flush=True)
        return clinical

    def _build_treatment_vectors(self):
        """Build per-patient 11-dim treatment vectors.

        Returns dict: case_id -> np.array(11,)
        """
        df = self.treatments_df.copy()
        patient_vecs = {}

        for case_id, grp in df.groupby("case_id"):
            vec = np.zeros(11, dtype=np.float32)
            for _, row in grp.iterrows():
                # Classify by treatment_type
                tt = row.get("treatment_type", "")
                tt_indices = classify_treatment_type(tt)
                for idx in tt_indices:
                    vec[idx] = 1.0

                # Classify by therapeutic_agents (drug name)
                agent = row.get("therapeutic_agents", "")
                drug_indices = classify_drug(agent)
                for idx in drug_indices:
                    vec[idx] = 1.0

            patient_vecs[case_id] = vec

        return patient_vecs

    def _build_mutation_index(self):
        """Build patient_id -> list of (gene, protein_change, mutation_type) from TCGA mutations.

        Also builds patient_id -> cancer_type mapping from mutation data.
        """
        patient_muts = {}
        patient_cancer = {}
        for _, row in self.mutations.iterrows():
            pid = row["patient_id"]
            gene = row["gene"]
            pc = row.get("protein_change", "")
            mt = row.get("mutation_type", "")
            ct = row.get("cancer_type", "")
            patient_muts.setdefault(pid, []).append((gene, pc, mt))
            if ct:
                patient_cancer[pid] = ct
        return patient_muts, patient_cancer

    def _load_graph_snapshot(self):
        """Load all model data from Neo4j in a single connection.

        One driver, multiple queries, all lookups populated.
        This is the single source of truth — no CSV fallbacks needed.
        """
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "openknowledgegraph"),
        )

        # Atlas (tiers 1-4)
        self.t1, self.t2, self.t3, self.t4 = {}, {}, {}, {}
        self._atlas_cancer_types = set()

        with driver.session() as s:
            # --- Atlas: Gene PROGNOSTIC_IN CancerType ---
            result = s.run("""
                MATCH (g:Gene)-[r:PROGNOSTIC_IN]->(ct:CancerType)
                RETURN g.name AS gene, ct.name AS cancer_type,
                       r.tier AS tier, r.hr AS hr,
                       r.ci_width AS ci_width, r.n_with AS n_with,
                       r.channel AS channel, r.protein_change AS protein_change,
                       r.confidence AS confidence
            """)
            for rec in result:
                entry = {
                    "hr": rec["hr"],
                    "ci_width": rec["ci_width"] or 1.0,
                    "tier": rec["tier"],
                    "n_with": rec["n_with"] or 0,
                }
                tier = rec["tier"]
                ct = rec["cancer_type"]
                gene = rec["gene"]
                self._atlas_cancer_types.add(ct)

                if tier == 1:
                    self.t1[(ct, gene, rec["protein_change"] or "")] = entry
                elif tier == 2:
                    self.t2[(ct, gene)] = entry
                elif tier == 4:
                    entry["confidence"] = rec["confidence"] or 0.0
                    self.t4[(ct, gene)] = entry

            # Atlas tier 3: Channel PROGNOSTIC_IN CancerType
            result = s.run("""
                MATCH (ch:Channel)-[r:PROGNOSTIC_IN]->(ct:CancerType)
                WHERE r.tier = 3
                RETURN ch.name AS channel, ct.name AS cancer_type,
                       r.hr AS hr, r.ci_width AS ci_width, r.n_with AS n_with
            """)
            for rec in result:
                self.t3[(rec["cancer_type"], rec["channel"])] = {
                    "hr": rec["hr"],
                    "ci_width": rec["ci_width"] or 1.0,
                    "tier": 3,
                    "n_with": rec["n_with"] or 0,
                }
                self._atlas_cancer_types.add(rec["cancer_type"])

            # --- Expression: Gene EXPRESSION_IN CancerType ---
            self.expr_lookup = {}
            result = s.run("""
                MATCH (g:Gene)-[r:EXPRESSION_IN]->(ct:CancerType)
                RETURN g.name AS gene, ct.name AS ct, r.z_score AS z
            """)
            for rec in result:
                if rec["z"] is not None:
                    self.expr_lookup[(rec["gene"], rec["ct"])] = float(rec["z"])

            # --- CNA: Gene CNA_IN CancerType ---
            self.cna_lookup = {}
            result = s.run("""
                MATCH (g:Gene)-[r:CNA_IN]->(ct:CancerType)
                RETURN g.name AS gene, ct.name AS ct,
                       r.amp_freq AS amp, r.del_freq AS del, r.mean_cna AS mean_cna
            """)
            for rec in result:
                self.cna_lookup[(rec["gene"], rec["ct"])] = {
                    "mean_cna": float(rec["mean_cna"] or 0),
                    "amp_freq": float(rec["amp"] or 0),
                    "del_freq": float(rec["del"] or 0),
                }

            # --- DepMap: Gene ESSENTIAL_IN Lineage ---
            self.depmap_lookup = {}
            result = s.run("""
                MATCH (g:Gene)-[r:ESSENTIAL_IN]->(l:Lineage)
                RETURN g.name AS gene, l.name AS lineage, r.dependency_score AS score
            """)
            for rec in result:
                if rec["score"] is not None:
                    self.depmap_lookup[(rec["gene"], rec["lineage"])] = float(rec["score"])

            # --- SL partners: Gene SL_PARTNER Gene ---
            self.sl_lookup = {}
            result = s.run("""
                MATCH (g1:Gene)-[:SL_PARTNER]-(g2:Gene)
                RETURN g1.name AS a, g2.name AS b
            """)
            for rec in result:
                self.sl_lookup.setdefault(rec["a"], set()).add(rec["b"])

            # --- CIViC sensitivity/resistance ---
            self.civic_sens = {}
            self.civic_res = {}
            self.civic_func = {}
            result = s.run("""
                MATCH (g:Gene)-[r:HAS_SENSITIVITY_EVIDENCE]->(t)
                RETURN g.name AS gene, r.protein_change AS pc
            """)
            for rec in result:
                self.civic_sens[(rec["gene"], rec["pc"] or "")] = True
            result = s.run("""
                MATCH (g:Gene)-[r:HAS_RESISTANCE_EVIDENCE]->(t)
                RETURN g.name AS gene, r.protein_change AS pc
            """)
            for rec in result:
                self.civic_res[(rec["gene"], rec["pc"] or "")] = True

        driver.close()

        # CIViC GOF/LOF from CSV (not yet edges — will be migrated)
        path = os.path.join(CIVIC_CACHE, "variant_functional_map.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                gene, pc = row["gene"], row["protein_change"]
                effects = str(row.get("oncogenic_effects", ""))
                directions = str(row.get("evidence_directions", ""))
                if "Sensitivity" in effects and "Supports" in directions:
                    self.civic_sens[(gene, pc)] = True
                if "Resistance" in effects and "Supports" in directions:
                    self.civic_res[(gene, pc)] = True
                inf = row.get("inferred_function", "")
                if isinstance(inf, str) and inf in ("GOF", "LOF"):
                    self.civic_func[(gene, pc)] = inf

        # Fallback: if Neo4j had sparse data, supplement from CSVs
        if len(self.expr_lookup) < 100:
            csv_expr = _load_expression_lookup()
            self.expr_lookup.update(csv_expr)
        if len(self.cna_lookup) < 50:
            csv_cna = _load_cna_lookup()
            self.cna_lookup.update(csv_cna)
        if len(self.depmap_lookup) < 100:
            csv_dep = _load_depmap_lookup()
            self.depmap_lookup.update(csv_dep)
        if len(self.sl_lookup) < 10:
            csv_sl = _load_sl_lookup()
            self.sl_lookup.update(csv_sl)

    def _load_atlas_from_neo4j(self):
        """Load all atlas entries from Neo4j PROGNOSTIC_IN edges.

        Returns (t1, t2, t3, t4) tier lookup dicts.
        All data comes from the graph — no CSV loading.
        """
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "openknowledgegraph"),
        )
        t1, t2, t3, t4 = {}, {}, {}, {}

        with driver.session() as s:
            result = s.run("""
                MATCH (g:Gene)-[r:PROGNOSTIC_IN]->(ct:CancerType)
                RETURN g.name AS gene, ct.name AS cancer_type,
                       r.tier AS tier, r.hr AS hr,
                       r.ci_width AS ci_width, r.n_with AS n_with,
                       r.channel AS channel, r.protein_change AS protein_change,
                       r.confidence AS confidence, r.source AS source
            """)
            for rec in result:
                entry = {
                    "hr": rec["hr"],
                    "ci_width": rec["ci_width"] or 1.0,
                    "tier": rec["tier"],
                    "n_with": rec["n_with"] or 0,
                }
                tier = rec["tier"]
                ct = rec["cancer_type"]
                gene = rec["gene"]

                if tier == 1:
                    pc = rec["protein_change"] or ""
                    t1[(ct, gene, pc)] = entry
                elif tier == 2:
                    t2[(ct, gene)] = entry
                elif tier == 3:
                    ch = rec["channel"] or ""
                    t3[(ct, ch)] = entry
                elif tier == 4:
                    # Graph-imputed: store like tier 2 but in separate dict
                    entry["confidence"] = rec["confidence"] or 0.0
                    t4[(ct, gene)] = entry

            # Load tier 3: Channel -[PROGNOSTIC_IN]-> CancerType
            result = s.run("""
                MATCH (ch:Channel)-[r:PROGNOSTIC_IN]->(ct:CancerType)
                WHERE r.tier = 3
                RETURN ch.name AS channel, ct.name AS cancer_type,
                       r.hr AS hr, r.ci_width AS ci_width,
                       r.n_with AS n_with
            """)
            for rec in result:
                entry = {
                    "hr": rec["hr"],
                    "ci_width": rec["ci_width"] or 1.0,
                    "tier": 3,
                    "n_with": rec["n_with"] or 0,
                }
                t3[(rec["cancer_type"], rec["channel"])] = entry

            # Cache the cancer types present in the atlas
            self._atlas_cancer_types = set()
            result = s.run("""
                MATCH ()-[:PROGNOSTIC_IN]->(ct:CancerType)
                RETURN DISTINCT ct.name AS name
            """)
            for rec in result:
                self._atlas_cancer_types.add(rec["name"])

        driver.close()
        return t1, t2, t3, t4

    def _get_atlas_cancer_type(self, tcga_cancer):
        """Map TCGA abbreviation to atlas cancer_type string.

        The atlas uses full names like 'Breast Cancer', 'Non-Small Cell Lung Cancer'.
        We need to check what cancer types exist in the atlas and find the best match.
        """
        if not hasattr(self, "_atlas_ct_map"):
            atlas_cts = getattr(self, "_atlas_cancer_types", set())
            self._atlas_ct_map = {}
            abbrev_to_name = {
                "ACC": "Adrenocortical Carcinoma",
                "BLCA": "Bladder Cancer",
                "BRCA": "Breast Cancer",
                "CESC": "Cervical Cancer",
                "CHOL": "Cholangiocarcinoma",
                "COADREAD": "Colorectal Cancer",
                "DLBC": "Diffuse Large B-Cell Lymphoma",
                "ESCA": "Esophagogastric Cancer",
                "GBM": "Glioblastoma",
                "HNSC": "Head and Neck Cancer",
                "KICH": "Renal Cell Carcinoma",
                "KIRC": "Renal Cell Carcinoma",
                "KIRP": "Renal Cell Carcinoma",
                "LAML": "Acute Myeloid Leukemia",
                "LGG": "Low-Grade Glioma",
                "LIHC": "Hepatobiliary Cancer",
                "LUAD": "Non-Small Cell Lung Cancer",
                "LUSC": "Non-Small Cell Lung Cancer",
                "MESO": "Mesothelioma",
                "OV": "Ovarian Cancer",
                "PAAD": "Pancreatic Cancer",
                "PCPG": "Pheochromocytoma",
                "PRAD": "Prostate Cancer",
                "SARC": "Soft Tissue Sarcoma",
                "SKCM": "Melanoma",
                "STAD": "Esophagogastric Cancer",
                "TGCT": "Germ Cell Tumor",
                "THCA": "Thyroid Cancer",
                "THYM": "Thymic Epithelial Tumor",
                "UCEC": "Endometrial Cancer",
                "UCS": "Uterine Sarcoma",
                "UVM": "Uveal Melanoma",
            }
            for abbrev, name in abbrev_to_name.items():
                if name in atlas_cts:
                    self._atlas_ct_map[abbrev] = name
                else:
                    for act in atlas_cts:
                        if name.split()[0].lower() in act.lower():
                            self._atlas_ct_map[abbrev] = act
                            break

        return self._atlas_ct_map.get(tcga_cancer)

    def _make_enriched_node(self, gene, pc, mt, hr, ci_width, tier, n_with,
                            cancer_type, patient_genes_set, treatment_vec=None):
        """Create 26-dim enriched node feature vector.

        Dims 0-21: base features (atlas + enrichment)
        Dims 22-25: per-node systemic treatment flags
            [22] chemotherapy, [23] endocrine, [24] targeted, [25] immunotherapy
        """
        feat = np.zeros(NODE_FEAT_DIM, dtype=np.float32)

        # Original 14 features
        feat[0] = np.log(max(hr, 0.01))
        feat[1] = min(ci_width, 3.0) / 3.0
        feat[2] = tier / 3.0
        feat[3] = 1.0 if gene in _HUB_SET else 0.0

        ch = CHANNEL_MAP.get(gene)
        if ch and ch in CHANNEL_NAMES:
            feat[4 + CHANNEL_NAMES.index(ch)] = 1.0

        feat[10] = 1.0 if hr > 1.1 else 0.0
        feat[11] = 1.0 if hr < 0.9 else 0.0
        feat[12] = np.log(max(n_with, 1)) / 10.0

        pos = parse_position(pc)
        plen = PROTEIN_LENGTHS.get(gene)
        feat[13] = pos / plen if (pos and plen) else 0.5

        # --- New 8 features ---

        # [14] expression_z
        feat[14] = self.expr_lookup.get((gene, cancer_type), 0.0)

        # [15] cna_score
        cna_entry = self.cna_lookup.get((gene, cancer_type), {})
        feat[15] = cna_entry.get("mean_cna", 0.0)

        # [16] depmap_dependency
        lineage = TCGA_TO_DEPMAP_LINEAGE.get(cancer_type, "")
        feat[16] = self.depmap_lookup.get((gene, lineage), 0.0)

        # [17] has_sl_partner_mutated
        sl_partners = self.sl_lookup.get(gene, set())
        feat[17] = 1.0 if len(sl_partners & patient_genes_set) > 0 else 0.0

        # [18] civic_sensitivity
        feat[18] = 1.0 if self.civic_sens.get((gene, pc), False) else 0.0

        # [19] civic_resistance
        feat[19] = 1.0 if self.civic_res.get((gene, pc), False) else 0.0

        # [20] gof_lof: +1 GOF, -1 LOF, 0 context/unknown
        # Check CIViC inferred_function first, then config GENE_FUNCTION
        civic_func = self.civic_func.get((gene, pc))
        if civic_func == "GOF":
            feat[20] = 1.0
        elif civic_func == "LOF":
            feat[20] = -1.0
        else:
            gf = GENE_FUNCTION.get(gene, "context")
            if gf == "GOF":
                feat[20] = 1.0
            elif gf == "LOF":
                # If truncating mutation, always LOF
                feat[20] = -1.0
            elif gf == "context":
                # Truncating -> LOF, missense -> 0 (ambiguous)
                if mt in TRUNCATING:
                    feat[20] = -1.0
                else:
                    feat[20] = 0.0

        # [21] cna_context: +1 if amp_freq > 0.05, -1 if del_freq > 0.05
        amp_freq = cna_entry.get("amp_freq", 0.0)
        del_freq = cna_entry.get("del_freq", 0.0)
        if amp_freq > 0.05:
            feat[21] = 1.0
        elif del_freq > 0.05:
            feat[21] = -1.0
        else:
            feat[21] = 0.0

        # [22-25] Per-node systemic treatment flags
        # Enables transformer to learn channel × treatment interactions
        # (e.g., DDR-channel gene + platinum chemo = sensitivity signal)
        if treatment_vec is not None:
            feat[22] = treatment_vec[2]   # chemotherapy
            feat[23] = treatment_vec[3]   # endocrine
            feat[24] = treatment_vec[4]   # targeted
            feat[25] = treatment_vec[5]   # immunotherapy

        return feat

    def _build_splits(self, clinical_df, treatment_vecs):
        """Build train/val/holdback splits with stratification.

        Returns dict with 'holdback', 'val', 'train' index arrays.
        """
        from sklearn.model_selection import StratifiedShuffleSplit

        splits_path = os.path.join(CACHE_DIR, "splits.json")
        if os.path.exists(splits_path):
            print(f"Loading existing splits from {splits_path}", flush=True)
            with open(splits_path) as f:
                splits = json.load(f)
            # Convert to numpy arrays
            return {k: np.array(v) for k, v in splits.items()}

        print("Building stratified splits...", flush=True)

        # Build composite stratum
        strata = []
        for i, row in clinical_df.iterrows():
            ct = row["cancer_type"]
            event = int(row["event"])
            # Summarize treatment arm
            vec = treatment_vecs.get(row["case_id"], np.zeros(11))
            arm_parts = []
            if vec[0] > 0: arm_parts.append("S")
            if vec[1] > 0: arm_parts.append("R")
            if vec[2] > 0: arm_parts.append("C")
            if vec[3] > 0: arm_parts.append("E")
            if vec[4] > 0: arm_parts.append("T")
            if vec[5] > 0: arm_parts.append("I")
            arm = "".join(arm_parts) if arm_parts else "none"
            strata.append(f"{ct}_{arm}_{event}")

        strata = np.array(strata)

        # Collapse rare strata (< 3 samples) to avoid split errors
        from collections import Counter
        counts = Counter(strata)
        strata_collapsed = np.array([
            s if counts[s] >= 3 else f"{s.rsplit('_', 1)[0]}_other"
            for s in strata
        ])
        # If still too rare, collapse further
        counts2 = Counter(strata_collapsed)
        strata_final = np.array([
            s if counts2[s] >= 3 else "rare_stratum"
            for s in strata_collapsed
        ])

        n = len(clinical_df)
        indices = np.arange(n)

        # First split: 15% holdback
        try:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            rest_idx, holdback_idx = next(sss1.split(indices, strata_final))
        except ValueError:
            # Fallback: random split if stratification fails
            print("  WARNING: Stratification failed, using random split", flush=True)
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            holdback_n = int(n * 0.15)
            holdback_idx = perm[:holdback_n]
            rest_idx = perm[holdback_n:]

        # Second split: 15% val from remaining (= 15/85 ~ 17.6% of rest)
        rest_strata = strata_final[rest_idx]
        counts3 = Counter(rest_strata)
        rest_strata_safe = np.array([
            s if counts3[s] >= 2 else "rare_stratum"
            for s in rest_strata
        ])

        try:
            val_frac = 0.15 / 0.85
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
            train_sub, val_sub = next(sss2.split(rest_idx, rest_strata_safe))
            val_idx = rest_idx[val_sub]
            train_idx = rest_idx[train_sub]
        except ValueError:
            print("  WARNING: Val stratification failed, using random split", flush=True)
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(rest_idx))
            val_n = int(len(rest_idx) * val_frac)
            val_idx = rest_idx[perm[:val_n]]
            train_idx = rest_idx[perm[val_n:]]

        splits = {
            "holdback": holdback_idx.tolist(),
            "val": val_idx.tolist(),
            "train": train_idx.tolist(),
        }

        # Save holdback patient IDs
        holdback_ids = clinical_df.iloc[holdback_idx]["case_id"].tolist()
        holdback_path = os.path.join(CACHE_DIR, "holdback_ids.json")
        if not os.path.exists(holdback_path):
            with open(holdback_path, "w") as f:
                json.dump(holdback_ids, f, indent=2)
            print(f"  Saved holdback IDs to {holdback_path}", flush=True)

        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"  Saved splits to {splits_path}", flush=True)
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Holdback: {len(holdback_idx)}", flush=True)

        return {k: np.array(v) for k, v in splits.items()}

    def _build_metabric_patients(self):
        """Build patient records from METABRIC data.

        Returns list of dicts with: case_id, cancer_type, time, event,
        treatment_vec, age_z, sex, mutations [(gene, pc, mt), ...]
        """
        if self.metabric_clinical is None:
            return []

        clin = self.metabric_clinical
        muts = self.metabric_mutations

        # Build per-patient mutation lists
        patient_muts = {}
        for _, row in muts.iterrows():
            pid = row["patient_id"]
            gene = row["gene"]
            pc = row.get("protein_change", "")
            mt = row.get("mutation_type", "Missense_Mutation")
            patient_muts.setdefault(pid, []).append((gene, str(pc), str(mt)))

        patients = []
        for _, row in clin.iterrows():
            pid = row["PATIENT_ID"]

            # Parse survival
            os_status = str(row.get("OS_STATUS", ""))
            event = 1 if os_status.startswith("1") else 0
            os_months = row.get("OS_MONTHS")
            if pd.isna(os_months) or os_months <= 0:
                continue
            time_days = float(os_months) * 30.44  # months to days

            # Treatment vector (indices 2-5 only, surgery/radiation zeroed)
            tvec = np.zeros(11, dtype=np.float32)
            if str(row.get("CHEMOTHERAPY", "")).upper() == "YES":
                tvec[2] = 1.0
            if str(row.get("HORMONE_THERAPY", "")).upper() == "YES":
                tvec[3] = 1.0  # endocrine
            # METABRIC doesn't distinguish targeted/immuno, leave as 0

            # Age
            age = row.get("AGE_AT_DIAGNOSIS", 60.0)
            age_z = (float(age) - 60.0) / 15.0 if pd.notna(age) else 0.0

            # Sex (all female in METABRIC)
            sex = 1.0

            patient_muts_list = patient_muts.get(pid, [])

            patients.append({
                "case_id": f"METABRIC_{pid}",
                "cancer_type": "BRCA",
                "time": time_days,
                "event": event,
                "treatment_vec": tvec,
                "age_z": age_z,
                "sex": sex,
                "mutations": patient_muts_list,
            })

        print(f"  METABRIC patients with survival: {len(patients)}", flush=True)
        return patients

    def build_features(self):
        """Build all tensors for the treatment-conditional survival model."""
        print("\nBuilding features...", flush=True)

        # 1. Build patient clinical table
        clinical = self._build_patient_table()

        # 2. Build treatment vectors
        treatment_vecs = self._build_treatment_vectors()

        # 3. Build mutation index from TCGA mutations
        # Need to link case_id (GDC UUID) to cBioPortal patient_id (TCGA barcode).
        # The mutations file uses TCGA barcodes (e.g. TCGA-A1-A0SK),
        # and the treatment file uses GDC case UUIDs.
        # We need to match via cancer_type + find a mapping.
        # The cBioPortal patient_id is the TCGA barcode; GDC case_id is a UUID.
        # We cannot directly join them. Instead, we build patient data from
        # cBioPortal only, using the mutation patient_ids.

        # Build per-(cbioportal patient_id) mutation lists
        patient_muts, patient_cancer_from_muts = self._build_mutation_index()

        # We need to reconcile GDC case_ids with cBioPortal patient_ids.
        # Strategy: for each GDC case_id in clinical, we match to cBioPortal
        # patients by sample_id overlap. But TCGA barcodes in cBioPortal
        # look like "TCGA-A1-A0SK" while GDC uses UUIDs.
        #
        # Alternative: build our patient table from the mutation data directly,
        # joining with treatment data via a barcode -> case_id mapping.
        # Since the treatment file has case_id (UUID) and project_id,
        # and the mutation file has patient_id (barcode) and cancer_type,
        # we need to fetch the mapping.
        #
        # Pragmatic approach: build a sample_id -> case_id mapping from the
        # treatment file's project_id and the mutation file's sample_id.
        # TCGA barcodes have structure: TCGA-{TSS}-{participant}.
        # We can try to match via the GDC API, but that is slow.
        #
        # Best approach: use the mutation data's patient_ids as the primary
        # key, and join treatment data by building a barcode -> case_id map
        # from the GDC. If this mapping file does not exist, create it.

        barcode_map = self._get_barcode_to_caseid_map()

        # Now build the combined patient list: patients that have both
        # treatment data AND are in our mutation data
        # (patients without mutations still get wild-type nodes)

        # Set of case_ids with treatment data
        treatment_case_ids = set(clinical["case_id"].unique())

        # Map cBioPortal patient_ids to case_ids
        mut_patient_ids = set(patient_muts.keys()) | set(patient_cancer_from_muts.keys())

        # Find overlap
        matched_patients = []  # list of (case_id, patient_id, cancer_type)
        unmatched_count = 0

        # clinical is indexed by case_id; try to map each to a cBioPortal patient_id
        for _, row in clinical.iterrows():
            case_id = row["case_id"]
            cancer = row["cancer_type"]

            # Try to find matching barcode
            barcode = barcode_map.get(case_id)
            if barcode and barcode in mut_patient_ids:
                matched_patients.append((case_id, barcode, cancer))
            elif barcode:
                # Patient has treatment data but no mutations in our gene set
                matched_patients.append((case_id, barcode, cancer))
            else:
                unmatched_count += 1

        # Also include patients who have a barcode mapped from the reverse direction
        caseid_to_barcode = barcode_map
        barcode_to_caseid = {v: k for k, v in barcode_map.items()}
        for pid in mut_patient_ids:
            cid = barcode_to_caseid.get(pid)
            if cid and cid in treatment_case_ids:
                # Already handled above
                pass

        print(f"Matched patients (treatment + mutations): {len(matched_patients)}", flush=True)
        if unmatched_count > 0:
            print(f"  Unmatched (no barcode mapping): {unmatched_count}", flush=True)

        if len(matched_patients) == 0:
            print("WARNING: No patients matched. Falling back to mutation-only patients.", flush=True)
            # Use mutation patients directly, with empty treatment vectors
            matched_patients = []
            for pid, cancer in patient_cancer_from_muts.items():
                matched_patients.append((pid, pid, cancer))

        # 4. Build cancer type mapping
        cancer_type_map = {}
        ct_idx = 0

        # 5. Assemble per-patient tensors
        print("Assembling per-patient tensors...", flush=True)

        # Build a clinical info lookup for case_ids
        clinical_lookup = {}
        for _, row in clinical.iterrows():
            clinical_lookup[row["case_id"]] = row

        all_node_feats = []
        all_node_masks = []
        all_channel_pos_ids = []
        all_atlas_sums = []
        all_times = []
        all_events = []
        all_cancer_types = []
        all_clinical = []  # (age_z, sex)
        all_treatment_vecs = []
        all_treatment_masks = []
        all_gene_names = []

        # For building the final clinical dataframe (for splits)
        patient_records = []

        for case_id, barcode, cancer in matched_patients:
            # Get clinical info
            clin_row = clinical_lookup.get(case_id)
            if clin_row is None:
                continue

            time_val = float(clin_row["time"])
            event_val = int(clin_row["event"])

            # Get atlas cancer type name
            atlas_ct = self._get_atlas_cancer_type(cancer)

            # Get treatment vector early (needed for per-node treatment features)
            tvec = treatment_vecs.get(case_id, np.zeros(11, dtype=np.float32))
            tvec[0] = 0.0  # surgery (confounder)
            tvec[1] = 0.0  # radiation (confounder)

            # Get patient mutations
            muts = patient_muts.get(barcode, [])
            patient_genes_set = set(g for g, _, _ in muts)

            # Match mutations to atlas tiers and build node features
            nodes = []
            cp_ids = []
            genes = []
            log_hrs = []

            for gene, pc, mt in muts:
                if gene not in CHANNEL_MAP:
                    continue

                ch = CHANNEL_MAP.get(gene)
                entry = None
                if atlas_ct:
                    if (atlas_ct, gene, pc) in self.t1:
                        entry = self.t1[(atlas_ct, gene, pc)]
                    elif (atlas_ct, gene) in self.t2:
                        entry = self.t2[(atlas_ct, gene)]
                    elif ch and (atlas_ct, ch) in self.t3:
                        entry = self.t3[(atlas_ct, ch)]
                    elif (atlas_ct, gene) in self.t4:
                        entry = self.t4[(atlas_ct, gene)]

                if entry is not None:
                    hr = entry["hr"]
                    ci_w = entry.get("ci_width", 1.0)
                    tier = entry["tier"]
                    n_w = entry.get("n_with", 50)
                else:
                    # Zero-confidence: neutral HR (1.0 = no effect),
                    # enriched features still carry signal
                    hr = 1.0
                    ci_w = 0.0
                    tier = 0
                    n_w = 0

                feat = self._make_enriched_node(
                    gene, pc, mt, hr, ci_w, tier, n_w,
                    cancer, patient_genes_set, tvec,
                )
                nodes.append(feat)
                cp_ids.append(get_channel_pos_id(gene))
                genes.append(gene)
                log_hrs.append(np.log(max(hr, 0.01)))

            if len(nodes) == 0:
                # Wild-type node
                nodes = [np.zeros(NODE_FEAT_DIM, dtype=np.float32)]
                cp_ids = [0]
                genes = ["WT"]
                atlas_sum = 0.0
            else:
                atlas_sum = float(sum(log_hrs))

            n_nodes = len(nodes)

            # Truncate if needed (keep highest |log_hr|)
            if n_nodes > MAX_NODES:
                abs_scores = np.abs([n[0] for n in nodes])
                top_idx = np.argsort(abs_scores)[-MAX_NODES:]
                nodes = [nodes[i] for i in top_idx]
                cp_ids = [cp_ids[i] for i in top_idx]
                genes = [genes[i] for i in top_idx]
                n_nodes = MAX_NODES

            mask = [1] * n_nodes

            # Pad to MAX_NODES
            while len(nodes) < MAX_NODES:
                nodes.append(np.zeros(NODE_FEAT_DIM, dtype=np.float32))
                cp_ids.append(0)
                genes.append("")
                mask.append(0)

            all_node_feats.append(np.stack(nodes))
            all_node_masks.append(mask)
            all_channel_pos_ids.append(cp_ids)
            all_atlas_sums.append(atlas_sum)
            all_gene_names.append(genes)

            all_times.append(time_val)
            all_events.append(event_val)

            if cancer not in cancer_type_map:
                cancer_type_map[cancer] = ct_idx
                ct_idx += 1
            all_cancer_types.append(cancer_type_map[cancer])

            # Clinical: age and sex (from treatment data, not available per-patient)
            # TCGA treatment data does not have age/sex. Use defaults.
            all_clinical.append([0.0, 0.0])  # placeholder age_z, sex

            # tvec already built above (with surgery/radiation zeroed)
            all_treatment_vecs.append(tvec)
            all_treatment_masks.append(np.ones(11, dtype=np.float32))

            patient_records.append({
                "case_id": case_id,
                "cancer_type": cancer,
                "event": event_val,
            })

        n_tcga = len(all_node_feats)
        print(f"TCGA patients assembled: {n_tcga}", flush=True)

        # === Add METABRIC patients ===
        metabric_patients = self._build_metabric_patients()
        n_metabric = 0
        for mp in metabric_patients:
            cancer = mp["cancer_type"]
            tvec = mp["treatment_vec"]
            muts = mp["mutations"]
            atlas_ct = self._get_atlas_cancer_type(cancer)
            patient_genes_set = set(g for g, _, _ in muts)

            nodes = []
            cp_ids = []
            genes = []
            log_hrs = []

            for gene, pc, mt in muts:
                if gene not in CHANNEL_MAP:
                    continue
                ch = CHANNEL_MAP.get(gene)
                entry = None
                if atlas_ct:
                    if (atlas_ct, gene, pc) in self.t1:
                        entry = self.t1[(atlas_ct, gene, pc)]
                    elif (atlas_ct, gene) in self.t2:
                        entry = self.t2[(atlas_ct, gene)]
                    elif ch and (atlas_ct, ch) in self.t3:
                        entry = self.t3[(atlas_ct, ch)]
                    elif (atlas_ct, gene) in self.t4:
                        entry = self.t4[(atlas_ct, gene)]

                if entry is not None:
                    hr = entry["hr"]
                    ci_w = entry.get("ci_width", 1.0)
                    tier = entry["tier"]
                    n_w = entry.get("n_with", 50)
                else:
                    hr = 1.0
                    ci_w = 0.0
                    tier = 0
                    n_w = 0

                feat = self._make_enriched_node(
                    gene, pc, mt, hr, ci_w, tier, n_w,
                    cancer, patient_genes_set, tvec,
                )
                nodes.append(feat)
                cp_ids.append(get_channel_pos_id(gene))
                genes.append(gene)
                log_hrs.append(np.log(max(hr, 0.01)))

            if len(nodes) == 0:
                nodes = [np.zeros(NODE_FEAT_DIM, dtype=np.float32)]
                cp_ids = [0]
                genes = ["WT"]
                atlas_sum = 0.0
            else:
                atlas_sum = float(sum(log_hrs))

            n_nodes = len(nodes)
            if n_nodes > MAX_NODES:
                abs_scores = np.abs([n[0] for n in nodes])
                top_idx = np.argsort(abs_scores)[-MAX_NODES:]
                nodes = [nodes[i] for i in top_idx]
                cp_ids = [cp_ids[i] for i in top_idx]
                genes = [genes[i] for i in top_idx]
                n_nodes = MAX_NODES

            mask = [1] * n_nodes
            while len(nodes) < MAX_NODES:
                nodes.append(np.zeros(NODE_FEAT_DIM, dtype=np.float32))
                cp_ids.append(0)
                genes.append("")
                mask.append(0)

            all_node_feats.append(np.stack(nodes))
            all_node_masks.append(mask)
            all_channel_pos_ids.append(cp_ids)
            all_atlas_sums.append(atlas_sum)
            all_gene_names.append(genes)

            all_times.append(mp["time"])
            all_events.append(mp["event"])

            if cancer not in cancer_type_map:
                cancer_type_map[cancer] = ct_idx
                ct_idx += 1
            all_cancer_types.append(cancer_type_map[cancer])

            all_clinical.append([mp["age_z"], mp["sex"]])
            all_treatment_vecs.append(tvec)
            all_treatment_masks.append(np.ones(11, dtype=np.float32))

            patient_records.append({
                "case_id": mp["case_id"],
                "cancer_type": cancer,
                "event": mp["event"],
            })
            n_metabric += 1

        print(f"METABRIC patients assembled: {n_metabric}", flush=True)

        n_patients = len(all_node_feats)
        print(f"Total patients: {n_patients} (TCGA: {n_tcga}, METABRIC: {n_metabric})", flush=True)

        if n_patients == 0:
            raise ValueError("No patients assembled. Check data files.")

        # Convert to tensors
        node_feats = torch.tensor(np.stack(all_node_feats), dtype=torch.float32)
        node_masks = torch.tensor(all_node_masks, dtype=torch.float32)
        channel_pos_ids = torch.tensor(all_channel_pos_ids, dtype=torch.long)
        atlas_sums = torch.tensor(all_atlas_sums, dtype=torch.float32).unsqueeze(1)
        times = torch.tensor(all_times, dtype=torch.float32)
        events = torch.tensor(all_events, dtype=torch.long)
        cancer_types = torch.tensor(all_cancer_types, dtype=torch.long)
        clinical_tensor = torch.tensor(all_clinical, dtype=torch.float32)
        treatment_vec = torch.tensor(np.stack(all_treatment_vecs), dtype=torch.float32)
        treatment_known_mask = torch.tensor(
            np.stack(all_treatment_masks), dtype=torch.float32
        )

        # Build splits
        patient_df = pd.DataFrame(patient_records)
        splits = self._build_splits(patient_df, treatment_vecs)

        # Print stats
        nodes_per_patient = node_masks.sum(dim=1)
        print(f"\n{'='*60}", flush=True)
        print("Dataset Summary", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Node features shape: {node_feats.shape}", flush=True)
        print(f"Mean nodes/patient: {nodes_per_patient.mean():.1f}", flush=True)
        print(f"Max nodes/patient: {nodes_per_patient.max():.0f}", flush=True)
        print(f"Wild-type only patients: {(nodes_per_patient <= 1).sum().item()}", flush=True)
        print(f"Cancer types: {ct_idx}", flush=True)
        print(f"Events: {events.sum().item()} / {n_patients} ({100*events.float().mean():.1f}%)", flush=True)
        print(f"Median survival (days): {times.median():.0f}", flush=True)
        print(f"Treatment vec mean: {treatment_vec.mean(dim=0).numpy()}", flush=True)
        print(f"Split sizes — train: {len(splits['train'])}, val: {len(splits['val'])}, holdback: {len(splits['holdback'])}", flush=True)

        return {
            "node_features": node_feats,
            "node_masks": node_masks,
            "channel_pos_ids": channel_pos_ids,
            "cancer_types": cancer_types,
            "clinical": clinical_tensor,
            "atlas_sums": atlas_sums,
            "treatment_vec": treatment_vec,
            "treatment_known_mask": treatment_known_mask,
            "times": times,
            "events": events,
            "split_indices": splits,
            "n_cancer_types": ct_idx,
            "cancer_type_map": cancer_type_map,
            "gene_names": all_gene_names,
        }

    def _get_barcode_to_caseid_map(self):
        """Build case_id (GDC UUID) -> TCGA barcode mapping.

        Uses the GDC API to fetch the mapping, caches to disk.
        """
        cache_path = os.path.join(CACHE_DIR, "caseid_barcode_map.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        print("Fetching case_id -> barcode mapping from GDC API...", flush=True)

        mapping = {}
        size = 1000
        offset = 0

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        while True:
            filters = json.dumps({
                "op": "=",
                "content": {
                    "field": "project.program.name",
                    "value": "TCGA",
                },
            })
            fields = "case_id,submitter_id"
            params = urllib.parse.urlencode({
                "filters": filters,
                "fields": fields,
                "from": offset,
                "size": size,
                "format": "JSON",
            })
            url = f"https://api.gdc.cancer.gov/cases?{params}"

            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                print(f"  GDC API error at offset {offset}: {e}", flush=True)
                break

            hits = data.get("data", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                case_id = hit.get("case_id", "")
                submitter_id = hit.get("submitter_id", "")
                if case_id and submitter_id:
                    mapping[case_id] = submitter_id

            total = data.get("data", {}).get("pagination", {}).get("total", 0)
            offset += size
            print(f"  Fetched {min(offset, total)}/{total} cases", flush=True)

            if offset >= total:
                break
            time.sleep(0.2)

        print(f"Mapped {len(mapping)} case_ids to barcodes", flush=True)

        with open(cache_path, "w") as f:
            json.dump(mapping, f)

        return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = TreatmentDataset()
    data = ds.build_features()

    print("\n" + "=" * 60, flush=True)
    print("Tensor shapes:", flush=True)
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape} ({val.dtype})", flush=True)
        elif isinstance(val, dict) and key == "split_indices":
            for sk, sv in val.items():
                print(f"  split_indices[{sk}]: {len(sv)}", flush=True)
        elif isinstance(val, dict):
            print(f"  {key}: dict with {len(val)} entries", flush=True)
        elif isinstance(val, list):
            print(f"  {key}: list of {len(val)}", flush=True)
        else:
            print(f"  {key}: {val}", flush=True)

    print("\nDone.", flush=True)
