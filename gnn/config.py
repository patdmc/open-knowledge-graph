"""
Centralized configuration for the Coupling-Channel GNN.

Paths, mutation types, dataset paths, and hyperparameters live here.
Gene-channel mappings, hub/leaf, GOF/LOF are loaded from the graph —
the graph is the source of truth for all biological knowledge.
"""

import os
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_CACHE = os.path.join(ROOT, "analysis", "cache")
GNN_CACHE = os.path.join(ROOT, "gnn", "data", "cache")
GNN_RESULTS = os.path.join(ROOT, "gnn", "results")

os.makedirs(GNN_CACHE, exist_ok=True)
os.makedirs(GNN_RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Graph-derived gene/channel data (loaded from Neo4j, cached to disk)
# ---------------------------------------------------------------------------

_GRAPH_DATA_CACHE = os.path.join(GNN_CACHE, "graph_derived_config.json")


def _load_from_graph():
    """Load gene-channel mappings, hub/leaf, GOF/LOF from Neo4j."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    channel_map = {}
    channel_names_set = set()
    gene_function = {}
    hub_genes = {}

    with driver.session() as s:
        # Gene → channel from graph
        result = s.run("""
            MATCH (g:Gene)
            WHERE g.channel IS NOT NULL
            RETURN g.name AS name, g.channel AS channel, properties(g) AS props
        """)
        for r in result:
            name = r['name']
            ch = r['channel']
            props = dict(r['props'])

            channel_map[name] = ch
            channel_names_set.add(ch)

            # GOF/LOF from CIViC data in graph
            func = props.get('civic_dominant_function')
            if func in ('GOF', 'LOF'):
                gene_function[name] = func

            # Hub detection: high connectivity in PPI
            # (will be refined below from PPI degree)

        # Hub genes: top 3 by PPI degree within each channel
        result = s.run("""
            MATCH (g:Gene)-[r:PPI]-(other:Gene)
            WHERE g.channel IS NOT NULL AND g.channel = other.channel
            RETURN g.name AS name, g.channel AS channel, count(r) AS degree
            ORDER BY g.channel, count(r) DESC
        """)
        ch_degrees = {}
        for r in result:
            ch = r['channel']
            if ch not in ch_degrees:
                ch_degrees[ch] = []
            ch_degrees[ch].append(r['name'])

        for ch, genes in ch_degrees.items():
            hub_genes[ch] = genes[:3]  # top 3 by PPI degree

        # Channels without PPI data: use DepMap essentiality as proxy
        for ch in channel_names_set:
            if ch not in hub_genes:
                hub_genes[ch] = []

    driver.close()

    # Stable channel ordering
    channel_names = sorted(channel_names_set)

    return {
        'channel_names': channel_names,
        'channel_map': channel_map,
        'gene_function': gene_function,
        'hub_genes': hub_genes,
    }


def _load_graph_config():
    """Load from cache or graph."""
    if os.path.exists(_GRAPH_DATA_CACHE):
        mtime = os.path.getmtime(_GRAPH_DATA_CACHE)
        import time
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:  # 24-hour cache
            with open(_GRAPH_DATA_CACHE, 'r') as f:
                return json.load(f)

    try:
        data = _load_from_graph()
        os.makedirs(os.path.dirname(_GRAPH_DATA_CACHE), exist_ok=True)
        with open(_GRAPH_DATA_CACHE, 'w') as f:
            json.dump(data, f)
        return data
    except Exception:
        # Fallback: if Neo4j is down, use cache even if stale
        if os.path.exists(_GRAPH_DATA_CACHE):
            with open(_GRAPH_DATA_CACHE, 'r') as f:
                return json.load(f)
        raise


_graph_config = _load_graph_config()

CHANNEL_NAMES = _graph_config['channel_names']
CHANNEL_MAP = _graph_config['channel_map']
GENE_FUNCTION = _graph_config.get('gene_function', {})

# Hub genes from graph PPI topology
HUB_GENES = {ch: set(genes) for ch, genes in _graph_config.get('hub_genes', {}).items()}

# Derived lookups
ALL_GENES = sorted(CHANNEL_MAP.keys())
GENE_TO_IDX = {g: i for i, g in enumerate(ALL_GENES)}
N_GENES = len(ALL_GENES)

CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
GENE_TO_CHANNEL_IDX = {g: CHANNEL_TO_IDX.get(ch, 0) for g, ch in CHANNEL_MAP.items()}

GENES_PER_CHANNEL = {}
for gene, ch in CHANNEL_MAP.items():
    GENES_PER_CHANNEL.setdefault(ch, []).append(gene)
for ch in GENES_PER_CHANNEL:
    GENES_PER_CHANNEL[ch].sort()

# Leaf genes = in channel but not hub
LEAF_GENES = {}
for ch in CHANNEL_NAMES:
    hubs = HUB_GENES.get(ch, set())
    ch_genes = set(GENES_PER_CHANNEL.get(ch, []))
    LEAF_GENES[ch] = ch_genes - hubs

# Flat lookup
GENE_POSITION = {}
for ch, genes in HUB_GENES.items():
    for g in genes:
        GENE_POSITION[g] = "hub"
for ch, genes in LEAF_GENES.items():
    for g in genes:
        if g not in GENE_POSITION:
            GENE_POSITION[g] = "leaf"

# ---------------------------------------------------------------------------
# Mutation types (not graph data — these are data format constants)
# ---------------------------------------------------------------------------

NON_SILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "Splice_Region",
    "Nonstop_Mutation", "Translation_Start_Site",
}

TRUNCATING = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "Nonstop_Mutation",
}

# ---------------------------------------------------------------------------
# MSK-IMPACT datasets
# ---------------------------------------------------------------------------

MSK_DATASETS = {
    "msk_impact_50k": {
        "mutations": os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        "clinical": os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"),
        "sample_clinical": os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_sample_clinical.csv"),
    },
    "msk_met_2021": {
        "mutations": os.path.join(ANALYSIS_CACHE, "msk_met_2021_mutations.csv"),
        "clinical": os.path.join(ANALYSIS_CACHE, "msk_met_2021_clinical.csv"),
        "sample_clinical": os.path.join(ANALYSIS_CACHE, "msk_met_2021_sample_clinical.csv"),
    },
    "msk_impact_2017": {
        "mutations": os.path.join(ANALYSIS_CACHE, "msk_impact_2017_mutations.csv"),
        "clinical": os.path.join(ANALYSIS_CACHE, "msk_impact_2017_clinical.csv"),
        "sample_clinical": os.path.join(ANALYSIS_CACHE, "msk_impact_2017_sample_clinical.csv"),
    },
}

# ---------------------------------------------------------------------------
# GNN hyperparameters
# ---------------------------------------------------------------------------

GNN_CONFIG = {
    "node_feat_dim": 18,
    "hidden_dim": 64,
    "num_gat_heads": 4,
    "num_channel_layers": 2,
    "cross_channel_heads": 4,
    "cross_channel_layers": 1,
    "readout_hidden": 32,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "patience": 20,
    "batch_size": 64,
    "n_folds": 5,
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# External API
# ---------------------------------------------------------------------------

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STRING_SPECIES = 9606
STRING_SCORE_THRESHOLD = 700
