"""
Graph-native precomputation — push computation into Neo4j, cache results.

Instead of pulling raw edges into Python and computing features in loops,
we precompute everything graph-native:

  1. All-pairs PPI shortest paths (Neo4j shortestPath)
  2. Gene centrality (PPI degree, path-based betweenness proxy)
  3. Cross-channel interaction matrix (Cypher aggregation)
  4. Community assignments (edges from Neo4j, Louvain in Python)
  5. Per-gene-pair edge summaries (all edge types, one query per type)

Everything caches to disk. The scorer does pure dict lookups — no Neo4j at scoring time.
"""

import os
import json
import time
import numpy as np
import networkx as nx
from collections import defaultdict


def _sigmoid(x, mid, k):
    """Sigmoid curve for non-linear edge weighting.

    Below mid: near-zero (noise). Above mid: near-one (signal).
    k controls steepness.
    """
    return 1.0 / (1.0 + np.exp(-k * (x - mid)))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver("bolt://localhost:7687",
                                auth=("neo4j", "openknowledgegraph"))


def _cache_path(name):
    return os.path.join(CACHE_DIR, f"precomputed_{name}.json")


def _load_cache(name, max_age_hours=24):
    path = _cache_path(name)
    if os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours < max_age_hours:
            with open(path, 'r') as f:
                return json.load(f)
    return None


def _save_cache(name, data):
    path = _cache_path(name)
    with open(path, 'w') as f:
        json.dump(data, f)


# -----------------------------------------------------------------------
# 1. All-pairs PPI shortest paths
# -----------------------------------------------------------------------

def precompute_ppi_distances(force=False):
    """Compute all-pairs shortest paths via Neo4j shortestPath.

    Returns dict: "gene1|gene2" → distance (int), where gene1 < gene2.
    Disconnected pairs are not included (caller should default to a large value).
    """
    cached = None if force else _load_cache("ppi_distances")
    if cached is not None:
        print(f"  PPI distances from cache: {len(cached)} pairs", flush=True)
        return cached

    print("  Computing all-pairs PPI shortest paths in Neo4j...", flush=True)
    t0 = time.time()
    driver = _neo4j_driver()

    with driver.session() as s:
        result = s.run("""
            MATCH (a:Gene), (b:Gene)
            WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
              AND a.name < b.name
            MATCH p = shortestPath((a)-[:PPI*..6]-(b))
            RETURN a.name AS g1, b.name AS g2, length(p) AS dist
        """)
        distances = {}
        for r in result:
            key = f"{r['g1']}|{r['g2']}"
            distances[key] = r['dist']

    driver.close()
    print(f"  PPI distances: {len(distances)} pairs in {time.time()-t0:.1f}s", flush=True)

    _save_cache("ppi_distances", distances)
    return distances


def get_ppi_distance(distances, gene_a, gene_b, default=10):
    """Look up PPI distance between two genes."""
    if gene_a == gene_b:
        return 0
    a, b = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
    return distances.get(f"{a}|{b}", default)


# -----------------------------------------------------------------------
# 2. Gene centrality metrics
# -----------------------------------------------------------------------

def precompute_gene_centrality(force=False):
    """Compute gene centrality from PPI topology.

    Returns dict: gene → {degree, betweenness_proxy, eigenvector, channel_degree}
    - degree: PPI degree within channel
    - betweenness_proxy: fraction of shortest paths through this gene
    - channel_degree: number of distinct channels this gene connects to via PPI
    """
    cached = None if force else _load_cache("gene_centrality")
    if cached is not None:
        print(f"  Gene centrality from cache: {len(cached)} genes", flush=True)
        return cached

    print("  Computing gene centrality from Neo4j...", flush=True)
    t0 = time.time()
    driver = _neo4j_driver()
    centrality = {}

    with driver.session() as s:
        # PPI degree + cross-channel connectivity
        result = s.run("""
            MATCH (g:Gene)
            WHERE g.channel IS NOT NULL
            OPTIONAL MATCH (g)-[:PPI]-(neighbor:Gene)
            WHERE neighbor.channel IS NOT NULL
            WITH g.name AS name, g.channel AS ch,
                 count(DISTINCT neighbor) AS degree,
                 count(DISTINCT neighbor.channel) AS n_channels
            RETURN name, ch, degree, n_channels
        """)
        for r in result:
            centrality[r['name']] = {
                'degree': r['degree'],
                'channel_degree': r['n_channels'],
                'channel': r['ch'],
            }

        # Betweenness proxy: for each gene, count how many other gene pairs
        # have a shortest path of length 2 through this gene
        # (full betweenness is expensive, this is a good proxy)
        result = s.run("""
            MATCH (a:Gene)-[:PPI]-(hub:Gene)-[:PPI]-(b:Gene)
            WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
              AND hub.channel IS NOT NULL
              AND a.name < b.name AND a <> hub AND b <> hub
            WITH hub.name AS name, count(*) AS path_count
            RETURN name, path_count
        """)
        for r in result:
            if r['name'] in centrality:
                centrality[r['name']]['betweenness_proxy'] = r['path_count']

        # Normalize betweenness
        max_btw = max((c.get('betweenness_proxy', 0) for c in centrality.values()), default=1)
        if max_btw > 0:
            for g in centrality:
                raw = centrality[g].get('betweenness_proxy', 0)
                centrality[g]['betweenness_proxy'] = raw
                centrality[g]['betweenness_norm'] = raw / max_btw

    driver.close()
    print(f"  Gene centrality: {len(centrality)} genes in {time.time()-t0:.1f}s", flush=True)

    _save_cache("gene_centrality", centrality)
    return centrality


# -----------------------------------------------------------------------
# 3. Cross-channel interaction matrix
# -----------------------------------------------------------------------

def precompute_cross_channel_matrix(force=False):
    """Compute cross-channel interaction strengths from all edge types.

    Returns dict with:
      - 'matrix': channel_a|channel_b → weight (float)
      - 'channel_names': sorted list of channels
      - 'edge_summary': channel_a|channel_b → {edge_type: count}
    """
    cached = None if force else _load_cache("cross_channel_matrix")
    if cached is not None:
        print(f"  Cross-channel matrix from cache", flush=True)
        return cached

    print("  Computing cross-channel interaction matrix...", flush=True)
    t0 = time.time()
    driver = _neo4j_driver()

    with driver.session() as s:
        # Get all channels
        result = s.run("""
            MATCH (g:Gene)
            WHERE g.channel IS NOT NULL
            RETURN DISTINCT g.channel AS ch ORDER BY ch
        """)
        channel_names = [r['ch'] for r in result]

        # Aggregate cross-channel edges by type with properties
        result = s.run("""
            MATCH (a:Gene)-[r]->(b:Gene)
            WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
              AND a.channel <> b.channel
              AND (r.deprecated IS NULL OR r.deprecated = false)
            WITH a.channel AS ch_a, b.channel AS ch_b, type(r) AS rtype,
                 count(*) AS cnt,
                 avg(CASE WHEN r.score IS NOT NULL THEN r.score ELSE null END) AS avg_score,
                 avg(CASE WHEN r.correlation IS NOT NULL THEN abs(r.correlation) ELSE null END) AS avg_corr,
                 avg(CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE null END) AS avg_weight,
                 sum(CASE WHEN r.count IS NOT NULL THEN r.count ELSE 0 END) AS total_count
            RETURN ch_a, ch_b, rtype, cnt, avg_score, avg_corr, avg_weight, total_count
        """)

        edge_summary = {}
        matrix = {}

        # Edge type weights for cross-channel scoring — calibrated linear
        EDGE_WEIGHTS = {
            'ATTENDS_TO': lambda cnt, avg_w, avg_corr, avg_score, total_count: (avg_w or 0) * 2 * cnt,
            'SYNTHETIC_LETHAL': lambda cnt, avg_w, avg_corr, avg_score, total_count: 0.5 * cnt,
            'PPI': lambda cnt, avg_w, avg_corr, avg_score, total_count: min((avg_score or 500) / 1000, 1) * 0.3 * cnt,
            'COOCCURS': lambda cnt, avg_w, avg_corr, avg_score, total_count: np.log1p(total_count) / 20.0,
            'COUPLES': lambda cnt, avg_w, avg_corr, avg_score, total_count: 0.3 * cnt,
            'CO_ESSENTIAL': lambda cnt, avg_w, avg_corr, avg_score, total_count: (avg_corr or 0) * 0.3 * cnt,
            'CO_EXPRESSED': lambda cnt, avg_w, avg_corr, avg_score, total_count: (avg_corr or 0) * 0.2 * cnt,
            'CO_CNA': lambda cnt, avg_w, avg_corr, avg_score, total_count: (avg_corr or 0) * 0.15 * cnt,
        }

        for r in result:
            ch_key = f"{r['ch_a']}|{r['ch_b']}"
            rtype = r['rtype']
            cnt = r['cnt']

            if ch_key not in edge_summary:
                edge_summary[ch_key] = {}
            edge_summary[ch_key][rtype] = cnt

            # Compute weight
            fn = EDGE_WEIGHTS.get(rtype)
            if fn:
                w = fn(cnt, r['avg_weight'], r['avg_corr'], r['avg_score'], r['total_count'])
            else:
                w = (r['avg_corr'] or 0) * 0.1 * cnt if r['avg_corr'] else 0.05 * cnt
            w = float(w)

            matrix[ch_key] = matrix.get(ch_key, 0) + w

        # Normalize by max
        max_w = max(matrix.values()) if matrix else 1
        if max_w > 0:
            for k in matrix:
                matrix[k] /= max_w

    driver.close()

    data = {
        'matrix': matrix,
        'channel_names': channel_names,
        'edge_summary': edge_summary,
    }
    print(f"  Cross-channel matrix: {len(matrix)} pairs in {time.time()-t0:.1f}s", flush=True)

    _save_cache("cross_channel_matrix", data)
    return data


# -----------------------------------------------------------------------
# 4. Community assignments (edges from Neo4j, Louvain in Python)
# -----------------------------------------------------------------------

def precompute_communities(force=False):
    """Detect sub-pathway communities per channel.

    Pulls PPI+COUPLES edges from Neo4j in batch, runs Louvain per channel.
    Returns dict: gene → {channel, community_idx, community_size}
    Plus: communities dict: channel → list of gene sets
    """
    cached = None if force else _load_cache("communities")
    if cached is not None:
        print(f"  Communities from cache: {len(cached['gene_community'])} genes", flush=True)
        return cached

    print("  Computing communities from graph topology...", flush=True)
    t0 = time.time()
    driver = _neo4j_driver()

    # Pull all intra-channel PPI + COUPLES edges in one query each
    channel_edges = defaultdict(list)  # channel → [(gene_a, gene_b, weight)]
    channel_genes = defaultdict(set)

    with driver.session() as s:
        # Get all genes with channels
        result = s.run("""
            MATCH (g:Gene) WHERE g.channel IS NOT NULL
            RETURN g.name AS name, g.channel AS ch
        """)
        for r in result:
            channel_genes[r['ch']].add(r['name'])

        # PPI edges within same channel
        result = s.run("""
            MATCH (a:Gene)-[r:PPI]-(b:Gene)
            WHERE a.channel IS NOT NULL AND a.channel = b.channel
              AND a.name < b.name
            RETURN a.channel AS ch, a.name AS g1, b.name AS g2,
                   r.score AS score
        """)
        for r in result:
            channel_edges[r['ch']].append(
                (r['g1'], r['g2'], float(r['score'] or 500)))

        # COUPLES edges within same channel
        result = s.run("""
            MATCH (a:Gene)-[r:COUPLES]-(b:Gene)
            WHERE a.channel IS NOT NULL AND a.channel = b.channel
              AND a.name < b.name
            RETURN a.channel AS ch, a.name AS g1, b.name AS g2
        """)
        for r in result:
            channel_edges[r['ch']].append((r['g1'], r['g2'], 200))

    driver.close()

    # Run Louvain per channel
    gene_community = {}
    communities = {}

    for ch in sorted(channel_genes.keys()):
        genes = channel_genes[ch]

        if len(genes) < 5:
            communities[ch] = [sorted(genes)]
            for g in genes:
                gene_community[g] = {'channel': ch, 'community_idx': 0,
                                     'community_size': len(genes)}
            continue

        G = nx.Graph()
        G.add_nodes_from(genes)
        for g1, g2, w in channel_edges.get(ch, []):
            if G.has_edge(g1, g2):
                G[g1][g2]['weight'] += w
            else:
                G.add_edge(g1, g2, weight=w)

        try:
            comms = list(nx.community.greedy_modularity_communities(G))
        except Exception:
            comms = [genes]

        # Split large communities (target ~20 genes per block)
        final_comms = []
        for comm in comms:
            comm = set(comm)
            if len(comm) <= 25:
                final_comms.append(sorted(comm))
            else:
                sub = G.subgraph(comm)
                sub_comps = list(nx.connected_components(sub))
                if len(sub_comps) > 1 and all(len(c) <= 25 for c in sub_comps):
                    final_comms.extend(sorted(c) for c in sub_comps)
                else:
                    sorted_nodes = sorted(comm,
                        key=lambda n: G.degree(n, weight='weight'), reverse=True)
                    mid = len(sorted_nodes) // 2
                    final_comms.append(sorted(sorted_nodes[:mid]))
                    final_comms.append(sorted(sorted_nodes[mid:]))

        communities[ch] = final_comms
        for ci, comm in enumerate(final_comms):
            for g in comm:
                gene_community[g] = {
                    'channel': ch,
                    'community_idx': ci,
                    'community_size': len(comm),
                }

        print(f"    {ch}: {len(genes)} genes → {len(final_comms)} blocks "
              f"(sizes: {[len(c) for c in final_comms[:6]]})", flush=True)

    data = {
        'gene_community': gene_community,
        'communities': communities,
    }
    print(f"  Communities: {len(gene_community)} genes in {time.time()-t0:.1f}s", flush=True)

    _save_cache("communities", data)
    return data


# -----------------------------------------------------------------------
# 5. Per-gene-pair edge summaries
# -----------------------------------------------------------------------

def precompute_pairwise_edges(force=False):
    """Precompute scored edge summaries for all gene pairs.

    Separates additive and multiplicative edge types:
    - Additive: statistical associations (COOCCURS, CO_EXPRESSED, CO_CNA, PPI proximity)
    - Multiplicative: functional interactions (SYNTHETIC_LETHAL, COUPLES, CO_ESSENTIAL, ATTENDS_TO)

    Returns dict: "gene1|gene2" → {
        'additive': float (additive pairwise score),
        'multiplier': float (multiplicative factor, ≥1.0),
        'score': float (total = additive, used for backward compat),
        'edge_types': list of edge type names present,
        'cooccur_count': int,
        'cooccur_by_ct': {cancer_type: count},
    }
    """
    cached = None if force else _load_cache("pairwise_edges")
    if cached is not None:
        print(f"  Pairwise edges from cache: {len(cached)} pairs", flush=True)
        return cached

    print("  Precomputing pairwise edge scores from Neo4j...", flush=True)
    t0 = time.time()
    driver = _neo4j_driver()

    # Additive edge types: statistical co-occurrence, proximity
    ADDITIVE_SCORE = {
        'COOCCURS': lambda p: np.log1p(float(p.get('count', 0))) / 20.0,
        'PPI': lambda p: min(float(p.get('score', 500)) / 1000.0, 1.0) * 0.3,
        'CO_EXPRESSED': lambda p: abs(float(p.get('correlation', 0))) * 0.2,
        'CO_CNA': lambda p: abs(float(p.get('correlation', 0))) * 0.15,
    }

    # Multiplicative edge types: functional interactions that amplify damage
    # These return a multiplier increment (added to base 1.0)
    MULTIPLICATIVE_SCORE = {
        'SYNTHETIC_LETHAL': lambda p: 0.5 if p.get('cross_channel') else 0.3,
        'COUPLES': lambda p: 0.2,
        'CO_ESSENTIAL': lambda p: abs(float(p.get('correlation', 0))) * 0.3,
        'ATTENDS_TO': lambda p: float(p.get('weight', 0)) * 0.5,
    }

    pairs = {}  # "g1|g2" → {additive, multiplier, score, edge_types, ...}

    with driver.session() as s:
        # Discover all edge types
        result = s.run("""
            MATCH (a:Gene)-[r]->(b:Gene)
            WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
              AND (r.deprecated IS NULL OR r.deprecated = false)
            WITH DISTINCT type(r) AS rtype
            RETURN rtype ORDER BY rtype
        """)
        edge_types = [r['rtype'] for r in result]

        # Load edges per type
        for etype in edge_types:
            result = s.run(f"""
                MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
                  AND (r.deprecated IS NULL OR r.deprecated = false)
                RETURN a.name AS g1, b.name AS g2, properties(r) AS props
            """)

            add_fn = ADDITIVE_SCORE.get(etype)
            mult_fn = MULTIPLICATIVE_SCORE.get(etype)
            n = 0
            for r in result:
                g1, g2 = r['g1'], r['g2']
                a, b = (g1, g2) if g1 < g2 else (g2, g1)
                key = f"{a}|{b}"

                if key not in pairs:
                    pairs[key] = {
                        'additive': 0.0,
                        'multiplier': 0.0,  # increment above 1.0
                        'score': 0.0,
                        'edge_types': [],
                        'cooccur_count': 0,
                        'cooccur_by_ct': {},
                    }

                entry = pairs[key]
                if etype not in entry['edge_types']:
                    entry['edge_types'].append(etype)

                props = dict(r['props'])
                for k in list(props.keys()):
                    if not isinstance(props[k], (int, float, str, bool, type(None))):
                        del props[k]

                try:
                    if add_fn:
                        val = add_fn(props)
                        entry['additive'] += val
                        entry['score'] += val
                    elif mult_fn:
                        val = mult_fn(props)
                        entry['multiplier'] += val
                        entry['score'] += val  # backward compat
                    else:
                        # Unknown edge type — additive with correlation
                        corr = props.get('correlation')
                        if corr is not None:
                            val = abs(float(corr)) * 0.1
                            entry['additive'] += val
                            entry['score'] += val
                except (TypeError, ValueError):
                    pass

                if etype == 'COOCCURS':
                    ct = props.get('cancer_type')
                    count = int(props.get('count', 0))
                    entry['cooccur_count'] += count
                    if ct:
                        entry['cooccur_by_ct'][ct] = \
                            entry['cooccur_by_ct'].get(ct, 0) + count

                n += 1
            print(f"    {etype}: {n:,} edges", flush=True)

    driver.close()

    # Convert numpy floats to plain floats for JSON
    for key in pairs:
        pairs[key]['score'] = float(pairs[key]['score'])
        pairs[key]['additive'] = float(pairs[key]['additive'])
        pairs[key]['multiplier'] = float(pairs[key]['multiplier'])

    print(f"  Pairwise edges: {len(pairs)} pairs in {time.time()-t0:.1f}s", flush=True)

    _save_cache("pairwise_edges", pairs)
    return pairs


# -----------------------------------------------------------------------
# 6. Gene properties (node-level features from graph)
# -----------------------------------------------------------------------

def precompute_gene_properties(force=False):
    """Load all gene node properties from Neo4j.

    Returns dict: gene → {all numeric and categorical properties}
    """
    cached = None if force else _load_cache("gene_properties")
    if cached is not None:
        print(f"  Gene properties from cache: {len(cached)} genes", flush=True)
        return cached

    print("  Loading gene properties from Neo4j...", flush=True)
    t0 = time.time()
    driver = _neo4j_driver()

    gene_props = {}
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)
            WHERE g.channel IS NOT NULL
            RETURN g.name AS name, properties(g) AS props
        """)
        for r in result:
            props = dict(r['props'])
            # Keep only serializable values
            clean = {}
            for k, v in props.items():
                if isinstance(v, (int, float, str, bool)):
                    clean[k] = v
                elif v is None:
                    clean[k] = None
            gene_props[r['name']] = clean

    driver.close()
    print(f"  Gene properties: {len(gene_props)} genes in {time.time()-t0:.1f}s", flush=True)

    _save_cache("gene_properties", gene_props)
    return gene_props


# -----------------------------------------------------------------------
# Master precompute
# -----------------------------------------------------------------------

class GraphPrecomputed:
    """All precomputed graph data in one object. Pure dict lookups at scoring time."""

    def __init__(self):
        self.ppi_distances = {}
        self.gene_centrality = {}
        self.cross_channel = {}
        self.communities_data = {}
        self.pairwise_edges = {}
        self.gene_properties = {}
        self._loaded = False

    def load(self, force=False):
        """Load all precomputed data (from cache or Neo4j)."""
        t0 = time.time()
        print("Loading precomputed graph data...", flush=True)

        self.ppi_distances = precompute_ppi_distances(force=force)
        self.gene_centrality = precompute_gene_centrality(force=force)
        self.cross_channel = precompute_cross_channel_matrix(force=force)
        self.communities_data = precompute_communities(force=force)
        self.pairwise_edges = precompute_pairwise_edges(force=force)
        self.gene_properties = precompute_gene_properties(force=force)

        self._loaded = True
        print(f"Precomputed data loaded in {time.time()-t0:.1f}s\n", flush=True)
        return self

    def get_ppi_dist(self, gene_a, gene_b, default=10):
        return get_ppi_distance(self.ppi_distances, gene_a, gene_b, default)

    def get_pairwise_score(self, gene_a, gene_b):
        """Get precomputed total pairwise score for a gene pair."""
        a, b = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
        entry = self.pairwise_edges.get(f"{a}|{b}")
        if entry:
            return entry['score']
        return 0.0

    def get_pairwise_additive(self, gene_a, gene_b):
        """Get additive pairwise score (statistical associations)."""
        a, b = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
        entry = self.pairwise_edges.get(f"{a}|{b}")
        if entry:
            return entry.get('additive', entry['score'])
        return 0.0

    def get_pairwise_multiplier(self, gene_a, gene_b):
        """Get multiplicative pairwise factor (functional interactions).

        Returns 1.0 + accumulated multiplier increments.
        Multiplicative edges amplify the individual gene effects.
        """
        a, b = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
        entry = self.pairwise_edges.get(f"{a}|{b}")
        if entry:
            return 1.0 + entry.get('multiplier', 0.0)
        return 1.0

    def get_pairwise_ct_cooccur(self, gene_a, gene_b, cancer_type):
        """Get cancer-type-specific cooccurrence count."""
        a, b = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
        entry = self.pairwise_edges.get(f"{a}|{b}")
        if entry and cancer_type:
            return entry.get('cooccur_by_ct', {}).get(cancer_type, 0)
        return 0

    def get_gene_community(self, gene):
        """Get community info: {channel, community_idx, community_size}."""
        return self.communities_data.get('gene_community', {}).get(gene)

    def get_channel_communities(self, channel):
        """Get list of gene lists per community in a channel."""
        return self.communities_data.get('communities', {}).get(channel, [])

    def get_cross_channel_weight(self, ch_a, ch_b):
        """Get normalized cross-channel interaction weight."""
        matrix = self.cross_channel.get('matrix', {})
        w = matrix.get(f"{ch_a}|{ch_b}", 0)
        w += matrix.get(f"{ch_b}|{ch_a}", 0)
        return w

    def get_gene_centrality_score(self, gene):
        """Get combined centrality score (0-1 range)."""
        c = self.gene_centrality.get(gene)
        if not c:
            return 0.0
        # Combine degree and betweenness
        degree_score = min(c.get('degree', 0) / 50.0, 1.0)  # cap at 50
        btw_score = c.get('betweenness_norm', 0)
        return 0.6 * degree_score + 0.4 * btw_score

    def get_gene_confidence(self, gene, atlas_tier):
        """Gene confidence weight based on atlas tier and centrality.

        Returns a weight in (0, 1] that modulates how much this gene
        contributes to the patient score. Genes with strong evidence
        (T1/T2 atlas, high centrality) get full weight. Genes with
        weak evidence get reduced weight to prevent noise dilution.
        """
        centrality = self.get_gene_centrality_score(gene)

        # Atlas tier confidence
        if atlas_tier == 1:
            atlas_conf = 1.0
        elif atlas_tier == 2:
            atlas_conf = 0.85
        elif atlas_tier == 3:
            atlas_conf = 0.5
        elif atlas_tier == 4:
            atlas_conf = 0.4
        else:
            atlas_conf = 0.15  # no atlas — minimal contribution

        # Combine: atlas dominates, centrality provides a floor
        # A gene with no atlas but high centrality still gets some credit
        return max(atlas_conf, centrality * 0.4)

    def invalidate(self):
        """Clear all caches."""
        for name in ['ppi_distances', 'gene_centrality', 'cross_channel_matrix',
                      'communities', 'pairwise_edges', 'gene_properties']:
            path = _cache_path(name)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Invalidated: {name}")


# Singleton
_PRECOMPUTED = None

def get_precomputed(force=False):
    global _PRECOMPUTED
    if _PRECOMPUTED is None or force:
        _PRECOMPUTED = GraphPrecomputed()
        _PRECOMPUTED.load(force=force)
    return _PRECOMPUTED


if __name__ == "__main__":
    print("=" * 70)
    print("  GRAPH PRECOMPUTE — push computation into Neo4j")
    print("=" * 70)

    gp = GraphPrecomputed()
    gp.load(force=True)

    print("\nSummary:")
    print(f"  PPI distances: {len(gp.ppi_distances)} pairs")
    print(f"  Gene centrality: {len(gp.gene_centrality)} genes")
    print(f"  Cross-channel pairs: {len(gp.cross_channel.get('matrix', {}))} ")
    print(f"  Communities: {len(gp.communities_data.get('gene_community', {}))} genes")
    print(f"  Pairwise edges: {len(gp.pairwise_edges)} pairs")
    print(f"  Gene properties: {len(gp.gene_properties)} genes")

    # Test lookups
    print("\nExample lookups:")
    print(f"  TP53-BRCA1 PPI dist: {gp.get_ppi_dist('TP53', 'BRCA1')}")
    print(f"  TP53-BRCA1 pairwise: {gp.get_pairwise_score('TP53', 'BRCA1'):.4f}")
    print(f"  TP53 centrality: {gp.get_gene_centrality_score('TP53'):.4f}")
    print(f"  TP53 community: {gp.get_gene_community('TP53')}")
    print(f"  DDR-PI3K_Growth interaction: {gp.get_cross_channel_weight('DDR', 'PI3K_Growth'):.4f}")
