"""
Dynamic graph schema discovery.

Queries Neo4j at runtime to discover:
  - All gene-gene edge types and their numeric properties
  - All gene node properties beyond the basics
  - Block/community assignments

Nothing is hardcoded. When new data sources add edge types or node
properties, the schema adapts automatically.
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache"
)


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver("bolt://localhost:7687",
                                auth=("neo4j", "openknowledgegraph"))


class GraphSchema:
    """Dynamically discovered graph schema — the graph defines its own features."""

    def __init__(self):
        self.edge_types = []           # ordered list of gene-gene edge types
        self.edge_properties = {}      # edge_type → list of numeric property names
        self.edge_feature_dim = 0      # total features per gene pair
        self.edge_feature_map = {}     # (edge_type, prop) → index in feature vector

        self.node_extra_props = []     # numeric Gene properties beyond basics
        self.node_extra_dim = 0
        self.node_prop_map = {}        # prop_name → index in extra feature vector

        self.gene_block = {}           # gene → {'channel': str, 'channel_id': int, 'block_id': int}
        self.n_blocks = 0
        self.n_channels = 0

        # Loaded data
        self.pairwise_edges = {}       # (ga, gb) → np.array of shape (edge_feature_dim,)
        self.node_extra_features = {}  # gene → np.array of shape (node_extra_dim,)

    def discover(self, force_refresh=False):
        """Query Neo4j to discover the full schema. Caches results."""
        cache_path = os.path.join(CACHE_DIR, "graph_schema.json")

        if not force_refresh and os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_hours = (time.time() - mtime) / 3600
            if age_hours < 1:  # cache valid for 1 hour
                self._load_cache(cache_path)
                return self

        print("Discovering graph schema from Neo4j...", flush=True)
        driver = _neo4j_driver()

        with driver.session() as s:
            self._discover_edge_types(s)
            self._discover_node_properties(s)
            self._discover_blocks(s)

        driver.close()
        self._save_cache(cache_path)
        return self

    def _discover_edge_types(self, session):
        """Find all gene-gene relationship types and their numeric properties."""
        # Get all relationship types between Gene nodes (skip deprecated)
        result = session.run("""
            MATCH (a:Gene)-[r]->(b:Gene)
            WHERE r.deprecated IS NULL OR r.deprecated = false
            WITH DISTINCT type(r) AS rtype
            RETURN rtype ORDER BY rtype
        """)
        self.edge_types = [r['rtype'] for r in result]
        print(f"  Edge types ({len(self.edge_types)}): {self.edge_types}", flush=True)

        # For each edge type, discover numeric properties
        # Exclude properties that have ANY string values (mixed-type = not truly numeric)
        self.edge_properties = {}
        for etype in self.edge_types:
            result = session.run(f"""
                MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                WITH r LIMIT 100
                UNWIND keys(r) AS k
                WITH DISTINCT k,
                     collect(DISTINCT CASE WHEN r[k] IS :: INTEGER OR r[k] IS :: FLOAT THEN 'num'
                                           WHEN r[k] IS :: STRING THEN 'str' END) AS types
                WHERE 'num' IN types AND NOT 'str' IN types
                RETURN k ORDER BY k
            """)
            props = [r['k'] for r in result]
            # Filter out metadata props
            props = [p for p in props if p not in ('created_at', 'updated_at', 'n_sources')]
            self.edge_properties[etype] = props
            print(f"    {etype}: {props}", flush=True)

        # Build feature index: each (edge_type, prop) gets an index
        # Also add a binary "exists" feature per edge type
        idx = 0
        self.edge_feature_map = {}
        for etype in self.edge_types:
            # Binary: does this edge type exist?
            self.edge_feature_map[(etype, '_exists')] = idx
            idx += 1
            # Numeric properties
            for prop in self.edge_properties[etype]:
                self.edge_feature_map[(etype, prop)] = idx
                idx += 1

        self.edge_feature_dim = idx
        print(f"  Edge feature dim: {self.edge_feature_dim}", flush=True)

    def _discover_node_properties(self, session):
        """Find all numeric Gene node properties beyond basics."""
        result = session.run("""
            MATCH (g:Gene)
            WHERE g.channel IS NOT NULL
            WITH g LIMIT 100
            UNWIND keys(g) AS k
            WITH DISTINCT k, g[k] AS v
            WHERE v IS NOT NULL AND (v IS :: INTEGER OR v IS :: FLOAT)
            RETURN DISTINCT k ORDER BY k
        """)
        # Filter out basic/identity props
        skip = {'name', 'channel', 'created_at', 'updated_at', 'n_sources'}
        self.node_extra_props = [r['k'] for r in result if r['k'] not in skip]
        self.node_extra_dim = len(self.node_extra_props)
        self.node_prop_map = {p: i for i, p in enumerate(self.node_extra_props)}
        print(f"  Node extra properties ({self.node_extra_dim}): {self.node_extra_props}", flush=True)

    def _discover_blocks(self, session):
        """Discover sub-pathway communities from graph structure."""
        from gnn.config import CHANNEL_MAP, CHANNEL_NAMES
        from gnn.data.block_assignments import load_block_assignments

        self.gene_block, self.n_blocks, self.n_channels = load_block_assignments()
        print(f"  Blocks: {self.n_blocks}, Channels: {self.n_channels}", flush=True)

    def load_edge_features(self, force_refresh=False):
        """Load all pairwise edge features from Neo4j."""
        cache_path = os.path.join(CACHE_DIR, "pairwise_edges_dynamic.npz")

        if not force_refresh and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            self.pairwise_edges = dict(data['edge_dict'].item())
            print(f"  Loaded {len(self.pairwise_edges)} cached pairwise edge features", flush=True)
            return

        print("Loading pairwise edge features from Neo4j...", flush=True)
        driver = _neo4j_driver()
        self.pairwise_edges = {}

        with driver.session() as s:
            for etype in self.edge_types:
                props = self.edge_properties[etype]
                # Build return clause for properties
                prop_returns = ", ".join(f"r.{p} AS `{p}`" for p in props)
                query = f"""
                    MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                    WHERE r.deprecated IS NULL OR r.deprecated = false
                    RETURN a.name AS f, b.name AS t{', ' + prop_returns if prop_returns else ''}
                """
                result = s.run(query)
                n = 0
                for r in result:
                    pair = (r['f'], r['t'])
                    if pair not in self.pairwise_edges:
                        self.pairwise_edges[pair] = np.zeros(self.edge_feature_dim, dtype=np.float32)

                    feat = self.pairwise_edges[pair]
                    # Set exists flag
                    feat[self.edge_feature_map[(etype, '_exists')]] = 1.0
                    # Set numeric properties
                    for prop in props:
                        val = r[prop]
                        if val is not None:
                            try:
                                fval = float(val)
                                if np.isfinite(fval):
                                    idx = self.edge_feature_map[(etype, prop)]
                                    feat[idx] = fval
                            except (ValueError, TypeError):
                                pass  # skip non-numeric values
                    n += 1
                print(f"    {etype}: {n:,} edges", flush=True)

        driver.close()

        # Normalize: log-scale large values, clip extremes
        for pair, feat in self.pairwise_edges.items():
            for (etype, prop), idx in self.edge_feature_map.items():
                if prop == '_exists':
                    continue
                if prop in ('count',):
                    feat[idx] = np.log1p(feat[idx]) / 10.0  # log-scale counts
                elif prop in ('score',):
                    feat[idx] = min(feat[idx] / 1000.0, 1.0)  # normalize PPI scores
                # correlation/weight values already in [-1, 1] or [0, 1]

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, edge_dict=self.pairwise_edges)
        print(f"  Cached {len(self.pairwise_edges)} pairwise edge features", flush=True)

    def load_node_features(self, force_refresh=False):
        """Load enriched gene node properties from Neo4j."""
        cache_path = os.path.join(CACHE_DIR, "gene_node_dynamic.npz")

        if not force_refresh and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            self.node_extra_features = dict(data['node_dict'].item())
            print(f"  Loaded {len(self.node_extra_features)} cached gene node features", flush=True)
            return

        print("Loading gene node extra features from Neo4j...", flush=True)
        driver = _neo4j_driver()
        self.node_extra_features = {}

        with driver.session() as s:
            result = s.run("""
                MATCH (g:Gene)
                WHERE g.channel IS NOT NULL
                RETURN g.name AS name, properties(g) AS props
            """)
            for r in result:
                name = r['name']
                props = dict(r['props'])
                feat = np.zeros(self.node_extra_dim, dtype=np.float32)

                for prop_name, idx in self.node_prop_map.items():
                    val = props.get(prop_name)
                    if val is not None:
                        try:
                            feat[idx] = float(val)
                        except (ValueError, TypeError):
                            pass

                self.node_extra_features[name] = feat

        driver.close()

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, node_dict=self.node_extra_features)
        print(f"  Cached {len(self.node_extra_features)} gene node features", flush=True)

    def load_all(self, force_refresh=False):
        """Discover schema and load all features."""
        self.discover(force_refresh=force_refresh)
        self.load_edge_features(force_refresh=force_refresh)
        self.load_node_features(force_refresh=force_refresh)
        return self

    def get_edge_feature_vector(self, gene_a, gene_b):
        """Get edge feature vector for a gene pair (checks both directions)."""
        feat = self.pairwise_edges.get((gene_a, gene_b))
        if feat is not None:
            return feat
        feat = self.pairwise_edges.get((gene_b, gene_a))
        if feat is not None:
            return feat
        return np.zeros(self.edge_feature_dim, dtype=np.float32)

    def get_node_extra_features(self, gene):
        """Get extra node features for a gene."""
        feat = self.node_extra_features.get(gene)
        if feat is not None:
            return feat
        return np.zeros(self.node_extra_dim, dtype=np.float32)

    def get_block_id(self, gene):
        """Get block ID for a gene. Returns -1 if unassigned."""
        info = self.gene_block.get(gene)
        return info['block_id'] if info else -1

    def get_channel_id(self, gene):
        """Get channel ID for a gene. Returns n_channels if unassigned."""
        info = self.gene_block.get(gene)
        return info['channel_id'] if info else self.n_channels

    def invalidate_cache(self):
        """Clear all caches — call when graph changes."""
        for fname in ['graph_schema.json', 'pairwise_edges_dynamic.npz',
                       'gene_node_dynamic.npz', 'block_assignments.json']:
            path = os.path.join(CACHE_DIR, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Invalidated cache: {fname}")

    def _save_cache(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'edge_types': self.edge_types,
                'edge_properties': self.edge_properties,
                'edge_feature_dim': self.edge_feature_dim,
                'edge_feature_map': {f"{k[0]}|{k[1]}": v
                                     for k, v in self.edge_feature_map.items()},
                'node_extra_props': self.node_extra_props,
                'node_extra_dim': self.node_extra_dim,
                'n_blocks': self.n_blocks,
                'n_channels': self.n_channels,
            }, f, indent=2)

    def _load_cache(self, path):
        with open(path, 'r') as f:
            cached = json.load(f)

        self.edge_types = cached['edge_types']
        self.edge_properties = cached['edge_properties']
        self.edge_feature_dim = cached['edge_feature_dim']
        self.edge_feature_map = {
            tuple(k.split('|', 1)): v
            for k, v in cached['edge_feature_map'].items()
        }
        self.node_extra_props = cached['node_extra_props']
        self.node_extra_dim = cached['node_extra_dim']
        self.n_blocks = cached['n_blocks']
        self.n_channels = cached['n_channels']
        self.node_prop_map = {p: i for i, p in enumerate(self.node_extra_props)}

        # Load block assignments
        from gnn.data.block_assignments import load_block_assignments
        self.gene_block, _, _ = load_block_assignments()

        print(f"  Schema from cache: {len(self.edge_types)} edge types, "
              f"{self.edge_feature_dim} edge features, "
              f"{self.node_extra_dim} node extra features, "
              f"{self.n_blocks} blocks", flush=True)


# Singleton for convenience
_SCHEMA = None

def get_schema(force_refresh=False):
    """Get or create the global graph schema."""
    global _SCHEMA
    if _SCHEMA is None or force_refresh:
        _SCHEMA = GraphSchema()
        _SCHEMA.load_all(force_refresh=force_refresh)
    return _SCHEMA


if __name__ == "__main__":
    schema = GraphSchema()
    schema.load_all(force_refresh=True)
    print(f"\nSchema summary:")
    print(f"  Edge types: {schema.edge_types}")
    print(f"  Edge feature dim: {schema.edge_feature_dim}")
    print(f"  Node extra dim: {schema.node_extra_dim}")
    print(f"  Blocks: {schema.n_blocks}")
    print(f"  Channels: {schema.n_channels}")
