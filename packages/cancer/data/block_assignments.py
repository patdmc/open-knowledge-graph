"""
Load sub-pathway block assignments from Neo4j graph communities.

Same community detection as hierarchical_scorer.py, but returns a
serializable mapping: gene → (channel_name, channel_id, block_id).

Block IDs are globally unique across all channels.
"""

import os
import json
import networkx as nx
from collections import defaultdict

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "block_assignments.json"
)


def discover_blocks(channel_genes, channel_names):
    """Discover sub-pathway communities via PPI + COUPLES graph community detection.

    Args:
        channel_genes: dict channel_name → set of gene names
        channel_names: ordered list of channel names

    Returns:
        gene_block: dict gene → {'channel': str, 'channel_id': int, 'block_id': int}
        communities: dict channel → list of gene sets
        n_blocks: total number of blocks
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    channel_idx = {ch: i for i, ch in enumerate(channel_names)}
    communities = {}
    gene_block = {}
    global_block_id = 0

    with driver.session() as s:
        for ch in sorted(channel_genes.keys()):
            genes = channel_genes[ch]
            if len(genes) < 5:
                communities[ch] = [list(genes)]
                for g in genes:
                    gene_block[g] = {
                        'channel': ch,
                        'channel_id': channel_idx[ch],
                        'block_id': global_block_id,
                    }
                global_block_id += 1
                continue

            G = nx.Graph()
            G.add_nodes_from(genes)

            result = s.run("""
                MATCH (a:Gene)-[r:PPI]->(b:Gene)
                WHERE a.channel = $ch AND b.channel = $ch
                RETURN a.name AS from, b.name AS to, r.score AS score
            """, ch=ch)
            for r in result:
                if r['from'] in genes and r['to'] in genes:
                    G.add_edge(r['from'], r['to'],
                               weight=float(r.get('score', 500)))

            result = s.run("""
                MATCH (a:Gene)-[r:COUPLES]->(b:Gene)
                WHERE a.channel = $ch AND b.channel = $ch
                RETURN a.name AS from, b.name AS to
            """, ch=ch)
            for r in result:
                if r['from'] in genes and r['to'] in genes:
                    if G.has_edge(r['from'], r['to']):
                        G[r['from']][r['to']]['weight'] += 200
                    else:
                        G.add_edge(r['from'], r['to'], weight=200)

            try:
                comms = list(nx.community.greedy_modularity_communities(G))
            except Exception:
                comms = [genes]

            final_comms = []
            for comm in comms:
                comm = set(comm)
                if len(comm) <= 25:
                    final_comms.append(list(comm))
                else:
                    sub = G.subgraph(comm)
                    sub_comps = list(nx.connected_components(sub))
                    if len(sub_comps) > 1 and all(len(c) <= 25 for c in sub_comps):
                        final_comms.extend([list(c) for c in sub_comps])
                    else:
                        sorted_nodes = sorted(comm,
                            key=lambda n: G.degree(n, weight='weight'), reverse=True)
                        mid = len(sorted_nodes) // 2
                        final_comms.append(sorted_nodes[:mid])
                        final_comms.append(sorted_nodes[mid:])

            communities[ch] = final_comms
            for ci, comm in enumerate(final_comms):
                for g in comm:
                    gene_block[g] = {
                        'channel': ch,
                        'channel_id': channel_idx[ch],
                        'block_id': global_block_id,
                    }
                global_block_id += 1

    driver.close()
    return gene_block, communities, global_block_id


def load_block_assignments(force_refresh=False):
    """Load cached block assignments, or discover from graph if not cached.

    Returns:
        gene_block: dict gene → {'channel': str, 'channel_id': int, 'block_id': int}
        n_blocks: total number of blocks
        n_channels: number of channels
    """
    from gnn.config import CHANNEL_MAP, CHANNEL_NAMES

    if not force_refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            cached = json.load(f)
        return cached['gene_block'], cached['n_blocks'], len(CHANNEL_NAMES)

    # Build channel_genes from config
    channel_genes = defaultdict(set)
    for gene, ch in CHANNEL_MAP.items():
        channel_genes[ch].add(gene)

    gene_block, communities, n_blocks = discover_blocks(channel_genes, CHANNEL_NAMES)

    # Cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump({
            'gene_block': gene_block,
            'n_blocks': n_blocks,
            'communities': {ch: [list(c) for c in comms]
                           for ch, comms in communities.items()},
        }, f, indent=2)

    print(f"  Block assignments: {len(gene_block)} genes → {n_blocks} blocks")
    return gene_block, n_blocks, len(CHANNEL_NAMES)


CROSS_BLOCK_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "cross_block_edges.json"
)


def discover_cross_block_edges(force_refresh=False):
    """Find PPI edges between genes in different blocks (cross-channel).

    Returns:
        edges: list of (block_i, block_j, n_ppi_edges, avg_score)
        edge_index: (2, E) numpy array of block pairs
        n_blocks: total blocks
    """
    if not force_refresh and os.path.exists(CROSS_BLOCK_CACHE):
        with open(CROSS_BLOCK_CACHE, 'r') as f:
            cached = json.load(f)
        import numpy as np
        edge_index = np.array(cached['edge_index'], dtype=np.int64)
        return cached['edges'], edge_index, cached['n_blocks']

    from neo4j import GraphDatabase
    gene_block, n_blocks, _ = load_block_assignments()

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    # Query all cross-channel PPI edges
    block_pair_stats = defaultdict(lambda: {'count': 0, 'total_score': 0.0})

    with driver.session() as s:
        result = s.run("""
            MATCH (a:Gene)-[r:PPI]->(b:Gene)
            WHERE a.channel <> b.channel
            RETURN a.name AS gene_a, b.name AS gene_b,
                   a.channel AS ch_a, b.channel AS ch_b,
                   r.score AS score
        """)
        for rec in result:
            ga, gb = rec['gene_a'], rec['gene_b']
            info_a = gene_block.get(ga)
            info_b = gene_block.get(gb)
            if info_a and info_b:
                ba, bb = info_a['block_id'], info_b['block_id']
                if ba != bb:
                    key = (min(ba, bb), max(ba, bb))
                    block_pair_stats[key]['count'] += 1
                    block_pair_stats[key]['total_score'] += float(rec.get('score', 0.4) or 0.4)

    driver.close()

    edges = []
    edge_index_list = [[], []]
    for (bi, bj), stats in sorted(block_pair_stats.items()):
        avg_score = stats['total_score'] / max(stats['count'], 1)
        edges.append({
            'block_i': bi, 'block_j': bj,
            'n_ppi': stats['count'], 'avg_score': round(avg_score, 4),
        })
        # Bidirectional
        edge_index_list[0].extend([bi, bj])
        edge_index_list[1].extend([bj, bi])

    import numpy as np
    edge_index = np.array(edge_index_list, dtype=np.int64)

    # Cache
    os.makedirs(os.path.dirname(CROSS_BLOCK_CACHE), exist_ok=True)
    with open(CROSS_BLOCK_CACHE, 'w') as f:
        json.dump({
            'edges': edges,
            'edge_index': edge_index.tolist(),
            'n_blocks': n_blocks,
        }, f, indent=2)

    print(f"  Cross-block edges: {len(edges)} unique pairs, "
          f"{edge_index.shape[1]} directed edges across {n_blocks} blocks")
    return edges, edge_index, n_blocks
