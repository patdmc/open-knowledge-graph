"""
Fetch and cache protein-protein interaction networks from STRING.
Build per-channel subgraphs for the coupling-channel GNN.
"""

import os
import json
import requests
import networkx as nx

from ..config import (
    ALL_GENES, CHANNEL_MAP, GENES_PER_CHANNEL, CHANNEL_NAMES,
    GNN_CACHE, STRING_SPECIES, STRING_SCORE_THRESHOLD,
)


PPI_CACHE_FILE = os.path.join(GNN_CACHE, "string_ppi_edges.json")


def fetch_string_interactions(genes, species=STRING_SPECIES,
                              score_threshold=STRING_SCORE_THRESHOLD):
    """Fetch PPI interactions from STRING API for a list of genes.

    Returns list of (gene1, gene2, combined_score) tuples.
    """
    url = "https://string-db.org/api/json/network"
    # STRING API limit is 2000 identifiers per request
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": score_threshold,
        "network_type": "functional",
        "caller_identity": "coupling_channel_gnn",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    edges = []
    for interaction in data:
        g1 = interaction["preferredName_A"]
        g2 = interaction["preferredName_B"]
        score = interaction["score"]
        if g1 in set(genes) and g2 in set(genes):
            edges.append((g1, g2, score))
    return edges


def build_ppi_networks(force_refresh=False):
    """Build per-channel PPI subgraphs.

    Returns:
        dict mapping channel_name -> nx.Graph
        Also includes "cross_channel" -> nx.Graph for inter-channel edges
    """
    # Load from cache if available
    if os.path.exists(PPI_CACHE_FILE) and not force_refresh:
        with open(PPI_CACHE_FILE) as f:
            cached = json.load(f)
        return _edges_to_graphs(cached)

    print("Fetching STRING PPI interactions for 92 coupling-channel genes...")
    all_edges = fetch_string_interactions(ALL_GENES)
    print(f"  Retrieved {len(all_edges)} interactions")

    # Classify edges as within-channel or cross-channel
    edge_dict = {"cross_channel": []}
    for ch in CHANNEL_NAMES:
        edge_dict[ch] = []

    gene_set = set(ALL_GENES)
    for g1, g2, score in all_edges:
        if g1 not in gene_set or g2 not in gene_set:
            continue
        ch1 = CHANNEL_MAP.get(g1)
        ch2 = CHANNEL_MAP.get(g2)
        if ch1 == ch2:
            edge_dict[ch1].append([g1, g2, score])
        else:
            edge_dict["cross_channel"].append([g1, g2, score])

    # Cache
    with open(PPI_CACHE_FILE, "w") as f:
        json.dump(edge_dict, f, indent=2)
    print(f"  Cached to {PPI_CACHE_FILE}")

    # Print summary
    for ch in CHANNEL_NAMES:
        print(f"  {ch}: {len(edge_dict[ch])} within-channel edges")
    print(f"  Cross-channel: {len(edge_dict['cross_channel'])} edges")

    return _edges_to_graphs(edge_dict)


def _edges_to_graphs(edge_dict):
    """Convert edge dict to networkx graphs."""
    graphs = {}
    for key, edges in edge_dict.items():
        G = nx.Graph()
        # Add all genes for this channel as nodes (even if no edges)
        if key in GENES_PER_CHANNEL:
            for g in GENES_PER_CHANNEL[key]:
                G.add_node(g)
        for g1, g2, score in edges:
            G.add_edge(g1, g2, weight=score)
        graphs[key] = G
    return graphs


def get_edge_index_for_channel(graphs, channel, gene_to_local_idx):
    """Convert a channel's networkx graph to a PyG edge_index tensor.

    Args:
        graphs: dict from build_ppi_networks()
        channel: channel name
        gene_to_local_idx: dict mapping gene name -> local node index

    Returns:
        (2, E) LongTensor edge_index
    """
    import torch

    G = graphs.get(channel, nx.Graph())
    edges = []
    for u, v in G.edges():
        if u in gene_to_local_idx and v in gene_to_local_idx:
            i, j = gene_to_local_idx[u], gene_to_local_idx[v]
            edges.append([i, j])
            edges.append([j, i])  # undirected
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
