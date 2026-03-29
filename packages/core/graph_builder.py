"""
Build PyG Data objects from patient mutation profiles.

Each patient becomes a graph with:
  - 92 nodes (one per coupling-channel gene)
  - Edges from STRING PPI (within-channel + high-confidence cross-channel)
  - Node features: mutation info + structural features (18-dim)
  - Survival target: (time, event)
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from ..config import (
    ALL_GENES, GENE_TO_IDX, GENE_TO_CHANNEL_IDX,
    CHANNEL_MAP, NON_SILENT, N_GENES,
)
from .feature_encoder import MutationFeatureEncoder
from .ppi_networks import build_ppi_networks, get_edge_index_for_channel


class PatientGraphBuilder:
    """Build PyG graphs for patients from mutation + clinical data."""

    def __init__(self, ppi_graphs=None):
        self.encoder = MutationFeatureEncoder()

        if ppi_graphs is None:
            self.ppi_graphs = build_ppi_networks()
        else:
            self.ppi_graphs = ppi_graphs

        # Precompute global edge_index from PPI
        self._edge_index = self._build_global_edge_index()

        # Channel assignment for all 92 nodes
        self._channel_assignment = torch.tensor(
            [GENE_TO_CHANNEL_IDX[g] for g in ALL_GENES],
            dtype=torch.long,
        )

    def _build_global_edge_index(self):
        """Build a single edge_index tensor for all PPI edges (within + cross channel)."""
        all_edges = []

        # Within-channel edges
        for ch_name, G in self.ppi_graphs.items():
            if ch_name == "cross_channel":
                continue
            for u, v in G.edges():
                if u in GENE_TO_IDX and v in GENE_TO_IDX:
                    i, j = GENE_TO_IDX[u], GENE_TO_IDX[v]
                    all_edges.append([i, j])
                    all_edges.append([j, i])

        # Cross-channel edges (high confidence)
        if "cross_channel" in self.ppi_graphs:
            for u, v in self.ppi_graphs["cross_channel"].edges():
                if u in GENE_TO_IDX and v in GENE_TO_IDX:
                    i, j = GENE_TO_IDX[u], GENE_TO_IDX[v]
                    all_edges.append([i, j])
                    all_edges.append([j, i])

        if not all_edges:
            # Fallback: add self-loops so GAT doesn't break
            all_edges = [[i, i] for i in range(N_GENES)]

        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index

    def build_graph(self, patient_id, patient_mutations, time, event,
                    cancer_type=None):
        """Build a single patient's graph.

        Args:
            patient_id: str
            patient_mutations: DataFrame of this patient's mutations
            time: float, OS_MONTHS
            event: int, 1=death 0=censored
            cancer_type: optional str

        Returns:
            torch_geometric.data.Data
        """
        # Filter to non-silent, channel-mapped mutations
        if patient_mutations is not None and len(patient_mutations) > 0:
            muts = patient_mutations[
                patient_mutations["mutationType"].isin(NON_SILENT) &
                patient_mutations["gene.hugoGeneSymbol"].isin(GENE_TO_IDX)
            ]
        else:
            muts = pd.DataFrame()

        # Encode features
        x, mutated_mask = self.encoder.encode_patient(muts)

        # Count channels severed
        channels_severed = set()
        for gene in muts["gene.hugoGeneSymbol"].unique() if len(muts) > 0 else []:
            if gene in CHANNEL_MAP:
                channels_severed.add(CHANNEL_MAP[gene])

        data = Data(
            x=x,
            edge_index=self._edge_index.clone(),
            channel_assignment=self._channel_assignment.clone(),
            y_time=torch.tensor([time], dtype=torch.float),
            y_event=torch.tensor([event], dtype=torch.float),
            mutated_mask=mutated_mask,
            n_channels_severed=torch.tensor([len(channels_severed)], dtype=torch.long),
        )
        return data
