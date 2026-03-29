"""
CouplingChannelGAT — the main model.

Architecture:
  1. Encode per-gene node features (mutation type, VAF, hub/leaf, GOF/LOF)
  2. Within-channel GAT message passing — SINGLE fused call across all channels
  3. Channel readout: scatter-based attention pooling per channel per patient
  4. Cross-channel transformer attention (learns co-severance interactions)
  5. Patient readout: attention over channels -> hazard score

PERFORMANCE:
  All patients share the same 99-node graph topology.
  Within-channel GAT runs as ONE call on all nodes — edges already separate
  channels (no cross-channel edges), so GAT naturally processes channels
  independently in a single fused operation.
  Per-channel edge indices precomputed at init, never recomputed.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ..config import (
    ALL_GENES, GENE_TO_IDX, GENE_TO_CHANNEL_IDX,
    CHANNEL_NAMES, N_GENES, CHANNEL_TO_IDX,
)
from .layers import (
    ChannelSubgraphLayer,
    CrossChannelAttention,
    PatientReadout,
)


class ChannelReadoutBatched(nn.Module):
    """Attention-weighted pooling over nodes, batched via scatter."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, batch_idx, num_graphs):
        scores = self.attn(x).squeeze(-1)
        max_scores = scatter(scores, batch_idx, dim=0, reduce='max',
                           dim_size=num_graphs)
        scores = scores - max_scores[batch_idx]
        exp_scores = scores.exp()
        sum_exp = scatter(exp_scores, batch_idx, dim=0, reduce='sum',
                         dim_size=num_graphs)
        weights = exp_scores / (sum_exp[batch_idx] + 1e-8)
        weighted = weights.unsqueeze(-1) * x
        return scatter(weighted, batch_idx, dim=0, reduce='sum',
                      dim_size=num_graphs)


class CouplingChannelGAT(nn.Module):

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config["node_feat_dim"]
        hidden_dim = config["hidden_dim"]
        heads = config["num_gat_heads"]
        dropout = config["dropout"]
        n_channel_layers = config["num_channel_layers"]

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.channel_layers = nn.ModuleList([
            ChannelSubgraphLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(n_channel_layers)
        ])

        self.channel_readout = ChannelReadoutBatched(hidden_dim)

        self.cross_channel = CrossChannelAttention(
            hidden_dim,
            num_heads=config["cross_channel_heads"],
            num_layers=config["cross_channel_layers"],
            dropout=dropout,
        )

        self.patient_readout = PatientReadout(
            hidden_dim,
            readout_hidden=config["readout_hidden"],
            dropout=dropout,
        )

        # Precompute per-channel edge masks for one patient's graph.
        # These are offsets relative to a single patient's 99 nodes.
        # At forward time we tile them across the batch.
        self._precompute_single_patient_channel_edges()

    def _precompute_single_patient_channel_edges(self):
        """Precompute within-channel edge indices for one 99-node patient graph.

        Stored as a single edge_index containing ONLY within-channel edges.
        Since channels don't share edges, running GAT on this edge_index
        is equivalent to running 6 separate GATs — but in one fused call.
        """
        # Build channel assignment for one patient
        ch_assign = torch.tensor(
            [GENE_TO_CHANNEL_IDX[g] for g in ALL_GENES], dtype=torch.long
        )
        self.register_buffer('_single_channel_assign', ch_assign)

    def _build_within_channel_edges(self, edge_index, channel_idx):
        """Filter edge_index to only within-channel edges.

        This is a single vectorized operation — no per-channel loop needed.
        Edges connecting nodes in the same channel are kept; cross-channel dropped.
        """
        src_ch = channel_idx[edge_index[0]]
        dst_ch = channel_idx[edge_index[1]]
        within_mask = (src_ch == dst_ch)
        return edge_index[:, within_mask]

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        channel_idx = data.channel_assignment
        batch = data.batch
        device = x.device

        B = batch.max().item() + 1

        # 1. Encode node features
        x = self.node_encoder(x)
        hidden_dim = x.shape[1]

        # 2. Filter to within-channel edges ONCE (one vectorized op)
        within_edges = self._build_within_channel_edges(edge_index, channel_idx)

        # 3. Run GAT on ALL nodes with within-channel edges only.
        #    Since there are no cross-channel edges, each channel's nodes
        #    form disconnected subgraphs — GAT processes them independently
        #    but in a SINGLE fused call. No per-channel loop needed.
        for layer in self.channel_layers:
            x = layer(x, within_edges)

        # 4. Channel readout — one scatter call per channel
        channel_embeddings = torch.zeros(B, 6, hidden_dim, device=device)
        for c in range(6):
            c_mask = (channel_idx == c)
            if c_mask.sum() == 0:
                continue
            channel_embeddings[:, c, :] = self.channel_readout(
                x[c_mask], batch[c_mask], B
            )

        # 5. Cross-channel attention
        channel_embeddings = self.cross_channel(channel_embeddings)

        # 6. Patient readout -> hazard
        return self.patient_readout(channel_embeddings)
