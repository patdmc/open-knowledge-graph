"""
CouplingChannelGATFast — precomputed neighbor lookup, lean aggregation.

Replaces GAT's dynamic edge computation with:
  1. Fixed neighbor index table (precomputed from PPI)
  2. Gather neighbor features via tensor indexing
  3. Learned weighted aggregation (not full multi-head attention)

This keeps the expressiveness of learning different neighbor importance
while avoiding the O(N * max_nbrs * heads * head_dim) memory blowup.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    ALL_GENES, GENE_TO_IDX, GENE_TO_CHANNEL_IDX, N_GENES,
)
from ..data.ppi_networks import build_ppi_networks
from .layers import CrossChannelAttention, PatientReadout


class NeighborAggLayer(nn.Module):
    """Learned neighbor aggregation with precomputed neighbor indices.

    For each node i with neighbors N(i):
      agg_i = sum_j( alpha_ij * W_v * x_j )
    where alpha_ij = softmax_j( LeakyReLU(a^T [W_q*x_i || W_k*x_j]) )

    This is the GAT attention mechanism but with fixed neighbor lists
    instead of dynamic edge_index, enabling pure tensor operations.
    """

    def __init__(self, in_dim, hidden_dim, neighbor_idx, neighbor_mask, dropout=0.3):
        super().__init__()
        self.W_self = nn.Linear(in_dim, hidden_dim)
        self.W_nbr = nn.Linear(in_dim, hidden_dim)
        # Attention: score how important each neighbor is
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.register_buffer('neighbor_idx', neighbor_idx)
        self.register_buffer('neighbor_mask', neighbor_mask)

    def forward(self, x):
        """
        Args:
            x: (B, N_GENES, in_dim)
        Returns:
            (B, N_GENES, hidden_dim)
        """
        B, N, D = x.shape
        residual = self.proj(x)
        max_nbrs = self.neighbor_idx.shape[1]

        # Project self and neighbors
        x_self = self.W_self(x)          # (B, N, H)
        x_nbr_proj = self.W_nbr(x)      # (B, N, H)

        # Gather neighbor projections
        safe_idx = self.neighbor_idx.clamp(min=0)  # (N, max_nbrs)
        nbr_feats = x_nbr_proj[:, safe_idx]        # (B, N, max_nbrs, H)

        # Compute attention scores
        # Expand self for concatenation: (B, N, max_nbrs, H)
        x_self_exp = x_self.unsqueeze(2).expand(-1, -1, max_nbrs, -1)
        # Concatenate self + neighbor: (B, N, max_nbrs, 2H)
        concat = torch.cat([x_self_exp, nbr_feats], dim=-1)
        # Score: (B, N, max_nbrs, 1) -> (B, N, max_nbrs)
        scores = self.attn_score(concat).squeeze(-1)

        # Mask invalid neighbors
        mask = self.neighbor_mask.unsqueeze(0)  # (1, N, max_nbrs)
        scores = scores.masked_fill(~mask, float('-inf'))

        # Check for nodes with no neighbors
        has_nbrs = self.neighbor_mask.any(dim=1)  # (N,)

        # Softmax over neighbors
        weights = F.softmax(scores, dim=-1)       # (B, N, max_nbrs)
        weights = self.dropout(weights)

        # Weighted aggregation
        # (B, N, max_nbrs, 1) * (B, N, max_nbrs, H) -> sum -> (B, N, H)
        agg = (weights.unsqueeze(-1) * nbr_feats).sum(dim=2)

        # Zero out isolated nodes
        agg = agg * has_nbrs.unsqueeze(0).unsqueeze(-1).float()

        # Combine with residual
        out = self.norm(agg + residual)
        out = F.elu(out)
        return out


class CouplingChannelGATFast(nn.Module):
    """Fast coupling-channel GNN with precomputed graph structure."""

    def __init__(self, config, ppi_graphs=None):
        super().__init__()
        node_feat_dim = config["node_feat_dim"]
        hidden_dim = config["hidden_dim"]
        dropout = config["dropout"]
        n_layers = config["num_channel_layers"]

        neighbor_idx, neighbor_mask = self._build_neighbor_table(ppi_graphs)

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Neighbor aggregation layers
        self.gat_layers = nn.ModuleList([
            NeighborAggLayer(hidden_dim, hidden_dim, neighbor_idx, neighbor_mask, dropout)
            for _ in range(n_layers)
        ])

        # Channel readout attention
        self.channel_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Precompute channel masks
        ch_assign = torch.tensor([GENE_TO_CHANNEL_IDX[g] for g in ALL_GENES])
        self._channel_gene_indices = {}
        for c in range(6):
            self._channel_gene_indices[c] = (ch_assign == c).nonzero(as_tuple=True)[0]

        # Cross-channel attention
        self.cross_channel = CrossChannelAttention(
            hidden_dim,
            num_heads=config["cross_channel_heads"],
            num_layers=config["cross_channel_layers"],
            dropout=dropout,
        )

        # Patient readout
        self.patient_readout = PatientReadout(
            hidden_dim,
            readout_hidden=config["readout_hidden"],
            dropout=dropout,
        )

    def _build_neighbor_table(self, ppi_graphs=None):
        if ppi_graphs is None:
            ppi_graphs = build_ppi_networks()

        neighbor_list = [set() for _ in range(N_GENES)]
        for ch_name, G in ppi_graphs.items():
            if ch_name == "cross_channel":
                continue
            for u, v in G.edges():
                if u in GENE_TO_IDX and v in GENE_TO_IDX:
                    i, j = GENE_TO_IDX[u], GENE_TO_IDX[v]
                    neighbor_list[i].add(j)
                    neighbor_list[j].add(i)

        max_nbrs = max((len(n) for n in neighbor_list), default=1)
        max_nbrs = max(max_nbrs, 1)

        neighbor_idx = torch.zeros(N_GENES, max_nbrs, dtype=torch.long)
        neighbor_mask = torch.zeros(N_GENES, max_nbrs, dtype=torch.bool)
        for i, nbrs in enumerate(neighbor_list):
            for j, n in enumerate(sorted(nbrs)):
                neighbor_idx[i, j] = n
                neighbor_mask[i, j] = True

        return neighbor_idx, neighbor_mask

    def forward(self, data):
        batch = data.batch
        B = batch.max().item() + 1

        # Reshape: (B*N_GENES, feat) -> (B, N_GENES, feat)
        x = data.x.view(B, N_GENES, -1)

        # 1. Encode
        x = self.node_encoder(x)

        # 2. Neighbor aggregation (precomputed graph, pure tensor ops)
        for layer in self.gat_layers:
            x = layer(x)

        # 3. Channel readout
        hidden_dim = x.shape[-1]
        scores = self.channel_attn(x).squeeze(-1)  # (B, N_GENES)

        channel_embeddings = torch.zeros(B, 6, hidden_dim, device=x.device)
        for c, gene_idx in self._channel_gene_indices.items():
            c_scores = scores[:, gene_idx]               # (B, n_c)
            c_x = x[:, gene_idx]                          # (B, n_c, H)
            weights = F.softmax(c_scores, dim=-1)         # (B, n_c)
            channel_embeddings[:, c] = (weights.unsqueeze(-1) * c_x).sum(dim=1)

        # 4. Cross-channel attention
        channel_embeddings = self.cross_channel(channel_embeddings)

        # 5. Hazard score
        return self.patient_readout(channel_embeddings)
