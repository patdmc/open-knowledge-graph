"""
Custom layers for the Coupling-Channel GNN.

Architecture:
  1. ChannelSubgraphLayer: GAT message-passing within a single channel
  2. ChannelReadout: Attention-weighted pooling over nodes in a channel
  3. CrossChannelAttention: Transformer attention across 6 channel embeddings
  4. PatientReadout: Final hazard score from channel embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ChannelSubgraphLayer(nn.Module):
    """Two-layer GAT within a single coupling channel subgraph."""

    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                            dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Projection for residual if dimensions differ
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x, edge_index):
        """
        Args:
            x: (N_channel, in_dim) node features for genes in this channel
            edge_index: (2, E_channel) edges within this channel

        Returns:
            (N_channel, hidden_dim) updated node features
        """
        residual = self.proj(x)
        x = self.gat1(x, edge_index)
        x = self.norm1(x + residual)
        x = F.elu(x)
        x = self.dropout(x)

        residual = x
        x = self.gat2(x, edge_index)
        x = self.norm2(x + residual)
        x = F.elu(x)
        return x


class ChannelReadout(nn.Module):
    """Attention-weighted mean pooling over nodes within a channel."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, hidden_dim) node features
            mask: optional (N,) boolean mask for valid nodes

        Returns:
            (hidden_dim,) single channel embedding
        """
        scores = self.attn(x).squeeze(-1)  # (N,)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=0)  # (N,)
        return (weights.unsqueeze(-1) * x).sum(dim=0)  # (hidden_dim,)


class CrossChannelAttention(nn.Module):
    """Transformer-style self-attention across 6 channel embeddings.

    Learns which channel co-severance patterns are synergistically lethal.
    """

    def __init__(self, hidden_dim, num_heads=4, num_layers=1, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.channel_pos = nn.Embedding(6, hidden_dim)

    def forward(self, channel_embeddings):
        """
        Args:
            channel_embeddings: (B, 6, hidden_dim) — 6 channel embeddings per patient

        Returns:
            (B, 6, hidden_dim) contextualized channel embeddings
        """
        B = channel_embeddings.shape[0]
        pos = self.channel_pos.weight.unsqueeze(0).expand(B, -1, -1)
        x = channel_embeddings + pos
        return self.transformer(x)


class PatientReadout(nn.Module):
    """Aggregate 6 channel embeddings into a single hazard score."""

    def __init__(self, hidden_dim, readout_hidden=32, dropout=0.3):
        super().__init__()
        # Learned attention over channels
        self.channel_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Final MLP: channel representation -> hazard
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, readout_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, channel_embeddings):
        """
        Args:
            channel_embeddings: (B, 6, hidden_dim)

        Returns:
            (B,) log partial hazard scores
        """
        # Attention-weighted pooling over channels
        scores = self.channel_attn(channel_embeddings).squeeze(-1)  # (B, 6)
        weights = F.softmax(scores, dim=-1)  # (B, 6)
        patient_repr = (weights.unsqueeze(-1) * channel_embeddings).sum(dim=1)  # (B, hidden_dim)
        return self.mlp(patient_repr).squeeze(-1)  # (B,)
