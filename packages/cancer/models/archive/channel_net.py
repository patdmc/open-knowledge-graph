"""
ChannelNet — Channel-level GNN with skip connections.

Instead of 99 gene nodes, pre-aggregate to 6 channel nodes.
Each channel node gets a rich feature vector:
  - n_mutations in this channel
  - n_genes_mutated in this channel
  - has_hub, has_leaf
  - fraction truncating vs missense
  - mean VAF
  - GOF/LOF ratio
  - channel identity (one-hot)

The GNN learns cross-channel interactions on a 6-node fully-connected graph.
Skip connection feeds raw channel features directly to the readout,
so the model always has the linear baseline available.

Architecture:
  1. Per-channel feature aggregation (preprocessing, no learned params)
  2. Channel encoder (Linear -> hidden)
  3. Cross-channel transformer (learns co-severance patterns)
  4. Skip connection: concat(transformer_out, raw_features)
  5. Readout MLP -> hazard score
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CrossChannelAttention


class ChannelNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Channel feature dim: computed in forward from pre-aggregated features
        # 6 channels x feature_dim per channel
        ch_feat_dim = config.get("channel_feat_dim", 16)
        hidden_dim = config["hidden_dim"]
        dropout = config["dropout"]

        # Channel encoder
        self.channel_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Cross-channel transformer
        self.cross_channel = CrossChannelAttention(
            hidden_dim,
            num_heads=config["cross_channel_heads"],
            num_layers=config.get("cross_channel_layers", 2),
            dropout=dropout,
        )

        # Channel attention for pooling
        self.channel_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Skip connection: raw features go directly to readout
        # Final MLP takes: transformer_pooled (hidden) + raw_pooled (ch_feat_dim*6)
        skip_dim = ch_feat_dim * 6
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        """
        Args:
            data: Batch with channel_features: (B*6, ch_feat_dim)
                  or reshaped to (B, 6, ch_feat_dim) via batch index

        Returns:
            (B,) hazard scores
        """
        # channel_features: (B, 6, ch_feat_dim) — pre-aggregated
        ch_feats = data.channel_features  # set by dataset
        B = ch_feats.shape[0]

        # Raw skip: flatten all 6 channels
        skip = ch_feats.reshape(B, -1)  # (B, 6*ch_feat_dim)

        # Encode channels
        h = self.channel_encoder(ch_feats)  # (B, 6, hidden_dim)

        # Cross-channel transformer
        h = self.cross_channel(h)  # (B, 6, hidden_dim)

        # Attention-weighted pooling over channels
        scores = self.channel_attn(h).squeeze(-1)  # (B, 6)
        weights = F.softmax(scores, dim=-1)  # (B, 6)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, hidden_dim)

        # Concat with skip connection
        combined = torch.cat([pooled, skip], dim=-1)  # (B, hidden_dim + 6*ch_feat_dim)

        return self.readout(combined).squeeze(-1)  # (B,)
