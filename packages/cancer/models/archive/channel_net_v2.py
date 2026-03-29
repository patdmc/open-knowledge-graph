"""
ChannelNetV2 — Enhanced model with:
  - 6-channel features + 3-tier hierarchy features
  - Cancer type embedding
  - Clinical covariates (age, sex)
  - Skip connections at every level
  - Two-level cross-attention: within-tier then cross-tier

Architecture:
  1. Encode 6 channels -> cross-channel transformer -> 6 channel embeddings
  2. Encode 3 tiers -> cross-tier transformer -> 3 tier embeddings
  3. Cancer type embedding + clinical covariates
  4. Skip: concat(channel_pooled, tier_pooled, cancer_embed, clinical, raw_features)
  5. Readout MLP -> hazard
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CrossChannelAttention


class ChannelNetV2(nn.Module):

    def __init__(self, config):
        super().__init__()
        ch_feat_dim = config.get("channel_feat_dim", 17)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        # --- Channel path (6 channels) ---
        self.ch_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.ch_transformer = CrossChannelAttention(
            hidden, num_heads=config["cross_channel_heads"],
            num_layers=config.get("cross_channel_layers", 2),
            dropout=dropout,
        )
        self.ch_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Tier path (3 tiers) ---
        self.tier_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.tier_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=config.get("cross_channel_layers", 2),
        )
        self.tier_pos = nn.Embedding(3, hidden)
        self.tier_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Cancer type embedding ---
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # --- Clinical encoder ---
        self.clinical_encoder = nn.Sequential(
            nn.Linear(2, hidden // 4),  # age + sex
            nn.ReLU(),
        )

        # --- Skip connection + readout ---
        # Inputs: ch_pooled(hidden) + tier_pooled(hidden) + cancer(hidden//2)
        #         + clinical(hidden//4) + raw_ch(6*ch_feat_dim) + raw_tier(3*ch_feat_dim)
        skip_raw = 6 * ch_feat_dim + 3 * ch_feat_dim
        total_dim = hidden + hidden + hidden // 2 + hidden // 4 + skip_raw

        self.readout = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        ch_feats = batch["channel_features"]   # (B, 6, 17)
        tier_feats = batch["tier_features"]     # (B, 3, 17)
        ct_idx = batch["cancer_type_idx"]       # (B,)
        age = batch["age"]                      # (B,)
        sex = batch["sex"]                      # (B,)
        B = ch_feats.shape[0]

        # --- Channel path ---
        h_ch = self.ch_encoder(ch_feats)                    # (B, 6, H)
        h_ch = self.ch_transformer(h_ch)                    # (B, 6, H)
        ch_scores = self.ch_attn(h_ch).squeeze(-1)          # (B, 6)
        ch_weights = F.softmax(ch_scores, dim=-1)
        ch_pooled = (ch_weights.unsqueeze(-1) * h_ch).sum(1)  # (B, H)

        # --- Tier path ---
        h_tier = self.tier_encoder(tier_feats)              # (B, 3, H)
        pos = self.tier_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h_tier = self.tier_transformer(h_tier + pos)        # (B, 3, H)
        tier_scores = self.tier_attn(h_tier).squeeze(-1)    # (B, 3)
        tier_weights = F.softmax(tier_scores, dim=-1)
        tier_pooled = (tier_weights.unsqueeze(-1) * h_tier).sum(1)  # (B, H)

        # --- Cancer type ---
        cancer = self.cancer_embed(ct_idx)                  # (B, H//2)

        # --- Clinical ---
        clinical = self.clinical_encoder(
            torch.stack([age, sex.float()], dim=-1)         # (B, 2)
        )                                                    # (B, H//4)

        # --- Skip: raw features ---
        raw_ch = ch_feats.reshape(B, -1)                    # (B, 6*17)
        raw_tier = tier_feats.reshape(B, -1)                # (B, 3*17)

        # --- Concat everything ---
        combined = torch.cat([
            ch_pooled, tier_pooled, cancer, clinical, raw_ch, raw_tier
        ], dim=-1)

        return self.readout(combined).squeeze(-1)           # (B,)
