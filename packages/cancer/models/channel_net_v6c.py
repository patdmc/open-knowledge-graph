"""
ChannelNetV6c — 8 channels + MSI-aware clinical features.

Same transformer architecture as V6 but clinical encoder takes 4 inputs:
  age, sex, msi_score (continuous), msi_high (binary)

The model can learn to discount chromatin channel mutations when MSI is high.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.channel_dataset_v6 import CHANNEL_FEAT_DIM_V6, N_CHANNELS_V6, N_TIERS_V6


class CrossChannelTransformerV6c(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.channel_pos = nn.Embedding(N_CHANNELS_V6, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h_channels):
        B = h_channels.shape[0]
        pos = self.channel_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h = self.transformer(h_channels + pos)
        scores = self.attn_pool(h).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(1)
        return pooled


class TierTransformerV6c(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.tier_pos = nn.Embedding(N_TIERS_V6, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h_tiers):
        B = h_tiers.shape[0]
        pos = self.tier_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h = self.transformer(h_tiers + pos)
        scores = self.attn_pool(h).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(1)
        return pooled


class ChannelNetV6c(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.get("channel_feat_dim", CHANNEL_FEAT_DIM_V6)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        # Channel path: 8 channels
        self.ch_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout),
        )
        self.ch_transformer = CrossChannelTransformerV6c(
            hidden, config["cross_channel_heads"],
            config.get("cross_channel_layers", 2), dropout,
        )

        # Tier path: 4 tiers
        self.tier_encoder = nn.Sequential(
            nn.Linear(5, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout),
        )
        self.tier_transformer = TierTransformerV6c(hidden, 2, dropout)

        # Cancer type embedding
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # Clinical encoder: age, sex, msi_score, msi_high, tmb = 5 inputs
        self.clinical_encoder = nn.Sequential(
            nn.Linear(5, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )

        # Skip: raw features
        skip_raw = N_CHANNELS_V6 * feat_dim + N_TIERS_V6 * 5
        total_dim = hidden + hidden + hidden // 2 + hidden // 4 + skip_raw

        self.readout = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        ch_feats = batch["channel_features"]
        tier_feats = batch["tier_features"]
        ct_idx = batch["cancer_type_idx"]
        age_val = batch["age"]
        sex_val = batch["sex"]
        msi_score = batch["msi_score"]
        msi_high = batch["msi_high"]
        tmb_val = batch["tmb"]
        B = ch_feats.shape[0]

        # Channel path
        h_ch = self.ch_encoder(ch_feats)
        ch_pooled = self.ch_transformer(h_ch)

        # Tier path
        h_tier = self.tier_encoder(tier_feats)
        tier_pooled = self.tier_transformer(h_tier)

        # Cancer type
        cancer = self.cancer_embed(ct_idx)

        # Clinical: 5 features
        clinical_input = torch.stack([
            age_val, sex_val.float(), msi_score, msi_high.float(), tmb_val
        ], dim=-1)
        clinical = self.clinical_encoder(clinical_input)

        # Skip
        raw = torch.cat([ch_feats.reshape(B, -1), tier_feats.reshape(B, -1)], dim=-1)

        combined = torch.cat([ch_pooled, tier_pooled, cancer, clinical, raw], dim=-1)
        return self.readout(combined).squeeze(-1)
