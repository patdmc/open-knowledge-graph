"""
ChannelNetV2_MSI — V2 architecture (6 channels) + MSI/TMB clinical covariates.

Ablation to test whether the 8-channel expansion adds signal beyond MSI/TMB.
Same as V2 but clinical encoder takes 5 inputs: age, sex, msi_score, msi_high, tmb.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CrossChannelAttention


class ChannelNetV2MSI(nn.Module):

    def __init__(self, config):
        super().__init__()
        ch_feat_dim = config.get("channel_feat_dim", 17)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        # Channel path (6 channels)
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

        # Tier path (3 tiers)
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

        # Cancer type embedding
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # Clinical encoder: age, sex, msi_score, msi_high, tmb = 5 inputs
        self.clinical_encoder = nn.Sequential(
            nn.Linear(5, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )

        # Skip + readout
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
        ch_feats = batch["channel_features"]   # (B, 6, feat_dim)
        tier_feats = batch["tier_features"]     # (B, 3, feat_dim)
        ct_idx = batch["cancer_type_idx"]
        age_val = batch["age"]
        sex_val = batch["sex"]
        msi_score = batch["msi_score"]
        msi_high = batch["msi_high"]
        tmb_val = batch["tmb"]
        B = ch_feats.shape[0]

        # Channel path
        h_ch = self.ch_encoder(ch_feats)
        h_ch = self.ch_transformer(h_ch)
        ch_scores = self.ch_attn(h_ch).squeeze(-1)
        ch_weights = F.softmax(ch_scores, dim=-1)
        ch_pooled = (ch_weights.unsqueeze(-1) * h_ch).sum(1)

        # Tier path
        h_tier = self.tier_encoder(tier_feats)
        pos = self.tier_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h_tier = self.tier_transformer(h_tier + pos)
        tier_scores = self.tier_attn(h_tier).squeeze(-1)
        tier_weights = F.softmax(tier_scores, dim=-1)
        tier_pooled = (tier_weights.unsqueeze(-1) * h_tier).sum(1)

        # Cancer type
        cancer = self.cancer_embed(ct_idx)

        # Clinical: 5 features
        clinical_input = torch.stack([
            age_val, sex_val.float(), msi_score, msi_high.float(), tmb_val
        ], dim=-1)
        clinical = self.clinical_encoder(clinical_input)

        # Skip
        raw_ch = ch_feats.reshape(B, -1)
        raw_tier = tier_feats.reshape(B, -1)

        combined = torch.cat([
            ch_pooled, tier_pooled, cancer, clinical, raw_ch, raw_tier
        ], dim=-1)

        return self.readout(combined).squeeze(-1)
