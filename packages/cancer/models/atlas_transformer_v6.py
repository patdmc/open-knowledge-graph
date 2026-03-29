"""
AtlasTransformer V6 — Hierarchical mutation-level transformer.

The computation follows the equivalence class hierarchy:

  Level 1: INTRA-BLOCK ATTENTION
    Mutations within the same sub-pathway block attend to each other.
    Block = PPI community within a channel (e.g., BRCA1-RAD51C-PALB2 in DDR).
    This is where mutation-mutation interactions within a functional unit are learned.
    Pairwise graph features (PPI distance, co-occurrence) bias attention.

  Level 2: BLOCK → CHANNEL POOLING
    Block representations are aggregated into channel-level damage scores.
    Attention-weighted pooling: not all blocks contribute equally.
    A channel with one deeply damaged block differs from uniformly light damage.

  Level 3: CROSS-CHANNEL ATTENTION
    Channel tokens attend to each other. DDR damage + Cell Cycle damage
    means something different than DDR damage alone. This captures the
    multi-pathway interaction signal.

  Level 4: CANCER TYPE CONDITIONS EACH LEVEL
    - Intra-block: CT-specific attention bias (DDR mutations interact
      differently in breast vs lung)
    - Block pooling: CT-specific block importance weights
    - Cross-channel: CT-specific channel interaction patterns

Node features are mutation-level only:
  - log_hr, ci_width, tier, protein_position, is_harmful, is_protective
  - biallelic_status, expression_context, gof_lof
  - NO is_hub, NO channel_onehot, NO channel_pos_embed

Gene identity flows through:
  - Block assignments (which sub-pathway community is this gene in)
  - Pairwise edge features (how mutations relate on the graph)
  - Channel membership (implicit via block → channel mapping)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================================
# Level 1: Intra-block attention
# =========================================================================

class IntraBlockAttention(nn.Module):
    """Mutations within the same block attend to each other.

    Uses pairwise graph features as attention bias. Cancer type conditions
    which mutation interactions matter for this tissue.
    """

    def __init__(self, hidden, n_heads, n_cancer_types, edge_feat_dim, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # Graph pairwise bias
        self.edge_bias_net = nn.Sequential(
            nn.Linear(edge_feat_dim, n_heads * 2),
            nn.ELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        # CT conditions intra-block attention
        self.ct_key_mod = nn.Embedding(n_cancer_types, hidden)
        self.ct_attn_bias = nn.Embedding(n_cancer_types, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, cancer_type, edge_features, block_mask, pad_mask):
        """
        x: (B, N, hidden) — mutation embeddings
        cancer_type: (B,) long
        edge_features: (B, N, N, edge_feat_dim)
        block_mask: (B, N, N) bool — True means DIFFERENT block (don't attend)
        pad_mask: (B, N) bool — True means padding
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # CT modulates keys
        ct_key = self.ct_key_mod(cancer_type).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + ct_key

        # QK attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # CT global bias
        ct_bias = self.ct_attn_bias(cancer_type)  # (B, n_heads)
        attn = attn + ct_bias[:, :, None, None]

        # Pairwise graph bias
        edge_bias = self.edge_bias_net(edge_features).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + edge_bias

        # Block-sparse: mask cross-block attention
        attn = attn.masked_fill(block_mask.unsqueeze(1), float('-inf'))

        # Pad mask
        full_pad = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)  # (B, N, N)
        attn = attn.masked_fill(full_pad.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden)
        return self.out_proj(out), attn.mean(dim=1)  # return avg attn for analysis


class IntraBlockLayer(nn.Module):
    def __init__(self, hidden, n_heads, n_cancer_types, edge_feat_dim, dropout=0.1):
        super().__init__()
        self.attn = IntraBlockAttention(
            hidden, n_heads, n_cancer_types, edge_feat_dim, dropout
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )

    def forward(self, x, cancer_type, edge_features, block_mask, pad_mask):
        attended, attn_w = self.attn(
            self.norm1(x), cancer_type, edge_features, block_mask, pad_mask
        )
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


# =========================================================================
# Level 2: Block → Channel pooling
# =========================================================================

class BlockToChannelPool(nn.Module):
    """Aggregate mutation embeddings into channel-level damage tokens.

    For each channel, gather all mutations in that channel's blocks,
    apply attention-weighted pooling (not all mutations contribute equally),
    and produce a channel token.

    Cancer type conditions which blocks/mutations matter most.
    """

    def __init__(self, hidden, n_cancer_types, n_channels, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels

        # Learned block importance within each channel (per CT)
        self.block_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        # CT conditions which mutations matter for this channel
        self.ct_channel_mod = nn.Embedding(n_cancer_types, n_channels)

        # Project pooled mutations → channel token
        self.channel_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, cancer_type, channel_ids, pad_mask):
        """
        x: (B, N, hidden) — mutation embeddings after intra-block attention
        cancer_type: (B,) long
        channel_ids: (B, N) long — channel index per mutation
        pad_mask: (B, N) bool — True = padding

        Returns:
            channel_tokens: (B, C, hidden) — one token per channel
            channel_active: (B, C) bool — True if channel has mutations
        """
        B, N, H = x.shape
        C = self.n_channels

        # Attention weights per mutation
        gate = self.block_gate(x).squeeze(-1)  # (B, N)
        gate = gate.masked_fill(pad_mask, float('-inf'))

        # CT modulation per channel
        ct_mod = self.ct_channel_mod(cancer_type)  # (B, C)

        channel_tokens = torch.zeros(B, C, H, device=x.device)
        channel_active = torch.zeros(B, C, dtype=torch.bool, device=x.device)

        for ch_id in range(C):
            ch_mask = (channel_ids == ch_id) & (~pad_mask)  # (B, N)
            has_ch = ch_mask.any(dim=1)  # (B,)
            if not has_ch.any():
                continue

            channel_active[:, ch_id] = has_ch

            # Attention-weighted pooling within this channel
            ch_gate = gate.clone()
            ch_gate[~ch_mask] = float('-inf')
            ch_weights = F.softmax(ch_gate, dim=1).nan_to_num(0.0)  # (B, N)

            # CT-scaled importance
            ch_weights = ch_weights * (1 + ct_mod[:, ch_id:ch_id+1] * 0.1)

            pooled = (x * ch_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
            channel_tokens[:, ch_id] = self.channel_proj(pooled)

        return channel_tokens, channel_active


# =========================================================================
# Level 3: Cross-channel attention
# =========================================================================

class CrossChannelAttention(nn.Module):
    """Channel tokens attend to each other.

    DDR damage + Cell Cycle damage has a different survival impact than
    either alone. This captures multi-pathway interaction patterns.

    Cancer type conditions which channel interactions matter.
    """

    def __init__(self, hidden, n_heads, n_cancer_types, n_channels, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # CT conditions cross-channel attention
        self.ct_channel_bias = nn.Embedding(n_cancer_types, n_channels * n_channels)
        self.ct_key_mod = nn.Embedding(n_cancer_types, hidden)

        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.n_channels = n_channels

    def forward(self, channel_tokens, cancer_type, channel_active):
        """
        channel_tokens: (B, C, hidden)
        cancer_type: (B,) long
        channel_active: (B, C) bool

        Returns: (B, C, hidden) — enriched channel tokens
        """
        B, C, _ = channel_tokens.shape
        normed = self.norm(channel_tokens)

        q = self.q_proj(normed).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        # CT modulates keys
        ct_key = self.ct_key_mod(cancer_type).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + ct_key

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, C, C)

        # CT-specific channel-channel bias
        ct_bias = self.ct_channel_bias(cancer_type).view(B, C, C)  # (B, C, C)
        # Average across heads
        attn = attn + ct_bias.unsqueeze(1) * 0.1

        # Mask inactive channels
        inactive = ~channel_active  # (B, C)
        attn = attn.masked_fill(inactive.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, C, self.hidden)
        out = self.out_proj(out)

        channel_tokens = channel_tokens + out
        channel_tokens = channel_tokens + self.ffn(self.norm2(channel_tokens))

        return channel_tokens


# =========================================================================
# Full model
# =========================================================================

class AtlasTransformerV6(nn.Module):
    """Hierarchical mutation-level transformer.

    Computation follows the equivalence class hierarchy:
      mutations → blocks → channels → patient score

    Each level is conditioned on cancer type. Gene identity flows through
    block assignments and pairwise graph structure, not through node features.

    Config must include:
        node_feat_dim: int — mutation-level features (18)
        hidden_dim: int
        edge_feat_dim: int — pairwise graph features
        n_channels: int — number of channels (8)
        n_blocks: int — total number of blocks
        n_cancer_types: int
        n_heads: int
        n_intra_layers: int — layers of intra-block attention
    """

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config.get('dropout', 0.1)
        n_cancer_types = config.get('n_cancer_types', 80)
        n_heads = config.get('n_heads', 4)
        n_intra_layers = config.get('n_intra_layers', 2)
        edge_feat_dim = config['edge_feat_dim']
        n_channels = config['n_channels']

        self.hidden = hidden
        self.n_channels = n_channels

        # Node embedding (mutation-level features only)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Per-CT sign coefficient on atlas features (log_hr, ci_width)
        # Initialized to 1.0 so default behavior is unchanged.
        # Can learn negative values for CTs where atlas HR direction is inverted.
        self.ct_sign = nn.Embedding(n_cancer_types, 1)
        nn.init.ones_(self.ct_sign.weight)

        # Age conditions each mutation
        self.age_mod = nn.Sequential(nn.Linear(1, hidden), nn.ELU())

        # Level 1: Intra-block attention (with graph-structured bias)
        self.intra_block_layers = nn.ModuleList([
            IntraBlockLayer(hidden, n_heads, n_cancer_types, edge_feat_dim, dropout)
            for _ in range(n_intra_layers)
        ])

        # Level 2: Block → Channel pooling
        self.block_to_channel = BlockToChannelPool(
            hidden, n_cancer_types, n_channels, dropout
        )

        # Level 3: Cross-channel attention
        self.cross_channel = CrossChannelAttention(
            hidden, n_heads, n_cancer_types, n_channels, dropout
        )

        # Patient readout
        self.channel_pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )
        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())
        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, hidden // 4),
            nn.ELU(),
        )

        # Sign × Magnitude decomposition
        # Per-mutation sign: predicts harmful (+1) or protective (-1) in this context
        # Supervised by atlas log_hr sign — 391K labeled examples
        self.sign_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Per-mutation magnitude: how much does this mutation matter (always positive)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),  # ensure positive
        )

        readout_in = hidden + hidden // 4 + hidden // 4 + hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _backbone(self, node_features, node_mask, cancer_type, clinical,
                  edge_features, block_ids, channel_ids):
        """Run the shared backbone: encode → intra-block → channel pool → cross-channel.

        Returns:
            node_hidden: (B, N, hidden) — per-node after intra-block attention
            patient_embed: (B, hidden) — pooled patient-level representation
            channel_tokens: (B, C, hidden)
            channel_active: (B, C) bool
        """
        B, N, _ = node_features.shape
        pad_mask = (node_mask == 0)

        # === Per-CT sign modulation of atlas features ===
        ct_s = self.ct_sign(cancer_type)  # (B, 1)
        ct_s_broad = ct_s.squeeze(-1).unsqueeze(1)  # (B, 1)
        sign_neg = (ct_s_broad < 0).float()  # (B, 1)

        modified_log_hr = node_features[:, :, 0:1] * ct_s_broad.unsqueeze(-1)
        unchanged_1_10 = node_features[:, :, 1:10]
        modified_harmful = (node_features[:, :, 10:11] * (1 - sign_neg.unsqueeze(-1))
                            + node_features[:, :, 11:12] * sign_neg.unsqueeze(-1))
        modified_protective = (node_features[:, :, 11:12] * (1 - sign_neg.unsqueeze(-1))
                               + node_features[:, :, 10:11] * sign_neg.unsqueeze(-1))
        rest = node_features[:, :, 12:]
        node_features = torch.cat([
            modified_log_hr, unchanged_1_10,
            modified_harmful, modified_protective, rest,
        ], dim=-1)

        # === Encode mutations ===
        x = self.node_encoder(node_features)

        # Age modulation
        age = clinical[:, 0:1]
        age_scale = self.age_mod(age)
        x = x * (1 + age_scale.unsqueeze(1) * 0.1)

        # === Level 1: Intra-block attention ===
        block_mask = (block_ids.unsqueeze(1) != block_ids.unsqueeze(2))

        for layer in self.intra_block_layers:
            x, attn_w = layer(x, cancer_type, edge_features, block_mask, pad_mask)

        node_hidden = x
        last_attn = attn_w  # (B, N, N) from last intra-block layer

        # === Level 2: Block → Channel pooling ===
        channel_tokens, channel_active = self.block_to_channel(
            x, cancer_type, channel_ids, pad_mask
        )

        # === Level 3: Cross-channel attention ===
        n_active = channel_active.float().sum(dim=1)
        has_multi = (n_active >= 2)
        if has_multi.any():
            channel_tokens = self.cross_channel(
                channel_tokens, cancer_type, channel_active
            )

        # === Pool channels → patient embedding ===
        ch_attn = self.channel_pool_attn(channel_tokens).squeeze(-1)
        ch_attn = ch_attn.masked_fill(~channel_active, float('-inf'))
        ch_attn = F.softmax(ch_attn, dim=1).nan_to_num(0.0)
        patient_embed = (channel_tokens * ch_attn.unsqueeze(-1)).sum(dim=1)

        return node_hidden, patient_embed, channel_tokens, channel_active, last_attn

    def encode(self, node_features, node_mask, cancer_type, clinical,
               edge_features, block_ids, channel_ids):
        """Return backbone representations without the survival readout.

        Returns:
            node_hidden: (B, N, hidden) — per-node after intra-block attention
            patient_embed: (B, hidden) — pooled patient-level embedding
        """
        node_hidden, patient_embed, _, _, _ = self._backbone(
            node_features, node_mask, cancer_type, clinical,
            edge_features, block_ids, channel_ids,
        )
        return node_hidden, patient_embed

    def encode_with_attention(self, node_features, node_mask, cancer_type, clinical,
                              edge_features, block_ids, channel_ids):
        """Like encode(), but also returns the last intra-block attention weights.

        Returns:
            node_hidden: (B, N, hidden)
            patient_embed: (B, hidden)
            attn_w: (B, N, N) — avg over heads from last intra-block layer
        """
        node_hidden, patient_embed, _, _, attn_w = self._backbone(
            node_features, node_mask, cancer_type, clinical,
            edge_features, block_ids, channel_ids,
        )
        return node_hidden, patient_embed, attn_w

    def forward(self, node_features, node_mask, cancer_type, clinical,
                atlas_sum, edge_features, block_ids, channel_ids):
        """
        Args:
            node_features: (B, N, node_feat_dim) — per-mutation features
            node_mask: (B, N) — 1 real, 0 pad
            cancer_type: (B,) long
            clinical: (B, 5) — [age_z, sex, norm_year, year_confidence, is_mature]
            atlas_sum: (B, 1) — additive baseline
            edge_features: (B, N, N, edge_feat_dim) — pairwise graph structure
            block_ids: (B, N) long — block assignment per mutation
            channel_ids: (B, N) long — channel assignment per mutation
        """
        node_hidden, patient_embed, channel_tokens, channel_active, _ = self._backbone(
            node_features, node_mask, cancer_type, clinical,
            edge_features, block_ids, channel_ids,
        )

        # === Sign × Magnitude decomposition ===
        # Per-mutation sign: tanh → [-1, +1]
        sign_logits = self.sign_head(node_hidden).squeeze(-1)   # (B, N)
        sign_pred = torch.tanh(sign_logits)                      # (B, N) in [-1, +1]

        # Per-mutation magnitude: softplus → [0, ∞)
        magnitude = self.magnitude_head(node_hidden).squeeze(-1)  # (B, N)

        # Per-mutation hazard contribution = sign × magnitude
        mutation_hazard = sign_pred * magnitude                   # (B, N)

        # Mask out padding and sum to get patient hazard component
        mutation_hazard = mutation_hazard * node_mask              # zero out pads
        hazard_from_mutations = mutation_hazard.sum(dim=1, keepdim=True)  # (B, 1)

        # === Patient-level readout (context: sex, atlas skip, temporal) ===
        ct_s = self.ct_sign(cancer_type)  # (B, 1)
        sex = clinical[:, 1:2]
        sex_feat = self.sex_encoder(sex)
        atlas_feat = self.atlas_skip(atlas_sum * ct_s)
        temporal = clinical[:, 2:5]
        temporal_feat = self.temporal_encoder(temporal)

        out = torch.cat([patient_embed, sex_feat, atlas_feat, temporal_feat], dim=-1)
        context_hazard = self.readout(out)  # (B, 1) residual from context

        # Final hazard = mutation contributions + context residual
        hazard = hazard_from_mutations + context_hazard

        # Cache for auxiliary losses
        self._last_sign_logits = sign_logits      # (B, N) — for sign supervision
        self._last_sign_pred = sign_pred           # (B, N)
        self._last_magnitude = magnitude           # (B, N)
        self._last_mutation_hazard = mutation_hazard  # (B, N)
        self._last_mutation_embeddings = node_hidden
        self._last_channel_tokens = channel_tokens
        self._last_channel_active = channel_active

        return hazard
