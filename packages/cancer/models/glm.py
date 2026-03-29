"""
Graph Language Model (GLM) — dense subspaces with sparse signed routing.

Architecture:
  Level 1: Dense intra-block attention (same as AtlasTransformerV6)
    Mutations within a sub-pathway block attend densely to each other.
    Graph pairwise features bias attention.

  Level 2: Mutation → Block pooling (NEW)
    Attention-weighted aggregation of mutations into 101 block tokens.
    Each block is a sub-pathway community from PPI-based detection.

  Level 3: BLOCK-LEVEL SPARSE SIGNED ROUTING (GLM core)
    101×101 routing with FIXED sparsity from PPI cross-channel edges.
    328 unique cross-block pairs (656 directed edges, 93.6% sparse).
    Each edge gets a CT-specific learned signed weight:
      R[i,j] > 0 → block i escalates through block j (worse prognosis)
      R[i,j] < 0 → block i modulated by block j (better prognosis)
    No sparsity learning needed — PPI defines the skeleton.

  Level 4: Block → Channel aggregation + escalation readout
    Block tokens aggregated to channels. Escalation classifier conditions
    the final survival prediction.

Key insight: 101×101 block routing captures sub-pathway interactions that
8×8 channel routing misses. BRCA1-block routing to RAD51C-block is different
from BRCA1-block routing to CHEK2-block, even though all three are DDR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .atlas_transformer_v6 import IntraBlockLayer


# =========================================================================
# Level 2: Mutation → Block pooling
# =========================================================================

class MutationToBlockPool(nn.Module):
    """Pool mutation embeddings into block-level tokens.

    Each block gets one token from attention-weighted aggregation of its
    mutations. CT conditions which mutations matter most per block.
    """

    def __init__(self, hidden, n_cancer_types, n_blocks, dropout=0.1):
        super().__init__()
        self.n_blocks = n_blocks

        self.block_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        # CT-specific block importance
        self.ct_block_mod = nn.Embedding(n_cancer_types, n_blocks)
        nn.init.zeros_(self.ct_block_mod.weight)

        self.block_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, cancer_type, block_ids, pad_mask):
        """
        Args:
            x: (B, N, hidden) — mutation embeddings after intra-block attention
            cancer_type: (B,) long
            block_ids: (B, N) long — block index per mutation
            pad_mask: (B, N) bool — True = padding

        Returns:
            block_tokens: (B, n_blocks, hidden)
            block_active: (B, n_blocks) bool
        """
        B, N, H = x.shape
        NB = self.n_blocks

        gate = self.block_gate(x).squeeze(-1)  # (B, N)
        gate = gate.masked_fill(pad_mask, float('-inf'))

        ct_mod = self.ct_block_mod(cancer_type)  # (B, NB)

        block_tokens = torch.zeros(B, NB, H, device=x.device)
        block_active = torch.zeros(B, NB, dtype=torch.bool, device=x.device)

        for bid in range(NB):
            b_mask = (block_ids == bid) & (~pad_mask)  # (B, N)
            has_b = b_mask.any(dim=1)  # (B,)
            if not has_b.any():
                continue

            block_active[:, bid] = has_b

            b_gate = gate.clone()
            b_gate[~b_mask] = float('-inf')
            b_weights = F.softmax(b_gate, dim=1).nan_to_num(0.0)  # (B, N)

            b_weights = b_weights * (1 + ct_mod[:, bid:bid+1] * 0.1)

            pooled = (x * b_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
            block_tokens[:, bid] = self.block_proj(pooled)

        return block_tokens, block_active


# =========================================================================
# Level 3: Block-level sparse signed routing
# =========================================================================

class BlockLevelRouter(nn.Module):
    """Sparse signed routing between blocks using PPI-defined edges.

    The sparsity pattern is FIXED from cross-channel PPI edges (328 pairs,
    656 directed). Only the signed magnitudes are learned, CT-specific.

    Message passing on the sparse graph:
      h_i' = h_i + gate * Σ_{j∈N(i)} w[i,j] * W_route(h_j)
    where N(i) is the PPI-defined neighborhood of block i.
    """

    def __init__(self, hidden, n_blocks, n_cancer_types, edge_index,
                 edge_scores=None, n_route_layers=2, dropout=0.1):
        """
        Args:
            hidden: embedding dimension
            n_blocks: total number of blocks (101)
            n_cancer_types: number of cancer types
            edge_index: (2, E) numpy array — directed block pairs from PPI
            edge_scores: (E,) optional PPI confidence scores for initialization
            n_route_layers: number of post-routing FFN layers
        """
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.n_edges = edge_index.shape[1]

        # Register the fixed sparsity pattern as a buffer (not a parameter)
        self.register_buffer('edge_src', torch.tensor(edge_index[0], dtype=torch.long))
        self.register_buffer('edge_dst', torch.tensor(edge_index[1], dtype=torch.long))

        # Per-edge global structure weight (learned, but PPI-initialized)
        # Higher PPI score → stronger initial connection
        init_vals = torch.ones(self.n_edges) * 0.5
        if edge_scores is not None:
            scores_t = torch.tensor(edge_scores, dtype=torch.float32)
            init_vals = scores_t.clamp(0.1, 1.0)
        self.edge_structure = nn.Parameter(init_vals)

        # CT-specific signed weights per edge
        self.ct_edge_weights = nn.Embedding(n_cancer_types, self.n_edges)
        nn.init.normal_(self.ct_edge_weights.weight, mean=0.0, std=0.1)

        # Message transform
        self.route_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )

        # Gate: how much of routed message to accept
        self.route_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )

        # Post-routing FFN layers
        self.route_layers = nn.ModuleList()
        for _ in range(n_route_layers):
            self.route_layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden),
                'ffn': nn.Sequential(
                    nn.Linear(hidden, hidden * 2),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, hidden),
                    nn.Dropout(dropout),
                ),
            }))

        self.dropout = nn.Dropout(dropout)

    def get_edge_weights(self, cancer_type):
        """Compute per-edge signed weights for this batch.

        Returns:
            weights: (B, E) — signed routing weight per edge
        """
        structure = torch.sigmoid(self.edge_structure)  # (E,)
        ct_w = self.ct_edge_weights(cancer_type)  # (B, E)
        return structure.unsqueeze(0) * ct_w  # (B, E)

    def forward(self, block_tokens, cancer_type, block_active):
        """
        Args:
            block_tokens: (B, NB, hidden)
            cancer_type: (B,) long
            block_active: (B, NB) bool

        Returns:
            routed_tokens: (B, NB, hidden)
            edge_weights: (B, E) — for analysis/regularization
        """
        B, NB, H = block_tokens.shape

        edge_w = self.get_edge_weights(cancer_type)  # (B, E)

        # Mask edges where source or destination block is inactive
        src_active = block_active[:, self.edge_src]  # (B, E)
        dst_active = block_active[:, self.edge_dst]  # (B, E)
        edge_w = edge_w * src_active.float() * dst_active.float()

        # Project source block tokens for routing
        h_route = self.route_proj(block_tokens)  # (B, NB, H)

        # Sparse message passing: gather source embeddings, weight, scatter to dst
        src_embeds = h_route[:, self.edge_src, :]  # (B, E, H)
        weighted_msgs = edge_w.unsqueeze(-1) * src_embeds  # (B, E, H)

        # Scatter-add to destination blocks
        messages = torch.zeros_like(block_tokens)  # (B, NB, H)
        dst_expanded = self.edge_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        messages.scatter_add_(1, dst_expanded, weighted_msgs)

        # Gated update
        gate_input = torch.cat([block_tokens, messages], dim=-1)  # (B, NB, 2H)
        gate = self.route_gate(gate_input)  # (B, NB, H)
        block_tokens = block_tokens + gate * self.dropout(messages)

        # Post-routing FFN
        for layer in self.route_layers:
            normed = layer['norm'](block_tokens)
            block_tokens = block_tokens + layer['ffn'](normed)

        return block_tokens, edge_w

    def routing_loss(self, lambda_sign=0.01):
        """Sign consistency loss across the batch."""
        # Structure regularization: keep edge weights from exploding
        structure = torch.sigmoid(self.edge_structure)
        return lambda_sign * (structure - 0.5).abs().mean()

    def routing_summary(self):
        """Human-readable routing summary."""
        with torch.no_grad():
            structure = torch.sigmoid(self.edge_structure)
            active = (structure > 0.5).sum().item()
            return {
                'n_edges': self.n_edges,
                'n_active': int(active),
                'n_suppressed': self.n_edges - int(active),
                'mean_structure': float(structure.mean()),
                'edge_src': self.edge_src.cpu().numpy(),
                'edge_dst': self.edge_dst.cpu().numpy(),
                'edge_strength': structure.cpu().numpy(),
            }


# =========================================================================
# Level 4a: Block → Channel aggregation (post-routing)
# =========================================================================

class BlockToChannelAggregate(nn.Module):
    """Aggregate routed block tokens into channel-level tokens.

    Each channel's token is the attention-weighted sum of its block tokens.
    """

    def __init__(self, hidden, n_channels, n_blocks, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_blocks = n_blocks

        self.agg_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, block_tokens, block_active, block_to_channel_map):
        """
        Args:
            block_tokens: (B, NB, hidden)
            block_active: (B, NB) bool
            block_to_channel_map: (NB,) long — channel_id for each block

        Returns:
            channel_tokens: (B, C, hidden)
            channel_active: (B, C) bool
        """
        B, NB, H = block_tokens.shape
        C = self.n_channels

        gate = self.agg_gate(block_tokens).squeeze(-1)  # (B, NB)
        gate = gate.masked_fill(~block_active, float('-inf'))

        channel_tokens = torch.zeros(B, C, H, device=block_tokens.device)
        channel_active = torch.zeros(B, C, dtype=torch.bool, device=block_tokens.device)

        for ch_id in range(C):
            ch_blocks = (block_to_channel_map == ch_id)  # (NB,)
            if not ch_blocks.any():
                continue

            ch_block_active = block_active & ch_blocks.unsqueeze(0)  # (B, NB)
            has_ch = ch_block_active.any(dim=1)  # (B,)
            if not has_ch.any():
                continue

            channel_active[:, ch_id] = has_ch

            ch_gate = gate.clone()
            ch_gate[~ch_block_active] = float('-inf')
            ch_weights = F.softmax(ch_gate, dim=1).nan_to_num(0.0)  # (B, NB)

            pooled = (block_tokens * ch_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
            channel_tokens[:, ch_id] = pooled

        return channel_tokens, channel_active


# =========================================================================
# Escalation classifier
# =========================================================================

class EscalationClassifier(nn.Module):
    """Classifies the damage pattern as DFS-like, BFS-like, or mixed.

    Uses channel tokens + routing statistics from block-level routing.
    """

    def __init__(self, hidden, n_channels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden + 4, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 3),  # DFS, BFS, mixed
        )

    def forward(self, channel_tokens, channel_active, edge_weights):
        """
        Args:
            channel_tokens: (B, C, hidden)
            channel_active: (B, C) bool
            edge_weights: (B, E) — block-level routing weights

        Returns:
            escalation_logits: (B, 3)
        """
        B, C, H = channel_tokens.shape

        mask = channel_active.float().unsqueeze(-1)  # (B, C, 1)
        n_active = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = (channel_tokens * mask).sum(dim=1) / n_active  # (B, H)

        # Routing statistics from edge weights
        ew_abs = edge_weights.abs()
        route_density = (ew_abs > 0.05).float().mean(dim=1)  # (B,)
        route_pos = (edge_weights > 0.05).float().mean(dim=1)  # (B,)
        route_neg = (edge_weights < -0.05).float().mean(dim=1)  # (B,)
        route_asym = (route_pos - route_neg).abs()  # (B,)

        stats = torch.stack([route_density, route_pos, route_neg, route_asym], dim=-1)
        logits = self.classifier(torch.cat([pooled, stats], dim=-1))
        return logits


# =========================================================================
# Full GLM
# =========================================================================

class GLM(nn.Module):
    """Graph Language Model — dense subspaces with sparse signed block routing.

    Level 1: Intra-block attention (dense within sub-pathway, from V6)
    Level 2: Mutation → Block pooling (101 block tokens)
    Level 3: Block-level sparse signed routing (328 PPI edges, CT-specific)
    Level 4: Block → Channel aggregation + escalation readout

    Config:
        node_feat_dim, hidden_dim, edge_feat_dim, n_channels, n_blocks,
        n_cancer_types, n_heads, n_intra_layers,
        cross_block_edge_index: (2, E) numpy array — required
        cross_block_edge_scores: (E,) numpy array — optional PPI scores
        block_to_channel: (n_blocks,) numpy array — channel_id per block
        n_route_layers (default 2)
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
        n_blocks = config['n_blocks']

        self.hidden = hidden
        self.n_channels = n_channels
        self.n_blocks = n_blocks

        # Register block→channel mapping as buffer
        b2c = config['block_to_channel']
        self.register_buffer(
            'block_to_channel_map',
            torch.tensor(b2c, dtype=torch.long)
        )

        # === Level 0: Node encoding ===
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.ct_sign = nn.Embedding(n_cancer_types, 1)
        nn.init.ones_(self.ct_sign.weight)

        self.age_mod = nn.Sequential(nn.Linear(1, hidden), nn.ELU())

        # === Level 1: Intra-block attention ===
        self.intra_block_layers = nn.ModuleList([
            IntraBlockLayer(hidden, n_heads, n_cancer_types, edge_feat_dim, dropout)
            for _ in range(n_intra_layers)
        ])

        # === Level 2: Mutation → Block pooling ===
        self.mutation_to_block = MutationToBlockPool(
            hidden, n_cancer_types, n_blocks, dropout
        )

        # === Level 3: Block-level sparse signed routing ===
        cross_edge_index = config['cross_block_edge_index']
        cross_edge_scores = config.get('cross_block_edge_scores')
        self.router = BlockLevelRouter(
            hidden, n_blocks, n_cancer_types,
            edge_index=cross_edge_index,
            edge_scores=cross_edge_scores,
            n_route_layers=config.get('n_route_layers', 2),
            dropout=dropout,
        )

        # === Level 4: Block → Channel + Escalation + Readout ===
        self.block_to_channel_agg = BlockToChannelAggregate(
            hidden, n_channels, n_blocks, dropout
        )

        self.escalation = EscalationClassifier(hidden, n_channels)

        # Channel pooling → patient embedding
        self.channel_pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # Clinical encoders
        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())
        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, hidden // 4), nn.ELU(),
        )
        self.escalation_encoder = nn.Sequential(
            nn.Linear(3, hidden // 4), nn.ELU(),
        )

        readout_in = hidden + hidden // 4 * 4
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _apply_ct_sign(self, node_features, cancer_type):
        """Apply per-CT sign modulation to atlas features."""
        ct_s = self.ct_sign(cancer_type)  # (B, 1)
        ct_s_broad = ct_s.squeeze(-1).unsqueeze(1)
        sign_neg = (ct_s_broad < 0).float()

        modified_log_hr = node_features[:, :, 0:1] * ct_s_broad.unsqueeze(-1)
        unchanged_1_10 = node_features[:, :, 1:10]
        modified_harmful = (node_features[:, :, 10:11] * (1 - sign_neg.unsqueeze(-1))
                            + node_features[:, :, 11:12] * sign_neg.unsqueeze(-1))
        modified_protective = (node_features[:, :, 11:12] * (1 - sign_neg.unsqueeze(-1))
                               + node_features[:, :, 10:11] * sign_neg.unsqueeze(-1))
        rest = node_features[:, :, 12:]
        return torch.cat([
            modified_log_hr, unchanged_1_10,
            modified_harmful, modified_protective, rest,
        ], dim=-1)

    def forward(self, node_features, node_mask, cancer_type, clinical,
                atlas_sum, edge_features, block_ids, channel_ids):
        """Same interface as AtlasTransformerV6.forward()."""
        B, N, _ = node_features.shape
        pad_mask = (node_mask == 0)

        # === CT sign modulation ===
        node_features = self._apply_ct_sign(node_features, cancer_type)

        # === Level 0: Encode mutations ===
        x = self.node_encoder(node_features)
        age = clinical[:, 0:1]
        age_scale = self.age_mod(age)
        x = x * (1 + age_scale.unsqueeze(1) * 0.1)

        # === Level 1: Dense intra-block attention ===
        block_mask = (block_ids.unsqueeze(1) != block_ids.unsqueeze(2))
        for layer in self.intra_block_layers:
            x, _ = layer(x, cancer_type, edge_features, block_mask, pad_mask)

        # === Level 2: Mutation → Block tokens ===
        block_tokens, block_active = self.mutation_to_block(
            x, cancer_type, block_ids, pad_mask
        )

        # === Level 3: Block-level sparse signed routing ===
        n_active_blocks = block_active.float().sum(dim=1)
        has_multi = (n_active_blocks >= 2)
        edge_weights = torch.zeros(B, self.router.n_edges, device=x.device)
        if has_multi.any():
            block_tokens, edge_weights = self.router(
                block_tokens, cancer_type, block_active
            )

        # === Level 4: Block → Channel aggregation ===
        channel_tokens, channel_active = self.block_to_channel_agg(
            block_tokens, block_active, self.block_to_channel_map
        )

        # Escalation classification
        escalation_logits = self.escalation(channel_tokens, channel_active, edge_weights)
        escalation_probs = F.softmax(escalation_logits, dim=-1)  # (B, 3)

        # Pool channels → patient
        ch_attn = self.channel_pool_attn(channel_tokens).squeeze(-1)
        ch_attn = ch_attn.masked_fill(~channel_active, float('-inf'))
        ch_attn = F.softmax(ch_attn, dim=1).nan_to_num(0.0)
        patient_embed = (channel_tokens * ch_attn.unsqueeze(-1)).sum(dim=1)

        # Clinical features
        ct_s = self.ct_sign(cancer_type)
        sex_feat = self.sex_encoder(clinical[:, 1:2])
        atlas_feat = self.atlas_skip(atlas_sum * ct_s)
        temporal_feat = self.temporal_encoder(clinical[:, 2:5])
        escalation_feat = self.escalation_encoder(escalation_probs)

        out = torch.cat([
            patient_embed, sex_feat, atlas_feat, temporal_feat, escalation_feat
        ], dim=-1)
        hazard = self.readout(out)

        # Cache for analysis and auxiliary losses
        self._last_block_tokens = block_tokens
        self._last_block_active = block_active
        self._last_channel_tokens = channel_tokens
        self._last_channel_active = channel_active
        self._last_edge_weights = edge_weights
        self._last_escalation_logits = escalation_logits

        return hazard

    def routing_loss(self, lambda_structure=0.01, lambda_sign_consistency=0.01):
        """Auxiliary losses for block routing."""
        loss = self.router.routing_loss(lambda_sign=lambda_structure)

        # Sign consistency across batch
        ew = self._last_edge_weights  # (B, E)
        if ew is not None and ew.shape[0] > 1:
            sign_var = ew.sign().var(dim=0).mean()
            loss = loss + lambda_sign_consistency * sign_var

        return loss

    def get_routing_analysis(self, block_names=None):
        """Return interpretable routing information."""
        summary = self.router.routing_summary()
        if block_names:
            edges = []
            src = summary['edge_src']
            dst = summary['edge_dst']
            strength = summary['edge_strength']
            for i in range(len(src)):
                if strength[i] > 0.5:
                    s_name = block_names.get(int(src[i]), f"block_{src[i]}")
                    d_name = block_names.get(int(dst[i]), f"block_{dst[i]}")
                    edges.append({
                        'from': s_name, 'to': d_name,
                        'strength': float(strength[i]),
                    })
            summary['named_edges'] = sorted(edges, key=lambda x: -x['strength'])
        return summary

    def load_pretrained_backbone(self, backbone_path, strict_intra=True):
        """Transfer weights from a pretrained V6 backbone into the GLM.

        Transfers:
          1. Node encoder (identical architecture)
          2. CT sign embeddings (padded if GLM has more CTs)
          3. Age modulation
          4. Intra-block attention layers (identical architecture)
          5. Cross-channel attention → block-level routing initialization
             The backbone's ct_channel_bias (n_ct, 8*8) encodes channel-pair
             interaction strengths. We project these to block-level edge weights
             by looking up the channel pair for each cross-block PPI edge.

        Args:
            backbone_path: path to pretrained_backbone.pt
            strict_intra: if True, require exact match for intra-block layers
        """
        import torch

        checkpoint = torch.load(backbone_path, map_location='cpu')
        backbone = checkpoint['backbone']
        bb_config = checkpoint.get('config', {})
        bb_n_ct = bb_config.get('n_cancer_types', 24)

        loaded = []
        skipped = []

        # --- 1. Node encoder ---
        for key in ['node_encoder.0.weight', 'node_encoder.0.bias',
                     'node_encoder.1.weight', 'node_encoder.1.bias']:
            if key in backbone:
                self_param = dict(self.named_parameters()).get(key)
                if self_param is None:
                    self_param = dict(self.named_buffers()).get(key)
                if self_param is not None and self_param.shape == backbone[key].shape:
                    self_param.data.copy_(backbone[key])
                    loaded.append(key)
                else:
                    skipped.append(f"{key} (shape mismatch)")

        # --- 2. CT sign (pad if backbone has fewer CTs) ---
        if 'ct_sign.weight' in backbone:
            bb_ct = backbone['ct_sign.weight']
            n_transfer = min(bb_ct.shape[0], self.ct_sign.weight.shape[0])
            with torch.no_grad():
                self.ct_sign.weight[:n_transfer] = bb_ct[:n_transfer]
            loaded.append(f'ct_sign.weight ({n_transfer}/{self.ct_sign.weight.shape[0]} CTs)')

        # --- 3. Age modulation ---
        for key in ['age_mod.0.weight', 'age_mod.0.bias']:
            if key in backbone:
                p = dict(self.named_parameters())[key]
                if p.shape == backbone[key].shape:
                    p.data.copy_(backbone[key])
                    loaded.append(key)

        # --- 4. Intra-block attention layers ---
        n_intra_loaded = 0
        for layer_idx in range(len(self.intra_block_layers)):
            prefix = f'intra_block_layers.{layer_idx}.'
            layer_keys = [k for k in backbone.keys() if k.startswith(prefix)]
            if not layer_keys:
                break

            layer_ok = True
            for key in layer_keys:
                self_params = dict(self.named_parameters())
                if key in self_params:
                    sp = self_params[key]
                    bp = backbone[key]
                    if sp.shape == bp.shape:
                        sp.data.copy_(bp)
                    elif len(sp.shape) == 2 and len(bp.shape) == 2 and sp.shape[1] == bp.shape[1]:
                        # Embedding with more rows in GLM (more CTs) — pad
                        n_transfer = min(sp.shape[0], bp.shape[0])
                        sp.data[:n_transfer] = bp[:n_transfer]
                    elif strict_intra:
                        layer_ok = False
                        skipped.append(f"{key} (shape {sp.shape} vs {bp.shape})")
                    else:
                        skipped.append(f"{key} (shape mismatch, non-strict)")

            if layer_ok:
                n_intra_loaded += 1
                loaded.append(f'intra_block_layers.{layer_idx} (all weights)')

        # --- 5. Cross-channel attention → block routing weights ---
        if 'cross_channel.ct_channel_bias.weight' in backbone:
            ct_bias = backbone['cross_channel.ct_channel_bias.weight']
            # ct_bias: (bb_n_ct, n_channels * n_channels)
            ct_bias = ct_bias.view(bb_n_ct, self.n_channels, self.n_channels)

            # Global mean channel interaction (average across CTs)
            mean_bias = ct_bias.mean(dim=0)  # (C, C)

            # Initialize edge_structure from channel-level interaction strength
            edge_src = self.router.edge_src.cpu()
            edge_dst = self.router.edge_dst.cpu()
            b2c = self.block_to_channel_map.cpu()

            with torch.no_grad():
                for idx in range(self.router.n_edges):
                    src_ch = b2c[edge_src[idx]].item()
                    dst_ch = b2c[edge_dst[idx]].item()
                    channel_strength = mean_bias[dst_ch, src_ch].item()
                    # Map channel interaction to edge structure logit
                    # sigmoid(logit) should match normalized strength
                    # strength in [-0.4, 0.4] → logit via inverse sigmoid
                    norm_strength = (channel_strength + 0.5)  # shift to [0.1, 0.9]
                    norm_strength = max(0.05, min(0.95, norm_strength))
                    logit = math.log(norm_strength / (1 - norm_strength))
                    self.router.edge_structure.data[idx] = logit

                # CT-specific edge weights from per-CT channel bias
                n_transfer_ct = min(bb_n_ct, self.router.ct_edge_weights.weight.shape[0])
                for ct_idx in range(n_transfer_ct):
                    ct_mat = ct_bias[ct_idx]  # (C, C)
                    for idx in range(self.router.n_edges):
                        src_ch = b2c[edge_src[idx]].item()
                        dst_ch = b2c[edge_dst[idx]].item()
                        self.router.ct_edge_weights.weight[ct_idx, idx] = \
                            ct_mat[dst_ch, src_ch] * 0.5  # scale down

            loaded.append(f'cross_channel → router ({self.router.n_edges} edges, '
                         f'{n_transfer_ct} CTs)')

        # --- 6. Transfer cross-channel projection → routing projection ---
        # The backbone's Q/K/V projections learned how to transform channel tokens
        # for cross-channel attention. Use V projection to init route_proj
        if 'cross_channel.v_proj.weight' in backbone:
            v_w = backbone['cross_channel.v_proj.weight']
            v_b = backbone['cross_channel.v_proj.bias']
            rp = dict(self.named_parameters())
            if ('router.route_proj.0.weight' in rp and
                    rp['router.route_proj.0.weight'].shape == v_w.shape):
                rp['router.route_proj.0.weight'].data.copy_(v_w)
                rp['router.route_proj.0.bias'].data.copy_(v_b)
                loaded.append('cross_channel.v_proj → router.route_proj')

        print(f"\n  Backbone transfer summary:")
        print(f"    Loaded: {len(loaded)} components")
        for item in loaded:
            print(f"      ✓ {item}")
        if skipped:
            print(f"    Skipped: {len(skipped)} components")
            for item in skipped:
                print(f"      ✗ {item}")

        return {'loaded': loaded, 'skipped': skipped}
