"""
AtlasTransformer V5b — Factored attention: compute once, apply per-patient.

Key insight: gene-gene attention is a property of the GRAPH, not the patient.
Many patients share the same mutated genes. The attention between TP53 and KRAS
is the same regardless of which patient has them.

Architecture:
  Phase 1 — Gene-level attention (runs ONCE on the full gene set):
    - Each of 509 genes gets a node embedding from its properties
    - Edge-informed attention within sub-pathway blocks (block-sparse)
    - Output: enriched gene embeddings + attention matrix

  Phase 2 — Patient scoring (fast, uses precomputed gene embeddings):
    - Select mutated genes for this patient
    - Gather their enriched embeddings
    - Cancer-type and clinical modulation
    - Readout: weighted sum → hazard score

Training still uses per-patient Cox loss, but the expensive attention
is shared across all patients in the batch. The gene embeddings are
computed once per forward pass, not once per patient.

This mirrors the hierarchical scorer: precompute gene-pair interactions,
then score patients by selecting the relevant subset.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneGraphEncoder(nn.Module):
    """Encode all genes using graph structure. Runs once, shared across patients.

    Input: all genes with node features + all pairwise edge features
    Output: enriched gene embeddings (n_genes, hidden)
    """

    def __init__(self, node_feat_dim, hidden, n_heads, n_edge_features,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            EdgeInformedBlock(hidden, n_heads, n_edge_features, dropout)
            for _ in range(n_layers)
        ])

        # Learned interaction score: how much does gene j's mutation
        # affect gene i's contribution to patient hazard?
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden * 2 + n_edge_features, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_features, edge_features, block_mask=None):
        """
        node_features: (G, node_feat_dim) — all genes
        edge_features: (G, G, n_edge_features) — all pairwise
        block_mask: (G, G) bool — True = don't attend

        Returns:
            gene_embeds: (G, hidden)
            interaction_scores: (G, G) — learned pairwise interaction weights
            attn_weights: (G, G) — last layer attention (for precipitation)
        """
        x = self.node_encoder(node_features)

        attn_w = None
        for layer in self.layers:
            x, attn_w = layer(x, edge_features, block_mask)

        # Compute pairwise interaction scores
        G = x.shape[0]
        xi = x.unsqueeze(1).expand(G, G, -1)  # (G, G, H)
        xj = x.unsqueeze(0).expand(G, G, -1)  # (G, G, H)
        pair_input = torch.cat([xi, xj, edge_features], dim=-1)  # (G, G, H*2+E)
        interaction_scores = self.interaction_head(pair_input).squeeze(-1)  # (G, G)

        return x, interaction_scores, attn_w


class EdgeInformedBlock(nn.Module):
    """Transformer block with edge-informed attention."""

    def __init__(self, hidden, n_heads, n_edge_features, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        self.edge_bias = nn.Sequential(
            nn.Linear(n_edge_features, n_heads * 2),
            nn.ELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, edge_features, block_mask=None):
        """
        x: (G, hidden)
        edge_features: (G, G, n_edge_features)
        block_mask: (G, G) bool
        """
        G = x.shape[0]
        normed = self.norm1(x)

        q = self.q_proj(normed).view(G, self.n_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(normed).view(G, self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(normed).view(G, self.n_heads, self.head_dim).transpose(0, 1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (H, G, G)

        e_bias = self.edge_bias(edge_features).permute(2, 0, 1)  # (H, G, G)
        attn = attn + e_bias

        if block_mask is not None:
            attn = attn.masked_fill(block_mask.unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (H, G, head_dim)
        out = out.transpose(0, 1).contiguous().view(G, self.hidden)
        out = self.out_proj(out)

        x = x + out
        x = x + self.ffn(self.norm2(x))

        # Average attention across heads for reporting
        attn_avg = attn.mean(dim=0)  # (G, G)
        return x, attn_avg


class AtlasTransformerV5b(nn.Module):
    """Factored transformer: gene attention computed once, patient scoring is fast.

    Config must include:
        node_feat_dim: int — gene node features (14 atlas + dynamic extras)
        hidden_dim: int
        n_edge_features: int — from GraphSchema
        n_channels: int
        n_cancer_types: int
    """

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config.get('dropout', 0.1)
        n_cancer_types = config.get('n_cancer_types', 80)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_block_layers', 2)
        n_channels = config['n_channels']
        n_edge_features = config['n_edge_features']

        self.hidden = hidden
        self.n_channels = n_channels
        self.config = config

        # Phase 1: Gene graph encoder (shared)
        self.gene_encoder = GeneGraphEncoder(
            node_feat_dim, hidden, n_heads, n_edge_features,
            n_layers=n_layers, dropout=dropout,
        )

        # Phase 2: Patient-level modules
        # Per-patient node encoder: atlas features (log_hr, ci, tier, etc.)
        # This is what makes each patient different — same gene, different HR
        patient_node_dim = config.get('patient_node_dim', 17)  # NODE_FEAT_DIM (mutation-level)
        self.node_encoder_patient = nn.Sequential(
            nn.Linear(patient_node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )

        self.cancer_mod = nn.Embedding(n_cancer_types, hidden)
        self.age_mod = nn.Sequential(nn.Linear(1, hidden // 2), nn.ELU())
        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())

        # Per-gene hazard projection (from enriched embedding)
        self.gene_hazard = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Channel-level aggregation
        self.channel_proj = nn.Linear(hidden, hidden)
        self.cross_channel_attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden)

        # Final readout
        readout_in = hidden + hidden // 2 + hidden // 4 + hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Cache for gene embeddings
        self._gene_embeds = None
        self._interaction_scores = None
        self._last_attn = None

    def encode_genes(self, gene_node_features, gene_edge_features,
                     block_ids=None):
        """Phase 1: Encode all genes. Call once, reuse across patients.

        gene_node_features: (G, node_feat_dim)
        gene_edge_features: (G, G, n_edge_features)
        block_ids: (G,) long — sub-pathway block per gene

        Returns gene_embeds, interaction_scores, attn
        """
        # Build block mask
        block_mask = None
        if block_ids is not None:
            block_mask = (block_ids.unsqueeze(0) != block_ids.unsqueeze(1))

        embeds, interactions, attn = self.gene_encoder(
            gene_node_features, gene_edge_features, block_mask
        )

        self._gene_embeds = embeds
        self._interaction_scores = interactions
        self._last_attn = attn
        return embeds, interactions, attn

    def forward(self, gene_indices, gene_masks, channel_ids,
                cancer_type, clinical, atlas_sum,
                node_features=None):
        """Phase 2: Score patients using precomputed gene interactions.

        gene_indices: (B, N) long — indices into gene vocabulary
        gene_masks: (B, N) float — 1 real, 0 pad
        channel_ids: (B, N) long — channel per mutation
        cancer_type: (B,) long
        clinical: (B, 2) — [age_z, sex]
        atlas_sum: (B, 1)
        node_features: (B, N, feat_dim) — per-patient atlas features (includes log_hr)
        """
        assert self._gene_embeds is not None, "Call encode_genes() first"

        B, N = gene_indices.shape
        pad_mask = (gene_masks == 0)

        # Start from gene graph embeddings (structure-aware)
        safe_idx = gene_indices.clamp(0, self._gene_embeds.shape[0] - 1)
        graph_embeds = self._gene_embeds[safe_idx]  # (B, N, hidden)

        # If we have per-patient node features (atlas HRs etc), encode and add
        if node_features is not None:
            patient_node_embeds = self.node_encoder_patient(node_features)
            patient_embeds = graph_embeds + patient_node_embeds
        else:
            patient_embeds = graph_embeds

        patient_embeds = patient_embeds * gene_masks.unsqueeze(-1)

        # Cancer type modulation
        ct_embed = self.cancer_mod(cancer_type)  # (B, hidden)
        patient_embeds = patient_embeds + ct_embed.unsqueeze(1) * gene_masks.unsqueeze(-1)

        # Age modulation
        age = clinical[:, 0:1]  # (B, 1)
        age_embed = self.age_mod(age)  # (B, hidden//2)
        age_scale = F.pad(age_embed, (0, self.hidden - age_embed.shape[-1]), value=0.0)
        patient_embeds = patient_embeds * (1 + age_scale.unsqueeze(1) * 0.1)

        # Gather precomputed interaction scores for this patient's gene pairs
        idx_i = safe_idx.unsqueeze(2).expand(B, N, N)
        idx_j = safe_idx.unsqueeze(1).expand(B, N, N)
        patient_interactions = self._interaction_scores[idx_i, idx_j]  # (B, N, N)

        pair_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        patient_interactions = patient_interactions.masked_fill(pair_mask, 0.0)

        # Interaction modulation: graph-learned gene pairs modulate embeddings
        inter_weights = torch.sigmoid(patient_interactions)
        inter_weights = inter_weights.masked_fill(pair_mask, 0.0)
        n_pairs = gene_masks.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        inter_context = torch.bmm(inter_weights, patient_embeds) / n_pairs
        patient_embeds = patient_embeds + inter_context

        # Channel pooling: average embeddings per channel
        channel_tokens = torch.zeros(B, self.n_channels, self.hidden,
                                     device=patient_embeds.device)
        channel_active = torch.zeros(B, self.n_channels,
                                     device=patient_embeds.device)

        for ch_id in range(self.n_channels):
            ch_mask = (channel_ids == ch_id) & (~pad_mask)
            has_ch = ch_mask.any(dim=1)
            if not has_ch.any():
                continue
            channel_active[:, ch_id] = has_ch.float()
            ch_count = ch_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            ch_embed = (patient_embeds * ch_mask.unsqueeze(-1).float()).sum(dim=1) / ch_count
            channel_tokens[:, ch_id] = self.channel_proj(ch_embed)

        # Cross-channel attention
        ch_pad = (channel_active == 0)
        n_active = channel_active.sum(dim=1)  # (B,)

        # Only run cross-channel on patients with >= 2 active channels
        has_multi = (n_active >= 2)
        if has_multi.any():
            # Need to ensure key_padding_mask doesn't have all-True rows
            # Force at least one channel unmasked per patient for attention stability
            safe_pad = ch_pad.clone()
            all_masked = safe_pad.all(dim=1)
            if all_masked.any():
                safe_pad[all_masked, 0] = False

            ct_out, _ = self.cross_channel_attn(
                channel_tokens, channel_tokens, channel_tokens,
                key_padding_mask=safe_pad,
            )
            ct_out = ct_out.nan_to_num(0.0)
            channel_tokens = channel_tokens + ct_out
            channel_tokens = self.cross_norm(channel_tokens)

        # Pool channels — zero embed for patients with no active channels
        ch_sum = n_active.unsqueeze(1).clamp(min=1)
        ch_weights = channel_active / ch_sum
        patient_embed = (channel_tokens * ch_weights.unsqueeze(-1)).sum(dim=1)

        # Readout
        sex_feat = self.sex_encoder(clinical[:, 1:2])
        age_feat = self.age_mod(clinical[:, 0:1])
        atlas_feat = self.atlas_skip(atlas_sum)

        out = torch.cat([patient_embed, age_feat, sex_feat, atlas_feat], dim=-1)
        return self.readout(out)

    def get_attention_maps(self):
        return {
            'gene_attn': self._last_attn,
            'interaction_scores': self._interaction_scores,
        }

    @staticmethod
    def config_from_schema(schema, n_cancer_types=80):
        from gnn.data.atlas_dataset import NODE_FEAT_DIM
        return {
            'node_feat_dim': NODE_FEAT_DIM + schema.node_extra_dim,
            'patient_node_dim': NODE_FEAT_DIM,  # mutation-level features
            'hidden_dim': 64,
            'dropout': 0.1,
            'n_cancer_types': n_cancer_types,
            'n_heads': 4,
            'n_block_layers': 2,
            'n_cross_layers': 1,
            'n_channels': schema.n_channels,
            'n_edge_features': schema.edge_feature_dim,
        }
