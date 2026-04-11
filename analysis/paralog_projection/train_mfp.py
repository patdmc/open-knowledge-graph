"""Masked feature prediction on the paralog projection cube.

Each paralog pair is a set of feature tokens. Continuous features are
bucket-embedded (per-feature learned slope+bias), categorical features are
embedded via lookup. A small transformer encoder predicts masked feature
values from the unmasked rest — self-supervised, no target leakage.

The learned [CLS] embedding per pair is the "precipitated dimension":
the synthesis of all projections in the cube, reconstructed from the
joint distribution rather than any single axis.

Apple Silicon: PyTorch MPS. Full cube fits in unified memory; no streaming.

Usage:
  # Smoke test on the current cube (whatever columns are present)
  python train_mfp.py \\
      --pair-table data/pair_table_gm12878_primary_heavy_coess.parquet \\
      --out-dir data/mfp/gm12878_primary

  # Train on a richer cube later with more feature columns
  python train_mfp.py \\
      --pair-table data/pair_table_gm12878_primary_full.parquet \\
      --features perc_id,coess_corr,hic_obs,hic_oe,linear_distance_bp,subtype_rank,ppi_jaccard,go_bp_sim,plm_cosine \\
      --d-model 192 --n-layers 6 --epochs 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Defaults — include anything we've got so the smoke test sees all columns
DEFAULT_FEATURES = [
    "perc_id",
    "perc_id_r1",
    "coess_corr",
    "hic_obs",
    "hic_oe",
    "linear_distance_bp",
    "subtype_rank",       # derived below from subtype ladder
    "same_chrom",         # bool → {0,1}
    "same_channel",       # bool → {0,1}
    "ppi_jaccard",        # TBD
    "go_bp_sim",          # TBD
    "go_mf_sim",          # TBD
    "plm_cosine",         # TBD
]


SUBTYPE_LADDER = [
    "Homo sapiens", "Hominidae", "Hominoidea", "Catarrhini", "Simiiformes",
    "Haplorrhini", "Primates", "Euarchontoglires", "Boreoeutheria", "Eutheria",
    "Theria", "Mammalia", "Amniota", "Tetrapoda", "Sarcopterygii", "Euteleostomi",
    "Vertebrata", "Gnathostomata", "Chordata", "Bilateria", "Opisthokonta",
]
SUBTYPE_TO_RANK = {s: i for i, s in enumerate(SUBTYPE_LADDER)}


def load_cube(
    pair_table: Path, features: list[str]
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, tuple[float, float]]]:
    """Load the pair table, select requested features, standardize.

    Returns:
      values: float tensor (N, F), NaN where missing
      mask:   bool tensor (N, F), True where OBSERVED (not NaN)
      feature_names: list of feature names actually used
      stats: dict feature_name → (mean, std) used for standardization
    """
    print(f"[mfp] loading {pair_table}", file=sys.stderr)
    df = pd.read_parquet(pair_table)
    print(f"[mfp] {len(df):,} pairs", file=sys.stderr)

    # Derive subtype_rank from subtype string
    if "subtype" in df.columns and "subtype_rank" not in df.columns:
        df["subtype_rank"] = df.subtype.map(SUBTYPE_TO_RANK).astype("float32")

    # Cast bool to float
    for col in ["same_chrom", "same_channel"]:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype("float32")

    # Log-transform linear distance (heavy tail) — fold into standardization
    if "linear_distance_bp" in df.columns:
        df["linear_distance_bp"] = np.log1p(df.linear_distance_bp.fillna(0))

    # Log-transform hic_obs (heavy tail)
    if "hic_obs" in df.columns:
        df["hic_obs"] = np.log1p(df.hic_obs.clip(lower=0))

    # Keep only requested features that actually exist
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[mfp] features missing from table: {missing}", file=sys.stderr)
    print(f"[mfp] features used ({len(present)}): {present}", file=sys.stderr)

    # Extract raw matrix with NaN preserved
    raw = df[present].astype("float32").values  # (N, F)
    mask = np.isfinite(raw)

    # Per-feature standardization using observed values only
    stats: dict[str, tuple[float, float]] = {}
    std_values = raw.copy()
    for i, name in enumerate(present):
        col = raw[:, i]
        m = mask[:, i]
        if m.sum() < 2:
            stats[name] = (0.0, 1.0)
            continue
        mean = float(np.nanmean(col))
        std = float(np.nanstd(col))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        std_values[:, i] = np.where(m, (col - mean) / std, 0.0)
        stats[name] = (mean, std)

    values = torch.from_numpy(std_values).float()
    mask_t = torch.from_numpy(mask).bool()
    return values, mask_t, present, stats


class FTMaskedEncoder(nn.Module):
    """FT-Transformer-style encoder with masked feature prediction head.

    Each feature is a token. A token's embedding is
      type_embed[feature_id] + value_scalar * value_slope[feature_id] + value_bias[feature_id]
    for continuous features. Masked tokens use a learned [MASK] vector
    regardless of the underlying value.

    A [CLS] token is prepended; its encoder output is the pair-level
    "precipitated dimension" for downstream use.

    The prediction head is per-feature: each feature's masked token
    produces a scalar prediction passed through its own linear readout
    so loss is comparable across features.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Token embeddings
        # 0 .. n_features-1 are the feature type ids; n_features is [CLS]
        self.type_embed = nn.Embedding(n_features + 1, d_model)
        # Per-feature affine map for continuous values
        self.value_slope = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.value_bias = nn.Parameter(torch.zeros(n_features, d_model))
        # Learned mask vector
        self.mask_vector = nn.Parameter(torch.randn(d_model) * 0.02)
        # Learned CLS vector
        self.cls_vector = nn.Parameter(torch.randn(d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # enable_nested_tensor=False avoids an MPS op
        # (_nested_tensor_from_mask_left_aligned) that isn't implemented yet.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Per-feature prediction head
        self.predict = nn.Linear(d_model, n_features)

    def build_tokens(
        self, values: torch.Tensor, mask_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the token sequence for a batch.

        Args:
          values: (B, F) standardized feature values (zero where missing).
          mask_in: (B, F) bool — True where the feature is OBSERVED
                              AND NOT MASKED for prediction.

        Returns:
          tokens: (B, 1+F, d_model) — [CLS] prepended.
          attn_mask_keypad: (B, 1+F) — True where the token should be IGNORED
                                       (i.e., missing features). CLS is always
                                       kept. Masked-for-prediction tokens are
                                       NOT ignored (they get the mask_vector
                                       and we want the model to attend to them).
        """
        B, F = values.shape
        device = values.device

        # Per-feature embedding for each batch element
        feat_ids = torch.arange(F, device=device).unsqueeze(0).expand(B, F)  # (B, F)
        type_e = self.type_embed(feat_ids)  # (B, F, d)
        # Affine value embedding — use value * slope + bias per feature
        slopes = self.value_slope[feat_ids]  # (B, F, d)
        biases = self.value_bias[feat_ids]   # (B, F, d)
        value_e = values.unsqueeze(-1) * slopes + biases  # (B, F, d)
        tokens = type_e + value_e  # (B, F, d)

        # Where a feature is NOT in mask_in but IS OBSERVED originally — means it's
        # being masked for prediction. Replace the value-derived component with
        # the learned mask vector so the model cannot read the value.
        # Caller passes the actual "visible" mask in `mask_in`.
        mv = self.mask_vector.view(1, 1, -1).expand_as(tokens)
        tokens = torch.where(mask_in.unsqueeze(-1), tokens, type_e + mv)

        # Prepend CLS
        cls = self.cls_vector.view(1, 1, -1).expand(B, 1, self.d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+F, d)

        # Build key-padding mask: True where IGNORED. Ignore features that were
        # never observed (NaN → not in mask at all). CLS never ignored.
        # For this we need the "originally missing" mask, which we must track
        # separately; here, we infer: if mask_in is False AND value == 0 after
        # standardization, it might be either masked-for-prediction or missing.
        # Let the caller pass present_mask explicitly via the forward method.
        return tokens, None  # pad mask set in forward

    def forward(
        self,
        values: torch.Tensor,         # (B, F) standardized, zero where missing/masked
        present: torch.Tensor,        # (B, F) bool — originally observed
        visible: torch.Tensor,        # (B, F) bool — observed AND NOT currently masked
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, F = values.shape
        tokens, _ = self.build_tokens(values, visible)
        # Key padding: True = ignore. CLS always kept; features ignored if not present.
        pad = torch.zeros(B, 1 + F, dtype=torch.bool, device=values.device)
        pad[:, 1:] = ~present
        h = self.encoder(tokens, src_key_padding_mask=pad)  # (B, 1+F, d)
        cls_emb = h[:, 0, :]  # (B, d)
        # Per-feature predictions from each feature's token position
        feat_tokens = h[:, 1:, :]  # (B, F, d)
        # Use a shared linear head that outputs a vector per feature position;
        # we want a scalar prediction per token, gathered from the diagonal.
        # Cleanest: take each feature-position's own readout index.
        all_preds = self.predict(feat_tokens)  # (B, F, n_features)
        idx = torch.arange(F, device=values.device).view(1, F, 1).expand(B, F, 1)
        preds = torch.gather(all_preds, 2, idx).squeeze(-1)  # (B, F)
        return preds, cls_emb


def make_batches(
    values: torch.Tensor,
    present: torch.Tensor,
    batch_size: int,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    n = values.shape[0]
    idx = torch.arange(n)
    ds = TensorDataset(idx)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


def train(
    values: torch.Tensor,
    present: torch.Tensor,
    feature_names: list[str],
    device: str,
    d_model: int,
    n_heads: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    mask_prob: float,
    lr: float,
    out_dir: Path,
) -> tuple[FTMaskedEncoder, torch.Tensor]:
    n, F = values.shape
    values = values.to(device)
    present = present.to(device)

    model = FTMaskedEncoder(
        n_features=F, d_model=d_model, n_heads=n_heads, n_layers=n_layers
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(
        f"[mfp] model: {sum(p.numel() for p in model.parameters())/1e3:.0f}K params, "
        f"device={device}, features={F}",
        file=sys.stderr,
    )

    loader = make_batches(values, present, batch_size)
    for epoch in range(epochs):
        t0 = time.time()
        running = 0.0
        count = 0
        model.train()
        for (idx,) in loader:
            v = values[idx]        # (B, F)
            p = present[idx]       # (B, F) — originally observed
            # Random per-token mask: True where we HIDE the value (force prediction)
            hide = (torch.rand_like(v) < mask_prob) & p  # only hide observed features
            visible = p & ~hide                         # observed and not hidden
            # Zero the hidden positions in values (model uses mask_vector anyway)
            v_in = torch.where(visible, v, torch.zeros_like(v))

            preds, _ = model(v_in, p, visible)

            # Loss only on hidden positions
            target = v
            per = (preds - target) ** 2
            mask = hide.float()
            denom = mask.sum().clamp_min(1.0)
            loss = (per * mask).sum() / denom

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item()) * int(mask.sum().item())
            count += int(mask.sum().item())
        elapsed = time.time() - t0
        avg = running / max(count, 1)
        print(
            f"[mfp] epoch {epoch+1}/{epochs}  loss={avg:.4f}  ({elapsed:.1f}s)",
            file=sys.stderr,
        )

    # Compute embeddings for the full dataset (no masking)
    # Also: compute per-feature reconstruction loss by masking each feature
    # one at a time and measuring prediction error. This quantifies how much
    # of each feature's variance is recoverable from the other features —
    # the lower the loss, the more "redundant" the feature is with the rest;
    # the higher the loss, the more orthogonal the feature is.
    model.eval()
    embeds = torch.empty((n, d_model), device=device, dtype=torch.float32)
    per_feat_sq_err = torch.zeros(F, device=device)
    per_feat_count = torch.zeros(F, device=device)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            v = values[start:stop]
            p = present[start:stop]
            visible = p
            v_in = torch.where(visible, v, torch.zeros_like(v))
            _, cls = model(v_in, p, visible)
            embeds[start:stop] = cls
            # Per-feature leave-one-out prediction
            for k in range(F):
                # Mask only feature k for pairs where it's present
                hide = torch.zeros_like(p)
                hide[:, k] = p[:, k]
                vis_k = p & ~hide
                v_in_k = torch.where(vis_k, v, torch.zeros_like(v))
                preds_k, _ = model(v_in_k, p, vis_k)
                err = ((preds_k[:, k] - v[:, k]) ** 2) * hide[:, k].float()
                per_feat_sq_err[k] += err.sum()
                per_feat_count[k] += hide[:, k].float().sum()

    per_feat_mse = (per_feat_sq_err / per_feat_count.clamp_min(1)).cpu().numpy()
    print("[mfp] per-feature leave-one-out MSE (on standardized values):",
          file=sys.stderr)
    for i, name in enumerate(feature_names):
        print(f"  {name:22s}  MSE={per_feat_mse[i]:.4f}", file=sys.stderr)

    return model, embeds.cpu(), per_feat_mse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-table", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/mfp"))
    ap.add_argument(
        "--features",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated feature list. Defaults to every known feature "
             "column present in the pair table.",
    )
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--mask-prob", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    pair_table = rel(args.pair_table)
    out_dir = rel(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = args.features or DEFAULT_FEATURES
    values, present, feat_names, stats = load_cube(pair_table, features)
    print(f"[mfp] cube: {values.shape}", file=sys.stderr)

    # Device sanity
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[mfp] MPS not available; falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    model, embeds, per_feat_mse = train(
        values, present, feat_names,
        device=args.device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_prob=args.mask_prob,
        lr=args.lr,
        out_dir=out_dir,
    )

    # Save embeddings + feature metadata
    tag = pair_table.stem
    emb_path = out_dir / f"embeds_{tag}.npy"
    meta_path = out_dir / f"meta_{tag}.json"
    np.save(emb_path, embeds.numpy())
    meta = {
        "pair_table": str(pair_table),
        "features": feat_names,
        "stats": {k: list(v) for k, v in stats.items()},
        "per_feature_mse": {name: float(per_feat_mse[i]) for i, name in enumerate(feat_names)},
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "mask_prob": args.mask_prob,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[mfp] wrote {emb_path}", file=sys.stderr)
    print(f"[mfp] wrote {meta_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
