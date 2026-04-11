"""Enrich a paralog pair table with ESM2 protein-language-model similarity.

ESM2 is a transformer trained on ~65M protein sequences. Its per-protein
mean-pooled embedding captures evolutionary and functional context that
pairwise alignment percent identity does not. Dennler & Ryan 2025 showed
PLM cosine similarity adds 0.05-0.10 AUROC over sequence identity for
predicting paralog functional equivalence.

This script:
  1. Reads the paralog pair table to get the unique gene set.
  2. Fetches canonical protein FASTA for each gene from the UniProt
     human proteome reference (UP000005640). Cached to disk.
  3. Runs ESM2-650M on the M5 Max GPU (MPS) to produce one embedding
     vector per gene. Cached to disk.
  4. Computes cosine similarity for each paralog pair and writes it as
     the `plm_cosine` column.

The two big disk caches are:
  data/plm/human_proteome.fasta.gz    — UniProt reference proteome (~30 MB)
  data/plm/esm2_650M_embeds.npz       — per-gene embeddings (~1200 dims × ~20K genes)

Re-runs skip the fetches / forward passes if caches exist.

Usage:
  python enrich_plm.py \\
      --pair-table data/pair_table_gm12878_primary_heavy_coess.parquet \\
      --out data/pair_table_gm12878_primary_heavy_coess_plm.parquet
"""

import argparse
import gzip
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


UNIPROT_HUMAN_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
    "reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"
)

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"


def download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[plm] cached {out_path} ({out_path.stat().st_size/1e6:.1f} MB)",
              file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[plm] downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "paralog-projection/0.1"})
    with urllib.request.urlopen(req, timeout=1800) as resp, out_path.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    print(f"[plm] wrote {out_path}", file=sys.stderr)


def parse_uniprot_fasta(fasta_gz: Path) -> dict[str, str]:
    """Return {gene_symbol: sequence} from the UniProt reference proteome FASTA.

    Header format:
      >sp|P38398|BRCA1_HUMAN Breast cancer type 1 susceptibility protein OS=Homo sapiens OX=9606 GN=BRCA1 PE=1 SV=2

    We extract the GN= token as the gene symbol. If a gene has multiple
    isoforms / entries, keep the LONGEST sequence (the canonical is
    usually the longest reviewed entry in SwissProt).
    """
    print(f"[plm] parsing {fasta_gz}", file=sys.stderr)
    seqs: dict[str, str] = {}
    current_gene: str | None = None
    current_seq: list[str] = []
    def flush():
        nonlocal current_gene, current_seq
        if current_gene and current_seq:
            seq = "".join(current_seq)
            prev = seqs.get(current_gene)
            if prev is None or len(seq) > len(prev):
                seqs[current_gene] = seq
        current_gene, current_seq = None, []

    with gzip.open(fasta_gz, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                # Extract GN=<symbol>
                gn = None
                for token in line.split():
                    if token.startswith("GN="):
                        gn = token[3:].strip()
                        break
                current_gene = gn
                current_seq = []
            else:
                current_seq.append(line)
        flush()
    print(f"[plm] {len(seqs)} unique gene symbols with sequences", file=sys.stderr)
    return seqs


def embed_sequences(
    sequences: dict[str, str],
    cache_path: Path,
    batch_tokens: int = 8000,
    max_len: int = 1022,
) -> dict[str, np.ndarray]:
    """Run ESM2 on the given sequences, returning {symbol: mean-pooled embedding}.

    Cached to cache_path (npz). If cache exists for the exact same keys,
    returned as-is. If it exists but key set differs, recomputed fully
    (simpler than incremental).
    """
    if cache_path.exists():
        try:
            npz = np.load(cache_path)
            cached_syms = set(npz["symbols"].tolist())
            if cached_syms == set(sequences.keys()):
                embeds = {s: npz["embeds"][i] for i, s in enumerate(npz["symbols"])}
                print(
                    f"[plm] loaded {len(embeds)} cached embeddings from {cache_path}",
                    file=sys.stderr,
                )
                return embeds
            else:
                print(
                    f"[plm] cache key mismatch ({len(cached_syms)} cached vs "
                    f"{len(sequences)} requested); recomputing",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[plm] cache load failed ({exc}); recomputing", file=sys.stderr)

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[plm] loading {ESM2_MODEL} on {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL).to(device).eval()

    # Sort sequences by length so batches are tighter.
    items = sorted(sequences.items(), key=lambda kv: len(kv[1]))
    symbols = [s for s, _ in items]
    seqs = [s[:max_len] for _, s in items]  # ESM2 max position embedding

    embeds = np.zeros((len(symbols), model.config.hidden_size), dtype=np.float32)
    i = 0
    t0 = time.time()
    last_print = t0
    with torch.inference_mode():
        while i < len(symbols):
            # Greedy batch: include sequences until token budget is hit
            batch_seqs = []
            batch_idx = []
            total_tokens = 0
            j = i
            while j < len(symbols):
                seq_len = len(seqs[j]) + 2  # CLS + EOS
                if batch_seqs and total_tokens + seq_len > batch_tokens:
                    break
                batch_seqs.append(seqs[j])
                batch_idx.append(j)
                total_tokens += seq_len
                j += 1

            enc = tok(batch_seqs, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len + 2)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state  # (B, L, D)
            # Mean-pool over attention-masked tokens (excl padding)
            attn = enc["attention_mask"].float().unsqueeze(-1)
            summed = (last * attn).sum(dim=1)
            counts = attn.sum(dim=1).clamp_min(1)
            pooled = (summed / counts).cpu().numpy()
            for k, idx_in_batch in enumerate(batch_idx):
                embeds[idx_in_batch] = pooled[k]

            i = j
            now = time.time()
            if now - last_print > 10:
                print(
                    f"[plm] embedded {i}/{len(symbols)} "
                    f"({(now-t0):.1f}s, {i/(now-t0):.0f} prot/s)",
                    file=sys.stderr,
                )
                last_print = now

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, symbols=np.array(symbols, dtype=object), embeds=embeds)
    print(f"[plm] wrote {cache_path} ({len(symbols)} embeddings)", file=sys.stderr)
    return {s: embeds[i] for i, s in enumerate(symbols)}


def cosine_similarity_pairs(
    pairs: pd.DataFrame,
    embeds: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized pairwise cosine similarity using a lookup table."""
    # Build a gene → row_idx map, stack only the genes we need.
    needed = pd.unique(pd.concat([pairs.gene_a, pairs.gene_b], ignore_index=True))
    present = [g for g in needed if g in embeds]
    print(f"[plm] pair genes with embedding: {len(present)}/{len(needed)}", file=sys.stderr)

    sym_to_row: dict[str, int] = {s: i for i, s in enumerate(present)}
    dim = next(iter(embeds.values())).shape[0]
    mat = np.zeros((len(present), dim), dtype=np.float32)
    for s, i in sym_to_row.items():
        mat[i] = embeds[s]
    # L2-normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms

    ia = pairs.gene_a.map(sym_to_row).values
    ib = pairs.gene_b.map(sym_to_row).values
    out = np.full(len(pairs), np.nan, dtype=np.float32)
    mask = pd.notna(ia) & pd.notna(ib)
    ia = ia[mask].astype(np.int64)
    ib = ib[mask].astype(np.int64)
    sims = (mat[ia] * mat[ib]).sum(axis=1)
    out[mask] = sims
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-table", type=Path, required=True)
    ap.add_argument("--plm-dir", type=Path, default=Path("data/plm"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-tokens", type=int, default=8000)
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    plm_dir = rel(args.plm_dir)
    fasta_path = plm_dir / "UP000005640_9606.fasta.gz"
    embed_cache = plm_dir / "esm2_650M_embeds.npz"

    download_if_missing(UNIPROT_HUMAN_FASTA_URL, fasta_path)
    all_seqs = parse_uniprot_fasta(fasta_path)

    # Only embed genes that appear in the paralog table (saves time)
    print(f"[plm] loading pair table", file=sys.stderr)
    pairs = pd.read_parquet(rel(args.pair_table))
    gene_set = set(pairs.gene_a.unique()) | set(pairs.gene_b.unique())
    need_seqs = {g: all_seqs[g] for g in gene_set if g in all_seqs}
    print(f"[plm] {len(need_seqs)}/{len(gene_set)} pair genes have a sequence",
          file=sys.stderr)

    embeds = embed_sequences(need_seqs, embed_cache, batch_tokens=args.batch_tokens)

    plm_cosine = cosine_similarity_pairs(pairs, embeds)
    pairs = pairs.copy()
    pairs["plm_cosine"] = plm_cosine

    n = int(np.isfinite(plm_cosine).sum())
    print(f"[plm] pairs with plm_cosine: {n:,}/{len(pairs):,}", file=sys.stderr)
    if n:
        print(
            f"[plm] distribution: "
            f"median={np.nanmedian(plm_cosine):.3f} "
            f"q25={np.nanquantile(plm_cosine,0.25):.3f} "
            f"q75={np.nanquantile(plm_cosine,0.75):.3f} "
            f"q99={np.nanquantile(plm_cosine,0.99):.3f}",
            file=sys.stderr,
        )

    out_path = rel(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"[plm] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
