"""Heavy-path paralog pair builder: continuous Hi-C contact frequency.

The light path (build_pair_table.py) joins paralog pairs against HiCCUPS
loop calls. That turned out to be too sparse: HiCCUPS loops span 10-100x
less than the median paralog separation (~29 Mb), and none of the loops
cross chromosomes, so the key signal (cross-chromosomal paralog pairs in
3D contact) is invisible.

This script joins against the raw .hic contact matrix instead, pulling
observed contact counts at a chosen resolution for every paralog pair —
including inter-chromosomal pairs.

Design: map-reduce by (chrom_a, chrom_b) shard.

  map step   — one shard per chromosome pair. Each shard:
                 1. Loads the cached canonical paralog subset for this
                    (chrom_a, chrom_b).
                 2. Opens the .hic file, pulls the zoom-data block for
                    this chrom pair at the chosen resolution.
                 3. Looks up observed contact counts at each pair's
                    gene-midpoint bin. Optionally pulls O/E too (intra only).
                 4. Writes data/heavy/<cell>/shards/<A>_<B>.parquet
                    atomically (.tmp → rename).

  reduce step — separate script concat_heavy_shards.py (not this file)
                reads every shard parquet and writes the final joined
                pair table.

Checkpointing: presence of the shard parquet is the source of truth. If
data/heavy/<cell>/shards/<A>_<B>.parquet exists, that shard is done and
will be skipped. Delete the file to force re-run.

Parallelism: this script runs ONE shard per invocation (--shard A B) or
enumerates shards (--list). Use xargs -P / gnu parallel / a bash loop to
spawn N workers — each invocation is its own Python process, no fork
safety concerns with hicstraw's C++ backend.

Usage:
  # one-time: cache the filtered paralog table so workers don't re-read the TSV
  python build_pair_table_heavy.py --cache-paralogs

  # list shards (for driving from shell)
  python build_pair_table_heavy.py --list

  # run one shard
  python build_pair_table_heavy.py --shard 17 17 \\
      --hic data/hic/GSE63525_GM12878_insitu_DpnII_combined_30.hic \\
      --cell-line gm12878_dpnII

  # run all shards serially (slow but simple)
  python build_pair_table_heavy.py --all \\
      --hic data/hic/GSE63525_GM12878_insitu_DpnII_combined_30.hic \\
      --cell-line gm12878_dpnII
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]

PARALOG_COLS = {
    "Gene stable ID": "gene_id_a",
    "Gene name": "gene_a",
    "Chromosome/scaffold name": "chrom_a",
    "Gene start (bp)": "start_a",
    "Gene end (bp)": "end_a",
    "Human paralogue gene stable ID": "gene_id_b",
    "Human paralogue associated gene name": "gene_b",
    "Human paralogue chromosome/scaffold name": "chrom_b",
    "Human paralogue chromosome/scaffold start (bp)": "start_b",
    "Human paralogue chromosome/scaffold end (bp)": "end_b",
    "Paralogue %id. target Human gene identical to query gene": "perc_id",
    "Paralogue %id. query gene identical to target Human gene": "perc_id_r1",
    "Human paralogue homology type": "orthology_type",
    "Paralogue last common ancestor with Human": "subtype",
}


def build_canonical_paralogs(paralogs_tsv: Path, cache_path: Path) -> pd.DataFrame:
    """Load, filter, and cache the canonical paralog table as parquet."""
    print(f"[heavy] reading {paralogs_tsv}", file=sys.stderr)
    df = pd.read_csv(
        paralogs_tsv,
        sep="\t",
        dtype={
            "Chromosome/scaffold name": str,
            "Human paralogue chromosome/scaffold name": str,
        },
    )
    df = df.rename(columns=PARALOG_COLS)
    df = df.dropna(subset=["gene_id_b"])
    canon = set(CANONICAL_CHROMS)
    df = df[df.chrom_a.isin(canon) & df.chrom_b.isin(canon)].copy()
    df["mid_a"] = ((df.start_a + df.end_a) // 2).astype(np.int64)
    df["mid_b"] = ((df.start_b + df.end_b) // 2).astype(np.int64)
    df["same_chrom"] = df.chrom_a == df.chrom_b
    df["linear_distance_bp"] = np.where(
        df.same_chrom,
        (df.mid_a - df.mid_b).abs(),
        np.nan,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[heavy] cached {len(df)} rows → {cache_path}", file=sys.stderr)
    return df


def load_canonical_paralogs(cache_path: Path, paralogs_tsv: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return build_canonical_paralogs(paralogs_tsv, cache_path)


def attach_channels(pairs: pd.DataFrame, channel_map_csv: Path) -> pd.DataFrame:
    ch = pd.read_csv(channel_map_csv)
    ch = ch.drop_duplicates("gene")
    pairs = pairs.merge(
        ch.rename(columns={"gene": "gene_a", "channel": "channel_a"}),
        on="gene_a",
        how="left",
    )
    pairs = pairs.merge(
        ch.rename(columns={"gene": "gene_b", "channel": "channel_b"}),
        on="gene_b",
        how="left",
    )
    pairs["same_channel"] = (
        pairs.channel_a.notna()
        & pairs.channel_b.notna()
        & (pairs.channel_a == pairs.channel_b)
    )
    return pairs


def shard_pairs(pairs: pd.DataFrame, chrom_a: str, chrom_b: str) -> pd.DataFrame:
    """Filter paralog pairs to those on exactly this (chrom_a, chrom_b) shard.

    A shard (A, B) holds pairs where {chrom_a, chrom_b} == {A, B} as an
    unordered set. For A != B we orient so the row's chrom_a matches A
    and chrom_b matches B (swap mid_a/mid_b as needed) before the lookup.
    """
    if chrom_a == chrom_b:
        mask = (pairs.chrom_a == chrom_a) & (pairs.chrom_b == chrom_b)
        return pairs[mask].copy()
    # Rows where chrom_a == A, chrom_b == B — keep as is.
    m_fwd = (pairs.chrom_a == chrom_a) & (pairs.chrom_b == chrom_b)
    fwd = pairs[m_fwd].copy()
    # Rows where chrom_a == B, chrom_b == A — swap to (A, B) orientation.
    m_rev = (pairs.chrom_a == chrom_b) & (pairs.chrom_b == chrom_a)
    rev = pairs[m_rev].copy()
    if len(rev):
        rev = rev.rename(
            columns={
                "gene_id_a": "gene_id_b_", "gene_id_b": "gene_id_a_",
                "gene_a": "gene_b_", "gene_b": "gene_a_",
                "chrom_a": "chrom_b_", "chrom_b": "chrom_a_",
                "start_a": "start_b_", "start_b": "start_a_",
                "end_a": "end_b_", "end_b": "end_a_",
                "mid_a": "mid_b_", "mid_b": "mid_a_",
            }
        )
        rev = rev.rename(columns=lambda c: c[:-1] if c.endswith("_") else c)
    shard = pd.concat([fwd, rev], ignore_index=True)
    # Dedupe: an unordered pair {A, B} may appear twice if Ensembl emits
    # both directions. Keep one row per (gene_id_a, gene_id_b) after orienting.
    shard = shard.drop_duplicates(subset=["gene_id_a", "gene_id_b"]).reset_index(drop=True)
    return shard


def query_hic_shard(
    hic_path: Path,
    chrom_a: str,
    chrom_b: str,
    pairs: pd.DataFrame,
    resolution: int,
    norm_intra: str,
    norm_inter: str,
) -> pd.DataFrame:
    """Pull observed contact counts for every pair on this shard.

    For intra-chrom shards, also pull observed/expected. Inter-chrom O/E
    is generally not available in Rao .hic files, so it's left NaN.
    """
    import hicstraw  # noqa: import-on-demand so --list doesn't need it

    is_intra = chrom_a == chrom_b
    norm = norm_intra if is_intra else norm_inter
    unit = "BP"
    datatype = "observed"

    hic = hicstraw.HiCFile(str(hic_path))
    try:
        m_obs = hic.getMatrixZoomData(
            chrom_a, chrom_b, datatype, norm, unit, resolution
        )
    except Exception as exc:
        # Fall back to no normalization if the chosen norm is missing
        # (common for inter-chrom in Rao .hic files).
        print(
            f"[heavy] {chrom_a} x {chrom_b}: norm={norm} failed ({exc}); "
            "falling back to NONE",
            file=sys.stderr,
        )
        m_obs = hic.getMatrixZoomData(
            chrom_a, chrom_b, datatype, "NONE", unit, resolution
        )
        norm = "NONE"

    # Determine query bounds from the pair table so we only pull the
    # region we actually need. Use a padded window, else the .hic library
    # sometimes returns empty for narrow ranges.
    pad = max(resolution * 4, 100_000)
    if len(pairs) == 0:
        return pd.DataFrame(
            {"hic_obs": [], "hic_oe": [], "hic_norm": []}
        )
    a_min = int(pairs.mid_a.min()) - pad
    a_max = int(pairs.mid_a.max()) + pad
    b_min = int(pairs.mid_b.min()) - pad
    b_max = int(pairs.mid_b.max()) + pad
    a_min = max(a_min, 0)
    b_min = max(b_min, 0)

    records = m_obs.getRecords(a_min, a_max, b_min, b_max)
    # Build a dict keyed by (binX, binY) → value.
    obs_map: dict[tuple[int, int], float] = {}
    for r in records:
        # BP coords: r.binX and r.binY are bp start positions of bins
        obs_map[(r.binX, r.binY)] = float(r.counts)

    # For intra-chrom shards, the matrix is symmetric but .hic only stores
    # the upper triangle — add the mirror.
    if is_intra:
        for (bx, by), v in list(obs_map.items()):
            if bx != by:
                obs_map.setdefault((by, bx), v)

    def lookup(mid_a_bp: int, mid_b_bp: int) -> float:
        bx = (int(mid_a_bp) // resolution) * resolution
        by = (int(mid_b_bp) // resolution) * resolution
        return obs_map.get((bx, by), 0.0)

    obs_vals = np.fromiter(
        (lookup(a, b) for a, b in zip(pairs.mid_a.values, pairs.mid_b.values)),
        dtype=np.float64,
        count=len(pairs),
    )

    # O/E for intra-chrom (removes the distance-decay background).
    oe_vals = np.full(len(pairs), np.nan, dtype=np.float64)
    if is_intra:
        try:
            m_oe = hic.getMatrixZoomData(
                chrom_a, chrom_b, "oe", norm, unit, resolution
            )
            oe_records = m_oe.getRecords(a_min, a_max, b_min, b_max)
            oe_map: dict[tuple[int, int], float] = {}
            for r in oe_records:
                oe_map[(r.binX, r.binY)] = float(r.counts)
            for (bx, by), v in list(oe_map.items()):
                if bx != by:
                    oe_map.setdefault((by, bx), v)

            def lookup_oe(mid_a_bp: int, mid_b_bp: int) -> float:
                bx = (int(mid_a_bp) // resolution) * resolution
                by = (int(mid_b_bp) // resolution) * resolution
                return oe_map.get((bx, by), np.nan)

            oe_vals = np.fromiter(
                (lookup_oe(a, b) for a, b in zip(pairs.mid_a.values, pairs.mid_b.values)),
                dtype=np.float64,
                count=len(pairs),
            )
        except Exception as exc:
            print(
                f"[heavy] {chrom_a} x {chrom_b}: oe fetch failed ({exc}); leaving NaN",
                file=sys.stderr,
            )

    out = pairs.copy()
    out["hic_obs"] = obs_vals
    out["hic_oe"] = oe_vals
    out["hic_norm"] = norm
    out["hic_resolution"] = resolution
    return out


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def shard_filename(shards_dir: Path, chrom_a: str, chrom_b: str) -> Path:
    return shards_dir / f"{chrom_a}_{chrom_b}.parquet"


def list_shards() -> list[tuple[str, str]]:
    """Unordered (chrom_a, chrom_b) pairs covering the canonical chroms."""
    out = []
    for i, a in enumerate(CANONICAL_CHROMS):
        for b in CANONICAL_CHROMS[i:]:
            out.append((a, b))
    return out


def run_shard(
    chrom_a: str,
    chrom_b: str,
    pairs: pd.DataFrame,
    channel_map_csv: Path,
    hic_path: Path,
    shards_dir: Path,
    resolution: int,
    norm_intra: str,
    norm_inter: str,
    force: bool,
) -> None:
    out_path = shard_filename(shards_dir, chrom_a, chrom_b)
    if out_path.exists() and not force:
        print(f"[heavy] skip {chrom_a} x {chrom_b} (exists)", file=sys.stderr)
        return
    t0 = time.time()
    sub = shard_pairs(pairs, chrom_a, chrom_b)
    if len(sub) == 0:
        # Write an empty shard so we don't keep re-evaluating it.
        write_atomic(
            pd.DataFrame(
                columns=list(pairs.columns) + ["hic_obs", "hic_oe", "hic_norm", "hic_resolution"]
            ),
            out_path,
        )
        print(f"[heavy] {chrom_a} x {chrom_b}: 0 pairs, empty shard", file=sys.stderr)
        return
    sub = attach_channels(sub, channel_map_csv)
    shard = query_hic_shard(
        hic_path, chrom_a, chrom_b, sub, resolution, norm_intra, norm_inter
    )
    write_atomic(shard, out_path)
    n_nonzero = int((shard.hic_obs > 0).sum())
    elapsed = time.time() - t0
    print(
        f"[heavy] {chrom_a} x {chrom_b}: {len(shard)} pairs, "
        f"{n_nonzero} with obs>0 ({elapsed:.1f}s)",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paralogs", type=Path, default=Path("data/paralogs.tsv"))
    ap.add_argument("--channels", type=Path, default=Path("../../data/channel_gene_map.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache/paralogs_canonical.parquet"))
    ap.add_argument("--hic", type=Path, help="Path to .hic file")
    ap.add_argument("--cell-line", default="gm12878", help="subdir name under data/heavy/")
    ap.add_argument("--resolution", type=int, default=25_000)
    ap.add_argument("--norm-intra", default="KR")
    ap.add_argument("--norm-inter", default="NONE")
    ap.add_argument(
        "--shard",
        nargs=2,
        metavar=("CHROM_A", "CHROM_B"),
        help="Run one shard (e.g. --shard 17 17).",
    )
    ap.add_argument(
        "--all", action="store_true", help="Run all shards serially in this process."
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Run all shards with N parallel worker subprocesses (each worker is "
             "its own Python process, hic file opened independently). 0 = serial.",
    )
    ap.add_argument(
        "--list", action="store_true", help="Print one 'A B' line per shard and exit."
    )
    ap.add_argument(
        "--cache-paralogs",
        action="store_true",
        help="(Re)build the canonical paralog cache and exit.",
    )
    ap.add_argument(
        "--force", action="store_true", help="Re-run shards even if output exists."
    )
    args = ap.parse_args()

    here = Path(__file__).parent
    rel = lambda p: p if p.is_absolute() else here / p
    paralogs_tsv = rel(args.paralogs)
    channels_csv = rel(args.channels)
    cache_path = rel(args.cache)

    if args.list:
        for a, b in list_shards():
            print(f"{a} {b}")
        return 0

    if args.cache_paralogs:
        build_canonical_paralogs(paralogs_tsv, cache_path)
        return 0

    if not args.hic:
        print("error: --hic is required for shard runs", file=sys.stderr)
        return 2
    hic_path = rel(args.hic)
    shards_dir = here / "data" / "heavy" / args.cell_line / "shards"

    pairs = load_canonical_paralogs(cache_path, paralogs_tsv)

    if args.shard:
        a, b = args.shard
        run_shard(
            a, b, pairs, channels_csv, hic_path, shards_dir,
            args.resolution, args.norm_intra, args.norm_inter, args.force,
        )
        return 0

    if args.all:
        for a, b in list_shards():
            run_shard(
                a, b, pairs, channels_csv, hic_path, shards_dir,
                args.resolution, args.norm_intra, args.norm_inter, args.force,
            )
        return 0

    if args.parallel > 0:
        # Driver mode: spawn one subprocess per remaining shard, N at a time.
        # Each subprocess re-invokes this script with --shard A B and opens
        # its own HiCFile handle — no fork-safety concerns.
        script = Path(__file__).resolve()
        shards_todo = [
            (a, b) for a, b in list_shards()
            if args.force or not shard_filename(shards_dir, a, b).exists()
        ]
        print(
            f"[heavy] driver: {len(shards_todo)} shards todo "
            f"(par={args.parallel}, cell={args.cell_line})",
            file=sys.stderr,
        )
        if not shards_todo:
            return 0

        def spawn(a: str, b: str) -> tuple[str, str, int, str]:
            cmd = [
                sys.executable, str(script),
                "--shard", a, b,
                "--hic", str(hic_path),
                "--cell-line", args.cell_line,
                "--paralogs", str(paralogs_tsv),
                "--channels", str(channels_csv),
                "--cache", str(cache_path),
                "--resolution", str(args.resolution),
                "--norm-intra", args.norm_intra,
                "--norm-inter", args.norm_inter,
            ]
            if args.force:
                cmd.append("--force")
            res = subprocess.run(cmd, capture_output=True, text=True)
            return a, b, res.returncode, (res.stderr or "").strip()

        failures = []
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futs = [pool.submit(spawn, a, b) for (a, b) in shards_todo]
            for fut in as_completed(futs):
                a, b, rc, stderr = fut.result()
                tag = "ok" if rc == 0 else f"FAIL rc={rc}"
                last = stderr.splitlines()[-1] if stderr else ""
                print(f"[heavy] {a} x {b}: {tag} | {last}", file=sys.stderr)
                if rc != 0:
                    failures.append((a, b, stderr))

        if failures:
            print(f"[heavy] {len(failures)} shard(s) failed", file=sys.stderr)
            for a, b, stderr in failures[:5]:
                print(f"  {a} x {b}:\n{stderr}", file=sys.stderr)
            return 1
        return 0

    print("error: pass --shard A B, --all, --parallel N, --list, or --cache-paralogs", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
