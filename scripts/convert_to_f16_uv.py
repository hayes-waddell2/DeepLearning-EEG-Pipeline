"""
@file convert_to_f16_uv.py
@brief convert preprocessed TUAB epochs from float64 volts to
       float16 (default) or float32 microvolts.

@details
Reads each `<recording>_epochs.npy` listed in a split's manifest, multiplies by
1e6 (V -> uV), casts to the target dtype, and writes it to a mirrored directory
tree. Originals are never modified.

Design notes:
  - Streams each recording in chunks, so memory stays bounded (~100 MB/worker).
  - Writes to `<name>.npy.tmp` then atomically renames, so an interrupted job
    never leaves a half-written file that looks valid.
  - Resumable: files already converted (matching shape + dtype) are skipped.
  - Per-file spot check: re-reads a few epochs and records the max absolute
    conversion error in uV.
  - Logs per-file stats (abs max, clipped count, non-finite count, error) to a
    CSV and writes a `conversion_meta.json` for provenance.

@par Layout (mirrors the existing one, including the doubled split folder):
@verbatim
<src-root>/<split>/<split>_manifest.csv
<src-root>/<split>/<split>/<recording>_epochs.npy
        -> <dst-root>/<split>/<split>_manifest.csv
           <dst-root>/<split>/<split>/<recording>_epochs.npy
           <dst-root>/<split>/conversion_log_<split>.csv
           <dst-root>/conversion_meta.json
@endverbatim

@par Usage (from repo root):
@verbatim
# small test run
python -m scripts.convert_to_f16_uv --splits train --limit 20 \
    --dst-root /shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv_test
# full run (normally via jobs/convert_f16.sh)
python -m scripts.convert_to_f16_uv --workers 16
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SRC = Path("/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab")
DEFAULT_DST = Path("/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv")

## @brief Epochs processed per chunk (256 x 19 x 2500 x 8 B ~= 97 MB float64).
CHUNK = 256

## @brief Epochs re-read per file for the post-write spot check.
N_CHECK_EPOCHS = 3


def _convert_one(job: tuple) -> dict:
    """
    @brief Convert one recording file. Runs inside a worker process.

    @param job Tuple `(src, dst, scale, dtype_name, overwrite, seed)`.
    @return Row dict for the conversion log.
    """
    src, dst, scale, dtype_name, overwrite, seed = job
    row: dict = {"filename": src.name}
    t0 = time.time()
    try:
        dtype = np.dtype(dtype_name)
        dmax = float(np.finfo(dtype).max)
        x = np.load(src, mmap_mode="r")
        row["n_epochs"] = int(x.shape[0])

        if dst.exists() and not overwrite:
            try:
                y = np.load(dst, mmap_mode="r")
                if y.shape == x.shape and y.dtype == dtype:
                    row["status"] = "skipped_exists"
                    return row
            except Exception:
                pass  # unreadable/partial -> reconvert

        tmp = dst.with_name(dst.name + ".tmp")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=x.shape)

        abs_max = 0.0
        n_clipped = 0
        n_nonfinite = 0
        for i in range(0, x.shape[0], CHUNK):
            block = np.asarray(x[i : i + CHUNK], dtype=np.float64) * scale
            finite = np.isfinite(block)
            n_nonfinite += int((~finite).sum())
            if finite.any():
                babs = np.abs(block[finite])
                abs_max = max(abs_max, float(babs.max()))
                n_clipped += int((babs > dmax).sum())
            np.clip(block, -dmax, dmax, out=block)
            out[i : i + CHUNK] = block.astype(dtype)
        out.flush()
        del out
        os.replace(tmp, dst)

        # Spot check: re-read a few epochs and compare against the source.
        y = np.load(dst, mmap_mode="r")
        rng = np.random.default_rng(seed)
        k = min(N_CHECK_EPOCHS, x.shape[0])
        idx = np.sort(rng.choice(x.shape[0], size=k, replace=False))
        ref = np.asarray(x[idx], dtype=np.float64) * scale
        got = np.asarray(y[idx], dtype=np.float64)
        ok = np.isfinite(ref)
        err = np.abs(got[ok] - np.clip(ref[ok], -dmax, dmax))

        row.update(
            status="converted",
            abs_max_uv=round(abs_max, 3),
            n_clipped=n_clipped,
            n_nonfinite=n_nonfinite,
            max_abs_err_uv=float(err.max()) if err.size else 0.0,
            seconds=round(time.time() - t0, 2),
        )
    except Exception as exc:  # noqa: BLE001 - log and continue
        row.update(status=f"error: {type(exc).__name__}: {exc}")
    return row


def _git_commit() -> str | None:
    """@brief Return the current git commit hash, or None if unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def convert_split(args: argparse.Namespace, split: str) -> dict:
    """
    @brief Convert every recording in one split's manifest.

    @param args Parsed CLI arguments.
    @param split "train" or "eval".
    @return Summary dict for this split.
    """
    src_dir = args.src_root / split / split
    src_manifest = args.src_root / split / f"{split}_manifest.csv"
    dst_dir = args.dst_root / split / split
    dst_manifest = args.dst_root / split / f"{split}_manifest.csv"

    if not src_manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {src_manifest}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(src_manifest)
    if args.limit:
        manifest = manifest.iloc[: args.limit].copy()

    jobs = [
        (src_dir / fn, dst_dir / fn, args.scale, args.dtype, args.overwrite, i)
        for i, fn in enumerate(manifest["filename"])
    ]
    missing = [str(j[0]) for j in jobs if not j[0].is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source files missing, e.g. {missing[:3]}")

    print(f"[{split}] converting {len(jobs)} recordings with {args.workers} workers")
    rows: list[dict] = []
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=1), 1):
            rows.append(row)
            if n % 100 == 0 or n == len(jobs):
                print(f"  [{split}] {n}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    log = pd.DataFrame(rows).sort_values("filename")
    log.to_csv(args.dst_root / split / f"conversion_log_{split}.csv", index=False)
    manifest.to_csv(dst_manifest, index=False)  # same columns; subset if --limit

    errors = log[log["status"].str.startswith("error")]
    conv = log[log["status"] == "converted"]
    summary = {
        "n_files": int(len(log)),
        "converted": int(len(conv)),
        "skipped_exists": int((log["status"] == "skipped_exists").sum()),
        "errors": int(len(errors)),
        "total_clipped_values": int(conv["n_clipped"].sum()) if len(conv) else 0,
        "total_nonfinite_values": int(conv["n_nonfinite"].sum()) if len(conv) else 0,
        "max_abs_uv": float(conv["abs_max_uv"].max()) if len(conv) else None,
        "median_file_abs_max_uv": float(conv["abs_max_uv"].median()) if len(conv) else None,
        "max_abs_err_uv": float(conv["max_abs_err_uv"].max()) if len(conv) else None,
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"[{split}] summary: {json.dumps(summary)}")
    if len(errors):
        print(f"[{split}] !! errors (first 5):\n{errors.head().to_string()}", file=sys.stderr)
    return summary


def main() -> int:
    """@brief CLI entrypoint. @return 0 on success, 1 if any file failed."""
    parser = argparse.ArgumentParser(description="Convert TUAB epochs to float16/32 uV")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST)
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--scale", type=float, default=1e6, help="V -> uV multiplier")
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    )
    parser.add_argument("--limit", type=int, default=0, help="Only first N recordings (testing)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dst_root.resolve() == args.src_root.resolve():
        print("ERROR: dst-root must differ from src-root", file=sys.stderr)
        return 2
    args.dst_root.mkdir(parents=True, exist_ok=True)

    summaries = {split: convert_split(args, split) for split in args.splits}

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "src_root": str(args.src_root),
        "dst_root": str(args.dst_root),
        "dtype": args.dtype,
        "units": "uV",
        "scale_applied": args.scale,
        "source_dtype": "float64",
        "source_units": "V (MNE default)",
        "limit": args.limit,
        "git_commit": _git_commit(),
        "numpy_version": np.__version__,
        "host": platform.node(),
        "splits": summaries,
    }
    (args.dst_root / "conversion_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.dst_root / 'conversion_meta.json'}")

    return 1 if any(s["errors"] for s in summaries.values()) else 0


if __name__ == "__main__":
    sys.exit(main())