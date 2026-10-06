"""
@file compute_norm_stats.py
@brief Phase 1: compute per-channel normalization statistics for the fixed-uV
       normalization option.

@details
The baseline z-scores every segment per channel, which removes absolute
amplitude and inter-channel amplitude asymmetry — both clinically meaningful
(low-voltage records, hemispheric asymmetry). The alternative is a *fixed*
affine transform: clip at +/-`clip_uv`, subtract a per-channel mean, divide by a
per-channel standard deviation, where mean and std are constants estimated once
from data. Amplitude information then survives into the model.

Statistics are estimated from the **tuning pool (T) only**. T never contributes
to a reported metric, so nothing derived from it can leak into the
cross-validation or held-out eval results. Estimating from E instead would put
(a very aggregate function of) test patients' data into the input transform.

Accumulates in float64 over a random subsample of segments per recording, so
the pass is a few GB of reads rather than the full 71 GB.

@par Privacy
Output is 19 means, 19 standard deviations and a few counts: aggregate
statistics with no identifiers, safe to commit. The split file's SHA-256 is
recorded so the stats can be tied to the exact pool they came from.

@par Usage (from repo root)
@verbatim
python -m scripts.compute_norm_stats
python -m scripts.compute_norm_stats --segments-per-recording 100 --clip-uv 500
@endverbatim
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_cnn_lstm.preprocessing.preprocessing import STANDARD_1020_CHANNELS

DEFAULT_SPLITS = Path("/shared/rc/eeg-cnn-lstm/data/splits/tuab_train_splits.csv")
DEFAULT_DATA_DIR = Path(
    "/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train"
)
DEFAULT_OUT = Path("configs/norm_stats_tuab.json")


def sha256_of(path: Path) -> str:
    """
    @brief SHA-256 of a file, used to tie stats to a specific split file.

    @param path File to hash.
    @return Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    """@brief CLI entrypoint. @return 0 on success, non-zero on error."""
    p = argparse.ArgumentParser(description="Per-channel normalization stats from pool T")
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--pool", default="T", help="pool to estimate from (default T)")
    p.add_argument("--segments-per-recording", type=int, default=50)
    p.add_argument("--clip-uv", type=float, default=800.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    splits = pd.read_csv(args.splits)
    pool = splits[splits["pool"] == args.pool]
    if pool.empty:
        print(f"ERROR: no rows with pool == {args.pool!r}", file=sys.stderr)
        return 2
    print(
        f"Pool {args.pool}: {len(pool)} recordings, "
        f"{pool['subject_id'].nunique()} patients; "
        f"sampling up to {args.segments_per_recording} segments each"
    )

    n_ch = len(STANDARD_1020_CHANNELS)
    rng = np.random.default_rng(args.seed)
    ch_sum = np.zeros(n_ch, dtype=np.float64)
    ch_sumsq = np.zeros(n_ch, dtype=np.float64)
    n_values = 0            # per channel
    n_clipped = 0           # values hitting the clip threshold
    n_segments = 0
    n_recordings = 0
    abs_max = 0.0
    missing: list[str] = []

    for i, row in enumerate(pool.itertuples(index=False), 1):
        path = args.data_dir / row.filename
        if not path.is_file():
            missing.append(row.filename)
            continue
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]
        k = min(args.segments_per_recording, n)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        x = np.asarray(arr[idx], dtype=np.float64)          # (k, 19, 2500), uV

        abs_max = max(abs_max, float(np.abs(x).max()))
        n_clipped += int((np.abs(x) > args.clip_uv).sum())
        np.clip(x, -args.clip_uv, args.clip_uv, out=x)

        ch_sum += x.sum(axis=(0, 2))
        ch_sumsq += (x**2).sum(axis=(0, 2))
        n_values += x.shape[0] * x.shape[2]
        n_segments += k
        n_recordings += 1

        if i % 100 == 0 or i == len(pool):
            print(f"  {i}/{len(pool)} recordings", flush=True)

    if n_values == 0:
        print("ERROR: no data read", file=sys.stderr)
        return 2

    mean = ch_sum / n_values
    var = ch_sumsq / n_values - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    # Pooled std across all channels (alternative single-scalar divisor).
    global_var = ch_sumsq.sum() / (n_values * n_ch) - (ch_sum.sum() / (n_values * n_ch)) ** 2
    global_std = float(np.sqrt(max(global_var, 1e-12)))

    stats = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimated_from_pool": args.pool,
        "splits_csv": str(args.splits),
        "splits_csv_sha256": sha256_of(args.splits),
        "data_dir": str(args.data_dir),
        "units": "uV",
        "clip_uv": args.clip_uv,
        "channels": list(STANDARD_1020_CHANNELS),
        "mean_uv": [round(float(v), 4) for v in mean],
        "std_uv": [round(float(v), 4) for v in std],
        "global_std_uv": round(global_std, 4),
        "n_recordings": n_recordings,
        "n_segments": n_segments,
        "n_values_per_channel": int(n_values),
        "segments_per_recording": args.segments_per_recording,
        "seed": args.seed,
        "abs_max_before_clip_uv": round(abs_max, 2),
        "clipped_value_fraction": round(n_clipped / (n_values * n_ch), 8),
        "missing_files": missing,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))

    print(f"\nWrote {args.out}")
    print(f"  segments used: {n_segments:,} from {n_recordings} recordings")
    print(f"  clip: +/-{args.clip_uv} uV; max seen {abs_max:,.0f} uV; "
          f"clipped {stats['clipped_value_fraction']:.6%} of values")
    print(f"  global std: {global_std:.2f} uV\n")
    print(f"{'channel':>8} {'mean_uV':>9} {'std_uV':>8}")
    for c, m, s in zip(STANDARD_1020_CHANNELS, mean, std):
        print(f"{c:>8} {m:>9.3f} {s:>8.3f}")
    if missing:
        print(f"\n!! {len(missing)} files missing, e.g. {missing[:3]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())