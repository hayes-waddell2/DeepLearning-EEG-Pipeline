"""
@file compare_runs.py
@brief Phase 2 / Step 1: compare runs and aggregation methods from saved
       per-segment predictions.

@details
Reads the `predictions_*.npz` files written by `train.py` (segment logits, the
recording each belongs to, and per-recording labels and patient IDs) and
produces the two tables the normalization A/B needs:

  1. **Run comparison** — one row per run, at a chosen aggregator, with
     recording-level AUC, partial AUC, sensitivity at a specificity floor, and
     specificity at a sensitivity floor, each with a patient-clustered
     bootstrap confidence interval.
  2. **Aggregator sweep** — for every run, all aggregation methods ranked by
     partial AUC.

Nothing here retrains anything: aggregation, thresholds and calibration are
post-hoc, so every option is compared on identical model output in seconds.

@par Why the bootstrap is clustered by patient
A patient can contribute up to 9 recordings in TUAB and those are correlated,
so resampling recordings would understate the uncertainty. Patient IDs are
saved alongside the predictions precisely so this can be done correctly.

@par Usage (from repo root)
@verbatim
# compare two runs
python -m scripts.compare_runs \
    zscore=/shared/rc/eeg-cnn-lstm/runs/ab-zscore_123/predictions_tune.npz \
    uv_scale=/shared/rc/eeg-cnn-lstm/runs/ab-uvscale_124/predictions_tune.npz

# one run, full aggregator sweep, saved as JSON + CSV
python -m scripts.compare_runs run=<path>.npz --out results/phase2/ab_normalization
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eeg_cnn_lstm.utils.metrics import (
    AGGREGATORS,
    DEFAULT_MAX_FPR,
    DEFAULT_SENS_FLOOR,
    DEFAULT_SPEC_FLOOR,
    aggregate_to_recordings,
    bootstrap_ci,
    compute_metrics,
    partial_auc,
    recording_metrics,
    sensitivity_at_specificity,
    sigmoid,
    specificity_at_sensitivity,
    sweep_aggregators,
)

PAUC_KEY = f"pauc_fpr{DEFAULT_MAX_FPR:g}"
SENS_KEY = f"sens_at_spec{DEFAULT_SPEC_FLOOR:g}"
SPEC_KEY = f"spec_at_sens{DEFAULT_SENS_FLOOR:g}"


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    """
    @brief Load one `predictions_*.npz` written by `train.py`.

    @param path Path to the `.npz` file.
    @return Dict of arrays: segment `logits`/`labels`/`rec_idx` plus
            per-recording `rec_label`, `rec_subject_id`, `rec_filename`.
    @throws FileNotFoundError If the file does not exist.
    @throws KeyError If an expected array is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"predictions file not found: {path}")
    z = np.load(path, allow_pickle=False)
    required = ("logits", "labels", "rec_idx", "rec_label", "rec_subject_id")
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    return {k: z[k] for k in z.files}


def segment_summary(pred: dict[str, np.ndarray]) -> dict[str, float | int]:
    """
    @brief Segment-level metrics, for comparison with the recording level.

    @param pred Output of `load_predictions`.
    @return Metric dict from `compute_metrics`.
    """
    import torch  # local import: only needed for the Phase 0 metric signature

    return compute_metrics(
        torch.from_numpy(pred["logits"].astype(np.float32)),
        torch.from_numpy(pred["labels"].astype(np.float32)),
    )


def run_row(
    name: str,
    pred: dict[str, np.ndarray],
    aggregator: str = "mean",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    @brief One comparison row: recording-level metrics with clustered CIs.

    @param name Label for the run.
    @param pred Output of `load_predictions`.
    @param aggregator Segment-to-recording method.
    @param n_boot Bootstrap replicates (0 disables the intervals).
    @param seed RNG seed for the bootstrap.
    @return Dict of metrics, CI bounds, and sizes.
    """
    probs = sigmoid(pred["logits"].astype(np.float64))
    rec_labels = pred["rec_label"].astype(int)
    patients = pred["rec_subject_id"]
    scores = aggregate_to_recordings(
        probs, pred["rec_idx"], aggregator, n_recordings=len(rec_labels)
    )

    seg = segment_summary(pred)
    rec = recording_metrics(rec_labels, scores)

    row: dict[str, Any] = {
        "run": name,
        "aggregator": aggregator,
        "recordings": rec.get("n_recordings"),
        "patients": int(len(np.unique(patients))),
        "segments": int(len(probs)),
        "segment_auc": seg.get("auc_roc"),
        "auc": rec.get("auc_roc"),
        "pauc": rec.get(PAUC_KEY),
        "sens_at_spec90": rec.get(SENS_KEY),
        "spec_at_sens95": rec.get(SPEC_KEY),
        "threshold_at_sens95": rec.get("threshold"),
        "sens_at_threshold": rec.get("sensitivity"),
        "spec_at_threshold": rec.get("specificity"),
    }

    if n_boot:
        for key, fn in (
            ("auc", lambda y, s: float(roc_auc_score(y, s))),
            ("pauc", lambda y, s: partial_auc(y, s)),
            ("sens_at_spec90", lambda y, s: sensitivity_at_specificity(y, s)[0]),
            ("spec_at_sens95", lambda y, s: specificity_at_sensitivity(y, s)[0]),
        ):
            _, lo, hi = bootstrap_ci(
                rec_labels, scores, fn, groups=patients, n_boot=n_boot, seed=seed
            )
            row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
    return row


def fmt(v: Any, width: int = 7, places: int = 4) -> str:
    """
    @brief Format a possibly-missing float for table output.

    @param v Value to format.
    @param width Field width.
    @param places Decimal places.
    @return Right-aligned string, or "n/a" when the value is missing.
    """
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{places}f}"


def print_comparison(rows: list[dict[str, Any]], n_boot: int) -> None:
    """
    @brief Print the run-comparison table.

    @param rows Output of `run_row`, one per run.
    @param n_boot Whether intervals were computed (0 = no).
    """
    print(f"\n{'=' * 96}\nRUN COMPARISON (recording level)\n{'=' * 96}")
    head = (
        f"{'run':<14}{'agg':<10}{'recs':>6}{'pats':>6}"
        f"{'segAUC':>9}{'AUC':>9}{'pAUC':>9}{'sens@sp.9':>11}{'spec@se.95':>12}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['run']:<14}{r['aggregator']:<10}{r['recordings']:>6}{r['patients']:>6}"
            f"{fmt(r['segment_auc'], 9)}{fmt(r['auc'], 9)}{fmt(r['pauc'], 9)}"
            f"{fmt(r['sens_at_spec90'], 11)}{fmt(r['spec_at_sens95'], 12)}"
        )
    if n_boot:
        print(f"\n95% CIs (patient-clustered bootstrap, {n_boot} replicates):")
        for r in rows:
            print(
                f"  {r['run']:<14} "
                f"AUC {fmt(r['auc'], 6)} [{fmt(r.get('auc_lo'), 6)},{fmt(r.get('auc_hi'), 6)}]   "
                f"pAUC {fmt(r['pauc'], 6)} [{fmt(r.get('pauc_lo'), 6)},{fmt(r.get('pauc_hi'), 6)}]   "
                f"spec@sens.95 {fmt(r['spec_at_sens95'], 6)} "
                f"[{fmt(r.get('spec_at_sens95_lo'), 6)},{fmt(r.get('spec_at_sens95_hi'), 6)}]"
            )

    if len(rows) == 2:
        a, b = rows
        print(
            f"\nDelta ({b['run']} - {a['run']}): "
            f"AUC {b['auc'] - a['auc']:+.4f}   pAUC {b['pauc'] - a['pauc']:+.4f}"
        )
        print(
            "  Overlapping CIs mean the difference is not resolved by this comparison."
        )


def print_sweep(name: str, rows: list[dict[str, Any]]) -> None:
    """
    @brief Print the aggregator sweep for one run.

    @param name Run label.
    @param rows Output of `sweep_aggregators`.
    """
    print(f"\n{'=' * 96}\nAGGREGATOR SWEEP: {name}  (ranked by pAUC)\n{'=' * 96}")
    head = f"{'aggregator':<18}{'AUC':>9}{'pAUC':>9}{'sens@sp.9':>11}{'spec@se.95':>12}{'sens':>8}{'spec':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{str(r['aggregator']):<18}{fmt(r.get('auc_roc'), 9)}{fmt(r.get(PAUC_KEY), 9)}"
            f"{fmt(r.get(SENS_KEY), 11)}{fmt(r.get(SPEC_KEY), 12)}"
            f"{fmt(r.get('sensitivity'), 8)}{fmt(r.get('specificity'), 8)}"
        )


def main() -> int:
    """@brief CLI entrypoint. @return 0 on success, non-zero on bad input."""
    p = argparse.ArgumentParser(
        description="Compare runs and aggregators from saved predictions"
    )
    p.add_argument(
        "runs",
        nargs="+",
        metavar="NAME=PATH",
        help="one or more runs, e.g. zscore=/path/predictions_tune.npz",
    )
    p.add_argument("--aggregator", default="mean", help="aggregator for the run table")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-sweep", action="store_true", help="skip the aggregator sweep")
    p.add_argument(
        "--out",
        default=None,
        help="path prefix for saved results (writes <prefix>.json and <prefix>_sweep.csv)",
    )
    args = p.parse_args()

    runs: list[tuple[str, dict[str, np.ndarray]]] = []
    for spec in args.runs:
        if "=" not in spec:
            print(f"ERROR: expected NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        runs.append((name, load_predictions(path)))

    if args.aggregator not in AGGREGATORS:
        print(
            f"ERROR: unknown aggregator {args.aggregator!r}; "
            f"choose from {sorted(AGGREGATORS)}",
            file=sys.stderr,
        )
        return 2

    rows = [
        run_row(name, pred, args.aggregator, args.n_boot, args.seed)
        for name, pred in runs
    ]
    print_comparison(rows, args.n_boot)

    sweeps: dict[str, list[dict[str, Any]]] = {}
    if not args.no_sweep:
        for name, pred in runs:
            probs = sigmoid(pred["logits"].astype(np.float64))
            sweep = sweep_aggregators(
                probs, pred["rec_idx"], pred["rec_label"].astype(int)
            )
            sweeps[name] = sweep
            print_sweep(name, sweep)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": {name: spec for name, spec in zip([r[0] for r in runs], args.runs)},
            "aggregator": args.aggregator,
            "n_boot": args.n_boot,
            "comparison": rows,
            "sweeps": sweeps,
        }
        out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
        pd.DataFrame(rows).to_csv(out.parent / f"{out.name}_comparison.csv", index=False)
        if sweeps:
            flat = [dict(run=n, **r) for n, rs in sweeps.items() for r in rs]
            pd.DataFrame(flat).to_csv(out.parent / f"{out.name}_sweep.csv", index=False)
        print(f"\nWrote {out.with_suffix('.json')} and CSVs alongside it.")
        print(
            "Note: these files contain aggregate metrics only (no filenames or "
            "patient IDs), so they are safe to commit."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())