"""
@file make_splits.py
@brief Phase 1 / Step 1: create the fixed patient-level splits used by every
       later experiment (HPO, cross-validation, calibration, final model).

@details
Produces two pools plus the CV folds, all grouped by patient so that no patient
ever appears on two sides of any split:

  - Pool T (tuning, ~20% of patients): every tuned decision is scored here —
    hyperparameters, normalization, aggregation, early stopping, calibration,
    decision threshold. T results are never reported as performance.
  - Pool E (~80% of patients): split into K folds. Each fold trains on K-1
    folds and is tested on the held-out one, so every E patient is predicted
    exactly once by a model that never trained on them. This is the reported
    generalization estimate. The final model trains on all of E.

Both splits use `StratifiedGroupKFold` with `subject_id` as the group and the
recording-level label as the stratification target. Recording-level labels are
used (not a per-patient summary label) because TUAB is balanced by recording
(50/50) but not by patient (59/41): abnormal patients have more sessions.

@par Privacy (data use agreement)
The split CSV contains recording filenames and subject IDs, so it is
dataset-derived and is written OUTSIDE the repository (default:
/shared/rc/eeg-cnn-lstm/data/splits/). Reproducibility does not depend on
committing it: this script plus the manifest plus `--seed` regenerate it
byte-for-byte, and the summary JSON (counts only, no identifiers) records the
CSV's SHA-256 so a run can be tied to an exact split.

@par Outputs
@verbatim
<out-dir>/tuab_train_splits.csv    filename,subject_id,label,n_epochs,pool,fold
                                   pool = "T" | "E";  fold = 0..K-1 (E), -1 (T)
<summary>                          aggregate counts + sha256 + seed (repo-safe)
@endverbatim

@par Usage (from repo root)
@verbatim
python -m scripts.make_splits                      # defaults
python -m scripts.make_splits --n-folds 5 --seed 7 --overwrite
@endverbatim
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from eeg_cnn_lstm.utils.dataset import load_manifest

DEFAULT_MANIFEST = Path(
    "/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train_manifest.csv"
)
DEFAULT_OUT_DIR = Path("/shared/rc/eeg-cnn-lstm/data/splits")
DEFAULT_SUMMARY = Path("results/phase1/splits_summary.json")

## @brief `fold` value used for rows in the tuning pool.
TUNE_FOLD = -1


def sha256_of(path: Path) -> str:
    """
    @brief SHA-256 of a file, for tying results to an exact split file.

    @param path File to hash.
    @return Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    """@brief Current git commit hash, or None if unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def grouped_holdout(
    manifest: pd.DataFrame, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    @brief Hold out approximately `frac` of patients, stratified by recording label.

    @details
    Implemented as one fold of a `StratifiedGroupKFold` with
    `n_splits = round(1 / frac)`; taking a single fold as the held-out pool
    gives a patient-disjoint, label-stratified split of the requested size.

    @param manifest Manifest with `subject_id` and `label` columns.
    @param frac Fraction of patients to hold out (e.g. 0.2).
    @param seed RNG seed.
    @return Tuple `(rest_idx, held_idx)` of positional row indices.
    @throws ValueError If `frac` does not yield at least 2 splits.
    """
    n_splits = int(round(1.0 / frac))
    if n_splits < 2:
        raise ValueError(f"--tune-frac {frac} is too large; needs 1/frac >= 2")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(
        splitter.split(manifest, manifest["label"], groups=manifest["subject_id"])
    )
    return rest_idx, held_idx


def assign_folds(pool: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """
    @brief Assign each recording in a pool to one of `n_folds` patient-disjoint folds.

    @param pool Rows of the E pool (must contain `subject_id` and `label`).
    @param n_folds Number of folds.
    @param seed RNG seed.
    @return Series of fold indices aligned to `pool`'s index.
    """
    folds = pd.Series(index=pool.index, dtype="int64")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(
        splitter.split(pool, pool["label"], groups=pool["subject_id"])
    ):
        folds.iloc[test_idx] = k
    return folds


def validate(splits: pd.DataFrame, manifest: pd.DataFrame, n_folds: int) -> None:
    """
    @brief Assert the split file satisfies every disjointness requirement.

    @param splits The assembled split table.
    @param manifest The source manifest.
    @param n_folds Expected number of folds.
    @throws AssertionError If any invariant fails.
    """
    assert len(splits) == len(manifest), "row count changed"
    assert splits["filename"].is_unique, "duplicate filenames"
    assert set(splits["filename"]) == set(manifest["filename"]), "filename set changed"

    # Every patient sits in exactly one pool.
    pools_per_patient = splits.groupby("subject_id")["pool"].nunique()
    assert (pools_per_patient == 1).all(), "patient(s) appear in both pools"

    # Every E patient sits in exactly one fold; T rows carry TUNE_FOLD.
    e_rows = splits[splits["pool"] == "E"]
    t_rows = splits[splits["pool"] == "T"]
    assert (t_rows["fold"] == TUNE_FOLD).all(), "T rows must have fold == -1"
    folds_per_patient = e_rows.groupby("subject_id")["fold"].nunique()
    assert (folds_per_patient == 1).all(), "patient(s) span multiple folds"
    assert sorted(e_rows["fold"].unique()) == list(range(n_folds)), "fold ids wrong"

    # No patient shared between any two folds (implied above, checked directly).
    per_fold = {k: set(g["subject_id"]) for k, g in e_rows.groupby("fold")}
    for a in range(n_folds):
        for b in range(a + 1, n_folds):
            assert not (per_fold[a] & per_fold[b]), f"folds {a},{b} share patients"
    assert not (set(t_rows["subject_id"]) & set(e_rows["subject_id"])), "T/E overlap"


def summarize(splits: pd.DataFrame, n_folds: int) -> dict:
    """
    @brief Build the aggregate (identifier-free) summary of a split table.

    @param splits The assembled split table.
    @param n_folds Number of folds.
    @return Dict of counts safe to commit to the repository.
    """

    def block(df: pd.DataFrame) -> dict:
        return {
            "recordings": int(len(df)),
            "patients": int(df["subject_id"].nunique()),
            "segments": int(df["n_epochs"].sum()),
            "abnormal_recordings": int(df["label"].sum()),
            "abnormal_frac": round(float(df["label"].mean()), 4),
        }

    e_rows = splits[splits["pool"] == "E"]
    return {
        "total": block(splits),
        "tuning_pool_T": block(splits[splits["pool"] == "T"]),
        "cv_pool_E": block(e_rows),
        "folds": {str(k): block(g) for k, g in e_rows.groupby("fold")},
        "n_folds": n_folds,
    }


def main() -> int:
    """@brief CLI entrypoint. @return 0 on success, non-zero on refusal/failure."""
    p = argparse.ArgumentParser(description="Create patient-level T/E pools and CV folds")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--tune-frac", type=float, default=0.2, help="patient fraction for T")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="replace an existing split file")
    args = p.parse_args()

    out_csv = args.out_dir / "tuab_train_splits.csv"
    if out_csv.exists() and not args.overwrite:
        print(
            f"ERROR: {out_csv} already exists.\n"
            "Splits must stay fixed once experiments have used them. "
            "Pass --overwrite only if you intend to invalidate earlier results.",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest(args.manifest).sort_values("filename").reset_index(drop=True)
    print(
        f"Manifest: {len(manifest)} recordings, "
        f"{manifest['subject_id'].nunique()} patients, "
        f"{int(manifest['n_epochs'].sum())} segments"
    )

    # ---- Pool T (tuning) vs pool E (train/test) ----
    e_idx, t_idx = grouped_holdout(manifest, args.tune_frac, args.seed)
    splits = manifest.copy()
    splits["pool"] = "E"
    splits.loc[t_idx, "pool"] = "T"
    splits["fold"] = TUNE_FOLD

    # ---- Folds within E ----
    e_rows = splits[splits["pool"] == "E"]
    splits.loc[e_rows.index, "fold"] = assign_folds(e_rows, args.n_folds, args.seed)

    splits = splits[["filename", "subject_id", "label", "n_epochs", "pool", "fold"]]
    splits = splits.sort_values("filename").reset_index(drop=True)

    validate(splits, manifest, args.n_folds)

    # ---- Write ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_csv, index=False)
    digest = sha256_of(out_csv)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "split_csv": str(out_csv),
        "split_csv_sha256": digest,
        "seed": args.seed,
        "tune_frac": args.tune_frac,
        "stratified_on": "recording label",
        "grouped_by": "subject_id",
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "sklearn_version": __import__("sklearn").__version__,
        "git_commit": git_commit(),
        "host": platform.node(),
        "counts": summarize(splits, args.n_folds),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))

    # ---- Report ----
    c = summary["counts"]
    print(f"\nWrote {out_csv}")
    print(f"  sha256 {digest}")
    print(f"Wrote {args.summary} (aggregate only; safe to commit)\n")
    print(f"{'pool/fold':>10} {'recordings':>11} {'patients':>9} {'segments':>10} {'abnormal':>9}")
    for name, key in (("T", "tuning_pool_T"), ("E", "cv_pool_E")):
        b = c[key]
        print(
            f"{name:>10} {b['recordings']:>11,} {b['patients']:>9,} "
            f"{b['segments']:>10,} {b['abnormal_frac']:>8.1%}"
        )
    for k in range(args.n_folds):
        b = c["folds"][str(k)]
        print(
            f"{'fold ' + str(k):>10} {b['recordings']:>11,} {b['patients']:>9,} "
            f"{b['segments']:>10,} {b['abnormal_frac']:>8.1%}"
        )
    print("\nAll disjointness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())