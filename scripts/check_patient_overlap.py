"""
@file check_patient_overlap.py
@brief Phase 0 / Step 4: verify TUAB train and eval partitions are patient-disjoint
       and summarize patient-level structure.

@details
Reads ONLY the two manifest CSVs (filenames, labels, epoch counts). No signal
data and no model predictions are touched, so the eval partition remains
untouched in every sense that matters for the final evaluation.

Reports, per partition:
  - recordings, patients, total segments, sampling rates present
  - recording- and patient-level class balance
  - recordings-per-patient distribution
  - patients with mixed labels across sessions
  - malformed subject IDs (anything not 8 lowercase letters)
Then reports the train/eval patient overlap (expected: 0).

Writes a JSON summary (and CSVs of any overlapping / mixed-label patients) to
`--out-dir` so the numbers are reproducible and citable in the writeup.

@par Usage (from the repo root, venv active):
@verbatim
python -m scripts.check_patient_overlap
python -m scripts.check_patient_overlap --eval-manifest <path> --out-dir results/phase0
@endverbatim

@return Exit code 0 if partitions are patient-disjoint, 1 if any overlap is found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils.dataset import load_manifest

## @brief Expected TUAB anonymized subject ID format.
SUBJECT_RE = re.compile(r"^[a-z]{8}$")

DEFAULT_ROOT = Path("/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab")


def summarize(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    @brief Compute patient-level summary statistics for one partition.

    @param df Manifest DataFrame from `load_manifest` (has `subject_id`).
    @return Tuple `(summary_dict, mixed_label_patients_df)`.
    """
    per_patient = df.groupby("subject_id").agg(
        n_recordings=("filename", "size"),
        n_distinct_labels=("label", "nunique"),
        n_abnormal_recordings=("label", "sum"),
        n_segments=("n_epochs", "sum"),
    )
    mixed = per_patient[per_patient["n_distinct_labels"] > 1]

    # Patient label = mode across recordings (same rule the current split uses).
    patient_label = df.groupby("subject_id")["label"].agg(
        lambda s: int(s.mode().iat[0])
    )

    rec_dist = per_patient["n_recordings"].value_counts().sort_index()
    malformed = sorted(
        s for s in per_patient.index if not SUBJECT_RE.match(str(s))
    )

    summary = {
        "n_recordings": int(len(df)),
        "n_patients": int(per_patient.shape[0]),
        "n_segments": int(df["n_epochs"].sum()),
        "duplicate_filenames": int(df["filename"].duplicated().sum()),
        "sfreq_values": sorted(float(v) for v in df["sfreq"].unique()),
        "recording_label_counts": {
            str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
        },
        "patient_label_counts_mode": {
            str(k): int(v) for k, v in patient_label.value_counts().sort_index().items()
        },
        "recordings_per_patient": {str(k): int(v) for k, v in rec_dist.items()},
        "max_recordings_per_patient": int(per_patient["n_recordings"].max()),
        "patients_with_multiple_recordings": int((per_patient["n_recordings"] > 1).sum()),
        "patients_with_mixed_labels": int(mixed.shape[0]),
        "recordings_from_mixed_label_patients": int(mixed["n_recordings"].sum()),
        "malformed_subject_ids": malformed,
    }
    return summary, mixed.reset_index()


def print_summary(name: str, s: dict) -> None:
    """
    @brief Print a human-readable summary block for one partition.

    @param name Partition name (e.g. "train").
    @param s Summary dict from `summarize`.
    """
    print(f"\n=== {name.upper()} ===")
    print(f"  recordings:              {s['n_recordings']:,}")
    print(f"  patients:                {s['n_patients']:,}")
    print(f"  segments:                {s['n_segments']:,}")
    print(f"  duplicate filenames:     {s['duplicate_filenames']}")
    print(f"  sfreq values:            {s['sfreq_values']}")
    print(f"  recording labels (0/1):  {s['recording_label_counts']}")
    print(f"  patient labels, mode:    {s['patient_label_counts_mode']}")
    print(f"  recordings per patient:  {s['recordings_per_patient']}")
    print(f"  multi-recording patients:{s['patients_with_multiple_recordings']:>6}")
    print(
        f"  mixed-label patients:    {s['patients_with_mixed_labels']} "
        f"({s['recordings_from_mixed_label_patients']} recordings)"
    )
    if s["malformed_subject_ids"]:
        print(f"  !! malformed subject IDs: {s['malformed_subject_ids'][:10]}")


def main() -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code (0 = disjoint, 1 = overlap found).
    """
    parser = argparse.ArgumentParser(description="Check TUAB train/eval patient overlap")
    parser.add_argument(
        "--train-manifest",
        default=str(DEFAULT_ROOT / "train" / "train_manifest.csv"),
    )
    parser.add_argument(
        "--eval-manifest",
        default=str(DEFAULT_ROOT / "eval" / "eval_manifest.csv"),
    )
    parser.add_argument("--out-dir", default="results/phase0")
    args = parser.parse_args()

    for p in (args.train_manifest, args.eval_manifest):
        if not Path(p).is_file():
            print(f"ERROR: manifest not found: {p}", file=sys.stderr)
            return 2

    train = load_manifest(args.train_manifest)
    evald = load_manifest(args.eval_manifest)

    train_sum, train_mixed = summarize(train)
    eval_sum, eval_mixed = summarize(evald)
    print_summary("train", train_sum)
    print_summary("eval", eval_sum)

    overlap = sorted(set(train["subject_id"]) & set(evald["subject_id"]))
    print("\n=== TRAIN / EVAL OVERLAP ===")
    if overlap:
        n_train_recs = int(train["subject_id"].isin(overlap).sum())
        n_eval_recs = int(evald["subject_id"].isin(overlap).sum())
        print(f"  !! {len(overlap)} patients appear in BOTH partitions")
        print(f"     ({n_train_recs} train recordings, {n_eval_recs} eval recordings)")
        print(f"     first few: {overlap[:10]}")
    else:
        print("  OK: no patient appears in both partitions.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_manifest": args.train_manifest,
        "eval_manifest": args.eval_manifest,
        "train": train_sum,
        "eval": eval_sum,
        "overlap_patients": overlap,
        "n_overlap_patients": len(overlap),
    }
    (out_dir / "patient_overlap_summary.json").write_text(json.dumps(summary, indent=2))
    train_mixed.to_csv(out_dir / "train_mixed_label_patients.csv", index=False)
    if overlap:
        pd.Series(overlap, name="subject_id").to_csv(
            out_dir / "overlap_patients.csv", index=False
        )
    print(f"\nWrote summary to {out_dir.resolve()}")

    return 1 if overlap else 0


if __name__ == "__main__":
    sys.exit(main())