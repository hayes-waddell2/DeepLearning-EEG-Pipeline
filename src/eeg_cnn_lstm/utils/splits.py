"""
@file splits.py
@brief Load the fixed patient-level split file and build the DataLoaders each
       experiment needs from it.

@details
`scripts/make_splits.py` writes one CSV that every later experiment reads:

@verbatim
filename,subject_id,label,n_epochs,pool,fold
  pool = "T"  tuning pool   (~20% of patients) -> chooses every setting
  pool = "E"  train/test pool (~80%)           -> reported performance
  fold = 0..K-1 for E rows, -1 for T rows
@endverbatim

Two usages:

  - **Tuning run** (`fold=None`): train on all of E, score on T. Used for the
    normalization A/B and for every HPO trial.
  - **CV fold** (`fold=k`): train on E minus fold k, use T for early stopping
    and calibration, test on fold k. Used for the reported result.

Both are built here so the two paths cannot drift apart.

@par Why the split file is loaded rather than recomputed
Re-deriving the split in each run would silently break the guarantee the whole
design rests on: that no setting was ever chosen using a patient whose result
is reported. `load_splits` revalidates the disjointness invariants on every
load, so a corrupted or hand-edited file fails loudly instead of quietly
leaking.

@par Privacy
The split CSV contains filenames and subject IDs, so it lives outside the
repository (default `/shared/rc/eeg-cnn-lstm/data/splits/`). Only its SHA-256
is recorded in run outputs.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from torch.utils.data import DataLoader

from eeg_cnn_lstm.utils.dataset import TUABEpochDataset, make_dataloader

## @brief Default location of the split file written by scripts/make_splits.py.
DEFAULT_SPLITS_CSV = Path("/shared/rc/eeg-cnn-lstm/data/splits/tuab_train_splits.csv")

## @brief `fold` value marking a tuning-pool row.
TUNE_FOLD: int = -1

## @brief Required columns in the split file.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "filename",
    "subject_id",
    "label",
    "n_epochs",
    "pool",
    "fold",
)


def splits_sha256(path: str | os.PathLike) -> str:
    """
    @brief SHA-256 of the split file, for recording which split a run used.

    @param path Path to the split CSV.
    @return Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_splits(path: str | os.PathLike = DEFAULT_SPLITS_CSV) -> pd.DataFrame:
    """
    @brief Load the split file and revalidate every disjointness invariant.

    @param path Path to the split CSV.
    @return DataFrame with the required columns, dtypes normalized.
    @throws FileNotFoundError If the file does not exist.
    @throws KeyError If required columns are missing.
    @throws ValueError If any disjointness or labelling invariant fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run `python -m scripts.make_splits` first."
        )
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Split file missing columns: {sorted(missing)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["n_epochs"] = df["n_epochs"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["pool"] = df["pool"].astype(str)

    if not df["filename"].is_unique:
        raise ValueError("Split file has duplicate filenames")
    bad_pools = set(df["pool"]) - {"T", "E"}
    if bad_pools:
        raise ValueError(f"Unexpected pool values: {sorted(bad_pools)}")

    # One pool per patient.
    if (df.groupby("subject_id")["pool"].nunique() > 1).any():
        raise ValueError("Some patients appear in both pools T and E")

    t_rows = df[df["pool"] == "T"]
    e_rows = df[df["pool"] == "E"]
    if not (t_rows["fold"] == TUNE_FOLD).all():
        raise ValueError(f"Tuning-pool rows must have fold == {TUNE_FOLD}")
    if (e_rows["fold"] < 0).any():
        raise ValueError("Evaluation-pool rows must have a non-negative fold")
    # One fold per patient.
    if (e_rows.groupby("subject_id")["fold"].nunique() > 1).any():
        raise ValueError("Some patients span multiple folds")
    if set(t_rows["subject_id"]) & set(e_rows["subject_id"]):
        raise ValueError("Patient overlap between pools T and E")

    return df


def n_folds(splits: pd.DataFrame) -> int:
    """
    @brief Number of cross-validation folds defined in a split table.

    @param splits Output of `load_splits`.
    @return Fold count (max fold id + 1).
    """
    e_rows = splits[splits["pool"] == "E"]
    return int(e_rows["fold"].max()) + 1 if len(e_rows) else 0


def pool_frame(splits: pd.DataFrame, pool: str) -> pd.DataFrame:
    """
    @brief Rows belonging to one pool.

    @param splits Output of `load_splits`.
    @param pool "T" or "E".
    @return Contiguous DataFrame of that pool's recordings.
    """
    return splits[splits["pool"] == pool].reset_index(drop=True)


def fold_frames(
    splits: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    @brief Training and test frames for one cross-validation fold.

    @param splits Output of `load_splits`.
    @param fold Fold index in `[0, n_folds)`.
    @return Tuple `(train_df, test_df)`: E minus fold k, and fold k.
    @throws ValueError If `fold` is out of range.
    """
    k = n_folds(splits)
    if not 0 <= fold < k:
        raise ValueError(f"fold must be in [0, {k}); got {fold}")
    e_rows = splits[splits["pool"] == "E"]
    train_df = e_rows[e_rows["fold"] != fold].reset_index(drop=True)
    test_df = e_rows[e_rows["fold"] == fold].reset_index(drop=True)
    return train_df, test_df


@dataclass
class SplitLoaders:
    """
    @brief The DataLoaders and metadata for one tuning run or CV fold.

    @var train Training loader (segments capped and optionally resampled).
    @var tune Tuning-pool loader (pool T, every segment) — early stopping,
         calibration, threshold, and the HPO objective are all scored here.
    @var test Held-out fold loader (every segment), or `None` for a tuning run
         where nothing is held out.
    @var info Counts and settings, for logging and run provenance.
    """

    train: DataLoader
    tune: DataLoader
    test: Optional[DataLoader] = None
    info: dict[str, Any] = field(default_factory=dict)


def _dataset(
    frame: pd.DataFrame,
    data_dir: str | os.PathLike,
    *,
    segments_per_recording: Optional[int],
    resample_each_epoch: bool,
    normalize: str,
    norm_stats: Any,
    clip_uv: Optional[float],
    seed: int,
) -> TUABEpochDataset:
    """
    @brief Build one `TUABEpochDataset` with the shared normalization settings.

    @param frame Recordings to include.
    @param data_dir Directory holding the `.npy` files.
    @param segments_per_recording Segment cap (`None` = every segment).
    @param resample_each_epoch Redraw the capped subset each epoch.
    @param normalize Normalization mode.
    @param norm_stats Stats path/dict, required for `"uv_scale"`.
    @param clip_uv Clip override for `"uv_scale"`.
    @param seed RNG seed.
    @return Configured dataset.
    """
    return TUABEpochDataset(
        frame,
        data_dir,
        segments_per_recording=segments_per_recording,
        resample_each_epoch=resample_each_epoch,
        normalize=normalize,
        norm_stats=norm_stats,
        clip_uv=clip_uv,
        seed=seed,
    )


def make_split_loaders(
    data_dir: str | os.PathLike,
    splits: str | os.PathLike | pd.DataFrame = DEFAULT_SPLITS_CSV,
    fold: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
    segments_per_recording: Optional[int] = None,
    resample_each_epoch: bool = False,
    normalize: str = "zscore",
    norm_stats: Any = None,
    clip_uv: Optional[float] = None,
) -> SplitLoaders:
    """
    @brief Build the loaders for a tuning run (`fold=None`) or a CV fold.

    @details
    Training uses the segment cap; the tuning and test loaders always use every
    segment, because a recording-level score must cover the whole recording.

    `fold=None` trains on all of E and scores on T. This is the configuration
    for the normalization A/B, for every HPO trial, and for the final model.
    `fold=k` trains on E minus fold k, still scores on T for early stopping and
    calibration, and reports on fold k.

    @param data_dir Directory holding the `.npy` files.
    @param splits Split CSV path, or an already-loaded DataFrame.
    @param fold Fold index, or `None` for a tuning run.
    @param batch_size Mini-batch size for all loaders.
    @param num_workers Worker processes per loader.
    @param seed RNG seed for segment subsampling.
    @param segments_per_recording Training-side segment cap (`None` = all).
    @param resample_each_epoch Redraw the training subset each epoch.
    @param normalize Normalization mode, shared by all three loaders.
    @param norm_stats Stats path/dict, required for `"uv_scale"`.
    @param clip_uv Clip override for `"uv_scale"`.
    @return `SplitLoaders` with train, tune, optional test, and `info`.
    """
    splits_df = splits if isinstance(splits, pd.DataFrame) else load_splits(splits)
    tune_df = pool_frame(splits_df, "T")

    if fold is None:
        train_df = pool_frame(splits_df, "E")
        test_df: Optional[pd.DataFrame] = None
    else:
        train_df, test_df = fold_frames(splits_df, fold)

    norm_kw = dict(normalize=normalize, norm_stats=norm_stats, clip_uv=clip_uv)
    train_set = _dataset(
        train_df,
        data_dir,
        segments_per_recording=segments_per_recording,
        resample_each_epoch=resample_each_epoch,
        seed=seed,
        **norm_kw,
    )
    tune_set = _dataset(
        tune_df,
        data_dir,
        segments_per_recording=None,
        resample_each_epoch=False,
        seed=seed + 1,
        **norm_kw,
    )
    test_set = (
        _dataset(
            test_df,
            data_dir,
            segments_per_recording=None,
            resample_each_epoch=False,
            seed=seed + 2,
            **norm_kw,
        )
        if test_df is not None
        else None
    )

    info: dict[str, Any] = {
        "fold": fold,
        "normalize": normalize,
        "splits_sha256": (
            None if isinstance(splits, pd.DataFrame) else splits_sha256(splits)
        ),
        "train": train_set.describe(),
        "tune": tune_set.describe(),
        "test": test_set.describe() if test_set is not None else None,
    }

    return SplitLoaders(
        train=make_dataloader(train_set, batch_size, True, num_workers),
        tune=make_dataloader(tune_set, batch_size, False, num_workers),
        test=(
            make_dataloader(test_set, batch_size, False, num_workers)
            if test_set is not None
            else None
        ),
        info=info,
    )


# ----------------------
# Smoke test entry point
# ----------------------


def _main() -> None:
    """
    @brief Print a summary of the split file: pools, folds, and patient counts.

    @details
    Loads and revalidates the split file, then prints the same table
    `make_splits.py` produced, as a quick check that the file on disk is intact
    and readable by the training code. Run as
    `python -m eeg_cnn_lstm.utils.splits [path]`.
    """
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SPLITS_CSV
    df = load_splits(path)
    k = n_folds(df)
    print(f"{path}\n  sha256 {splits_sha256(path)}\n")
    print(f"{'pool/fold':>10} {'recordings':>11} {'patients':>9} {'segments':>10} {'abnormal':>9}")

    def row(name: str, sub: pd.DataFrame) -> None:
        print(
            f"{name:>10} {len(sub):>11,} {sub['subject_id'].nunique():>9,} "
            f"{sub['n_epochs'].sum():>10,} {sub['label'].mean():>8.1%}"
        )

    row("T", pool_frame(df, "T"))
    row("E", pool_frame(df, "E"))
    e_rows = pool_frame(df, "E")
    for f in range(k):
        row(f"fold {f}", e_rows[e_rows["fold"] == f])
    print("\nAll disjointness checks passed.")


if __name__ == "__main__":
    _main()