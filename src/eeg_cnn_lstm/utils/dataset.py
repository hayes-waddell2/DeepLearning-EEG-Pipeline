"""
@file dataset.py
@brief PyTorch Dataset and DataLoader utilities for the TUH EEG Abnormal Corpus (TUAB).

@details
This module wraps the preprocessed TUAB `.npy` files (float16, microvolts) into
a PyTorch-compatible dataset for training a CNN+LSTM binary classifier
(0 = normal, 1 = abnormal).

Key features:
  - Lazy memory-mapped loading: never loads the full ~71 GB train set into RAM.
  - Compact index: parallel NumPy arrays rather than one object per segment,
    so a 600k-segment index costs a few MB.
  - Recording identity is returned with every sample (`rec_idx`), which is what
    lets segment predictions be aggregated into one score per recording.
  - Selectable normalization (Phase 1): per-segment z-score (the baseline) or a
    fixed microvolt affine transform that preserves absolute and inter-channel
    amplitude.
  - Optional per-recording segment subsampling, with `set_epoch()` to redraw a
    fresh subset each epoch (segments overlap 50%, so neighbours are
    near-duplicates; resampling adds variety per unit of compute).
  - float16 (uV) -> float32 cast at load time for numerically stable scaling.

@par Expected on-disk layout:
@verbatim
data_dir/
    <recording_id>_epochs.npy   # shape (n_epochs, 19, 2500), dtype float16, units uV
manifest.csv                    # columns: filename,label,n_epochs,sfreq
@endverbatim

@par Subject ID convention:
TUAB filenames have the form `<subject>_s<session>_t<token>_epochs.npy` (e.g.
`<subject>_s001_t000_epochs.npy`). The substring before the first underscore is
treated as the anonymized subject ID and is used to keep all recordings from a
given subject in exactly one split.

@par Normalization modes
  - `"zscore"`: per segment, per channel, subtract mean and divide by std.
    The Phase 0 baseline. Removes absolute amplitude and inter-channel
    amplitude asymmetry, both of which are clinically meaningful.
  - `"uv_scale"`: clip to +/-`clip_uv`, then apply a *fixed* per-channel
    (x - mean) / std using constants estimated once from the tuning pool by
    `scripts/compute_norm_stats.py`. Amplitude information survives.
  - `"none"`: raw microvolts (debugging only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

## @brief Number of EEG channels in the standard 10-20 montage used by preprocessing.
N_CHANNELS: int = 19

## @brief Number of time samples per epoch (10 s window @ 250 Hz).
N_TIMESTEPS: int = 2500

## @brief Sampling frequency of the preprocessed data, in Hz.
SAMPLE_RATE: int = 250

## @brief Supported values for the `normalize` argument.
NORMALIZE_MODES: tuple[str, ...] = ("zscore", "uv_scale", "none")

## @brief Channel std below this (in uV) is treated as flat/disconnected and not divided by.
FLAT_STD_UV: float = 1e-2

## @brief Default clip threshold (uV) for `"uv_scale"`; above this is artifact, not EEG.
DEFAULT_CLIP_UV: float = 800.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_subject_id(filename: str) -> str:
    """
    @brief Extract the subject identifier from a TUAB recording filename.

    @details
    TUAB filenames take the form `<subject>_s<session>_t<token>_epochs.npy`.
    The portion before the first underscore is the anonymized subject ID and is
    shared across all recordings (sessions / tokens) for that subject.

    @param filename Bare filename (no directory) of a recording's `.npy` file.
    @return The subject ID string.
    @throws ValueError If the filename has no underscore and therefore no
            parseable subject prefix.
    """
    base = os.path.basename(filename)
    if "_" not in base:
        raise ValueError(f"Cannot parse subject id from: {filename!r}")
    return base.split("_", 1)[0]


def load_manifest(manifest_path: str | os.PathLike) -> pd.DataFrame:
    """
    @brief Load a TUAB preprocessing manifest CSV and attach a `subject_id` column.

    @param manifest_path Path to the manifest CSV (e.g. `train_manifest.csv`).
    @return DataFrame with columns: `filename, label, n_epochs, sfreq, subject_id`.
    @throws FileNotFoundError If the manifest does not exist.
    @throws KeyError If required columns are missing from the manifest.
    """
    df = pd.read_csv(manifest_path)
    required = {"filename", "label", "n_epochs", "sfreq"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["subject_id"] = df["filename"].map(parse_subject_id)
    df["label"] = df["label"].astype(np.int64)
    df["n_epochs"] = df["n_epochs"].astype(np.int64)
    return df


def load_norm_stats(source: str | os.PathLike | dict[str, Any]) -> dict[str, Any]:
    """
    @brief Load and validate per-channel normalization statistics.

    @details
    Accepts either an already-parsed dict or a path to the JSON written by
    `scripts/compute_norm_stats.py`. Validates that the vectors have one entry
    per channel and that no standard deviation is degenerate.

    @param source Path to the stats JSON, or the parsed dict.
    @return Dict with at least `mean_uv`, `std_uv`, `clip_uv`, `channels`.
    @throws KeyError If required keys are missing.
    @throws ValueError If vector lengths or values are invalid.
    """
    stats = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    for key in ("mean_uv", "std_uv"):
        if key not in stats:
            raise KeyError(f"norm stats missing {key!r}")
        if len(stats[key]) != N_CHANNELS:
            raise ValueError(
                f"norm stats {key!r} has {len(stats[key])} entries; expected {N_CHANNELS}"
            )
    if min(stats["std_uv"]) <= 0:
        raise ValueError("norm stats std_uv contains a non-positive value")
    stats.setdefault("clip_uv", DEFAULT_CLIP_UV)
    return stats


def build_subject_disjoint_split(
    manifest: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    @brief Split a manifest into train / val DataFrames with no subject overlap.

    @deprecated Superseded in Phase 1 by the fixed split file produced by
    `scripts/make_splits.py` (pools T/E plus CV folds, stratified on recording
    labels). Retained because `scripts/verify_f16_checkpoint.py` needs it to
    reproduce the Phase 0 baseline validation fold exactly.

    @param manifest Manifest DataFrame (must contain `subject_id` and `label`).
    @param val_frac Fraction of subjects to assign to the validation split.
    @param seed RNG seed for reproducibility.
    @return Tuple `(train_df, val_df)`, each a contiguous DataFrame.
    @throws ValueError If `val_frac` is not strictly between 0 and 1.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")

    rng = np.random.default_rng(seed)

    # Assign each subject a single representative label (mode across their
    # recordings) purely to stratify the val sampling. Disjointness is enforced
    # by the final isin() check, not by this label.
    subject_label = manifest.groupby("subject_id")["label"].agg(
        lambda s: int(s.mode().iat[0])
    )

    val_subjects: set[str] = set()
    for label_value in sorted(subject_label.unique()):
        subjects = subject_label[subject_label == label_value].index.to_numpy()
        rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_frac)))
        val_subjects.update(subjects[:n_val].tolist())

    is_val = manifest["subject_id"].isin(val_subjects)
    train_df = manifest.loc[~is_val].reset_index(drop=True)
    val_df = manifest.loc[is_val].reset_index(drop=True)
    return train_df, val_df


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class TUABEpochDataset(Dataset):
    """
    @brief PyTorch Dataset over individual segments from preprocessed TUAB recordings.

    @details
    Each sample is one 10-second EEG segment of shape (19, 2500), its binary
    label, and the index of the recording it came from. Files are opened lazily
    via `np.load(..., mmap_mode='r')` and cached per worker so consecutive reads
    from the same recording skip the open() syscall.

    @par Returned tuple:
      - `x`: torch.float32, shape (19, 2500), normalized per `normalize`.
      - `y`: torch.float32 scalar (0.0 or 1.0), float so it pairs cleanly with
             `BCEWithLogitsLoss`.
      - `rec_idx`: torch.int64 scalar, row position in `self.recordings`. Group
             predictions by this to obtain one score per recording.

    @par Segment subsampling
    `segments_per_recording=None` (or 0) uses every segment, which is required
    for validation and test so that recording-level scores are computed over the
    whole recording. For training, a cap plus `set_epoch()` draws a different
    subset each epoch.

    @note Worker safety: NumPy mmap handles do not cross fork boundaries
          cleanly. The internal handle cache is keyed by worker PID and rebuilt
          the first time a worker calls `__getitem__`.
    @note `set_epoch()` must be called before the epoch's iterator is created;
          with `persistent_workers=True` the change reaches workers on the next
          epoch's first iteration.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_dir: str | os.PathLike,
        segments_per_recording: Optional[int] = None,
        normalize: str = "zscore",
        norm_stats: Optional[str | os.PathLike | dict[str, Any]] = None,
        clip_uv: Optional[float] = None,
        resample_each_epoch: bool = False,
        seed: int = 42,
    ) -> None:
        """
        @brief Build the segment index for this dataset.

        @param manifest DataFrame as returned by `load_manifest` (or a split
               frame with the same columns), restricted to the desired subset.
        @param data_dir Directory containing the `<recording>_epochs.npy` files
               referenced by the manifest's `filename` column.
        @param segments_per_recording If set and > 0, use at most this many
               randomly chosen segments per recording. `None` or `0` uses all
               segments (required for validation / test).
        @param normalize One of `NORMALIZE_MODES`.
        @param norm_stats Path to (or parsed contents of) the normalization
               stats JSON. Required when `normalize == "uv_scale"`.
        @param clip_uv Clip threshold in microvolts for `"uv_scale"`. Defaults
               to the value in `norm_stats`, else `DEFAULT_CLIP_UV`.
        @param resample_each_epoch If True, `set_epoch(n)` redraws the segment
               subset. Only meaningful with `segments_per_recording` set.
        @param seed RNG seed for segment subsampling.
        @throws ValueError If `normalize` is unknown, or `"uv_scale"` is
                requested without stats.
        """
        super().__init__()
        if normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}; got {normalize!r}")

        self._data_dir = Path(data_dir)
        self._normalize = normalize
        self._cap = int(segments_per_recording or 0)
        self._resample = bool(resample_each_epoch)
        self._seed = int(seed)
        self._epoch = 0

        # Per-channel affine constants, shaped (19, 1) for broadcasting.
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._clip_uv: float = DEFAULT_CLIP_UV if clip_uv is None else float(clip_uv)
        if normalize == "uv_scale":
            if norm_stats is None:
                raise ValueError('normalize="uv_scale" requires norm_stats')
            stats = load_norm_stats(norm_stats)
            self._mean = np.asarray(stats["mean_uv"], dtype=np.float32).reshape(-1, 1)
            self._std = np.asarray(stats["std_uv"], dtype=np.float32).reshape(-1, 1)
            if clip_uv is None:
                self._clip_uv = float(stats["clip_uv"])

        # Recording table: row position is the rec_idx returned by __getitem__.
        cols = [c for c in ("filename", "subject_id", "label", "n_epochs") if c in manifest]
        self.recordings: pd.DataFrame = manifest[cols].reset_index(drop=True).copy()
        if "subject_id" not in self.recordings:
            self.recordings["subject_id"] = self.recordings["filename"].map(parse_subject_id)

        self._paths: list[str] = [
            str(self._data_dir / f) for f in self.recordings["filename"]
        ]
        self._n_epochs = self.recordings["n_epochs"].to_numpy(dtype=np.int64)
        self._labels = self.recordings["label"].to_numpy(dtype=np.int8)

        # Per-worker mmap handle cache, populated lazily so forked DataLoader
        # workers each get their own handles.
        self._mmap_cache: dict[str, np.ndarray] = {}
        self._cache_owner_pid: int = -1

        self._build_index()

    # -------------------------------------------------------------------------
    # Index construction
    # -------------------------------------------------------------------------

    def _build_index(self) -> None:
        """
        @brief (Re)build the flat segment index as parallel NumPy arrays.

        @details
        Populates `_rec_of_sample` (recording row position) and
        `_seg_of_sample` (segment index within that recording). With no cap the
        index is every segment of every recording; with a cap, a random subset
        per recording, drawn from `seed` and the current epoch so that each
        epoch sees a different subset but any run is reproducible.
        """
        if not self._cap:
            rec = np.repeat(
                np.arange(len(self._n_epochs), dtype=np.int32), self._n_epochs
            )
            seg = np.concatenate(
                [np.arange(n, dtype=np.int32) for n in self._n_epochs]
            ) if len(self._n_epochs) else np.empty(0, dtype=np.int32)
        else:
            rng = np.random.default_rng((self._seed, self._epoch))
            recs, segs = [], []
            for i, n in enumerate(self._n_epochs):
                k = min(self._cap, int(n))
                chosen = (
                    np.arange(n, dtype=np.int32)
                    if k == n
                    else rng.choice(int(n), size=k, replace=False).astype(np.int32)
                )
                recs.append(np.full(k, i, dtype=np.int32))
                segs.append(chosen)
            rec = np.concatenate(recs) if recs else np.empty(0, dtype=np.int32)
            seg = np.concatenate(segs) if segs else np.empty(0, dtype=np.int32)

        self._rec_of_sample = rec
        self._seg_of_sample = seg

    def set_epoch(self, epoch: int) -> None:
        """
        @brief Redraw the per-recording segment subset for a new epoch.

        @details
        No-op unless the dataset was built with `resample_each_epoch=True` and a
        segment cap. Call before creating the epoch's DataLoader iterator.

        @param epoch Zero-based epoch number; seeds the redraw.
        """
        if not (self._resample and self._cap):
            return
        if epoch == self._epoch:
            return
        self._epoch = int(epoch)
        self._build_index()

    # -------------------------------------------------------------------------
    # PyTorch Dataset API
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        @brief Number of segments currently indexed.
        @return Length of the flat segment index.
        """
        return int(self._rec_of_sample.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @brief Load and return one EEG segment, its label, and its recording index.

        @param idx Index into the flat segment list, in `[0, len(self))`.
        @return Tuple `(x, y, rec_idx)`:
                - `x`: `torch.float32`, shape (19, 2500), normalized.
                - `y`: `torch.float32` scalar (0.0 or 1.0).
                - `rec_idx`: `torch.int64` scalar, row in `self.recordings`.
        @throws IndexError If `idx` is out of range.
        """
        rec_idx = int(self._rec_of_sample[idx])
        seg_idx = int(self._seg_of_sample[idx])
        arr = self._get_mmap(self._paths[rec_idx])
        # Materialize one segment as float32. The mmap is read-only float16 (uV);
        # np.asarray with a dtype forces a copy + cast.
        seg = np.asarray(arr[seg_idx], dtype=np.float32)

        if self._normalize == "zscore":
            mean = seg.mean(axis=1, keepdims=True)
            std = seg.std(axis=1, keepdims=True)
            std = np.where(std < FLAT_STD_UV, 1.0, std)  # < 0.01 uV = flat channel
            seg = (seg - mean) / std
        elif self._normalize == "uv_scale":
            np.clip(seg, -self._clip_uv, self._clip_uv, out=seg)
            seg = (seg - self._mean) / self._std

        x = torch.from_numpy(seg)
        y = torch.tensor(float(self._labels[rec_idx]), dtype=torch.float32)
        return x, y, torch.tensor(rec_idx, dtype=torch.int64)

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def segments_per_recording_used(self) -> np.ndarray:
        """
        @brief Number of indexed segments per recording, aligned to `recordings`.
        @return int64 array of length `len(self.recordings)`.
        """
        return np.bincount(self._rec_of_sample, minlength=len(self.recordings)).astype(
            np.int64
        )

    def describe(self) -> dict[str, Any]:
        """
        @brief Summary of this dataset's configuration and size.
        @return Dict of counts and settings, suitable for logging.
        """
        return {
            "recordings": int(len(self.recordings)),
            "patients": int(self.recordings["subject_id"].nunique()),
            "segments_indexed": len(self),
            "segments_available": int(self._n_epochs.sum()),
            "segments_per_recording_cap": self._cap or None,
            "resample_each_epoch": self._resample,
            "normalize": self._normalize,
            "clip_uv": self._clip_uv if self._normalize == "uv_scale" else None,
            "abnormal_recording_frac": round(float(self.recordings["label"].mean()), 4),
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _get_mmap(self, path: str) -> np.ndarray:
        """
        @brief Return a memory-mapped view of a recording, caching by file path.

        @details
        Opens the file with `np.load(path, mmap_mode='r')` on first access and
        retains the handle so subsequent segment reads from the same recording
        skip the open() syscall. The cache is reset whenever the owning process
        ID changes, which handles the DataLoader's `fork`-based worker startup
        cleanly.

        @param path Absolute path to a recording's `.npy` file.
        @return Memory-mapped ndarray of shape (n_epochs, 19, 2500), dtype
                float16 (uV).
        """
        pid = os.getpid()
        if pid != self._cache_owner_pid:
            self._mmap_cache = {}
            self._cache_owner_pid = pid
        arr = self._mmap_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            self._mmap_cache[path] = arr
        return arr


# -----------------------------------------------------------------------------
# DataLoader factories
# -----------------------------------------------------------------------------


def make_dataloader(
    dataset: TUABEpochDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """
    @brief Wrap a dataset in a DataLoader with the project's standard settings.

    @param dataset The dataset to iterate.
    @param batch_size Mini-batch size.
    @param shuffle Whether to shuffle (True for training, False for eval).
    @param num_workers Worker processes.
    @return Configured DataLoader yielding `(x, y, rec_idx)` batches.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def make_train_val_dataloaders(
    manifest_path: str | os.PathLike,
    data_dir: str | os.PathLike,
    batch_size: int = 64,
    val_frac: float = 0.2,
    max_epochs_per_recording: Optional[int] = None,
    num_workers: int = 8,
    seed: int = 42,
    normalize: str = "zscore",
    norm_stats: Optional[str | os.PathLike | dict[str, Any]] = None,
    clip_uv: Optional[float] = None,
    resample_each_epoch: bool = False,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    @brief Build subject-disjoint train and validation DataLoaders from a manifest.

    @deprecated Phase 1 experiments use the fixed split file via
    `eeg_cnn_lstm.utils.splits`. This wrapper is kept so the Phase 0 baseline
    configuration remains runnable and comparable.

    @details
    Note the asymmetry: `max_epochs_per_recording` caps *training* segments
    only. Validation always uses every segment, because a recording-level score
    must be computed over the whole recording.

    @param manifest_path Path to `train_manifest.csv`.
    @param data_dir Directory containing the `<recording>_epochs.npy` files.
    @param batch_size Mini-batch size for both loaders.
    @param val_frac Fraction of *subjects* held out for validation.
    @param max_epochs_per_recording Training-side segment cap; `None` uses all.
    @param num_workers Worker processes for each DataLoader.
    @param seed RNG seed for the split and for segment subsampling.
    @param normalize One of `NORMALIZE_MODES`.
    @param norm_stats Stats JSON path/dict, required for `"uv_scale"`.
    @param clip_uv Clip threshold override for `"uv_scale"`.
    @param resample_each_epoch Redraw the training subset each epoch.
    @return Tuple `(train_loader, val_loader, info)` where `info` holds
            split-size diagnostics.
    """
    manifest = load_manifest(manifest_path)
    train_df, val_df = build_subject_disjoint_split(
        manifest, val_frac=val_frac, seed=seed
    )

    common = dict(
        normalize=normalize,
        norm_stats=norm_stats,
        clip_uv=clip_uv,
    )
    train_set = TUABEpochDataset(
        train_df,
        data_dir,
        segments_per_recording=max_epochs_per_recording,
        resample_each_epoch=resample_each_epoch,
        seed=seed,
        **common,
    )
    val_set = TUABEpochDataset(
        val_df,
        data_dir,
        segments_per_recording=None,  # always evaluate on every segment
        seed=seed + 1,
        **common,
    )

    train_loader = make_dataloader(train_set, batch_size, True, num_workers)
    val_loader = make_dataloader(val_set, batch_size, False, num_workers)

    info = {
        "train_recordings": int(train_df.shape[0]),
        "val_recordings": int(val_df.shape[0]),
        "train_subjects": int(train_df["subject_id"].nunique()),
        "val_subjects": int(val_df["subject_id"].nunique()),
        "train_epochs_used": len(train_set),
        "val_epochs_used": len(val_set),
        "train_class_balance": train_df["label"].value_counts().to_dict(),
        "val_class_balance": val_df["label"].value_counts().to_dict(),
        "normalize": normalize,
    }
    return train_loader, val_loader, info


# -----------------------------------------------------------------------------
# Smoke test entry point
# -----------------------------------------------------------------------------


def _main() -> None:
    """
    @brief Minimal CLI smoke test: load one batch and print its shapes.

    @details
    Run as `python -m eeg_cnn_lstm.utils.dataset` from the project root.
    Override the default paths via environment variables `TUAB_MANIFEST` and
    `TUAB_DATA_DIR`. Note the doubled `train/train/` quirk in the cluster's
    on-disk layout: `data_dir` points to the inner folder holding the `.npy`
    files, not the outer one holding the manifest.
    """
    import sys

    manifest = os.environ.get(
        "TUAB_MANIFEST",
        "/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train_manifest.csv",
    )
    data_dir = os.environ.get(
        "TUAB_DATA_DIR",
        "/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train",
    )

    train_loader, val_loader, info = make_train_val_dataloaders(
        manifest_path=manifest,
        data_dir=data_dir,
        batch_size=8,
        max_epochs_per_recording=30,
        num_workers=2,
        resample_each_epoch=True,
    )
    print("Split info:", info, file=sys.stderr)
    print("Train dataset:", train_loader.dataset.describe(), file=sys.stderr)
    x, y, rec = next(iter(train_loader))
    print(
        f"x.shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"y.shape={tuple(y.shape)}, rec.shape={tuple(rec.shape)}, "
        f"label_mean={y.mean().item():.3f}, rec_idx[:4]={rec[:4].tolist()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    _main()