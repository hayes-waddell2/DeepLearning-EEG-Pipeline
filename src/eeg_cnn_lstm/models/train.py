"""
@file train.py
@brief Training entrypoint for the CNN+LSTM EEG classifier.

@details
One-command training script for the TUH EEG Abnormal Corpus binary classifier.
Reads all settings from a YAML config, builds the DataLoaders, instantiates the
CNN+LSTM model, and runs a mixed-precision training loop with gradient
clipping. Each epoch logs loss plus segment-level and recording-level metrics,
and the best checkpoint is saved.

Phase 0 / Step 2 additions (no change to model, loss, optimizer or data):
  - CLI overrides for data dir, output dir, epoch count and a step cap, so job
    scripts can point at a node-local copy of the data without editing YAML.
  - Per-step timing split into data-wait (blocked on the DataLoader) and
    compute (forward/backward/step). `loss.item()` synchronizes the GPU every
    step, so the split is accurate without extra CUDA syncs.
  - Per-epoch throughput / timing written to `<output_dir>/benchmark.json`.

Phase 0 / Step 3 (autocast dtype): the precision is selected by
`train.amp_dtype` ("fp16" or "bf16"); FP16 is the default and BF16 must not be
used with this architecture. cuDNN provides no fused BF16 LSTM kernel, so BF16
autocast falls back to an unfused per-timestep path: measured throughput drops
from ~6,100 to ~615 samples/s (~9x slower) at identical validation AUC. The
GradScaler is enabled only for FP16, which needs dynamic loss scaling; BF16 has
FP32's dynamic range and does not. Evaluation always runs in FP32.

Phase 1 additions:
  - **Fixed patient splits.** With `data.splits_csv` set, training uses the
    pools from `scripts/make_splits.py`: train on E (or E minus one fold), and
    score on T. Every tuned decision is then scored on patients that are never
    reported on. Without it, the legacy subject-disjoint 80/20 split is used so
    the Phase 0 baseline stays runnable.
  - **Recording-level metrics.** Segment scores are aggregated into one score
    per recording and evaluated with the sensitivity-first metric set (partial
    AUC, sensitivity at a specificity floor, specificity at a sensitivity
    floor). A clinical decision is per recording, not per 10-second window.
  - **Saved predictions.** The best epoch's per-segment scores are written to
    `predictions_tune.npz` (and `predictions_test.npz` for a fold run), so
    aggregator sweeps, calibration and threshold selection can be done
    afterward with no retraining.
  - **Checkpoint selection** by `train.select_metric`: recording-level partial
    AUC (default when splits are used) or segment-level AUC (Phase 0
    behaviour).
  - Batches are `(x, y, rec_idx)`; `set_epoch()` redraws the capped training
    subset each epoch; normalization is selectable for the A/B.

Phase 2 / Step 2 additions (early stopping and LR scheduling):
  - **Sub-epoch validation.** Validation runs every `val_every_frac` of an
    epoch rather than only at the epoch boundary. The A/B showed the usable
    window is about four epochs wide, with the metric collapsing in the fifth;
    checking twice per epoch doubles the resolution over that window at the
    cost of one extra validation pass per epoch.
  - **Early stopping** on the same metric used for checkpoint selection,
    counted in validation checks, with `min_delta`, a patience counter and
    restore-best-weights at the end. `train.num_epochs` becomes an upper bound.
  - **ReduceLROnPlateau** stepped on the same checks, sharing the stopper's
    `min_delta` so both agree on what counts as progress.
  - The run record gains a `checks` list (one entry per validation check, with
    the learning rate and the running training loss since the previous check),
    while `epochs` still carries one entry per epoch boundary so existing
    plotting and comparison scripts keep working unchanged.

@par Usage:
@verbatim
python -m eeg_cnn_lstm.models.train --config configs/baseline.yaml
python -m eeg_cnn_lstm.models.train --config configs/baseline.yaml --fold 3
python -m eeg_cnn_lstm.models.train --config configs/baseline.yaml \
    --train-data-dir /tmp/$USER/$SLURM_JOB_ID/train --output-dir runs/bench --num-epochs 1
@endverbatim

@par Config schema:
@verbatim
data:
  train_manifest: <path>      # legacy split path; also used for the final eval
  train_data_dir: <path>
  splits_csv: <path>          # Phase 1: enables the T/E pools
  eval_manifest:  <path>      # required only if eval.run_final_eval = true
  eval_data_dir:  <path>      # required only if eval.run_final_eval = true
loader:
  batch_size: int
  val_frac: float             # legacy path only
  max_epochs_per_recording: int | null   # TRAINING segment cap; eval uses all
  num_workers: int
  seed: int
  normalize: "zscore" | "uv_scale" | "none"   # optional; default zscore
  norm_stats: <path>          # required when normalize = uv_scale
  clip_uv: float              # optional override for uv_scale
  resample_each_epoch: bool   # optional; redraw the capped subset each epoch
train:
  num_epochs: int             # UPPER BOUND when early stopping is enabled
  lr: float
  weight_decay: float
  grad_clip: float            # 0.0 disables
  use_amp: bool               # auto-disabled on CPU
  amp_dtype: "fp16" | "bf16"  # optional; default fp16 (bf16 is ~9x slower here)
  device: "auto" | "cpu" | "cuda" | "cuda:N"
  output_dir: str             # checkpoints + log written here
  log_every: int              # optional; steps between throughput logs (default 200)
  fold: int | null            # optional; CV fold index (needs splits_csv)
  select_metric: str          # optional; "recording_pauc" | "segment_auc"
  aggregator: str             # optional; segment->recording method, default mean
  early_stopping:             # optional; omit or enabled:false for Phase 1 behaviour
    enabled: bool
    val_every_frac: float     # validate every this fraction of an epoch (0 = epoch end only)
    patience: int             # non-improving CHECKS tolerated (scales with the cadence)
    min_delta: float          # absolute improvement that counts as progress
    min_checks: int           # optional warm-up before stopping is allowed
    restore_best: bool        # reload best.pt when the loop ends
  lr_schedule:                # optional; omit or enabled:false to keep a fixed LR
    enabled: bool
    kind: "plateau" | "none"
    factor: float             # LR multiplier on plateau
    patience: int             # non-improving CHECKS before reducing; < stopper patience
    min_lr: float
    cooldown: int
eval:
  run_final_eval: bool        # if true, evaluates best checkpoint on eval split
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from eeg_cnn_lstm.utils.dataset import (
    TUABEpochDataset,
    load_manifest,
    make_train_val_dataloaders,
)
from eeg_cnn_lstm.utils.metrics import (
    DEFAULT_MAX_FPR,
    aggregate_to_recordings,
    compute_metrics,
    format_metrics,
    format_recording_metrics,
    recording_metrics,
    sigmoid,
)
from eeg_cnn_lstm.utils.splits import make_split_loaders
from eeg_cnn_lstm.utils.stopping import (
    EarlyStopping,
    build_scheduler,
    current_lr,
)
from eeg_cnn_lstm.models.model import CNN_LSTM, ModelConfig

## @brief Supported autocast dtypes, selected by `train.amp_dtype`.
AMP_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

## @brief Key of the recording-level partial AUC in a `recording_metrics` dict.
PAUC_KEY = f"pauc_fpr{DEFAULT_MAX_FPR:g}"

# --------------
# Set up helpers
# --------------


def set_seeds(seed: int) -> None:
    """
    @brief Seed Python's `random`, NumPy, and PyTorch for reproducible runs.

    @param seed RNG seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(spec: str) -> torch.device:
    """
    @brief Resolve a device spec into a `torch.device`.

    @param spec One of "auto", "cpu", "cuda", or "cuda:N". "auto" picks CUDA
                if a GPU is visible to PyTorch, otherwise CPU.
    @return Resolved device.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_config(path: str) -> dict[str, Any]:
    """
    @brief Load a YAML training config into a dict.

    @param path Filesystem path to the YAML file.
    @return Parsed config as a nested dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_eval_loader(
    manifest_path: str | os.PathLike,
    data_dir: str | os.PathLike,
    batch_size: int,
    num_workers: int,
    seed: int,
    normalize: str = "zscore",
    norm_stats: Any = None,
    clip_uv: float | None = None,
) -> DataLoader:
    """
    @brief Build a DataLoader over an entire manifest with no internal split.

    @details
    Used for the final eval pass on the held-out test set: one loader covering
    every recording, using every segment of each recording. Normalization must
    match training, so the same options are threaded through.

    @param manifest_path Path to the eval manifest CSV.
    @param data_dir Directory containing the eval `.npy` files.
    @param batch_size Mini-batch size.
    @param num_workers DataLoader worker count.
    @param seed RNG seed (unused for eval but kept for parity).
    @param normalize Normalization mode; must match training.
    @param norm_stats Stats JSON path/dict, required for `"uv_scale"`.
    @param clip_uv Clip threshold override for `"uv_scale"`.
    @return DataLoader yielding `(x, y, rec_idx)` from the eval set.
    """
    manifest = load_manifest(manifest_path)
    dataset = TUABEpochDataset(
        manifest,
        data_dir,
        segments_per_recording=None,
        normalize=normalize,
        norm_stats=norm_stats,
        clip_uv=clip_uv,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


# ------------------
# Train / eval phase
# ------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    max_steps: int = 0,
    log_every: int = 200,
    eval_hook: Optional[Callable[[int, float], bool]] = None,
    eval_every: int = 0,
) -> tuple[float, dict[str, float | int], bool]:
    """
    @brief Run one training epoch with mixed precision, grad clipping, and timing.

    @details
    The forward pass runs in `amp_dtype` autocast (FP16 by default). Under FP16
    the GradScaler is enabled: the loss is scaled, backpropagated, then unscaled
    before gradient clipping, so the clip threshold applies to the true
    (un-scaled) gradients. Under BF16 (or FP32) the scaler is disabled and
    `scale`/`unscale_`/`step`/`update` are pass-throughs, so the same code path
    clips true-scale gradients either way.

    Batches are `(x, y, rec_idx)`; the recording index is unused during
    training and discarded here.

    Timing: for each step, *data wait* is the time the loop is blocked waiting
    for the DataLoader to yield the next batch; *compute* is everything from
    receiving the batch to the end of the step. `loss.item()` forces a GPU
    sync, so compute time is measured accurately. The first batch (which
    includes worker start-up) is reported separately and excluded from the
    data-wait share.

    Mid-epoch validation: when `eval_hook` and `eval_every` are given, the hook
    is called every `eval_every` steps with `(step, loss_since_last_hook)` and
    returns True to abort the epoch (early stopping). The hook's wall time is
    measured separately and excluded from both the data-wait accounting and the
    reported epoch time, so throughput numbers stay comparable with runs that
    validate only at the epoch boundary. The hook is responsible for restoring
    train mode on the model; this function does so as well, defensively.

    @param model The model in train mode.
    @param loader Train DataLoader.
    @param criterion Binary classification loss (e.g., `BCEWithLogitsLoss`).
    @param optimizer Optimizer (e.g., `AdamW`).
    @param scaler `torch.amp.GradScaler`; enabled for FP16, pass-through otherwise.
    @param device Target device for tensors.
    @param grad_clip Max gradient norm. Pass `0.0` to disable.
    @param use_amp Whether to run the forward pass under autocast.
    @param amp_dtype Autocast dtype (`torch.float16` or `torch.bfloat16`).
    @param max_steps Stop after this many steps (0 = full epoch).
    @param log_every Log throughput every N steps (0 = never).
    @param eval_hook Called every `eval_every` steps as
           `hook(step, loss_since_last_call)`; returns True to stop.
    @param eval_every Steps between `eval_hook` calls (0 = never).
    @return Tuple `(mean_loss, timing_stats, stopped_early)`.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    t_epoch = time.perf_counter()
    t_prev_end = t_epoch
    first_batch_s = 0.0
    data_s = 0.0
    compute_s = 0.0
    hook_s = 0.0
    win_samples = 0
    win_data_s = 0.0
    win_start = t_epoch
    chk_loss = 0.0
    chk_samples = 0
    step = 0
    stopped = False

    for step, (x, y, _) in enumerate(loader, start=1):
        t_ready = time.perf_counter()
        wait = t_ready - t_prev_end
        if step == 1:
            first_batch_s = wait
        else:
            data_s += wait
            win_data_s += wait

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        loss_val = loss.item()  # GPU sync point
        total_loss += loss_val * bs
        total_samples += bs
        win_samples += bs
        chk_loss += loss_val * bs
        chk_samples += bs

        t_prev_end = time.perf_counter()
        compute_s += t_prev_end - t_ready

        if log_every and step % log_every == 0:
            win_s = t_prev_end - win_start
            logger.info(
                f"  step {step:>6}  loss={loss_val:.4f}  "
                f"{win_samples / win_s:,.0f} samples/s  "
                f"data_wait={win_data_s / win_s:.1%}"
            )
            win_samples, win_data_s, win_start = 0, 0.0, t_prev_end

        if max_steps and step >= max_steps:
            break

        if eval_hook is not None and eval_every and step % eval_every == 0:
            t_hook = time.perf_counter()
            stopped = bool(eval_hook(step, chk_loss / max(1, chk_samples)))
            hook_s += time.perf_counter() - t_hook
            chk_loss, chk_samples = 0.0, 0
            model.train()  # the hook ran evaluation; restore train mode
            # Charge none of the hook's wall time to data wait or throughput.
            t_prev_end = time.perf_counter()
            win_samples, win_data_s, win_start = 0, 0.0, t_prev_end
            if stopped:
                break

    epoch_s = time.perf_counter() - t_epoch - hook_s
    steady_s = data_s + compute_s
    stats = {
        "steps": step,
        "samples": total_samples,
        "epoch_seconds": round(epoch_s, 2),
        "samples_per_sec": round(total_samples / epoch_s, 1) if epoch_s else 0.0,
        "first_batch_seconds": round(first_batch_s, 2),
        "data_wait_seconds": round(data_s, 2),
        "compute_seconds": round(compute_s, 2),
        "data_wait_frac": round(data_s / steady_s, 4) if steady_s else 0.0,
        "mid_epoch_eval_seconds": round(hook_s, 2),
        "tail_loss": round(chk_loss / chk_samples, 6) if chk_samples else None,
    }
    return total_loss / max(1, total_samples), stats, stopped


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, np.ndarray]]:
    """
    @brief Run a full FP32 evaluation pass and return every per-segment score.

    @details
    No autocast: evaluation is cheap relative to training, and FP32 keeps the
    logits (and therefore the metrics) precision-independent. Returning the raw
    arrays rather than only summary metrics is what allows aggregation,
    calibration and thresholds to be re-derived later without re-running the
    model.

    @param model The model (set to eval mode internally).
    @param loader Loader yielding `(x, y, rec_idx)`.
    @param criterion Same loss as training, for tracking the eval loss.
    @param device Target device for tensors.
    @return Tuple `(avg_loss, arrays)` where `arrays` holds `logits` (float32),
            `labels` (int8) and `rec_idx` (int32), one entry per segment.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_l: list[torch.Tensor] = []
    labels_l: list[torch.Tensor] = []
    recs_l: list[torch.Tensor] = []

    for x, y, rec in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        logits_l.append(logits.detach().float().cpu())
        labels_l.append(y.detach().cpu())
        recs_l.append(rec.detach().cpu())

    if total_samples == 0:
        empty = np.empty(0)
        return 0.0, {"logits": empty, "labels": empty, "rec_idx": empty}

    arrays = {
        "logits": torch.cat(logits_l).numpy().astype(np.float32),
        "labels": torch.cat(labels_l).numpy().astype(np.int8),
        "rec_idx": torch.cat(recs_l).numpy().astype(np.int32),
    }
    return total_loss / total_samples, arrays


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int]]:
    """
    @brief Segment-level evaluation pass (Phase 0 signature, kept for callers).

    @param model The model in eval mode.
    @param loader Eval / val DataLoader.
    @param criterion Same loss as training, for tracking the eval loss.
    @param device Target device for tensors.
    @return Tuple `(avg_loss, segment_metrics)`.
    """
    loss, arrays = collect_predictions(model, loader, criterion, device)
    if len(arrays["logits"]) == 0:
        return 0.0, {}
    metrics = compute_metrics(
        torch.from_numpy(arrays["logits"]),
        torch.from_numpy(arrays["labels"].astype(np.float32)),
    )
    return loss, metrics


def summarize_predictions(
    arrays: dict[str, np.ndarray],
    dataset: TUABEpochDataset,
    aggregator: str = "mean",
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """
    @brief Compute segment-level and recording-level metrics from saved scores.

    @param arrays Output of `collect_predictions`.
    @param dataset The dataset the scores came from; supplies the recording
           table (labels per recording).
    @param aggregator Segment-to-recording method (see `metrics.AGGREGATORS`).
    @return Tuple `(segment_metrics, recording_metrics)`.
    """
    if len(arrays["logits"]) == 0:
        return {}, {}
    seg = compute_metrics(
        torch.from_numpy(arrays["logits"]),
        torch.from_numpy(arrays["labels"].astype(np.float32)),
    )
    probs = sigmoid(arrays["logits"].astype(np.float64))
    rec_labels = dataset.recordings["label"].to_numpy()
    scores = aggregate_to_recordings(
        probs, arrays["rec_idx"], aggregator, n_recordings=len(rec_labels)
    )
    rec = recording_metrics(rec_labels, scores)
    rec["aggregator"] = aggregator  # type: ignore[assignment]
    return seg, rec


def save_predictions(
    path: Path,
    arrays: dict[str, np.ndarray],
    dataset: TUABEpochDataset,
) -> None:
    """
    @brief Write per-segment scores plus the recording table to a `.npz`.

    @details
    Everything needed for post-hoc work is stored: segment logits, the
    recording each belongs to, and per-recording filenames, patient IDs and
    labels (patient IDs make the clustered bootstrap possible).

    @param path Destination `.npz` path.
    @param arrays Output of `collect_predictions`.
    @param dataset The dataset the scores came from.
    """
    recs = dataset.recordings
    np.savez_compressed(
        path,
        logits=arrays["logits"],
        labels=arrays["labels"],
        rec_idx=arrays["rec_idx"],
        rec_filename=recs["filename"].to_numpy().astype("U64"),
        rec_subject_id=recs["subject_id"].to_numpy().astype("U16"),
        rec_label=recs["label"].to_numpy().astype(np.int8),
        rec_n_epochs=recs["n_epochs"].to_numpy().astype(np.int32),
    )


# ------------
# Main routine
# ------------


def train(
    config_path: str,
    train_data_dir: str | None = None,
    output_dir: str | None = None,
    num_epochs: int | None = None,
    max_steps: int = 0,
    fold: Optional[int] = None,
) -> dict[str, Any]:
    """
    @brief Run the full training routine driven by a YAML config.

    @param config_path Path to the YAML config file.
    @param train_data_dir Optional override for `data.train_data_dir`
           (e.g. a node-local staged copy).
    @param output_dir Optional override for `train.output_dir`.
    @param num_epochs Optional override for `train.num_epochs`.
    @param max_steps Optional cap on training steps per epoch (0 = no cap).
    @param fold Optional CV fold index; requires `data.splits_csv`.
    @return Dict with the best score, the best epoch's metrics and output
            paths, so an HPO driver can consume the result directly.
    """
    config = load_config(config_path)

    data_cfg = config["data"]
    loader_cfg = config["loader"]
    train_cfg = config["train"]
    eval_cfg = config.get("eval", {})

    # ---- CLI overrides (recorded in the log for provenance) ----
    overrides: dict[str, Any] = {}
    if train_data_dir:
        data_cfg["train_data_dir"] = overrides["train_data_dir"] = train_data_dir
    if output_dir:
        train_cfg["output_dir"] = overrides["output_dir"] = output_dir
    if num_epochs:
        train_cfg["num_epochs"] = overrides["num_epochs"] = num_epochs
    if max_steps:
        overrides["max_steps"] = max_steps
    if fold is not None:
        train_cfg["fold"] = overrides["fold"] = fold

    set_seeds(int(loader_cfg.get("seed", 42)))
    device = select_device(str(train_cfg.get("device", "auto")))

    output_dir_p = Path(train_cfg.get("output_dir", "outputs"))
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(output_dir_p / "train.log"), level="DEBUG")

    logger.info(f"Config: {config_path}")
    logger.info(f"CLI overrides: {overrides or 'none'}")
    logger.info(f"Device: {device}")
    logger.info(f"Train data dir: {data_cfg['train_data_dir']}")
    logger.info(f"Output dir: {output_dir_p.resolve()}")

    # ---- Dataloaders ----
    num_workers = int(loader_cfg.get("num_workers", 0))
    batch_size = int(loader_cfg["batch_size"])
    seed = int(loader_cfg.get("seed", 42))
    normalize = str(loader_cfg.get("normalize", "zscore"))
    norm_stats = loader_cfg.get("norm_stats")
    clip_uv = loader_cfg.get("clip_uv")
    resample_each_epoch = bool(loader_cfg.get("resample_each_epoch", False))
    segment_cap = loader_cfg.get("max_epochs_per_recording")
    splits_csv = data_cfg.get("splits_csv")
    fold_idx = train_cfg.get("fold")
    aggregator = str(train_cfg.get("aggregator", "mean"))

    test_loader: Optional[DataLoader] = None
    if splits_csv:
        # Phase 1 path: train on E (minus a fold), score on T.
        sl = make_split_loaders(
            data_dir=data_cfg["train_data_dir"],
            splits=splits_csv,
            fold=fold_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            segments_per_recording=segment_cap,
            resample_each_epoch=resample_each_epoch,
            normalize=normalize,
            norm_stats=norm_stats,
            clip_uv=clip_uv,
        )
        train_loader, val_loader, test_loader = sl.train, sl.tune, sl.test
        info = sl.info
        val_name = "T"
        default_select = "recording_pauc"
    else:
        # Legacy Phase 0 path: subject-disjoint 80/20 of the training manifest.
        train_loader, val_loader, info = make_train_val_dataloaders(
            manifest_path=data_cfg["train_manifest"],
            data_dir=data_cfg["train_data_dir"],
            batch_size=batch_size,
            val_frac=float(loader_cfg["val_frac"]),
            max_epochs_per_recording=segment_cap,
            num_workers=num_workers,
            seed=seed,
            normalize=normalize,
            norm_stats=norm_stats,
            clip_uv=clip_uv,
            resample_each_epoch=resample_each_epoch,
        )
        val_name = "val"
        default_select = "segment_auc"

    select_metric = str(train_cfg.get("select_metric", default_select))
    if select_metric not in {"recording_pauc", "segment_auc"}:
        raise ValueError(
            "train.select_metric must be 'recording_pauc' or 'segment_auc'; "
            f"got {select_metric!r}"
        )
    if select_metric == "recording_pauc" and not splits_csv:
        logger.warning(
            "select_metric=recording_pauc without data.splits_csv: selecting on "
            "the legacy validation split."
        )

    logger.info(f"Split info: {json.dumps(info, default=str)}")
    logger.info(f"Selection metric: {select_metric}; aggregator: {aggregator}")

    # ---- Model ----
    model = CNN_LSTM(ModelConfig()).to(device)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # ---- Optimizer / loss / AMP ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.BCEWithLogitsLoss()

    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    amp_dtype_name = str(train_cfg.get("amp_dtype", "fp16")).lower()
    if amp_dtype_name not in AMP_DTYPES:
        raise ValueError(
            f"train.amp_dtype must be one of {sorted(AMP_DTYPES)}; got {amp_dtype_name!r}"
        )
    amp_dtype = AMP_DTYPES[amp_dtype_name]
    # FP16 needs dynamic loss scaling; BF16 has FP32's range and does not.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype is torch.float16
    )
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 200))

    # ---- Early stopping / LR schedule (Step 2) ----
    # Both are driven by validation checks, not epochs, and share one min_delta
    # so "no progress" means the same thing to the stopper and the scheduler.
    es_cfg = dict(train_cfg.get("early_stopping") or {})
    es_enabled = bool(es_cfg.get("enabled", False))
    min_delta = float(es_cfg.get("min_delta", 0.002))
    restore_best = bool(es_cfg.get("restore_best", True))
    val_every_frac = float(es_cfg.get("val_every_frac") or 0.0)
    if not 0.0 <= val_every_frac <= 1.0:
        raise ValueError(
            "train.early_stopping.val_every_frac must be in [0, 1]; "
            f"got {val_every_frac}"
        )

    stopper = (
        EarlyStopping(
            patience=int(es_cfg.get("patience", 12)),
            min_delta=min_delta,
            mode="max",  # both selection metrics are larger-is-better
            min_checks=int(es_cfg.get("min_checks", 0)),
        )
        if es_enabled
        else None
    )
    scheduler = build_scheduler(
        optimizer,
        train_cfg.get("lr_schedule"),
        mode="max",
        min_delta=min_delta,
        stop_patience=stopper.patience if stopper else None,
    )

    # Mid-epoch validation cadence. A check that would land on the final step
    # is dropped: the epoch-boundary check already covers that point.
    steps_per_epoch = len(train_loader)
    if max_steps:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    eval_every = (
        int(round(steps_per_epoch * val_every_frac)) if val_every_frac > 0 else 0
    )
    if eval_every >= steps_per_epoch:
        eval_every = 0

    precision = amp_dtype_name if use_amp else "fp32"
    logger.info(
        f"AMP: {precision} (GradScaler "
        f"{'on' if scaler.is_enabled() else 'off'}); grad clip: {grad_clip}"
    )
    if use_amp and amp_dtype is torch.bfloat16:
        logger.warning(
            "bf16 autocast disables the fused cuDNN LSTM kernel; expect ~9x "
            "slower training with this architecture."
        )

    # ---- Run record ----
    bench: dict[str, Any] = {
        "config": config_path,
        "overrides": overrides,
        "train_data_dir": data_cfg["train_data_dir"],
        "splits_csv": splits_csv,
        "fold": fold_idx,
        "device": str(device),
        "amp_dtype": precision,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "normalize": normalize,
        "clip_uv": clip_uv,
        "resample_each_epoch": resample_each_epoch,
        "segments_per_recording": segment_cap,
        "select_metric": select_metric,
        "aggregator": aggregator,
        "early_stopping": {**es_cfg, "steps_per_epoch": steps_per_epoch,
                           "eval_every_steps": eval_every},
        "lr_schedule": dict(train_cfg.get("lr_schedule") or {}),
        "lr_initial": current_lr(optimizer),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "stage_seconds": float(os.environ["STAGE_SECONDS"]) if os.environ.get("STAGE_SECONDS") else None,
        "split_info": info,
        "epochs": [],
        "checks": [],
    }
    bench_path = output_dir_p / "benchmark.json"

    # ---- Training loop ----
    n_epochs = int(train_cfg.get("num_epochs", 2))
    best_ckpt_path = output_dir_p / "best.pt"
    pred_path = output_dir_p / "predictions_tune.npz"

    if eval_every:
        cadence = (
            f"every {eval_every:,} steps ({val_every_frac:g} epoch) "
            "plus the epoch boundary"
        )
    else:
        cadence = "at the epoch boundary only"
    if stopper is not None:
        policy = (
            f"early stopping on {select_metric}, patience {stopper.patience} "
            f"checks, min_delta {min_delta:g}"
        )
    else:
        policy = "early stopping disabled, running the full budget"
    logger.info(
        f"Epoch budget: {n_epochs} x {steps_per_epoch:,} steps; "
        f"validating {cadence}; {policy}."
    )
    if scheduler is not None:
        logger.info(
            f"LR schedule: ReduceLROnPlateau(factor="
            f"{scheduler.factor}, patience={scheduler.patience} checks, "
            f"min_lr={scheduler.min_lrs[0]:g}) from lr={current_lr(optimizer):.2e}"
        )

    ## Mutable state shared with the validation hook below.
    state: dict[str, Any] = {
        "epoch": 0,
        "best_score": -float("inf"),
        "best_epoch": -1,
        "best_check": -1,
        "best_frac": float("nan"),
        "best_seg": {},
        "best_rec": {},
        "last_check_step": -1,
        "last_record": None,
    }

    def run_check(step_in_epoch: int, window_loss: float) -> bool:
        """
        @brief One validation check: score, record, select, schedule, decide.

        @details
        Called both mid-epoch (from `train_one_epoch`) and at the epoch
        boundary, so there is exactly one code path for evaluation and
        checkpointing regardless of cadence.

        `is_record` and the stop decision come from the stopper, which
        separates them: a score may be the best yet (keep the weights) without
        clearing `min_delta` (so patience still advances).

        @param step_in_epoch Step index within the current epoch.
        @param window_loss Mean training loss since the previous check.
        @return True if the run should stop now.
        """
        epoch = int(state["epoch"])
        t_val = time.perf_counter()
        val_loss, arrays = collect_predictions(model, val_loader, criterion, device)
        seg_metrics, rec_metrics = summarize_predictions(
            arrays, val_loader.dataset, aggregator
        )
        val_s = time.perf_counter() - t_val

        if select_metric == "recording_pauc":
            score = float(rec_metrics.get(PAUC_KEY, float("nan")))
        else:
            score = float(seg_metrics.get("auc_roc", float("nan")))
        if score != score:  # NaN: fall back to segment accuracy
            score = float(seg_metrics.get("accuracy", -float("inf")))

        if stopper is not None:
            res = stopper.update(score)
            is_record, stop, n_bad, check_no = (
                res.is_record,
                res.should_stop,
                res.n_bad,
                res.n_checks,
            )
        else:
            is_record = score > float(state["best_score"])
            stop, n_bad = False, 0
            check_no = len(bench["checks"]) + 1

        lr_before = current_lr(optimizer)
        if scheduler is not None:
            scheduler.step(score)
        lr_after = current_lr(optimizer)

        frac = (epoch - 1) + step_in_epoch / max(1, steps_per_epoch)
        logger.info(
            f"  check {check_no:>3} @ epoch {frac:.2f}  "
            f"train_loss={window_loss:.4f}  {val_name}_loss={val_loss:.4f}  "
            f"{select_metric}={score:.4f}  lr={lr_after:.2e}  "
            f"bad={n_bad}/{stopper.patience if stopper else '-'}  "
            f"eval {val_s / 60:.1f} min"
        )
        logger.info(
            f"    {format_recording_metrics(rec_metrics, prefix=f'{val_name}rec')}"
        )
        if lr_after < lr_before:
            logger.info(f"    LR reduced {lr_before:.2e} -> {lr_after:.2e}")

        record = {
            "check": check_no,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "epoch_frac": round(frac, 4),
            "train_loss": window_loss,
            "val_loss": val_loss,
            "segment_metrics": seg_metrics,
            "recording_metrics": rec_metrics,
            "score": score,
            "lr": lr_after,
            "n_bad": n_bad,
            "is_best": bool(is_record),
            "val_seconds": round(val_s, 2),
        }
        bench["checks"].append(record)
        state["last_check_step"] = step_in_epoch
        state["last_record"] = record

        if is_record:
            state.update(
                best_score=score,
                best_epoch=epoch,
                best_check=check_no,
                best_frac=round(frac, 4),
                best_seg=seg_metrics,
                best_rec=rec_metrics,
            )
            torch.save(model.state_dict(), best_ckpt_path)
            save_predictions(pred_path, arrays, val_loader.dataset)
            logger.info(
                f"    ↑ new best ({select_metric}={score:.4f}); saved "
                f"{best_ckpt_path.name} and {pred_path.name}"
            )

        bench_path.write_text(json.dumps(bench, indent=2, default=str))
        if stop:
            logger.info(f"  early stop: {stopper.reason}")
        return stop

    stopped_early = False
    epochs_run = 0

    for epoch in range(1, n_epochs + 1):
        state["epoch"] = epoch
        state["last_check_step"] = -1
        # Redraw the per-recording segment subset (no-op unless the training set
        # has a cap and resample_each_epoch is on).
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        train_loss, t_stats, stopped_early = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_clip,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_steps=max_steps,
            log_every=log_every,
            eval_hook=run_check if eval_every else None,
            eval_every=eval_every,
        )
        epochs_run = epoch
        steps_done = int(t_stats["steps"])

        # Epoch-boundary check, unless a mid-epoch check already landed on this
        # exact step or the run stopped inside the epoch.
        if not stopped_early and state["last_check_step"] != steps_done:
            tail = t_stats.get("tail_loss")
            stopped_early = run_check(
                steps_done, float(tail if tail is not None else train_loss)
            )

        last = state["last_record"] or {}
        logger.info(
            f"Epoch {epoch}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"{val_name}_loss={last.get('val_loss', float('nan')):.4f}  "
            f"{format_metrics(last.get('segment_metrics', {}), prefix=f'{val_name}seg')}"
        )
        logger.info(
            f"  timing: train {t_stats['epoch_seconds'] / 60:.1f} min "
            f"({t_stats['samples_per_sec']:,.0f} samples/s, "
            f"data_wait {t_stats['data_wait_frac']:.1%}, "
            f"first batch {t_stats['first_batch_seconds']:.1f}s)  "
            f"validation {(t_stats['mid_epoch_eval_seconds'] + last.get('val_seconds', 0.0)) / 60:.1f} min"
        )

        bench["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": last.get("val_loss"),
                "segment_metrics": last.get("segment_metrics", {}),
                "recording_metrics": last.get("recording_metrics", {}),
                "train_timing": t_stats,
                "val_seconds": last.get("val_seconds"),
                "lr": last.get("lr"),
                "complete": not stopped_early,
            }
        )
        bench_path.write_text(json.dumps(bench, indent=2, default=str))

        if stopped_early:
            break

    best_score = float(state["best_score"])
    best_epoch = int(state["best_epoch"])
    best_seg = state["best_seg"]
    best_rec = state["best_rec"]

    if stopped_early:
        logger.info(
            f"Stopped early after {epochs_run} epoch(s) / "
            f"{len(bench['checks'])} checks."
        )
    else:
        logger.info(f"Reached the {n_epochs}-epoch budget without early stopping.")
    logger.info(
        f"Best {select_metric}={best_score:.4f} at check {state['best_check']} "
        f"(epoch {state['best_frac']})."
    )

    # Restore the best weights so anything downstream in this process -- the
    # fold test pass, the final eval, an HPO driver reading the model -- sees
    # the selected model rather than the last (possibly overfit) one.
    if restore_best and best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        logger.info(f"Restored best weights from {best_ckpt_path.name}.")

    bench["best"] = {
        "epoch": best_epoch,
        "epoch_frac": state["best_frac"],
        "check": state["best_check"],
        "select_metric": select_metric,
        "score": best_score,
        "segment_metrics": best_seg,
        "recording_metrics": best_rec,
        "checkpoint": str(best_ckpt_path),
        "predictions": str(pred_path),
    }
    bench["stopping"] = {
        "epochs_run": epochs_run,
        "epoch_budget": n_epochs,
        "checks_run": len(bench["checks"]),
        "stopped_early": stopped_early,
        "restored_best": bool(restore_best and best_ckpt_path.exists()),
        "lr_final": current_lr(optimizer),
        **(stopper.state() if stopper else {"enabled": False}),
    }

    # ---- Held-out fold (CV runs only) ----
    if test_loader is not None:
        logger.info(f"Scoring held-out fold {fold_idx} with the best checkpoint...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        test_loss, test_arrays = collect_predictions(
            model, test_loader, criterion, device
        )
        test_seg, test_rec = summarize_predictions(
            test_arrays, test_loader.dataset, aggregator
        )
        test_pred_path = output_dir_p / "predictions_test.npz"
        save_predictions(test_pred_path, test_arrays, test_loader.dataset)
        logger.info(f"  fold {fold_idx} loss={test_loss:.4f}")
        logger.info(f"  {format_recording_metrics(test_rec, prefix='testrec')}")
        bench["test"] = {
            "fold": fold_idx,
            "loss": test_loss,
            "segment_metrics": test_seg,
            "recording_metrics": test_rec,
            "predictions": str(test_pred_path),
        }

    # ---- Final eval on the held-out TUAB partition (opt-in only) ----
    if eval_cfg.get("run_final_eval", False):
        logger.info("Running final eval on the held-out TUAB eval partition...")
        eval_loader = build_eval_loader(
            manifest_path=data_cfg["eval_manifest"],
            data_dir=data_cfg["eval_data_dir"],
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            normalize=normalize,
            norm_stats=norm_stats,
            clip_uv=clip_uv,
        )
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        final_loss, final_arrays = collect_predictions(
            model, eval_loader, criterion, device
        )
        final_seg, final_rec = summarize_predictions(
            final_arrays, eval_loader.dataset, aggregator
        )
        final_pred_path = output_dir_p / "predictions_eval.npz"
        save_predictions(final_pred_path, final_arrays, eval_loader.dataset)
        logger.info(f"Eval loss: {final_loss:.4f}")
        logger.info(format_metrics(final_seg, prefix="evalseg"))
        logger.info(format_recording_metrics(final_rec, prefix="evalrec"))
        bench["final_eval"] = {
            "loss": final_loss,
            "segment_metrics": final_seg,
            "recording_metrics": final_rec,
            "predictions": str(final_pred_path),
        }

    bench_path.write_text(json.dumps(bench, indent=2, default=str))
    logger.info(f"Run record written to {bench_path}")
    return bench


def parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace with the config path and optional overrides.
    """
    parser = argparse.ArgumentParser(description="Train CNN+LSTM EEG classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config (e.g., configs/baseline.yaml)",
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=None,
        help="Override data.train_data_dir (e.g. node-local staged copy)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override train.output_dir"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="Override train.num_epochs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap training steps per epoch (0 = full epoch); for quick smoke tests",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="CV fold index to hold out (requires data.splits_csv)",
    )
    return parser.parse_args()


def main() -> None:
    """@brief CLI entrypoint."""
    args = parse_args()
    train(
        args.config,
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        fold=args.fold,
    )


if __name__ == "__main__":
    main()