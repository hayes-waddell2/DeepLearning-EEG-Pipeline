"""
@file train.py
@brief Training entrypoint for the CNN+LSTM EEG classifier.

@details
One-command training script for the TUH EEG Abnormal Corpus binary classifier.
Reads all settings from a YAML config, builds subject-disjoint train/val
DataLoaders, instantiates the CNN+LSTM model, and runs a mixed-precision
training loop with gradient clipping. Each epoch logs train/val loss and the
full metric set. The best checkpoint (by val AUC, falling back to accuracy
when AUC is undefined) is saved for the optional final-eval pass.

@par Usage:
@verbatim
python -m src.train --config configs/demo.yaml
@endverbatim

@par Config schema:
@verbatim
data:
  train_manifest: <path>
  train_data_dir: <path>
  eval_manifest:  <path>      # required only if eval.run_final_eval = true
  eval_data_dir:  <path>      # required only if eval.run_final_eval = true
loader:
  batch_size: int
  val_frac: float
  max_epochs_per_recording: int | null
  num_workers: int
  seed: int
train:
  num_epochs: int
  lr: float
  weight_decay: float
  grad_clip: float            # 0.0 disables
  use_amp: bool               # auto-disabled on CPU
  device: "auto" | "cpu" | "cuda" | "cuda:N"
  output_dir: str             # checkpoints + log written here
eval:
  run_final_eval: bool        # if true, evaluates best checkpoint on eval split
@endverbatim
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.utils.dataset import (
    TUABEpochDataset,
    load_manifest,
    make_train_val_dataloaders,
)
from src.utils.metrics import compute_metrics, format_metrics
from src.eeg_cnn_lstm.models.model_b import CNN_LSTM, ModelConfig

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
) -> DataLoader:
    """
    @brief Build a DataLoader over an entire manifest with no internal split.

    @details
    Used for the final eval pass on the held-out test set. Unlike
    `make_train_val_dataloaders`, this does not split by subject; it returns
    one loader covering every recording in the manifest.

    @param manifest_path Path to the eval manifest CSV.
    @param data_dir Directory containing the eval `.npy` files.
    @param batch_size Mini-batch size.
    @param num_workers DataLoader worker count.
    @param seed RNG seed (unused for eval but kept for parity).
    @return DataLoader yielding (x, y) pairs from the eval set.
    """
    manifest = load_manifest(manifest_path)
    dataset = TUABEpochDataset(
        manifest,
        data_dir,
        max_epochs_per_recording=None,
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
) -> float:
    """
    @brief Run one full training epoch with optional mixed precision and grad clipping.

    @details
    Loss is computed in autocast scope, scaled, backpropagated, then unscaled
    before gradient clipping so the clip threshold is interpreted in the true
    (un-scaled) loss landscape.

    @param model The model in train mode.
    @param loader Train DataLoader.
    @param criterion Binary classification loss (e.g., `BCEWithLogitsLoss`).
    @param optimizer Optimizer (e.g., `AdamW`).
    @param scaler `torch.amp.GradScaler`. May be disabled (no-op on CPU).
    @param device Target device for tensors.
    @param grad_clip Max gradient norm. Pass `0.0` to disable.
    @return Mean per-sample training loss across the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = scaler.is_enabled()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int]]:
    """
    @brief Run a full evaluation pass and compute classification metrics.

    @details
    Accumulates logits and labels across all batches, then calls
    `compute_metrics` once on the full set. Avoids per-batch metric noise.

    @param model The model in eval mode.
    @param loader Eval / val DataLoader.
    @param criterion Same loss as training, for tracking val_loss.
    @param device Target device for tensors.
    @return Tuple `(avg_loss, metrics_dict)`.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    if total_samples == 0:
        return 0.0, {}

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    metrics = compute_metrics(logits_tensor, labels_tensor)
    return total_loss / total_samples, metrics


# ------------
# Main routine
# ------------


def train(config_path: str) -> None:
    """
    @brief Run the full training routine driven by a YAML config.

    @param config_path Path to the YAML config file.
    """
    config = load_config(config_path)

    data_cfg = config["data"]
    loader_cfg = config["loader"]
    train_cfg = config["train"]
    eval_cfg = config.get("eval", {})

    set_seeds(int(loader_cfg.get("seed", 42)))
    device = select_device(str(train_cfg.get("device", "auto")))

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(output_dir / "train.log"), level="DEBUG")

    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Output dir: {output_dir.resolve()}")

    # ---- Dataloaders ----
    train_loader, val_loader, info = make_train_val_dataloaders(
        manifest_path=data_cfg["train_manifest"],
        data_dir=data_cfg["train_data_dir"],
        batch_size=int(loader_cfg["batch_size"]),
        val_frac=float(loader_cfg["val_frac"]),
        max_epochs_per_recording=loader_cfg.get("max_epochs_per_recording"),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        seed=int(loader_cfg.get("seed", 42)),
    )
    logger.info(f"Split info: {info}")

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    logger.info(f"AMP enabled: {use_amp}; grad clip: {grad_clip}")

    # ---- Training loop ----
    num_epochs = int(train_cfg.get("num_epochs", 2))
    best_score = -float("inf")
    best_ckpt_path = output_dir / "best.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_clip,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"{format_metrics(val_metrics, prefix='val')}"
        )

        # Track best by AUC; fall back to accuracy when AUC is NaN.
        score = val_metrics.get("auc_roc", float("nan"))
        if score != score:  # NaN check
            score = val_metrics.get("accuracy", -float("inf"))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(
                f"  ↑ new best (score={score:.4f}); saved {best_ckpt_path.name}"
            )

    logger.info(f"Training complete. Best val score: {best_score:.4f}")

    # ---- Final eval (optional) ----
    if eval_cfg.get("run_final_eval", False):
        logger.info("Running final eval on held-out test set...")
        eval_loader = build_eval_loader(
            manifest_path=data_cfg["eval_manifest"],
            data_dir=data_cfg["eval_data_dir"],
            batch_size=int(loader_cfg["batch_size"]),
            num_workers=int(loader_cfg.get("num_workers", 0)),
            seed=int(loader_cfg.get("seed", 42)),
        )
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        test_loss, test_metrics = evaluate(model, eval_loader, criterion, device)
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(format_metrics(test_metrics, prefix="test"))


def parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace with `config` attribute pointing to the YAML config path.
    """
    parser = argparse.ArgumentParser(description="Train CNN+LSTM EEG classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config (e.g., configs/demo.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """@brief CLI entrypoint."""
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
