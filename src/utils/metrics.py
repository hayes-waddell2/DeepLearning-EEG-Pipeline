"""
@file metrics.py
@brief Binary classification metrics for the CNN+LSTM EEG classifier.

@details
Compute accuracy, AUC-ROC, F1, and confusion-matrix counts from a tensor of
unnormalized logits and a tensor of binary labels. Designed to be called once
per validation pass on the accumulated outputs of all batches:

@verbatim
all_logits = torch.cat(per_batch_logits)
all_labels = torch.cat(per_batch_labels)
m = compute_metrics(all_logits, all_labels)
print(format_metrics(m, prefix="val"))
@endverbatim

The model returns logits (not probabilities). `compute_metrics` thresholds at
zero for the class prediction (equivalent to `sigmoid(z) >= 0.5`) and applies
the sigmoid internally only where it's actually needed — for the AUC
computation, which ranks samples by predicted probability.
"""

from __future__ import annotations
from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

# ----------
# Public API
# ----------


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.0,
) -> dict[str, float | int]:
    """
    @brief Compute binary classification metrics from logits and labels.

    @details
    Accepts the model's raw logit output (1-D tensor of shape (N,)) and the
    ground-truth labels (1-D tensor of shape (N,) with values in {0, 1}). All
    inputs are moved to CPU and detached internally so it is safe to call on
    GPU tensors that still have grad attached.

    @param logits 1-D tensor of unnormalized logits, one per sample.
    @param labels 1-D tensor of binary labels with values in {0, 1}.
    @param threshold Logit threshold for the positive class. Default 0.0 is
           equivalent to `sigmoid(logit) >= 0.5`.
    @return Dict with keys:
      - `accuracy`: float in [0, 1]
      - `auc_roc`:  float in [0, 1]; `NaN` if only one class is present
      - `f1`:       float in [0, 1]
      - `tp`, `fp`, `fn`, `tn`: confusion-matrix counts (int)
      - `n_pos`, `n_neg`: per-class support (int)
    @throws ValueError If `logits` and `labels` have mismatched lengths.
    """
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Length mismatch: logits {tuple(logits.shape)} "
            f"vs labels {tuple(labels.shape)}"
        )

    logits_np = logits.detach().cpu().numpy().ravel()
    labels_np = labels.detach().cpu().numpy().ravel().astype(int)

    preds = (logits_np >= threshold).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits_np))

    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())

    n = len(labels_np)
    accuracy = (tp + tn) / n if n > 0 else 0.0

    if len(np.unique(labels_np)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels_np, probs))

    f1 = float(f1_score(labels_np, preds, zero_division=0))

    return {
        "accuracy": float(accuracy),
        "auc_roc": auc,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pos": int((labels_np == 1).sum()),
        "n_neg": int((labels_np == 0).sum()),
    }


def format_metrics(
    metrics: Mapping[str, float | int],
    prefix: str = "",
) -> str:
    """
    @brief Render a metrics dict as a single-line human-readable string.

    @details
    Intended for stdout / logger output during training. Pairs nicely with
    `loguru` or `print` once per epoch. The confusion-matrix counts are
    appended at the end so a quick visual check shows whether the classifier
    is collapsing to one class.

    @param metrics Output of `compute_metrics`.
    @param prefix Optional label prefix, e.g., "val" → "val acc=0.7321 ...".
    @return One-line summary string.
    """
    p = f"{prefix} " if prefix else ""
    return (
        f"{p}acc={metrics['accuracy']:.4f}  "
        f"auc={metrics['auc_roc']:.4f}  "
        f"f1={metrics['f1']:.4f}  "
        f"tp={metrics['tp']} fp={metrics['fp']} "
        f"fn={metrics['fn']} tn={metrics['tn']}"
    )


# ----------------------
# Smoke test entry point
# ----------------------


def _main() -> None:
    """
    @brief Smoke test on synthetic data: print metrics on a balanced toy set.

    @details
    Generates 200 random binary labels and creates logits that are mildly
    correlated with the labels (label * 1.5 + standard normal noise) so AUC
    comes out above chance (~0.85) and the formatted output looks realistic.
    """
    torch.manual_seed(0)
    n = 200
    labels = torch.randint(0, 2, (n,))
    logits = labels.float() * 1.5 + torch.randn(n) * 1.0
    m = compute_metrics(logits, labels)
    print(format_metrics(m, prefix="smoke"))
    print(m)


if __name__ == "__main__":
    _main()
