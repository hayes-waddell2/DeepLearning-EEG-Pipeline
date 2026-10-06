"""
@file metrics.py
@brief Classification metrics for the CNN+LSTM EEG classifier, at both the
       segment level and the recording level.

@details
Two layers:

  1. **Segment level** (`compute_metrics`, `format_metrics`) — the Phase 0
     metrics, kept unchanged so training logs stay comparable across phases.
  2. **Recording level** (everything below the divider) — Phase 1. A clinical
     decision is made per recording, not per 10-second window, so the reported
     metrics aggregate each recording's segment scores into one score, then
     evaluate those.

@par Why recording level
A recording contributes ~274 overlapping segments. Scoring segments
independently (a) answers the wrong question, and (b) inflates the apparent
sample size ~274x, because segments from one recording are highly correlated.
Confidence intervals here resample *patients*, which is the unit that actually
varies.

@par Sensitivity-first metrics
A screening tool is judged in the low-false-alarm region, so the headline
numbers are partial AUC over FPR <= 0.10, sensitivity at a specificity floor,
and specificity at a sensitivity floor (the operating point for triage).

@par Aggregators
`AGGREGATORS` maps a name to a function over one recording's segment
probabilities. All are post-hoc: given saved segment scores, every aggregator
and every threshold can be swept in seconds with no retraining.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

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
    Segment-level metrics, as used throughout Phase 0. Accepts the model's raw
    logit output (1-D tensor of shape (N,)) and the ground-truth labels. All
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

    @param metrics Output of `compute_metrics`.
    @param prefix Optional label prefix, e.g., "val" -> "val acc=0.7321 ...".
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


# =============================================================================
# Recording-level metrics (Phase 1)
# =============================================================================

## @brief Default FPR ceiling for partial AUC (the screening-relevant region).
DEFAULT_MAX_FPR: float = 0.10

## @brief Default specificity floor for `sensitivity_at_specificity`.
DEFAULT_SPEC_FLOOR: float = 0.90

## @brief Default sensitivity floor for the triage operating point.
DEFAULT_SENS_FLOOR: float = 0.95


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """
    @brief Numerically stable logistic function.

    @param logits Array of logits.
    @return Array of probabilities in (0, 1).
    """
    out = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    e = np.exp(logits[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _top_frac_mean(p: np.ndarray, frac: float) -> float:
    """
    @brief Mean of the highest `frac` of a recording's segment probabilities.

    @details
    Length-invariant (unlike a fixed top-k) and far less brittle than the
    maximum, which a single artifact segment can dominate.

    @param p Segment probabilities for one recording.
    @param frac Fraction of segments to average, in (0, 1].
    @return Aggregated score.
    """
    k = max(1, int(np.ceil(len(p) * frac)))
    return float(np.mean(np.sort(p)[-k:]))


def _trimmed_mean(p: np.ndarray, trim: float) -> float:
    """
    @brief Mean after discarding the lowest and highest `trim` of values.

    @param p Segment probabilities for one recording.
    @param trim Fraction removed from each tail, in [0, 0.5).
    @return Aggregated score.
    """
    if len(p) < 3 or trim <= 0:
        return float(np.mean(p))
    k = int(np.floor(len(p) * trim))
    s = np.sort(p)
    core = s[k : len(p) - k] if len(p) - 2 * k > 0 else s
    return float(np.mean(core))


## @brief Aggregators mapping one recording's segment probabilities to a score.
AGGREGATORS: dict[str, Callable[[np.ndarray], float]] = {
    # Denoising baseline: lowest variance, dilutes focal events.
    "mean": lambda p: float(np.mean(p)),
    # Mean in log-odds space: weights confident segments more heavily.
    "mean_logit": lambda p: float(
        np.mean(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
    ),
    # Classic multiple-instance assumption; artifact-brittle and length-biased.
    "max": lambda p: float(np.max(p)),
    # Compromise: focal sensitivity without letting one segment decide.
    "top10pct": lambda p: _top_frac_mean(p, 0.10),
    "top20pct": lambda p: _top_frac_mean(p, 0.20),
    "top5pct": lambda p: _top_frac_mean(p, 0.05),
    # Smoothed top-fraction; robust by construction.
    "q90": lambda p: float(np.quantile(p, 0.90)),
    # Interpretable: "what share of this recording looks abnormal".
    "frac_above_0.5": lambda p: float(np.mean(p > 0.5)),
    # Robust to artifacts in both tails; trims focal evidence too.
    "trimmed_mean_10": lambda p: _trimmed_mean(p, 0.10),
}


def aggregate_to_recordings(
    probs: np.ndarray,
    rec_idx: np.ndarray,
    method: str = "mean",
    n_recordings: Optional[int] = None,
) -> np.ndarray:
    """
    @brief Collapse segment probabilities into one score per recording.

    @param probs 1-D array of segment probabilities.
    @param rec_idx 1-D array of recording indices, same length as `probs`.
    @param method Key into `AGGREGATORS`.
    @param n_recordings Expected number of recordings; inferred if omitted.
           Pass it when some recordings may contribute no segments.
    @return 1-D array of length `n_recordings`; `NaN` where a recording has
            no segments.
    @throws KeyError If `method` is not a known aggregator.
    @throws ValueError If `probs` and `rec_idx` lengths differ.
    """
    if method not in AGGREGATORS:
        raise KeyError(f"unknown aggregator {method!r}; have {sorted(AGGREGATORS)}")
    if len(probs) != len(rec_idx):
        raise ValueError("probs and rec_idx must be the same length")

    n = int(n_recordings if n_recordings is not None else (rec_idx.max() + 1))
    fn = AGGREGATORS[method]
    out = np.full(n, np.nan, dtype=np.float64)

    # Group by recording with one sort rather than n boolean masks.
    order = np.argsort(rec_idx, kind="stable")
    sorted_rec = rec_idx[order]
    sorted_probs = probs[order]
    bounds = np.searchsorted(sorted_rec, np.arange(n + 1))
    for r in range(n):
        lo, hi = bounds[r], bounds[r + 1]
        if hi > lo:
            out[r] = fn(sorted_probs[lo:hi])
    return out


def partial_auc(
    labels: np.ndarray, scores: np.ndarray, max_fpr: float = DEFAULT_MAX_FPR
) -> float:
    """
    @brief Standardized partial AUC over FPR in [0, `max_fpr`].

    @details
    Uses sklearn's McClish-standardized partial AUC, which rescales the region
    so 0.5 is chance and 1.0 is perfect, making it readable like a full AUC.
    This is the HPO objective: it focuses on the low-false-alarm region that
    matters for screening, and is threshold-free so trials are not penalized
    for a poorly placed default cut-off.

    @param labels Binary labels (0/1).
    @param scores Higher means more likely positive.
    @param max_fpr FPR ceiling, in (0, 1].
    @return Standardized partial AUC, or `NaN` if only one class is present.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def sensitivity_at_specificity(
    labels: np.ndarray, scores: np.ndarray, min_specificity: float = DEFAULT_SPEC_FLOOR
) -> tuple[float, float]:
    """
    @brief Best sensitivity achievable while holding specificity at or above a floor.

    @param labels Binary labels (0/1).
    @param scores Higher means more likely positive.
    @param min_specificity Specificity floor, e.g. 0.90.
    @return Tuple `(sensitivity, threshold)`; `(nan, nan)` if undefined.
    """
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    ok = (1.0 - fpr) >= min_specificity
    if not ok.any():
        return float("nan"), float("nan")
    i = int(np.argmax(np.where(ok, tpr, -np.inf)))
    return float(tpr[i]), float(thr[i])


def specificity_at_sensitivity(
    labels: np.ndarray, scores: np.ndarray, min_sensitivity: float = DEFAULT_SENS_FLOOR
) -> tuple[float, float]:
    """
    @brief Best specificity achievable while holding sensitivity at or above a floor.

    @details
    This is the triage operating point: "catch at least `min_sensitivity` of
    abnormal recordings; how many normal studies can then be deprioritized?"

    @param labels Binary labels (0/1).
    @param scores Higher means more likely positive.
    @param min_sensitivity Sensitivity floor, e.g. 0.95.
    @return Tuple `(specificity, threshold)`; `(nan, nan)` if undefined.
    """
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    ok = tpr >= min_sensitivity
    if not ok.any():
        return float("nan"), float("nan")
    spec = 1.0 - fpr
    i = int(np.argmax(np.where(ok, spec, -np.inf)))
    return float(spec[i]), float(thr[i])


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    @brief Threshold maximizing Youden's J (sensitivity + specificity - 1).

    @param labels Binary labels (0/1).
    @param scores Higher means more likely positive.
    @return Threshold value, or `NaN` if undefined.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def metrics_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """
    @brief Confusion matrix and derived rates at a fixed decision threshold.

    @param labels Binary labels (0/1).
    @param scores Higher means more likely positive.
    @param threshold Scores >= this are predicted positive.
    @return Dict with tp/fp/fn/tn, sensitivity, specificity, precision,
            accuracy, f1 and the threshold used.
    """
    pred = (scores >= threshold).astype(int)
    lab = labels.astype(int)
    tp = int(((pred == 1) & (lab == 1)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "accuracy": float((tp + tn) / max(1, len(lab))),
        "f1": float(f1),
    }


def recording_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    max_fpr: float = DEFAULT_MAX_FPR,
    spec_floor: float = DEFAULT_SPEC_FLOOR,
    sens_floor: float = DEFAULT_SENS_FLOOR,
) -> dict[str, float | int]:
    """
    @brief Full recording-level metric set: threshold-free plus one operating point.

    @param labels Recording labels (0/1); `NaN` scores are dropped.
    @param scores One score per recording.
    @param threshold Operating point to report. If `None`, uses the threshold
           that achieves `sens_floor` (falling back to Youden's J when that is
           unreachable).
    @param max_fpr FPR ceiling for partial AUC.
    @param spec_floor Specificity floor for `sens_at_spec`.
    @param sens_floor Sensitivity floor for `spec_at_sens`.
    @return Dict of metrics, including `n_recordings` and `n_pos`/`n_neg`.
    """
    keep = ~np.isnan(scores)
    labels = np.asarray(labels)[keep].astype(int)
    scores = np.asarray(scores)[keep].astype(float)

    out: dict[str, float | int] = {
        "n_recordings": int(len(labels)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }
    if len(np.unique(labels)) < 2:
        return out

    out["auc_roc"] = float(roc_auc_score(labels, scores))
    out[f"pauc_fpr{max_fpr:g}"] = partial_auc(labels, scores, max_fpr)
    sens, _ = sensitivity_at_specificity(labels, scores, spec_floor)
    spec, thr_sens = specificity_at_sensitivity(labels, scores, sens_floor)
    out[f"sens_at_spec{spec_floor:g}"] = sens
    out[f"spec_at_sens{sens_floor:g}"] = spec

    if threshold is None:
        threshold = thr_sens if np.isfinite(thr_sens) else youden_threshold(labels, scores)
    out.update(metrics_at_threshold(labels, scores, float(threshold)))
    return out


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    groups: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    @brief Bootstrap confidence interval, optionally clustered by patient.

    @details
    When `groups` is given, whole patients are resampled rather than
    recordings. A patient can contribute up to 9 recordings in TUAB, and those
    are correlated, so resampling recordings would understate the uncertainty.

    @param labels Recording labels (0/1).
    @param scores One score per recording.
    @param metric_fn Callable `(labels, scores) -> float`.
    @param groups Patient identifier per recording, or `None` to resample
           recordings directly.
    @param n_boot Number of bootstrap replicates.
    @param alpha Two-sided significance level (0.05 -> 95% CI).
    @param seed RNG seed.
    @return Tuple `(point_estimate, ci_low, ci_high)`.
    """
    keep = ~np.isnan(scores)
    labels = np.asarray(labels)[keep].astype(int)
    scores = np.asarray(scores)[keep].astype(float)
    point = float(metric_fn(labels, scores))

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(labels)

    if groups is None:
        # Resample recordings directly.
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(labels[idx])) < 2:
                continue
            v = metric_fn(labels[idx], scores[idx])
            if np.isfinite(v):
                vals.append(float(v))
    else:
        # Cluster bootstrap: resample whole patients, keeping their recordings
        # together, so correlated recordings do not count as independent.
        g = np.asarray(groups)[keep]
        uniq = np.unique(g)
        blocks = [np.flatnonzero(g == u) for u in uniq]
        n_units = len(blocks)
        for _ in range(n_boot):
            pick = rng.integers(0, n_units, size=n_units)
            idx = np.concatenate([blocks[i] for i in pick])
            if len(np.unique(labels[idx])) < 2:
                continue
            v = metric_fn(labels[idx], scores[idx])
            if np.isfinite(v):
                vals.append(float(v))

    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def sweep_aggregators(
    probs: np.ndarray,
    rec_idx: np.ndarray,
    rec_labels: np.ndarray,
    methods: Optional[Sequence[str]] = None,
    max_fpr: float = DEFAULT_MAX_FPR,
    spec_floor: float = DEFAULT_SPEC_FLOOR,
    sens_floor: float = DEFAULT_SENS_FLOOR,
) -> list[dict[str, float | int | str]]:
    """
    @brief Evaluate every aggregator on one set of saved segment predictions.

    @details
    Aggregation is post-hoc, so this is seconds of CPU on scores that already
    exist: no retraining, and every method is compared on identical model
    output.

    @param probs Segment probabilities.
    @param rec_idx Recording index per segment.
    @param rec_labels Label per recording, indexed by `rec_idx`.
    @param methods Aggregator names; defaults to all of `AGGREGATORS`.
    @param max_fpr FPR ceiling for partial AUC.
    @param spec_floor Specificity floor.
    @param sens_floor Sensitivity floor.
    @return List of metric dicts, one per aggregator, sorted by partial AUC
            (descending).
    """
    names = list(methods) if methods else list(AGGREGATORS)
    n_rec = len(rec_labels)
    rows: list[dict[str, float | int | str]] = []
    for m in names:
        scores = aggregate_to_recordings(probs, rec_idx, m, n_recordings=n_rec)
        row: dict[str, float | int | str] = {"aggregator": m}
        row.update(
            recording_metrics(
                np.asarray(rec_labels),
                scores,
                max_fpr=max_fpr,
                spec_floor=spec_floor,
                sens_floor=sens_floor,
            )
        )
        rows.append(row)
    key = f"pauc_fpr{max_fpr:g}"
    rows.sort(key=lambda r: (-(r.get(key) or -np.inf)))
    return rows


def format_recording_metrics(
    metrics: Mapping[str, float | int], prefix: str = ""
) -> str:
    """
    @brief One-line summary of a recording-level metric dict, for logs.

    @param metrics Output of `recording_metrics`.
    @param prefix Optional label prefix, e.g. "T".
    @return One-line summary string.
    """
    p = f"{prefix} " if prefix else ""
    pauc = next((v for k, v in metrics.items() if k.startswith("pauc")), float("nan"))
    sens_spec = next(
        (v for k, v in metrics.items() if k.startswith("sens_at_spec")), float("nan")
    )
    spec_sens = next(
        (v for k, v in metrics.items() if k.startswith("spec_at_sens")), float("nan")
    )
    return (
        f"{p}n={metrics.get('n_recordings', 0)}  "
        f"auc={metrics.get('auc_roc', float('nan')):.4f}  "
        f"pauc={pauc:.4f}  "
        f"sens@spec={sens_spec:.4f}  "
        f"spec@sens={spec_sens:.4f}  "
        f"sens={metrics.get('sensitivity', float('nan')):.4f} "
        f"spec={metrics.get('specificity', float('nan')):.4f}"
    )


# ----------------------
# Smoke test entry point
# ----------------------


def _main() -> None:
    """
    @brief Smoke test on synthetic data: segment and recording metrics.

    @details
    Builds 300 synthetic recordings in which only 5% of an abnormal recording's
    segments look abnormal, with segment counts spanning 50-2000 (like TUAB's
    20x duration range). That combination is what separates the aggregators:
    top-fraction methods find the focal evidence, `max` suffers from length
    bias, and `trimmed_mean` discards exactly the segments that matter. Run as
    `python -m eeg_cnn_lstm.utils.metrics`.
    """
    rng = np.random.default_rng(0)
    n_rec = 300
    rec_labels = (rng.random(n_rec) < 0.5).astype(int)
    probs, rec_idx = [], []
    for r in range(n_rec):
        n_seg = int(rng.integers(50, 2000))
        base = rng.beta(2, 6, n_seg)
        if rec_labels[r]:
            # focal: only 5% of segments carry the abnormality
            k = max(1, int(n_seg * 0.05))
            base[rng.choice(n_seg, k, replace=False)] = rng.beta(4, 3, k)
        probs.append(base)
        rec_idx.append(np.full(n_seg, r))
    probs = np.concatenate(probs)
    rec_idx = np.concatenate(rec_idx)
    seg_labels = rec_labels[rec_idx]

    logits = torch.tensor(np.log(probs / (1 - probs)))
    seg = compute_metrics(logits, torch.tensor(seg_labels, dtype=torch.float32))
    print("segment level:", format_metrics(seg, prefix="seg"))

    print(f"\n{'aggregator':>16} {'auc':>7} {'pauc':>7} {'sens@spec.9':>12} {'spec@sens.95':>13}")
    for row in sweep_aggregators(probs, rec_idx, rec_labels):
        print(
            f"{row['aggregator']:>16} {row['auc_roc']:>7.4f} "
            f"{row['pauc_fpr0.1']:>7.4f} {row['sens_at_spec0.9']:>12.4f} "
            f"{row['spec_at_sens0.95']:>13.4f}"
        )

    best = aggregate_to_recordings(probs, rec_idx, "top10pct", n_recordings=n_rec)
    point, lo, hi = bootstrap_ci(
        rec_labels, best, lambda y, s: float(roc_auc_score(y, s)), n_boot=500
    )
    print(f"\ntop10pct AUC {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    _main()