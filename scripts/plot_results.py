"""
@file plot_results.py
@brief Render the Phase 2 / Step 1 figures from saved run records and predictions.

@details
Reads what the training runs already wrote - each run's `benchmark.json`
(per-epoch metrics) and `predictions_tune.npz` (per-segment scores) - and
produces presentation-ready PNGs. Nothing is recomputed from the model.

Figures:
  1. `training_curves.png`  - recording-level partial AUC per epoch, one line
     per arm, best epoch marked. The overfitting evidence for early stopping.
  2. `loss_curves.png`      - train vs tuning loss per arm (small multiples;
     both panels share the loss axis, so no dual-axis comparison is implied).
  3. `segment_vs_recording.png` - the aggregation gain, as a paired bar chart.
  4. `aggregator_sweep.png` - partial AUC per aggregation method, both arms.
  5. `roc_recording.png`    - recording-level ROC with the 95%-sensitivity
     operating point marked, plus the FPR <= 0.10 region shaded.

@par Design notes
Two series throughout (one per normalization arm), drawn in the first two
categorical slots (blue #2a78d6, orange #eb6834), which pass the lightness,
chroma, CVD-separation, normal-vision and contrast checks on a light surface.
Every chart is directly labeled as well as legended, so identity never depends
on color alone. Grid and axes are recessive; no chart uses two y-scales.

@par Usage (from repo root)
@verbatim
python -m scripts.plot_results \
    zscore=/shared/rc/eeg-cnn-lstm/runs/ab-zscore_123 \
    uv_scale=/shared/rc/eeg-cnn-lstm/runs/ab-uvscale_124 \
    --out results/phase2/figs
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from eeg_cnn_lstm.utils.metrics import (  # noqa: E402
    DEFAULT_MAX_FPR,
    DEFAULT_SENS_FLOOR,
    aggregate_to_recordings,
    sigmoid,
    specificity_at_sensitivity,
    sweep_aggregators,
)

PAUC_KEY = f"pauc_fpr{DEFAULT_MAX_FPR:g}"

## @brief Categorical slots 1-2; validated on the light surface (see dataviz skill).
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"


def style_axes(ax: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """
    @brief Apply the shared recessive styling to one axes.

    @param ax Axes to style.
    @param xlabel X-axis label.
    @param ylabel Y-axis label.
    @param title Axes title.
    """
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def new_fig(w: float, h: float) -> tuple[plt.Figure, Any]:
    """
    @brief Create a figure on the chart surface.

    @param w Width in inches.
    @param h Height in inches.
    @return Tuple `(figure, axes)`.
    """
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def load_run(path: str | Path) -> dict[str, Any]:
    """
    @brief Load one run directory's benchmark record and predictions.

    @param path Run directory containing `benchmark.json` and
           `predictions_tune.npz`.
    @return Dict with `bench` and `pred` entries.
    @throws FileNotFoundError If either file is missing.
    """
    path = Path(path)
    bench_p, pred_p = path / "benchmark.json", path / "predictions_tune.npz"
    for p in (bench_p, pred_p):
        if not p.is_file():
            raise FileNotFoundError(f"missing {p}")
    return {
        "bench": json.loads(bench_p.read_text()),
        "pred": {k: v for k, v in np.load(pred_p, allow_pickle=False).items()},
        "dir": str(path),
    }


def curve_points(bench: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    """
    @brief Read a per-check curve, falling back to per-epoch for older runs.

    @details
    Step 2 runs validate several times per epoch and record each check in
    `bench["checks"]` with a fractional epoch position; Step 1 runs have only
    `bench["epochs"]`. Both are plotted on the same fractional-epoch x-axis so
    a run with early stopping and one without are directly comparable.

    @param bench Parsed `benchmark.json`.
    @param key Either `"pauc"` (recording-level partial AUC), `"train_loss"`
           or `"val_loss"`.
    @return Tuple `(x_in_epochs, y)`.
    """
    checks = bench.get("checks")
    rows = checks if checks else bench.get("epochs", [])
    xs, ys = [], []
    for row in rows:
        xs.append(float(row.get("epoch_frac", row.get("epoch", 0))))
        if key == "pauc":
            ys.append(float(row.get("recording_metrics", {}).get(PAUC_KEY, np.nan)))
        else:
            v = row.get(key)
            ys.append(float(v) if v is not None else np.nan)
    return xs, ys


def lr_drop_points(bench: dict[str, Any]) -> list[float]:
    """
    @brief Fractional-epoch positions at which the learning rate was reduced.

    @param bench Parsed `benchmark.json`.
    @return List of x positions (empty when no scheduler ran).
    """
    checks = bench.get("checks") or []
    drops, prev = [], None
    for row in checks:
        lr = row.get("lr")
        if lr is None:
            continue
        if prev is not None and lr < prev:
            drops.append(float(row.get("epoch_frac", row.get("epoch", 0))))
        prev = lr
    return drops


def fig_training_curves(runs: dict[str, dict[str, Any]], out: Path) -> None:
    """
    @brief Recording-level partial AUC over training, with the best point marked.

    @details
    Plotted against fractional epochs so sub-epoch validation checks (Step 2)
    and epoch-boundary checks (Step 1) share one axis. Learning-rate reductions
    are marked with a tick on the line, and a run that stopped early ends with
    a square terminator and an annotation naming the epochs saved.

    @param runs Mapping of run name to loaded run.
    @param out Output directory.
    """
    fig, ax = new_fig(7.6, 4.4)
    x_max = max(
        (max(curve_points(r["bench"], "pauc")[0] or [0]) for r in runs.values()),
        default=1.0,
    )
    notes: list[tuple[float, float, str, str, int]] = []  # x, y, text, color, slot

    for i, (name, r) in enumerate(runs.items()):
        bench = r["bench"]
        xs, vals = curve_points(bench, "pauc")
        if not xs:
            continue
        dense = len(xs) > 14  # hide per-point markers on a long sub-epoch curve
        ax.plot(xs, vals, color=SERIES[i], linewidth=2,
                marker="" if dense else "o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)

        b = int(np.nanargmax(vals))
        ax.scatter([xs[b]], [vals[b]], s=170, facecolor="none",
                   edgecolor=SERIES[i], linewidth=2, zorder=4)
        notes.append((xs[b], vals[b], f"best {vals[b]:.3f} @ epoch {xs[b]:g}",
                      SERIES[i], i))

        # Learning-rate reductions, as a tick on the line.
        for d in lr_drop_points(bench):
            j = int(np.argmin(np.abs(np.asarray(xs) - d)))
            ax.plot([d], [vals[j]], marker="|", markersize=13, color=SERIES[i],
                    markeredgewidth=2, zorder=5)

        stop = bench.get("stopping") or {}
        if stop.get("stopped_early"):
            ax.scatter([xs[-1]], [vals[-1]], s=55, marker="s", color=SERIES[i],
                       edgecolor=SURFACE, linewidth=1.2, zorder=5)
            saved = int(stop.get("epoch_budget", 0)) - int(stop.get("epochs_run", 0))
            notes.append((xs[-1], vals[-1],
                          f"stopped early, {saved} epochs saved", SERIES[i], i + 2))

    # Annotations flip to right-alignment in the right third of the plot so
    # nothing runs off the canvas; vertical slots keep them from colliding.
    for x, y, text, color, slot in notes:
        right = x > 0.66 * x_max
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(-10 if right else 10, (12, -22, -22, -34)[slot % 4]),
                    ha="right" if right else "left",
                    color=color, fontsize=9.5)

    style_axes(ax, "Epoch", "Recording-level partial AUC (FPR ≤ 0.10)",
               "Tuning-set performance peaks early, then degrades")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.06, y=0.14)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", facecolor=SURFACE)
    plt.close(fig)


def fig_loss_curves(runs: dict[str, dict[str, Any]], out: Path) -> None:
    """
    @brief Train vs tuning loss per arm, as small multiples on a shared axis.

    @param runs Mapping of run name to loaded run.
    @param out Output directory.
    """
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.8), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, runs.items()):
        eps, tr = curve_points(r["bench"], "train_loss")
        _, va = curve_points(r["bench"], "val_loss")
        if not eps:
            continue
        dense = len(eps) > 14
        mk = "" if dense else "o"
        ax.plot(eps, tr, color=SERIES[0], linewidth=2, marker=mk, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label="train")
        ax.plot(eps, va, color=SERIES[1], linewidth=2,
                marker="" if dense else "s", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label="tuning set")
        ax.annotate("train", (eps[-1], tr[-1]), textcoords="offset points",
                    xytext=(-6, -14), color=SERIES[0], fontsize=9, ha="right")
        ax.annotate("tuning", (eps[-1], va[-1]), textcoords="offset points",
                    xytext=(-6, 8), color=SERIES[1], fontsize=9, ha="right")
        style_axes(ax, "Epoch", "Loss" if ax is axes[0] else "", name)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Training loss falls while tuning loss rises: overfitting",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "loss_curves.png", facecolor=SURFACE)
    plt.close(fig)


def fig_segment_vs_recording(runs: dict[str, dict[str, Any]], out: Path) -> None:
    """
    @brief Paired bars contrasting segment-level and recording-level AUC.

    @param runs Mapping of run name to loaded run.
    @param out Output directory.
    """
    names = list(runs)
    seg, rec = [], []
    for r in runs.values():
        bench = r["bench"]
        best = bench.get("best") or max(
            bench.get("checks") or bench["epochs"],
            key=lambda e: e["recording_metrics"].get(PAUC_KEY, -1),
        )
        seg.append(best["segment_metrics"]["auc_roc"])
        rec.append(best["recording_metrics"]["auc_roc"])

    fig, ax = new_fig(6.6, 4.0)
    x = np.arange(len(names))
    w = 0.34
    for off, vals, lab, c in (
        (-w / 2 - 0.01, seg, "per 10-s segment", SERIES[0]),
        (w / 2 + 0.01, rec, "per recording", SERIES[1]),
    ):
        ax.bar(x + off, vals, w, color=c, label=lab, zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", color=INK, fontsize=10)
    style_axes(ax, "", "AUC on the tuning set",
               "Aggregating segments into one score per recording")
    ax.set_xticks(x, names, color=INK_2)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "segment_vs_recording.png", facecolor=SURFACE)
    plt.close(fig)


def fig_aggregator_sweep(runs: dict[str, dict[str, Any]], out: Path) -> None:
    """
    @brief Partial AUC for every aggregation method, both arms.

    @param runs Mapping of run name to loaded run.
    @param out Output directory.
    """
    sweeps = {}
    for name, r in runs.items():
        p = r["pred"]
        probs = sigmoid(p["logits"].astype(np.float64))
        rows = sweep_aggregators(probs, p["rec_idx"], p["rec_label"].astype(int))
        sweeps[name] = {str(row["aggregator"]): row[PAUC_KEY] for row in rows}

    order = list(sweeps[list(sweeps)[0]])          # ranked by the first arm
    y = np.arange(len(order))[::-1]
    h = 0.36
    fig, ax = new_fig(7.4, 4.6)
    for i, (name, vals) in enumerate(sweeps.items()):
        off = (h / 2 + 0.01) * (1 if i == 0 else -1)
        ax.barh(y + off, [vals.get(a, np.nan) for a in order], h,
                color=SERIES[i], label=name, zorder=3)
    for i, a in enumerate(order):
        v = sweeps[list(sweeps)[0]][a]
        ax.text(v + 0.004, y[i] + h / 2 + 0.01, f"{v:.3f}", va="center",
                color=INK, fontsize=9)
    style_axes(ax, "Recording-level partial AUC (FPR ≤ 0.10)", "",
               "Mean beats focal aggregators: abnormality looks diffuse")
    ax.set_yticks(y, order, color=INK_2)
    ax.set_xlim(0.6, max(max(v.values()) for v in sweeps.values()) + 0.03)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "aggregator_sweep.png", facecolor=SURFACE)
    plt.close(fig)


def fig_roc(runs: dict[str, dict[str, Any]], out: Path, aggregator: str = "mean") -> None:
    """
    @brief Recording-level ROC curves with the triage operating point marked.

    @param runs Mapping of run name to loaded run.
    @param out Output directory.
    @param aggregator Segment-to-recording method used for the scores.
    """
    fig, ax = new_fig(5.6, 5.2)
    ax.axvspan(0, DEFAULT_MAX_FPR, color=GRID, alpha=0.45, zorder=1)
    ax.text(DEFAULT_MAX_FPR / 2, 0.03, "pAUC region", ha="center",
            color=INK_2, fontsize=8.5)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, linestyle=(0, (4, 4)), zorder=2)

    for i, (name, r) in enumerate(runs.items()):
        p = r["pred"]
        probs = sigmoid(p["logits"].astype(np.float64))
        labels = p["rec_label"].astype(int)
        scores = aggregate_to_recordings(probs, p["rec_idx"], aggregator,
                                         n_recordings=len(labels))
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, color=SERIES[i], linewidth=2, label=name, zorder=3)
        spec, _ = specificity_at_sensitivity(labels, scores, DEFAULT_SENS_FLOOR)
        if np.isfinite(spec):
            ax.scatter([1 - spec], [DEFAULT_SENS_FLOOR], s=90, color=SERIES[i],
                       edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.annotate(f"{name}: spec {spec:.2f} @ sens 0.95",
                        (1 - spec, DEFAULT_SENS_FLOOR), textcoords="offset points",
                        xytext=(8, -4 - 14 * i), color=SERIES[i], fontsize=9)
    style_axes(ax, "False positive rate (1 − specificity)", "Sensitivity",
               "Recording-level ROC on the tuning set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_recording.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    """@brief CLI entrypoint. @return 0 on success, non-zero on bad input."""
    p = argparse.ArgumentParser(description="Plot Phase 2 step-1 figures")
    p.add_argument("runs", nargs="+", metavar="NAME=RUNDIR")
    p.add_argument("--out", default="results/phase2/figs")
    p.add_argument("--aggregator", default="mean")
    args = p.parse_args()

    runs: dict[str, dict[str, Any]] = {}
    for spec in args.runs:
        if "=" not in spec:
            print(f"ERROR: expected NAME=RUNDIR, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        runs[name] = load_run(path)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_training_curves(runs, out)
    fig_loss_curves(runs, out)
    fig_segment_vs_recording(runs, out)
    fig_aggregator_sweep(runs, out)
    fig_roc(runs, out, args.aggregator)
    for f in sorted(out.glob("*.png")):
        print(f"wrote {f}")
    print("\nFigures contain aggregate metrics only; safe to share.")
    return 0


if __name__ == "__main__":
    sys.exit(main())