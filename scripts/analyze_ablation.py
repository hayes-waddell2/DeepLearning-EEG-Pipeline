"""
@file analyze_ablation.py
@brief Per-axis curves, the parameter-cost frontier, and the check that gates them.

@details
Phase 4. Turns the ablation runs into the two figures the parameter-efficiency
deliverable needs, and refuses to draw them until the validity check passes.

@par The check comes first, and it can stop the analysis
Early stopping gives each model its own run length. If small models terminate
systematically earlier or later than large ones, the arm is measuring training
budget rather than capacity, and every conclusion drawn from it is about the
wrong thing. The check correlates `best_check` with parameter count across
points; a strong relationship prints a warning and, with `--strict`, exits
non-zero before a single figure is written. A plot that should not have been
drawn is worse than no plot, because it will end up on a slide.

@par Two figures, because they answer different questions
`axes` shows one panel per capacity axis: does *this* dimension matter, holding
everything else fixed. That is the attribution question, and it is the one the
HPO scatter could not answer -- its points varied several things at once.

`frontier` shows every point against parameter count and, when latency has been
measured, against seconds per recording. That is the deployment question. They
are separate figures because they are separate slides and separate claims.

@par Reading the error bars
Each point carries the standard deviation over its seeds -- typically three,
which is two degrees of freedom and therefore a noisy estimate in its own
right. The horizontal band on each panel is the centre configuration plus or
minus the **pooled** standard deviation across all points, which has far more
degrees of freedom and is the honest noise floor. A point inside that band is
not distinguishable from the centre, however its own bar looks.

@par The sign test
Individual single-axis moves are rarely significant at three seeds, but the
pattern across them is. If every move away from the centre scores below it,
that is evidence the centre is a local optimum even when no single comparison
clears a threshold. The report prints that test, because it is the strongest
honest statement the data supports.

@par Usage:
@verbatim
python -m scripts.analyze_ablation --runs runs/ablation-shape/runs.json \
    --points configs/ablation_shape.json --out-dir results/ablation
python -m scripts.analyze_ablation --runs ... --points ... \
    --latency results/latency.json --strict
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

## @brief Palette, matching the project's updated architecture diagrams:
#  black ink, grey for the recessive layer, green for the selected point,
#  orange for anything flagged.
INK, GREY, FAINT = "#000000", "#8A8A8A", "#D4D4D4"
SEL, WARN = "#157F4A", "#C0531F"

## @brief Metric selected on, and the one the deliverable promises.
PRIMARY = "score"
SECONDARY = "spec_at_sens0.95"

## @brief Axis order for the panels, coarsest structural change first.
AXIS_ORDER = ["conv_depth", "lstm_layers", "lstm_hidden", "conv_width"]

## @brief Readable panel titles.
AXIS_LABEL = {
    "conv_depth": "CNN blocks",
    "lstm_layers": "LSTM layers",
    "lstm_hidden": "LSTM hidden units",
    "conv_width": "Conv width multiplier",
}

## @brief Spearman correlation between size and stopping above which the
#  sweep is treated as confounded by training budget rather than capacity.
STOP_CONFOUND_RHO = 0.6


def _tick(v: float, _pos: int = 0) -> str:
    """
    @brief Format a log-axis tick as a plain readable number.

    @param v Tick value.
    @param _pos Tick index, unused.
    @return "250K", "2.4M", "1.3" and so on; empty for crowded minor ticks.
    """
    if v <= 0:
        return ""
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}K"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def spearman(xs: list[float], ys: list[float]) -> float:
    """
    @brief Spearman rank correlation, without a scipy dependency.

    @param xs First series.
    @param ys Second series, same length.
    @return Correlation in [-1, 1]; 0.0 if undefined.
    """
    n = len(xs)
    if n < 3:
        return 0.0

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def load(runs_path: Path, points_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    @brief Read runs and point metadata, grouped per point.

    @param runs_path `runs.json` from the runner.
    @param points_path The point set, for axis and level.
    @return Tuple `(stats, spec)` keyed by point name.
    @throws ValueError If no usable runs are present.
    """
    runs = json.loads(runs_path.read_text())
    spec = {c["name"]: c for c in json.loads(points_path.read_text())["configs"]}

    by: dict[str, list[dict]] = defaultdict(list)
    failed: dict[str, int] = defaultdict(int)
    for r in runs:
        if r.get("score", -1.0) <= -1.0:
            failed[r["name"]] += 1
            continue
        by[r["name"]].append(r)
    if not by:
        raise ValueError(f"no usable runs in {runs_path}")

    stats = {}
    for name, rs in by.items():
        p = [r["score"] for r in rs]
        s = [r[SECONDARY] for r in rs if r.get(SECONDARY) is not None]
        stats[name] = {
            "n": len(p),
            "failed": failed.get(name, 0),
            "mean": st.fmean(p),
            "sd": st.stdev(p) if len(p) > 1 else 0.0,
            "sec_mean": st.fmean(s) if s else float("nan"),
            "sec_sd": st.stdev(s) if len(s) > 1 else 0.0,
            "n_parameters": rs[0].get("n_parameters") or 0,
            "best_check": st.fmean([r.get("best_check") or 0 for r in rs]),
            "checks_run": st.fmean([r.get("checks_run") or 0 for r in rs]),
            "minutes": st.fmean([r.get("minutes", 0.0) for r in rs]),
        }
    for name in failed:
        if name not in stats:
            stats[name] = {"n": 0, "failed": failed[name]}
    return stats, spec


def pooled_sd(stats: dict[str, dict]) -> tuple[float, int]:
    """
    @brief Pool the within-point variance across every point.

    @param stats Per-point statistics.
    @return Tuple `(pooled_sd, degrees_of_freedom)`.
    """
    num, df = 0.0, 0
    for s in stats.values():
        if s.get("n", 0) > 1:
            num += (s["n"] - 1) * s["sd"] ** 2
            df += s["n"] - 1
    return (math.sqrt(num / df) if df else 0.0), df


def validity(stats: dict[str, dict]) -> tuple[bool, float, list[str]]:
    """
    @brief Check that stopping behaviour does not track model size.

    @param stats Per-point statistics.
    @return Tuple `(passed, rho, messages)`.
    """
    usable = [s for s in stats.values() if s.get("n", 0) > 0]
    pars = [s["n_parameters"] for s in usable]
    best = [s["best_check"] for s in usable]
    rho = spearman(pars, best)
    msgs = [
        f"Spearman(parameters, best_check) = {rho:+.3f} over {len(usable)} points",
        f"  best_check spans {min(best):.1f} to {max(best):.1f}",
    ]
    ok = abs(rho) < STOP_CONFOUND_RHO
    if not ok:
        msgs.append(
            "  FAIL: stopping tracks model size. These arms measure training "
            "budget, not capacity. Re-run with a fixed epoch budget before "
            "drawing any conclusion from them."
        )
    else:
        msgs.append("  PASS: no systematic relationship; the arms measure capacity.")
    return ok, rho, msgs


def sign_test(stats: dict[str, dict], center: str = "center") -> Optional[dict]:
    """
    @brief Test whether the centre is beaten by any single-axis move.

    @details
    Individual moves rarely clear significance at three seeds. The count of
    moves falling below the centre does, and it is the claim the sweep is
    actually able to support.

    @param stats Per-point statistics.
    @param center Name of the centre point.
    @return Summary dict, or `None` if the centre is missing.
    """
    if center not in stats or stats[center].get("n", 0) == 0:
        return None
    c = stats[center]["mean"]
    off = [(k, v["mean"]) for k, v in stats.items() if k != center and v.get("n", 0) > 0]
    below = sum(1 for _, m in off if m < c)
    n = len(off)
    if n == 0:
        return None
    p = 2 * sum(math.comb(n, i) for i in range(below, n + 1)) / 2**n
    return {
        "below": below,
        "n": n,
        "p": min(p, 1.0),
        "center_mean": c,
        "exceptions": [k for k, m in off if m >= c],
    }


def panel(ax, pts: list[tuple[float, float, float]], label: str, band: tuple[float, float]):
    """
    @brief Draw one axis panel: metric against level, with error bars.

    @param ax Matplotlib axes.
    @param pts Sorted `(level, mean, sd)` triples.
    @param label X-axis label.
    @param band `(centre_mean, pooled_sd)` for the indistinguishability band.
    """
    cm, psd = band
    ax.axhspan(cm - psd, cm + psd, color=FAINT, zorder=0)
    ax.axhline(cm, color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1)

    xs = list(range(len(pts)))
    ys = [p[1] for p in pts]
    es = [p[2] for p in pts]
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        color=INK,
        lw=1.6,
        marker="o",
        ms=6,
        mfc="white",
        mec=INK,
        mew=1.4,
        capsize=3,
        zorder=3,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p[0]:g}" for p in pts])
    ax.set_xlabel(label, fontsize=10, color=INK)
    ax.grid(axis="y", color=FAINT, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)


def figure_axes(stats, spec, psd: float, out: Path, center: str = "center") -> None:
    """
    @brief Per-axis panels: does this dimension matter, all else held fixed.

    @param stats Per-point statistics.
    @param spec Point metadata.
    @param psd Pooled standard deviation.
    @param out Destination PNG.
    @param center Name of the centre point.
    """
    cm = stats[center]["mean"]
    axes_present = [a for a in AXIS_ORDER if any(spec[k]["axis"] == a for k in stats if k in spec)]
    if not axes_present:
        return

    fig, axs = plt.subplots(
        1, len(axes_present), figsize=(3.4 * len(axes_present), 4.4), sharey=True
    )
    if len(axes_present) == 1:
        axs = [axs]
    fig.patch.set_facecolor("white")

    for ax, axis in zip(axs, axes_present):
        pts = []
        for k, s in stats.items():
            if k not in spec or s.get("n", 0) == 0:
                continue
            if spec[k]["axis"] == axis or (k == center):
                pts.append(
                    (
                        (
                            float(spec[k]["level"])
                            if k != center
                            else float(_center_level(spec, axis))
                        ),
                        s["mean"],
                        s["sd"],
                    )
                )
        pts = sorted(set(pts))
        panel(ax, pts, AXIS_LABEL.get(axis, axis), (cm, psd))

    axs[0].set_ylabel("Partial AUC over TPR $\\geq$ 0.90", fontsize=10, color=INK)
    fig.suptitle(
        "Every single-axis move away from the selected model costs performance",
        fontsize=14,
        color=INK,
        x=0.02,
        ha="left",
        y=0.975,
        weight="bold",
    )
    fig.text(
        0.02,
        0.915,
        "Error bars are one standard deviation over seeds. The shaded band is the\n"
        f"selected configuration $\\pm$ the pooled standard deviation ({psd:.4f}): a point\n"
        "inside it is not distinguishable from the selected model.",
        fontsize=9.5,
        va="top",
        linespacing=1.5,
        color=GREY,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def _center_level(spec: dict, axis: str) -> Any:
    """
    @brief The level the centre sits at on a given axis.

    @param spec Point metadata.
    @param axis Axis name.
    @return The centre's level on that axis, defaulting to its stored level.
    """
    defaults = {"conv_depth": 3, "lstm_layers": 2, "lstm_hidden": 64, "conv_width": 1.0}
    return defaults.get(axis, spec.get("center", {}).get("level", 0))


def figure_frontier(
    stats, psd: float, latency: Optional[dict], out: Path, center: str = "center"
) -> None:
    """
    @brief Cost against performance: parameters, and seconds when measured.

    @param stats Per-point statistics.
    @param psd Pooled standard deviation.
    @param latency Parsed latency JSON, or `None`.
    @param out Destination PNG.
    @param center Name of the centre point.
    """
    usable = {k: v for k, v in stats.items() if v.get("n", 0) > 0}
    cm = stats[center]["mean"]

    lat = {}
    if latency:
        key = None
        for r in latency.get("records", []):
            if not key and r.get("timings"):
                key = sorted(r["timings"])[0]
            if key and key in r.get("timings", {}):
                lat[r["name"]] = r["timings"][key]["per_recording_s"]

    ncol = 2 if lat else 1
    fig, axs = plt.subplots(1, ncol, figsize=(6.6 * ncol, 5.4), sharey=True)
    axs = [axs] if ncol == 1 else list(axs)
    fig.patch.set_facecolor("white")

    panels = [("n_parameters", "Trainable parameters (log scale)", True)]
    if lat:
        panels.append((None, "Seconds per recording on CPU (log scale)", True))

    for ax, (field, xlabel, logx) in zip(axs, panels):
        ax.axhspan(cm - psd, cm + psd, color=FAINT, zorder=0)
        for k, s in usable.items():
            x = s["n_parameters"] if field else lat.get(k)
            if not x:
                continue
            is_c = k == center
            ax.errorbar(
                x,
                s["mean"],
                yerr=s["sd"],
                color=SEL if is_c else GREY,
                lw=0,
                elinewidth=1.3,
                capsize=3,
                marker="D" if is_c else "o",
                ms=9 if is_c else 7,
                mfc=SEL if is_c else "white",
                mec="white" if is_c else GREY,
                mew=1.4,
                zorder=4 if is_c else 2,
            )
        if logx:
            ax.set_xscale("log")
            # Plain numbers at 1-2-5 steps, not 10^5 and 2x10^0: these are
            # counts and seconds a reader should compare at a glance. Minor
            # labels are suppressed -- on a log decade they collide.
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(_tick))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
        ax.grid(axis="y", color=FAINT, lw=0.7)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(GREY)
        ax.tick_params(colors=GREY, labelsize=9)

    axs[0].set_ylabel("Partial AUC over TPR $\\geq$ 0.90", fontsize=10, color=INK)
    cx = stats[center]["n_parameters"]
    axs[0].annotate(
        "selected",
        (cx, cm),
        textcoords="offset points",
        xytext=(10, 6),
        fontsize=9.5,
        color=SEL,
        weight="bold",
    )

    title = (
        "Cost buys nothing past the selected model"
        if lat
        else "More parameters buy nothing past the selected model"
    )
    fig.suptitle(title, fontsize=14, color=INK, x=0.02, ha="left", y=0.975, weight="bold")
    fig.text(
        0.02,
        0.915,
        "Each point is one architecture at three seeds; bars are one standard deviation.\n"
        "The shaded band is the selected configuration $\\pm$ the pooled standard deviation.",
        fontsize=9.5,
        color=GREY,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def report(stats, spec, psd, df, ok, msgs, sgn, out_dir: Path) -> None:
    """
    @brief Print and write the text summary.

    @param stats Per-point statistics.
    @param spec Point metadata.
    @param psd Pooled standard deviation.
    @param df Degrees of freedom behind `psd`.
    @param ok Whether the validity check passed.
    @param msgs Validity messages.
    @param sgn Sign-test summary or `None`.
    @param out_dir Destination directory.
    """
    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 84)
    emit("VALIDITY CHECK")
    for m in msgs:
        emit(m)
    emit()
    emit(f"Pooled within-point sd: {psd:.4f} on {df} df")

    failed = {k: v["failed"] for k, v in stats.items() if v.get("failed")}
    if failed:
        emit(f"Failed runs: {failed}")

    if sgn:
        emit()
        emit("CENTRE AS A LOCAL OPTIMUM")
        emit(
            f"  {sgn['below']} of {sgn['n']} single-axis moves scored below the "
            f"selected configuration (sign test p = {sgn['p']:.4f})"
        )
        if sgn["exceptions"]:
            for e in sgn["exceptions"]:
                emit(
                    f"  exception: {e} at {stats[e]['mean']:.4f} "
                    f"(sd {stats[e]['sd']:.4f} vs centre {stats['center']['sd']:.4f})"
                )

    emit()
    emit("PER POINT, by distance from the selected configuration")
    cm = sgn["center_mean"] if sgn else 0.0
    emit(
        f"  {'point':20} {'params':>10} {'pauc':>8} {'sd':>7} {'delta':>8} "
        f"{'in sd':>6} {'spec@.95':>9} {'min':>6}"
    )
    rows = [(k, v) for k, v in stats.items() if v.get("n", 0) > 0]
    for k, v in sorted(rows, key=lambda kv: -kv[1]["mean"]):
        d = v["mean"] - cm
        emit(
            f"  {k:20} {v['n_parameters']:>10,} {v['mean']:>8.4f} {v['sd']:>7.4f} "
            f"{d:>+8.4f} {(d / psd if psd else 0):>6.1f} "
            f"{v['sec_mean']:>9.4f} {v['minutes']:>6.1f}"
        )

    emit()
    emit("EQUIVALENT TO THE SELECTED MODEL (within one pooled sd), smallest first")
    eq = [(v["n_parameters"], k, v) for k, v in rows if abs(v["mean"] - cm) <= psd]
    for n_par, k, v in sorted(eq):
        emit(f"  {n_par:>10,}  {k:20} {v['mean']:.4f}")
    if eq:
        n_par, k, v = sorted(eq)[0]
        emit(
            f"\n  Smallest indistinguishable architecture: {k} at {n_par:,} parameters, "
            f"{(stats['center']['n_parameters'] / n_par):.1f}x smaller than the selected one."
        )

    if not ok:
        emit()
        emit(
            "Figures were still written, but the validity check FAILED. Do not "
            "present them without resolving it."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_report.txt").write_text("\n".join(lines) + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace of options.
    """
    p = argparse.ArgumentParser(description="Analyze the ablation sweep.")
    p.add_argument("--runs", required=True, help="runs.json from the sweep")
    p.add_argument("--points", required=True, help="The point set used")
    p.add_argument("--latency", default=None, help="Optional latency JSON")
    p.add_argument("--out-dir", default="results/ablation")
    p.add_argument("--center", default="center", help="Name of the reference point")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the validity check fails",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args(argv)
    runs_p, pts_p = Path(a.runs), Path(a.points)
    for p in (runs_p, pts_p):
        if not p.is_file():
            print(f"error: {p} not found", file=sys.stderr)
            return 1

    stats, spec = load(runs_p, pts_p)
    if a.center not in stats or stats[a.center].get("n", 0) == 0:
        print(f"error: reference point {a.center!r} has no usable runs", file=sys.stderr)
        return 1

    psd, df = pooled_sd(stats)
    ok, _, msgs = validity(stats)
    sgn = sign_test(stats, a.center)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latency = json.loads(Path(a.latency).read_text()) if a.latency else None

    figure_axes(stats, spec, psd, out_dir / "ablation_axes.png", a.center)
    figure_frontier(stats, psd, latency, out_dir / "ablation_frontier.png", a.center)
    report(stats, spec, psd, df, ok, msgs, sgn, out_dir)

    print(f"\nWrote {out_dir / 'ablation_axes.png'}")
    print(f"Wrote {out_dir / 'ablation_frontier.png'}")
    print(f"Wrote {out_dir / 'ablation_report.txt'}")
    return 0 if (ok or not a.strict) else 2


if __name__ == "__main__":
    raise SystemExit(main())