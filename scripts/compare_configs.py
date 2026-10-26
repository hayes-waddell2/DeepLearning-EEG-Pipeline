"""
@file compare_configs.py
@brief Paired, pooled-variance comparison of configurations across seeds or folds.

@details
Phase 3, stage 2b. `seed_validate.py` reports each configuration's mean and its
own standard deviation. With three repeats that standard deviation rests on two
degrees of freedom and is close to worthless on its own, and comparing two
configurations by their independent means throws away the single most useful
property of the design: **every configuration was run on the same seeds**.

This script fixes both.

@par Pooled variance
Every configuration is assumed to share a common repeat-to-repeat variance --
reasonable here, since they differ only in capacity and regularization, not in
the training procedure. Pooling five configurations x 3 repeats gives 10
degrees of freedom instead of 2, which tightens every interval without a single
additional GPU-minute.

@par Pairing
Running all configurations on seeds {42, 1337, 2024} is *common random
numbers*, a standard variance-reduction technique. If seed 42 is a hard draw,
it is hard for everyone, and that shared difficulty cancels in the
configuration-to-configuration difference. The paired standard deviation is
therefore usually smaller -- sometimes much smaller -- than the unpaired one,
and the script reports the ratio so the gain is visible rather than assumed.

@par Equivalence, not superiority
The question is not "is A better than B". The search already answered that
badly: the top configurations differ by less than the metric's own noise, and
the power calculation says separating a 0.0009 difference at a repeat-to-repeat
sd of 0.009 would take on the order of 800 repeats per configuration. The
useful question is **"which configurations can I not tell apart"**, answered by
two one-sided tests (TOST) against a pre-specified margin. A TOST pass is a
positive claim of equivalence; a non-significant t-test is not.

@par Reusable for cross-validation
`--block fold` treats fold index as the repeat unit, so the same analysis runs
over 10-fold CV results. Pairing matters more there, not less: fold difficulty
varies far more than seed difficulty, and removing it is the difference between
a usable comparison and a wash.

@par Usage:
@verbatim
python -m scripts.compare_configs --runs runs/seedval/runs.json
python -m scripts.compare_configs --runs runs/seedval/runs.json --margin 0.015
python -m scripts.compare_configs --runs runs/cv/runs.json --block fold \
    --metric pauc_tpr0.9
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover - scipy ships with scikit-learn
    sps = None

## @brief Default equivalence margin on the objective.
#
#  0.01 is just above the pooled repeat-to-repeat sd measured on this model and
#  roughly a fifth of the improvement HPO delivered over the Step 2 baseline. A
#  difference smaller than this would not change a clinical decision, so it is
#  the point below which two configurations are interchangeable.
DEFAULT_MARGIN = 0.01

## @brief Metric selected on.
PRIMARY = "score"

## @brief Metric the deployed system is judged by.
SECONDARY = "spec_at_sens0.95"

## @brief Score recorded for a failed run; excluded everywhere.
FAILED_SCORE = -1.0


def t_crit(df: int, p: float) -> float:
    """
    @brief Upper-tail critical value of Student's t.

    @param df Degrees of freedom.
    @param p Cumulative probability (e.g. 0.975 for a two-sided 95% interval).
    @return The critical value, or the normal approximation if scipy is absent.
    """
    if sps is not None:
        return float(sps.t.ppf(p, df))
    z = {0.95: 1.6449, 0.975: 1.9600}.get(p, 1.9600)
    return z * (1.0 + 2.0 / max(df, 1))  # crude widening; scipy is preferred


def t_sf(t: float, df: int) -> float:
    """
    @brief Upper-tail survival function of Student's t.

    @param t Test statistic.
    @param df Degrees of freedom.
    @return P(T > t), or a normal approximation if scipy is absent.
    """
    if sps is not None:
        return float(sps.t.sf(t, df))
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def load(path: Path, block: str, metric: str) -> dict[str, dict[Any, float]]:
    """
    @brief Read runs.json into {config: {block_value: metric}}.

    @param path The runs.json written by the runner.
    @param block Field identifying the repeat unit: `seed` or `fold`.
    @param metric Field to analyze.
    @return Nested mapping, failed runs omitted.
    @throws ValueError If no usable runs are present.
    """
    runs = json.loads(path.read_text())
    out: dict[str, dict[Any, float]] = {}
    for r in runs:
        if r.get("score", FAILED_SCORE) <= FAILED_SCORE:
            continue
        v = r.get(metric)
        if v is None:
            continue
        out.setdefault(r["name"], {})[r[block]] = float(v)
    if not out:
        raise ValueError(f"no usable runs with metric {metric!r} in {path}")
    return out


def pooled_sd(data: dict[str, dict[Any, float]]) -> tuple[float, int]:
    """
    @brief Pool the within-configuration variance across configurations.

    @details
    Each configuration contributes `n_i - 1` degrees of freedom. With five
    configurations at three repeats this is 10 df rather than the 2 df any
    single configuration offers, which is the whole point.

    @param data Output of `load`.
    @return Tuple `(pooled_sd, degrees_of_freedom)`.
    """
    num = 0.0
    df = 0
    for vals in data.values():
        v = list(vals.values())
        if len(v) > 1:
            num += (len(v) - 1) * st.variance(v)
            df += len(v) - 1
    return (math.sqrt(num / df) if df else 0.0), df


def block_effect(data: dict[str, dict[Any, float]], block: str) -> None:
    """
    @brief Report how much of the spread is shared across configurations.

    @details
    If one seed is hard for every configuration, that difficulty is a nuisance
    factor the paired analysis removes. The spread of the per-block means is a
    direct read on how much there is to remove.

    @param data Output of `load`.
    @param block Name of the repeat unit, for labelling.
    """
    blocks = sorted({b for v in data.values() for b in v})
    rows = []
    for b in blocks:
        vals = [v[b] for v in data.values() if b in v]
        if vals:
            rows.append((b, st.fmean(vals), len(vals)))
    if len(rows) < 2:
        return
    print(f"\n{block.upper()} EFFECT -- mean across configurations")
    for b, m, n in rows:
        print(f"  {block} {str(b):>6}  mean {m:.4f}  (n={n})")
    spread = max(r[1] for r in rows) - min(r[1] for r in rows)
    print(f"  spread across {block}s: {spread:.4f}")
    if spread > 0.005:
        print(
            f"  -> {block.capitalize()}s differ enough to matter. Pairing removes "
            "this, which is why the paired comparison below is the right one."
        )


def describe(data: dict[str, dict[Any, float]], label: str) -> list[tuple]:
    """
    @brief Per-configuration summary, ranked.

    @param data Output of `load`.
    @param label Metric name for the header.
    @return Rows of `(name, n, mean, sd, min, max)` sorted by mean descending.
    """
    rows = []
    for name, vals in data.items():
        v = list(vals.values())
        rows.append((name, len(v), st.fmean(v), st.stdev(v) if len(v) > 1 else 0.0, min(v), max(v)))
    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"\nPER-CONFIGURATION -- {label}")
    print(f"  {'config':14} {'n':>2} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8}")
    for r in rows:
        print(f"  {r[0]:14} {r[1]:>2} {r[2]:>8.4f} {r[3]:>8.4f} {r[4]:>8.4f} {r[5]:>8.4f}")
    return rows


def pairwise(
    data: dict[str, dict[Any, float]],
    order: list[str],
    margin: float,
    psd: float,
    pdf: int,
) -> None:
    """
    @brief Paired differences, confidence intervals and TOST equivalence.

    @details
    For each pair, the per-block differences are formed and tested against
    zero (superiority) and against +/- `margin` (equivalence). Both answers are
    reported because they are different claims: failing to show a difference is
    not the same as showing there is none.

    The unpaired interval, built from the pooled sd, is printed alongside so the
    variance reduction pairing bought is visible.

    @param data Output of `load`.
    @param order Configuration names, best first.
    @param margin Equivalence margin.
    @param psd Pooled within-configuration sd.
    @param pdf Degrees of freedom behind `psd`.
    """
    print(f"\nPAIRWISE -- paired differences, equivalence margin +/-{margin:g}")
    print(
        f"  {'A':13} {'B':13} {'diff':>8} {'sd(d)':>7} {'95% CI':>18} "
        f"{'unpaired CI':>18} {'verdict':>12}"
    )
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            shared = sorted(set(data[a]) & set(data[b]))
            if len(shared) < 2:
                continue
            d = [data[a][k] - data[b][k] for k in shared]
            n = len(d)
            md, sd = st.fmean(d), st.stdev(d)
            se = sd / math.sqrt(n)
            tc = t_crit(n - 1, 0.975)
            lo, hi = md - tc * se, md + tc * se

            # Same comparison without pairing, using the pooled sd.
            se_un = psd * math.sqrt(2.0 / n)
            tc_un = t_crit(pdf, 0.975)
            lo_u, hi_u = md - tc_un * se_un, md + tc_un * se_un

            # TOST: equivalent when both one-sided tests reject at alpha.
            if se > 0:
                p_lo = t_sf((md + margin) / se, n - 1)  # H0: diff <= -margin
                p_hi = t_sf((margin - md) / se, n - 1)  # H0: diff >= +margin
                equiv = max(p_lo, p_hi) < 0.05
            else:
                equiv = abs(md) < margin
            differs = lo > 0 or hi < 0

            verdict = "DIFFERENT" if differs else "equivalent" if equiv else "inconclusive"
            print(
                f"  {a:13} {b:13} {md:>+8.4f} {sd:>7.4f} "
                f"[{lo:>+7.4f},{hi:>+7.4f}] [{lo_u:>+7.4f},{hi_u:>+7.4f}] "
                f"{verdict:>12}"
            )

    print(
        "\n  DIFFERENT    = the 95% CI on the paired difference excludes zero\n"
        "  equivalent   = TOST rejects both one-sided nulls; the pair is\n"
        "                 interchangeable within the margin (a positive claim)\n"
        "  inconclusive = neither; more repeats would help THIS pair"
    )


def variance_reduction(data: dict[str, dict[Any, float]], order: list[str], psd: float) -> None:
    """
    @brief Report how much pairing shrank the comparison noise.

    @param data Output of `load`.
    @param order Configuration names.
    @param psd Pooled within-configuration sd.
    """
    ratios = []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            shared = sorted(set(data[a]) & set(data[b]))
            if len(shared) < 2:
                continue
            d = [data[a][k] - data[b][k] for k in shared]
            sd_paired = st.stdev(d)
            sd_unpaired = psd * math.sqrt(2.0)
            if sd_paired > 0:
                ratios.append(sd_unpaired / sd_paired)
    if ratios:
        print(
            f"\nPairing shrank the comparison sd by a median factor of "
            f"{st.median(ratios):.2f}x "
            f"(range {min(ratios):.2f}-{max(ratios):.2f})."
        )


def power_note(psd: float, order: list[str], data: dict[str, dict[Any, float]]) -> None:
    """
    @brief State how many repeats the closest comparison would actually need.

    @param psd Pooled within-configuration sd.
    @param order Configuration names, best first.
    @param data Output of `load`.
    """
    if len(order) < 2 or psd <= 0:
        return
    a, b = order[0], order[1]
    shared = sorted(set(data[a]) & set(data[b]))
    if not shared:
        return
    delta = abs(st.fmean([data[a][k] - data[b][k] for k in shared]))
    if delta <= 0:
        return
    # n per group for 80% power, two-sided alpha=0.05, unpaired.
    n = 16.0 * (psd / delta) ** 2
    print(
        f"\nTo resolve the {a} vs {b} gap of {delta:.4f} at the pooled sd of "
        f"{psd:.4f}\nwould take about {n:,.0f} repeats per configuration "
        "(80% power, alpha 0.05)."
    )
    if n > 50:
        print(
            "That is not a budget problem -- it says the difference is not "
            "there.\nSelect on a pre-specified secondary criterion and report "
            "the equivalence."
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace of options.
    """
    p = argparse.ArgumentParser(description="Paired, pooled-variance comparison of configurations.")
    p.add_argument("--runs", required=True, help="runs.json from the runner")
    p.add_argument(
        "--block",
        default="seed",
        choices=["seed", "fold"],
        help="Repeat unit: seed for seed validation, fold for CV",
    )
    p.add_argument("--metric", default=PRIMARY, help="Primary metric field")
    p.add_argument("--secondary", default=SECONDARY, help="Secondary metric field")
    p.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Equivalence margin; state it before looking at results",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args(argv)
    path = Path(a.runs)
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    if sps is None:
        print("warning: scipy not found; p-values are approximate", file=sys.stderr)

    data = load(path, a.block, a.metric)
    print("=" * 94)
    print(f"{path}   block={a.block}   metric={a.metric}   margin=+/-{a.margin:g}")
    print("=" * 94)

    rows = describe(data, a.metric)
    order = [r[0] for r in rows]
    psd, pdf = pooled_sd(data)
    own = rows[0][3]
    print(
        f"\nPooled within-configuration sd: {psd:.4f} on {pdf} df "
        f"(vs {own:.4f} on {max(rows[0][1] - 1, 0)} df for {order[0]} alone)"
    )

    block_effect(data, a.block)
    pairwise(data, order, a.margin, psd, pdf)
    variance_reduction(data, order, psd)
    power_note(psd, order, data)

    try:
        sec = load(path, a.block, a.secondary)
    except ValueError:
        print(f"\n(no {a.secondary} recorded; skipping secondary analysis)")
        return 0

    print("\n" + "=" * 94)
    print(f"SECONDARY METRIC -- {a.secondary} (the deployed operating point)")
    print("=" * 94)
    srows = describe(sec, a.secondary)
    spsd, spdf = pooled_sd(sec)
    print(f"\nPooled sd: {spsd:.4f} on {spdf} df")
    pairwise(sec, [r[0] for r in srows], a.margin, spsd, spdf)
    print(
        "\nIf the primary metric cannot separate the configurations and this one "
        "can,\nthis is the metric to select on -- it is what the deliverable "
        "promises."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
