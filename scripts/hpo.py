"""
@file hpo.py
@brief Optuna hyperparameter search over the tuning pool, with ASHA pruning.

@details
Phase 3. Trains on the training pool (E) and scores on the tuning pool (T),
which no reported result is ever drawn from. The search space and the
reasoning behind each range live in `eeg_cnn_lstm.utils.search_space`.

@par Objective
`recording_pauc_sens` by default: the standardized partial AUC over the
TPR >= 0.90 region. The earlier objective measured FPR <= 0.10, the
few-false-alarms corner, but the triage operating point ("catch 95% of
abnormal recordings") sits near FPR 0.46 -- the search would have been tuning
a part of the ROC the deployed system never visits. The high-sensitivity
region is where it does operate, and an area statistic is far less noisy than
the single-threshold `spec_at_sens0.95`, which matters when check-to-check
variation is already ~0.02.

@par Pruning
Each trial reports its score at every validation check, several times per
epoch, and the Hyperband/ASHA pruner kills trials that are behind their
cohort. This is what makes ~70 trials fit in an overnight job: most trials
die within an epoch or two, and the budget concentrates on survivors.

Pruning is deliberately conservative here. The tuning metric varies ~0.02
between checks, so an aggressive pruner would kill good trials on noise -- the
same failure that removed the learning-rate scheduler. `--prune-warmup-checks`
holds pruning off until a trial has enough checks for its mean to mean
something.

@par Surviving a 10-hour job
Three safeguards, because this runs unattended overnight:
  - **SQLite storage.** Results persist per trial. If the job dies at hour 9,
    rerunning with the same `--study-name` and `--storage` resumes rather
    than restarting.
  - **Wall-clock budget.** `--timeout-hours` stops the study cleanly with
    results written, rather than being killed mid-trial by SLURM. Set it
    below the job's time limit.
  - **Failed trials do not kill the study.** A trial that raises is recorded
    as FAILED and the search continues, except for errors that would affect
    every trial (missing data, bad config), which stop immediately.

@par Usage:
@verbatim
# verify first -- see scripts/hpo_verify.py
python -m scripts.hpo_verify --mode offline

# the real search
python -m scripts.hpo --config configs/step2_earlystop.yaml \
    --n-trials 70 --timeout-hours 9.5 \
    --storage sqlite:////shared/rc/eeg-cnn-lstm/runs/hpo/study.db

# resume after an interruption (same study name and storage)
python -m scripts.hpo --config configs/step2_earlystop.yaml --n-trials 70 \
    --storage sqlite:////shared/rc/eeg-cnn-lstm/runs/hpo/study.db

# report only, no training
python -m scripts.hpo --storage sqlite:///.../study.db --report-only
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from loguru import logger
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from eeg_cnn_lstm.utils.search_space import (
    GROUPS,
    SEARCH_SPACE,
    STARTUP_TRIALS,
    describe,
    suggest_all,
)

## @brief Selection metric the search optimizes; see the module docstring.
DEFAULT_OBJECTIVE = "recording_pauc_sens"

## @brief Score returned for a trial that fails in a way worth continuing past.
FAILED_SCORE = -1.0

## @brief Exceptions that mean every trial will fail, so stop the study.
FATAL = (FileNotFoundError, ImportError, KeyError)


def make_study(
    storage: Optional[str],
    study_name: str,
    seed: int,
    n_startup: int,
    prune: str,
    prune_warmup_checks: int,
) -> optuna.Study:
    """
    @brief Create or resume the study, with sampler and pruner configured.

    @details
    `load_if_exists` is what makes an interrupted overnight job resumable:
    the same name and storage pick up the completed trials and keep going.

    @param storage SQLAlchemy URL, e.g. `sqlite:///path/study.db`; `None`
           keeps the study in memory (verification only -- nothing survives).
    @param study_name Study identifier; the resume key.
    @param seed Sampler seed, for a reproducible search.
    @param n_startup Random trials before TPE's model takes over.
    @param prune `"hyperband"`, `"median"` or `"none"`.
    @param prune_warmup_checks Checks a trial gets before it may be pruned.
    @return The study.
    @throws ValueError If `prune` is unknown.
    """
    if prune == "hyperband":
        pruner: optuna.pruners.BasePruner = HyperbandPruner(
            min_resource=prune_warmup_checks,
            reduction_factor=3,
        )
    elif prune == "median":
        pruner = MedianPruner(
            n_startup_trials=n_startup,
            n_warmup_steps=prune_warmup_checks,
        )
    elif prune == "none":
        pruner = NopPruner()
    else:
        raise ValueError(f"--prune must be hyperband, median or none; got {prune!r}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,   # parameters interact; model them jointly
            group=True,
        ),
        pruner=pruner,
        load_if_exists=True,
    )
    return study


def enqueue_startup(study: optuna.Study) -> int:
    """
    @brief Enqueue the bracketing startup points, skipping any already run.

    @details
    On a resumed study these have usually been run already; enqueuing them
    again would waste GPU-hours re-evaluating known points. Matching is on
    the exact parameter dict.

    @param study The study.
    @return How many were enqueued.
    """
    seen = [t.params for t in study.get_trials(deepcopy=False)]
    added = 0
    for point in STARTUP_TRIALS:
        if point in seen:
            continue
        study.enqueue_trial(point, skip_if_exists=True)
        added += 1
    return added


def make_objective(
    config_path: str,
    out_root: Path,
    objective_metric: str,
    train_fn: Callable[..., dict[str, Any]],
    base_overrides: Optional[dict[str, Any]] = None,
    max_steps: int = 0,
    num_epochs: Optional[int] = None,
    keep_artifacts: bool = False,
) -> Callable[[optuna.Trial], float]:
    """
    @brief Build the Optuna objective closure.

    @details
    `train_fn` is injected rather than imported directly so the verification
    script can substitute a cheap stand-in and exercise this exact code path
    -- sampling, pruning, storage, failure handling -- without a GPU.

    Each trial reports its score at every validation check and raises
    `TrialPruned` when the pruner says so; `train()` calls the hook after the
    checkpoint and run record are written, so a pruned trial still leaves a
    readable history behind.

    @param config_path Base YAML config.
    @param out_root Directory for per-trial run directories.
    @param objective_metric `train.select_metric` value to optimize.
    @param train_fn Callable with `train()`'s signature.
    @param base_overrides Config overrides applied to every trial, before the
           sampled ones.
    @param max_steps Cap on training steps per epoch (0 = uncapped).
    @param num_epochs Override the epoch budget.
    @param keep_artifacts Keep checkpoints and predictions for every trial.
           Off by default: 70 trials of weights is tens of gigabytes, and only
           the top few get re-run anyway.
    @return The objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        params, overrides = suggest_all(trial)
        run_dir = out_root / f"trial_{trial.number:04d}"

        merged: dict[str, Any] = {}
        for src in (base_overrides or {}, overrides):
            for section, values in src.items():
                if isinstance(values, dict):
                    merged.setdefault(section, {}).update(values)
                else:
                    merged[section] = values
        merged.setdefault("train", {})
        merged["train"]["select_metric"] = objective_metric
        merged["train"]["output_dir"] = str(run_dir)

        logger.info(
            f"--- trial {trial.number}: "
            + "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params.items()
            )
        )

        def on_check(check_no: int, score: float) -> bool:
            """Report to the pruner; raise TrialPruned when it says stop."""
            if not math.isfinite(score):
                return False
            trial.report(score, step=check_no)
            if trial.should_prune():
                logger.info(f"    pruned at check {check_no} (score {score:.4f})")
                raise optuna.TrialPruned()
            return False

        t0 = time.perf_counter()
        try:
            bench = train_fn(
                config_path,
                output_dir=str(run_dir),
                num_epochs=num_epochs,
                max_steps=max_steps,
                on_check=on_check,
                config_overrides=merged,
            )
        except optuna.TrialPruned:
            raise
        except FATAL as exc:
            logger.error(f"trial {trial.number} hit a fatal error: {exc!r}")
            raise                      # every trial would fail the same way
        except Exception as exc:       # noqa: BLE001 - one bad config is survivable
            logger.exception(f"trial {trial.number} failed: {exc!r}")
            return FAILED_SCORE
        minutes = (time.perf_counter() - t0) / 60

        best = bench.get("best", {})
        score = float(best.get("score", FAILED_SCORE))
        rec = best.get("recording_metrics", {}) or {}
        stop = bench.get("stopping", {}) or {}

        # Everything needed to rank and re-run trials later, with no GPU.
        for key, value in {
            "n_parameters": bench.get("n_parameters"),
            "epochs_run": stop.get("epochs_run"),
            "checks_run": stop.get("checks_run"),
            "best_check": best.get("check"),
            "best_epoch_frac": best.get("epoch_frac"),
            "minutes": round(minutes, 2),
            "auc_roc": rec.get("auc_roc"),
            "pauc_fpr0.1": rec.get("pauc_fpr0.1"),
            "pauc_tpr0.9": rec.get("pauc_tpr0.9"),
            "sens_at_spec0.9": rec.get("sens_at_spec0.9"),
            "spec_at_sens0.95": rec.get("spec_at_sens0.95"),
            "run_dir": str(run_dir),
        }.items():
            if value is not None:
                trial.set_user_attr(key, value)

        logger.info(
            f"    trial {trial.number}: {objective_metric}={score:.4f}  "
            f"spec@sens.95={rec.get('spec_at_sens0.95', float('nan')):.4f}  "
            f"params={bench.get('n_parameters', 0):,}  {minutes:.1f} min"
        )

        if not keep_artifacts:
            for name in ("best.pt", "predictions_tune.npz"):
                target = run_dir / name
                if target.exists():
                    target.unlink()

        return score

    return objective


def report(study: optuna.Study, top_k: int, out_dir: Optional[Path]) -> dict[str, Any]:
    """
    @brief Summarize the study and write the top-k configurations.

    @details
    The top-k file is the input to the seed-validation stage. The best trial's
    score is the maximum over many noisy estimates and is therefore optimistic
    -- the gap between it and the mean of fresh-seed re-runs measures that
    selection bias directly, which is why the re-run stage is not optional.

    @param study The study.
    @param top_k How many configurations to export.
    @param out_dir Where to write `top_k.json` and `trials.csv`; `None` skips.
    @return The summary dict.
    """
    done = [t for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and t.value is not None
            and t.value > FAILED_SCORE]
    states: dict[str, int] = {}
    for t in study.get_trials(deepcopy=False):
        states[t.state.name] = states.get(t.state.name, 0) + 1

    ranked = sorted(done, key=lambda t: t.value, reverse=True)[:top_k]
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": "maximize",
        "n_trials": len(study.get_trials(deepcopy=False)),
        "states": states,
        "n_usable": len(done),
        "top": [
            {
                "rank": i + 1,
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "user_attrs": dict(t.user_attrs),
            }
            for i, t in enumerate(ranked)
        ],
    }

    logger.info("=" * 72)
    logger.info(f"Study: {study.study_name}")
    logger.info(f"Trials: {summary['n_trials']}  states: {states}")
    if not ranked:
        logger.warning("No usable completed trials.")
        return summary

    spread = ranked[0].value - ranked[-1].value
    logger.info(f"\nTop {len(ranked)} by objective:")
    for row in summary["top"]:
        ua = row["user_attrs"]
        logger.info(
            f"  #{row['rank']} trial {row['trial']:>3}  score {row['score']:.4f}  "
            f"spec@sens.95 {ua.get('spec_at_sens0.95', float('nan')):.4f}  "
            f"params {ua.get('n_parameters', 0):,}  {ua.get('minutes', 0):.0f} min"
        )
        logger.info(f"       {row['params']}")

    # Which part of the space the winners came from -- a direct check on
    # whether weighting the budget toward regularization was the right call.
    logger.info("\nSpread across the top-k:")
    for p in SEARCH_SPACE:
        vals = [t.params.get(p.name) for t in ranked if p.name in t.params]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            logger.info(f"  {p.name:24} [{p.group}] min {min(vals):.4g}  max {max(vals):.4g}")
        elif vals:
            logger.info(f"  {p.name:24} [{p.group}] {sorted(set(map(str, vals)))}")

    logger.info(
        f"\nTop-k score spread: {spread:.4f}. "
        + (
            "Tightly clustered -- re-run 5 configs x 3 seeds."
            if spread < 0.01
            else "Clearly separated -- re-run 3 configs x 5 seeds."
        )
    )
    logger.info(
        "The best score is the maximum of many noisy estimates and is "
        "optimistic. Select on the mean of fresh-seed re-runs, not on this."
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "top_k.json").write_text(json.dumps(summary, indent=2, default=str))
        rows = ["trial,state,score," + ",".join(p.name for p in SEARCH_SPACE)]
        for t in study.get_trials(deepcopy=False):
            vals = ",".join(str(t.params.get(p.name, "")) for p in SEARCH_SPACE)
            rows.append(f"{t.number},{t.state.name},{t.value if t.value is not None else ''},{vals}")
        (out_dir / "trials.csv").write_text("\n".join(rows) + "\n")
        logger.info(f"\nWrote {out_dir / 'top_k.json'} and {out_dir / 'trials.csv'}")
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @param argv Argument list; `None` uses `sys.argv`.
    @return Parsed namespace.
    """
    p = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    p.add_argument("--config", default="configs/step2_earlystop.yaml",
                   help="Base YAML config; the search overrides leaves of it")
    p.add_argument("--study-name", default="eeg-hpo-v1",
                   help="Study name; the resume key")
    p.add_argument("--storage", default=None,
                   help="SQLAlchemy URL, e.g. sqlite:///runs/hpo/study.db. "
                        "Without it the study is in memory and is lost on exit")
    p.add_argument("--train-data-dir", default=None,
                   help="Override data.train_data_dir for every trial, e.g. a "
                        "node-local staged copy")
    p.add_argument("--out-dir", default="runs/hpo",
                   help="Per-trial run directories and the final report")
    p.add_argument("--n-trials", type=int, default=70, help="Trials to run")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="Stop cleanly after this long; set below the SLURM limit")
    p.add_argument("--num-epochs", type=int, default=None,
                   help="Override the epoch budget per trial")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Cap steps per epoch (0 = uncapped); for smoke runs")
    p.add_argument("--objective", default=DEFAULT_OBJECTIVE,
                   help="train.select_metric to optimize")
    p.add_argument("--prune", default="hyperband",
                   choices=["hyperband", "median", "none"])
    p.add_argument("--prune-warmup-checks", type=int, default=8,
                   help="Checks a trial gets before it may be pruned. The "
                        "metric varies ~0.02 between checks, so pruning early "
                        "kills good trials on noise")
    p.add_argument("--n-startup", type=int, default=15,
                   help="Random trials before TPE's model takes over")
    p.add_argument("--seed", type=int, default=42, help="Sampler seed")
    p.add_argument("--top-k", type=int, default=5,
                   help="Configurations to export for seed validation")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="Keep every trial's checkpoint and predictions")
    p.add_argument("--no-startup-trials", action="store_true",
                   help="Skip the bracketing startup points")
    p.add_argument("--report-only", action="store_true",
                   help="Summarize an existing study and exit")
    p.add_argument("--describe", action="store_true",
                   help="Print the search space and exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @param argv Argument list; `None` uses `sys.argv`.
    @return Process exit code.
    """
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.describe:
        print(describe())
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(out_dir / "hpo.log"), level="DEBUG")

    if args.storage is None and not args.report_only:
        logger.warning(
            "No --storage: the study is in memory and will be lost if this "
            "process dies. Pass sqlite:///<path>/study.db for any real run."
        )

    study = make_study(
        storage=args.storage,
        study_name=args.study_name,
        seed=args.seed,
        n_startup=args.n_startup,
        prune=args.prune,
        prune_warmup_checks=args.prune_warmup_checks,
    )

    if args.report_only:
        report(study, args.top_k, out_dir)
        return 0

    from eeg_cnn_lstm.models.train import train as train_fn  # heavy import

    logger.info(describe())
    logger.info(
        f"\nObjective: {args.objective} (maximize) | pruner: {args.prune} "
        f"(warmup {args.prune_warmup_checks} checks) | "
        f"budget: {args.n_trials} trials"
        + (f", {args.timeout_hours:g} h" if args.timeout_hours else "")
    )

    already = len([t for t in study.get_trials(deepcopy=False)
                   if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    if already:
        logger.info(f"Resuming: {already} trial(s) already recorded.")
    if not args.no_startup_trials:
        n = enqueue_startup(study)
        logger.info(f"Enqueued {n} bracketing startup point(s).")

    base_overrides: dict[str, Any] = {}
    if args.train_data_dir:
        base_overrides["data"] = {"train_data_dir": args.train_data_dir}
        logger.info(f"Every trial reads data from {args.train_data_dir}")

    objective = make_objective(
        config_path=args.config,
        out_root=out_dir,
        objective_metric=args.objective,
        train_fn=train_fn,
        base_overrides=base_overrides,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        keep_artifacts=args.keep_artifacts,
    )

    remaining = max(0, args.n_trials - already)
    try:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=args.timeout_hours * 3600 if args.timeout_hours else None,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted; reporting what has completed so far.")

    report(study, args.top_k, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())