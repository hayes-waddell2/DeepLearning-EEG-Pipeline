"""
@file cross_validate.py
@brief Run patient-level K-fold cross-validation for one or more frozen configurations.

@details
This is the reported-performance stage. Everything that chooses a setting --
the search, the seed validation, the capacity ablation -- happened against the
tuning pool T. Cross-validation runs on pool E, where each fold is a set of
patients no model in that fold has trained on.

`train.py` already implements the fold mechanics: with `train.fold = k` it
trains on E minus fold k, keeps using T for early stopping and checkpoint
selection, and scores the held-out fold with the restored best checkpoint. The
score this runner records is therefore the held-out fold's recording-level
metric from `bench["test"]`, **not** the tuning-pool score in `bench["best"]`.
Recording the latter would report the number the model was stopped on, which is
optimistic by construction.

Folds are the inner loop and configurations the outer one, the opposite of
`seed_validate.py`. A cross-validation estimate needs every fold, so if the
wall-clock budget runs out it is better to hold one complete K-fold estimate
than two half-finished ones.

One seed is used throughout. Fold is the blocking variable here, and both
configurations see the same folds in the same order, so a comparison between
them is paired on fold and the fold-to-fold variation cancels in the
difference. Seed variation was already characterized in the seed validation
stage and is not re-measured here.

Usage:

python -m scripts.cross_validate --describe
python -m scripts.cross_validate --dry-run
python -m scripts.cross_validate \
    --config configs/ablation_base.yaml \
    --configs-json configs/cv.json \
    --out-dir runs/cv \
    --timeout-hours 5.5

python -m scripts.cross_validate --report-only --out-dir runs/cv
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics as st
import time
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from eeg_cnn_lstm.utils.search_space import params_to_overrides

## @brief Metric the runs optimize and stop on; must be a key of `train.SELECT_METRICS`.
DEFAULT_OBJECTIVE = "recording_pauc_sens"

## @brief Recording-level key read off the held-out fold; the reported metric.
PRIMARY_KEY = "pauc_tpr0.9"

## @brief Secondary metric reported alongside, because it is the deployed one.
SECONDARY_KEY = "spec_at_sens0.95"

## @brief Further recording-level keys carried through when present.
EXTRA_KEYS = ("auc_roc", "sens_at_spec0.9")

## @brief Score recorded for a run that raised. Never aggregated.
FAILED_SCORE = -1.0

## @brief Default fold indices, matching `make_splits.py --n-folds 10`.
DEFAULT_FOLDS = list(range(10))

## @brief Default seed. Fold is the blocking variable, so this stays fixed.
DEFAULT_SEED = 42

## @brief Exceptions that indicate a broken setup rather than a bad config.
FATAL = (FileNotFoundError, ImportError, KeyError)


def load_configs(path: Path) -> tuple[list[dict[str, Any]], list[int], int]:
    """
    @brief Read the frozen configuration set to cross-validate.

    @details
    Accepts the same schema as the ablation point sets, so a configuration can
    be lifted straight out of `configs/ablation_small.json` without being
    retyped. `folds` and `seed` are optional and fall back to the module
    defaults.

    @param path JSON holding `configs`, and optionally `folds` and `seed`.
    @return Tuple `(configs, folds, seed)`.
    @throws ValueError If the document is missing or malformed.
    """
    doc = json.loads(path.read_text())
    configs = doc.get("configs")
    if not configs:
        raise ValueError(f"{path} must contain a non-empty 'configs'")
    for c in configs:
        if "name" not in c or not ("params" in c or "overrides" in c):
            raise ValueError(f"{path}: every config needs 'name' and 'params' or 'overrides'")
    folds = [int(f) for f in doc.get("folds", DEFAULT_FOLDS)]
    if not folds:
        raise ValueError(f"{path}: 'folds' must not be empty")
    return configs, folds, int(doc.get("seed", DEFAULT_SEED))


def build_overrides(
    config: dict[str, Any],
    fold: int,
    seed: int,
    output_dir: str,
    objective: str,
    train_data_dir: Optional[str],
    num_epochs: Optional[int],
) -> dict[str, Any]:
    """
    @brief Turn one configuration entry into a full config override tree.

    @details
    Mirrors `seed_validate.build_overrides` so a configuration runs identically
    in both stages, with one addition: `train.fold` selects the held-out fold.
    `train.py` reads that key, trains on E minus the fold, and scores the fold
    with the restored best checkpoint.

    The fold does not touch `loader.seed`. Initialization and batch order stay
    fixed across folds so that fold-to-fold variation is attributable to the
    patients in the fold rather than to the training process.

    @param config One entry from the configuration set.
    @param fold Held-out fold index.
    @param seed RNG seed, held constant across folds.
    @param output_dir Run directory for this fold.
    @param objective `train.select_metric` to optimize and stop on.
    @param train_data_dir Optional node-local staged copy of the training data.
    @param num_epochs Optional epoch-budget override.
    @return Nested override tree to merge over the YAML config.
    @throws KeyError If the entry carries neither `params` nor `overrides`.
    """
    if "overrides" in config:
        ov = copy.deepcopy(config["overrides"])
    else:
        ov = params_to_overrides(config["params"])
    ov.setdefault("loader", {})["seed"] = int(seed)
    train = ov.setdefault("train", {})
    train["output_dir"] = output_dir
    train["select_metric"] = objective
    train["fold"] = int(fold)
    if num_epochs is not None:
        train["num_epochs"] = int(num_epochs)
    if train_data_dir:
        ov.setdefault("data", {})["train_data_dir"] = train_data_dir
    return ov


def read_fold_score(bench: dict[str, Any]) -> dict[str, Any]:
    """
    @brief Extract the held-out fold's metrics from a finished run record.

    @details
    `bench["test"]` is written only when the run had a fold. Its absence means
    the fold index never reached `train.py`, usually because `data.splits_csv`
    is unset -- a silent tuning run would otherwise be recorded as a fold
    result, so this is treated as a failure rather than ignored.

    @param bench Record returned by `train.run`.
    @return Partial run record with `score` and any available metrics.
    """
    test = bench.get("test") or {}
    recm = test.get("recording_metrics") or {}
    if PRIMARY_KEY not in recm:
        return {"score": FAILED_SCORE, "error": "no held-out fold metrics in bench['test']"}
    out: dict[str, Any] = {"score": float(recm[PRIMARY_KEY])}
    for k in (SECONDARY_KEY, *EXTRA_KEYS):
        if k in recm:
            out[k] = recm[k]
    best = bench.get("best", {}) or {}
    out["tune_score"] = best.get("score")
    out["best_check"] = best.get("check")
    out["checks_run"] = (bench.get("stopping", {}) or {}).get("checks_run")
    out["n_parameters"] = bench.get("n_parameters")
    return out


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    @brief Aggregate per-fold runs into one row per configuration.

    @param runs Flat list of run records.
    @return One row per configuration, best mean first.
    """
    by: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        by.setdefault(r["name"], []).append(r)

    rows = []
    for name, rs in by.items():
        ok = [r for r in rs if r.get("score", FAILED_SCORE) > FAILED_SCORE]
        scores = [r["score"] for r in ok]
        sec = [r[SECONDARY_KEY] for r in ok if r.get(SECONDARY_KEY) is not None]
        rows.append(
            {
                "name": name,
                "n_folds": len(ok),
                "failed": len(rs) - len(ok),
                "mean": st.fmean(scores) if scores else float("nan"),
                "sd": st.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores) if scores else float("nan"),
                "max": max(scores) if scores else float("nan"),
                "sec_mean": st.fmean(sec) if sec else float("nan"),
                "n_parameters": next((r.get("n_parameters") for r in ok), None),
                "minutes": st.fmean([r.get("minutes", 0.0) for r in rs]),
            }
        )
    rows.sort(key=lambda r: (-r["mean"] if r["mean"] == r["mean"] else 0.0))
    return rows


def report(runs: list[dict[str, Any]], folds: list[int], out_dir: Optional[Path]) -> list[str]:
    """
    @brief Render the cross-validation summary and write it out.

    @details
    Reports the fold-to-fold spread rather than a single number, because the
    spread is what says how much a one-off estimate on any single split could
    have differed. A configuration whose folds span more than the seed-to-seed
    noise measured earlier is varying with the patients, not with the training
    process, and that is the quantity cross-validation exists to expose.

    @param runs Flat list of run records.
    @param folds Fold indices that were requested.
    @param out_dir Where to write `cv_report.txt`, `cv.json` and `cv.csv`;
           `None` skips writing.
    @return The report lines.
    """
    rows = summarize(runs)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("=" * 84)
    emit(f"CROSS-VALIDATION -- {len(folds)} patient-level folds, metric {PRIMARY_KEY}")
    emit("=" * 84)
    emit()
    emit(
        f"  {'config':20} {'params':>10} {'folds':>6} {'mean':>8} {'sd':>7} "
        f"{'min':>8} {'max':>8} {SECONDARY_KEY:>10}"
    )
    for r in rows:
        par = f"{r['n_parameters']:,}" if r["n_parameters"] else "-"
        emit(
            f"  {r['name']:20} {par:>10} {r['n_folds']:>6} {r['mean']:>8.4f} "
            f"{r['sd']:>7.4f} {r['min']:>8.4f} {r['max']:>8.4f} {r['sec_mean']:>10.4f}"
        )
        if r["failed"]:
            emit(f"  {'':20} {r['failed']} fold(s) failed and are excluded from the mean")

    emit()
    for r in rows:
        if r["n_folds"] > 1:
            emit(f"  {r['name']}: folds span {r['max'] - r['min']:.4f}")

    if len(rows) > 1:
        emit()
        emit("Two or more configurations were cross-validated on the same folds.")
        emit("Compare them paired on fold rather than from the means above:")
        emit(
            "  python -m scripts.compare_configs --runs <out-dir>/runs.json "
            "--block fold --margin 0.01"
        )

    emit()
    emit("=" * 84)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cv_report.txt").write_text("\n".join(lines) + "\n")
        (out_dir / "cv.json").write_text(
            json.dumps({"folds": folds, "metric": PRIMARY_KEY, "rows": rows}, indent=2)
        )
        head = "name,n_folds,failed,mean,sd,min,max,sec_mean,n_parameters"
        csv = [head] + [
            f"{r['name']},{r['n_folds']},{r['failed']},{r['mean']:.6f},{r['sd']:.6f},"
            f"{r['min']:.6f},{r['max']:.6f},{r['sec_mean']:.6f},{r['n_parameters'] or ''}"
            for r in rows
        ]
        (out_dir / "cv.csv").write_text("\n".join(csv) + "\n")
        logger.info(f"Wrote {out_dir / 'cv_report.txt'}, cv.json and cv.csv")

    return lines


def run_all(
    configs: list[dict[str, Any]],
    folds: list[int],
    seed: int,
    base_config: str,
    out_dir: Path,
    objective: str,
    train_data_dir: Optional[str],
    num_epochs: Optional[int],
    train_fn: Callable[..., dict[str, Any]],
    deadline: Optional[float],
) -> list[dict[str, Any]]:
    """
    @brief Run every (configuration, fold) pair, tolerating individual failures.

    @details
    Configuration is the outer loop. A cross-validation estimate is only
    meaningful over the complete fold set, so a budget overrun should leave one
    finished estimate rather than two partial ones.

    @param configs Frozen configurations to cross-validate.
    @param folds Held-out fold indices.
    @param seed RNG seed, constant across folds.
    @param base_config Path to the YAML config the overrides merge over.
    @param out_dir Run and report root.
    @param objective `train.select_metric`.
    @param train_data_dir Optional node-local staged copy.
    @param num_epochs Optional epoch-budget override.
    @param train_fn Injected `train.run`, for testing.
    @param deadline `time.monotonic()` value after which no new run starts.
    @return Flat list of run records.
    """
    runs: list[dict[str, Any]] = []
    total = len(configs) * len(folds)
    done = 0

    for cfg in configs:
        name = cfg["name"]
        for fold in folds:
            if deadline is not None and time.monotonic() > deadline:
                logger.warning(
                    f"Wall-clock budget reached after {done}/{total} runs; "
                    "stopping cleanly with what is finished."
                )
                return runs

            run_dir = out_dir / f"{name}__fold{fold}"
            done += 1
            logger.info(f"[{done}/{total}] {name} fold {fold} -> {run_dir}")

            rec: dict[str, Any] = {
                "name": name,
                "fold": fold,
                "seed": seed,
                "score": FAILED_SCORE,
            }
            t0 = time.monotonic()
            try:
                bench = train_fn(
                    config_path=base_config,
                    config_overrides=build_overrides(
                        cfg,
                        fold,
                        seed,
                        str(run_dir),
                        objective,
                        train_data_dir,
                        num_epochs,
                    ),
                )
                rec.update(read_fold_score(bench))
            except FATAL:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad fold must not end the stage
                logger.exception(f"{name} fold {fold} failed: {exc}")
                rec["error"] = repr(exc)
            rec["minutes"] = (time.monotonic() - t0) / 60.0
            runs.append(rec)

            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "runs.json").write_text(json.dumps(runs, indent=2, default=str))

    return runs


def describe(configs: list[dict[str, Any]], folds: list[int], seed: int) -> None:
    """
    @brief Print the plan without running anything.

    @param configs Frozen configurations.
    @param folds Held-out fold indices.
    @param seed RNG seed.
    """
    total = len(configs) * len(folds)
    logger.info(f"{len(configs)} configuration(s) x {len(folds)} folds = {total} runs, seed {seed}")
    logger.info(f"Folds: {folds}")
    for c in configs:
        model = (c.get("overrides") or {}).get("model", {})
        logger.info(f"  {c['name']:20} {json.dumps(model, sort_keys=True)[:96]}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.

    @param argv Argument vector, or `None` for `sys.argv`.
    @return Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description="Patient-level K-fold cross-validation for frozen configurations."
    )
    p.add_argument("--config", default="configs/ablation_base.yaml", help="Base YAML config")
    p.add_argument("--configs-json", default="configs/cv.json", help="Frozen configuration set")
    p.add_argument("--out-dir", default="runs/cv", help="Run and report root")
    p.add_argument("--folds", default=None, help="Comma-separated folds, overriding the JSON")
    p.add_argument("--seed", type=int, default=None, help="Seed, overriding the JSON")
    p.add_argument("--train-data-dir", default=None, help="Node-local staged copy of the data")
    p.add_argument("--num-epochs", type=int, default=None, help="Override the epoch budget")
    p.add_argument("--objective", default=DEFAULT_OBJECTIVE, help="train.select_metric")
    p.add_argument("--timeout-hours", type=float, default=None, help="Stop starting new runs after")
    p.add_argument("--describe", action="store_true", help="Print the plan and exit")
    p.add_argument("--dry-run", action="store_true", help="Build every override tree and exit")
    p.add_argument("--report-only", action="store_true", help="Re-render the report from runs.json")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args(argv)
    out_dir = Path(a.out_dir)

    if a.report_only:
        runs_path = out_dir / "runs.json"
        if not runs_path.is_file():
            logger.error(f"{runs_path} not found")
            return 1
        runs = json.loads(runs_path.read_text())
        folds = sorted({r["fold"] for r in runs})
        print("\n".join(report(runs, folds, out_dir)))
        return 0

    cfg_path = Path(a.configs_json)
    if not cfg_path.is_file():
        logger.error(f"{cfg_path} not found")
        return 1
    configs, folds, seed = load_configs(cfg_path)
    if a.folds:
        folds = [int(f) for f in a.folds.split(",") if f.strip()]
    if a.seed is not None:
        seed = int(a.seed)

    if a.describe:
        describe(configs, folds, seed)
        return 0

    if a.dry_run:
        for cfg in configs:
            for fold in folds:
                build_overrides(
                    cfg,
                    fold,
                    seed,
                    str(out_dir / f"{cfg['name']}__fold{fold}"),
                    a.objective,
                    a.train_data_dir,
                    a.num_epochs,
                )
        logger.info(f"Built {len(configs) * len(folds)} override trees; all valid.")
        return 0

    from eeg_cnn_lstm.models.train import run as train_run

    deadline = None if a.timeout_hours is None else time.monotonic() + a.timeout_hours * 3600.0
    describe(configs, folds, seed)
    runs = run_all(
        configs,
        folds,
        seed,
        a.config,
        out_dir,
        a.objective,
        a.train_data_dir,
        a.num_epochs,
        train_run,
        deadline,
    )
    if not runs:
        logger.error("No runs completed.")
        return 1
    print("\n".join(report(runs, folds, out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
