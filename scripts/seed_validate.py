"""
@file seed_validate.py
@brief Re-run the top HPO configurations across fresh seeds and select on the mean.

@details
Phase 3, stage 2. The search reported `pauc_tpr0.9 = 0.8026` for trial 46, but
that number is the maximum over 42 noisy estimates and is optimistic by
construction: the winner is partly whichever configuration drew the kindest
validation noise. The top-10 spread was 0.0100 against ~0.02 check-to-check
variation, which means the ranking the search produced is one statistical tie,
not an ordering. This stage breaks the tie properly -- every configuration is
re-run from several seeds and selected on the **mean**, with the spread
reported so the selection is auditable.

@par Why ten configurations and not five
Two reasons, both measured rather than assumed.

First, the two metrics disagree. Trial 46 leads on `pauc_tpr0.9` but ranks 9th
on `spec_at_sens0.95`; trial 32 is the reverse -- 10th on pAUC, 1st on the
operating-point metric the deliverable actually promises. Ranking on either
alone discards a real contender, so the completed configurations here are the
seven best by the mean of their two ranks.

Second, three configurations were pruned at check 8 despite having already
peaked above the best completed trial's #10. Optuna's pruner compares the
**latest reported** value while the objective returns the **best**; with a
metric that swings ~0.02 between checks, a trial that happens to be on a
down-swing at the decision point dies regardless of its peak. Trial 68 is the
clear case: at check 6 it scored 0.7955 against the eventual winner's 0.7908,
then reported 0.7573 at check 8 and was killed. These three are included
because the evidence does not distinguish them from the winners, not because
they are expected to win.

@par What this does not fix
Re-running from fresh seeds measures the variance of the training process --
initialization, batch order, and segment sampling. It does **not** widen the
tuning pool, so a configuration that suits pool T specifically will still look
good here. The one-shot evaluation on the held-out TUAB eval partition remains
the only unbiased estimate, and it happens once, after this stage selects.

@par Usage:
@verbatim
python -m scripts.seed_validate --describe
python -m scripts.seed_validate --dry-run
python -m scripts.seed_validate \
    --config configs/step2_earlystop.yaml \
    --configs-json configs/seed_validation.json \
    --out-dir runs/seedval \
    --train-data-dir /tmp/$USER/$SLURM_JOB_ID/train \
    --timeout-hours 10.0
python -m scripts.seed_validate --report-only --out-dir runs/seedval
@endverbatim
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from eeg_cnn_lstm.utils.search_space import params_to_overrides

## @brief Metric the selection is made on; must be a key of `train.SELECT_METRICS`.
DEFAULT_OBJECTIVE = "recording_pauc_sens"

## @brief Secondary metric reported alongside, because it is the deployed one.
SECONDARY_KEY = "spec_at_sens0.95"

## @brief Score recorded for a run that raised. Never selected.
FAILED_SCORE = -1.0

## @brief Exceptions that indicate a broken setup rather than a bad config.
FATAL = (FileNotFoundError, ImportError, KeyError)


def load_configs(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """
    @brief Read the configuration set written by the HPO report stage.

    @param path JSON produced alongside `top_k.json`.
    @return Tuple `(configs, seeds)`.
    @throws ValueError If the document is missing required keys.
    """
    doc = json.loads(path.read_text())
    configs = doc.get("configs")
    seeds = doc.get("seeds")
    if not configs or not seeds:
        raise ValueError(f"{path} must contain non-empty 'configs' and 'seeds'")
    for c in configs:
        if "name" not in c or not ("params" in c or "overrides" in c):
            raise ValueError(f"{path}: every config needs 'name' and 'params' or 'overrides'")
    return configs, [int(s) for s in seeds]


def build_overrides(
    config: dict[str, Any],
    seed: int,
    output_dir: str,
    objective: str,
    train_data_dir: Optional[str],
    num_epochs: Optional[int],
) -> dict[str, Any]:
    """
    @brief Turn one configuration entry into a full config override tree.

    @details
    Two input forms are accepted, because two stages feed this runner.

    `params` holds search-space values and goes through `params_to_overrides`,
    the same function the search used, so a configuration re-run here is
    bit-for-bit the one the search scored -- `conv_width_mult` and `downsample`
    pass through their transforms rather than being re-derived, which is where
    a hand-written re-run would drift.

    `overrides` holds a config tree directly. The parameter-efficiency ablation
    needs this: it varies CNN depth, which changes `conv_channels` and
    `pool_sizes` together and cannot be expressed in a search space whose pool
    spec is fixed at three blocks, and it reaches capacities the search never
    offered. An ablation is not bound by the space the search explored.

    The seed lands on `loader.seed`, which drives initialization, batch order
    and segment sampling. It does **not** move the patient-level T/E split:
    that comes from `data.splits_csv` and stays fixed, or the runs would not be
    comparable.

    @param config One entry from the configuration set, carrying either
           `params` or `overrides`.
    @param seed RNG seed for this repeat.
    @param output_dir Run directory for this repeat.
    @param objective `train.select_metric` to optimize and report.
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
    if num_epochs is not None:
        train["num_epochs"] = int(num_epochs)
    if train_data_dir:
        ov.setdefault("data", {})["train_data_dir"] = train_data_dir
    return ov


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    @brief Aggregate per-seed runs into one row per configuration.

    @details
    Selection is on `mean`. `spread` (max - min) is carried because a
    configuration with a high mean and a wide spread is not a better
    configuration, it is a less predictable one, and the final model has to be
    reproducible from the committed config.

    @param runs Flat list of completed run records.
    @return One row per configuration, sorted by mean descending.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        by_name.setdefault(r["name"], []).append(r)

    rows = []
    for name, rs in by_name.items():
        ok = [r for r in rs if r["score"] > FAILED_SCORE]
        scores = [r["score"] for r in ok]
        sec = [r[SECONDARY_KEY] for r in ok if r.get(SECONDARY_KEY) is not None]
        if not scores:
            rows.append({"name": name, "n": 0, "mean": FAILED_SCORE, "failed": len(rs)})
            continue
        rows.append(
            {
                "name": name,
                "source_trial": rs[0].get("source_trial"),
                "source_state": rs[0].get("source_state"),
                "hpo_score": rs[0].get("hpo_score"),
                "n": len(scores),
                "failed": len(rs) - len(scores),
                "mean": statistics.fmean(scores),
                "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
                "spread": max(scores) - min(scores),
                f"mean_{SECONDARY_KEY}": statistics.fmean(sec) if sec else None,
                "n_parameters": rs[0].get("n_parameters"),
                "scores": scores,
            }
        )
    return sorted(rows, key=lambda r: r["mean"], reverse=True)


def report(rows: list[dict[str, Any]], out_dir: Optional[Path]) -> dict[str, Any]:
    """
    @brief Print the selection table and write it to disk.

    @details
    The headline comparison is `mean` against the configuration's original HPO
    score. The gap between them is the selection bias the search carried, and
    printing it makes that bias a measured quantity in the write-up rather than
    a caveat.

    @param rows Output of `summarize`.
    @param out_dir Where to write `seed_validation.json`; `None` skips.
    @return The summary document.
    """
    logger.info("=" * 86)
    if not rows or rows[0]["n"] == 0:
        logger.warning("No usable runs.")
        return {"rows": rows}

    logger.info(
        f"{'config':>12} {'src':>5} {'n':>2} {'mean':>7} {'sd':>6} "
        f"{'spread':>7} {'hpo':>7} {'bias':>7} {'spec@.95':>8} {'params':>9}"
    )
    for r in rows:
        if r["n"] == 0:
            logger.warning(f"{r['name']:>12}  all {r['failed']} runs failed")
            continue
        hpo = r.get("hpo_score")
        bias = (hpo - r["mean"]) if hpo is not None else float("nan")
        sec = r.get(f"mean_{SECONDARY_KEY}")
        logger.info(
            f"{r['name']:>12} {str(r.get('source_state', ''))[:5]:>5} {r['n']:>2} "
            f"{r['mean']:>7.4f} {r['sd']:>6.4f} {r['spread']:>7.4f} "
            f"{(hpo if hpo is not None else float('nan')):>7.4f} {bias:>+7.4f} "
            f"{(sec if sec is not None else float('nan')):>8.4f} "
            f"{(r.get('n_parameters') or 0):>9,}"
        )

    usable = [r for r in rows if r["n"] > 1]
    best = rows[0]
    logger.info("")
    logger.info(f"Selected on the mean: {best['name']} ({best['mean']:.4f})")

    if len(rows) > 1 and rows[1]["n"] > 0:
        margin = best["mean"] - rows[1]["mean"]
        typical_sd = statistics.fmean([r["sd"] for r in usable]) if usable else 0.0
        logger.info(
            f"Margin over #2 ({rows[1]['name']}): {margin:.4f}; "
            f"typical within-config sd: {typical_sd:.4f}"
        )
        if margin < typical_sd:
            logger.warning(
                "Margin is inside the seed-to-seed noise. These two are still "
                "not separated -- prefer the one with the smaller parameter "
                "count or the better spec_at_sens0.95, and say so explicitly."
            )

    biases = [r["hpo_score"] - r["mean"] for r in rows if r.get("hpo_score") and r["n"]]
    if biases:
        logger.info(
            f"Mean HPO-score optimism across configs: {statistics.fmean(biases):+.4f} "
            "(how much the search's reported score overstated the truth)"
        )

    doc = {"objective": DEFAULT_OBJECTIVE, "selected": best["name"], "rows": rows}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "seed_validation.json").write_text(json.dumps(doc, indent=2))
        hdr = ["config", "source_trial", "n", "mean", "sd", "spread", "hpo_score"]
        lines = [",".join(hdr)]
        for r in rows:
            lines.append(
                f"{r['name']},{r.get('source_trial', '')},{r['n']},"
                f"{r['mean']:.6f},{r.get('sd', 0):.6f},{r.get('spread', 0):.6f},"
                f"{r.get('hpo_score', '')}"
            )
        (out_dir / "seed_validation.csv").write_text("\n".join(lines) + "\n")
        logger.info(f"\nWrote {out_dir / 'seed_validation.json'} and .csv")
    return doc


def run_all(
    configs: list[dict[str, Any]],
    seeds: list[int],
    base_config: str,
    out_dir: Path,
    objective: str,
    train_data_dir: Optional[str],
    num_epochs: Optional[int],
    deadline: Optional[float],
    train_fn: Callable[..., Any],
) -> list[dict[str, Any]]:
    """
    @brief Run every (configuration, seed) pair, tolerating individual failures.

    @details
    Seeds are the outer loop on purpose. If the job runs out of wall clock, the
    result is every configuration at one or two seeds rather than half the
    configurations at three -- a usable partial answer instead of a truncated
    one.

    @param configs Configuration set.
    @param seeds Seeds to repeat each configuration over.
    @param base_config Path to the YAML the overrides are merged into.
    @param out_dir Root for per-run directories.
    @param objective `train.select_metric`.
    @param train_data_dir Optional node-local staged data.
    @param num_epochs Optional epoch-budget override.
    @param deadline `time.monotonic()` value to stop before; `None` for none.
    @param train_fn Injected trainer, for testing.
    @return Flat list of run records.
    @throws FileNotFoundError, ImportError, KeyError On a broken setup.
    """
    runs: list[dict[str, Any]] = []
    total = len(configs) * len(seeds)
    done = 0

    for seed in seeds:
        for cfg in configs:
            if deadline is not None and time.monotonic() > deadline:
                logger.warning(
                    f"Wall-clock budget reached after {done}/{total} runs; "
                    "stopping cleanly with what is finished."
                )
                return runs

            name = cfg["name"]
            run_dir = out_dir / f"{name}__seed{seed}"
            done += 1
            logger.info(f"[{done}/{total}] {name} seed {seed} -> {run_dir}")

            rec: dict[str, Any] = {
                "name": name,
                "seed": seed,
                "source_trial": cfg.get("source_trial"),
                "source_state": cfg.get("source_state"),
                "hpo_score": cfg.get("hpo_score"),
                "score": FAILED_SCORE,
            }
            t0 = time.monotonic()
            try:
                bench = train_fn(
                    config_path=base_config,
                    config_overrides=build_overrides(
                        cfg,
                        seed,
                        str(run_dir),
                        objective,
                        train_data_dir,
                        num_epochs,
                    ),
                )
                best = bench.get("best", {}) or {}
                recm = best.get("recording_metrics", {}) or {}
                rec["score"] = float(best.get("score", FAILED_SCORE))
                rec["n_parameters"] = bench.get("n_parameters")
                rec["best_check"] = best.get("check")
                rec["checks_run"] = (bench.get("stopping", {}) or {}).get("checks_run")
                for k in (SECONDARY_KEY, "auc_roc", "pauc_tpr0.9", "sens_at_spec0.9"):
                    if k in recm:
                        rec[k] = recm[k]
            except FATAL:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad run must not end the stage
                logger.exception(f"{name} seed {seed} failed: {exc}")
                rec["error"] = repr(exc)
            rec["minutes"] = (time.monotonic() - t0) / 60.0
            runs.append(rec)

            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "runs.json").write_text(json.dumps(runs, indent=2, default=str))

    return runs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace of options.
    """
    p = argparse.ArgumentParser(
        description="Re-run top HPO configurations across seeds; select on the mean."
    )
    p.add_argument(
        "--config",
        default="configs/step2_earlystop.yaml",
        help="Base YAML; the configuration overrides are merged over it",
    )
    p.add_argument(
        "--configs-json",
        default="configs/seed_validation.json",
        help="Configuration set exported from the HPO report",
    )
    p.add_argument("--out-dir", default="runs/seedval", help="Run and report root")
    p.add_argument(
        "--train-data-dir",
        default=None,
        help="Override data.train_data_dir, e.g. a node-local staged copy",
    )
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds, overriding the JSON (e.g. 42,1337,2024)",
    )
    p.add_argument("--num-epochs", type=int, default=None, help="Override the epoch budget")
    p.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help="Stop cleanly after this long; set below the SLURM limit",
    )
    p.add_argument("--objective", default=DEFAULT_OBJECTIVE, help="train.select_metric")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Summarize an existing runs.json and exit",
    )
    p.add_argument("--describe", action="store_true", help="Print the plan and exit")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every override tree and validate it, but train nothing",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args(argv)
    out_dir = Path(a.out_dir)
    cfg_path = Path(a.configs_json)

    if a.report_only:
        runs_p = out_dir / "runs.json"
        if not runs_p.is_file():
            print(f"error: {runs_p} not found", file=sys.stderr)
            return 1
        report(summarize(json.loads(runs_p.read_text())), out_dir)
        return 0

    if not cfg_path.is_file():
        print(f"error: configuration set not found: {cfg_path}", file=sys.stderr)
        return 1
    configs, seeds = load_configs(cfg_path)
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    total = len(configs) * len(seeds)
    logger.info(f"{len(configs)} configurations x {len(seeds)} seeds = {total} runs")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Objective: {a.objective}")

    if a.describe or a.dry_run:
        for c in configs:
            ov = build_overrides(
                c,
                seeds[0],
                str(out_dir / c["name"]),
                a.objective,
                a.train_data_dir,
                a.num_epochs,
            )
            src = f"{c.get('source_state', '?')} trial {c.get('source_trial', '?')}"
            logger.info(f"\n{c['name']}  ({src}, hpo {c.get('hpo_score', float('nan')):.4f})")
            logger.info(f"  {json.dumps(ov, default=str)}")
        logger.info(f"\n{'--describe' if a.describe else '--dry-run'}: nothing trained")
        return 0

    from eeg_cnn_lstm.models.train import train

    deadline = time.monotonic() + a.timeout_hours * 3600.0 if a.timeout_hours else None
    runs = run_all(
        configs=configs,
        seeds=seeds,
        base_config=a.config,
        out_dir=out_dir,
        objective=a.objective,
        train_data_dir=a.train_data_dir,
        num_epochs=a.num_epochs,
        deadline=deadline,
        train_fn=train,
    )
    report(summarize(runs), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
