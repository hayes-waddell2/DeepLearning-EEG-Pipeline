"""
@file final_eval.py
@brief One-shot evaluation of the frozen configurations on the held-out partition.

@details
This is the last stage. Every setting was fixed before this script ran: the
architectures come from `configs/cv.json`, the seed is pre-specified, and both
configurations are evaluated once. Nothing here selects anything.

Two things are done differently from the earlier stages, and both matter.

**The operating point is carried, not re-derived.** `spec_at_sens0.95` computed
directly on the held-out partition would choose the threshold using the test
labels, which is a small but real form of fitting to the test set. Instead the
threshold is found on the tuning pool, where every other decision was made, and
then applied unchanged to the held-out recordings. The sensitivity actually
realised there becomes a result rather than being pinned to 0.95 by
construction.

**Uncertainty comes from a clustered bootstrap.** The held-out partition is 276
recordings, about one cross-validation fold's worth, and fold-to-fold variation
was the largest effect measured anywhere in this work. A bare point estimate
would be over-read, so the partial AUC is reported with a bootstrap interval
that resamples whole patients.

A single seed is used. Seed variance was characterised separately and is small
relative to the patient sampling that dominates a partition this size.

Usage:

python -m scripts.final_eval --describe
python -m scripts.final_eval --dry-run
python -m scripts.final_eval \
    --config configs/ablation_base.yaml \
    --configs-json configs/cv.json \
    --out-dir runs/final

python -m scripts.final_eval --report-only --out-dir runs/final
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from eeg_cnn_lstm.utils.metrics import (
    bootstrap_ci,
    recording_metrics,
    partial_auc_high_sens,
    sigmoid,
)
from eeg_cnn_lstm.utils.search_space import params_to_overrides

## @brief Metric the runs optimize and stop on.
DEFAULT_OBJECTIVE = "recording_pauc_sens"

## @brief Sensitivity floor defining the deployed operating point.
SENS_FLOOR = 0.95

## @brief Lower TPR bound of the reported partial AUC.
MIN_TPR = 0.90

## @brief Seed for the evaluation runs. Pre-specified; do not sweep.
DEFAULT_SEED = 42

## @brief Bootstrap replicates for the held-out interval.
N_BOOT = 2000

## @brief Score recorded for a run that raised.
FAILED_SCORE = -1.0

## @brief Exceptions that indicate a broken setup rather than a bad config.
FATAL = (FileNotFoundError, ImportError, KeyError)


def load_configs(path: Path) -> tuple[list[dict[str, Any]], int]:
    """
    @brief Read the frozen configuration set.

    @param path JSON holding `configs`, optionally `seed`.
    @return Tuple `(configs, seed)`.
    @throws ValueError If the document is malformed.
    """
    doc = json.loads(path.read_text())
    configs = doc.get("configs")
    if not configs:
        raise ValueError(f"{path} must contain a non-empty 'configs'")
    for c in configs:
        if "name" not in c or not ("params" in c or "overrides" in c):
            raise ValueError(f"{path}: every config needs 'name' and 'params' or 'overrides'")
    return configs, int(doc.get("seed", DEFAULT_SEED))


def build_overrides(
    config: dict[str, Any],
    seed: int,
    output_dir: str,
    objective: str,
    train_data_dir: Optional[str],
) -> dict[str, Any]:
    """
    @brief Turn one configuration entry into a full config override tree.

    @details
    No `fold` is set, so the run trains on all of pool E and scores the tuning
    pool, exactly as the search and ablation runs did. `eval.run_final_eval`
    additionally scores the held-out partition with the restored best
    checkpoint.

    @param config One entry from the configuration set.
    @param seed RNG seed.
    @param output_dir Run directory.
    @param objective `train.select_metric`.
    @param train_data_dir Optional node-local staged copy of the training data.
    @return Nested override tree to merge over the YAML config.
    """
    if "overrides" in config:
        ov = copy.deepcopy(config["overrides"])
    else:
        ov = params_to_overrides(config["params"])
    ov.setdefault("loader", {})["seed"] = int(seed)
    train = ov.setdefault("train", {})
    train["output_dir"] = output_dir
    train["select_metric"] = objective
    ov.setdefault("eval", {})["run_final_eval"] = True
    if train_data_dir:
        ov.setdefault("data", {})["train_data_dir"] = train_data_dir
    return ov


def recordings_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Aggregate segment scores in a predictions file to recording level.

    @details
    Mean of the per-segment probabilities, which is the aggregator the
    evaluation protocol fixed. Patient identifiers are carried through because
    the bootstrap resamples patients rather than recordings.

    @param path A `predictions_*.npz` written by the trainer.
    @return Tuple `(labels, scores, subjects)`, one entry per recording.
    @throws FileNotFoundError If the file is absent.
    """
    d = np.load(path, allow_pickle=False)
    probs = sigmoid(d["logits"].astype(np.float64).ravel())
    rec_idx = d["rec_idx"].astype(int).ravel()
    n_rec = len(d["rec_label"])
    sums = np.zeros(n_rec, dtype=np.float64)
    counts = np.zeros(n_rec, dtype=np.float64)
    np.add.at(sums, rec_idx, probs)
    np.add.at(counts, rec_idx, 1.0)
    scores = np.divide(sums, counts, out=np.full(n_rec, np.nan), where=counts > 0)
    return d["rec_label"].astype(int), scores, d["rec_subject_id"]


def threshold_at_sensitivity(labels: np.ndarray, scores: np.ndarray, floor: float) -> float:
    """
    @brief Highest threshold whose sensitivity still meets a floor.

    @details
    Thresholding at a positive score reaches sensitivity `floor` when at least
    that fraction of positives sit at or above it. Taking the highest such
    threshold gives the best specificity consistent with the floor, which is
    the operating point the deployed system would use.

    @param labels Recording labels (0/1).
    @param scores One score per recording.
    @param floor Required sensitivity, e.g. 0.95.
    @return The threshold.
    @throws ValueError If there are no positive recordings.
    """
    pos = np.sort(scores[labels == 1])
    if pos.size == 0:
        raise ValueError("no positive recordings; cannot set a sensitivity floor")
    k = int(np.floor((1.0 - floor) * pos.size))
    k = min(max(k, 0), pos.size - 1)
    return float(pos[k])


def analyze(run_dir: Path, seed: int) -> dict[str, Any]:
    """
    @brief Derive the reported held-out numbers from one finished run.

    @param run_dir Directory holding `predictions_tune.npz` and
           `predictions_eval.npz`.
    @param seed Bootstrap seed.
    @return Metrics for this configuration.
    @throws FileNotFoundError If either predictions file is missing.
    """
    t_lab, t_sco, _ = recordings_from_npz(run_dir / "predictions_tune.npz")
    e_lab, e_sco, e_sub = recordings_from_npz(run_dir / "predictions_eval.npz")

    thr = threshold_at_sensitivity(t_lab, t_sco, SENS_FLOOR)

    def pauc(lab: np.ndarray, sco: np.ndarray) -> float:
        return partial_auc_high_sens(lab, sco, MIN_TPR)

    point, lo, hi = bootstrap_ci(
        e_lab, e_sco, pauc, groups=e_sub, n_boot=N_BOOT, alpha=0.05, seed=seed
    )
    # The full metric set at the carried threshold: threshold-free figures
    # (auc_roc, both partial AUCs) plus the operating point and its confusion
    # matrix. auc_roc is the only number here directly comparable to the TUAB
    # literature, which does not report partial AUC.
    full = recording_metrics(e_lab, e_sco, threshold=thr)

    return {
        "n_patients": int(len(np.unique(e_sub))),
        "threshold": thr,
        "threshold_source": "tuning pool",
        "pauc": point,
        "pauc_lo": lo,
        "pauc_hi": hi,
        "tune_pauc": float(pauc(t_lab, t_sco)),
        **{k: v for k, v in full.items()},
    }


def report(runs: list[dict[str, Any]], out_dir: Optional[Path]) -> list[str]:
    """
    @brief Render the held-out summary and write it out.

    @param runs Flat list of run records.
    @param out_dir Where to write `final_eval.txt` and `.json`; `None` skips.
    @return The report lines.
    """
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("=" * 78)
    emit(
        f"HELD-OUT EVALUATION -- operating point carried from the tuning pool at "
        f"sensitivity {SENS_FLOOR:g}"
    )
    emit("=" * 78)
    for r in runs:
        emit()
        emit(f"{r['name']}  ({r.get('n_parameters') or '?'} parameters)")
        if r.get("error"):
            emit(f"  FAILED: {r['error']}")
            continue
        emit(f"  recordings {r['n_recordings']}, patients {r['n_patients']}")
        emit(f"  tuning-pool pAUC   {r['tune_pauc']:.3f}")
        emit(f"  held-out pAUC      {r['pauc']:.3f}  [{r['pauc_lo']:.3f}, {r['pauc_hi']:.3f}]")
        emit(f"  threshold          {r['threshold']:.4f}  (from the tuning pool)")
        emit(f"  realised sens      {r['sensitivity']:.3f}   specificity {r['specificity']:.3f}")
        emit(f"  full AUC           {r['auc_roc']:.3f}")
        emit(f"  accuracy           {r['accuracy']:.3f}   F1 {r['f1']:.3f}")
        emit(f"  confusion          tp {r['tp']}  fp {r['fp']}  fn {r['fn']}  tn {r['tn']}")
    emit()
    emit("The realised sensitivity is what the carried threshold achieves here. It is")
    emit("not fixed at the floor, because the floor was set on different patients.")
    emit("=" * 78)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_eval.txt").write_text("\n".join(lines) + "\n")
        (out_dir / "final_eval.json").write_text(json.dumps(runs, indent=2, default=str))
        logger.info(f"Wrote {out_dir / 'final_eval.txt'} and .json")
    return lines


def run_all(
    configs: list[dict[str, Any]],
    seed: int,
    base_config: str,
    out_dir: Path,
    objective: str,
    train_data_dir: Optional[str],
    train_fn: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    @brief Train and evaluate every frozen configuration once.

    @param configs Frozen configurations.
    @param seed RNG seed.
    @param base_config Path to the YAML config.
    @param out_dir Run and report root.
    @param objective `train.select_metric`.
    @param train_data_dir Optional node-local staged copy.
    @param train_fn Injected `train.train`, for testing.
    @return Flat list of run records.
    """
    runs: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, start=1):
        name = cfg["name"]
        run_dir = out_dir / name
        logger.info(f"[{i}/{len(configs)}] {name} -> {run_dir}")

        rec: dict[str, Any] = {"name": name, "seed": seed, "score": FAILED_SCORE}
        t0 = time.monotonic()
        try:
            bench = train_fn(
                config_path=base_config,
                config_overrides=build_overrides(
                    cfg, seed, str(run_dir), objective, train_data_dir
                ),
            )
            if "final_eval" not in bench:
                raise RuntimeError("no final_eval in bench; is eval.run_final_eval reaching it?")
            rec["n_parameters"] = bench.get("n_parameters")
            rec.update(analyze(run_dir, seed))
            rec["score"] = rec["pauc"]
        except FATAL:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad run must not end the stage
            logger.exception(f"{name} failed: {exc}")
            rec["error"] = repr(exc)
        rec["minutes"] = (time.monotonic() - t0) / 60.0
        runs.append(rec)

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "runs.json").write_text(json.dumps(runs, indent=2, default=str))
    return runs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @param argv Argument vector, or `None` for `sys.argv`.
    @return Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description="One-shot held-out evaluation of the frozen configurations."
    )
    p.add_argument("--config", default="configs/ablation_base.yaml", help="Base YAML config")
    p.add_argument("--configs-json", default="configs/cv.json", help="Frozen configuration set")
    p.add_argument("--out-dir", default="runs/final", help="Run and report root")
    p.add_argument("--seed", type=int, default=None, help="Seed, overriding the JSON")
    p.add_argument("--train-data-dir", default=None, help="Node-local staged copy of the data")
    p.add_argument("--objective", default=DEFAULT_OBJECTIVE, help="train.select_metric")
    p.add_argument("--describe", action="store_true", help="Print the plan and exit")
    p.add_argument("--dry-run", action="store_true", help="Build every override tree and exit")
    p.add_argument("--report-only", action="store_true", help="Re-render from runs.json")
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
        print("\n".join(report(json.loads(runs_path.read_text()), out_dir)))
        return 0

    cfg_path = Path(a.configs_json)
    if not cfg_path.is_file():
        logger.error(f"{cfg_path} not found")
        return 1
    configs, seed = load_configs(cfg_path)
    if a.seed is not None:
        seed = int(a.seed)

    if a.describe:
        logger.info(f"{len(configs)} configuration(s), seed {seed}, one held-out pass each")
        for c in configs:
            model = (c.get("overrides") or {}).get("model", {})
            logger.info(f"  {c['name']:12} {json.dumps(model, sort_keys=True)[:88]}")
        return 0

    if a.dry_run:
        for cfg in configs:
            build_overrides(cfg, seed, str(out_dir / cfg["name"]), a.objective, a.train_data_dir)
        from eeg_cnn_lstm.models.train import train as _check  # noqa: F401

        logger.info(f"Built {len(configs)} override trees; all valid.")
        return 0

    from eeg_cnn_lstm.models.train import train as train_fn

    runs = run_all(configs, seed, a.config, out_dir, a.objective, a.train_data_dir, train_fn)
    if not runs:
        logger.error("No runs completed.")
        return 1
    print("\n".join(report(runs, out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
