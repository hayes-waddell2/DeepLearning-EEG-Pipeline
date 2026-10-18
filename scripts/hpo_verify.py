"""
@file hpo_verify.py
@brief Prove the HPO machinery works before committing ten GPU-hours to it.

@details
Phase 3. A hyperparameter search is the worst place to discover a bug: the
failure shows up hours in, overnight, with nobody watching, and the usual
symptom is not a crash but a quietly wrong result -- a parameter that never
reached the model, a pruner that killed every trial, a study that was never
written to disk. This script turns each of those into a check that runs in
seconds.

Two modes, meant to be run in order.

**`--mode offline`** (seconds, CPU, no data, no GPU). Replaces the training
function with a cheap deterministic stand-in and runs the real driver against
it, so the plumbing is exercised without the compute: parameter sampling and
bounds, nested config override construction, the pruner actually pruning,
SQLite persistence, resume after a simulated crash, survival of a failing
trial, the wall-clock timeout, and top-k export. It also validates every
searched parameter name against the real `ModelConfig`, which is the single
most likely silent failure -- a typo there costs the whole run.

**`--mode gpu`** (a few minutes, one GPU, real data). Runs two short trials
with capped steps through the real training function, proving the integration
that the offline mode stubs out: that configs build, data loads, the model
accepts every searched parameter, scores come back finite, and the pruning
hook fires at the real cadence.

Exit code is 0 only if every check passes, so this drops into a job script
ahead of the search:

@par Usage:
@verbatim
python -m scripts.hpo_verify --mode offline          # always, first
python -m scripts.hpo_verify --mode gpu --config configs/step2_earlystop.yaml
python -m scripts.hpo_verify --mode both --config configs/step2_earlystop.yaml
@endverbatim
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from optuna.trial import TrialState

from eeg_cnn_lstm.utils.search_space import (
    SEARCH_SPACE,
    STARTUP_TRIALS,
    describe,
    params_to_overrides,
    set_by_path,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

## @brief Pass/fail marks.
OK, BAD = "  PASS", "  FAIL"


class Checks:
    """
    @brief Tiny check recorder, so one failure does not hide the rest.

    @details
    Every check runs even after an earlier one fails: when a search is broken
    it is usually broken in more than one way, and a single run should surface
    all of it rather than making the user re-run after each fix.
    """

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, fn: Callable[[], Any]) -> Any:
        """
        @brief Run one check, recording the outcome.

        @param name Human-readable check name.
        @param fn Callable; raising (including `AssertionError`) means failure.
        @return The callable's return value, or None on failure.
        """
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - reporting every failure is the point
            detail = f"{type(exc).__name__}: {exc}"
            self.failed.append((name, detail))
            print(f"{BAD}  {name}\n          {detail}")
            if not isinstance(exc, AssertionError):
                tb = traceback.format_exc(limit=3).splitlines(True)
                print("".join("          " + line for line in tb))
            return None
        self.passed += 1
        print(f"{OK}  {name}")
        return value

    def summary(self) -> int:
        """
        @brief Print the tally.
        @return 0 if everything passed, 1 otherwise.
        """
        total = self.passed + len(self.failed)
        print(f"\n{self.passed}/{total} checks passed")
        if self.failed:
            print("\nFailures:")
            for name, detail in self.failed:
                print(f"  - {name}: {detail}")
            return 1
        return 0


# ---------------------------------------------------------------------------
# Offline mode: the driver, against a stand-in trainer
# ---------------------------------------------------------------------------


def fake_train_factory(
    behavior: str = "normal",
    n_checks: int = 24,
) -> Callable[..., dict[str, Any]]:
    """
    @brief Build a stand-in for `train()` that returns in microseconds.

    @details
    Mimics the parts of the real contract the driver depends on: it calls
    `on_check` repeatedly with a rising-then-falling score (the shape the real
    runs show), honours a True return by stopping, lets `TrialPruned`
    propagate, and returns a `bench` dict with the keys the objective reads.

    The score depends on the sampled parameters in a crude but monotone way,
    so TPE has a real signal to optimize and the "search improves over random"
    check means something.

    @param behavior `"normal"`, `"fail"` (raises mid-trial) or `"nan"`
           (returns a non-finite score).
    @param n_checks How many validation checks to simulate.
    @return A callable with `train()`'s keyword interface.
    """

    def fake_train(
        config_path: str,
        output_dir: Optional[str] = None,
        num_epochs: Optional[int] = None,
        max_steps: int = 0,
        on_check: Optional[Callable[[int, float], bool]] = None,
        config_overrides: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        cfg = config_overrides or {}
        model = cfg.get("model", {})
        train_c = cfg.get("train", {})
        loader_c = cfg.get("loader", {})

        if behavior == "fail":
            raise RuntimeError("simulated trial failure")

        # A toy objective with an interior optimum, so the search has
        # something real to find: moderate dropout and decay are best.
        dropout = float(model.get("dropout", 0.2))
        wd = float(train_c.get("weight_decay", 1e-4))
        hidden = float(model.get("lstm_hidden", 128))
        segs = float(loader_c.get("max_epochs_per_recording", 30))
        peak = (
            0.80
            - 2.2 * (dropout - 0.30) ** 2
            - 0.02 * (math.log10(wd) + 3.0) ** 2
            - 0.00002 * abs(hidden - 128)
            - 0.00004 * abs(segs - 30)
        )
        if behavior == "nan":
            peak = float("nan")

        best = -float("inf")
        for i in range(1, n_checks + 1):
            # Rise to the peak around check 8, then decay: the real shape.
            frac = min(i / 8.0, 1.0)
            score = peak * (0.82 + 0.18 * frac) - 0.004 * max(0, i - 8)
            best = max(best, score) if math.isfinite(score) else best
            if on_check is not None and on_check(i, score):
                break

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "benchmark.json").write_text("{}")

        return {
            "best": {"score": best, "check": 8, "epoch_frac": 2.0,
                     "recording_metrics": {
                         "auc_roc": best + 0.09, "pauc_fpr0.1": best - 0.01,
                         "pauc_tpr0.9": best, "sens_at_spec0.9": best - 0.03,
                         "spec_at_sens0.95": best - 0.27}},
            "stopping": {"epochs_run": 5, "checks_run": n_checks,
                         "stopped_early": True},
            "n_parameters": int(1e5 + hidden * 800),
        }

    return fake_train


def run_offline(checks: Checks, keep: bool = False) -> None:
    """
    @brief Exercise the driver end to end without a GPU.

    @param checks Recorder to report into.
    @param keep Keep the temporary working directory for inspection.
    """
    from scripts.hpo import FAILED_SCORE, enqueue_startup, make_objective, make_study, report

    tmp = Path(tempfile.mkdtemp(prefix="hpo_verify_"))
    db = f"sqlite:///{tmp / 'study.db'}"
    print(f"\n[offline] workspace: {tmp}\n")

    # --- 1. the search space itself -------------------------------------
    def space_is_well_formed() -> None:
        names = [p.name for p in SEARCH_SPACE]
        assert len(names) == len(set(names)), "duplicate parameter names"
        paths = [p.path for p in SEARCH_SPACE]
        assert len(paths) == len(set(paths)), "two parameters write the same path"
        for p in SEARCH_SPACE:
            assert p.why.strip(), f"{p.name} has no rationale"
    checks.check("search space is well formed", space_is_well_formed)

    def model_params_exist() -> None:
        """The silent killer: a model.* name that ModelConfig does not have."""
        try:
            from eeg_cnn_lstm.models.model import ModelConfig
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"cannot import ModelConfig ({exc!r}); run this from the repo "
                "root with the package installed (pip install -e .)"
            ) from exc
        fields = {f.name for f in dataclasses.fields(ModelConfig)}
        wanted = {p.path.split(".", 1)[1] for p in SEARCH_SPACE
                  if p.path.startswith("model.")}
        missing = sorted(wanted - fields)
        assert not missing, (
            f"ModelConfig has no field(s) {missing}. These would be sampled "
            f"every trial and silently ignored. Available: {sorted(fields)}"
        )
    checks.check("every model.* parameter exists on ModelConfig", model_params_exist)

    def startup_points_valid() -> None:
        names = {p.name for p in SEARCH_SPACE}
        for i, point in enumerate(STARTUP_TRIALS):
            extra = sorted(set(point) - names)
            assert not extra, f"startup point {i} sets unknown {extra}"
            for p in SEARCH_SPACE:
                if p.name not in point:
                    continue
                v = point[p.name]
                if p.kind == "categorical":
                    assert v in p.choices, f"startup {i}: {p.name}={v} not in choices"
                else:
                    assert p.low <= v <= p.high, f"startup {i}: {p.name}={v} out of range"
    checks.check("startup points are inside the search space", startup_points_valid)

    def overrides_nest_correctly() -> None:
        ov = params_to_overrides({p.name: (p.choices[0] if p.kind == "categorical"
                                           else p.low) for p in SEARCH_SPACE})
        assert "train" in ov and "model" in ov and "loader" in ov, ov
        assert ov["train"]["lr"] == 3e-4, ov["train"]
        assert ov["loader"]["max_epochs_per_recording"] == 10, ov["loader"]
        nested: dict[str, Any] = {}
        set_by_path(nested, "a.b.c", 1)
        assert nested == {"a": {"b": {"c": 1}}}
    checks.check("parameters map to the right nested config paths", overrides_nest_correctly)

    # --- 2. sampling and bounds -----------------------------------------
    def sampling_respects_bounds() -> None:
        study = make_study(None, "bounds", 0, 5, "none", 4)
        study.optimize(make_objective(
            "unused.yaml", tmp / "bounds", "recording_pauc_sens",
            fake_train_factory()), n_trials=25)
        for t in study.get_trials(deepcopy=False):
            for p in SEARCH_SPACE:
                v = t.params[p.name]
                if p.kind == "categorical":
                    assert v in p.choices, f"{p.name}={v} outside choices"
                else:
                    assert p.low <= v <= p.high, f"{p.name}={v} outside [{p.low},{p.high}]"
                if p.kind == "int":
                    assert float(v).is_integer(), f"{p.name}={v} not an integer"
    checks.check("sampled values stay inside every declared range", sampling_respects_bounds)

    def pos_weight_never_below_one() -> None:
        study = make_study(None, "posw", 1, 5, "none", 4)
        study.optimize(make_objective(
            "unused.yaml", tmp / "posw", "recording_pauc_sens",
            fake_train_factory()), n_trials=20)
        lows = [t.params["pos_weight"] for t in study.get_trials(deepcopy=False)]
        assert min(lows) >= 1.0, f"pos_weight sampled below 1.0: {min(lows)}"
    checks.check("pos_weight is never sampled below 1.0 (sensitivity guard)",
                 pos_weight_never_below_one)

    # --- 3. pruning -------------------------------------------------------
    def pruner_prunes() -> None:
        study = make_study(None, "prune", 2, 4, "median", 3)
        study.optimize(make_objective(
            "unused.yaml", tmp / "prune", "recording_pauc_sens",
            fake_train_factory()), n_trials=30)
        states = [t.state for t in study.get_trials(deepcopy=False)]
        n_pruned = sum(s == TrialState.PRUNED for s in states)
        n_done = sum(s == TrialState.COMPLETE for s in states)
        assert n_pruned > 0, "pruner never fired: the budget will not stretch"
        assert n_done > 0, "pruner killed everything: warmup is too aggressive"
    checks.check("pruner kills some trials but not all", pruner_prunes)

    def warmup_is_respected() -> None:
        """A trial must get its warmup checks before it can be pruned."""
        seen: list[int] = []
        base = fake_train_factory()

        def recording_train(*a: Any, **kw: Any) -> dict[str, Any]:
            hook = kw.get("on_check")
            count = {"n": 0}

            def wrapped(check_no: int, score: float) -> bool:
                count["n"] = check_no
                try:
                    return hook(check_no, score) if hook else False
                except optuna.TrialPruned:
                    seen.append(count["n"])
                    raise
            kw["on_check"] = wrapped
            return base(*a, **kw)

        study = make_study(None, "warm", 3, 4, "median", 6)
        study.optimize(make_objective(
            "unused.yaml", tmp / "warm", "recording_pauc_sens",
            recording_train), n_trials=30)
        if seen:
            assert min(seen) >= 6, f"pruned at check {min(seen)}, before warmup 6"
    checks.check("no trial is pruned before its warmup checks", warmup_is_respected)

    # --- 4. persistence and resume ---------------------------------------
    def storage_persists() -> None:
        study = make_study(db, "resume-test", 4, 5, "none", 4)
        enqueue_startup(study)
        study.optimize(make_objective(
            "unused.yaml", tmp / "resume", "recording_pauc_sens",
            fake_train_factory()), n_trials=8)
        assert (tmp / "study.db").exists(), "no SQLite file written"
        del study
        reopened = make_study(db, "resume-test", 4, 5, "none", 4)
        n = len(reopened.get_trials(deepcopy=False))
        assert n >= 8, f"resumed study has {n} trials, expected >= 8"
        reopened.optimize(make_objective(
            "unused.yaml", tmp / "resume", "recording_pauc_sens",
            fake_train_factory()), n_trials=3)
        assert len(reopened.get_trials(deepcopy=False)) >= n + 3, "resume added no trials"
    checks.check("study survives process death and resumes", storage_persists)

    def startup_not_repeated_on_resume() -> None:
        study = make_study(db, "resume-test", 4, 5, "none", 4)
        added = enqueue_startup(study)
        assert added == 0, f"re-enqueued {added} startup point(s) already run"
    checks.check("resume does not re-run the startup points", startup_not_repeated_on_resume)

    # --- 5. failure handling ----------------------------------------------
    def one_bad_trial_does_not_stop_the_study() -> None:
        calls = {"n": 0}
        good, bad = fake_train_factory(), fake_train_factory("fail")

        def flaky(*a: Any, **kw: Any) -> dict[str, Any]:
            calls["n"] += 1
            return bad(*a, **kw) if calls["n"] % 4 == 0 else good(*a, **kw)

        study = make_study(None, "flaky", 5, 3, "none", 4)
        study.optimize(make_objective(
            "unused.yaml", tmp / "flaky", "recording_pauc_sens", flaky), n_trials=12)
        done = [t for t in study.get_trials(deepcopy=False)
                if t.state == TrialState.COMPLETE]
        assert len(done) == 12, "study aborted on a failing trial"
        penalized = [t for t in done if t.value is not None and t.value <= FAILED_SCORE]
        assert penalized, "failing trials were not penalized"
    checks.check("a failing trial is penalized, not fatal", one_bad_trial_does_not_stop_the_study)

    def nan_scores_are_handled() -> None:
        study = make_study(None, "nan", 6, 3, "median", 2)
        study.optimize(make_objective(
            "unused.yaml", tmp / "nan", "recording_pauc_sens",
            fake_train_factory("nan")), n_trials=6)
        assert len(study.get_trials(deepcopy=False)) == 6, "NaN scores aborted the study"
    checks.check("non-finite scores do not abort the study", nan_scores_are_handled)

    # --- 6. the search actually searches ----------------------------------
    def tpe_beats_random() -> None:
        """TPE should find better points than random on a space with an optimum."""
        from optuna.samplers import RandomSampler
        obj = make_objective("unused.yaml", tmp / "cmp", "recording_pauc_sens",
                             fake_train_factory())
        rnd = optuna.create_study(direction="maximize", sampler=RandomSampler(seed=7))
        rnd.optimize(obj, n_trials=40)
        tpe = make_study(None, "tpe", 7, 10, "none", 4)
        tpe.optimize(obj, n_trials=40)
        assert tpe.best_value >= rnd.best_value - 1e-9, (
            f"TPE {tpe.best_value:.4f} did not match random {rnd.best_value:.4f}"
        )
    checks.check("TPE finds at least as good an optimum as random search", tpe_beats_random)

    def timeout_stops_cleanly() -> None:
        slow = fake_train_factory()

        def slow_train(*a: Any, **kw: Any) -> dict[str, Any]:
            time.sleep(0.12)
            return slow(*a, **kw)

        study = make_study(None, "timeout", 8, 3, "none", 4)
        t0 = time.perf_counter()
        study.optimize(make_objective(
            "unused.yaml", tmp / "timeout", "recording_pauc_sens", slow_train),
            n_trials=10_000, timeout=1.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 6.0, f"timeout overran: {elapsed:.1f}s"
        assert len(study.get_trials(deepcopy=False)) > 0, "timeout produced no trials"
    checks.check("wall-clock timeout stops the study cleanly", timeout_stops_cleanly)

    # --- 7. the report the next stage consumes -----------------------------
    def report_exports_topk() -> None:
        study = make_study(None, "report", 9, 5, "none", 4)
        study.optimize(make_objective(
            "unused.yaml", tmp / "report", "recording_pauc_sens",
            fake_train_factory()), n_trials=12)
        out = tmp / "report_out"
        report(study, 5, out)
        assert (out / "top_k.json").is_file(), "top_k.json not written"
        assert (out / "trials.csv").is_file(), "trials.csv not written"
        loaded = json.loads((out / "top_k.json").read_text())
        assert len(loaded["top"]) == 5, f"expected 5 entries, got {len(loaded['top'])}"
        scores = [r["score"] for r in loaded["top"]]
        assert scores == sorted(scores, reverse=True), "top-k is not ranked"
        names = {p.name for p in SEARCH_SPACE}
        assert set(loaded["top"][0]["params"]) == names, "exported params are incomplete"
        csv_head = (out / "trials.csv").read_text().splitlines()[0]
        for p in SEARCH_SPACE:
            assert p.name in csv_head, f"{p.name} missing from trials.csv"
    checks.check("top-k export is complete and correctly ranked", report_exports_topk)

    if keep:
        print(f"\n[offline] kept workspace at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# GPU mode: the real training function, briefly
# ---------------------------------------------------------------------------


def run_gpu(
    checks: Checks,
    config: str,
    max_steps: int,
    out_dir: Path,
    train_data_dir: Optional[str] = None,
) -> None:
    """
    @brief Run two short real trials to prove the training integration.

    @details
    Covers what the offline mode cannot: that the YAML loads, the data is
    reachable, every searched parameter is accepted by the real model, scores
    come back finite, and the pruning hook fires at the real cadence. Steps
    are capped so this costs minutes rather than hours.

    @param checks Recorder to report into.
    @param config Base YAML config path.
    @param max_steps Training steps per epoch.
    @param out_dir Directory for the throwaway run directories.
    @param train_data_dir Override for `data.train_data_dir`, so the checks
           read the same staged copy the search will.
    """
    from scripts.hpo import make_objective, make_study

    print(f"\n[gpu] config: {config}  max_steps: {max_steps}\n")

    def config_exists() -> None:
        assert Path(config).is_file(), f"config not found: {config}"
    checks.check("base config exists", config_exists)

    def extreme_configs_build() -> None:
        """Both corners of the capacity space must construct a real model."""
        from eeg_cnn_lstm.models.model import CNN_LSTM
        from eeg_cnn_lstm.models.train import build_model_config
        for label, point in (("smallest", STARTUP_TRIALS[3]),
                             ("largest", STARTUP_TRIALS[1])):
            ov = params_to_overrides(point)
            cfg = build_model_config(ov.get("model"))
            n = CNN_LSTM(cfg).num_parameters()
            assert n > 0, f"{label} config has no parameters"
            print(f"          {label}: {n:,} parameters")
    checks.check("smallest and largest sampled models both build",
                 extreme_configs_build)

    def real_trials_run() -> None:
        from eeg_cnn_lstm.models.train import train as train_fn
        hook_calls: list[int] = []

        def counting_train(*a: Any, **kw: Any) -> dict[str, Any]:
            hook = kw.get("on_check")

            def wrapped(check_no: int, score: float) -> bool:
                hook_calls.append(check_no)
                assert math.isfinite(score), f"check {check_no} scored {score}"
                return hook(check_no, score) if hook else False
            kw["on_check"] = wrapped
            return train_fn(*a, **kw)

        base = ({"data": {"train_data_dir": train_data_dir}}
                if train_data_dir else None)
        study = make_study(None, "gpu-smoke", 42, 2, "none", 2)
        study.optimize(make_objective(
            config_path=config,
            out_root=out_dir,
            objective_metric="recording_pauc_sens",
            train_fn=counting_train,
            base_overrides=base,
            max_steps=max_steps,
            num_epochs=1,
        ), n_trials=2)

        done = [t for t in study.get_trials(deepcopy=False)
                if t.state == TrialState.COMPLETE]
        assert len(done) == 2, f"only {len(done)}/2 trials completed"
        for t in done:
            assert t.value is not None and math.isfinite(t.value), \
                f"trial {t.number} returned {t.value}"
            assert t.value > -1.0, f"trial {t.number} failed inside train()"
            assert "n_parameters" in t.user_attrs, "run record did not reach the trial"
            assert "pauc_tpr0.9" in t.user_attrs, "high-sensitivity pAUC not reported"
        assert hook_calls, "the validation hook never fired -- pruning would never work"
        print(f"          {len(hook_calls)} validation check(s) across 2 trials; "
              f"scores {done[0].value:.4f}, {done[1].value:.4f}")
    checks.check("two real trials complete with finite scores", real_trials_run)

    def objective_metric_is_the_sensitivity_one() -> None:
        from eeg_cnn_lstm.models.train import SELECT_METRICS
        assert "recording_pauc_sens" in SELECT_METRICS, "objective metric not registered"
        source, key = SELECT_METRICS["recording_pauc_sens"]
        assert source == "recording" and key == "pauc_tpr0.9", (source, key)
    checks.check("objective resolves to the high-sensitivity pAUC",
                 objective_metric_is_the_sensitivity_one)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @param argv Argument list; `None` uses `sys.argv`.
    @return Parsed namespace.
    """
    p = argparse.ArgumentParser(description="Verify the HPO setup before a long run.")
    p.add_argument("--mode", default="offline", choices=["offline", "gpu", "both"])
    p.add_argument("--config", default="configs/step2_earlystop.yaml")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Training steps per epoch in gpu mode")
    p.add_argument("--train-data-dir", default=None,
                   help="Override data.train_data_dir in gpu mode")
    p.add_argument("--out-dir", default="runs/hpo_verify")
    p.add_argument("--keep", action="store_true",
                   help="Keep the offline temporary workspace")
    p.add_argument("--describe", action="store_true",
                   help="Print the search space and exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @param argv Argument list; `None` uses `sys.argv`.
    @return 0 if every check passed, 1 otherwise.
    """
    args = parse_args(argv)
    if args.describe:
        print(describe())
        return 0

    print("=" * 72)
    print("HPO verification")
    print("=" * 72)
    checks = Checks()

    if args.mode in ("offline", "both"):
        print("\n--- offline: driver machinery, no GPU ---")
        run_offline(checks, keep=args.keep)

    if args.mode in ("gpu", "both"):
        print("\n--- gpu: real training, capped steps ---")
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        run_gpu(checks, args.config, args.max_steps, out, args.train_data_dir)

    code = checks.summary()
    print("\n" + ("Ready to launch the search." if code == 0
                  else "Do NOT launch the search until these are fixed."))
    return code


if __name__ == "__main__":
    raise SystemExit(main())