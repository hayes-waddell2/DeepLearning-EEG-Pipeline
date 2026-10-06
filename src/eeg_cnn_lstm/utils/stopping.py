"""
@file stopping.py
@brief Early stopping and learning-rate scheduling for validation-check cadence.

@details
Phase 2 / Step 2. The A/B runs showed this model overfits within a handful of
epochs: recording-level partial AUC peaked at epoch 4 (0.8212) and collapsed to
0.7005 by epoch 5, while training loss kept falling (0.377 -> 0.253). Selecting
the best epoch after the fact salvages the score but wastes compute and leaves
the stopping point unjustified. This module makes stopping a decision the run
takes for itself.

Two ideas drive the design:

  1. **Checks, not epochs.** One epoch over pool E is a long time to be blind.
     Validation runs several times per epoch, and every quantity below
     (patience, scheduler patience, the recorded curve) is counted in
     *validation checks*, not epochs. Measured on this corpus a validation
     pass over pool T costs ~0.3 min against a ~2 min training epoch, so four
     checks per epoch add ~60% to epoch wall time and buy ~16 observations
     across the ~4-epoch usable window. Because patience is counted in checks
     it must be rescaled whenever the cadence changes: 12 checks at
     `val_every_frac = 0.25` expresses the same "3 epochs of no progress" as
     6 checks at 0.5.

  2. **Two notions of "best".** A score can be good enough to keep the weights
     but not good enough to count as progress. `min_delta` separates them:
     `is_record` (strictly better than anything seen; save the checkpoint) and
     `improved` (better by more than `min_delta`; reset the patience counter).
     Conflating them lets a run drift forward on noise forever.

The scheduler shares the same clock. `ReduceLROnPlateau` is stepped once per
check with `threshold_mode="abs"` and `threshold=min_delta`, so "no progress"
means the same thing to the scheduler and to the stopper. Its patience must be
strictly smaller than the stopper's or the run halts before a reduced learning
rate has any chance to help; `build_scheduler` warns when it is not.

@par Usage:
@verbatim
stopper = EarlyStopping(patience=12, min_delta=0.002, mode="max")
scheduler = build_scheduler(optimizer, {"kind": "plateau", "patience": 4},
                            mode="max", min_delta=0.002,
                            stop_patience=stopper.patience)

for score in validation_checks:          # several per epoch
    res = stopper.update(score)
    if res.is_record:
        torch.save(model.state_dict(), best_path)
    if scheduler is not None:
        scheduler.step(score)
    if res.should_stop:
        break
@endverbatim
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from loguru import logger

## @brief Comparison directions accepted by `EarlyStopping` and `build_scheduler`.
MODES = ("max", "min")

## @brief Scheduler kinds accepted by `build_scheduler`.
SCHEDULER_KINDS = ("plateau", "none")


@dataclass(frozen=True)
class CheckResult:
    """
    @brief Outcome of a single validation check.

    @details
    `is_record` and `improved` are deliberately distinct. The first governs the
    checkpoint (keep the genuinely best weights, however small the margin); the
    second governs patience (only a margin larger than `min_delta` counts as
    progress). A run that creeps upward by 0.0001 per check should keep its
    checkpoint current and still be stopped.
    """

    ## @brief Score passed to `update`.
    score: float
    ## @brief Strictly better than every previous score: save the checkpoint.
    is_record: bool
    ## @brief Better than the reference by more than `min_delta`: reset patience.
    improved: bool
    ## @brief Consecutive checks without a `min_delta` improvement.
    n_bad: int
    ## @brief 1-based index of this check.
    n_checks: int
    ## @brief True once patience is exhausted (and `min_checks` is satisfied).
    should_stop: bool
    ## @brief Human-readable reason, set only when `should_stop` is True.
    reason: Optional[str] = None


@dataclass
class EarlyStopping:
    """
    @brief Patience-based early stopping measured in validation checks.

    @details
    Keras semantics, with the record/improvement split described above. A NaN
    score never improves and never sets a record, so a diverged run burns
    patience and stops rather than poisoning the reference with NaN.

    @param patience Consecutive non-improving checks tolerated before stopping.
    @param min_delta Minimum absolute change that counts as an improvement.
    @param mode `"max"` for metrics where larger is better (AUC, pAUC),
           `"min"` for loss.
    @param min_checks Checks that must elapse before stopping is permitted.
           A warm-up guard for noisy early validation; 0 disables it.
    """

    patience: int = 12
    min_delta: float = 0.01
    mode: str = "max"
    min_checks: int = 0

    ## @brief Reference score for the `min_delta` comparison.
    best: float = field(init=False)
    ## @brief Best score seen at any margin (what the checkpoint tracks).
    best_record: float = field(init=False)
    ## @brief Check index that set `best_record`.
    best_check: int = field(init=False, default=0)
    ## @brief Number of checks seen so far.
    n_checks: int = field(init=False, default=0)
    ## @brief Consecutive non-improving checks.
    n_bad: int = field(init=False, default=0)
    ## @brief Latch: True once the stop condition has fired.
    should_stop: bool = field(init=False, default=False)
    ## @brief Reason the run stopped, or None.
    reason: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}; got {self.mode!r}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1; got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0; got {self.min_delta}")
        worst = -math.inf if self.mode == "max" else math.inf
        self.best = worst
        self.best_record = worst

    def _beats(self, score: float, reference: float, delta: float) -> bool:
        """
        @brief Direction-aware comparison with a margin.

        @param score Candidate score.
        @param reference Score to beat.
        @param delta Required margin (0 for a strict comparison).
        @return True if `score` beats `reference` by more than `delta`.
        """
        if math.isnan(score):
            return False
        if self.mode == "max":
            return score > reference + delta
        return score < reference - delta

    def update(self, score: float) -> CheckResult:
        """
        @brief Record one validation check and decide whether to stop.

        @param score The selection metric at this check.
        @return A `CheckResult`; act on `is_record` and `should_stop`.
        """
        self.n_checks += 1
        score = float(score)

        is_record = self._beats(score, self.best_record, 0.0)
        if is_record:
            self.best_record = score
            self.best_check = self.n_checks

        improved = self._beats(score, self.best, self.min_delta)
        if improved:
            self.best = score
            self.n_bad = 0
        else:
            self.n_bad += 1
            if self.n_bad >= self.patience and self.n_checks >= self.min_checks:
                self.should_stop = True
                self.reason = (
                    f"no improvement > {self.min_delta:g} for {self.n_bad} "
                    f"checks (best {self.best_record:.4f} at check "
                    f"{self.best_check})"
                )

        return CheckResult(
            score=score,
            is_record=is_record,
            improved=improved,
            n_bad=self.n_bad,
            n_checks=self.n_checks,
            should_stop=self.should_stop,
            reason=self.reason,
        )

    def state(self) -> dict[str, Any]:
        """
        @brief Serializable summary for the run record.
        @return Dict of configuration and final counters.
        """
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "min_checks": self.min_checks,
            "n_checks": self.n_checks,
            "n_bad": self.n_bad,
            "best_score": None if math.isinf(self.best_record) else self.best_record,
            "best_check": self.best_check,
            "stopped_early": self.should_stop,
            "reason": self.reason,
        }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Optional[dict[str, Any]],
    mode: str = "max",
    min_delta: float = 0.002,
    stop_patience: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    @brief Build the validation-check learning-rate scheduler, or None.

    @details
    Only `ReduceLROnPlateau` is supported, deliberately: a schedule indexed by
    epoch (cosine, step) is meaningless when the number of epochs is decided at
    runtime by early stopping. The plateau scheduler is driven by the same
    score and the same `min_delta` as the stopper, so the two agree on what
    "no progress" means.

    `threshold_mode="abs"` matters. The default `"rel"` scales the threshold by
    the current best, which for a metric already near 0.8 makes the effective
    margin ~0.0016 rather than the configured value, and makes the scheduler's
    sensitivity depend on where the metric happens to sit.

    @param optimizer Optimizer whose learning rate will be reduced.
    @param cfg Scheduler config dict, or None / `{"enabled": false}` to disable.
           Keys: `kind` ("plateau"|"none"), `factor`, `patience` (in checks),
           `min_lr`, `cooldown`.
    @param mode `"max"` or `"min"`; must match the stopper.
    @param min_delta Absolute improvement threshold, shared with the stopper.
    @param stop_patience The stopper's patience, for the ordering warning.
    @return The scheduler, or None when disabled.
    """
    if not cfg or not cfg.get("enabled", True):
        return None

    kind = str(cfg.get("kind", "plateau")).lower()
    if kind not in SCHEDULER_KINDS:
        raise ValueError(
            f"lr_schedule.kind must be one of {SCHEDULER_KINDS}; got {kind!r}"
        )
    if kind == "none":
        return None
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")

    patience = int(cfg.get("patience", 4))
    factor = float(cfg.get("factor", 0.5))
    min_lr = float(cfg.get("min_lr", 1e-6))
    cooldown = int(cfg.get("cooldown", 0))

    if not 0.0 < factor < 1.0:
        raise ValueError(f"lr_schedule.factor must be in (0, 1); got {factor}")

    if stop_patience is not None and patience >= stop_patience:
        logger.warning(
            f"lr_schedule.patience ({patience}) >= early_stopping.patience "
            f"({stop_patience}): the run will stop before a reduced learning "
            "rate can take effect. Set the scheduler patience lower."
        )

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=min_delta,
        threshold_mode="abs",
        cooldown=cooldown,
        min_lr=min_lr,
    )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    @brief Learning rate of the optimizer's first parameter group.

    @param optimizer Optimizer to read.
    @return The current learning rate.
    """
    return float(optimizer.param_groups[0]["lr"])