"""
@file search_space.py
@brief The HPO search space, and the reasoning that sets each range.

@details
Phase 3. The space is deliberately asymmetric. Step 2 produced two findings
that decide where the trial budget should go:

  1. **Overfitting begins at the end of the first pass.** Training error falls
     monotonically while tuning performance peaks around pass 2 and collapses
     by pass 5. That is the signature of a model with more capacity than 1,663
     patients support, so the parameters that change the capacity-to-data ratio
     get wide ranges and most of the budget.

  2. **Reducing the learning rate did not slow the memorizing.** After two
     halvings the model never beat its earlier peak. Learning rate therefore
     gets a narrow range around the value already known to work (1e-3): the
     search refines it rather than exploring it, and the trials saved go to
     regularization.

The second asymmetry is toward sensitivity. `pos_weight` above 1 scales the
loss on abnormal recordings, buying recall at the cost of precision, which is
the trade a triage tool wants; it is searched over a range that is strictly
weighted toward recall rather than centred on 1.

Model capacity is searched as a regularizer, not only as a cost. Narrower and
shallower models memorize less, and they also serve the lightweight deployment
question, so the two goals point the same way here instead of trading off.

@par Weighting with TPE
Optuna's sampler has no notion of a parameter being "more important", so the
weighting is expressed three ways instead:
  - **Range width.** Regularization spans orders of magnitude; the learning
    rate spans one factor of ten.
  - **Resolution.** Capacity parameters are small categorical sets, so trials
    are not spent distinguishing 96 hidden units from 100.
  - **Startup coverage.** `STARTUP_TRIALS` are enqueued before the sampler
    takes over, chosen to bracket the regularization axis from barely
    regularized to heavily regularized, so TPE's first model of the objective
    is built on points that actually separate on that axis.

@par Adding a parameter
Append to `SEARCH_SPACE`. `path` is a dotted location in the YAML config, so
`model.dropout` sets `config["model"]["dropout"]`, which `build_model_config`
validates against the real `ModelConfig` fields. A name that does not exist
there raises at trial setup rather than being silently ignored -- run
`scripts/hpo_verify.py` to surface that before a long run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

## @brief Parameter groups, used only for reporting how the budget was spent.
GROUPS = ("regularization", "capacity", "optimization", "sensitivity")


@dataclass(frozen=True)
class Param:
    """
    @brief One searchable hyperparameter.

    @param name Optuna's name for it; must be unique.
    @param path Dotted path into the config, e.g. `"train.lr"`.
    @param kind `"float"`, `"int"` or `"categorical"`.
    @param low Lower bound (float/int only).
    @param high Upper bound (float/int only).
    @param log Sample on a log scale (float/int only).
    @param step Grid step (int only).
    @param choices Allowed values (categorical only).
    @param group One of `GROUPS`, for the budget report.
    @param why One line on why this range; printed by the verifier so the
           rationale travels with the code rather than living in a slide.
    """

    name: str
    path: str
    kind: Literal["float", "int", "categorical"]
    group: str
    why: str
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[int] = None
    choices: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: group must be one of {GROUPS}")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: categorical needs choices")
        elif self.low is None or self.high is None:
            raise ValueError(f"{self.name}: {self.kind} needs low and high")
        elif self.low >= self.high:
            raise ValueError(f"{self.name}: low must be < high")

    def suggest(self, trial: Any) -> Any:
        """
        @brief Draw a value for this parameter from an Optuna trial.

        @param trial An `optuna.Trial`.
        @return The sampled value.
        """
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.kind == "int":
            return trial.suggest_int(
                self.name,
                int(self.low),
                int(self.high),
                step=self.step or 1,
                log=self.log,
            )
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)


## @brief The search space. Order is presentation only; TPE is order-free.
SEARCH_SPACE: tuple[Param, ...] = (
    # ---- Regularization: the widest ranges, because this is the diagnosis ----
    Param(
        "dropout",
        "model.dropout",
        "float",
        "regularization",
        "Overfitting starts at pass 1; the default was too weak to be the "
        "answer, so the range spans near-none to aggressive.",
        low=0.05,
        high=0.6,
    ),
    Param(
        "weight_decay",
        "train.weight_decay",
        "float",
        "regularization",
        "AdamW decouples decay from the adaptive step, so this is a true L2 "
        "penalty whose magnitude is meaningful. Four orders of magnitude.",
        low=1e-5,
        high=1e-1,
        log=True,
    ),
    Param(
        "segments_per_recording",
        "loader.max_epochs_per_recording",
        "int",
        "regularization",
        "Controls how much each patient contributes. More segments is more "
        "data but more CORRELATED data, so this trades coverage against "
        "memorizing individual patients. Probably the most underrated knob.",
        low=10,
        high=60,
        step=5,
    ),
    # ---- Capacity: regularizer and deployment cost at once ----
    Param(
        "lstm_hidden",
        "model.lstm_hidden",
        "categorical",
        "capacity",
        "Shrinking the recurrent state limits what can be memorized and "
        "directly cuts parameters. Coarse set: 96 vs 100 is not a question.",
        choices=[32, 64, 128, 192, 256],
    ),
    Param(
        "lstm_layers",
        "model.lstm_layers",
        "categorical",
        "capacity",
        "A second recurrent layer roughly doubles recurrent parameters; worth "
        "asking whether it earns them.",
        choices=[1, 2],
    ),
    Param(
        "conv_width_mult",
        "model.conv_width_mult",
        "categorical",
        "capacity",
        "Scales every conv block's channel count. Width is a smooth capacity "
        "dial, unlike depth, which changes the receptive field and is run as "
        "a separate ablation.",
        choices=[0.5, 0.75, 1.0, 1.5],
    ),
    Param(
        "downsample_factor",
        "model.downsample_factor",
        "categorical",
        "capacity",
        "How much the convolutional stack reduces the sequence before the "
        "LSTM. More reduction is a shorter sequence, fewer parameters, and a "
        "coarser view of time.",
        choices=[2, 4, 8],
    ),
    # ---- Optimization: narrow, because the evidence says it is not the problem ----
    Param(
        "lr",
        "train.lr",
        "float",
        "optimization",
        "1e-3 already works and mid-run reduction did not help, so this "
        "refines a known-good region rather than exploring. Note the evidence "
        "is against mid-run REDUCTION; a different initial rate is untested, "
        "which is why it stays in the search at all.",
        low=3e-4,
        high=3e-3,
        log=True,
    ),
    # ---- Sensitivity: asymmetric on purpose ----
    Param(
        "pos_weight",
        "train.pos_weight",
        "float",
        "sensitivity",
        "Scales the loss on abnormal recordings. The range starts at 1.0 and "
        "goes up only: a triage tool should never be pushed toward missing "
        "positives, so weights below 1 are excluded by construction.",
        low=1.0,
        high=4.0,
        log=True,
    ),
)

## @brief Points enqueued before TPE takes over, bracketing the regularization axis.
#
#  Random startup sampling can leave the regularization axis poorly covered,
#  and TPE's first surrogate is then built on points that do not separate on
#  the dimension the search most needs to resolve. These five do separate:
#  barely regularized through heavily regularized, plus one small-capacity
#  point and one high-sensitivity point.
STARTUP_TRIALS: tuple[dict[str, Any], ...] = (
    # The Step 2 baseline, so the search has the current model as a reference.
    {
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "segments_per_recording": 30,
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "conv_width_mult": 1.0,
        "downsample_factor": 4,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Barely regularized: should overfit hard, anchoring the low end.
    {
        "dropout": 0.05,
        "weight_decay": 1e-5,
        "segments_per_recording": 60,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "conv_width_mult": 1.5,
        "downsample_factor": 2,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Heavily regularized: the opposite anchor.
    {
        "dropout": 0.55,
        "weight_decay": 5e-2,
        "segments_per_recording": 10,
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "conv_width_mult": 0.5,
        "downsample_factor": 8,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Small capacity, light explicit regularization: separates the two routes
    # to the same effect, which the search would otherwise confound.
    {
        "dropout": 0.1,
        "weight_decay": 1e-4,
        "segments_per_recording": 30,
        "lstm_hidden": 32,
        "lstm_layers": 1,
        "conv_width_mult": 0.5,
        "downsample_factor": 4,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Sensitivity-weighted, moderate regularization.
    {
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "segments_per_recording": 30,
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "conv_width_mult": 1.0,
        "downsample_factor": 4,
        "lr": 1e-3,
        "pos_weight": 2.5,
    },
)


def set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    """
    @brief Set a dotted path in a nested dict, creating sections as needed.

    @param config Dict to modify in place.
    @param path Dotted path, e.g. `"model.dropout"`.
    @param value Value to set.
    @throws ValueError If a path segment exists but is not a dict.
    """
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        nxt = node.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"cannot descend into {path!r}: {part!r} is not a section")
        node = nxt
    node[parts[-1]] = value


def params_to_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """
    @brief Turn sampled parameter values into a nested config override dict.

    @param params Mapping of `Param.name` to value.
    @return Nested partial config for `train(config_overrides=...)`.
    @throws KeyError If a name is not in `SEARCH_SPACE`.
    """
    by_name = {p.name: p for p in SEARCH_SPACE}
    out: dict[str, Any] = {}
    for name, value in params.items():
        if name not in by_name:
            raise KeyError(f"{name!r} is not in SEARCH_SPACE")
        set_by_path(out, by_name[name].path, value)
    return out


def suggest_all(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    @brief Sample every parameter and build the config overrides.

    @param trial An `optuna.Trial`.
    @return Tuple `(params, overrides)`.
    """
    params = {p.name: p.suggest(trial) for p in SEARCH_SPACE}
    return params, params_to_overrides(params)


def describe() -> str:
    """
    @brief Human-readable table of the space, grouped, with the rationale.
    @return Multi-line string.
    """
    lines = [f"Search space: {len(SEARCH_SPACE)} parameters"]
    for group in GROUPS:
        members = [p for p in SEARCH_SPACE if p.group == group]
        if not members:
            continue
        lines.append(f"\n  [{group}]  {len(members)} parameter(s)")
        for p in members:
            if p.kind == "categorical":
                rng = f"choices={list(p.choices)}"
            else:
                scale = " (log)" if p.log else ""
                rng = f"{p.low:g} .. {p.high:g}{scale}"
            lines.append(f"    {p.name:24} -> {p.path:34} {rng}")
            lines.append(f"      {p.why}")
    return "\n".join(lines)
