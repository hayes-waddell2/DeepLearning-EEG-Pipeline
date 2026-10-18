"""
@file search_space.py
@brief The HPO search space, and the reasoning that sets each range.

@details
Phase 3. The space is deliberately asymmetric. Step 2 produced two findings
that decide where the trial budget should go:

  1. **Overfitting begins at the end of the first pass.** Training error falls
     monotonically while tuning performance peaks around pass 2 and collapses
     by pass 5. That is the signature of a model with more capacity than 1,663
     patients support, so the parameters that change the capacity-to-data
     ratio get wide ranges and most of the budget. Ten of the twelve
     parameters here are regularization or capacity.

  2. **Reducing the learning rate did not slow the memorizing.** After two
     halvings the model never beat its earlier peak. Learning rate therefore
     gets a narrow range around the value already known to work (1e-3): the
     search refines it rather than exploring it, and the trials saved go to
     regularization.

The second asymmetry is toward sensitivity. `pos_weight` above 1 scales the
loss on abnormal recordings, buying recall at the cost of precision, which is
the trade a triage tool wants; it is searched over a range strictly weighted
toward recall rather than centred on 1.

Model capacity is searched as a regularizer, not only as a cost. Narrower and
shallower models memorize less, and they also serve the lightweight deployment
question, so the two goals point the same way here instead of trading off.

@par Derived parameters
`ModelConfig` takes `conv_channels` and `pool_sizes` as tuples, which are
awkward to search directly: TPE handles a scalar dial far better than a
structured value, and a tuple is unreadable in a results table. So two
parameters are sampled in an interpretable form and transformed into the real
field at trial setup:

  - `conv_width_mult` scales the default `(32, 64, 128)` channel widths.
  - `downsample` is a `"4,4,2"`-style string parsed into `pool_sizes`.

The transform runs in `params_to_overrides`, so what reaches the model is
always a genuine `ModelConfig` field and the verifier's name check still
protects against typos.

@par What is NOT searched, and why
`max_epochs_per_recording` (segments per recording) is a powerful regularizer
and was in an earlier draft of this space. It is excluded because it changes
the number of optimizer steps in an epoch, and the pruner compares trials at
equal *check* numbers. A trial drawing 10 segments per recording would reach
check 12 having seen a fraction of the data of a trial drawing 150, and the
pruner would kill the slower-feeding trial for being behind when it is merely
earlier in training. Fixing it keeps every trial on the same clock. It is
cheap to ablate separately afterward, at the winning configuration.

`conv_kernels` is fixed: kernel width interacts with `pool_sizes` to set the
receptive field, and searching both makes the winner uninterpretable.

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
`model.conv_dropout` sets `config["model"]["conv_dropout"]`, which
`build_model_config` validates against the real `ModelConfig` fields. A name
that does not exist there raises at trial setup rather than being silently
ignored -- run `scripts/hpo_verify.py` to surface that before a long run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

## @brief Parameter groups, used for reporting how the budget was spent.
GROUPS = ("regularization", "capacity", "optimization", "sensitivity")

## @brief Default conv widths, from `ModelConfig`. `conv_width_mult` scales these.
BASE_CONV_CHANNELS: tuple[int, ...] = (32, 64, 128)

## @brief Smallest channel count a width multiplier may produce.
MIN_CHANNELS = 8


def scale_conv_channels(mult: float) -> tuple[int, ...]:
    """
    @brief Scale the default conv channel widths by a multiplier.

    @details
    Keeps the 1:2:4 shape of the default stack while moving its overall
    capacity, so the multiplier stays interpretable: 0.5 is half as wide
    everywhere, not an arbitrary new architecture.

    @param mult Width multiplier.
    @return Channel counts per block, each at least `MIN_CHANNELS`.
    @throws ValueError If `mult` is not positive.
    """
    if mult <= 0:
        raise ValueError(f"conv_width_mult must be positive; got {mult}")
    return tuple(max(MIN_CHANNELS, int(round(c * mult))) for c in BASE_CONV_CHANNELS)


def parse_pool_sizes(spec: str) -> tuple[int, ...]:
    """
    @brief Parse a `"4,4,2"` pooling spec into a `pool_sizes` tuple.

    @details
    Sampled as a string because Optuna's categorical values must be plain
    scalars, and because a string reads clearly in `trials.csv` where a tuple
    would not.

    @param spec Comma-separated pool strides, one per CNN block.
    @return The pool sizes as a tuple of ints.
    @throws ValueError If the spec is malformed or has the wrong block count.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != len(BASE_CONV_CHANNELS):
        raise ValueError(
            f"pool spec {spec!r} has {len(parts)} entries; "
            f"expected {len(BASE_CONV_CHANNELS)} (one per CNN block)"
        )
    try:
        sizes = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"pool spec {spec!r} is not all integers") from exc
    if any(s < 1 for s in sizes):
        raise ValueError(f"pool spec {spec!r} has a stride below 1")
    return sizes


def sequence_length(spec: str, n_timesteps: int = 2500) -> int:
    """
    @brief Length of the sequence the LSTM sees, for a given pooling spec.

    @param spec Comma-separated pool strides.
    @param n_timesteps Input samples per epoch (2500 = 10 s at 250 Hz).
    @return Sequence length after the CNN stack.
    """
    total = 1
    for s in parse_pool_sizes(spec):
        total *= s
    return n_timesteps // total


@dataclass(frozen=True)
class Param:
    """
    @brief One searchable hyperparameter.

    @param name Optuna's name for it; must be unique.
    @param path Dotted path into the config, e.g. `"model.conv_dropout"`.
    @param kind `"float"`, `"int"` or `"categorical"`.
    @param group One of `GROUPS`, for the budget report.
    @param why One line on why this range; printed by the verifier so the
           rationale travels with the code rather than living in a slide.
    @param low Lower bound (float/int only).
    @param high Upper bound (float/int only).
    @param log Sample on a log scale (float/int only).
    @param step Grid step (int only).
    @param choices Allowed values (categorical only).
    @param transform Maps the sampled value to the value written at `path`.
           Lets a parameter be sampled in an interpretable form while the
           model still receives a real field value.
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
    transform: Optional[Callable[[Any], Any]] = None

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
        @return The sampled value, before any transform.
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

    def apply(self, value: Any) -> Any:
        """
        @brief Convert a sampled value into the value written to the config.

        @param value The sampled value.
        @return The transformed value, or `value` unchanged when no transform.
        """
        return self.transform(value) if self.transform else value


## @brief The search space. Order is presentation only; TPE is order-free.
SEARCH_SPACE: tuple[Param, ...] = (
    # ---- Regularization: widest ranges, because this is the diagnosis -------
    Param(
        "conv_dropout",
        "model.conv_dropout",
        "float",
        "regularization",
        "Dropout after each CNN block. Overfitting starts at pass 1, so the "
        "default 0.3 was not enough on its own; the range spans near-none to "
        "aggressive.",
        low=0.05,
        high=0.6,
    ),
    Param(
        "lstm_dropout",
        "model.lstm_dropout",
        "float",
        "regularization",
        "Inter-layer LSTM dropout. Note it only takes effect when "
        "lstm_layers > 1, so roughly half the trials ignore it -- that is a "
        "property of the architecture, not a bug in the space.",
        low=0.0,
        high=0.6,
    ),
    Param(
        "head_dropout",
        "model.head_dropout",
        "float",
        "regularization",
        "Dropout in the MLP head, where the model is closest to the label and "
        "memorizing is easiest.",
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
    # ---- Capacity: regularizer and deployment cost at once ------------------
    Param(
        "conv_width_mult",
        "model.conv_channels",
        "categorical",
        "capacity",
        "Scales the default (32, 64, 128) widths, keeping the 1:2:4 shape. "
        "Width is a smooth capacity dial; depth changes the receptive field "
        "and is run as a separate ablation.",
        choices=[0.5, 0.75, 1.0, 1.5],
        transform=scale_conv_channels,
    ),
    Param(
        "downsample",
        "model.pool_sizes",
        "categorical",
        "capacity",
        "Pooling strides per block; the product sets the sequence length the "
        "LSTM sees (default 4,4,2 = 32x = 78 steps). More pooling is a "
        "shorter sequence, a cheaper LSTM and a coarser view of time.",
        choices=["4,2,2", "4,4,2", "4,4,4", "5,5,4"],
        transform=parse_pool_sizes,
    ),
    Param(
        "lstm_hidden",
        "model.lstm_hidden",
        "categorical",
        "capacity",
        "Hidden size per direction. Shrinking the recurrent state limits what "
        "can be memorized and cuts parameters directly. Coarse set: 96 vs "
        "100 is not a question worth a trial.",
        choices=[32, 64, 128, 192, 256],
    ),
    Param(
        "lstm_layers",
        "model.lstm_layers",
        "categorical",
        "capacity",
        "A second recurrent layer roughly doubles recurrent parameters; the "
        "default is 2, so this asks whether it earns them.",
        choices=[1, 2],
    ),
    Param(
        "lstm_bidirectional",
        "model.lstm_bidirectional",
        "categorical",
        "capacity",
        "Bidirectional doubles recurrent parameters and the head's input "
        "width. Justified offline, but with overfitting this early it is "
        "worth testing rather than assuming.",
        choices=[True, False],
    ),
    Param(
        "head_hidden",
        "model.head_hidden",
        "categorical",
        "capacity",
        "Width of the MLP head. Small, but it sits where the model is closest " "to the label.",
        choices=[32, 64, 128],
    ),
    # ---- Optimization: narrow, because the evidence says it is not the problem
    Param(
        "lr",
        "train.lr",
        "float",
        "optimization",
        "1e-3 already works and mid-run reduction did not help, so this "
        "refines a known-good region rather than exploring. The evidence is "
        "against mid-run REDUCTION; a different initial rate is untested, "
        "which is why it stays in the search at all.",
        low=3e-4,
        high=3e-3,
        log=True,
    ),
    # ---- Sensitivity: asymmetric on purpose ---------------------------------
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
#  the dimension the search most needs to resolve. These five do separate.
STARTUP_TRIALS: tuple[dict[str, Any], ...] = (
    # The current baseline, so the search has today's model as a reference.
    {
        "conv_dropout": 0.3,
        "lstm_dropout": 0.3,
        "head_dropout": 0.3,
        "weight_decay": 1e-4,
        "conv_width_mult": 1.0,
        "downsample": "4,4,2",
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "lstm_bidirectional": True,
        "head_hidden": 64,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Barely regularized and wide: should overfit hard, anchoring the low end.
    {
        "conv_dropout": 0.05,
        "lstm_dropout": 0.0,
        "head_dropout": 0.05,
        "weight_decay": 1e-5,
        "conv_width_mult": 1.5,
        "downsample": "4,2,2",
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "lstm_bidirectional": True,
        "head_hidden": 128,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Heavily regularized and narrow: the opposite anchor.
    {
        "conv_dropout": 0.55,
        "lstm_dropout": 0.5,
        "head_dropout": 0.55,
        "weight_decay": 5e-2,
        "conv_width_mult": 0.5,
        "downsample": "5,5,4",
        "lstm_hidden": 32,
        "lstm_layers": 1,
        "lstm_bidirectional": False,
        "head_hidden": 32,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Small capacity, light explicit regularization: separates the two routes
    # to the same effect, which the search would otherwise confound.
    {
        "conv_dropout": 0.1,
        "lstm_dropout": 0.0,
        "head_dropout": 0.1,
        "weight_decay": 1e-4,
        "conv_width_mult": 0.5,
        "downsample": "4,4,4",
        "lstm_hidden": 32,
        "lstm_layers": 1,
        "lstm_bidirectional": False,
        "head_hidden": 32,
        "lr": 1e-3,
        "pos_weight": 1.0,
    },
    # Baseline capacity, sensitivity-weighted, moderate regularization.
    {
        "conv_dropout": 0.35,
        "lstm_dropout": 0.3,
        "head_dropout": 0.35,
        "weight_decay": 1e-3,
        "conv_width_mult": 1.0,
        "downsample": "4,4,2",
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "lstm_bidirectional": True,
        "head_hidden": 64,
        "lr": 1e-3,
        "pos_weight": 2.5,
    },
)


def set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    """
    @brief Set a dotted path in a nested dict, creating sections as needed.

    @param config Dict to modify in place.
    @param path Dotted path, e.g. `"model.conv_dropout"`.
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

    @details
    Applies each parameter's transform, so `conv_width_mult: 0.5` becomes
    `model.conv_channels: (16, 32, 64)` -- a real `ModelConfig` field.

    @param params Mapping of `Param.name` to sampled value.
    @return Nested partial config for `train(config_overrides=...)`.
    @throws KeyError If a name is not in `SEARCH_SPACE`.
    """
    by_name = {p.name: p for p in SEARCH_SPACE}
    out: dict[str, Any] = {}
    for name, value in params.items():
        if name not in by_name:
            raise KeyError(f"{name!r} is not in SEARCH_SPACE")
        p = by_name[name]
        set_by_path(out, p.path, p.apply(value))
    return out


def suggest_all(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    @brief Sample every parameter and build the config overrides.

    @param trial An `optuna.Trial`.
    @return Tuple `(params, overrides)`; `params` holds the sampled values as
            Optuna recorded them, `overrides` the transformed config.
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
            arrow = " ->" if p.transform else "  "
            lines.append(f"    {p.name:20}{arrow} {p.path:26} {rng}")
            lines.append(f"      {p.why}")
    lines.append("\n  Sequence length into the LSTM, per pooling choice:")
    for spec in ("4,2,2", "4,4,2", "4,4,4", "5,5,4"):
        total = 2500 // sequence_length(spec)
        lines.append(f"    {spec:8} -> {sequence_length(spec):4} steps ({total}x)")
    return "\n".join(lines)
