"""
@file make_ablation_points.py
@brief Generate the point sets for the parameter-efficiency ablation.

@details
Phase 4. The scatter produced by the hyperparameter search is **observational**:
its 42 points differ in dropout, learning rate, class weighting and three
capacity axes simultaneously, and 27 of them sit at just two parameter counts.
"Performance peaks near 250K" is a reasonable reading of that chart, but it is
not a controlled result and it would not survive a question about confounding.

This script emits the point sets for the controlled version. Everything is held
at the selected configuration and **one capacity axis moves at a time**, so a
difference between two points is attributable to that axis and nothing else.

@par Two jobs, two sets
`--set shape` is the sweep: 14 points at 3 seeds, establishing the shape of each
axis. `--set decision` is the follow-up: a handful of points at 5 seeds, chosen
from the shape results, where the equivalence claim is made. Splitting them
means the expensive precision is spent only where a conclusion depends on it,
and the first job's results inform the second job's points.

@par Why depth needs direct overrides
`conv_width_mult` and `downsample` are search-space parameters with transforms,
but the pool spec is fixed at three blocks by `parse_pool_sizes`, so depth
cannot be expressed there at all. Depth also moves two fields together --
`conv_channels` and `pool_sizes` -- and moving only one would silently change
the sequence length as well. The points here are therefore written as direct
config overrides, which `seed_validate.py` accepts alongside search-space
`params`.

@par Holding depth honest
Changing the number of CNN blocks changes the number of pooling stages, and
therefore the sequence length the LSTM sees. A naive depth sweep varies depth
*and* sequence length and can attribute neither. Every depth here keeps the
**total pooling product at 16** (156 timesteps) and the **final channel width at
128**, so the LSTM's input is identical across the arm and depth is the only
thing moving:

  | depth | conv_channels      | pool_sizes   | product | LSTM input |
  |-------|--------------------|--------------|---------|------------|
  | 2     | (64, 128)          | (4, 4)       | 16      | 156 x 128  |
  | 3     | (32, 64, 128)      | (4, 2, 2)    | 16      | 156 x 128  |
  | 4     | (16, 32, 64, 128)  | (2, 2, 2, 2) | 16      | 156 x 128  |

@par Reaching past the search space
The ablation deliberately goes outside the ranges the search offered:
`lstm_hidden` down to 16, `conv_width_mult` to 0.25 and 2.0, `lstm_layers` to 3.
The search space was drawn to find a good model; the ablation is drawn to find
where the model breaks, and those are different jobs. The smallest HPO model,
at 70.9K parameters, still beat the Step 2 baseline -- so the floor has not
been found, and nothing below 70.9K has ever been tried.

@par Usage:
@verbatim
python -m scripts.make_ablation_points --set shape --out configs/ablation_shape.json
python -m scripts.make_ablation_points --set shape --describe      # table, no file
python -m scripts.make_ablation_points --set decision \
    --points center lstm_hidden_32 conv_width_0.5 lstm_layers_1 \
    --out configs/ablation_decision.json
@endverbatim
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Optional

## @brief The selected configuration (trial 61), the hub of every axis.
#
#  Seed validation measured this at mean 0.7921, sd 0.0032 over three seeds --
#  statistically equivalent to the nominal HPO winner but four times more
#  reproducible at the same 246,497 parameters.
CENTER: dict[str, Any] = {
    "model": {
        "conv_channels": [32, 64, 128],
        "pool_sizes": [4, 2, 2],
        "conv_dropout": 0.06833770752158194,
        "lstm_hidden": 64,
        "lstm_layers": 2,
        "lstm_bidirectional": True,
        "lstm_dropout": 0.3899726212809172,
        "head_hidden": 64,
        "head_dropout": 0.0792873966634228,
    },
    "train": {
        "lr": 0.002070212608238921,
        "pos_weight": 1.2517970214422354,
        "weight_decay": 2.388470451335015e-05,
    },
}

## @brief Base widths the conv-width axis scales, matching `search_space`.
BASE_CONV_CHANNELS = (32, 64, 128)

## @brief Floor on any scaled channel count, matching `search_space`.
MIN_CHANNELS = 8

## @brief Depth arm. Total pooling 16 and final width 128 in every entry, so
#  the LSTM input is identical and only depth moves.
DEPTHS: dict[int, tuple[list[int], list[int]]] = {
    2: ([64, 128], [4, 4]),
    3: ([32, 64, 128], [4, 2, 2]),
    4: ([16, 32, 64, 128], [2, 2, 2, 2]),
}

## @brief Levels per axis. The centre value appears in each and is emitted once.
AXES: dict[str, list[Any]] = {
    "conv_depth": [2, 3, 4],
    "lstm_layers": [1, 2, 3],
    "lstm_hidden": [16, 32, 64, 128, 256],
    "conv_width": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
}

## @brief Level of each axis that the centre configuration already sits at.
CENTER_LEVEL: dict[str, Any] = {
    "conv_depth": 3,
    "lstm_layers": 2,
    "lstm_hidden": 64,
    "conv_width": 1.0,
}

## @brief Seeds for the shape sweep. Shared across every point so comparisons
#  are blocked: a hard seed is hard everywhere and cancels in the difference.
SHAPE_SEEDS = [42, 1337, 2024]

## @brief Seeds for the decision points. Five where a conclusion rests on it.
DECISION_SEEDS = [42, 1337, 2024, 7, 101]


def scale_channels(mult: float) -> list[int]:
    """
    @brief Scale the base conv widths, keeping the 1:2:4 shape.

    @param mult Width multiplier.
    @return Channel counts per block, each at least `MIN_CHANNELS`.
    @throws ValueError If `mult` is not positive.
    """
    if mult <= 0:
        raise ValueError(f"conv width multiplier must be positive; got {mult}")
    return [max(MIN_CHANNELS, int(round(c * mult))) for c in BASE_CONV_CHANNELS]


def point(axis: str, level: Any) -> dict[str, Any]:
    """
    @brief Build one ablation point: the centre with a single axis moved.

    @param axis One of `AXES`.
    @param level The level to set on that axis.
    @return A configuration entry with `name`, `axis`, `level` and `overrides`.
    @throws ValueError If the axis is unknown or the level is unusable.
    """
    ov = copy.deepcopy(CENTER)
    model = ov["model"]

    if axis == "conv_depth":
        if level not in DEPTHS:
            raise ValueError(f"no depth recipe for {level}; have {sorted(DEPTHS)}")
        model["conv_channels"], model["pool_sizes"] = (list(x) for x in DEPTHS[level])
    elif axis == "lstm_layers":
        model["lstm_layers"] = int(level)
    elif axis == "lstm_hidden":
        model["lstm_hidden"] = int(level)
    elif axis == "conv_width":
        model["conv_channels"] = scale_channels(float(level))
    else:
        raise ValueError(f"unknown axis {axis!r}; have {sorted(AXES)}")

    is_center = level == CENTER_LEVEL[axis]
    name = "center" if is_center else f"{axis}_{level}"
    return {
        "name": name,
        "axis": "center" if is_center else axis,
        "level": level,
        "overrides": ov,
    }


def shape_points() -> list[dict[str, Any]]:
    """
    @brief The full shape sweep: centre plus every off-centre level.

    @details
    The centre sits on all four axes at once, so it is emitted once and shared.
    Fourteen points rather than seventeen, and every arm is anchored to the
    same measurement rather than to three separate estimates of it.

    @return Configuration entries, centre first.
    """
    out = [point("conv_depth", CENTER_LEVEL["conv_depth"])]
    for axis, levels in AXES.items():
        for lv in levels:
            if lv == CENTER_LEVEL[axis]:
                continue
            out.append(point(axis, lv))
    return out


def count_parameters(overrides: dict[str, Any]) -> Optional[int]:
    """
    @brief Instantiate the model and count trainable parameters.

    @details
    Knowing the x-axis coverage before committing GPU hours is the point of
    generating these up front: a sweep whose parameter counts clump the way the
    search's did would be worth redesigning, not running. Returns `None` when
    torch or the model package is unavailable, so the generator still works off
    the cluster.

    @param overrides The point's config override tree.
    @return Trainable parameter count, or `None` if it cannot be computed.
    """
    try:
        from eeg_cnn_lstm.models.model import CNNLSTM, ModelConfig
    except Exception:  # noqa: BLE001 - absence is expected off the cluster
        return None
    try:
        fields = {f for f in ModelConfig.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in overrides["model"].items() if k in fields}
        unknown = set(overrides["model"]) - fields
        if unknown:
            raise ValueError(f"ModelConfig has no field(s) {sorted(unknown)}")
        model = CNNLSTM(ModelConfig(**kwargs))
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception as exc:  # noqa: BLE001 - report, do not abort generation
        print(f"  warning: parameter count failed: {exc}", file=sys.stderr)
        return None


def describe(points: list[dict[str, Any]], seeds: list[int]) -> None:
    """
    @brief Print the sweep as a table, with parameter counts where available.

    @param points Configuration entries.
    @param seeds Seeds each point will be run at.
    """
    print(f"{len(points)} points x {len(seeds)} seeds = {len(points) * len(seeds)} runs")
    print(f"seeds: {seeds}\n")
    print(f"  {'name':22} {'axis':12} {'level':>6} {'params':>10}  conv / pool / lstm")
    counts = []
    for p in points:
        m = p["overrides"]["model"]
        n = count_parameters(p["overrides"])
        if n:
            counts.append(n)
        arch = (
            f"{m['conv_channels']} / {m['pool_sizes']} / " f"{m['lstm_hidden']}x{m['lstm_layers']}"
        )
        print(
            f"  {p['name']:22} {p['axis']:12} {str(p['level']):>6} "
            f"{(f'{n:,}' if n else '-'):>10}  {arch}"
        )
    if counts:
        lo, hi = min(counts), max(counts)
        print(f"\nparameter coverage: {lo:,} to {hi:,}  ({hi / lo:.0f}x span)")
        print(
            "The search's own scatter put 27 of 42 points at two values; this "
            "sweep spreads them deliberately."
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace of options.
    """
    p = argparse.ArgumentParser(description="Generate ablation point sets.")
    p.add_argument("--set", default="shape", choices=["shape", "decision"])
    p.add_argument("--out", default=None, help="Destination JSON; omit to print only")
    p.add_argument(
        "--points",
        nargs="*",
        default=None,
        help="For --set decision: names from the shape sweep to re-run at 5 seeds",
    )
    p.add_argument("--seeds", default=None, help="Comma-separated seed override")
    p.add_argument("--describe", action="store_true", help="Print the table and exit")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing --out")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args(argv)
    allp = shape_points()

    if a.set == "shape":
        points, seeds = allp, SHAPE_SEEDS
        note = (
            "Shape sweep for the parameter-efficiency ablation. Everything is "
            "held at the selected configuration (trial 61) and one capacity "
            "axis moves at a time, so the HPO scatter's confounding is removed. "
            "Depth keeps total pooling at 16 and final width at 128, so the "
            "LSTM input is identical across that arm and only depth moves. "
            "Three seeds per point, shared across all points so comparisons are "
            "blocked on seed -- seed-to-seed variation exceeded "
            "configuration-to-configuration variation during selection. These "
            "points establish curve shape; precision is bought separately in "
            "the decision set."
        )
    else:
        if not a.points:
            print("error: --set decision needs --points", file=sys.stderr)
            return 1
        index = {p["name"]: p for p in allp}
        missing = [n for n in a.points if n not in index]
        if missing:
            print(f"error: unknown point(s) {missing}", file=sys.stderr)
            print(f"available: {sorted(index)}", file=sys.stderr)
            return 1
        points = [index[n] for n in a.points]
        seeds = DECISION_SEEDS
        note = (
            "Decision points for the parameter-efficiency ablation, selected "
            "from the shape sweep. Five seeds rather than three, because the "
            "equivalence claim -- the smallest model statistically "
            "indistinguishable from the best -- rests on these and nothing "
            "else does. Compare with scripts/compare_configs.py using the "
            "margin stated before the runs."
        )

    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    describe(points, seeds)

    if a.describe or a.out is None:
        print("\nnothing written")
        return 0

    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f"error: {out} exists; pass --overwrite", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "study": f"ablation-{a.set}",
                "objective": "recording_pauc_sens",
                "seeds": seeds,
                "center": "trial_0061",
                "note": note,
                "configs": points,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
