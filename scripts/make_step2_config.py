"""
@file make_step2_config.py
@brief Derive the Step 2 (early stopping + LR schedule) config from a baseline.

@details
Phase 2 / Step 2. The early-stopping config differs from the Step 1 baseline
only in the `train` block; every data path, loader setting and normalization
choice must be inherited unchanged or the comparison is not a comparison. So
this script reads the baseline config and rewrites only the keys that Step 2
introduces, rather than asking anyone to retype paths that are already correct.

@par Choosing the cadence
Patience is counted in **validation checks**, not epochs, so it has to be
rescaled whenever `val_every_frac` changes. `--patience-epochs` keeps that
honest: it is expressed in epochs and converted, so the stopping rule means the
same thing at any cadence.

The default cadence comes from measurement, not taste. On this corpus a
validation pass over pool T costs ~0.3 min against a ~2 min training epoch:

  | val_every_frac | checks/epoch | epoch wall | checks over a 4-epoch window |
  |----------------|--------------|------------|------------------------------|
  | 1.0            | 1            | 2.3 min    |  4                           |
  | 0.5            | 2            | 2.6 min    |  8                           |
  | 0.25 (default) | 4            | 3.2 min    | 16                           |
  | 0.125          | 8            | 4.4 min    | 32                           |

0.25 is the knee. Finer mostly resamples noise: pool T holds ~543 recordings
and the partial-AUC confidence interval is about +/-0.05, so eight checks per
epoch resolve sampling variation rather than the peak.

@par Usage:
@verbatim
python -m scripts.make_step2_config                       # all defaults
python -m scripts.make_step2_config --val-every-frac 0.5  # cheaper cadence
python -m scripts.make_step2_config --dry-run             # print, write nothing
@endverbatim
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

## @brief Keys Step 2 adds or changes; everything else is inherited verbatim.
STEP2_KEYS = ("num_epochs", "output_dir", "early_stopping", "lr_schedule")


def build(
    baseline: dict[str, Any],
    output_dir: str,
    num_epochs: int,
    val_every_frac: float,
    patience_epochs: float,
    lr_patience_epochs: float,
    min_delta: float,
    factor: float,
    min_lr: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    @brief Produce the Step 2 config and the check counts it implies.

    @details
    Epoch-denominated patience is converted to checks at the configured
    cadence and floored at 1. The scheduler must get a strictly smaller
    patience than the stopper, otherwise the run halts before a reduced
    learning rate can show any effect; that ordering is enforced here rather
    than left to the person editing YAML.

    @param baseline Parsed baseline config (not mutated).
    @param output_dir Run directory for the Step 2 run.
    @param num_epochs Upper bound on epochs; early stopping decides the actual
           length.
    @param val_every_frac Fraction of an epoch between validation checks.
    @param patience_epochs Epochs of no improvement tolerated before stopping.
    @param lr_patience_epochs Epochs of no improvement before halving the LR.
    @param min_delta Absolute improvement that counts as progress.
    @param factor LR multiplier on plateau.
    @param min_lr Learning-rate floor.
    @return Tuple `(config, derived)` where `derived` holds the check counts.
    @throws ValueError If the cadence is outside (0, 1].
    """
    if not 0.0 < val_every_frac <= 1.0:
        raise ValueError(f"--val-every-frac must be in (0, 1]; got {val_every_frac}")

    checks_per_epoch = 1.0 / val_every_frac
    patience = max(1, int(round(patience_epochs * checks_per_epoch)))
    lr_patience = max(1, int(round(lr_patience_epochs * checks_per_epoch)))
    if lr_patience >= patience:
        lr_patience = max(1, patience - 1)

    cfg = yaml.safe_load(yaml.safe_dump(baseline))  # deep copy
    cfg.setdefault("train", {})
    cfg["train"].update(
        {
            "num_epochs": num_epochs,  # UPPER BOUND, not a target
            "output_dir": output_dir,
            "early_stopping": {
                "enabled": True,
                "val_every_frac": val_every_frac,
                "patience": patience,
                "min_delta": min_delta,
                "min_checks": 0,
                "restore_best": True,
            },
            "lr_schedule": {
                "enabled": False,
                "kind": "plateau",
                "factor": factor,
                "patience": lr_patience,
                "min_lr": min_lr,
                "cooldown": 0,
            },
        }
    )
    derived = {
        "checks_per_epoch": int(math.ceil(checks_per_epoch)),
        "stop_patience_checks": patience,
        "lr_patience_checks": lr_patience,
    }
    return cfg, derived


def parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace of config-generation options.
    """
    p = argparse.ArgumentParser(
        description="Derive the Step 2 early-stopping config from a baseline."
    )
    p.add_argument("--baseline", default="configs/baseline.yaml",
                   help="Config to inherit data and loader settings from")
    p.add_argument("--out", default="configs/step2_earlystop.yaml",
                   help="Destination config path")
    p.add_argument("--output-dir", default="runs/step2-earlystop",
                   help="train.output_dir for the Step 2 run")
    p.add_argument("--num-epochs", type=int, default=20,
                   help="Upper bound on epochs (early stopping ends the run)")
    p.add_argument("--val-every-frac", type=float, default=0.25,
                   help="Fraction of an epoch between validation checks")
    p.add_argument("--patience-epochs", type=float, default=3.0,
                   help="Epochs of no improvement before stopping")
    p.add_argument("--lr-patience-epochs", type=float, default=1.0,
                   help="Epochs of no improvement before halving the LR")
    p.add_argument("--min-delta", type=float, default=0.01,
                   help="Absolute improvement that counts as progress")
    p.add_argument("--factor", type=float, default=0.5,
                   help="LR multiplier on plateau")
    p.add_argument("--min-lr", type=float, default=1e-6, help="LR floor")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace the destination if it already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the train block and write nothing")
    return p.parse_args()


def main() -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args()
    base_p, out_p = Path(a.baseline), Path(a.out)
    if not base_p.is_file():
        print(f"error: baseline config not found: {base_p}", file=sys.stderr)
        return 1
    if out_p.exists() and not (a.overwrite or a.dry_run):
        print(f"error: {out_p} exists; pass --overwrite to replace it",
              file=sys.stderr)
        return 1

    baseline = yaml.safe_load(base_p.read_text())
    cfg, derived = build(
        baseline,
        output_dir=a.output_dir,
        num_epochs=a.num_epochs,
        val_every_frac=a.val_every_frac,
        patience_epochs=a.patience_epochs,
        lr_patience_epochs=a.lr_patience_epochs,
        min_delta=a.min_delta,
        factor=a.factor,
        min_lr=a.min_lr,
    )

    # Inheritance check: anything Step 2 did not deliberately change must be
    # byte-identical to the baseline, or the two runs are not comparable.
    changed = [
        k for k in set(baseline.get("train", {})) | set(cfg["train"])
        if baseline.get("train", {}).get(k) != cfg["train"].get(k)
    ]
    unexpected = sorted(set(changed) - set(STEP2_KEYS))
    for section in ("data", "loader", "eval"):
        if baseline.get(section) != cfg.get(section):
            unexpected.append(section)
    if unexpected:
        print(f"error: unexpected divergence from the baseline: {unexpected}",
              file=sys.stderr)
        return 1

    print(f"Baseline:      {base_p}")
    print(f"Cadence:       every {a.val_every_frac:g} epoch "
          f"(~{derived['checks_per_epoch']} checks/epoch)")
    print(f"Stop patience: {derived['stop_patience_checks']} checks "
          f"(= {a.patience_epochs:g} epochs)")
    print(f"LR patience:   {derived['lr_patience_checks']} checks "
          f"(= {a.lr_patience_epochs:g} epochs), factor {a.factor:g}")
    print(f"Epoch budget:  {a.num_epochs} (upper bound)")
    print(f"Inherited:     data, loader, eval and all other train keys\n")
    print(yaml.safe_dump({"train": cfg["train"]}, sort_keys=False))

    if a.dry_run:
        print("--dry-run: nothing written")
        return 0
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())