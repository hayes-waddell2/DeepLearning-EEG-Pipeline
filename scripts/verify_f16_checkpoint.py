"""
@file verify_f16_checkpoint.py
@brief Phase 0 / Step 1 verification: confirm float16 uV data gives the same
       results as the original float64 V data.

@details
Rebuilds the exact baseline validation fold (same manifest, val_frac, seed) and
runs two checks on both the old and the new data directory:

  1. Numerical: for a random sample of segments, compare the z-scored float32
     tensors the model would actually receive. Because the current dataset
     z-scores each channel per segment, the V -> uV scale factor cancels out,
     so this isolates float16 precision loss.
  2. Functional: run the saved baseline checkpoint over the full validation
     fold on both directories and compare logits, AUC, and confusion matrices.

Sanity check: the old-data AUC should reproduce the baseline's epoch-3 value
(0.8851). If it does not, the split or checkpoint differs from the baseline run.

Decision rule: PASS if |delta AUC| < 0.002.

@par Usage (from repo root, on a GPU node; normally via jobs/verify_f16.sh):
@verbatim
python -m scripts.verify_f16_checkpoint \
    --old-data-dir /shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab/train/train \
    --new-data-dir /shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.eeg_cnn_lstm.models.model_b import CNN_LSTM, ModelConfig
from src.utils.dataset import TUABEpochDataset, build_subject_disjoint_split, load_manifest
from src.utils.metrics import compute_metrics

EXPECTED_BASELINE_AUC = 0.8851
AUC_TOLERANCE = 0.002


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    dataset: TUABEpochDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    @brief Run the model over a dataset in order (no shuffle), FP32, like `evaluate`.

    @return Tuple `(logits, labels)` as 1-D CPU tensors.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main() -> int:
    """@brief CLI entrypoint. @return 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser(description="Verify float16 uV conversion")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--old-data-dir", required=True)
    parser.add_argument("--new-data-dir", required=True)
    parser.add_argument(
        "--checkpoint", default="/shared/rc/eeg-cnn-lstm/runs/baseline_v1/best.pt"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-numeric-samples", type=int, default=500)
    parser.add_argument("--out", default="results/phase0/f16_verification.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["loader"].get("seed", 42))
    val_frac = float(cfg["loader"]["val_frac"])

    # Rebuild the exact baseline validation fold.
    manifest = load_manifest(cfg["data"]["train_manifest"])
    _, val_df = build_subject_disjoint_split(manifest, val_frac=val_frac, seed=seed)
    # Baseline used seed + 1 and no subsampling for the val set.
    ds_old = TUABEpochDataset(val_df, args.old_data_dir, max_epochs_per_recording=None, seed=seed + 1)
    ds_new = TUABEpochDataset(val_df, args.new_data_dir, max_epochs_per_recording=None, seed=seed + 1)
    assert len(ds_old) == len(ds_new), "Datasets differ in length"
    print(f"Validation fold: {len(val_df)} recordings, {len(ds_old):,} segments")

    # ---- 1. Numerical check on the tensors the model sees ----
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds_old), size=min(args.n_numeric_samples, len(ds_old)), replace=False)
    max_abs, mean_abs = 0.0, []
    for i in idx:
        a, _ = ds_old[int(i)]
        b, _ = ds_new[int(i)]
        d = (a - b).abs()
        max_abs = max(max_abs, float(d.max()))
        mean_abs.append(float(d.mean()))
    numeric = {
        "n_segments_checked": int(len(idx)),
        "max_abs_diff_zscore_units": max_abs,
        "mean_abs_diff_zscore_units": float(np.mean(mean_abs)),
    }
    print(f"Numerical: {numeric}")

    # ---- 2. Functional check with the baseline checkpoint ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(ModelConfig()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Running checkpoint on OLD data...")
    lo_old, y_old = collect_logits(model, ds_old, device, args.batch_size, args.num_workers)
    print("Running checkpoint on NEW data...")
    lo_new, y_new = collect_logits(model, ds_new, device, args.batch_size, args.num_workers)
    assert torch.equal(y_old, y_new), "Label order differs between runs"

    m_old = compute_metrics(lo_old, y_old)
    m_new = compute_metrics(lo_new, y_new)
    p_old, p_new = torch.sigmoid(lo_old), torch.sigmoid(lo_new)
    flips = int(((lo_old >= 0) != (lo_new >= 0)).sum())
    d_auc = m_new["auc_roc"] - m_old["auc_roc"]

    functional = {
        "old": m_old,
        "new": m_new,
        "delta_auc": d_auc,
        "max_abs_logit_diff": float((lo_old - lo_new).abs().max()),
        "mean_abs_prob_diff": float((p_old - p_new).abs().mean()),
        "prediction_flips": flips,
        "prediction_flip_rate": flips / len(lo_old),
        "old_auc_matches_baseline": abs(m_old["auc_roc"] - EXPECTED_BASELINE_AUC) < 5e-4,
    }
    passed = abs(d_auc) < AUC_TOLERANCE

    print("\n=== RESULTS ===")
    print(f"Old AUC: {m_old['auc_roc']:.4f}  (baseline epoch 3: {EXPECTED_BASELINE_AUC})")
    print(f"New AUC: {m_new['auc_roc']:.4f}   delta = {d_auc:+.5f}")
    print(f"Old CM: tp={m_old['tp']} fp={m_old['fp']} fn={m_old['fn']} tn={m_old['tn']}")
    print(f"New CM: tp={m_new['tp']} fp={m_new['fp']} fn={m_new['fn']} tn={m_new['tn']}")
    print(f"Prediction flips: {flips} ({functional['prediction_flip_rate']:.4%})")
    print(f"Max |logit diff|: {functional['max_abs_logit_diff']:.5f}")
    if not functional["old_auc_matches_baseline"]:
        print("!! Old-data AUC does not reproduce the baseline; check split/checkpoint.")
    print(f"\nVERDICT: {'PASS - keep float16' if passed else 'FAIL - use float32'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"numerical": numeric, "functional": functional, "passed": passed,
             "auc_tolerance": AUC_TOLERANCE, "checkpoint": args.checkpoint,
             "old_data_dir": args.old_data_dir, "new_data_dir": args.new_data_dir},
            indent=2,
        )
    )
    print(f"Wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())