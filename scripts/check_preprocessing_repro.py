"""
@file check_preprocessing_repro.py
@brief Phase 0 / Step 1 wrap-up: confirm the UPDATED preprocessing pipeline
       reproduces the converted float16 uV dataset from raw EDFs.

@details
Picks a few raw training EDFs (half normal, half abnormal), runs them through
the exact preprocessing chain used by `preprocessing.main()` (including the
updated `save_epochs`), writes the output to a temporary directory, and
compares it against the corresponding file in the converted dataset.

Expected result: bit-identical arrays. Both paths compute (float64 V) * 1e6
and cast to float16, so rounding is identical. Differences of at most one
float16 step are tolerated in case of MNE/NumPy version drift.

Passing this check means the float64 data is no longer needed: the float16
dataset can be regenerated from the raw EDFs with version-controlled code.

@par Usage (from repo root; normally via jobs/check_repro.sh):
@verbatim
python -m scripts.check_preprocessing_repro --n-files 4
python -m scripts.check_preprocessing_repro --module src.preprocessing.preprocessing
@endverbatim
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_TRAIN = Path("/shared/rc/eeg-cnn-lstm/data/raw-datasets/tuab/v3.0.1/edf/train")
CONVERTED = Path("/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train")

## @brief Where preprocessing.py may live; the first importable one is used.
CANDIDATE_MODULES = [
    "src.eeg_cnn_lstm.preprocessing.preprocessing",
    "src.preprocessing.preprocessing",
]


def import_preprocessing(name: str | None):
    """
    @brief Import the preprocessing module by explicit name or by trying candidates.

    @param name Dotted module path, or None to try `CANDIDATE_MODULES`.
    @return The imported module.
    @throws ImportError If no candidate can be imported.
    """
    names = [name] if name else CANDIDATE_MODULES
    errors = []
    for n in names:
        try:
            mod = importlib.import_module(n)
            print(f"Using preprocessing module: {n}")
            return mod
        except ImportError as exc:
            errors.append(f"{n}: {exc}")
    raise ImportError("Could not import preprocessing module:\n  " + "\n  ".join(errors)
                      + "\nPass --module <dotted.path> explicitly.")


def run_pipeline(pp, edf_file: Path, out_dir: Path) -> dict:
    """
    @brief Run one EDF through the same chain as `preprocessing.main()`.

    @param pp Imported preprocessing module.
    @param edf_file Raw EDF path.
    @param out_dir Directory to write the resulting .npy.
    @return Manifest row dict returned by `save_epochs`.
    """
    raw = pp.load_edf(edf_file)
    raw = pp.clean_channel_names(raw)
    raw = pp.remove_non_eeg_channels(raw)
    raw = pp.select_1020_channels(raw)
    raw = pp.filter_raw(raw)
    raw = pp.resample_raw(raw)
    raw = pp.apply_common_average_montage(raw)
    epochs = pp.segment_raw(raw)
    label = pp.extract_label(edf_file)
    return pp.save_epochs(epochs, edf_file, out_dir, label)


def compare(new_path: Path, ref_path: Path) -> dict:
    """
    @brief Compare a freshly preprocessed file against the converted reference.

    @return Dict with shape/dtype checks, exact-match flag, and diff stats.
    """
    new = np.load(new_path)
    ref = np.load(ref_path, mmap_mode="r")
    res = {
        "new_shape": list(new.shape),
        "ref_shape": list(ref.shape),
        "new_dtype": str(new.dtype),
        "ref_dtype": str(ref.dtype),
    }
    if new.shape != ref.shape:
        res.update(match="shape_mismatch", passed=False)
        return res

    ref = np.asarray(ref)
    exact = bool(np.array_equal(new, ref))
    diff = np.abs(new.astype(np.float32) - ref.astype(np.float32))
    ulp = np.spacing(np.abs(ref)).astype(np.float32)  # one float16 step at each value
    within_one_step = bool(np.all(diff <= ulp))
    res.update(
        exact_match=exact,
        n_values_differing=int((diff > 0).sum()),
        max_abs_diff_uv=float(diff.max()),
        within_one_float16_step=within_one_step,
        passed=(new.dtype == np.float16) and (exact or within_one_step),
    )
    return res


def main() -> int:
    """@brief CLI entrypoint. @return 0 if all files pass, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Check preprocessing reproduces float16 data")
    parser.add_argument("--module", default=None, help="Dotted path to preprocessing module")
    parser.add_argument("--raw-dir", type=Path, default=RAW_TRAIN)
    parser.add_argument("--converted-dir", type=Path, default=CONVERTED)
    parser.add_argument("--n-files", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/phase0/preprocessing_repro.json")
    args = parser.parse_args()

    pp = import_preprocessing(args.module)
    manifest = pd.read_csv(args.converted_dir / "train_manifest.csv").set_index("filename")
    npy_dir = args.converted_dir / "train"

    # Pick half normal / half abnormal, only EDFs that have a converted counterpart.
    rng = np.random.default_rng(args.seed)
    chosen: list[Path] = []
    for label_dir in ("normal", "abnormal"):
        edfs = sorted((args.raw_dir / label_dir).rglob("*.edf"))
        edfs = [e for e in edfs if (npy_dir / f"{e.stem}_epochs.npy").is_file()]
        k = min(len(edfs), (args.n_files + 1) // 2)
        chosen += [edfs[i] for i in sorted(rng.choice(len(edfs), size=k, replace=False))]
    if not chosen:
        print("ERROR: no matching EDF / converted file pairs found", file=sys.stderr)
        return 2

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for edf in chosen:
            print(f"\n--- {edf.name} ---")
            row = run_pipeline(pp, edf, tmp_dir)
            res = compare(tmp_dir / row["filename"], npy_dir / row["filename"])
            res["filename"] = row["filename"]
            res["label"] = int(row["label"])
            res["n_epochs"] = int(row["n_epochs"])
            res["manifest_n_epochs"] = int(manifest.loc[row["filename"], "n_epochs"])
            res["passed"] = res["passed"] and res["n_epochs"] == res["manifest_n_epochs"]
            results.append(res)

    print("\n=== RESULTS ===")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = (
            "bit-identical" if r.get("exact_match")
            else f"max diff {r.get('max_abs_diff_uv', float('nan')):.4f} uV, "
                 f"{r.get('n_values_differing', '?')} values differ"
        )
        print(f"{status}  {r['filename']}  label={r['label']}  "
              f"epochs={r['n_epochs']} (manifest {r['manifest_n_epochs']})  "
              f"dtype={r['new_dtype']}  {detail}")
    all_passed = all(r["passed"] for r in results)
    print(f"\nVERDICT: {'PASS - float64 data can be deleted' if all_passed else 'FAIL - keep float64 data'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": all_passed, "files": results}, indent=2))
    print(f"Wrote {out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())