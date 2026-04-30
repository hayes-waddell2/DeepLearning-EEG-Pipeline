# TUH EEG Abnormal Corpus — Preprocessing Summary

Run date: 2026-04-29
Source: TUH EEG Abnormal Corpus v3.0.1
Cluster: RIT RC, partition `sporc-cpu`, node `spr-a-01`
Job IDs: 21242747 (train), 21242748 (eval)

## Pipeline applied

For each `.edf` file:

1. Load with MNE
2. Clean channel names (strip `EEG ` prefix and `-REF` suffix, normalize case)
3. Drop non-EEG channels (ROC, LOC, EKG, EMG, photic, IBI, bursts, suppr, numeric channels)
4. Select the standard 10–20 montage (19 channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2)
5. Notch filter at 60 Hz
6. High-pass filter at 0.3 Hz
7. Resample to 250 Hz (most files were already at 250 Hz)
8. Common-average reference montage
9. Segment into 10-second epochs with 5-second overlap
10. Save as `.npy` with shape `(n_epochs, 19, 2500)`, dtype `float64`

## Output layout

```
/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab/
├── train/
│   ├── train/                       # 2,717 .npy files
│   │   └── <recording_id>_epochs.npy
│   └── train_manifest.csv
└── eval/
    ├── eval/                        # 276 .npy files
    │   └── <recording_id>_epochs.npy
    └── eval_manifest.csv
```

Manifest columns: `filename, label, n_epochs, sfreq`. Label 0 = normal, 1 = abnormal.

Note: the doubled `train/train/` and `eval/eval/` is a quirk of `preprocessing.py` mirroring the input split structure inside the output directory. Cosmetic; downstream loaders should reference the inner folder.

## Class balance

| Split | Label 0 (normal) | Label 1 (abnormal) | Total recordings |
|-------|------------------|--------------------|------------------|
| Train | 1,371            | 1,346              | 2,717            |
| Eval  | 150              | 126                | 276              |

| Split | Label 0 epochs | Label 1 epochs | Total epochs |
|-------|----------------|----------------|--------------|
| Train | 366,708        | 376,870        | 743,578      |
| Eval  | 39,726         | 34,008         | 73,734       |

Class balance is essentially 50/50 in both splits. No class-weighting or oversampling needed during training.

## Sample-level data check

Verified on `train/train/aaaaaoyy_s001_t000_epochs.npy`:

| Field | Value | Interpretation |
|-------|-------|----------------|
| Shape | `(251, 19, 2500)` | 251 epochs × 19 channels × 2500 samples (10 s × 250 Hz) |
| dtype | `float64` | 8 bytes per sample |
| Mean  | ≈ −4.9e-24 | Essentially zero — common-average montage worked correctly |
| Std   | ≈ 1.95e-5 (≈ 19.5 µV) | Typical EEG amplitude |
| Min   | ≈ −459 µV | Within range for occasional artifact spikes |
| Max   | ≈ 429 µV | Within range for occasional artifact spikes |

## Resource usage

| Job   | Split | Elapsed | MaxRSS | CPUs | Outcome |
|-------|-------|---------|--------|------|---------|
| 21242747 | train | 34 min 24 s | ~16 GB | 2  | Data saved cleanly |
| 21242748 | eval  | 4 min 48 s  | ~16 GB | 2  | Data saved cleanly |

Both jobs printed the closing `Finished:` banner. The non-zero exit code (`127`) was caused by a stray heredoc terminator (`PREPROCESS_DEBUG_EOF`) left at the end of `jobs/process_tuh.sh`, which bash tried to execute as a command after the work was done. That has been removed; future runs will exit 0.

`MaxRSS` of ~16 GB sat right at the requested `--mem=16g` ceiling, so memory has been bumped to 32 GB for future runs.

## Notes for downstream code

- `*_manifest.csv` uses `\r\n` line endings (Python `csv` module / RFC 4180 default). pandas/numpy/Python `csv` reads handle this transparently. Bash tools like `awk` need `gsub(/\r/, "")` to strip the trailing carriage return.
- `.npy` files are `float64`. EEG doesn't need 64-bit precision; switching to `float32` in `preprocessing.py` (`epochs.astype(np.float32)` before `np.save`) would halve storage — current train data is ~280 GB, would drop to ~140 GB.
- Storage layout suggests using the `*_manifest.csv` as the source of truth in your `Dataset` class (read filename + label from the CSV, then load each `.npy` lazily) rather than walking the directory tree.
