## @package tests.test_data_integrity
# Unit tests to verify that the sample dataset has been generated correctly
# and is fully compatible with the preprocessing pipeline.
#
# # Generated with the assistance of Claude (Anthropic), April 2026
#
# These tests satisfy the code review requirement for "a unit test which can be
# run to verify your data has been downloaded/scraped/preprocessed correctly and
# is in the correct format for your models."
#
# Run after create_sample_data.py:
#   python create_sample_data.py
#   pytest tests/test_data_integrity.py -v


import numpy as np
import mne
from pathlib import Path

# Adjust these paths if your config.yaml points elsewhere.
SAMPLE_DATA_DIR = Path("data/raw/edf")
PROCESSED_DATA_DIR = Path("data/processed")

EXPECTED_CHANNELS_AFTER_PREPROCESSING = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]
VALID_SFREQS = {250.0, 256.0, 512.0}
MIN_RECORDING_DURATION_SEC = 10.0  # Must be long enough for at least one epoch


# ===== Helpers =====


def get_edf_files(split):
    """Return all .edf files under data/raw/edf/<split>/."""
    return list((SAMPLE_DATA_DIR / split).rglob("*.edf"))


def load_raw(edf_path):
    """Load an EDF file with MNE, suppressing verbose output."""
    return mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)


# ===== 1. Dataset existence =====


## Verifies that the sample data directory has been created.
#
# Fails with a descriptive message if create_sample_data.py has not been run.
def test_sample_data_root_exists():
    assert SAMPLE_DATA_DIR.exists(), (
        "Sample data not found. Generate it first:\n" "  python create_sample_data.py"
    )


## Verifies that both train and eval splits are present.
def test_both_splits_exist():
    for split in ["train", "eval"]:
        assert (
            SAMPLE_DATA_DIR / split
        ).exists(), f"Missing split directory: {SAMPLE_DATA_DIR / split}"


## Verifies that normal and abnormal subdirectories exist in each split.
def test_both_labels_per_split():
    for split in ["train", "eval"]:
        for label in ["normal", "abnormal"]:
            assert (
                SAMPLE_DATA_DIR / split / label
            ).exists(), f"Missing label directory: {SAMPLE_DATA_DIR / split / label}"


## Verifies that each split contains at least one EDF file per class.
def test_minimum_edf_file_count():
    for split in ["train", "eval"]:
        for label in ["normal", "abnormal"]:
            edfs = list((SAMPLE_DATA_DIR / split / label).rglob("*.edf"))
            assert (
                len(edfs) >= 1
            ), f"Expected at least 1 .edf file in {split}/{label}, found {len(edfs)}"


# ===== 2. EDF format validity =====


## Verifies that every EDF file in the train split can be opened by MNE.
def test_edf_files_are_readable():
    edfs = get_edf_files("train")
    assert len(edfs) > 0, "No EDF files found in train split."

    for edf in edfs:
        raw = load_raw(edf)
        assert raw is not None, f"Failed to load: {edf}"
        assert len(raw.ch_names) > 0, f"No channels found in: {edf}"


## Verifies that each EDF file has a supported sampling rate.
def test_sampling_rates_are_valid():
    edfs = get_edf_files("train")
    for edf in edfs:
        raw = load_raw(edf)
        assert raw.info["sfreq"] in VALID_SFREQS, (
            f"{edf.name}: unexpected sampling rate {raw.info['sfreq']} Hz. "
            f"Expected one of {VALID_SFREQS}."
        )


## Verifies that each EDF file is long enough for at least one 10-second epoch.
def test_recordings_meet_minimum_duration():
    edfs = get_edf_files("train")
    for edf in edfs:
        raw = load_raw(edf)
        duration = raw.times[-1]
        assert duration >= MIN_RECORDING_DURATION_SEC, (
            f"{edf.name}: duration {duration:.1f}s is shorter than the required "
            f"{MIN_RECORDING_DURATION_SEC}s minimum."
        )


## Verifies that EDF files contain the expected TUH-format channel names.
#
# Checks that at least 19 channels with the "EEG" prefix exist, matching
# the TUH EEG Abnormal Corpus naming convention.
def test_edf_channel_names_match_tuh_format():
    edfs = get_edf_files("train")
    for edf in edfs:
        raw = load_raw(edf)
        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
        assert len(eeg_channels) >= 19, (
            f"{edf.name}: found only {len(eeg_channels)} EEG channels. "
            "Expected ≥19 channels with 'EEG' in the name (TUH format)."
        )


# ===== 3. Label encoding via directory structure =====


## Verifies that extract_label() correctly reads 0 for files under normal/ and
# 1 for files under abnormal/.
def test_label_extraction_from_directory_structure():
    from src.preprocessing.preprocessing import extract_label

    for split in ["train", "eval"]:
        normal_edfs = list((SAMPLE_DATA_DIR / split / "normal").rglob("*.edf"))
        abnormal_edfs = list((SAMPLE_DATA_DIR / split / "abnormal").rglob("*.edf"))

        for edf in normal_edfs:
            assert (
                extract_label(edf) == 0
            ), f"Expected label 0 (normal) for {edf}, got {extract_label(edf)}"
        for edf in abnormal_edfs:
            assert (
                extract_label(edf) == 1
            ), f"Expected label 1 (abnormal) for {edf}, got {extract_label(edf)}"


# ===== 4. Preprocessing pipeline compatibility =====


## Integration test: runs the full preprocessing pipeline on one sample file
# and asserts that the output epochs have the correct shape.
#
# Expected output: (n_epochs, 19, 2500) — 19 channels, 2500 samples at 250 Hz.
def test_full_pipeline_produces_valid_epochs(tmp_path):
    from src.preprocessing.preprocessing import (
        load_edf,
        clean_channel_names,
        remove_non_eeg_channels,
        select_1020_channels,
        filter_raw,
        resample_raw,
        apply_common_average_montage,
        segment_raw,
    )

    edfs = get_edf_files("train")
    assert len(edfs) > 0, "No EDF files to test against."

    edf = edfs[0]
    raw = load_edf(str(edf))
    raw = clean_channel_names(raw)
    raw = remove_non_eeg_channels(raw)
    raw = select_1020_channels(raw)
    raw = filter_raw(raw)
    raw = resample_raw(raw)
    raw = apply_common_average_montage(raw)
    epochs = segment_raw(raw)

    assert len(epochs) > 0, "Pipeline produced zero epochs."

    _, n_channels, n_times = epochs.get_data().shape
    assert (
        n_channels == 19
    ), f"Expected 19 channels after channel selection, got {n_channels}."
    assert (
        n_times == 2500
    ), f"Expected 2500 samples per epoch (10s × 250 Hz), got {n_times}."


## Verifies that the channel names after preprocessing match the standard 10-20 set.
def test_pipeline_channel_names_after_preprocessing():
    from src.preprocessing.preprocessing import (
        load_edf,
        clean_channel_names,
        remove_non_eeg_channels,
        select_1020_channels,
    )

    edfs = get_edf_files("train")
    edf = edfs[0]
    raw = load_edf(str(edf))
    raw = clean_channel_names(raw)
    raw = remove_non_eeg_channels(raw)
    raw = select_1020_channels(raw)

    assert set(raw.ch_names) == set(EXPECTED_CHANNELS_AFTER_PREPROCESSING), (
        f"Channel mismatch after preprocessing.\n"
        f"  Got:      {sorted(raw.ch_names)}\n"
        f"  Expected: {sorted(EXPECTED_CHANNELS_AFTER_PREPROCESSING)}"
    )


## Verifies that the final resampled recording is at 250 Hz.
def test_pipeline_resamples_to_250hz():
    from src.preprocessing.preprocessing import (
        load_edf,
        clean_channel_names,
        remove_non_eeg_channels,
        select_1020_channels,
        filter_raw,
        resample_raw,
    )

    edfs = get_edf_files("train")
    edf = edfs[0]
    raw = load_edf(str(edf))
    raw = clean_channel_names(raw)
    raw = remove_non_eeg_channels(raw)
    raw = select_1020_channels(raw)
    raw = filter_raw(raw)
    raw = resample_raw(raw)

    assert (
        raw.info["sfreq"] == 250.0
    ), f"Expected 250 Hz after resampling, got {raw.info['sfreq']} Hz."


## Verifies that save_epochs produces a valid .npy file with the correct array shape.
def test_pipeline_saves_valid_npy(tmp_path):
    from src.preprocessing.preprocessing import (
        load_edf,
        clean_channel_names,
        remove_non_eeg_channels,
        select_1020_channels,
        filter_raw,
        resample_raw,
        apply_common_average_montage,
        segment_raw,
        save_epochs,
        extract_label,
    )

    edfs = get_edf_files("train")
    edf = edfs[0]
    raw = load_edf(str(edf))
    raw = clean_channel_names(raw)
    raw = remove_non_eeg_channels(raw)
    raw = select_1020_channels(raw)
    raw = filter_raw(raw)
    raw = resample_raw(raw)
    raw = apply_common_average_montage(raw)
    epochs = segment_raw(raw)

    label = extract_label(edf)
    row = save_epochs(epochs, edf, tmp_path, label)

    npy_path = tmp_path / row["filename"]
    assert npy_path.exists(), f"Expected .npy output at {npy_path}"

    saved = np.load(npy_path)
    assert saved.shape == epochs.get_data().shape, (
        f"Saved array shape {saved.shape} does not match "
        f"epochs shape {epochs.get_data().shape}."
    )
    assert saved.dtype in [
        np.float32,
        np.float64,
    ], f"Unexpected dtype {saved.dtype}. Expected float32 or float64."
