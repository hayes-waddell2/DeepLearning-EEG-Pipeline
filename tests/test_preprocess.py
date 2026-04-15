## @package tests.test_preprocess
# Unit tests for the preprocessing module.

import pytest
import numpy as np
import mne
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.eeg_cnn_lstm.preprocessing.preprocessing import (
    filter_raw,
    load_config,
    load_edf,
    clean_channel_names,
    remove_non_eeg_channels,
    select_1020_channels,
    resample_raw,
    apply_common_average_montage,
    segment_raw,
    extract_label,
    save_epochs,
)


# ===== Helpers =======


def make_synthetic_raw(freqs, sfreq=250.0, duration=10.0, amplitude=1.0):
    """Create a single-channel Raw object containing sinusoids at the given frequencies.

    Uses a real MNE RawArray so that actual MNE filtering can be applied in tests.

    @param freqs  List of frequencies (Hz) to include in the signal.
    @param sfreq  Sampling rate in Hz (default: 250.0).
    @param duration  Signal duration in seconds (default: 10.0).
    @param amplitude  Amplitude of each sinusoid (default: 1.0).
    @return mne.io.RawArray Synthetic single-channel raw object.
    """
    times = np.arange(0, duration, 1.0 / sfreq)
    data = sum(amplitude * np.sin(2 * np.pi * f * times) for f in freqs)
    info = mne.create_info(ch_names=["EEG000"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data[np.newaxis, :], info, verbose=False)


def make_raw_with_channels(ch_names, sfreq=250.0, duration=60.0):
    """Create a multi-channel Raw object with random EEG data.

    Used for testing channel selection, removal, referencing, and segmentation.

    @param ch_names  List of channel name strings.
    @param sfreq  Sampling rate in Hz (default: 250.0).
    @param duration  Signal duration in seconds (default: 60.0).
    @return mne.io.RawArray Synthetic multi-channel raw object.
    """
    n_times = int(sfreq * duration)
    data = np.random.randn(len(ch_names), n_times) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def band_power(raw, fmin, fmax):
    """Return mean PSD power in a frequency band.

    @param raw  MNE Raw object to analyse.
    @param fmin  Lower bound of frequency band (Hz).
    @param fmax  Upper bound of frequency band (Hz).
    @return float Mean power across the band.
    """
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    return float(psd.get_data().mean())


# ====== Tests for load_config =======


## Tests that load_config correctly reads a YAML file.
def test_load_config(tmp_path):
    # Write a minimal config file to a temp location
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data_path: /some/path\n")

    config = load_config(str(config_file))

    assert config["data_path"] == "/some/path"


## Tests that load_config raises an error when the file does not exist.
def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


# ====== Tests for load_edf =======


## Tests that load_edf returns an MNE Raw object with expected attributes.
def test_load_edf_returns_raw(tmp_path):
    # Mock mne.io.read_raw_edf to avoid needing a real .edf file in tests
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG1", "EEG2"]
    mock_raw.info = {"sfreq": 256}
    mock_raw.times = [0, 1, 2, 3]

    with patch(
        "src.eeg_cnn_lstm.preprocessing.preprocessing.mne.io.read_raw_edf",
        return_value=mock_raw,
    ):
        result = load_edf("fake_file.edf")

    assert result.ch_names == ["EEG1", "EEG2"]
    assert result.info["sfreq"] == 256


# ====== Tests for clean_channel_names =======


## Tests that "EEG " prefix and "-REF" suffix are stripped and capitalization applied.
def test_clean_channel_names_strips_prefix_and_suffix():
    raw = make_raw_with_channels(["EEG FP1-REF", "EEG CZ-REF", "EEG O1-REF"])
    result = clean_channel_names(raw)
    assert result.ch_names == ["Fp1", "Cz", "O1"]


## Tests that channels without prefix or suffix are only capitalized.
def test_clean_channel_names_no_prefix_suffix():
    raw = make_raw_with_channels(["FP1", "CZ", "O1"])
    result = clean_channel_names(raw)
    assert result.ch_names == ["Fp1", "Cz", "O1"]


## Tests that clean_channel_names returns the same Raw object (in-place).
def test_clean_channel_names_returns_same_object():
    raw = make_raw_with_channels(["EEG FP1-REF"])
    result = clean_channel_names(raw)
    assert result is raw


# ====== Tests for remove_non_eeg_channels =======


## Tests that EKG, EMG, and ocular channels are dropped.
def test_remove_non_eeg_channels_drops_excluded_types():
    raw = make_raw_with_channels(["Fp1", "EKG", "EMG", "ROC", "LOC", "Cz"])
    result = remove_non_eeg_channels(raw)
    assert "EKG" not in result.ch_names
    assert "EMG" not in result.ch_names
    assert "ROC" not in result.ch_names
    assert "LOC" not in result.ch_names


## Tests that standard EEG channels are preserved after removal.
def test_remove_non_eeg_channels_preserves_eeg():
    raw = make_raw_with_channels(["Fp1", "Cz", "O1", "EKG"])
    result = remove_non_eeg_channels(raw)
    assert "Fp1" in result.ch_names
    assert "Cz" in result.ch_names
    assert "O1" in result.ch_names


## Tests that purely numeric channel names are dropped.
def test_remove_non_eeg_channels_drops_numeric():
    raw = make_raw_with_channels(["Fp1", "1", "22"])
    result = remove_non_eeg_channels(raw)
    assert "1" not in result.ch_names
    assert "22" not in result.ch_names
    assert "Fp1" in result.ch_names


# ====== Tests for select_1020_channels =======


## Tests that only standard 10-20 channels are retained.
def test_select_1020_channels_keeps_standard():
    ch_names = ["Fp1", "Fp2", "Fz", "Cz", "Oz", "ExtraChannel"]
    raw = make_raw_with_channels(ch_names)
    result = select_1020_channels(raw)
    assert "ExtraChannel" not in result.ch_names
    assert all(ch in ["Fp1", "Fp2", "Fz", "Cz"] for ch in result.ch_names)


## Tests that a full 19-channel montage is retained when all channels are present.
def test_select_1020_channels_full_montage():
    all_channels = [
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
    raw = make_raw_with_channels(all_channels)
    result = select_1020_channels(raw)
    assert len(result.ch_names) == 19


## Tests that a warning is printed when fewer than 19 channels are found.
def test_select_1020_channels_warns_on_missing(capsys):
    raw = make_raw_with_channels(["Fp1", "Cz"])
    select_1020_channels(raw)
    captured = capsys.readouterr()
    assert "Warning" in captured.out


# ====== filter_raw tests ======


## Tests that the notch filter reduces 60 Hz power by at least 99%.
def test_notch_filter_attenuates_60hz():
    raw = make_synthetic_raw(freqs=[60.0])
    power_before = band_power(raw, fmin=59.0, fmax=61.0)
    filter_raw(raw)
    power_after = band_power(raw, fmin=59.0, fmax=61.0)
    assert power_after < power_before * 0.01


## Tests that the high-pass filter reduces sub-0.3 Hz drift power by at least 99%.
def test_highpass_filter_attenuates_slow_drift():
    raw = make_synthetic_raw(freqs=[0.1], duration=120.0)  # Longer duration to better capture low frequencies
    power_before = band_power(raw, fmin=0.05, fmax=0.2)
    filter_raw(raw)
    power_after = band_power(raw, fmin=0.05, fmax=0.2)
    assert power_after < power_before * 0.01


## Tests that core EEG band power (1-40 Hz) is preserved after filtering (>90%).
def test_filter_preserves_eeg_band():
    raw = make_synthetic_raw(freqs=[10.0, 20.0, 40.0])
    power_before = band_power(raw, fmin=1.0, fmax=45.0)
    filter_raw(raw)
    power_after = band_power(raw, fmin=1.0, fmax=45.0)
    assert power_after > power_before * 0.90


## Tests that filter_raw returns the same Raw object it received (in-place).
def test_filter_returns_same_raw_object():
    raw = make_synthetic_raw(freqs=[10.0])
    result = filter_raw(raw)
    assert result is raw


## Tests that filter_raw works with non-default filter parameters.
def test_filter_raw_custom_params():
    raw = make_synthetic_raw(freqs=[50.0, 0.05])
    # Should not raise with custom notch and highpass values
    result = filter_raw(raw, notch_freq=50.0, highpass_freq=1.0)
    assert result is raw


# ====== Tests for resample_raw =======


## Tests that a 512 Hz recording is downsampled to 250 Hz.
def test_resample_raw_downsamples_512():
    raw = make_raw_with_channels(["Fp1"], sfreq=512.0)
    result = resample_raw(raw, target_sfreq=250.0)
    assert result.info["sfreq"] == 250.0


## Tests that a 256 Hz recording is resampled to 250 Hz.
def test_resample_raw_downsamples_256():
    raw = make_raw_with_channels(["Fp1"], sfreq=256.0)
    result = resample_raw(raw, target_sfreq=250.0)
    assert result.info["sfreq"] == 250.0


## Tests that a recording already at the target rate is returned unchanged.
def test_resample_raw_skips_if_already_target():
    raw = make_raw_with_channels(["Fp1"], sfreq=250.0)
    result = resample_raw(raw, target_sfreq=250.0)
    assert result is raw
    assert result.info["sfreq"] == 250.0


# ====== Tests for apply_common_average_montage =======


## Tests that the mean across channels is approximately zero after re-referencing.
def test_average_montage_zero_mean():
    raw = make_raw_with_channels(["Fp1", "Cz", "O1", "Fz"])
    result = apply_common_average_montage(raw)
    channel_means = result.get_data().mean(axis=0)
    assert np.allclose(channel_means, 0, atol=1e-12)


## Tests that apply_common_average_montage returns the same Raw object (in-place).
def test_average_montage_returns_same_object():
    raw = make_raw_with_channels(["Fp1", "Cz", "O1"])
    result = apply_common_average_montage(raw)
    assert result is raw


# ====== Tests for segment_raw =======


## Tests that segment_raw produces epochs of the correct shape.
def test_segment_raw_epoch_shape():
    ch_names = [
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
    raw = make_raw_with_channels(ch_names, sfreq=250.0, duration=60.0)
    epochs = segment_raw(raw, window_sec=10.0, overlap_sec=5.0)
    _, n_channels, n_times = epochs.get_data().shape
    assert n_channels == 19
    assert n_times == 2500  # 10s * 250 Hz


## Tests that segment_raw produces more than one epoch from a long recording.
def test_segment_raw_multiple_epochs():
    raw = make_raw_with_channels(["Fp1"], sfreq=250.0, duration=60.0)
    epochs = segment_raw(raw, window_sec=10.0, overlap_sec=5.0)
    assert len(epochs) > 1


## Tests that segment_raw respects a custom window length.
def test_segment_raw_custom_window():
    raw = make_raw_with_channels(["Fp1"], sfreq=250.0, duration=60.0)
    epochs = segment_raw(raw, window_sec=5.0, overlap_sec=0.0)
    _, _, n_times = epochs.get_data().shape
    assert n_times == 1250  # 5s * 250 Hz


# ====== Tests for extract_label =======


## Tests that a path containing 'normal' returns label 0.
def test_extract_label_normal():
    path = Path("/data/edf/train/normal/01_tcp_ar/file.edf")
    assert extract_label(path) == 0


## Tests that a path containing 'abnormal' returns label 1.
def test_extract_label_abnormal():
    path = Path("/data/edf/train/abnormal/01_tcp_ar/file.edf")
    assert extract_label(path) == 1


## Tests that a path containing neither 'normal' nor 'abnormal' raises ValueError.
def test_extract_label_invalid_path():
    path = Path("/data/edf/train/unknown/file.edf")
    with pytest.raises(ValueError):
        extract_label(path)


# ====== Tests for save_epochs =======


## Tests that save_epochs writes a .npy file to the output directory.
def test_save_epochs_writes_npy(tmp_path):
    raw = make_raw_with_channels(["Fp1", "Cz", "O1"], sfreq=250.0, duration=60.0)
    epochs = segment_raw(raw, window_sec=10.0, overlap_sec=5.0)
    edf_file = Path("train/normal/test_recording.edf")

    save_epochs(epochs, edf_file, tmp_path, label=0)

    assert (tmp_path / "test_recording_epochs.npy").exists()


## Tests that the saved array shape matches the epochs object.
def test_save_epochs_array_shape(tmp_path):
    raw = make_raw_with_channels(["Fp1", "Cz", "O1"], sfreq=250.0, duration=60.0)
    epochs = segment_raw(raw, window_sec=10.0, overlap_sec=5.0)
    edf_file = Path("train/normal/test_recording.edf")

    save_epochs(epochs, edf_file, tmp_path, label=0)

    data = np.load(tmp_path / "test_recording_epochs.npy")
    assert data.shape == epochs.get_data().shape


## Tests that save_epochs returns a manifest row with the correct keys and values.
def test_save_epochs_returns_manifest_row(tmp_path):
    raw = make_raw_with_channels(["Fp1", "Cz", "O1"], sfreq=250.0, duration=60.0)
    epochs = segment_raw(raw, window_sec=10.0, overlap_sec=5.0)
    edf_file = Path("train/abnormal/test_recording.edf")

    row = save_epochs(epochs, edf_file, tmp_path, label=1)

    assert row["filename"] == "test_recording_epochs.npy"
    assert row["label"] == 1
    assert row["n_epochs"] == len(epochs)
    assert row["sfreq"] == 250.0
