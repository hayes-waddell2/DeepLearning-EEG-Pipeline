## @package tests.test_preprocess
# Unit tests for the preprocessing module.

import pytest
import numpy as np
import mne
from unittest.mock import patch, MagicMock
from src.eeg_cnn_lstm.preprocessing.preprocessing import (
    filter_raw,
    load_config,
    load_edf,
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
    raw = make_synthetic_raw(freqs=[0.1])
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
