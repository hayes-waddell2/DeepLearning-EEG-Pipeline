## @package tests.test_preprocess
# Unit tests for the preprocessing module.

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.eeg_cnn_lstm.preprocessing.preprocess import load_config, load_edf


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


## Tests that load_edf returns an MNE Raw object with expected attributes.
def test_load_edf_returns_raw(tmp_path):
    # Mock mne.io.read_raw_edf to avoid needing a real .edf file in tests
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG1", "EEG2"]
    mock_raw.info = {"sfreq": 256}
    mock_raw.times = [0, 1, 2, 3]

    with patch(
        "src.eeg_cnn_lstm.preprocessing.preprocess.mne.io.read_raw_edf",
        return_value=mock_raw,
    ):
        result = load_edf("fake_file.edf")

    assert result.ch_names == ["EEG1", "EEG2"]
    assert result.info["sfreq"] == 256
