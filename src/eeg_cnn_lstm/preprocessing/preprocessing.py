## @package preprocessing
# Preprocessing pipeline for TUH EEG Abnormal Corpus dataset.
#
# Loads raw .edf EEG files and applies signal preprocessing.
# steps include notch filtering, high-pass filtering and common average re-referencing.

import mne
import yaml
import argparse
from pathlib import Path


## Loads configuration settings from a YAML file.
#
# @param config_path str Path to the YAML config file.
# @return dict Configuration parameters.
# @throws FileNotFoundError If the config file does not exist.
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


## Loads a single .edf EEG file into an MNE Raw object.
#
# Reads the file with preloading so all data is held in memory,
# which is required before filtering can be applied.
#
# @param edf_path str or Path Path to the .edf file.
# @return mne.io.Raw The loaded raw EEG data.
# @throws FileNotFoundError If the .edf file does not exist.
def load_edf(edf_path):
    print(f"Loading: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Log basic recording metadata for verification
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} seconds")
    print(f"  Channel names: {raw.ch_names}")

    return raw


## Applies notch and high-pass filters to a raw EEG recording.
#
# Notch filter removes US power line noise at 60 Hz.
# High-pass filter attenuates slow drift below 0.3 Hz.
# Both filters are applied in-place using MNE's default FIR filter design.
#
# @param raw mne.io.Raw Preloaded raw EEG object (modified in-place).
# @param notch_freq float Notch filter frequency in Hz (default: 60.0).
# @param highpass_freq float High-pass cutoff frequency in Hz (default: 0.3).
# @return mne.io.Raw The filtered raw object (same object, modified in-place).
def filter_raw(raw, notch_freq=60.0, highpass_freq=0.3):
    print(f"Applying notch filter at {notch_freq} Hz")
    raw.notch_filter(notch_freq, verbose=False)

    print(f"Applying high-pass filter at {highpass_freq} Hz")
    raw.filter(
        l_freq=highpass_freq,
        h_freq=None,
        verbose=False,
    )
    return raw


## Parses command-line arguments for the preprocessing script.
#
# Input and output paths are optional and fall back to config.yaml
# values if not provided on the command line.
#
# @return argparse.Namespace Parsed arguments.
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess EEG .edf files")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to .edf file or directory (overrides config)",
    )
    parser.add_argument(
        "--output", default=None, help="Output directory (overrides config)"
    )
    return parser.parse_args()


## Entry point for the preprocessing pipeline.
#
# Loads config and input EEG file(s), applies preprocessing,
# and saves results to the specified output directory.
def main():
    args = parse_args()
    config = load_config(args.config)

    # Use CLI args if provided, otherwise fall back to config.yaml
    input_path = Path(args.input if args.input else config["data"]["raw_data_path"])
    output_path = Path(
        args.output if args.output else config["data"]["processed_data_path"]
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO: add filtering and montage steps after load is verified
    if input_path.is_file():
        raw = load_edf(input_path)
        filter_raw(raw)
    elif input_path.is_dir():
        edf_files = list(input_path.rglob("*.edf"))
        print(f"Found {len(edf_files)} .edf files")
        for edf_file in edf_files:
            raw = load_edf(edf_file)
            filter_raw(raw)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
