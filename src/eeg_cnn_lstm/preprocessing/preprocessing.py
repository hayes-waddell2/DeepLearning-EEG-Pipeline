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


## Standardizes EEG channel names by stripping common EDF prefixes and suffixes.
#
# Removes "EEG " prefixes and "-REF" suffixes that appear in many EDF recordings,
# then applies title-case capitalization to match the standard 10-20 naming
# convention (e.g., "EEG FP1-REF" becomes "Fp1").
#
# @param raw mne.io.Raw Preloaded raw EEG object (modified in-place).
# @return mne.io.Raw The raw object with renamed channels.
def clean_channel_names(raw):
    print("Cleaning channel names")

    cleaned_names = {}

    for ch in raw.ch_names:
        new_ch = ch

        # Remove common prefixes/suffixes
        if "EEG " in new_ch:
            new_ch = new_ch.replace("EEG ", "")
        if "-REF" in new_ch:
            new_ch = new_ch.replace("-REF", "")

        new_ch = new_ch.capitalize()  # Standardize capitalization

        cleaned_names[ch] = new_ch

    raw.rename_channels(cleaned_names)
    print(f"  Cleaned channel names: {list(cleaned_names.values())}")
    return raw


## Removes non-EEG physiological and artifact channels from the recording.
#
# Drops channels associated with cardiac (EKG/ECG), muscular (EMG), ocular
# (ROC/LOC), and stimulus/derived (PHOTIC, IBI, BURSTS, SUPPR) signals, as
# well as any channels whose names are purely numeric.
#
# @param raw mne.io.Raw Preloaded raw EEG object (modified in-place).
# @return mne.io.Raw The raw object with non-EEG channels removed.
def remove_non_eeg_channels(raw):
    print("Removing non-EEG channels")

    exclude_types = [
        "EKG",
        "ECG",
        "EMG",
        "ROC",
        "LOC",
        "PHOTIC",
        "IBI",
        "BURSTS",
        "SUPPR",
    ]

    bad_channels = [
        ch
        for ch in raw.ch_names
        if any(ex_type in ch.upper() for ex_type in exclude_types)
        or ch.replace(" ", "").isdigit()  # Exclude channels that are just numbers
    ]

    print(f"  Excluding channels: {bad_channels}")

    raw.drop_channels(bad_channels)
    return raw


# Select standard 10-20 EEG channels from the recording, if available.
STANDARD_1020_CHANNELS = [
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


## Retains only the 19 standard 10-20 system EEG channels.
#
# Intersects the recording's channel list with STANDARD_1020_CHANNELS and
# keeps only those present. Logs a warning if fewer than 19 channels are
# found, as downstream processing assumes a full 19-channel montage.
# Must be called after clean_channel_names so channel names match the
# expected 10-20 format.
#
# @param raw mne.io.Raw Preloaded raw EEG object (modified in-place).
# @return mne.io.Raw The raw object containing only available 10-20 channels.
def select_1020_channels(raw):
    print("Selecting standard 10-20 channels")

    available_channels = [ch for ch in STANDARD_1020_CHANNELS if ch in raw.ch_names]

    print(f"  Available channels: {available_channels}")

    if len(available_channels) < 19:
        print("  Warning: Less than 19 standard channels found.")

    raw.pick(available_channels)

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
