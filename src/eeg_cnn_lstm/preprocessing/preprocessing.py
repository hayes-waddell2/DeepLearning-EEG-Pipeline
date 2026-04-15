## @package preprocessing
# Preprocessing pipeline for TUH EEG Abnormal Corpus dataset.
#
# Loads raw .edf EEG files and applies signal preprocessing.
# steps include notch filtering, high-pass filtering and common average re-referencing.

import mne
import yaml
import argparse
import numpy as np
import csv
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


## Resamples raw EEG data to a target sampling frequency.
#
# Skips resampling if the recording is already at the target rate.
# Default target is 250 Hz, matching the majority of TUH recordings.
#
# @param raw mne.io.Raw Preloaded raw EEG object.
# @param target_sfreq float Target sampling frequency in Hz (default: 250.0).
# @return mne.io.Raw The resampled raw object.
def resample_raw(raw, target_sfreq=250.0):
    current_sfreq = raw.info["sfreq"]

    if current_sfreq == target_sfreq:
        print(f"Sampling rate already at {target_sfreq} Hz, skipping resampling")
        return raw

    print(f"Resampling from {current_sfreq} Hz to {target_sfreq} Hz")
    raw.resample(target_sfreq, verbose=False)
    return raw


## Applies a common average montage to the EEG recording.
#
# Subtracts the mean signal across all channels from each channel.
# Should be called after resampling and channel selection so that
# only the 19 standard 10-20 channels contribute to the average.
#
# @param raw mne.io.Raw Preloaded raw EEG object.
# @return mne.io.Raw The re-referenced raw object.
def apply_common_average_montage(raw):
    print("Applying common average montage")
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    return raw


## Segments a continuous raw EEG recording into fixed-length epochs with 50% overlap.
#
# Uses MNE's make_fixed_length_epochs to create non-overlapping or overlapping
# windows. At 250 Hz, a 10-second window produces epochs of shape (19, 2500).
#
# @param raw mne.io.Raw Preprocessed raw EEG object.
# @param window_sec float Epoch length in seconds (default: 10.0).
# @param overlap_sec float Overlap between consecutive epochs in seconds (default: 5.0).
# @return mne.Epochs Fixed-length epochs object.
def segment_raw(raw, window_sec=10.0, overlap_sec=5.0):
    print(
        f"Segmenting raw data into {window_sec}-second epochs"
        f"with {overlap_sec}-second overlap"
    )
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=window_sec,
        overlap=overlap_sec,
        preload=True,
        verbose=False,
    )
    print(f"  Created {len(epochs)} epochs of shape {epochs.get_data().shape}")
    return epochs


## Extracts the binary label from a TUH EEG Abnormal Corpus file path.
#
# The corpus encodes the label in the directory structure:
#   edf/train/normal/...  -> 0
#   edf/train/abnormal/.. -> 1
#
# @param edf_path Path Path to the .edf file.
# @return int 0 for normal, 1 for abnormal.
# @throws ValueError If neither 'normal' nor 'abnormal' appears in the path.
def extract_label(edf_path):
    parts = Path(edf_path).parts
    if "abnormal" in parts:
        return 1
    elif "normal" in parts:
        return 0
    else:
        raise ValueError(f"Cannot determine label from path: {edf_path}")


## Saves segmented EEG epochs to disk as a NumPy array and returns a manifest row.
#
# Writes a single .npy file per recording containing all epochs:
#   - <stem>_epochs.npy : float64 array of shape
#     (n_epochs, n_channels, n_timepoints)
#
# Returns a metadata dict to be aggregated into the split manifest CSV
# by the caller.
#
# @param epochs mne.Epochs Segmented epochs object from segment_raw().
# @param edf_file Path Path to the original .edf file, used to derive
#     the output filename.
# @param output_path Path Directory where the .npy file will be written.
# @param label int Binary label for the recording (0=normal, 1=abnormal).
# @return dict Row dict with keys: filename, label, n_epochs, sfreq.
def save_epochs(epochs, edf_file, out_path, label):
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    stem = edf_file.stem
    npy_file = out_path / f"{stem}_epochs.npy"
    np.save(npy_file, data)
    print(f"Saved epochs to {npy_file}")

    return {
        "filename": npy_file.name,
        "label": label,
        "n_epochs": len(epochs),
        "sfreq": epochs.info["sfreq"],
    }


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
# Loads config and resolves input/output paths from CLI args or config.yaml.
# Detects the dataset split (train or eval) from the input path and writes
# all processed epochs to the corresponding split subdirectory.
#
# For each .edf file, applies the full preprocessing pipeline:
# load -> clean channels -> remove non-EEG -> select 10-20 -> filter
# -> resample -> average reference -> segment -> save
#
# After all files are processed, writes a manifest CSV to the output root
# mapping each saved .npy file to its label, epoch count, and sampling rate.
def main():
    args = parse_args()
    config = load_config(args.config)

    # Use CLI args if provided, otherwise fall back to config.yaml
    input_path = Path(args.input if args.input else config["data"]["raw_data_path"])
    output_path = Path(
        args.output if args.output else config["data"]["processed_data_path"]
    )
    output_path.mkdir(parents=True, exist_ok=True)

    split = "train" if "train" in input_path.parts else "eval"
    split_output = output_path / split
    split_output.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        edf_files = [input_path]
    elif input_path.is_dir():
        edf_files = list(input_path.rglob("*.edf"))
        print(f"Found {len(edf_files)} .edf files")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    manifest_rows = []
    for edf_file in edf_files:
        raw = load_edf(edf_file)
        raw = clean_channel_names(raw)
        raw = remove_non_eeg_channels(raw)
        raw = select_1020_channels(raw)
        raw = filter_raw(raw)
        raw = resample_raw(raw)
        raw = apply_common_average_montage(raw)
        epochs = segment_raw(raw)
        label = extract_label(edf_file)
        row = save_epochs(epochs, edf_file, split_output, label)
        manifest_rows.append(row)

    manifest_path = output_path / f"{split}_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "label", "n_epochs", "sfreq"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
