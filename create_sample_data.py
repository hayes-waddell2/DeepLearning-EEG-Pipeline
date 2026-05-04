## @package create_sample_data
# Generates a small synthetic EEG dataset in TUH EEG Abnormal Corpus directory format.
#
#
# Generated with the assistance of Claude (Anthropic), April 2026.
#
# Creates synthetic .edf files that can be used to test and demonstrate the
# preprocessing pipeline without requiring credentialed access to the full TUH
# dataset. Signals include realistic EEG characteristics (alpha oscillations,
# 60 Hz line noise, slow drift) so that the notch, high-pass, and CAR steps
# have visible effects.
#
# Output directory structure mirrors the TUH corpus layout expected by
# preprocessing.py and extract_label():
#
#   <output>/edf/train/normal/01_tcp_ar/sample_normal_train_000.edf
#   <output>/edf/train/abnormal/01_tcp_ar/sample_abnormal_train_000.edf
#   <output>/edf/eval/normal/01_tcp_ar/sample_normal_eval_000.edf
#   <output>/edf/eval/abnormal/01_tcp_ar/sample_abnormal_eval_000.edf
#
# Usage:
#   python create_sample_data.py
#   python create_sample_data.py --output data/raw
#   --n_normal 3 --n_abnormal 3 --duration 65

import argparse
import numpy as np
import pyedflib
from pathlib import Path

# TUH EEG Abnormal Corpus channel names (19 standard 10-20 channels in TUH format).
# clean_channel_names() strips "EEG " and "-REF" then .capitalize(), yielding:
#   "EEG FP1-REF" -> "Fp1", "EEG CZ-REF" -> "Cz", etc.
TUH_EEG_CHANNELS = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F7-REF",
    "EEG F3-REF",
    "EEG FZ-REF",
    "EEG F4-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG C3-REF",
    "EEG CZ-REF",
    "EEG C4-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG P3-REF",
    "EEG PZ-REF",
    "EEG P4-REF",
    "EEG T6-REF",
    "EEG O1-REF",
    "EEG O2-REF",
]

# Non-EEG channels included to exercise remove_non_eeg_channels().
NON_EEG_CHANNELS = ["EKG", "EMG", "PHOTIC PH"]

ALL_CHANNELS = TUH_EEG_CHANNELS + NON_EEG_CHANNELS


## Generates a multi-channel synthetic EEG signal array.
#
# Combines coloured noise, alpha oscillations, 60 Hz line noise, slow drift,
# and (for abnormal recordings) high-amplitude delta waves. All amplitudes
# are in Volts so they can be written directly to EDF.
#
# @param n_times int Number of time samples.
# @param sfreq float Sampling frequency in Hz.
# @param label str "normal" or "abnormal".
# @param seed int Random seed for reproducibility.
# @return np.ndarray Float64 array of shape (n_channels, n_times) in Volts.
def generate_signals(n_times, sfreq, label, seed=42):
    rng = np.random.RandomState(seed)
    times = np.arange(n_times) / sfreq
    n_eeg = len(TUH_EEG_CHANNELS)

    # Base: low-amplitude Gaussian noise (~10 µV)
    data = rng.randn(n_eeg, n_times) * 10e-6

    # Alpha rhythm (10 Hz, ~20 µV) on posterior channels (O1, O2)
    alpha = 20e-6 * np.sin(2 * np.pi * 10.0 * times)
    data[17] += alpha  # O1-REF
    data[18] += alpha  # O2-REF

    # 60 Hz power-line noise (~5 µV) on all channels — removed by notch filter
    line_noise = 5e-6 * np.sin(2 * np.pi * 60.0 * times)
    data += line_noise

    # Slow drift (0.1 Hz, ~50 µV) on frontal channels — removed by high-pass filter
    drift = 50e-6 * np.sin(2 * np.pi * 0.1 * times)
    data[:3] += drift

    if label == "abnormal":
        # High-amplitude delta waves (1–4 Hz, ~150 µV) simulating abnormal activity
        for ch_idx in range(n_eeg):
            freq = rng.uniform(1.0, 4.0)
            phase = rng.uniform(0, 2 * np.pi)
            data[ch_idx] += 150e-6 * np.sin(2 * np.pi * freq * times + phase)

    # Non-EEG channels: larger amplitude noise
    non_eeg = rng.randn(len(NON_EEG_CHANNELS), n_times) * 50e-6
    return np.vstack([data, non_eeg])


## Saves a signal array to an EDF file using pyedflib.
#
# All channels are written with a ±500 µV physical range, which comfortably
# covers the synthetic signals. The channel labels match TUH format so that
# the preprocessing pipeline can parse them correctly.
#
# @param data np.ndarray Array of shape (n_channels, n_times) in Volts.
# @param ch_names list[str] List of channel label strings.
# @param sfreq float Sampling frequency in Hz.
# @param filepath Path Destination .edf file path.
def save_edf(data, ch_names, sfreq, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_channels = len(ch_names)
    data_uv = data * 1e6  # Convert Volts to µV for EDF physical units

    with pyedflib.EdfWriter(str(filepath), n_channels) as f:
        channel_headers = []
        for ch in ch_names:
            channel_headers.append(
                {
                    "label": ch,
                    "dimension": "uV",
                    "sample_frequency": int(sfreq),
                    "physical_min": -500.0,
                    "physical_max": 500.0,
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
            )
        f.setSignalHeaders(channel_headers)
        f.writeSamples(data_uv)

    print(f"  Saved: {filepath}")


## Creates a complete synthetic sample dataset in TUH directory format.
#
# For each combination of split (train/eval) and label (normal/abnormal),
# generates n_normal or n_abnormal EDF files and saves them under the
# appropriate subdirectory. Each file uses a deterministic seed so that
# results are reproducible.
#
# @param output_dir str or Path Root output directory (e.g. "data/raw").
# @param n_normal int Number of normal recordings per split.
# @param n_abnormal int Number of abnormal recordings per split.
# @param duration float Recording duration in seconds.
# @param sfreq float Sampling frequency in Hz.
def create_sample_dataset(
    output_dir="data/raw",
    n_normal=3,
    n_abnormal=3,
    duration=65.0,
    sfreq=256.0,
):
    output_dir = Path(output_dir)
    n_times = int(sfreq * duration)

    for split in ["train", "eval"]:
        for label in ["normal", "abnormal"]:
            n_files = n_normal if label == "normal" else n_abnormal
            split_dir = output_dir / "edf" / split / label / "01_tcp_ar"

            print(f"\nGenerating {n_files} {label} recordings for split='{split}'...")
            for i in range(n_files):
                # Deterministic seed per (split, label, index) combination
                seed = abs(hash(f"{split}{label}{i}")) % (2**31)
                data = generate_signals(n_times, sfreq, label, seed=seed)
                filename = f"sub{label[0]}{split[0]}{i:03d}_s001_t000.edf"
                save_edf(data, ALL_CHANNELS, sfreq, split_dir / filename)

    print(f"\nSample dataset written to: {output_dir / 'edf'}")
    print(f"  Train: {n_normal} normal, {n_abnormal} abnormal")
    print(f"  Eval:  {n_normal} normal, {n_abnormal} abnormal")
    print(
        f"\nPass the train split to preprocessing.py with:\n"
        f"  python src/preprocessing/preprocessing.py "
        f"--input {output_dir / 'edf' / 'train'} --output data/processed"
    )


## Parses command-line arguments.
#
# @return argparse.Namespace Parsed arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic EEG dataset in TUH corpus format."
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Root output directory (default: data/raw)",
    )
    parser.add_argument(
        "--n_normal",
        type=int,
        default=3,
        help="Number of normal recordings per split (default: 3)",
    )
    parser.add_argument(
        "--n_abnormal",
        type=int,
        default=3,
        help="Number of abnormal recordings per split (default: 3)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=65.0,
        help=(
            "Recording duration in seconds (default: 65.0). "
            "Must be >10s for at least one epoch.",
        ),
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=256.0,
        help="Sampling frequency in Hz (default: 256.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_sample_dataset(
        output_dir=args.output,
        n_normal=args.n_normal,
        n_abnormal=args.n_abnormal,
        duration=args.duration,
        sfreq=args.sfreq,
    )
