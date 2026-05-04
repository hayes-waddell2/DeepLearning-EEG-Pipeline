## @package data_explore
# Exploratory analysis of the TUH EEG Abnormal Corpus dataset.
#
# Scans all .edf files across train and eval splits, collects
# metadata from each file header, and prints aggregate statistics
# to describe the full dataset without loading signal data into memory.

import os
import mne
import yaml
import numpy as np
from collections import Counter

mne.set_log_level("WARNING")  # Suppress verbose MNE output


## Loads configuration settings from a YAML file.
#
# @param config_path str Path to the YAML config file.
# @return dict Configuration parameters.
def load_config(config_path="configs/preprocessing.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


## Collects file paths and labels from a directory split.
#
# Walks a split directory (e.g. train/) and collects all .edf
# file paths along with their normal/abnormal label and split name.
#
# @param split_path str Path to the split directory.
# @param split_name str Name of the split (e.g. "train", "eval").
# @return list of dict Each dict contains file, label, and split.
def collect_files(split_path, split_name):
    records = []
    for label in ["normal", "abnormal"]:
        label_path = os.path.join(split_path, label)
        if not os.path.exists(label_path):
            continue
        for root, _, files in os.walk(label_path):
            for file in files:
                if file.endswith(".edf"):
                    records.append(
                        {
                            "file": os.path.join(root, file),
                            "label": label,
                            "split": split_name,
                        }
                    )
    return records


## Reads metadata from a single .edf file without loading signal data.
#
# @param record dict Dict containing file path, label, and split.
# @return dict Metadata including duration, sfreq, and channel info.
def read_metadata(record):
    raw = mne.io.read_raw_edf(record["file"], preload=False, verbose=False)
    duration = raw.n_times / raw.info["sfreq"]
    return {
        "file": record["file"],
        "label": record["label"],
        "split": record["split"],
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names,
        "duration_sec": duration,
        "duration_min": duration / 60,
    }


## Prints a section header for readability.
#
# @param title str Title to display.
def section(title):
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")


## Entry point for dataset exploration.
#
# Collects metadata from all files and prints aggregate statistics
# including file counts, durations, sampling rates, and channel info.
def main():
    config = load_config()
    data_path = config["data"]["raw_data_path"]

    # Collect all files across train and eval splits
    all_records = []
    for split in ["train", "eval"]:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            records = collect_files(split_path, split)
            all_records.extend(records)
            print(f"Collected {len(records)} records from {split} split.")

    print(f"Total records collected: {len(all_records)}")
    print("Reading metadata from all files (this may take a moment)...")

    # Read metadata for all records
    metadata = []
    for i, record in enumerate(all_records):
        if i % 100 == 0:
            print(f" Processing file {i + 1}/{len(all_records)}")
        metadata.append(read_metadata(record))

    durations = [m["duration_sec"] for m in metadata]
    sfreqs = [m["sfreq"] for m in metadata]
    n_channels = [m["n_channels"] for m in metadata]

    # --- Dataset Overview ---
    section("Dataset Overview")
    total = len(metadata)
    for split in ["train", "eval"]:
        split_data = [m for m in metadata if m["split"] == split]
        normal = sum(1 for m in split_data if m["label"] == "normal")
        abnormal = sum(1 for m in split_data if m["label"] == "abnormal")
        print(f"\n {split.upper()} split: {len(split_data)} files")
        print(f"  Normal: {normal} ({normal / len(split_data) * 100:.1f}%)")
        print(f"  Abnormal: {abnormal} ({abnormal / len(split_data) * 100:.1f}%)")

    # --- Duration Statistics ---
    section("Duration Statistics")
    print(f" Min: {np.min(durations):.1f}s")
    print(f" Max: {np.max(durations):.1f}s")
    print(f" Mean: {np.mean(durations):.1f}s")
    print(f" Std: {np.std(durations):.1f}s")
    total_hours = np.sum(durations) / 3600
    print(f" Total data: {total_hours:.1f} hours")

    # --- Sampling Rate Statistics ---
    section("Sampling Rate (Hz)")
    sfreq_counts = Counter(sfreqs)
    for rate, count in sorted(sfreq_counts.items()):
        print(f" {rate} Hz: {count} files ({count / total * 100:.1f}%)")

    # --- Channel Statistics ---
    section("Channel Statistics")
    ch_counts = Counter(n_channels)
    for n, count in sorted(ch_counts.items()):
        print(f" {n} channels: {count} files ({count / total * 100:.1f}%)")

    # --- Channel Name Consistency ---
    section("Channel Names")
    unique_montages = Counter(tuple(m["ch_names"]) for m in metadata)
    print(f" Unique channel montages: {len(unique_montages)}")
    print(f"\n Most common montages ({unique_montages.most_common(1)[0][1]} files):")
    for ch in unique_montages.most_common(1)[0][0]:
        print(f"  {ch}")

    # --- Sample File ---
    section("Sample File")
    sample = metadata[0]
    print(f" File: {sample['file']}")
    print(f" Label: {sample['label']}")
    print(f" Split: {sample['split']}")
    print(f" Sfreq: {sample['sfreq']} Hz")
    print(f" Channels: {sample['n_channels']}")
    print(f" Duration: {sample['duration_sec']:.1f}s ({sample['duration_min']:.1f}m)")


if __name__ == "__main__":
    main()
