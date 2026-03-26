import os
import mne
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["data"]["raw_data_path"]

# Walk through training directory
edf_files = []
labels = []

for label in ["normal", "abnormal"]:
    label_path = os.path.join(data_path, label)

    for root, dirs, files in os.walk(label_path):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
                labels.append(label)

# Check how many files we found
print(f"\nTotal files found: {len(edf_files)}")

# Take one sample
sample_file = edf_files[0]
sample_label = labels[0]

print(f"\nSample file: {sample_file}")
print(f"Label: {sample_label}")

# Load EEG file
raw = mne.io.read_raw_edf(sample_file, preload=False)

# "Head" equivalent
print("\n===== BASIC INFO =====")
print(raw)

print("\n===== CHANNELS =====")
print(raw.info["ch_names"])

print("\n===== SAMPLING RATE =====")
print(raw.info["sfreq"])

duration = raw.n_times / raw.info["sfreq"]
print("\n===== DURATION (seconds) =====")
print(duration)

# Optional: first few signal values
data, times = raw[:, :10]

print("\n===== FIRST SIGNAL VALUES =====")
print(data)
