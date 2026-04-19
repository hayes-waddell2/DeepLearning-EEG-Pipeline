# Abnormal EEG Classifier

End-to-end deep learning pipeline for classifying abnormal EEG recordings using CNN+LSTM and CNN+Transformer architectures, trained on the TUH EEG Abnormal Corpus.

---

## Requirements

- Python 3.11
- pip

**Dependencies** (`requirements.txt`)

| Package | Purpose |
| mne | EEG file I/O, processing |
| numpy | Array operations |
| pyedflib | Writing synthetic EEG files |
| pyyaml | Config file parsing |
| pytest | Test runner|

### Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/[TODO: your-username]/capstone-project.git
cd capstone-project
pip install -r requirements.txt
```

---

## Data Access

### TUH EEG Abnormal Corpus (Full Dataset)

The full dataset is provided by the Temple University Hospital (TUH) EEG Corpus and requires credentialed access.

1. Request access at [https://isip.piconepress.com/projects/nedc/html/tuh_eeg/](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).
2. Once approved, download the **TUH EEG Abnormal Corpus (TUAB)** with rsync using your provided credentials.
3. Store the data in a secure directory.

### Sample Data

A script (`create_sample_data.py`) is provided to generate a small synthetic dataset in the TUH EEG Abnormal Corupus directory format. The synthetic files mimic realistic EEG content so all preprocessing steps have visible effects.

```bash
python create_sample_data
```

This creates the following structure under `data/raw/`:

```text
data/raw/edf/
├── train/
│   ├── normal/01_tcp_ar/    # 3 synthetic normal recordings
│   └── abnormal/01_tcp_ar/  # 3 synthetic abnormal recordings
└── eval/
├── normal/01_tcp_ar/    # 3 synthetic normal recordings
└── abnormal/01_tcp_ar/  # 3 synthetic abnormal recordings
```
   
---

## Data Preparation

### Preprocessing

The preprocessing pipeline (`src/preprocessing/preprocessing.py`) processes raw `.edf` files through the following steps:

load → clean channel names → remove non-EEG channels → select 19 standard 10-20 channels → notch filter (60 Hz) → high-pass filter (0.3 Hz) → resample to 250 Hz → common average reference → segment into 10-second epochs with 50% overlap → save as `.npy`

Preprocess the train split:

```bash
python src/preprocessing/preprocessing.py \
  --input data/raw/edf/train \
  --output data/processed
```

Preprocess the eval split:

```bash
python src/preprocessing/preprocessing.py \
  --input data/raw/edf/eval \
  --output data/processed
```

Processed files are written to `data/processed/train/` and `data/processed/eval/`, with a manifest CSV mapping each `.npy` file to its label, epoch count, and sampling rate.

### Config

Pipeline paths and parameters are controlled via `config.yaml`:

```yaml
data:
  raw_data_path: data/raw/edf/train
  processed_data_path: data/processed
```

### Verifying Processed Data

Run the data integrity tests to confirm the sample data is correctly formatted and compatible with the preprocessing pipeline:

```bash
pytest tests/test_sample_data.py -v
```

These tests verify that the expected directory structure is present, all EDF files are readable, channel names match TUH format, labels are correctly encoded in the directory structure, and the full preprocessing pipeline produces epochs of the expected shape `(n_epochs, 19, 2500)`.

---

## Model Training

# TODO:

---

## Data Exploration and Visualization

# TODO:

---

## Expected Results:

# TODO:

---

## Running All Tests

```bash
pytest tests/ -v
```

---

## Running Full End-To-End Pipeline

# TODO:

---

## Project Structure

```text
capstone-project
├── README.md
├── __pycache__
├── config.yaml
├── conftest.py
├── create_sample_data.py
├── data
│   ├── processed
│   │   ├── eval
│   │   ├── eval_manifest.csv
│   │   ├── train
│   │   └── train_manifest.csv
│   └── raw
│       └── edf
│           ├── eval
│           │   ├── abnormal
│           │   │   └── 01_tcp_ar
│           │   └── normal
│           │       └── 01_tcp_ar
│           └── train
│               ├── abnormal
│               │   └── 01_tcp_ar
│               └── normal
│                   └── 01_tcp_ar
├── environment.yml
├── jobs
├── psd_after.png
├── psd_before.png
├── pyproject.toml
├── requirements.txt
├── results
│   └── data_exploration.txt
├── src
│   ├── eeg_cnn_lstm
│   │   ├── models
│   │   └── utils
│   ├── eeg_cnn_lstm.egg-info
│   ├── preprocessing
│   │   ├── data_explore.py
│   │   ├── debug_filter_check.py
│   │   └── preprocessing.py
│   └── visualize.py
└── tests
    ├── test_preprocess.py
    └── test_sample_data.py
```



