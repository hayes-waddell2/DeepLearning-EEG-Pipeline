# Abnormal EEG Classifier

End-to-end deep learning pipeline for classifying abnormal EEG recordings using CNN+LSTM architecture, trained and validated on the TUH Abnormal EEG Corpus (TUAB). A CNN+Transformer comparison model and an attention-based interpretability module are planned extensions.

---

## Requirements

- Python 3.11
- Optional but recommended for training: NVIDIA GPU with CUDA 12.x driver

**Dependencies** (`requirements.txt`)

| Package | Purpose |
| `mne` | EEG file I/O, processing |
| `numpy` | Array operations |
| `pyedflib` | Writing synthetic EEG files |
| `pyyaml` | Config file parsing |
| `pytest` | Test runner |
| `pandas` | Manifest and tabular data handling |
| `scipy ` | Filtering and statisitcal utilization |
| `scikit-learn`  | Metrics (AUC, F1) for evaluation         |
| `torch`         | Deep learning model and training loop    |
| `loguru`        | Structured logging                       |
| `pyyaml`        | Config file parsing                      |
| `matplotlib`    | Plotting and visualization               |
| `black`         | Code formatting (dev only)               |
| `flake8`        | Lint checks (dev only)                   |


### Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/[TODO: your-username]/capstone-project.git
cd capstone-project

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

This `pip install -e ".[dev]"` command installs the project in editable mode with all development dependencies.

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

Pipeline paths and parameters are controlled via 'configs/preprocessing.yaml`:

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

The CNN+LSTM model lives at `src/eeg_cnn_lstm/models/model_b.py`. The training entry point is `src/eeg_cnn_lstm/models/train_b.py` and is driven entirely by a YAML config.

### Local Demo Run

A small demo configuration runs the full pipeline (~30-60 seconds on GPU) on the synthetic sample data:

```bash
python -m src.eeg_cnn_lstm.models.train_b --config configs/demo.yaml
```

Outputs (best checkpoint and training log) are written to `outputs/demo/`. Becuase the synthetic classes are deliberately sperable, the demo typicaly reports perferct validation accuracy. The run is a smoke test of the pipeline, not a meaningful classification result.

### Cluster Baseline Run

The full TUAB training run is launched via SLURM. The job script `jobs/train_baeline.sh` requests a single A100 GPU and reads `configs/baseline.yaml`:

```bash
sbatch jobs/train_baseline.sh
squeue -u $USER
```

Outputs (best checkpoint, training log, SLURM stdout/stderr) are written to `/shared/rc/eeg-cnn-lstm/runs/baseline_v1/`. Adjust paths in the job script and config for your own cluster setup.

A debug-partition smoke job (`jobs/train_baseline_debug.sh`) is also provided for fast cluster-environment validation against the full data with heavy per-recording subsampling.

---

## Preliminary Results

CNN+LSTM baseline trained on the full TUAB training partitions and evaluated on a 415-subject held-out validation fold:

| Metric         | Value  |
|----------------|--------|
| AUC-ROC        | 0.885  |
| Accuracy       | 0.7705 |
| Sensitivity    | 0.602  |
| Specificity    | 0.953  |
| F1 (positive)  | 0.7314 |

---

## Project Structure

```text
capstone-project/
├── README.md
├── conftest.py                       # pytest path setup
├── pyproject.toml                    # main package + dependencies
├── environment.yml                   # conda spec (cluster users)
├── requirements.txt                  # pip-freeze fallback
├── .gitignore
├── .flake8
├── .github/workflows/                # CI: flake8 + pytest
├── configs/
│   ├── preprocessing.yaml            # preprocessing pipeline config
│   ├── demo.yaml                     # local demo training config
│   ├── baseline.yaml                 # cluster training config
│   └── baseline_smoke.yaml           # cluster smoke-test config
├── data/                             # gitignored
│   ├── raw/                          # raw EDF files
│   └── processed/                    # preprocessed .npy + manifests
├── jobs/                             # SLURM job scripts
│   ├── download_tuh_eeg.sh
│   ├── process_tuh.sh
│   ├── train_baseline_debug.sh
│   └── train_baseline.sh
├── outputs/                          # gitignored: local run artifacts
├── runs/                             # gitignored: cluster run artifacts
├── results/
│   ├── data_exploration.txt
│   ├── processing_summary.md
│   └── figures/                      # committed plots only
├── scripts/                          # standalone helper scripts
│   ├── create_sample_data.py
│   ├── data_explore.py
│   ├── debug_filter_check.py
│   └── run_demo.sh
├── src/
│   ├── eeg_cnn_lstm/
│   │   ├── init.py
│   │   └── models/
│   │       ├── init.py
│   │       ├── model_b.py            # CNN+LSTM architecture
│   │       └── train_b.py            # training entry point
│   ├── preprocessing/
│   │   ├── init.py
│   │   └── preprocessing.py
│   └── utils/
│       ├── init.py
│       ├── dataset.py                # TUABEpochDataset + dataloader factory
│       └── metrics.py                # accuracy, AUC, F1, confusion matrix
└── tests/
├── init.py
├── test_preprocess.py
└── test_sample_data.py
```

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



