# Abnormal EEG Classifier

End-to-end deep learning pipeline for classifying abnormal EEG recordings using CNN+LSTM and CNN+Transformer architectures, trained on the TUH EEG Abnormal Corpus.

---

## Requirements

- Python 3.11
- Conda

### Instalation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/[TODO: your-username]/capstone-project.git
cd capstone-project
conda env create -f environment.yml
conda activate eeg-env
```

Alternatively, install dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## Data Access

### TUH EEG Abnormal Corpus (Full Dataset_)

The full dataset is provided by the Temple University Hospital (TUH) EEG Corpus and requires credentialed access.

1. Request access at [https://isip.piconepress.com/projects/nedc/html/tuh_eeg/](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).
2. Once approved, download the **TUH EEG Abnormal Corpus (TUAB)** with rsync using your provided credentials.
3. Sotre the data in a secure directory.
   
---

## Data Preparation

### Preporcessing

The full preprocessing pipeline, `preprocessing.py` segments raw EEG recordings, applies filtering methods, and outputs fixed-length numpy arrays with label ready for model input. 

To preproces the full dataset (from the project root):

```bash
python src/preprocessing.py --config.yaml
```

Processed data will be written to `data/processed/`. 

### Verifying Processed Data

Run the data unit tests to confirm that preprocessing completed correctly and that output are in the expected format. 

# TODO:

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






