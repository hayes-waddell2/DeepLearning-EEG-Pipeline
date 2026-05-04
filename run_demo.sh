#!/usr/bin/env bash
# scripts/run_demo.sh — full pipeline from raw EDFs to model results.

set -euo pipefail

echo "=== 0/4: Cleaning old data ==="
rm -rf data/raw/edf data/processed

echo "=== 1/4: Generating synthetic raw EDFs ==="
python create_sample_data.py

echo ""
echo "=== 2/4: Preprocessing train split ==="
python src/preprocessing/preprocessing.py --input data/raw/edf/train --output data/processed

echo ""
echo "=== 3/4: Preprocessing eval split ==="
python src/preprocessing/preprocessing.py --input data/raw/edf/eval  --output data/processed

echo ""
echo "=== 4/4: Training + evaluation ==="
python -m src.eeg_cnn_lstm.models.train_b --config demo.yaml

echo ""
echo "=== Demo complete ==="