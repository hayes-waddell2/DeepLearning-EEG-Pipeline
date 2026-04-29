#!/bin/bash -l
#SBATCH --job-name=tuh-eeg-preprocess-debug
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-cpu
#SBATCH --output=/shared/rc/eeg-cnn-lstm/preprocess_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/preprocess_%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g

# Debug-partition test of the preprocessing pipeline.
# Usage:
#   sbatch jobs/preprocess_debug.sh train
#   sbatch jobs/preprocess_debug.sh eval

set -euo pipefail

SPLIT="${1:-train}"
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "eval" ]]; then
    echo "ERROR: split must be 'train' or 'eval' (got '${SPLIT}')" >&2
    exit 1
fi

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
DATA_ROOT=/shared/rc/eeg-cnn-lstm/data
INPUT_DIR="${DATA_ROOT}/raw-datasets/tuab/v3.0.1/edf/${SPLIT}"
OUTPUT_DIR="${DATA_ROOT}/processed-datasets/tuab/${SPLIT}"

mkdir -p "${OUTPUT_DIR}"
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: input dir does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

. /tools/spack/share/spack/setup-env.sh
spack unload --all 2>/dev/null || true
spack load /khlktry

VENV="${REPO_DIR}/.venv"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV}" >&2
    exit 1
fi
source "${VENV}/bin/activate"

cd "${REPO_DIR}"

echo "============================================================"
echo "Job:     ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:    $(hostname)"
echo "Split:   ${SPLIT}"
echo "Input:   ${INPUT_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "Python:  $(which python)"
echo "Started: $(date)"
echo "============================================================"

python -u src/preprocessing/preprocessing.py \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
PREPROCESS_DEBUG_EOF

chmod +x jobs/preprocess_debug.sh