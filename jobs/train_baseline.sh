#!/bin/bash -l
#SBATCH --job-name=eeg-cnn-lstm-baseline
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu             
#SBATCH --gres=gpu:a100:1                
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/baseline_v1/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/baseline_v1/slurm_%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g

# Train CNN+LSTM on TUAB (real data, 80/20 subject-disjoint split).
# Eval set untouched until the model is finalized.

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
RUN_DIR=/shared/rc/eeg-cnn-lstm/runs/baseline_v1

mkdir -p "${RUN_DIR}"

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
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "Python:  $(which python)"
echo "Started: $(date)"
echo "============================================================"
nvidia-smi
echo

python -u -m src.eeg_cnn_lstm.models.train_b --config configs/baseline.yaml

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"