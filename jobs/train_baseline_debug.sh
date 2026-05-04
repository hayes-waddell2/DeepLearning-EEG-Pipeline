#!/bin/bash -l
#SBATCH --job-name=eeg-cnn-lstm-debug
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=debug                  # ← confirm exact name
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/baseline_smoke/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/baseline_smoke/slurm_%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
RUN_DIR=/shared/rc/eeg-cnn-lstm/runs/baseline_smoke

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

python -u -m src.eeg_cnn_lstm.models.train_b --config configs/baseline_smoke.yaml

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"