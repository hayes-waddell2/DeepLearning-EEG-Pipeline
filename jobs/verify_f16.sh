#!/bin/bash -l
#SBATCH --job-name=tuab-verify-f16
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/phase0/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/phase0/slurm_%x_%j.err
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=48g

# Phase 0 / Step 1 verification: baseline checkpoint on float64 V vs float16 uV data.

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets

. /tools/spack/share/spack/setup-env.sh
spack unload --all 2>/dev/null || true
spack load /khlktry
source "${REPO_DIR}/.venv/bin/activate"
cd "${REPO_DIR}"

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Started: $(date)"
nvidia-smi

python -u -m scripts.verify_f16_checkpoint \
    --old-data-dir "${DATA}/tuab/train/train" \
    --new-data-dir "${DATA}/tuab_f16uv/train/train" \
    --num-workers 8 \
    "$@"

echo "Finished: $(date)"