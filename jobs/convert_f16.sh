#!/bin/bash -l
#SBATCH --job-name=tuab-convert-f16
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-cpu
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/phase0/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/phase0/slurm_%x_%j.err
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g

# Phase 0 / Step 1: convert preprocessed TUAB float64 V -> float16 uV.
# Extra args are passed straight to the Python script, e.g.:
#   sbatch jobs/convert_f16.sh --splits train --limit 20 \
#       --dst-root /shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv_test
#   sbatch jobs/convert_f16.sh                       # full run, train + eval

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project

. /tools/spack/share/spack/setup-env.sh
spack unload --all 2>/dev/null || true
spack load /khlktry
source "${REPO_DIR}/.venv/bin/activate"
cd "${REPO_DIR}"

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs"
echo "Started: $(date)"

python -u -m scripts.convert_to_f16_uv --workers "${SLURM_CPUS_PER_TASK}" "$@"

echo "Finished: $(date)"