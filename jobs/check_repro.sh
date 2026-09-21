#!/bin/bash -l
#SBATCH --job-name=tuab-check-repro
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-cpu
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/phase0/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/phase0/slurm_%x_%j.err
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g

# Phase 0 / Step 1 wrap-up: updated preprocessing.py must reproduce the
# converted float16 uV files from raw EDFs. Extra args go to the Python script.

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project

. /tools/spack/share/spack/setup-env.sh
spack unload --all 2>/dev/null || true
spack load /khlktry
source "${REPO_DIR}/.venv/bin/activate"
cd "${REPO_DIR}"

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Started: $(date)"

python -u -m scripts.check_preprocessing_repro "$@"

echo "Finished: $(date)"