#!/bin/bash -l
#SBATCH --job-name=eeg-train-staged
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.err
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64g

# Train CNN+LSTM with the float16 uV dataset staged to node-local /tmp.
#
# Environment variables (optional, pass with --export=ALL,VAR=value):
#   CONFIG   YAML config            (default: configs/baseline.yaml)
#   RUN_DIR  output directory       (default: runs/<job-name>_<job-id>)
# Extra arguments are passed to train_b.py, e.g. --num-epochs 1
#
# Phase 0 / Step 2 benchmark:
#   sbatch --job-name=bench-f16-staged jobs/train_staged.sh --num-epochs 1

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train
CONFIG="${CONFIG:-configs/baseline.yaml}"
RUN_DIR="${RUN_DIR:-/shared/rc/eeg-cnn-lstm/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
LOCAL="/tmp/${USER}/${SLURM_JOB_ID}"
GPU_LOG_PID=""

cleanup() {
    [[ -n "${GPU_LOG_PID}" ]] && kill "${GPU_LOG_PID}" 2>/dev/null || true
    rm -rf "${LOCAL}"
}
trap cleanup EXIT

mkdir -p "${RUN_DIR}" "${LOCAL}"

. /tools/spack/share/spack/setup-env.sh
spack unload --all 2>/dev/null || true
spack load /khlktry
source "${REPO_DIR}/.venv/bin/activate"
cd "${REPO_DIR}"

echo "============================================================"
echo "Job:     ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:    $(hostname)"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "Config:  ${CONFIG}"
echo "Run dir: ${RUN_DIR}"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Started: $(date)"
echo "============================================================"
nvidia-smi
df -h /tmp

# ---- Stage data to node-local disk ----
echo "Staging ${SRC_DATA} -> ${LOCAL}/train ..."
t0=$(date +%s)
mkdir -p "${LOCAL}/train"
find "${SRC_DATA}" -maxdepth 1 -name '*.npy' -print0 \
    | xargs -0 -P 8 -n 32 cp -t "${LOCAL}/train/"
t1=$(date +%s)
export STAGE_SECONDS=$((t1 - t0))
echo "Staged $(du -sh "${LOCAL}/train" | cut -f1) in ${STAGE_SECONDS}s"

# ---- GPU utilization log (every 10 s) ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv -l 10 > "${RUN_DIR}/gpu_util.csv" &
GPU_LOG_PID=$!

# ---- Train ----
python -u -m src.eeg_cnn_lstm.models.train_b \
    --config "${CONFIG}" \
    --train-data-dir "${LOCAL}/train" \
    --output-dir "${RUN_DIR}" \
    "$@"

# ---- GPU utilization summary ----
python - "${RUN_DIR}/gpu_util.csv" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], skipinitialspace=True)
u = df.iloc[:, 1].astype(str).str.replace("%", "").str.strip().astype(float)
print(f"GPU utilization: mean {u.mean():.0f}%, median {u.median():.0f}%, "
      f"samples {len(u)} (includes validation and idle start-up)")
EOF

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"