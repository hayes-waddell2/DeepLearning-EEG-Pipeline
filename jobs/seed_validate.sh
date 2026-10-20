#!/bin/bash -l
#SBATCH --job-name=eeg-seedval
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
#
# Seed validation for the eeg-hpo-v1 top configurations.
#
# The search produced a top-10 whose score spread (0.0100) is inside the
# check-to-check noise (~0.02), so its ranking is a tie rather than an order.
# This job re-runs each configuration from fresh seeds and selects on the mean.
#
# Budget: 10 configurations x 3 seeds = 30 runs. Completed HPO trials averaged
# ~16 min; one configuration (trial 47, lstm_hidden 256) ran ~26 min. Expect
# ~8.5 h. The internal timeout stops cleanly at 10 h, inside the 12 h wall, so
# the report is always written.
#
# Seeds are the OUTER loop in the runner, so a timeout leaves every
# configuration measured at one or two seeds rather than two thirds of them
# measured at three.
#
# Usage:
#   sbatch jobs/seed_validate.sh
#   sbatch --export=ALL,SEEDS=42,1337 jobs/seed_validate.sh     # shorter run
#   sbatch --export=ALL,OUT=runs/seedval-v2 jobs/seed_validate.sh

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train

CONFIG="${CONFIG:-configs/step2_earlystop.yaml}"
CONFIGS_JSON="${CONFIGS_JSON:-configs/seed_validation.json}"
OUT="${OUT:-/shared/rc/eeg-cnn-lstm/runs/seedval}"
TIMEOUT_H="${TIMEOUT_H:-10.0}"
SEEDS="${SEEDS:-}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

LOCAL="/tmp/${USER}/${SLURM_JOB_ID}"
GPU_LOG_PID=""

cleanup() {
    [[ -n "${GPU_LOG_PID}" ]] && kill "${GPU_LOG_PID}" 2>/dev/null || true
    rm -rf "${LOCAL}"
}
trap cleanup EXIT

mkdir -p "${OUT}" "${LOCAL}"
cd "${REPO_DIR}"
source .venv/bin/activate

echo "========================================================================"
echo "Job:      ${SLURM_JOB_ID} on $(hostname)"
echo "Config:   ${CONFIG}"
echo "Set:      ${CONFIGS_JSON}"
echo "Out:      ${OUT}"
echo "Timeout:  ${TIMEOUT_H} h (SLURM wall 12 h)"
echo "Started:  $(date)"
echo "========================================================================"

# ---- Stage data node-local -------------------------------------------------
# Thirty runs over shared storage is thirty times the metadata traffic. Staging
# once costs a few minutes and is repaid on the first configuration.
echo "Staging training data to ${LOCAL}/train ..."
mkdir -p "${LOCAL}/train"
cp -r "${SRC_DATA}/." "${LOCAL}/train/"
echo "Staged $(find "${LOCAL}/train" -name '*.npy' | wc -l) files."

# ---- Pre-flight ------------------------------------------------------------
# Every override tree is built and validated against the real ModelConfig
# before a single GPU-minute is spent. This is the check that caught three
# invalid parameter names before the 10-hour HPO run; the same class of error
# would waste 8 hours here.
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo ""
    echo "---- Pre-flight: building every override tree ----"
    python -m scripts.seed_validate \
        --config "${CONFIG}" \
        --configs-json "${CONFIGS_JSON}" \
        --out-dir "${OUT}" \
        --train-data-dir "${LOCAL}/train" \
        --dry-run || {
        echo "PRE-FLIGHT FAILED - not starting the runs." >&2
        exit 1
    }
    echo "---- Pre-flight passed ----"
fi

# ---- GPU utilization log ---------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv -l 60 > "${OUT}/gpu_util.csv" &
GPU_LOG_PID=$!

# ---- Run -------------------------------------------------------------------
echo ""
SEED_ARG=()
[[ -n "${SEEDS}" ]] && SEED_ARG=(--seeds "${SEEDS}")

python -m scripts.seed_validate \
    --config "${CONFIG}" \
    --configs-json "${CONFIGS_JSON}" \
    --out-dir "${OUT}" \
    --train-data-dir "${LOCAL}/train" \
    --timeout-hours "${TIMEOUT_H}" \
    "${SEED_ARG[@]}"

echo ""
echo "========================================================================"
echo "Finished: $(date)"
echo "Report:   ${OUT}/seed_validation.json"
echo "Table:    ${OUT}/seed_validation.csv"
echo "Raw:      ${OUT}/runs.json"
echo "Re-report without re-running:"
echo "  python -m scripts.seed_validate --report-only --out-dir ${OUT}"
echo "========================================================================"