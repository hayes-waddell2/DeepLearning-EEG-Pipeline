#!/bin/bash -l
#SBATCH --job-name=eeg-final
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.err
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64g
#
# One-shot evaluation on the held-out TUAB partition.
#
# Two configurations, one run each, pre-specified before any held-out number
# was seen: the selected configuration (246,497 parameters) and the reduced
# one (65,457). The selected configuration is the headline; the reduced one is
# the parameter-efficiency comparison. Nothing is selected here.
#
# Each run trains on all of pool E with early stopping on the tuning pool,
# then scores the held-out partition with the restored best checkpoint. At
# roughly 12 min per run plus staging, 2.5 h is generous.
#
# The operating point is NOT re-derived on the held-out data. The runner finds
# the 95% sensitivity threshold on the tuning pool and applies it unchanged,
# so the specificity reported is one the deployed system could actually set.
#
# Usage:
#   sbatch jobs/final_eval.sh

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train

CONFIG="${CONFIG:-configs/ablation_base.yaml}"
CONFIGS_JSON="${CONFIGS_JSON:-configs/cv.json}"
OUT="${OUT:-/shared/rc/eeg-cnn-lstm/runs/final}"
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
echo "Configs:  ${CONFIGS_JSON}"
echo "Out:      ${OUT}"
echo "Started:  $(date)"
echo "========================================================================"

# ---- Protocol guard --------------------------------------------------------
if grep -A2 '^lr_schedule:\|^  lr_schedule:' "${CONFIG}" | grep -q 'enabled: true'; then
    echo "ABORT: ${CONFIG} has lr_schedule enabled." >&2
    exit 1
fi
echo "Protocol guard: LR scheduler off. OK."

# ---- Held-out partition guard ----------------------------------------------
# Without these keys the final-eval block never runs and the stage silently
# produces a tuning run. The runner also raises in that case, but failing here
# costs seconds instead of half an hour.
for key in eval_manifest eval_data_dir; do
    if ! grep -q "${key}" "${CONFIG}"; then
        echo "ABORT: ${CONFIG} has no data.${key}; the held-out pass needs it." >&2
        exit 1
    fi
done
echo "Held-out guard: eval_manifest and eval_data_dir present. OK."

# ---- Stage data node-local -------------------------------------------------
echo "Staging training data to ${LOCAL}/train ..."
mkdir -p "${LOCAL}/train"
cp -r "${SRC_DATA}/." "${LOCAL}/train/"
echo "Staged $(find "${LOCAL}/train" -name '*.npy' | wc -l) files."

# ---- Pre-flight ------------------------------------------------------------
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo ""
    echo "---- Pre-flight ----"
    python -m scripts.final_eval \
        --config "${CONFIG}" \
        --configs-json "${CONFIGS_JSON}" \
        --out-dir "${OUT}" \
        --train-data-dir "${LOCAL}/train" \
        --dry-run || {
        echo "PRE-FLIGHT FAILED - not starting the evaluation." >&2
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
python -m scripts.final_eval \
    --config "${CONFIG}" \
    --configs-json "${CONFIGS_JSON}" \
    --out-dir "${OUT}" \
    --train-data-dir "${LOCAL}/train"

echo ""
echo "========================================================================"
echo "Finished: $(date)"
echo "Report:   ${OUT}/final_eval.txt"
echo "========================================================================"
