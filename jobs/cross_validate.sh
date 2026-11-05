#!/bin/bash -l
#SBATCH --job-name=eeg-cv
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.err
#SBATCH --time=0-07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64g
#
# Patient-level 10-fold cross-validation: the reported-performance stage.
#
# Two frozen configurations at ten folds, 20 runs. Nothing is selected here.
# Every setting was chosen against the tuning pool T; this measures how the
# frozen choices generalize to patients no model in a fold has trained on.
#
# Budget. The centre ran ~10.9 min and small_2 ~9.9 min on the tuning pass
# under the trimmed protocol. A fold trains on 90% of E rather than all of it
# and adds one scoring pass over the held-out fold, so ~11-12 min each,
# ~230 min total. The 5.5 h internal timeout sits inside the 7 h wall so the
# report is always written.
#
# Configurations are the outer loop inside the runner, so a timeout leaves one
# complete ten-fold estimate rather than two half-finished ones. A partial
# cross-validation is not an estimate.
#
# Usage:
#   sbatch jobs/cross_validate.sh
#   sbatch --export=ALL,FOLDS=0,1,2 jobs/cross_validate.sh        # short check
#   sbatch --export=ALL,OUT=/path/elsewhere jobs/cross_validate.sh

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train

CONFIG="${CONFIG:-configs/ablation_base.yaml}"
CONFIGS_JSON="${CONFIGS_JSON:-configs/cv.json}"
OUT="${OUT:-/shared/rc/eeg-cnn-lstm/runs/cv}"
TIMEOUT_H="${TIMEOUT_H:-5.5}"
FOLDS="${FOLDS:-}"
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
echo "Timeout:  ${TIMEOUT_H} h (SLURM wall 7 h)"
echo "Started:  $(date)"
echo "========================================================================"

# ---- Protocol guard --------------------------------------------------------
# The LR scheduler has silently come back once already, through a cleanup
# commit. It fires on validation noise and cost 0.023 pAUC when measured. The
# cross-validation runs must use the same recipe every earlier stage used.
if grep -A2 '^lr_schedule:\|^  lr_schedule:' "${CONFIG}" | grep -q 'enabled: true'; then
    echo "ABORT: ${CONFIG} has lr_schedule enabled. Regenerate it without --lr-schedule." >&2
    exit 1
fi
echo "Protocol guard: LR scheduler off. OK."

# ---- Splits guard ----------------------------------------------------------
# A fold index is silently ignored without data.splits_csv, and the run becomes
# an ordinary tuning run scored on T. The runner treats a missing bench["test"]
# as a failure, but catching it here saves ten hours of finding out.
if ! grep -q 'splits_csv' "${CONFIG}"; then
    echo "ABORT: ${CONFIG} has no data.splits_csv; fold selection needs it." >&2
    exit 1
fi
echo "Splits guard: data.splits_csv present. OK."

# ---- Stage data node-local -------------------------------------------------
echo "Staging training data to ${LOCAL}/train ..."
mkdir -p "${LOCAL}/train"
cp -r "${SRC_DATA}/." "${LOCAL}/train/"
echo "Staged $(find "${LOCAL}/train" -name '*.npy' | wc -l) files."

# ---- Pre-flight ------------------------------------------------------------
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo ""
    echo "---- Pre-flight: building every override tree ----"
    python -m scripts.cross_validate \
        --config "${CONFIG}" \
        --configs-json "${CONFIGS_JSON}" \
        --out-dir "${OUT}" \
        --train-data-dir "${LOCAL}/train" \
        --dry-run || {
        echo "PRE-FLIGHT FAILED - not starting cross-validation." >&2
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
FOLD_ARG=()
[[ -n "${FOLDS}" ]] && FOLD_ARG=(--folds "${FOLDS}")

python -m scripts.cross_validate \
    --config "${CONFIG}" \
    --configs-json "${CONFIGS_JSON}" \
    --out-dir "${OUT}" \
    --train-data-dir "${LOCAL}/train" \
    --timeout-hours "${TIMEOUT_H}" \
    "${FOLD_ARG[@]}"

echo ""
echo "========================================================================"
echo "Finished: $(date)"
echo "Raw:      ${OUT}/runs.json"
echo "Report:   ${OUT}/cv_report.txt"
echo ""
echo "Both configurations saw the same folds, so compare them paired on fold:"
echo "  python -m scripts.compare_configs --runs ${OUT}/runs.json \\"
echo "      --block fold --margin 0.01"
echo "========================================================================"