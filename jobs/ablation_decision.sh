#!/bin/bash -l
#SBATCH --job-name=eeg-abldec
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.err
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64g
#
# Parameter-efficiency ablation, job 2 of 2: the decision points.
#
# A handful of points re-run at five seeds instead of three. These are chosen
# from the shape sweep -- the best point, the smallest point still close to it,
# and the knee on either side -- and they are the only points a conclusion
# rests on, so they are the only ones worth the extra precision.
#
# The claim this job supports is the equivalence one: "the smallest model
# statistically indistinguishable from the best". That needs a variance
# estimate at the specific points being compared, which three seeds does not
# reliably give -- a standard deviation on two degrees of freedom is close to
# worthless, as seed validation demonstrated when trial 61's apparent fourfold
# reproducibility advantage just missed significance.
#
# Generate the point set from the shape results first:
#   python -m scripts.make_ablation_points --set decision \
#       --points center lstm_hidden_32 conv_width_0.5 lstm_layers_1 \
#       --out configs/ablation_decision.json
#
# Budget. Four or five points at five seeds is 20-25 runs. At ~11 min that is
# 3.7-4.6 h; the 5 h internal timeout sits inside the 6 h wall.
#
# Usage:
#   sbatch jobs/ablation_decision.sh
#   sbatch --export=ALL,SEEDS=42,1337,2024 jobs/ablation_decision.sh

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train

CONFIG="${CONFIG:-configs/ablation_base.yaml}"
CONFIGS_JSON="${CONFIGS_JSON:-configs/ablation_decision.json}"
OUT="${OUT:-/shared/rc/eeg-cnn-lstm/runs/ablation-decision}"
TIMEOUT_H="${TIMEOUT_H:-5.0}"
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
echo "Points:   ${CONFIGS_JSON}"
echo "Out:      ${OUT}"
echo "Timeout:  ${TIMEOUT_H} h (SLURM wall 6 h)"
echo "Started:  $(date)"
echo "========================================================================"

# ---- Protocol guard --------------------------------------------------------
# The LR scheduler has silently come back once already, through a cleanup
# commit. It fires on validation noise and cost 0.023 pAUC when measured. If it
# is on here it would fire across all 42 runs and the sweep would be measuring
# the scheduler rather than capacity.
if grep -A2 '^lr_schedule:\|^  lr_schedule:' "${CONFIG}" | grep -q 'enabled: true'; then
    echo "ABORT: ${CONFIG} has lr_schedule enabled. Regenerate it without --lr-schedule." >&2
    exit 1
fi
echo "Protocol guard: LR scheduler off. OK."

# ---- Stage data node-local -------------------------------------------------
echo "Staging training data to ${LOCAL}/train ..."
mkdir -p "${LOCAL}/train"
cp -r "${SRC_DATA}/." "${LOCAL}/train/"
echo "Staged $(find "${LOCAL}/train" -name '*.npy' | wc -l) files."

# ---- Pre-flight ------------------------------------------------------------
# Every override tree is built and checked before a GPU-minute is spent. This
# class of check caught three invalid parameter names before the 10-hour HPO
# run; here it would catch a conv_channels/pool_sizes length mismatch in the
# depth arm, which is the one thing hand-written overrides get wrong.
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo ""
    echo "---- Pre-flight: building every override tree ----"
    python -m scripts.seed_validate \
        --config "${CONFIG}" \
        --configs-json "${CONFIGS_JSON}" \
        --out-dir "${OUT}" \
        --train-data-dir "${LOCAL}/train" \
        --dry-run || {
        echo "PRE-FLIGHT FAILED - not starting the sweep." >&2
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
echo "Raw:      ${OUT}/runs.json"
echo ""
echo "Next -- the equivalence test, with the margin fixed before the runs:"


echo "  python -m scripts.compare_configs --runs ${OUT}/runs.json --margin 0.01"
echo "========================================================================"
