#!/bin/bash -l
#SBATCH --job-name=eeg-ablshape
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.out
#SBATCH --error=/shared/rc/eeg-cnn-lstm/runs/slurm_%x_%j.err
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64g
#
# Parameter-efficiency ablation, job 1 of 2: the shape sweep.
#
# Fourteen points at three seeds, 42 runs. Everything held at the selected
# configuration with one capacity axis moving at a time, which is what the HPO
# scatter could not give us -- its 42 points varied dropout, learning rate and
# three capacity axes simultaneously, and 27 of them sat at two parameter
# counts.
#
# Budget. Under the trimmed protocol the centre config runs ~10 min. Runtime
# tracks LSTM hidden size far more than conv width, because the recurrent pass
# is sequential over 156 timesteps while the convolutions parallelise: in the
# search, hidden 256 cost 26 min against hidden 64 at 16 min, while width 1.5
# and width 1.0 were indistinguishable. Estimated ~147 min per seed, ~7.5 h
# total. The 9 h internal timeout sits inside the 10 h wall so the report is
# always written.
#
# Seeds are the outer loop inside the runner, so a timeout leaves every point
# measured at one or two seeds rather than two thirds of the points measured at
# three. A partial sweep with every point present is still a usable curve; a
# truncated one is not.
#
# Usage:
#   sbatch jobs/ablation_shape.sh
#   sbatch --export=ALL,SEEDS=42,1337 jobs/ablation_shape.sh    # shorter
#   sbatch --export=ALL,OUT=/path/to/elsewhere jobs/ablation_shape.sh

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train

CONFIG="${CONFIG:-configs/ablation_base.yaml}"
CONFIGS_JSON="${CONFIGS_JSON:-configs/ablation_shape.json}"
OUT="${OUT:-/shared/rc/eeg-cnn-lstm/runs/ablation-shape}"
TIMEOUT_H="${TIMEOUT_H:-9.0}"
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
echo "Timeout:  ${TIMEOUT_H} h (SLURM wall 10 h)"
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
echo "Next, before interpreting anything -- check that stopping behaviour does"
echo "not track capacity. If models stop differently as size changes, the arm"
echo "measures training budget rather than capacity:"
echo "  python -m scripts.compare_configs --runs ${OUT}/runs.json --margin 0.01"
echo "========================================================================"
