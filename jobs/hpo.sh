#!/bin/bash -l
#SBATCH --job-name=eeg-hpo
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

# Overnight hyperparameter search.
#
# Runs the verification suite first and refuses to start the search if any
# check fails. The whole point of the verifier is that a ten-hour unattended
# job should not be the thing that discovers a typo, so an early exit here is
# a success, not a failure.
#
# Environment variables (pass with --export=ALL,VAR=value):
#   CONFIG       base YAML                (default: configs/step2_earlystop.yaml)
#   N_TRIALS     trial budget             (default: 70)
#   TIMEOUT_H    wall-clock budget, hours (default: 10.0; must stay under
#                                          --time minus staging and reporting)
#   STUDY        study name / resume key  (default: eeg-hpo-v1)
#   SKIP_VERIFY  set to 1 to skip the checks (only when resuming a run whose
#                                          checks already passed today)
#
#   sbatch jobs/hpo.sh
#   sbatch --export=ALL,N_TRIALS=100,TIMEOUT_H=10 jobs/hpo.sh
#
# Resuming after an interruption: submit the same command again. The study is
# keyed by STUDY in a SQLite file on shared storage, so completed trials are
# picked up rather than repeated.

set -euo pipefail

REPO_DIR=/shared/rc/eeg-cnn-lstm/waddell-capstone-project
SRC_DATA=/shared/rc/eeg-cnn-lstm/data/processed-datasets/tuab_f16uv/train/train
HPO_ROOT=/shared/rc/eeg-cnn-lstm/runs/hpo

CONFIG="${CONFIG:-configs/step2_earlystop.yaml}"
N_TRIALS="${N_TRIALS:-70}"
TIMEOUT_H="${TIMEOUT_H:-10.0}"
STUDY="${STUDY:-eeg-hpo-v1}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

OUT_DIR="${HPO_ROOT}/${STUDY}"
STORAGE="sqlite:///${HPO_ROOT}/${STUDY}.db"
LOCAL="/tmp/${USER}/${SLURM_JOB_ID}"
GPU_LOG_PID=""

cleanup() {
    [[ -n "${GPU_LOG_PID}" ]] && kill "${GPU_LOG_PID}" 2>/dev/null || true
    rm -rf "${LOCAL}"
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}" "${LOCAL}"

. /tools/spack/share/spack/setup-env.sh
spack unload --all 2>/dev/null || true
spack load /khlktry
source "${REPO_DIR}/.venv/bin/activate"
cd "${REPO_DIR}"

echo "============================================================"
echo "Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:     $(hostname)"
echo "Config:   ${CONFIG}"
echo "Study:    ${STUDY}"
echo "Storage:  ${STORAGE}"
echo "Budget:   ${N_TRIALS} trials / ${TIMEOUT_H} h"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Started:  $(date)"
echo "============================================================"
nvidia-smi

python -c "import optuna; print(f'optuna {optuna.__version__}')" || {
    echo "optuna is not installed in the venv. Run: pip install optuna"
    exit 1
}

# ---- Stage data to node-local disk ----
echo "Staging ${SRC_DATA} -> ${LOCAL}/train ..."
t0=$(date +%s)
find "${SRC_DATA}" -maxdepth 1 -name '*.npy' -print0 \
    | xargs -0 -P 8 -n 32 cp -t "${LOCAL}/train/" 2>/dev/null \
    || { mkdir -p "${LOCAL}/train"; find "${SRC_DATA}" -maxdepth 1 -name '*.npy' -print0 \
         | xargs -0 -P 8 -n 32 cp -t "${LOCAL}/train/"; }
t1=$(date +%s)
export STAGE_SECONDS=$((t1 - t0))
echo "Staged $(du -sh "${LOCAL}/train" | cut -f1) in ${STAGE_SECONDS}s"

# ---- Verification: do not spend ten hours on a broken setup ----
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo
    echo "------------------- verification -------------------"
    python -u -m scripts.hpo_verify --mode offline || {
        echo "Offline checks FAILED. Search not started."; exit 2; }
    python -u -m scripts.hpo_verify --mode gpu \
        --config "${CONFIG}" --max-steps 30 \
        --out-dir "${LOCAL}/verify" --train-data-dir "${LOCAL}/train" || {
        echo "GPU checks FAILED. Search not started."; exit 3; }
    echo "------------------ checks passed -------------------"
    echo
else
    echo "SKIP_VERIFY=1: checks skipped."
fi

# ---- GPU utilization log ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv -l 30 > "${OUT_DIR}/gpu_util.csv" &
GPU_LOG_PID=$!

# ---- The search ----
python -u -m scripts.hpo \
    --config "${CONFIG}" \
    --study-name "${STUDY}" \
    --storage "${STORAGE}" \
    --out-dir "${OUT_DIR}" \
    --n-trials "${N_TRIALS}" \
    --timeout-hours "${TIMEOUT_H}" \
    --train-data-dir "${LOCAL}/train" \
    --top-k 5

echo "============================================================"
echo "Finished: $(date)"
echo "Report:   ${OUT_DIR}/top_k.json"
echo "Trials:   ${OUT_DIR}/trials.csv"
echo "Resume:   sbatch --export=ALL,STUDY=${STUDY} jobs/hpo.sh"
echo "============================================================"