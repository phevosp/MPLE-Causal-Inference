#!/bin/bash
#SBATCH --job-name=test-eval
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=mit_normal
#SBATCH --output=/dev/stdout
#SBATCH --error=/dev/stderr

set -euo pipefail

SCRIPT_START_TIME="$(date +%s)"

format_elapsed_time() {
  local elapsed="$1"
  local hours=$((elapsed / 3600))
  local minutes=$(((elapsed % 3600) / 60))
  local seconds=$((elapsed % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

report_script_runtime() {
  local exit_code="$1"
  local end_time
  end_time="$(date +%s)"
  echo "Script runtime: $(format_elapsed_time "$((end_time - SCRIPT_START_TIME))") (exit_code=${exit_code})"
}

trap 'report_script_runtime $?' EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

FIT_MANIFEST_PATH="${FIT_MANIFEST_PATH:-experiments/Synthetic/train_fit_manifest.csv}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
GIBBS_SWEEPS="${GIBBS_SWEEPS:-}"
SEED="${SEED:-}"

if [[ ! -f "${FIT_MANIFEST_PATH}" ]]; then
  echo "FIT_MANIFEST_PATH does not exist: ${FIT_MANIFEST_PATH}" >&2
  exit 1
fi

DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

sampling_args=()
if [[ -n "${NUM_SAMPLES}" ]]; then
  sampling_args+=(--num_samples "${NUM_SAMPLES}")
fi
if [[ -n "${GIBBS_SWEEPS}" ]]; then
  sampling_args+=(--gibbs_sweeps "${GIBBS_SWEEPS}")
fi
if [[ -n "${SEED}" ]]; then
  sampling_args+=(--seed "${SEED}")
fi

pixi run python -u run_test_evaluation.py \
  --fit_manifest_path "${FIT_MANIFEST_PATH}" \
  "${sampling_args[@]}"
