#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mit_normal
#SBATCH --output=/dev/stdout         # Send SLURM output to stdout (captured by exec below)
#SBATCH --error=/dev/stderr          # Send SLURM errors to stderr (captured by exec below)

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

GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH:-data/configs/generation_spec.yaml}"
GENERATION_OVERWRITE="${GENERATION_OVERWRITE:-false}"

EXPERIMENT_SLUG="${1:?missing experiment_slug}"

OVERWRITE_FLAG=()
if [[ "${GENERATION_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi


DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

pixi run python -u run_generation_pipeline.py \
  --spec_path "${GENERATION_SPEC_PATH}" \
  --run_request \
  --experiment_slug "${EXPERIMENT_SLUG}" \
  "${OVERWRITE_FLAG[@]}"
