#!/bin/bash
#SBATCH --job-name=cv
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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

# Ensure the log directory exists
DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
CV_SPEC_PATH="${CV_SPEC_PATH:-data/configs/cv_spec.yaml}"
CV_NUM_FOLDS="${CV_NUM_FOLDS:-}"  # Optional override; uses CV spec default if not set
CV_OVERWRITE="${CV_OVERWRITE:-false}"
CV_CONTINUE="${CV_CONTINUE:-false}"
EXECUTION_MODE="${EXECUTION_MODE:-cv}"

EXPERIMENT_SLUG="${1:?missing experiment_slug}"
SEARCH_SLUG="${2:?missing search_slug}"

OVERWRITE_FLAG=()
if [[ "${CV_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

CONTINUE_FLAG=()
if [[ "${CV_CONTINUE}" == "true" ]]; then
  CONTINUE_FLAG=(--continue)
fi

NUM_FOLDS_FLAG=()
if [[ -n "${CV_NUM_FOLDS}" ]]; then
  NUM_FOLDS_FLAG=(--num_folds "${CV_NUM_FOLDS}")
fi

pixi run python -u run_cv_folds.py \
  --generation_manifest_path "${GENERATION_MANIFEST_PATH}" \
  --cv_spec_path "${CV_SPEC_PATH}" \
  --execution_mode "${EXECUTION_MODE}" \
  --run_request \
  --experiment_slug "${EXPERIMENT_SLUG}" \
  --search_slug "${SEARCH_SLUG}" \
  "${NUM_FOLDS_FLAG[@]}" \
  "${CONTINUE_FLAG[@]}" \
  "${OVERWRITE_FLAG[@]}"
