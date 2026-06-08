#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=mit_normal
#SBATCH --output=/dev/stdout         # Send SLURM output to stdout (captured by exec below)
#SBATCH --error=/dev/stderr          # Send SLURM errors to stderr (captured by exec below)

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-data/configs/fits_spec.yaml}"
CV_SPEC_PATH="${CV_SPEC_PATH:-}"
SEARCH_SLUG="${SEARCH_SLUG:-}"
FIT_MODE="${FIT_MODE:-standard}"
SPLIT_KIND="${SPLIT_KIND:-}"
NUM_FOLDS="${NUM_FOLDS:-}"
OUTER_NUM_FOLDS="${OUTER_NUM_FOLDS:-}"
TEST_FOLD_ID="${TEST_FOLD_ID:-}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"

EXPERIMENT_SLUG="${1:?missing experiment_slug}"
VARIANT_SLUG="${2:-}"
POSITIONAL_SEARCH_SLUG="${3:-}"

if [[ "${FIT_MODE}" != "standard" && "${FIT_MODE}" != "outer_masked" ]]; then
  echo "FIT_MODE must be 'standard' or 'outer_masked', got '${FIT_MODE}'." >&2
  exit 1
fi

if [[ "${FIT_MODE}" == "standard" && -z "${VARIANT_SLUG}" ]]; then
  echo "missing variant_slug" >&2
  exit 1
fi

if [[ "${FIT_MODE}" == "outer_masked" ]]; then
  if [[ -z "${CV_SPEC_PATH}" ]]; then
    echo "CV_SPEC_PATH is required when FIT_MODE=outer_masked." >&2
    exit 1
  fi
  if [[ -n "${POSITIONAL_SEARCH_SLUG}" ]]; then
    SEARCH_SLUG="${POSITIONAL_SEARCH_SLUG}"
  fi
  if [[ -z "${SEARCH_SLUG}" ]]; then
    echo "SEARCH_SLUG is required when FIT_MODE=outer_masked." >&2
    exit 1
  fi
fi

OVERWRITE_FLAG=()
if [[ "${FIT_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

split_args=()
if [[ -n "${SPLIT_KIND}" ]]; then
  split_args+=(--split_kind "${SPLIT_KIND}")
fi
if [[ -n "${NUM_FOLDS}" ]]; then
  split_args+=(--num_folds "${NUM_FOLDS}")
fi
if [[ -n "${OUTER_NUM_FOLDS}" ]]; then
  split_args+=(--outer_num_folds "${OUTER_NUM_FOLDS}")
fi
if [[ -n "${TEST_FOLD_ID}" ]]; then
  split_args+=(--test_fold_id "${TEST_FOLD_ID}")
fi

# Ensure the log directory exists
DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

if [[ "${FIT_MODE}" == "standard" ]]; then
  pixi run python -u run_fit_pipeline.py \
    --manifest_path "${GENERATION_MANIFEST_PATH}" \
    --fits_spec_path "${FITS_SPEC_PATH}" \
    --run_request \
    --experiment_slug "${EXPERIMENT_SLUG}" \
    --variant_slug "${VARIANT_SLUG}" \
    "${OVERWRITE_FLAG[@]}"
else
  variant_args=()
  if [[ -n "${VARIANT_SLUG}" ]]; then
    variant_args=(--variant_slug "${VARIANT_SLUG}")
  fi
  pixi run python -u run_fit_pipeline.py \
    --fit_mode outer_masked \
    --manifest_path "${GENERATION_MANIFEST_PATH}" \
    --cv_spec_path "${CV_SPEC_PATH}" \
    --search_slug "${SEARCH_SLUG}" \
    --run_request \
    --experiment_slug "${EXPERIMENT_SLUG}" \
    "${variant_args[@]}" \
    "${split_args[@]}" \
    "${OVERWRITE_FLAG[@]}"
fi
