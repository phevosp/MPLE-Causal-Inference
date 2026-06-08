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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  local candidate=""
  for candidate in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}" "${SCRIPT_DIR}/.."; do
    [[ -n "${candidate}" ]] || continue
    if ! candidate="$(cd "${candidate}" 2>/dev/null && pwd)"; then
      continue
    fi
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/pixi.toml" || -f "${candidate}/pyproject.toml" ]]; then
        printf "%s\n" "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "Could not locate repo root containing pixi.toml or pyproject.toml." >&2
  exit 1
}
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-data/configs/fits_spec.yaml}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"

EXPERIMENT_SLUG="${1:?missing experiment_slug}"
VARIANT_SLUG="${2:?missing variant_slug}"

OVERWRITE_FLAG=()
if [[ "${FIT_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

# Ensure the log directory exists
DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

pixi run python -u run_fit_pipeline.py \
  --manifest_path "${GENERATION_MANIFEST_PATH}" \
  --fits_spec_path "${FITS_SPEC_PATH}" \
  --run_request \
  --experiment_slug "${EXPERIMENT_SLUG}" \
  --variant_slug "${VARIANT_SLUG}" \
  "${OVERWRITE_FLAG[@]}"
