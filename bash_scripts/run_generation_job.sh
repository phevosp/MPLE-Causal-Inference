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
