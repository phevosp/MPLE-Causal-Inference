#!/bin/bash
#SBATCH --job-name=posterior-predictive
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mit_normal
#SBATCH --output=/dev/stdout         # Send SLURM output to stdout (captured by exec below)
#SBATCH --error=/dev/stderr          # Send SLURM errors to stderr (captured by exec below)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GEN_MANIFEST="${GEN_MANIFEST:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FIT_MANIFEST="${FIT_MANIFEST:-experiments/SyntheticHybridExperiments/fit_manifest.csv}"
TARGET_PAIRS_PATH="${TARGET_PAIRS_PATH:-data/configs/posterior_predictive_target_pairs.csv}"
POSTERIOR_PREDICTIVE_SPEC_PATH="${POSTERIOR_PREDICTIVE_SPEC_PATH:-data/configs/posterior_predictive_spec.yaml}"
POSTERIOR_PREDICTIVE_OVERWRITE="${POSTERIOR_PREDICTIVE_OVERWRITE:-false}"

EXPERIMENT_NAME="${1:?missing experiment_name}"
SOURCE_TYPE="${2:?missing source_type}"
VARIANT_NAME="${3-}"
INTERVENTION_SOURCE="${4:?missing intervention_source}"
INTERVENTION_NAME="${5-}"
RUN_NAME="${6:?missing run_name}"

OVERWRITE_FLAG=()
if [[ "${POSTERIOR_PREDICTIVE_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path "${GEN_MANIFEST}" \
  --fit_manifest_path "${FIT_MANIFEST}" \
  --target_pairs_path "${TARGET_PAIRS_PATH}" \
  --spec_path "${POSTERIOR_PREDICTIVE_SPEC_PATH}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --source_type "${SOURCE_TYPE}" \
  --variant_name "${VARIANT_NAME}" \
  --intervention_source "${INTERVENTION_SOURCE}" \
  --intervention_name "${INTERVENTION_NAME}" \
  --run_name "${RUN_NAME}" \
  "${OVERWRITE_FLAG[@]}"
