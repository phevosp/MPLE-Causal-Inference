#!/bin/bash
#SBATCH --job-name=mple-tests
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=mit_normal
#SBATCH --output=/dev/stdout         # Send SLURM output to stdout (captured by exec below)
#SBATCH --error=/dev/stderr          # Send SLURM errors to stderr (captured by exec below)

set -euo pipefail

DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH:-data/configs/generation_spec.yaml}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-data/configs/fits_spec.yaml}"
INTERVENTION_LIBRARY_SPEC_PATH="${INTERVENTION_LIBRARY_SPEC_PATH:-data/configs/intervention_library_spec.yaml}"
TARGET_PAIRS_PATH="${TARGET_PAIRS_PATH:-data/configs/posterior_predictive_target_pairs.csv}"
POSTERIOR_PREDICTIVE_SPEC_PATH="${POSTERIOR_PREDICTIVE_SPEC_PATH:-data/configs/posterior_predictive_spec.yaml}"

GENERATION_OVERWRITE="${GENERATION_OVERWRITE:-false}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"
POSTERIOR_PREDICTIVE_OVERWRITE="${POSTERIOR_PREDICTIVE_OVERWRITE:-false}"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SACCT_BIN="${SACCT_BIN:-sacct}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SLEEP_BIN="${SLEEP_BIN:-sleep}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-10}"

GENERATION_SUBMITTER="${GENERATION_SUBMITTER:-submit_generation_jobs.sh}"
FIT_SUBMITTER="${FIT_SUBMITTER:-submit_fit_jobs.sh}"
POSTERIOR_PREDICTIVE_SUBMITTER="${POSTERIOR_PREDICTIVE_SUBMITTER:-submit_posterior_predictive_jobs.sh}"

GENERATION_WORKER_SCRIPT="${GENERATION_WORKER_SCRIPT:-run_generation_job.sh}"
FIT_WORKER_SCRIPT="${FIT_WORKER_SCRIPT:-run_fit_job.sh}"
POSTERIOR_PREDICTIVE_WORKER_SCRIPT="${POSTERIOR_PREDICTIVE_WORKER_SCRIPT:-run_posterior_predictive_job.sh}"

GENERATION_REPORT_JOB_NAME="${GENERATION_REPORT_JOB_NAME:-generation-refresh}"
FIT_REPORT_JOB_NAME="${FIT_REPORT_JOB_NAME:-fit-refresh}"
POSTERIOR_PREDICTIVE_REPORT_JOB_NAME="${POSTERIOR_PREDICTIVE_REPORT_JOB_NAME:-posterior-predictive-report}"

resolve_generation_manifest_path() {
  pixi run python - <<'PY' "${GENERATION_SPEC_PATH}"
import sys
from run_generation_pipeline import generation_manifest_path_for_spec

print(generation_manifest_path_for_spec(sys.argv[1]))
PY
}

resolve_fit_manifest_path() {
  pixi run python - <<'PY' "${FITS_SPEC_PATH}"
import sys
from run_fit_pipeline import fit_manifest_path_for_spec

print(fit_manifest_path_for_spec(sys.argv[1]))
PY
}

GEN_MANIFEST="${GEN_MANIFEST:-$(resolve_generation_manifest_path)}"
FIT_MANIFEST="${FIT_MANIFEST:-$(resolve_fit_manifest_path)}"

get_job_state() {
  local job_id="$1"
  local sacct_output=""
  local state=""
  if sacct_output="$("${SACCT_BIN}" -n -X -P -j "${job_id}" --format State 2>/dev/null)"; then
    state="$(printf "%s\n" "${sacct_output}" | awk -F'|' 'NF {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1; exit}')"
    if [[ -n "${state}" ]]; then
      printf "%s\n" "${state}"
      return 0
    fi
  fi

  local squeue_output=""
  if squeue_output="$("${SQUEUE_BIN}" -h -j "${job_id}" 2>/dev/null)"; then
    if [[ -n "${squeue_output}" ]]; then
      printf "%s\n" "PENDING"
      return 0
    fi
  fi

  printf "\n"
}

wait_for_job() {
  local job_id="$1"
  local stage_name="$2"
  local state=""
  while true; do
    state="$(get_job_state "${job_id}")"
    if [[ "${state}" == COMPLETED* ]]; then
      echo "${stage_name} barrier ${job_id} completed."
      return 0
    fi
    if [[ "${state}" == FAILED* ]] || [[ "${state}" == CANCELLED* ]] || [[ "${state}" == TIMEOUT* ]] || [[ "${state}" == OUT_OF_MEMORY* ]] || [[ "${state}" == BOOT_FAIL* ]] || [[ "${state}" == DEADLINE* ]] || [[ "${state}" == NODE_FAIL* ]] || [[ "${state}" == PREEMPTED* ]] || [[ "${state}" == REVOKED* ]] || [[ "${state}" == SPECIAL_EXIT* ]]; then
      echo "${stage_name} barrier ${job_id} failed with state ${state}." >&2
      exit 1
    fi
    "${SLEEP_BIN}" "${WAIT_POLL_SECONDS}"
  done
}

submit_generation_stage() {
  GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH}" \
  GENERATION_OVERWRITE="${GENERATION_OVERWRITE}" \
  SBATCH_BIN="${SBATCH_BIN}" \
  WORKER_SCRIPT="${GENERATION_WORKER_SCRIPT}" \
  REPORT_JOB_NAME="${GENERATION_REPORT_JOB_NAME}" \
  bash "${GENERATION_SUBMITTER}"
}

submit_fit_stage() {
  GENERATION_MANIFEST_PATH="${GEN_MANIFEST}" \
  FITS_SPEC_PATH="${FITS_SPEC_PATH}" \
  FIT_OVERWRITE="${FIT_OVERWRITE}" \
  SBATCH_BIN="${SBATCH_BIN}" \
  WORKER_SCRIPT="${FIT_WORKER_SCRIPT}" \
  REPORT_JOB_NAME="${FIT_REPORT_JOB_NAME}" \
  bash "${FIT_SUBMITTER}"
}

run_intervention_stage() {
  if [[ -n "${INTERVENTION_LIBRARY_SCRIPT:-}" ]]; then
    bash "${INTERVENTION_LIBRARY_SCRIPT}"
    return 0
  fi
  pixi run python -u run_intervention_library.py \
    --generation_manifest_path "${GEN_MANIFEST}" \
    --spec_path "${INTERVENTION_LIBRARY_SPEC_PATH}"
}

submit_posterior_predictive_stage() {
  GEN_MANIFEST="${GEN_MANIFEST}" \
  FIT_MANIFEST="${FIT_MANIFEST}" \
  TARGET_PAIRS_PATH="${TARGET_PAIRS_PATH}" \
  POSTERIOR_PREDICTIVE_SPEC_PATH="${POSTERIOR_PREDICTIVE_SPEC_PATH}" \
  POSTERIOR_PREDICTIVE_OVERWRITE="${POSTERIOR_PREDICTIVE_OVERWRITE}" \
  SBATCH_BIN="${SBATCH_BIN}" \
  WORKER_SCRIPT="${POSTERIOR_PREDICTIVE_WORKER_SCRIPT}" \
  REPORT_JOB_NAME="${POSTERIOR_PREDICTIVE_REPORT_JOB_NAME}" \
  bash "${POSTERIOR_PREDICTIVE_SUBMITTER}"
}

# echo "Submitting generation jobs..."
# generation_barrier_job_id="$(submit_generation_stage)"
# wait_for_job "${generation_barrier_job_id}" "Generation"

# echo "Submitting fit jobs..."
# fit_barrier_job_id="$(submit_fit_stage)"
# wait_for_job "${fit_barrier_job_id}" "Fit"

# echo "Running intervention library..."
# run_intervention_stage

echo "Submitting posterior predictive jobs..."
posterior_predictive_report_job_id="$(submit_posterior_predictive_stage)"
wait_for_job "${posterior_predictive_report_job_id}" "Posterior predictive"

echo "Job finished at $(date)"

# Ensure the log directory exists
mkdir -p "slurm-logs/$(date +%Y-%m-%d)"
