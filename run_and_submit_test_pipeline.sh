#!/bin/bash
#SBATCH --job-name=mple-tests
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=mit_normal
#SBATCH --output=/dev/stdout
#SBATCH --error=/dev/stderr

set -euo pipefail

DATE=$(date +%F)
LOG_DIR="slurm-logs/$DATE"
mkdir -p "$LOG_DIR"
OUT_PATH="$LOG_DIR/${SLURM_JOB_ID:-manual}_${SLURM_JOB_NAME:-run-and-submit-full-pipeline}.out"
ERR_PATH="$LOG_DIR/${SLURM_JOB_ID:-manual}_${SLURM_JOB_NAME:-run-and-submit-full-pipeline}.err"
exec >"$OUT_PATH" 2>"$ERR_PATH"

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH:-data/configs/generation_spec.yaml}"
CV_SPEC_PATH="${CV_SPEC_PATH:-data/configs/cv_spec.yaml}"
SEARCH_SLUG="${SEARCH_SLUG:-}"
GEN_MANIFEST="${GEN_MANIFEST:-}"
FIT_MANIFEST_PATH="${FIT_MANIFEST_PATH:-}"

GENERATION_OVERWRITE="${GENERATION_OVERWRITE:-false}"
CV_OVERWRITE="${CV_OVERWRITE:-false}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"

BUILD_SPLITS_SEED="${BUILD_SPLITS_SEED:-${SEED:-}}"
BUILD_SPLITS_RECURSIVE="${BUILD_SPLITS_RECURSIVE:-${RECURSIVE:-}}"
BUILD_SPLITS_CONTIGUOUS="${BUILD_SPLITS_CONTIGUOUS:-${CONTIGUOUS:-}}"
BUILD_SPLITS_TOLERANCE="${BUILD_SPLITS_TOLERANCE:-${TOLERANCE:-}}"
BUILD_SPLITS_OVERWRITE="${BUILD_SPLITS_OVERWRITE:-${OVERWRITE:-false}}"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SACCT_BIN="${SACCT_BIN:-sacct}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SLEEP_BIN="${SLEEP_BIN:-sleep}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-10}"

GENERATION_SUBMITTER="${GENERATION_SUBMITTER:-submit_generation_jobs.sh}"
CV_SUBMITTER="${CV_SUBMITTER:-submit_cv_jobs.sh}"
FIT_SUBMITTER="${FIT_SUBMITTER:-submit_fit_jobs.sh}"
TEST_EVALUATION_SUBMITTER="${TEST_EVALUATION_SUBMITTER:-submit_test_evaluation_jobs.sh}"

GENERATION_WORKER_SCRIPT="${GENERATION_WORKER_SCRIPT:-run_generation_job.sh}"
CV_WORKER_SCRIPT="${CV_WORKER_SCRIPT:-run_cv_job.sh}"
FIT_WORKER_SCRIPT="${FIT_WORKER_SCRIPT:-run_fit_job.sh}"
TEST_EVALUATION_WORKER_SCRIPT="${TEST_EVALUATION_WORKER_SCRIPT:-run_test_evaluation_job.sh}"

GENERATION_REPORT_JOB_NAME="${GENERATION_REPORT_JOB_NAME:-generation-refresh}"
CV_REPORT_JOB_NAME="${CV_REPORT_JOB_NAME:-cv-refresh}"
FIT_REPORT_JOB_NAME="${FIT_REPORT_JOB_NAME:-fit-refresh}"
TEST_EVALUATION_JOB_NAME="${TEST_EVALUATION_JOB_NAME:-test-eval}"

BUILD_SPLITS_SCRIPT="${BUILD_SPLITS_SCRIPT:-${BUILD_CV_FOLDS_SCRIPT:-}}"

resolve_generation_manifest_path() {
  if [[ -n "${RESOLVE_GENERATION_MANIFEST_SCRIPT:-}" ]]; then
    bash "${RESOLVE_GENERATION_MANIFEST_SCRIPT}" "${GENERATION_SPEC_PATH}"
    return 0
  fi
  pixi run python - <<'PY' "${GENERATION_SPEC_PATH}"
import sys
from run_generation_pipeline import generation_manifest_path_for_spec

print(generation_manifest_path_for_spec(sys.argv[1]))
PY
}

resolve_train_fit_manifest_path() {
  if [[ -n "${RESOLVE_TRAIN_FIT_MANIFEST_SCRIPT:-}" ]]; then
    bash "${RESOLVE_TRAIN_FIT_MANIFEST_SCRIPT}" "${GEN_MANIFEST}" "${SEARCH_SLUG}"
    return 0
  fi
  pixi run python - <<'PY' "${GEN_MANIFEST}" "${SEARCH_SLUG}"
import sys
from run_fit_pipeline import train_fit_manifest_path_for_scope

print(train_fit_manifest_path_for_scope(sys.argv[1], search_slug=sys.argv[2] or None))
PY
}

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

build_splits_stage() {
  if [[ -n "${BUILD_SPLITS_SCRIPT}" ]]; then
    bash "${BUILD_SPLITS_SCRIPT}" "${GEN_MANIFEST}" "${CV_SPEC_PATH}"
    return 0
  fi

  local split_args=(
    --generation_manifest_path "${GEN_MANIFEST}"
    --cv_spec_path "${CV_SPEC_PATH}"
  )
  if [[ -n "${BUILD_SPLITS_SEED}" ]]; then
    split_args+=(--seed "${BUILD_SPLITS_SEED}")
  fi
  if [[ "${BUILD_SPLITS_RECURSIVE}" == "true" ]]; then
    split_args+=(--recursive)
  fi
  if [[ "${BUILD_SPLITS_CONTIGUOUS}" == "true" ]]; then
    split_args+=(--contiguous)
  fi
  if [[ -n "${BUILD_SPLITS_TOLERANCE}" ]]; then
    split_args+=(--tolerance "${BUILD_SPLITS_TOLERANCE}")
  fi
  if [[ "${BUILD_SPLITS_OVERWRITE}" == "true" ]]; then
    split_args+=(--overwrite)
  fi
  pixi run python -u build_splits.py "${split_args[@]}"
}

submit_cv_stage() {
  GENERATION_MANIFEST_PATH="${GEN_MANIFEST}" \
  CV_SPEC_PATH="${CV_SPEC_PATH}" \
  CV_OVERWRITE="${CV_OVERWRITE}" \
  EXECUTION_MODE="cv" \
  SBATCH_BIN="${SBATCH_BIN}" \
  WORKER_SCRIPT="${CV_WORKER_SCRIPT}" \
  REPORT_JOB_NAME="${CV_REPORT_JOB_NAME}" \
  bash "${CV_SUBMITTER}"
}

submit_fit_stage() {
  local fit_env=(
    "GENERATION_MANIFEST_PATH=${GEN_MANIFEST}"
    "CV_SPEC_PATH=${CV_SPEC_PATH}"
    "FIT_MODE=outer_masked"
    "FIT_OVERWRITE=${FIT_OVERWRITE}"
    "SBATCH_BIN=${SBATCH_BIN}"
    "WORKER_SCRIPT=${FIT_WORKER_SCRIPT}"
    "REPORT_JOB_NAME=${FIT_REPORT_JOB_NAME}"
  )
  if [[ -n "${SEARCH_SLUG}" ]]; then
    fit_env+=("SEARCH_SLUG=${SEARCH_SLUG}")
  fi
  env "${fit_env[@]}" bash "${FIT_SUBMITTER}"
}

submit_test_evaluation_stage() {
  FIT_MANIFEST_PATH="${FIT_MANIFEST_PATH}" \
  SBATCH_BIN="${SBATCH_BIN}" \
  WORKER_SCRIPT="${TEST_EVALUATION_WORKER_SCRIPT}" \
  WORKER_JOB_NAME="${TEST_EVALUATION_JOB_NAME}" \
  bash "${TEST_EVALUATION_SUBMITTER}"
}

echo "Submitting generation jobs..."
generation_barrier_job_id="$(submit_generation_stage)"
wait_for_job "${generation_barrier_job_id}" "Generation"

if [[ -z "${GEN_MANIFEST}" ]]; then
  GEN_MANIFEST="$(resolve_generation_manifest_path)"
fi

echo "Building split bundles..."
build_splits_stage

echo "Submitting CV jobs..."
cv_barrier_job_id="$(submit_cv_stage)"
wait_for_job "${cv_barrier_job_id}" "CV"

echo "Submitting outer-masked fit jobs..."
fit_barrier_job_id="$(submit_fit_stage)"
wait_for_job "${fit_barrier_job_id}" "Fit"

if [[ -z "${FIT_MANIFEST_PATH}" ]]; then
  FIT_MANIFEST_PATH="$(resolve_train_fit_manifest_path)"
fi

echo "Submitting test-set evaluation jobs..."
test_evaluation_job_id="$(submit_test_evaluation_stage)"
wait_for_job "${test_evaluation_job_id}" "Test evaluation"

echo "Job finished at $(date)"
