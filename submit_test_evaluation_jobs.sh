#!/bin/bash

set -euo pipefail

FIT_MANIFEST_PATH="${FIT_MANIFEST_PATH:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
GIBBS_SWEEPS="${GIBBS_SWEEPS:-}"
SEED="${SEED:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
WORKER_SCRIPT="${WORKER_SCRIPT:-run_test_evaluation_job.sh}"
WORKER_JOB_NAME="${WORKER_JOB_NAME:-test-eval}"

TEST_EVAL_TIME="${TEST_EVAL_TIME:-08:00:00}"
TEST_EVAL_CPUS="${TEST_EVAL_CPUS:-4}"
TEST_EVAL_MEM="${TEST_EVAL_MEM:-8G}"
TEST_EVAL_PARTITION="${TEST_EVAL_PARTITION:-mit_normal}"

if [[ -z "${FIT_MANIFEST_PATH}" ]]; then
  echo "FIT_MANIFEST_PATH is required." >&2
  exit 1
fi

worker_args=(
  --parsable
  --job-name "${WORKER_JOB_NAME}"
  --time "${TEST_EVAL_TIME}"
  --ntasks 1
  --cpus-per-task "${TEST_EVAL_CPUS}"
  --mem "${TEST_EVAL_MEM}"
)
if [[ -n "${TEST_EVAL_PARTITION}" ]]; then
  worker_args+=(--partition "${TEST_EVAL_PARTITION}")
fi

submit_output="$(
  FIT_MANIFEST_PATH="${FIT_MANIFEST_PATH}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  GIBBS_SWEEPS="${GIBBS_SWEEPS}" \
  SEED="${SEED}" \
  "${SBATCH_BIN}" "${worker_args[@]}" "${WORKER_SCRIPT}"
)"
printf "%s\n" "${submit_output%%;*}"
