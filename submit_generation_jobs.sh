#!/bin/bash

set -euo pipefail

GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH:-data/configs/generation_spec.yaml}"
GENERATION_OVERWRITE="${GENERATION_OVERWRITE:-false}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
WORKER_SCRIPT="${WORKER_SCRIPT:-run_generation_job.sh}"
WORKER_JOB_NAME="${WORKER_JOB_NAME:-generation}"
REPORT_JOB_NAME="${REPORT_JOB_NAME:-generation-refresh}"

GEN_TIME="${GEN_TIME:-04:00:00}"
GEN_CPUS="${GEN_CPUS:-8}"
GEN_MEM="${GEN_MEM:-16G}"
GEN_PARTITION="${GEN_PARTITION:-mit_normal}"

GEN_REPORT_TIME="${GEN_REPORT_TIME:-00:30:00}"
GEN_REPORT_CPUS="${GEN_REPORT_CPUS:-1}"
GEN_REPORT_MEM="${GEN_REPORT_MEM:-4G}"
GEN_REPORT_PARTITION="${GEN_REPORT_PARTITION:-mit_normal}"

# Ensure the log directory exists
mkdir -p "slurm-logs/$(date +%Y-%m-%d)"

pixi run python -u run_generation_pipeline.py \
  --spec_path "${GENERATION_SPEC_PATH}" \
  --write_requests >/dev/null

REQUESTS_PATH="$(
  pixi run python - <<'PY' "${GENERATION_SPEC_PATH}"
import sys
from run_generation_pipeline import generation_requests_path_for_spec

print(generation_requests_path_for_spec(sys.argv[1]))
PY
)"

job_ids=()

while IFS=$'\t' read -r generation_spec_path experiment_name experiment_slug experiment_path; do
  worker_args=(
    --parsable
    --job-name "${WORKER_JOB_NAME}"
    --time "${GEN_TIME}"
    --ntasks 1
    --cpus-per-task "${GEN_CPUS}"
    --mem "${GEN_MEM}"
  )
  if [[ -n "${GEN_PARTITION}" ]]; then
    worker_args+=(--partition "${GEN_PARTITION}")
  fi
  submit_output="$(
    GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH}" \
    GENERATION_OVERWRITE="${GENERATION_OVERWRITE}" \
    "${SBATCH_BIN}" "${worker_args[@]}" "${WORKER_SCRIPT}" "${experiment_slug}"
  )"
  job_ids+=("${submit_output%%;*}")
done < <(
  pixi run python - <<'PY' "${REQUESTS_PATH}"
import csv
import sys

with open(sys.argv[1], "r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        print(
            "\t".join(
                [
                    row.get("generation_spec_path", "").strip(),
                    row.get("experiment_name", "").strip(),
                    row.get("experiment_slug", "").strip(),
                    row.get("experiment_path", "").strip(),
                ]
            )
        )
PY
)

if [[ ${#job_ids[@]} -eq 0 ]]; then
  exit 1
fi

dependency=$(IFS=:; echo "${job_ids[*]}")
report_args=(
  --parsable
  --job-name "${REPORT_JOB_NAME}"
  --dependency "afterok:${dependency}"
  --time "${GEN_REPORT_TIME}"
  --ntasks 1
  --cpus-per-task "${GEN_REPORT_CPUS}"
  --mem "${GEN_REPORT_MEM}"
)
if [[ -n "${GEN_REPORT_PARTITION}" ]]; then
  report_args+=(--partition "${GEN_REPORT_PARTITION}")
fi

report_job_id="$(
  "${SBATCH_BIN}" "${report_args[@]}" \
    --wrap "pixi run python -u run_generation_pipeline.py --spec_path '${GENERATION_SPEC_PATH}' --refresh_manifest"
)"
printf "%s\n" "${report_job_id%%;*}"
