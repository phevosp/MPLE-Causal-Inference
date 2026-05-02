#!/bin/bash

set -euo pipefail

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
CV_SPEC_PATH="${CV_SPEC_PATH:-data/configs/cv_spec.yaml}"
CV_NUM_FOLDS="${CV_NUM_FOLDS:-}"  # Optional override; uses CV spec default if not set
CV_OVERWRITE="${CV_OVERWRITE:-false}"
EXECUTION_MODE="${EXECUTION_MODE:-cv}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
WORKER_SCRIPT="${WORKER_SCRIPT:-run_cv_job.sh}"
WORKER_JOB_NAME="${WORKER_JOB_NAME:-cv}"
REPORT_JOB_NAME="${REPORT_JOB_NAME:-cv-refresh}"

CV_TIME="${CV_TIME:-08:00:00}"
CV_CPUS="${CV_CPUS:-16}"
CV_MEM="${CV_MEM:-32G}"
CV_PARTITION="${CV_PARTITION:-mit_normal}"

CV_REPORT_TIME="${CV_REPORT_TIME:-00:30:00}"
CV_REPORT_CPUS="${CV_REPORT_CPUS:-1}"
CV_REPORT_MEM="${CV_REPORT_MEM:-4G}"
CV_REPORT_PARTITION="${CV_REPORT_PARTITION:-mit_normal}"

pixi run python -u run_cv_folds.py \
  --generation_manifest_path "${GENERATION_MANIFEST_PATH}" \
  --cv_spec_path "${CV_SPEC_PATH}" \
  --execution_mode "${EXECUTION_MODE}" \
  --write_requests >/dev/null

REQUESTS_PATH="$(
  pixi run python - <<'PY' "${CV_SPEC_PATH}" "${EXECUTION_MODE}"
import sys
from run_cv_folds import model_selection_requests_path_for_spec

print(model_selection_requests_path_for_spec(sys.argv[1], execution_mode=sys.argv[2]))
PY
)"

job_ids=()

# Extract unique (experiment_slug, search_slug) pairs from cv_requests
while IFS=$'\t' read -r experiment_slug search_slug; do
  worker_args=(
    --parsable
    --job-name "${WORKER_JOB_NAME}"
    --time "${CV_TIME}"
    --ntasks 1
    --cpus-per-task "${CV_CPUS}"
    --mem "${CV_MEM}"
  )
  if [[ -n "${CV_PARTITION}" ]]; then
    worker_args+=(--partition "${CV_PARTITION}")
  fi
  submit_output="$(
    GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH}" \
    CV_SPEC_PATH="${CV_SPEC_PATH}" \
    CV_NUM_FOLDS="${CV_NUM_FOLDS}" \
    CV_OVERWRITE="${CV_OVERWRITE}" \
    EXECUTION_MODE="${EXECUTION_MODE}" \
    "${SBATCH_BIN}" "${worker_args[@]}" "${WORKER_SCRIPT}" "${experiment_slug}" "${search_slug}"
  )"
  job_ids+=("${submit_output%%;*}")
done < <(
  pixi run python - <<'PY' "${REQUESTS_PATH}"
import csv
import sys

seen = set()
with open(sys.argv[1], "r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        pair = (
            row.get("experiment_slug", "").strip(),
            row.get("search_slug", "").strip(),
        )
        if pair not in seen:
            print("\t".join(pair))
            seen.add(pair)
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
  --time "${CV_REPORT_TIME}"
  --ntasks 1
  --cpus-per-task "${CV_REPORT_CPUS}"
  --mem "${CV_REPORT_MEM}"
)
if [[ -n "${CV_REPORT_PARTITION}" ]]; then
  report_args+=(--partition "${CV_REPORT_PARTITION}")
fi

report_job_id="$(
  "${SBATCH_BIN}" "${report_args[@]}" \
    --wrap "pixi run python -u run_cv_folds.py --refresh_manifest --cv_requests_path '${REQUESTS_PATH}' --execution_mode '${EXECUTION_MODE}'"
)"
printf "%s\n" "${report_job_id%%;*}"
