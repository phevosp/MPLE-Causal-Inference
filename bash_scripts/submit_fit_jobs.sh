#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-data/configs/fits_spec.yaml}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${SCRIPT_DIR}/run_fit_job.sh}"
WORKER_JOB_NAME="${WORKER_JOB_NAME:-fit}"
REPORT_JOB_NAME="${REPORT_JOB_NAME:-fit-refresh}"

FIT_TIME="${FIT_TIME:-08:00:00}"
FIT_CPUS="${FIT_CPUS:-16}"
FIT_MEM="${FIT_MEM:-32G}"
FIT_PARTITION="${FIT_PARTITION:-mit_normal}"

FIT_REPORT_TIME="${FIT_REPORT_TIME:-00:30:00}"
FIT_REPORT_CPUS="${FIT_REPORT_CPUS:-1}"
FIT_REPORT_MEM="${FIT_REPORT_MEM:-4G}"
FIT_REPORT_PARTITION="${FIT_REPORT_PARTITION:-mit_normal}"

pixi run python -u run_fit_pipeline.py \
  --manifest_path "${GENERATION_MANIFEST_PATH}" \
  --fits_spec_path "${FITS_SPEC_PATH}" \
  --write_requests >/dev/null

REQUESTS_PATH="$(
  pixi run python - <<'PY' "${FITS_SPEC_PATH}"
import sys
from run_fit_pipeline import fit_requests_path_for_spec

print(fit_requests_path_for_spec(sys.argv[1]))
PY
)"

job_ids=()

while IFS=$'\t' read -r generation_manifest_path fits_spec_path experiment_name experiment_slug variant_name variant_slug fit_path; do
  worker_args=(
    --parsable
    --job-name "${WORKER_JOB_NAME}"
    --time "${FIT_TIME}"
    --ntasks 1
    --cpus-per-task "${FIT_CPUS}"
    --mem "${FIT_MEM}"
  )
  if [[ -n "${FIT_PARTITION}" ]]; then
    worker_args+=(--partition "${FIT_PARTITION}")
  fi
  submit_output="$(
    GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH}" \
    FITS_SPEC_PATH="${FITS_SPEC_PATH}" \
    FIT_OVERWRITE="${FIT_OVERWRITE}" \
    "${SBATCH_BIN}" --chdir "${REPO_ROOT}" "${worker_args[@]}" "${WORKER_SCRIPT}" "${experiment_slug}" "${variant_slug}"
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
                    row.get("generation_manifest_path", "").strip(),
                    row.get("fits_spec_path", "").strip(),
                    row.get("experiment_name", "").strip(),
                    row.get("experiment_slug", "").strip(),
                    row.get("variant_name", "").strip(),
                    row.get("variant_slug", "").strip(),
                    row.get("fit_path", "").strip(),
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
  --time "${FIT_REPORT_TIME}"
  --ntasks 1
  --cpus-per-task "${FIT_REPORT_CPUS}"
  --mem "${FIT_REPORT_MEM}"
)
if [[ -n "${FIT_REPORT_PARTITION}" ]]; then
  report_args+=(--partition "${FIT_REPORT_PARTITION}")
fi

report_job_id="$(
  "${SBATCH_BIN}" --chdir "${REPO_ROOT}" "${report_args[@]}" \
    --wrap "pixi run python -u run_fit_pipeline.py --manifest_path '${GENERATION_MANIFEST_PATH}' --fits_spec_path '${FITS_SPEC_PATH}' --refresh_manifest"
)"
printf "%s\n" "${report_job_id%%;*}"
