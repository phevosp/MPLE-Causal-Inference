#!/bin/bash

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

GEN_MANIFEST="${GEN_MANIFEST:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FIT_MANIFEST="${FIT_MANIFEST:-experiments/SyntheticHybridExperiments/fit_manifest.csv}"
TARGET_PAIRS_PATH="${TARGET_PAIRS_PATH:-data/configs/posterior_predictive_target_pairs.csv}"
POSTERIOR_PREDICTIVE_SPEC_PATH="${POSTERIOR_PREDICTIVE_SPEC_PATH:-data/configs/posterior_predictive_spec.yaml}"
POSTERIOR_PREDICTIVE_OVERWRITE="${POSTERIOR_PREDICTIVE_OVERWRITE:-false}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${SCRIPT_DIR}/run_posterior_predictive_job.sh}"
REPORT_JOB_NAME="${REPORT_JOB_NAME:-posterior-predictive-report}"

job_ids=()
FIELD_SEP=$'\x1f'

while IFS="${FIELD_SEP}" read -r experiment_name source_type variant_name intervention_source intervention_name run_name; do
  submit_output="$(
    GEN_MANIFEST="${GEN_MANIFEST}" \
    FIT_MANIFEST="${FIT_MANIFEST}" \
    TARGET_PAIRS_PATH="${TARGET_PAIRS_PATH}" \
    POSTERIOR_PREDICTIVE_SPEC_PATH="${POSTERIOR_PREDICTIVE_SPEC_PATH}" \
    POSTERIOR_PREDICTIVE_OVERWRITE="${POSTERIOR_PREDICTIVE_OVERWRITE}" \
    "${SBATCH_BIN}" \
      --chdir "${REPO_ROOT}" \
      --export "ALL,REPO_ROOT=${REPO_ROOT}" \
      --parsable "${WORKER_SCRIPT}" \
      "${experiment_name}" \
      "${source_type}" \
      "${variant_name}" \
      "${intervention_source}" \
      "${intervention_name}" \
      "${run_name}"
  )"
  job_ids+=("${submit_output%%;*}")
done < <(
  pixi run python - <<'PY' "${TARGET_PAIRS_PATH}" "${POSTERIOR_PREDICTIVE_SPEC_PATH}"
import csv
import sys
from utils.t6_pipeline_spec_utils import expand_named_entries

target_pairs_path = sys.argv[1]
spec_path = sys.argv[2]
runs = expand_named_entries(spec_path, "runs")
with open(target_pairs_path, "r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        for run in runs:
            print(
                "\x1f".join(
                    [
                        row.get("experiment_name", "").strip(),
                        row.get("source_type", "").strip(),
                        row.get("variant_name", "").strip(),
                        (row.get("intervention_source", "").strip() or "observed_experiment"),
                        row.get("intervention_name", "").strip(),
                        str(run["name"]).strip(),
                    ]
                )
            )
PY
)

if [[ ${#job_ids[@]} -eq 0 ]]; then
  echo "No posterior predictive jobs were submitted."
  exit 1
fi

dependency=$(IFS=:; echo "${job_ids[*]}")

report_job_id="$(
  GEN_MANIFEST="${GEN_MANIFEST}" \
    "${SBATCH_BIN}" \
    --chdir "${REPO_ROOT}" \
    --export "ALL,REPO_ROOT=${REPO_ROOT}" \
    --parsable \
    --job-name "${REPORT_JOB_NAME}" \
    --dependency "afterok:${dependency}" \
    --wrap "cd '${REPO_ROOT}' && pixi run python -c \"from utils.t8_posterior_predictive_reporting import refresh_and_write_posterior_predictive_reports as f; f(r'${GEN_MANIFEST}')\""
)"
printf "%s\n" "${report_job_id%%;*}"
