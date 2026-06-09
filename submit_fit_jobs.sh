#!/bin/bash

set -euo pipefail

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-data/configs/fits_spec.yaml}"
CV_SPEC_PATH="${CV_SPEC_PATH:-}"
SEARCH_SLUG="${SEARCH_SLUG:-}"
FIT_MODE="${FIT_MODE:-standard}"
SPLIT_KIND="${SPLIT_KIND:-}"
NUM_FOLDS="${NUM_FOLDS:-}"
OUTER_NUM_FOLDS="${OUTER_NUM_FOLDS:-}"
TEST_FOLD_ID="${TEST_FOLD_ID:-}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
WORKER_SCRIPT="${WORKER_SCRIPT:-run_fit_job.sh}"
WORKER_JOB_NAME="${WORKER_JOB_NAME:-fit}"
REPORT_JOB_NAME="${REPORT_JOB_NAME:-fit-refresh}"

FIT_TIME="${FIT_TIME:-08:00:00}"
FIT_CPUS="${FIT_CPUS:-8}"
FIT_MEM="${FIT_MEM:-16G}"
FIT_PARTITION="${FIT_PARTITION:-mit_normal}"

FIT_REPORT_TIME="${FIT_REPORT_TIME:-00:30:00}"
FIT_REPORT_CPUS="${FIT_REPORT_CPUS:-1}"
FIT_REPORT_MEM="${FIT_REPORT_MEM:-4G}"
FIT_REPORT_PARTITION="${FIT_REPORT_PARTITION:-mit_normal}"
FIELD_SEP=$'\x1f'

if [[ "${FIT_MODE}" != "standard" && "${FIT_MODE}" != "outer_masked" ]]; then
  echo "FIT_MODE must be 'standard' or 'outer_masked', got '${FIT_MODE}'." >&2
  exit 1
fi

split_args=()
if [[ -n "${SPLIT_KIND}" ]]; then
  split_args+=(--split_kind "${SPLIT_KIND}")
fi
if [[ -n "${NUM_FOLDS}" ]]; then
  split_args+=(--num_folds "${NUM_FOLDS}")
fi
if [[ -n "${OUTER_NUM_FOLDS}" ]]; then
  split_args+=(--outer_num_folds "${OUTER_NUM_FOLDS}")
fi
if [[ -n "${TEST_FOLD_ID}" ]]; then
  split_args+=(--test_fold_id "${TEST_FOLD_ID}")
fi

if [[ "${FIT_MODE}" == "standard" ]]; then
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
else
  if [[ -z "${CV_SPEC_PATH}" ]]; then
    echo "CV_SPEC_PATH is required when FIT_MODE=outer_masked." >&2
    exit 1
  fi
  if [[ -z "${SEARCH_SLUG}" ]]; then
    echo "SEARCH_SLUG is required when FIT_MODE=outer_masked." >&2
    exit 1
  fi

  pixi run python -u run_fit_pipeline.py \
    --fit_mode outer_masked \
    --manifest_path "${GENERATION_MANIFEST_PATH}" \
    --cv_spec_path "${CV_SPEC_PATH}" \
    --search_slug "${SEARCH_SLUG}" \
    "${split_args[@]}" \
    --write_requests >/dev/null

  REQUESTS_PATH="$(
    pixi run python - <<'PY' "${GENERATION_MANIFEST_PATH}" "${SEARCH_SLUG}"
import sys
from pathlib import Path

print(
    Path(sys.argv[1]).resolve().parent
    / f"train_fit_requests__{str(sys.argv[2]).strip()}.csv"
)
PY
  )"
fi

job_ids=()

if [[ "${FIT_MODE}" == "standard" ]]; then
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
      FIT_MODE="${FIT_MODE}" \
      FIT_OVERWRITE="${FIT_OVERWRITE}" \
      "${SBATCH_BIN}" "${worker_args[@]}" "${WORKER_SCRIPT}" "${experiment_slug}" "${variant_slug}"
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
else
  while IFS="${FIELD_SEP}" read -r generation_manifest_path cv_spec_path search_name search_slug split_kind num_folds outer_num_folds test_fold_id experiment_name experiment_slug variant_name variant_slug fit_path; do
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
      CV_SPEC_PATH="${cv_spec_path}" \
      SEARCH_SLUG="${search_slug}" \
      FIT_MODE="${FIT_MODE}" \
      SPLIT_KIND="${split_kind}" \
      NUM_FOLDS="${num_folds}" \
      OUTER_NUM_FOLDS="${outer_num_folds}" \
      TEST_FOLD_ID="${test_fold_id}" \
      FIT_OVERWRITE="${FIT_OVERWRITE}" \
      "${SBATCH_BIN}" "${worker_args[@]}" "${WORKER_SCRIPT}" "${experiment_slug}" "${variant_slug}" "${search_slug}"
    )"
    job_ids+=("${submit_output%%;*}")
  done < <(
    pixi run python - <<'PY' "${REQUESTS_PATH}"
import csv
import sys

with open(sys.argv[1], "r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        fields = [
            row.get("generation_manifest_path", "").strip(),
            row.get("cv_spec_path", "").strip(),
            row.get("search_name", "").strip(),
            row.get("search_slug", "").strip(),
            row.get("split_kind", "").strip(),
            row.get("num_folds", "").strip(),
        ]
        if "outer_num_folds" in row:
            fields.append(row.get("outer_num_folds", "").strip())
        else:
            fields.append("")
        if "test_fold_id" in row:
            fields.append(row.get("test_fold_id", "").strip())
        else:
            fields.append("")
        fields.extend([
            row.get("experiment_name", "").strip(),
            row.get("experiment_slug", "").strip(),
            row.get("variant_name", "").strip(),
            row.get("variant_slug", "").strip(),
            row.get("fit_path", "").strip(),
        ])
        print("\x1f".join(fields))
PY
  )
fi

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

if [[ "${FIT_MODE}" == "standard" ]]; then
  report_job_id="$(
    "${SBATCH_BIN}" "${report_args[@]}" \
      --wrap "pixi run python -u run_fit_pipeline.py --manifest_path '${GENERATION_MANIFEST_PATH}' --fits_spec_path '${FITS_SPEC_PATH}' --refresh_manifest"
  )"
else
  refresh_cmd="pixi run python -u run_fit_pipeline.py --fit_mode outer_masked --manifest_path '${GENERATION_MANIFEST_PATH}' --cv_spec_path '${CV_SPEC_PATH}' --search_slug '${SEARCH_SLUG}'"
  if [[ -n "${SPLIT_KIND}" ]]; then
    refresh_cmd="${refresh_cmd} --split_kind '${SPLIT_KIND}'"
  fi
  if [[ -n "${NUM_FOLDS}" ]]; then
    refresh_cmd="${refresh_cmd} --num_folds '${NUM_FOLDS}'"
  fi
  if [[ -n "${OUTER_NUM_FOLDS}" ]]; then
    refresh_cmd="${refresh_cmd} --outer_num_folds '${OUTER_NUM_FOLDS}'"
  fi
  if [[ -n "${TEST_FOLD_ID}" ]]; then
    refresh_cmd="${refresh_cmd} --test_fold_id '${TEST_FOLD_ID}'"
  fi
  refresh_cmd="${refresh_cmd} --refresh_manifest"
  report_job_id="$(
    "${SBATCH_BIN}" "${report_args[@]}" \
      --wrap "${refresh_cmd}"
  )"
fi
printf "%s\n" "${report_job_id%%;*}"
