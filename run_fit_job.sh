#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --output=slurm-%j-fit.out
#SBATCH --error=slurm-%j-fit.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=mit_normal

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-data/configs/fits_spec.yaml}"
FIT_OVERWRITE="${FIT_OVERWRITE:-false}"

EXPERIMENT_SLUG="${1:?missing experiment_slug}"
VARIANT_SLUG="${2:?missing variant_slug}"

OVERWRITE_FLAG=()
if [[ "${FIT_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

pixi run python -u run_fit_pipeline.py \
  --manifest_path "${GENERATION_MANIFEST_PATH}" \
  --fits_spec_path "${FITS_SPEC_PATH}" \
  --run_request \
  --experiment_slug "${EXPERIMENT_SLUG}" \
  --variant_slug "${VARIANT_SLUG}" \
  "${OVERWRITE_FLAG[@]}"
