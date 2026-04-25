#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --output=slurm-logs/$(date +%Y-%m-%d)/slurm-%j-generation.out
#SBATCH --error=slurm-logs/$(date +%Y-%m-%d)/slurm-%j-generation.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mit_normal

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

GENERATION_SPEC_PATH="${GENERATION_SPEC_PATH:-data/configs/generation_spec.yaml}"
GENERATION_OVERWRITE="${GENERATION_OVERWRITE:-false}"

EXPERIMENT_SLUG="${1:?missing experiment_slug}"

OVERWRITE_FLAG=()
if [[ "${GENERATION_OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

# Ensure the log directory exists
mkdir -p "slurm-logs/$(date +%Y-%m-%d)"

pixi run python -u run_generation_pipeline.py \
  --spec_path "${GENERATION_SPEC_PATH}" \
  --run_request \
  --experiment_slug "${EXPERIMENT_SLUG}" \
  "${OVERWRITE_FLAG[@]}"
