#!/bin/bash
#SBATCH --job-name=mple-tests
#SBATCH --output=slurm-%j-mple-tests.out
#SBATCH --error=slurm-%j-mple-tests.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=mit_normal

# ---- safety + debugging ----
set -euo pipefail
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# ---- control threading (VERY important for numpy/scipy) ----
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- optional: load environment if needed ----
# module load python
# source ~/.bashrc

# ---- paths ----
GEN_MANIFEST="experiments/SyntheticHybridExperiments/generation_manifest.csv"
FIT_MANIFEST="experiments/SyntheticHybridExperiments/fit_manifest.csv"

# # ---- run pipeline ----
echo "Running generation pipeline..."
pixi run python -u run_generation_pipeline.py

echo "Running fit pipeline..."
pixi run python -u run_fit_pipeline.py \
    --manifest_path "$GEN_MANIFEST"

echo "Running intervention library..."
pixi run python -u run_intervention_library.py \
    --generation_manifest_path "$GEN_MANIFEST" \
    --spec_path data/configs/intervention_library_spec.yaml

echo "Submitting posterior predictive jobs..."
GEN_MANIFEST="$GEN_MANIFEST" \
FIT_MANIFEST="$FIT_MANIFEST" \
TARGET_PAIRS_PATH="data/configs/posterior_predictive_target_pairs.csv" \
POSTERIOR_PREDICTIVE_SPEC_PATH="data/configs/posterior_predictive_spec.yaml" \
bash submit_posterior_predictive_jobs.sh

echo "Job finished at $(date)"
