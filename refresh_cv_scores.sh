#!/bin/bash
#SBATCH --job-name=refresh-cv-scores
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/refresh_cv_scores_%j.log
#SBATCH --error=logs/refresh_cv_scores_%j.log

set -e

if [ $# -eq 0 ]; then
    echo "Usage: sbatch refresh_cv_scores.sh <cv_requests_path> [execution_mode]"
    echo "Example: sbatch refresh_cv_scores.sh data/cv_runs/my_search/cv_requests.csv cv"
    exit 1
fi

CV_REQUESTS_PATH="$1"
EXECUTION_MODE="${2:-cv}"

if [ ! -f "$CV_REQUESTS_PATH" ]; then
    echo "Error: CV requests file not found: $CV_REQUESTS_PATH"
    exit 1
fi

mkdir -p logs

echo "Refreshing CV scores from: $CV_REQUESTS_PATH"
echo "Execution mode: $EXECUTION_MODE"
echo "Job ID: $SLURM_JOB_ID"
echo ""

python -u run_cv_folds.py \
    --refresh_scores \
    --cv_requests_path "$CV_REQUESTS_PATH" \
    --execution_mode "$EXECUTION_MODE"

echo "Done!"
