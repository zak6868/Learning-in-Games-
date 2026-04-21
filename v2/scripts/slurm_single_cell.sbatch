#!/bin/bash
# One cell per array task (fine-grained; easy to rerun failed indices).
#SBATCH --job-name=modexp-fine
#SBATCH --array=0-503
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
source .venv/bin/activate

mkdir -p logs experiments_overnight/out
python -m experiments_overnight.runner \
    --cells "${SLURM_ARRAY_TASK_ID}" \
    --workers ${SLURM_CPUS_PER_TASK} \
    --outdir experiments_overnight/out \
    --summary-csv experiments_overnight/out/summary_${SLURM_ARRAY_TASK_ID}.csv
