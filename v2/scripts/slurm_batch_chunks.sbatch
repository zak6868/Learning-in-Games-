#!/bin/bash
# Split the 504-cell batch into 12 chunks of 42, run one per array task.
# Adjust --array range if you change the chunk size or total cell count.
#SBATCH --job-name=modexp
#SBATCH --array=0-11
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

# ---- activate your env ------------------------------------------------
source .venv/bin/activate
# If Gurobi is via module: module load gurobi/10.0
# export GRB_LICENSE_FILE=$HOME/gurobi.lic

# ---- split --------------------------------------------------------------
CHUNK_SIZE=42
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE - 1 ))

mkdir -p logs experiments_overnight/out

python -m experiments_overnight.runner \
    --cells "${START}-${END}" \
    --workers ${SLURM_CPUS_PER_TASK} \
    --outdir experiments_overnight/out \
    --summary-csv experiments_overnight/out/summary_${SLURM_ARRAY_TASK_ID}.csv

echo "task $SLURM_ARRAY_TASK_ID done"
