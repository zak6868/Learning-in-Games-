# CLUSTER.md — running the overnight batch on a SLURM cluster

This document walks through (1) cloning, (2) environment setup, (3)
splitting the 504-cell batch across array jobs, (4) collecting and
aggregating the results.  Templates for SLURM and PBS Pro are in
`scripts/`.

---

## 1. Clone the repo on the cluster

```bash
ssh your_user@cluster.login
cd $SCRATCH    # or wherever your fast storage is
git clone <git-url> moderator_experiments
cd moderator_experiments
```

---

## 2. Environment setup

### Python

Any Python ≥ 3.10 works.  The CI tests run on 3.14.

```bash
module load python/3.11     # adjust to your cluster's module system
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Gurobi (recommended)

The inner CE reconstruction LP is small but called many times.
`gurobipy` is a strong dependency performance-wise; the code falls
back to `scipy.optimize.linprog` when Gurobi is unavailable.

```bash
module load gurobi/10.0
pip install gurobipy
export GRB_LICENSE_FILE=$HOME/gurobi.lic   # point at your license
```

### Verify

```bash
pytest experiments_overnight/tests/ -v
```

All 9 tests must pass.  Total runtime: a few seconds.

---

## 3. Inspect the configured cells

```bash
python -m experiments_overnight.runner --list-only | head -20
python -m experiments_overnight.runner --list-only | wc -l
```

You should see 504 cells.  They are grouped (but not sorted) by
game family; see `experiments_overnight/configs/overnight_cells.py`
to adjust what is included.

---

## 4. Split the batch across SLURM array tasks

The cleanest pattern: one array job, each task handles a contiguous
range of cells and uses multiprocessing within the node.

### Option A — one task per cell (fine-grained, easy reruns)

```bash
#!/bin/bash
#SBATCH --job-name=modexp
#SBATCH --array=0-503          # one task per cell
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out

source .venv/bin/activate
python -m experiments_overnight.runner \
    --cells "$SLURM_ARRAY_TASK_ID" \
    --workers 2 \
    --outdir experiments_overnight/out \
    --summary-csv experiments_overnight/out/summary_${SLURM_ARRAY_TASK_ID}.csv
```

### Option B — one task per chunk (fewer jobs, bigger nodes)

Say we have 24 cores per node and want 12 array tasks covering 504
cells → 42 cells per task, 12 processes per task:

```bash
#!/bin/bash
#SBATCH --job-name=modexp
#SBATCH --array=0-11
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.out

CHUNK_SIZE=42
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE - 1))

source .venv/bin/activate
python -m experiments_overnight.runner \
    --cells "${START}-${END}" \
    --workers 12 \
    --outdir experiments_overnight/out \
    --summary-csv experiments_overnight/out/summary_${SLURM_ARRAY_TASK_ID}.csv
```

Submit with:

```bash
mkdir -p logs experiments_overnight/out
sbatch scripts/slurm_batch_chunks.sbatch
```

### PBS Pro

See `scripts/pbs_batch.pbs` for the same logic under PBS.

---

## 5. Monitor progress

Each cell streams a one-line summary as it completes:

```
[17/42] braess_n50_BR_s0: ok  regret=3.415
```

Across an array job, `squeue -u $USER` shows what's running; `tail
logs/<jobid>_<task>.out` shows per-task output.

The summary CSVs (`summary_*.csv`) are small and can be concatenated
post-run to get a single table of every cell's outcome.

---

## 6. Post-run aggregation

Once the array finishes, consolidate and produce mean ± std plots:

```bash
# Concatenate per-task summary CSVs into a master file
head -1 experiments_overnight/out/summary_0.csv > experiments_overnight/out/summary_all.csv
for f in experiments_overnight/out/summary_*.csv; do tail -n +2 "$f"; done \
    >> experiments_overnight/out/summary_all.csv

# Produce one 2x2 aggregated plot per configuration (across seeds)
python -m experiments_overnight.aggregate_plots \
    --npz-dir experiments_overnight/out \
    --outdir experiments_overnight/out/aggregated
```

---

## 7. Rsync back to your laptop

```bash
rsync -avz \
    cluster:$SCRATCH/moderator_experiments/experiments_overnight/out/ \
    ./experiments_overnight/out_cluster/
```

---

## 8. Resource recommendations

- 8 GB of RAM per worker is ample for all cells in the default config.
- At 24 workers per node, Pigou $n{=}5000$ and Braess $n{=}1000$ can
  take 2–4 hours each — budget accordingly.
- Gurobi is single-threaded here; no need for the LP+MIP multi-thread
  license variants.

---

## 9. Common issues

**Cell status "timeout":** the signal-based 300-second timeout inside
the runner fired before JLB converged.  Increase `max_rounds` or
`jlb_max_iters` in the config for that cell family, or just rerun the
specific indices:

```bash
python -m experiments_overnight.runner --cells 123,145,212
```

**Cell status "error:…":** inspect the traceback in the Python side
(these fire on e.g. JLB reconstruction LP returning `None` on a
pathological θ).  Increase the outer radius, or skip the cell.

**Gurobi license conflicts on shared nodes:** set
`GRB_LICENSE_FILE` per-task in your sbatch to a local token file or
use a WLS license.

---

## 10. Minimum working example

On any Linux/macOS with Python:

```bash
git clone <url> moderator_experiments
cd moderator_experiments
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m experiments_overnight.runner --cells 0-3 --workers 2
ls experiments_overnight/out/
```

Should produce 4 `.npz` files + 16 PNGs + a `summary.csv`.
