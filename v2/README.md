# moderator_experiments

End-to-end experimental stack for the **cutting-plane moderator** on
succinct games.  Each run plays one round at a time:

1. The moderator proposes $\theta_t$ (current centre of the outer
   ellipsoid / ACCPM knowledge set in $\theta$-space).
2. The **JLB inner oracle** computes a correlated equilibrium of the
   game-at-$\theta_t$.
3. One profile is sampled from that CE and passed to each agent as a
   recommendation.
4. Each agent plays best-response (or $\gamma$-best-response) under
   the **true** $\theta^\star$.  Deviations are observed.
5. If anyone deviated, a cut in $\theta$-space is added to the outer
   cutting-plane.

The pipeline saves every metric we might want later — trajectory of
$\theta_t$, per-agent $\varphi_i$, # deviating agents per round,
inner-solver time, mixture support size, etc.

## Repo layout

```
moderator_experiments/
├── README.md                       ← this file
├── CLUSTER.md                      ← how to run at scale on a cluster
├── pyproject.toml
├── recommend_unknown_games/        ← moderator + JLB backend
├── experiments_overnight/
│   ├── environments/               ← FeatureMaps for each game family
│   │   ├── base_fm.py
│   │   ├── congestion_fm.py
│   │   ├── polymatrix_fm.py
│   │   └── graphical_fm.py
│   ├── harness.py                  ← run_experiment(cfg) → metrics + npz
│   ├── feedback.py                 ← BR / γ-BR
│   ├── plot_utils.py               ← per-run 4-panel plots
│   ├── aggregate_plots.py          ← mean ± std across seeds
│   ├── runner.py                   ← run_one_cell / run_batch + CLI
│   ├── configs/
│   │   └── overnight_cells.py      ← 504-cell config spec
│   └── tests/                      ← smoke + integration (9 tests, all green)
└── scripts/                        ← SLURM / PBS examples
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional:

- **Gurobi** for the inner LP reconstruction (much faster).  Install
  `gurobipy` and configure a license.  Otherwise the code falls back
  to `scipy.optimize.linprog`.

## Quickstart (local, small)

Run the full smoke-test suite (9 tests, <2 s total):

```bash
pytest experiments_overnight/tests/ -v
```

Run a single cell and inspect the artefacts:

```bash
python -m experiments_overnight.runner \
    --cells 0 \
    --workers 1 \
    --outdir experiments_overnight/out
```

This runs the first configured cell (`pigou_n10_BR_s0` by default)
and writes:

- `experiments_overnight/out/pigou_n10_BR_s0.npz` — full per-round
  metrics.
- `experiments_overnight/out/pigou_n10_BR_s0_{regret,deviations,maxphi,inner_time}.png`
  — 4 plots.
- `experiments_overnight/out/summary.csv` — one row with config +
  top-level outcome.

## List configured cells

```bash
python -m experiments_overnight.runner --list-only
```

Prints all 504 cells with their index and label.  Pipe to `wc -l` to
count, or `grep sfgrid3` to filter.

## Run a subset of cells

```bash
# cells 0 through 49
python -m experiments_overnight.runner --cells 0-49 --workers 8

# specific indices
python -m experiments_overnight.runner --cells 17,89,144 --workers 3
```

## Aggregated plots (mean ± std over seeds)

After one or more batches have finished:

```bash
python -m experiments_overnight.aggregate_plots \
    --npz-dir experiments_overnight/out \
    --outdir experiments_overnight/out/aggregated
```

For each `_s{seed}`-stripped config, produces one 2×2 PNG with:

- cumulative regret (mean ± std)
- # deviating agents (mean ± std bars)
- max single-agent $\varphi$ (mean ± std, log-y on some panels)
- inner-solver wall time (mean ± std, log-y)

## Cluster execution

See **[CLUSTER.md](CLUSTER.md)** for SLURM-based deployment: array
jobs, environment setup, post-run aggregation, and rsync templates.

## What each cell produces

Per-round metrics (all saved to `.npz`):

| Key | Description |
|---|---|
| `theta` | current $\theta_t \in \mathbf{R}^d$ |
| `recommendation`, `response` | length-$n$ tuples |
| `n_deviated` | # agents whose response $\neq$ recommendation |
| `per_agent_phi` | length-$n$ array of $\varphi_i$ |
| `max_phi`, `mean_phi` | summary stats of above |
| `regret`, `cum_regret` | moderator's regret per round and cumulative |
| `cut_norm` | $\|q_\theta\|_2$, 0 if no deviation |
| `inner_solver_time` | JLB call wall time (s) |
| `ce_cached` | did we reuse the cached CE? |
| `mixture_size` | support size of that round's CE mixture |
| `round_wall_time` | total round wall time (s) |

Final metrics:

| Key | Description |
|---|---|
| `final_theta` | terminal $\theta$ |
| `verify_ce` | `is_ce`, `max_violation`, `n_violated`, `worst_violation` |
| `total_rounds`, `last_dev_round` | |
| `total_wall_time` | |

## Two feedback models

- **BR** (γ = 0): agent deviates on any strict utility improvement.
- **γ-BR**: agent deviates only if the improvement exceeds γ.  With
  γ large enough, no agent ever deviates (useful as a sanity check /
  to study approximate equilibria).

Both are plugged via `experiments_overnight.feedback.make_feedback(...)`.

## Licence

MIT.
