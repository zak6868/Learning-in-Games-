"""Runner for overnight experiments.

Does NOT auto-launch anything.  Exposes:

    run_one_cell(cell: dict) -> ExperimentResult
    run_batch(cells: list[dict], n_workers: int, out_dir: str) -> None

`run_batch` uses multiprocessing.Pool, writes one .npz per cell and one
PNG set per cell via make_plots.

A cell dict has keys:
    game_family : 'pigou' | 'braess' | 'sfgrid3' | 'polymatrix_pricing' |
                  'graph_coloring' | 'bestshot_pg'
    n, m, seed, feedback, gamma, box_radius, max_rounds, stop_after_no_dev
    ...

Call from the command line:

    python -m experiments_overnight.runner --config configs/overnight.yaml \
          --workers 6 --outdir out/
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from recommend_unknown_games.congestion_game import (
    pigou_game, braess_game, sioux_falls_game,
)

from .environments.congestion_fm import CongestionFM
from .environments.polymatrix_fm import PolymatrixPricingFeatureMap
from .environments.graphical_fm import (
    GraphColoringFeatureMap, BestShotPGFeatureMap,
)
from .harness import ExperimentConfig, run_experiment, ExperimentResult
from .plot_utils import make_plots


def _grid_adj(rows: int, cols: int) -> list[tuple[int, ...]]:
    out = [[] for _ in range(rows * cols)]
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[i].append(nr * cols + nc)
    return [tuple(x) for x in out]


def _cycle_adj(n: int) -> list[tuple[int, ...]]:
    return [((i - 1) % n, (i + 1) % n) for i in range(n)]


def build_feature_map(cell: dict):
    """Dispatch on `game_family` and return a FeatureMap."""
    family = cell["game_family"]
    n = cell["n"]
    m = cell["m"]
    seed = cell.get("seed", 0)

    if family == "pigou":
        return CongestionFM(pigou_game(n_agents=n))
    if family == "braess":
        return CongestionFM(braess_game(n_agents=n, free_shortcut=True))
    if family == "sfgrid3":
        return CongestionFM(
            sioux_falls_game(n_agents=n, grid_size=3, k_routes=m)
        )
    if family == "sfgrid4":
        return CongestionFM(
            sioux_falls_game(n_agents=n, grid_size=4, k_routes=m)
        )
    if family == "polymatrix_pricing":
        return PolymatrixPricingFeatureMap(
            n=n, k=m, degree=cell.get("degree", 3), seed=seed,
        )
    if family == "graph_coloring":
        size = cell.get("grid_size", 3)
        adj = _grid_adj(size, size)
        assert size * size == n, f"n={n} incompatible with grid {size}×{size}"
        return GraphColoringFeatureMap(
            adj, n_colors=m,
            true_collision_cost=cell.get("true_collision_cost", 1.0),
        )
    if family == "bestshot_pg":
        topology = cell.get("topology", "cycle")
        if topology == "cycle":
            adj = _cycle_adj(n)
        else:
            raise ValueError(topology)
        return BestShotPGFeatureMap(
            adj,
            true_benefit=cell.get("true_benefit", 1.0),
            true_cost=cell.get("true_cost", 0.4),
        )
    raise ValueError(family)


def run_one_cell(cell: dict) -> dict:
    """Run one cell and return a summary dict (also saves .npz and plots)."""
    out_dir = Path(cell["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = build_feature_map(cell)
    label = cell["label"]
    cfg = ExperimentConfig(
        label=label,
        feature_map=fm,
        box_radius=cell.get("box_radius", 10.0),
        box_lower=cell.get("box_lower", 0.0),
        outer=cell.get("outer", "ellipsoid"),
        max_rounds=cell.get("max_rounds", 120),
        stop_after_no_dev=cell.get("stop_after_no_dev", 10),
        feedback_kind=cell.get("feedback", "BR"),
        gamma=cell.get("gamma", 0.0),
        jlb_max_iters=cell.get("jlb_max_iters", 100_000),
        jlb_initial_radius=cell.get("jlb_initial_radius", 10.0),
        jlb_reconstruct_every=cell.get("jlb_reconstruct_every", 100),
        use_structured_y_fallback=cell.get("use_structured_y_fallback", True),
        seed=cell.get("seed", 0),
    )

    try:
        res = run_experiment(cfg)
    except Exception as e:
        return {
            "label": label, "status": f"error:{type(e).__name__}:{e}",
            "cell": cell,
        }

    # Save .npz and plots.
    npz_path = out_dir / f"{label}.npz"
    res.to_npz(str(npz_path))
    make_plots(res, str(out_dir))

    summary = {
        "label": label,
        "game_family": cell["game_family"],
        "n": cell["n"], "m": cell["m"],
        "seed": cell.get("seed", 0),
        "feedback": cell.get("feedback", "BR"),
        "gamma": cell.get("gamma", 0.0),
        "total_rounds": res.final["total_rounds"],
        "last_dev_round": res.final["last_dev_round"],
        "total_wall_time": res.final["total_wall_time"],
        "is_ce": (
            res.final["verify_ce"]["is_ce"]
            if res.final["verify_ce"] else None
        ),
        "max_violation": (
            res.final["verify_ce"]["max_violation"]
            if res.final["verify_ce"] else None
        ),
        "cum_regret_final": (
            res.per_round[-1]["cum_regret"] if res.per_round else 0.0
        ),
        "status": "ok",
        "npz_path": str(npz_path),
    }
    return summary


def run_batch(
    cells: list[dict],
    n_workers: int = 4,
    summary_csv: str = "experiments_overnight/out/summary.csv",
) -> None:
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(cells)} cells with {n_workers} workers", flush=True)
    fieldnames = [
        "label", "game_family", "n", "m", "seed", "feedback", "gamma",
        "total_rounds", "last_dev_round", "total_wall_time",
        "is_ce", "max_violation", "cum_regret_final", "status", "npz_path",
    ]
    with open(summary_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_one_cell, cells)):
                with open(summary_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(
                        {k: row.get(k, "") for k in fieldnames}
                    )
                print(f"[{i+1}/{len(cells)}] {row.get('label')}: "
                      f"{row.get('status')}  "
                      f"regret={row.get('cum_regret_final', 'NA')}",
                      flush=True)
    else:
        for i, cell in enumerate(cells):
            row = run_one_cell(cell)
            with open(summary_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(
                    {k: row.get(k, "") for k in fieldnames}
                )
            print(f"[{i+1}/{len(cells)}] {row.get('label')}: "
                  f"{row.get('status')}", flush=True)


__all__ = ["build_feature_map", "run_one_cell", "run_batch"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_range(spec: str, max_len: int) -> list[int]:
    """Parse 'a-b' range, 'a,b,c' list, or 'all'."""
    if spec == "all":
        return list(range(max_len))
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), min(int(b) + 1, max_len)))
    return [int(x) for x in spec.split(",") if x]


def _load_cells_from_module(modpath: str) -> list[dict]:
    import importlib
    mod = importlib.import_module(modpath)
    return mod.all_cells()


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Run overnight moderator experiments.",
    )
    ap.add_argument(
        "--config",
        default="experiments_overnight.configs.overnight_cells",
        help="Module path exposing `all_cells()`.",
    )
    ap.add_argument("--cells", default="all",
                    help="'all', 'a-b', or comma list of indices")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--outdir", default="experiments_overnight/out")
    ap.add_argument("--summary-csv",
                    default="experiments_overnight/out/summary.csv")
    ap.add_argument("--list-only", action="store_true",
                    help="Just print cell indices + labels and exit.")
    args = ap.parse_args()

    cells_all = _load_cells_from_module(args.config)
    idx = _parse_range(args.cells, len(cells_all))
    cells = [cells_all[i] for i in idx]
    for c in cells:
        c.setdefault("out_dir", args.outdir)

    if args.list_only:
        for i, c in zip(idx, cells):
            print(f"{i:4d}  {c['label']}")
        return

    print(f"Selected {len(cells)}/{len(cells_all)} cells, "
          f"{args.workers} workers, outdir={args.outdir}")
    run_batch(cells, n_workers=args.workers,
              summary_csv=args.summary_csv)


if __name__ == "__main__":
    _main()
