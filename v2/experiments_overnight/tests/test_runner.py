"""End-to-end test of run_one_cell: plot + npz outputs are produced."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..runner import run_one_cell


def _minimal_cell(out_dir: str) -> dict:
    return dict(
        game_family="pigou", n=5, m=2, seed=0,
        feedback="BR", gamma=0.0,
        label="runner_e2e_pigou5",
        box_radius=5.0, box_lower=0.0,
        outer="ellipsoid",
        max_rounds=6, stop_after_no_dev=10,
        jlb_max_iters=500, jlb_reconstruct_every=20,
        out_dir=out_dir,
    )


def test_run_one_cell_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as td:
        summary = run_one_cell(_minimal_cell(td))
        assert summary["status"] == "ok"
        assert summary["total_rounds"] >= 1
        # .npz was saved
        npz = Path(summary["npz_path"])
        assert npz.exists()
        # Plots
        for suffix in ("regret", "deviations", "maxphi", "inner_time"):
            p = Path(td) / f"runner_e2e_pigou5_{suffix}.png"
            assert p.exists(), f"missing {p}"
        # .npz contents
        data = np.load(npz, allow_pickle=True)
        assert "pr_n_deviated" in data.files
        assert "pr_per_agent_phi" in data.files
        assert "pr_cum_regret" in data.files
        assert "pr_max_phi" in data.files
        assert "pr_inner_solver_time" in data.files
        assert "cfg" in data.files


def test_graph_coloring_cell() -> None:
    with tempfile.TemporaryDirectory() as td:
        cell = dict(
            game_family="graph_coloring", n=9, m=3, seed=0,
            grid_size=3,
            feedback="BR", gamma=0.0,
            label="runner_e2e_coloring",
            box_radius=3.0, box_lower=0.0,
            max_rounds=5, stop_after_no_dev=10,
            jlb_max_iters=500,
            true_collision_cost=1.0,
            out_dir=td,
        )
        summary = run_one_cell(cell)
        assert summary["status"] == "ok"
