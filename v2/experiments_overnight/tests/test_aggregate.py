"""Verify aggregate_plots produces a per-config mean±std plot."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..runner import run_one_cell
from ..aggregate_plots import aggregate_and_plot


def test_aggregate_multi_seed() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for seed in range(3):
            cell = dict(
                game_family="pigou", n=5, m=2, seed=seed,
                feedback="BR", gamma=0.0,
                label=f"agg_test_pigou5_s{seed}",
                box_radius=5.0, box_lower=0.0,
                max_rounds=6, stop_after_no_dev=10,
                jlb_max_iters=500, jlb_reconstruct_every=20,
                out_dir=str(td),
            )
            summary = run_one_cell(cell)
            assert summary["status"] == "ok"

        out_agg = td / "aggregated"
        aggregate_and_plot(str(td), str(out_agg))

        # Expect one aggregated plot for "agg_test_pigou5" (seed suffix stripped).
        agg_png = out_agg / "agg_test_pigou5_agg.png"
        assert agg_png.exists(), f"missing aggregated plot {agg_png}"
