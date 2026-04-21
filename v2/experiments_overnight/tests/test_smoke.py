"""Smoke tests: run every environment on a tiny instance end-to-end and
verify the harness completes and produces well-formed metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommend_unknown_games.congestion_game import pigou_game, braess_game

from ..harness import ExperimentConfig, run_experiment
from ..environments.congestion_fm import CongestionFM
from ..environments.polymatrix_fm import PolymatrixPricingFeatureMap
from ..environments.graphical_fm import (
    GraphColoringFeatureMap, BestShotPGFeatureMap,
)


def _cycle(n: int) -> list[tuple[int, ...]]:
    return [((i - 1) % n, (i + 1) % n) for i in range(n)]


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


# ---------------------------------------------------------------------
# Per-environment smoke: each config runs ≤ 5s.
# ---------------------------------------------------------------------


def test_smoke_pigou() -> None:
    fm = CongestionFM(pigou_game(n_agents=5))
    cfg = ExperimentConfig(
        label="smoke_pigou", feature_map=fm,
        box_radius=5.0, box_lower=0.0,
        outer="ellipsoid", max_rounds=8, stop_after_no_dev=10,
        feedback_kind="BR", gamma=0.0,
        jlb_max_iters=1000, jlb_reconstruct_every=20, seed=0,
    )
    res = run_experiment(cfg)
    assert len(res.per_round) >= 1
    assert res.final["total_rounds"] == len(res.per_round)
    assert "verify_ce" in res.final
    # θ dimension check
    for r in res.per_round:
        assert r["theta"].shape == (fm.d,)
        assert r["per_agent_phi"].shape == (fm.spec.n,)
        assert r["n_deviated"] >= 0


def test_smoke_braess() -> None:
    fm = CongestionFM(braess_game(n_agents=3, free_shortcut=True))
    cfg = ExperimentConfig(
        label="smoke_braess", feature_map=fm,
        box_radius=5.0, max_rounds=8,
        jlb_max_iters=1000, seed=0,
    )
    res = run_experiment(cfg)
    assert len(res.per_round) >= 1


def test_smoke_polymatrix() -> None:
    fm = PolymatrixPricingFeatureMap(n=5, k=3, degree=3, seed=0)
    cfg = ExperimentConfig(
        label="smoke_polymatrix", feature_map=fm,
        box_radius=150.0, box_lower=0.0,
        max_rounds=8, jlb_max_iters=1000, seed=0,
    )
    res = run_experiment(cfg)
    assert len(res.per_round) >= 1


def test_smoke_coloring() -> None:
    adj = _grid_adj(3, 3)     # 9 nodes
    fm = GraphColoringFeatureMap(adj, n_colors=3, true_collision_cost=1.0)
    cfg = ExperimentConfig(
        label="smoke_coloring", feature_map=fm,
        box_radius=3.0, box_lower=0.0,
        max_rounds=8, jlb_max_iters=1000, seed=0,
    )
    res = run_experiment(cfg)
    assert len(res.per_round) >= 1


def test_smoke_bestshot() -> None:
    fm = BestShotPGFeatureMap(
        _cycle(6), true_benefit=1.0, true_cost=0.4,
    )
    cfg = ExperimentConfig(
        label="smoke_bestshot", feature_map=fm,
        box_radius=2.0, box_lower=0.0,
        max_rounds=8, jlb_max_iters=1000, seed=0,
    )
    res = run_experiment(cfg)
    assert len(res.per_round) >= 1


# ---------------------------------------------------------------------
# γ-BR sanity: with a huge γ, agents never deviate.
# ---------------------------------------------------------------------


def test_gamma_br_no_devs() -> None:
    fm = CongestionFM(pigou_game(n_agents=5))
    cfg = ExperimentConfig(
        label="smoke_gammaBR", feature_map=fm,
        box_radius=5.0, box_lower=0.0,
        outer="ellipsoid", max_rounds=15, stop_after_no_dev=5,
        feedback_kind="gamma_BR", gamma=100.0,  # huge → never deviate
        jlb_max_iters=1000, seed=0,
    )
    res = run_experiment(cfg)
    # With γ=100 and pigou at n=5 (small utility range), no deviation
    # should ever fire.
    total_dev = sum(r["n_deviated"] for r in res.per_round)
    assert total_dev == 0
    # Should early-stop after 5 no-dev rounds.
    assert len(res.per_round) <= 5 + 2
