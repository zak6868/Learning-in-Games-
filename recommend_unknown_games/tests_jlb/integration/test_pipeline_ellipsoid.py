"""End-to-end JLB-with-ellipsoid pipeline tests.

Validates that the full rigorous outer loop (ellipsoid + JLB oracle)
produces valid CEs on benchmark games.  This is the rigorous replacement
for the random-y / structured-y drivers.
"""

from __future__ import annotations

import numpy as np
import pytest
from gmpy2 import mpq

from ...compact_game import NormalFormCompactGame
from ...congestion_game import pigou_game, braess_game
from ...jlb_ellipsoid import run_jlb_ellipsoid
from ..fixtures.benchmark_games import (
    prisoners_dilemma,
    matching_pennies,
    shapley_game,
    chicken_game,
)
from ..utils.exact_validation import Mixture, max_incentive_violation


def _mixture(game, profiles, alpha):
    combined: dict[tuple[int, ...], object] = {}
    for p, a in zip(profiles, alpha):
        if a > 0:
            combined[p] = combined.get(p, mpq(0)) + mpq(float(a))
    total = mpq(0)
    for w in combined.values():
        total = total + mpq(w)
    weights = tuple(mpq(w) / total for w in combined.values())
    return Mixture(game.spec.action_counts, tuple(combined.keys()), weights)


class TestEllipsoidOnBenchmarks:
    @pytest.mark.parametrize(
        "name,game_fn",
        [
            ("PD", prisoners_dilemma),
            ("MP", matching_pennies),
            ("Chicken", chicken_game),
            ("Shapley", shapley_game),
        ],
    )
    def test_benchmark_produces_valid_ce(self, name, game_fn):
        game = NormalFormCompactGame(game_fn())
        result = run_jlb_ellipsoid(game, max_iterations=500,
                                   initial_radius=10.0,
                                   reconstruct_every=10)
        assert result.reconstructed, (
            f"{name}: ellipsoid failed to reconstruct CE, stopped={result.stopped_reason}"
        )
        mixture = _mixture(game, result.profiles, result.alpha)
        v = max_incentive_violation(game, mixture)
        assert v <= mpq(1, 10**6), f"{name}: violation {v}"


class TestEllipsoidOnCongestion:
    def test_pigou_small(self):
        g = pigou_game(n_agents=3)
        result = run_jlb_ellipsoid(g, max_iterations=500, reconstruct_every=5)
        assert result.reconstructed
        mixture = _mixture(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6)

    def test_braess_no_shortcut(self):
        g = braess_game(n_agents=3, free_shortcut=False)
        result = run_jlb_ellipsoid(g, max_iterations=500, reconstruct_every=5)
        assert result.reconstructed
        mixture = _mixture(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6)

    def test_pigou_mid_size(self):
        """Pigou n=10 -- the case where random-y failed."""
        g = pigou_game(n_agents=10)
        result = run_jlb_ellipsoid(g, max_iterations=2000, reconstruct_every=10)
        assert result.reconstructed, f"Pigou n=10 failed: {result.stopped_reason}"
        mixture = _mixture(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6), f"violation {v}"


class TestEllipsoidInvariants:
    def test_volume_decreases(self):
        """Over the full run, ellipsoid volume must decrease."""
        game = NormalFormCompactGame(shapley_game())
        result = run_jlb_ellipsoid(game, max_iterations=300)
        assert result.timings["final_log_vol_ratio"] < 0, (
            f"volume did not decrease: {result.timings}"
        )

    def test_distinct_profiles(self):
        """Every profile in the support is distinct (no duplicates)."""
        game = NormalFormCompactGame(shapley_game())
        result = run_jlb_ellipsoid(game, max_iterations=300)
        assert len(set(result.profiles)) == len(result.profiles)
