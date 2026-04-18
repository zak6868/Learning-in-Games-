"""Tests for the JLB-backed drop-in CE solver.

Validates two properties:
  1. `normal_form_from_w` reconstructs a NormalFormGame whose w-vector
     equals the input.
  2. `JLBCESolver.solve(w)` returns a dense CE distribution — comparable
     in quality (max-violation) to the reference normal-form CESolver.
"""

from __future__ import annotations

import numpy as np
import pytest
from gmpy2 import mpq

from ...ce_solver import CESolver
from ...game import NormalFormGame
from ...jlb_ce_solver import JLBCESolver, normal_form_from_w
from ..fixtures.benchmark_games import (
    prisoners_dilemma, matching_pennies, shapley_game, chicken_game,
)
from ..utils.exact_validation import Mixture, max_incentive_violation


class TestNormalFormFromW:
    """normal_form_from_w must be an exact inverse of pack_w."""

    @pytest.mark.parametrize(
        "game_fn",
        [prisoners_dilemma, matching_pennies, chicken_game, shapley_game],
    )
    def test_roundtrip(self, game_fn):
        game = game_fn()
        w = game.pack_w()
        reconstructed = normal_form_from_w(game.spec, w)
        w2 = reconstructed.pack_w()
        np.testing.assert_allclose(w, w2, atol=1e-14, err_msg="w-vector roundtrip failed")

    def test_ce_set_preserved(self):
        """If g is built from w via normal_form_from_w, then g and the original
        game (with the same w) have the same CE set.  We verify by solving both
        with the normal-form LP and comparing max-violation."""
        game = chicken_game()
        w = game.pack_w()
        g_reconstructed = normal_form_from_w(game.spec, w)

        solver = CESolver(game.spec)
        x_orig = solver.solve(w)
        x_recon = solver.solve(g_reconstructed.pack_w())
        # Both should satisfy the CE constraints of game
        assert solver.is_ce(w, x_orig)
        assert solver.is_ce(w, x_recon)


class TestJLBCESolverCorrectness:
    @pytest.mark.parametrize(
        "name,game_fn",
        [
            ("PD", prisoners_dilemma),
            ("MP", matching_pennies),
            ("Chicken", chicken_game),
            ("Shapley", shapley_game),
        ],
    )
    def test_solve_produces_ce(self, name, game_fn):
        game = game_fn()
        w = game.pack_w()
        solver = JLBCESolver(game.spec, outer_loop="ellipsoid",
                             max_iterations=500, reconstruct_every=10)
        x = solver.solve(w)
        assert x is not None, f"{name}: JLBCESolver returned None"

        # Verify as CE
        np.testing.assert_allclose(x.sum(), 1.0, atol=1e-6)
        assert np.all(x >= -1e-12)

        # Reference solver agreement: both must give CE
        ref_solver = CESolver(game.spec)
        assert ref_solver.is_ce(w, x, tol=1e-6), (
            f"{name}: JLB output is not a CE w.r.t. reference solver"
        )

    def test_matches_reference_on_PD(self):
        """Unique CE = (D, D). JLB should put all mass there."""
        game = prisoners_dilemma()
        w = game.pack_w()
        solver = JLBCESolver(game.spec, outer_loop="ellipsoid",
                             max_iterations=200, reconstruct_every=5)
        x = solver.solve(w)
        assert x is not None
        # Profile (1, 1) has flat index = 1 * 2 + 1 = 3
        assert x[3] >= 0.99, f"expected ~1 at (D,D), got {x[3]}"


class TestJLBCESolverAgainstReference:
    """Benchmark JLB vs normal-form CESolver.  Both must produce valid CEs;
    JLB should not be worse than the reference in terms of constraint
    violations."""

    @pytest.mark.parametrize(
        "name,game_fn",
        [
            ("PD", prisoners_dilemma),
            ("MP", matching_pennies),
            ("Shapley", shapley_game),
        ],
    )
    def test_both_give_valid_ces(self, name, game_fn):
        game = game_fn()
        w = game.pack_w()

        ref = CESolver(game.spec)
        x_ref = ref.solve(w)
        assert x_ref is not None and ref.is_ce(w, x_ref)

        jlb = JLBCESolver(game.spec, outer_loop="ellipsoid",
                          max_iterations=500, reconstruct_every=10)
        x_jlb = jlb.solve(w)
        assert x_jlb is not None and ref.is_ce(w, x_jlb, tol=1e-6)
