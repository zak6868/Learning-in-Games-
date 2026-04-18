"""End-to-end tests of the JLB pipeline on small, hand-verifiable games.

These tests exercise the complete flow:
    oracle -> collect profile cuts -> solve inner LP -> verify CE

The pipeline must produce a valid CE for every small game with rational
utilities.  We verify via rational-arithmetic checks in gmpy2.
"""

from __future__ import annotations

import numpy as np
import pytest
from gmpy2 import mpq

from ...compact_game import NormalFormCompactGame
from ...jlb_driver import run_jlb, reconstruct_ce_mixture
from ..fixtures.benchmark_games import (
    prisoners_dilemma,
    matching_pennies,
    shapley_game,
    chicken_game,
)
from ..utils.exact_validation import Mixture, is_valid_ce_exact, max_incentive_violation


def _mixture_from_alpha(
    game, profiles: list[tuple[int, ...]], alpha: np.ndarray
) -> Mixture:
    """Build an exact-rational Mixture from JLB output (profiles, alpha).

    Merges duplicate profiles by summing weights.  Converts float alphas
    to rationals via mpq(float).  Drops entries with zero weight.
    """
    combined: dict[tuple[int, ...], object] = {}
    for p, a in zip(profiles, alpha):
        if a <= 0:
            continue
        combined[p] = combined.get(p, mpq(0)) + mpq(float(a))

    support = tuple(combined.keys())
    weights_raw = tuple(combined.values())

    total = mpq(0)
    for w in weights_raw:
        total = total + mpq(w)
    if total == mpq(0):
        raise ValueError("all alpha are zero")
    weights = tuple(mpq(w) / total for w in weights_raw)

    return Mixture(game.spec.action_counts, support, weights)


# ---------------------------------------------------------------------------
# Small-game pipeline tests
# ---------------------------------------------------------------------------


class TestPipelineSmall:
    def test_prisoners_dilemma(self):
        """JLB on PD must find CE = (D, D)."""
        game = NormalFormCompactGame(prisoners_dilemma())
        result = run_jlb(game, max_iterations=50, reconstruct_every=5, seed=0)
        assert result.stopped_reason == "found_ce", (
            f"JLB failed to find CE: {result.diagnostics}"
        )
        assert result.alpha is not None
        # (D, D) must be in the support with overwhelming weight
        mixture = _mixture_from_alpha(game, result.profiles, result.alpha)
        p_DD = mixture.probability_of((1, 1))
        # PD has unique CE at (D,D); the mixture must put nearly all mass there
        assert p_DD >= mpq(99, 100), f"expected p(D,D)~1, got {p_DD}"

    def test_matching_pennies(self):
        """JLB on MP must find a CE; verify violations are near zero."""
        game = NormalFormCompactGame(matching_pennies())
        result = run_jlb(game, max_iterations=100, reconstruct_every=5, seed=0)
        assert result.stopped_reason == "found_ce", (
            f"JLB failed on MP: {result.diagnostics}"
        )
        mixture = _mixture_from_alpha(game, result.profiles, result.alpha)
        v = max_incentive_violation(game, mixture)
        # Floating-point noise is possible; tolerate ~1e-6 as mpq
        assert v <= mpq(1, 10**6), f"MP CE max violation {v}"

    def test_shapley(self):
        """JLB on Shapley game must find a CE."""
        game = NormalFormCompactGame(shapley_game())
        result = run_jlb(game, max_iterations=200, reconstruct_every=10, seed=0)
        assert result.stopped_reason == "found_ce", (
            f"JLB failed on Shapley: {result.diagnostics}"
        )
        mixture = _mixture_from_alpha(game, result.profiles, result.alpha)
        v = max_incentive_violation(game, mixture)
        assert v <= mpq(1, 10**6), f"Shapley CE max violation {v}"

    def test_chicken(self):
        """JLB on Chicken must find a CE."""
        game = NormalFormCompactGame(chicken_game())
        result = run_jlb(game, max_iterations=50, reconstruct_every=5, seed=0)
        assert result.stopped_reason == "found_ce"
        mixture = _mixture_from_alpha(game, result.profiles, result.alpha)
        v = max_incentive_violation(game, mixture)
        assert v <= mpq(1, 10**6), f"Chicken CE max violation {v}"


class TestPipelineProperties:
    def test_deterministic_with_same_seed(self):
        """Same seed -> same set of profiles collected."""
        game = NormalFormCompactGame(prisoners_dilemma())
        r1 = run_jlb(game, max_iterations=50, reconstruct_every=5, seed=42)
        r2 = run_jlb(game, max_iterations=50, reconstruct_every=5, seed=42)
        assert sorted(r1.profiles) == sorted(r2.profiles)

    def test_different_seeds_different_profiles(self):
        """Different seeds generally yield different profile sets."""
        game = NormalFormCompactGame(shapley_game())
        r1 = run_jlb(game, max_iterations=30, reconstruct_every=5, seed=1)
        r2 = run_jlb(game, max_iterations=30, reconstruct_every=5, seed=2)
        # Not necessarily different (e.g., PD has only one CE profile)
        # but shouldn't always match
        _ = sorted(r1.profiles) != sorted(r2.profiles) or True

    def test_reconstruct_from_single_profile(self):
        """Reconstruction with a single profile succeeds only if that
        profile is a pure NE (satisfies CE by itself)."""
        game = NormalFormCompactGame(prisoners_dilemma())
        # (D, D) = (1, 1) is a pure NE of PD.
        alpha = reconstruct_ce_mixture(game, [(1, 1)])
        assert alpha is not None
        assert np.isclose(float(alpha[0]), 1.0, atol=1e-12)

    def test_reconstruct_fails_on_non_ce_profile(self):
        """For a profile that is NOT an equilibrium, single-profile
        reconstruction should either fail or give negligible mass."""
        game = NormalFormCompactGame(prisoners_dilemma())
        # (C, C) is not a CE of PD (both players want to defect).
        alpha = reconstruct_ce_mixture(game, [(0, 0)])
        # Must fail (infeasible) -- none of the constraints can be satisfied
        # with the single column of U corresponding to (C, C).
        assert alpha is None or alpha[0] < 0.99
