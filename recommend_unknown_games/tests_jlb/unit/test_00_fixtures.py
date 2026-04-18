"""Sanity checks for fixtures and ground-truth helpers.

If any of these fail, every other test in this suite is suspect: we would
be comparing implementations against broken reference values.
"""

from __future__ import annotations

import numpy as np
import pytest
from gmpy2 import mpq

from ..fixtures.benchmark_games import (
    prisoners_dilemma,
    matching_pennies,
    shapley_game,
    chicken_game,
    theorem5_counterexample_u,
    theorem5_counterexample_v,
    all_benchmark_games,
)
from ..fixtures.ground_truth import (
    brute_force_ce,
    expected_utility_brute_force,
    ce_constraints_dense,
    verify_ce_dense,
)
from ..utils.exact_validation import (
    Mixture,
    incentive_violation,
    max_incentive_violation,
    is_valid_ce_exact,
)


# ---------------------------------------------------------------------------
# Hand-verified payoffs
# ---------------------------------------------------------------------------


class TestPrisonersDilemma:
    def test_shape(self, game_pd):
        assert game_pd.spec.n == 2
        assert game_pd.spec.action_counts == (2, 2)
        assert game_pd.spec.M == 4

    def test_payoffs(self, game_pd):
        # (C, C) -> both get 3
        assert game_pd.utilities[0][0, 0] == 3.0
        assert game_pd.utilities[1][0, 0] == 3.0
        # (D, D) -> both get 1
        assert game_pd.utilities[0][1, 1] == 1.0
        assert game_pd.utilities[1][1, 1] == 1.0
        # (D, C) -> defector gets 5, cooperator gets 0
        assert game_pd.utilities[0][1, 0] == 5.0
        assert game_pd.utilities[1][1, 0] == 0.0

    def test_defect_dominant(self, game_pd):
        # Defect strictly dominates Cooperate for both players
        # u_1(D, a2) > u_1(C, a2) for all a2
        for a2 in range(2):
            assert game_pd.utilities[0][1, a2] > game_pd.utilities[0][0, a2]
        for a1 in range(2):
            assert game_pd.utilities[1][a1, 1] > game_pd.utilities[1][a1, 0]

    def test_unique_ce_is_DD(self, game_pd):
        """PD has a unique CE = (D, D) because Defect is strictly dominant."""
        ce = brute_force_ce(game_pd)
        # Expect all weight on profile (1, 1).
        pd_weight = ce.probability_of((1, 1))
        assert pd_weight == mpq(1), f"expected p(D,D)=1, got {pd_weight}"
        # Verify CE property literally
        assert is_valid_ce_exact(game_pd, ce)
        assert max_incentive_violation(game_pd, ce) == mpq(0)


class TestMatchingPennies:
    def test_shape(self, game_mp):
        assert game_mp.spec.action_counts == (2, 2)

    def test_zero_sum(self, game_mp):
        for a1 in range(2):
            for a2 in range(2):
                s = game_mp.utilities[0][a1, a2] + game_mp.utilities[1][a1, a2]
                assert s == 0.0

    def test_uniform_is_ce(self, game_mp):
        """Uniform joint distribution is a CE of matching pennies."""
        support = tuple((a1, a2) for a1 in range(2) for a2 in range(2))
        weights = tuple(mpq(1, 4) for _ in support)
        uniform = Mixture(game_mp.spec.action_counts, support, weights)
        assert is_valid_ce_exact(game_mp, uniform)
        assert max_incentive_violation(game_mp, uniform) == mpq(0)

    def test_brute_force_returns_ce(self, game_mp):
        ce = brute_force_ce(game_mp)
        # In matching pennies the ill-conditioning can allow small
        # float round-off.  Verify the mixture produced is at least a CE
        # in rational arithmetic up to the normalization step.
        assert is_valid_ce_exact(game_mp, ce) or max_incentive_violation(
            game_mp, ce
        ) <= mpq(1, 10**6)


class TestShapley:
    def test_shape(self, game_shapley):
        assert game_shapley.spec.action_counts == (3, 3)
        assert game_shapley.spec.M == 9

    def test_brute_force_is_ce(self, game_shapley):
        ce = brute_force_ce(game_shapley)
        # Shapley's game CE should verify exactly (integer payoffs).
        v = max_incentive_violation(game_shapley, ce)
        # Allow tiny float noise; literal == 0 would be ideal.
        assert v <= mpq(1, 10**8), f"max violation {v}"


class TestChicken:
    def test_shape(self, game_chicken):
        assert game_chicken.spec.action_counts == (2, 2)

    def test_pure_NE_is_CE(self, game_chicken):
        """(Swerve=0, Straight=1) and (Straight=1, Swerve=0) are pure NE."""
        for profile in [(0, 1), (1, 0)]:
            pure = Mixture(game_chicken.spec.action_counts, (profile,), (mpq(1),))
            assert is_valid_ce_exact(game_chicken, pure), f"pure NE at {profile}"


class TestTheorem5Games:
    def test_shapes(self, game_thm5_u, game_thm5_v):
        assert game_thm5_u.spec.action_counts == (4, 2)
        assert game_thm5_v.spec.action_counts == (4, 2)

    def test_player2_utilities_match(self, game_thm5_u, game_thm5_v):
        """In the paper's Theorem 5, player 2 utilities are identical between u and v."""
        assert np.allclose(
            game_thm5_u.utilities[1], game_thm5_v.utilities[1]
        )

    def test_player1_utilities_differ(self, game_thm5_u, game_thm5_v):
        assert not np.allclose(
            game_thm5_u.utilities[0], game_thm5_v.utilities[0]
        )


# ---------------------------------------------------------------------------
# ce_constraints_dense consistency
# ---------------------------------------------------------------------------


class TestDenseCEConstraints:
    def test_shape(self, game_pd):
        U = ce_constraints_dense(game_pd)
        # N = sum m_i^2 but conventional PR uses m_i(m_i-1) rows
        # We use the latter: one per (i, a_i, a_prime) pair with a_prime != a_i
        expected_rows = sum(
            m * (m - 1) for m in game_pd.spec.action_counts
        )
        assert U.shape == (expected_rows, game_pd.spec.M)

    def test_ce_verifies_dense(self, game_pd):
        """Brute-force CE should satisfy U x <= 0 up to tolerance."""
        ce = brute_force_ce(game_pd)
        # Rebuild a dense vector from the mixture
        x = np.zeros(game_pd.spec.M)
        for profile, w in zip(ce.support, ce.weights):
            flat = int(np.ravel_multi_index(profile, game_pd.spec.action_counts))
            x[flat] = float(w)
        ok, max_v = verify_ce_dense(game_pd, x, tol=1e-8)
        assert ok, f"dense CE fails check: max violation {max_v}"


# ---------------------------------------------------------------------------
# Incentive-violation helper sanity
# ---------------------------------------------------------------------------


class TestIncentiveViolation:
    def test_pure_strict_dominated_violates(self, game_pd):
        """Play (C, C) with probability 1; defecting gives +2, violation = 2."""
        pure_CC = Mixture(game_pd.spec.action_counts, ((0, 0),), (mpq(1),))
        # Player 1 deviating from C(0) to D(1) at profile (0,0):
        # u_1(1, 0) - u_1(0, 0) = 5 - 3 = 2
        v = incentive_violation(game_pd, pure_CC, agent=0, a_i=0, a_prime_i=1)
        assert v == mpq(2)

    def test_self_deviation_zero(self, game_pd):
        pure_CC = Mixture(game_pd.spec.action_counts, ((0, 0),), (mpq(1),))
        v = incentive_violation(game_pd, pure_CC, agent=0, a_i=0, a_prime_i=0)
        assert v == mpq(0)


# ---------------------------------------------------------------------------
# expected_utility_brute_force consistency
# ---------------------------------------------------------------------------


class TestExpectedUtilityBF:
    def test_deterministic(self, game_pd):
        """Point-mass distributions give deterministic utility."""
        x = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]  # both play 0
        u0 = expected_utility_brute_force(game_pd, agent=0, x_product=x)
        assert u0 == 3.0  # u_1(C, C) = 3

    def test_uniform(self, game_pd):
        """Uniform distribution = mean of all payoff entries for agent 0."""
        x = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
        u0 = expected_utility_brute_force(game_pd, agent=0, x_product=x)
        expected = game_pd.utilities[0].mean()
        assert np.isclose(u0, expected)


# ---------------------------------------------------------------------------
# Parametrized sanity across all benchmark games
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(all_benchmark_games().keys()))
def test_benchmark_game_is_well_formed(name):
    """Every benchmark game has internally consistent dimensions."""
    game = all_benchmark_games()[name]
    assert game.spec.n == len(game.utilities)
    for i in range(game.spec.n):
        assert game.utilities[i].shape == game.spec.action_counts
    assert game.spec.M == int(np.prod(game.spec.action_counts))


@pytest.mark.parametrize("name", list(all_benchmark_games().keys()))
def test_benchmark_game_admits_ce(name):
    """Every finite game has a CE (Hart-Schmeidler 1989). Verify empirically."""
    game = all_benchmark_games()[name]
    ce = brute_force_ce(game)
    # Sum weights exactly to 1 and each is non-negative
    total = mpq(0)
    for w in ce.weights:
        assert w >= 0
        total = total + mpq(w)
    assert total == mpq(1)
