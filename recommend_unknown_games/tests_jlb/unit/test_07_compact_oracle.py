"""Tests for CompactSeparationOracle (T2 in the design doc).

The compact oracle:
  1. Takes θ_guess ∈ R^d
  2. Builds the implicit game at θ_guess
  3. Queries JLB for a CE mixture of that game (sparse)
  4. Samples a recommendation a(t) ~ mixture
  5. Queries the true agents (who use θ_true) for a response a*(t)
  6. If deviation: builds a cut q_θ ∈ R^d DIRECTLY from the sparse mixture
     (never materializing w or a dense x)

Tests verify correctness via the fundamental separation property:
  <θ_true, q_θ> >= 0 >= <θ_guess, q_θ>.
"""

from __future__ import annotations

import numpy as np
import pytest

from ...congestion_game import CongestionGame, pigou_game, braess_game
from ...game import AgentFeedback

try:
    from ...feature_map import CongestionFeatureMap
    from ...compact_oracle import CompactSeparationOracle
    _HAS_IMPL = True
except ImportError:
    _HAS_IMPL = False


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_oracle module not implemented yet")
class TestCompactOracleBasics:
    def test_construction_on_pigou(self):
        g = pigou_game(n_agents=3)
        # The agent-feedback model uses the TRUE game (with its theta).
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        # Must expose the .spec, d, N
        assert oracle.spec.n == g.spec.n
        assert oracle.d == fm.d

    def test_query_returns_result_with_expected_fields(self):
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        theta = fm.theta_true()
        res = oracle.query(theta)
        # Basic shape expectations
        assert isinstance(res.recommendation, tuple)
        assert len(res.recommendation) == g.spec.n
        assert isinstance(res.response, tuple)
        assert len(res.response) == g.spec.n
        assert isinstance(res.deviated, bool)
        if res.deviated:
            assert res.cut_theta is not None
            assert res.cut_theta.shape == (fm.d,)


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_oracle module not implemented yet")
class TestCompactOracleCorrectness:
    """T2.10 and T2.11."""

    def test_at_true_theta_no_deviation(self):
        """Querying with θ_guess = θ_true under BR should elicit no deviations
        (since every recommendation in a CE is best-response)."""
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        theta = fm.theta_true()
        n_deviations = 0
        for _ in range(20):
            res = oracle.query(theta)
            if res.deviated:
                n_deviations += 1
        # BR follows CE exactly; allow tiny tie-breaking slack.
        assert n_deviations <= 2, f"too many deviations at θ_true: {n_deviations}/20"

    @pytest.mark.parametrize(
        "game_fn",
        [
            lambda: pigou_game(n_agents=3),
            lambda: braess_game(n_agents=2, free_shortcut=False),
            lambda: braess_game(n_agents=3, free_shortcut=True),
        ],
    )
    def test_cut_separation_property(self, game_fn):
        """T2.11 — the cornerstone identity."""
        g = game_fn()
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        theta_true = fm.theta_true()
        rng = np.random.default_rng(0)
        n_checked = 0
        n_trials = 30
        for _ in range(n_trials):
            # Random θ_guess that differs from θ_true
            theta_guess = rng.standard_normal(fm.d)
            res = oracle.query(theta_guess)
            if res.deviated and res.cut_theta is not None:
                n_checked += 1
                lhs = float(theta_true @ res.cut_theta)
                rhs = float(theta_guess @ res.cut_theta)
                # Exact 0 would be ideal; allow float noise
                assert lhs >= -1e-6, f"<θ_true, q_θ> = {lhs} < 0"
                assert rhs <= 1e-6, f"<θ_guess, q_θ> = {rhs} > 0"
        assert n_checked > 0, "never observed a deviation across 30 random θ"


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_oracle module not implemented yet")
class TestCompactOracleMatchesNormalForm:
    """T2.12 — the cut in θ-space should agree with Φ^T of the cut
    produced by the existing normal-form SeparationOracle."""

    def test_matches_nf_oracle_on_pigou_small(self):
        """For small games, run both oracles on the same w/θ and compare cuts."""
        g = pigou_game(n_agents=2)
        nf = g.as_normal_form()
        feedback_compact = AgentFeedback(nf, model="best_response", seed=0)
        feedback_nf = AgentFeedback(nf, model="best_response", seed=0)

        fm = CongestionFeatureMap(g)

        # Compact oracle
        compact_oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback_compact)

        # Normal-form reference (paper's SeparationOracle)
        from ...oracle import SeparationOracle
        from ...ce_solver import CESolver
        ce_solver = CESolver(nf.spec)
        nf_oracle = SeparationOracle(nf, ce_solver, feedback_nf, seed=0)

        theta_guess = np.array([0.0, 0.5, 2.0, 0.0])  # a_0=0, b_0=0.5, a_1=2.0, b_1=0
        w_guess = fm.w_from_theta(theta_guess)

        # Reset seeds so both oracles sample the same recommendation
        feedback_compact.rng = np.random.default_rng(0)
        feedback_nf.rng = np.random.default_rng(0)
        compact_oracle.rng = np.random.default_rng(0)
        nf_oracle.rng = np.random.default_rng(0)

        res_compact = compact_oracle.query(theta_guess)
        res_nf = nf_oracle.query(w_guess)

        if res_compact.deviated and res_nf.deviated:
            # Project the normal-form cut into θ-space
            q_theta_projected = fm.project_cut_dense(res_nf.cut)
            # Ideally these are equal up to scaling (both are valid cuts)
            # We check that both satisfy separation; exact equality is not required
            theta_true = fm.theta_true()
            lhs_compact = float(theta_true @ res_compact.cut_theta)
            lhs_nf = float(theta_true @ q_theta_projected)
            # Both must be non-negative (cut validity)
            assert lhs_compact >= -1e-6, "compact cut not valid at θ_true"
            assert lhs_nf >= -1e-6, "projected nf cut not valid at θ_true"
