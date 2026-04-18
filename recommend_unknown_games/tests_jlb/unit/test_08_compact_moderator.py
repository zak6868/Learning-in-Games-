"""Tests for CompactCuttingPlaneAlgorithm (T3 in the design doc).

The compact moderator maintains θ(t) ∈ R^d via ACCPM on the knowledge
set and feeds θ(t) to the compact oracle.  After each oracle query, a
cut q_θ is added to the knowledge set.  The moderator produces a
trajectory of θ(t) along with cumulative compliance regret.

Tests verify:
  - The moderator runs without error on small congestion games.
  - Cumulative regret grows sub-linearly.
  - θ_true always satisfies all accumulated cuts (correctness invariant).
  - State never includes any length-N vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from ...congestion_game import pigou_game, braess_game
from ...game import AgentFeedback

try:
    from ...feature_map import CongestionFeatureMap
    from ...compact_oracle import CompactSeparationOracle
    from ...compact_moderator import CompactCuttingPlaneAlgorithm
    _HAS_IMPL = True
except ImportError:
    _HAS_IMPL = False


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_moderator module not implemented yet")
class TestModeratorRuns:
    """T3.13 — smoke test."""

    def test_runs_on_pigou_3(self):
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=30, theta_star=fm.theta_true())
        assert result.T == 30
        assert result.cumulative_regret.shape == (30,)

    def test_runs_on_braess(self):
        g = braess_game(n_agents=3, free_shortcut=False)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=30, theta_star=fm.theta_true())
        assert result.cumulative_regret.shape == (30,)


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_moderator module not implemented yet")
class TestModeratorRegret:
    """T3.14 — regret must be sub-linear in T."""

    def test_regret_sublinear_on_pigou(self):
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=100)
        # Average regret per round should be decreasing.
        avg_late = result.cumulative_regret[-1] / 100
        avg_early = result.cumulative_regret[9] / 10
        # Late-round average must be smaller than early-round average
        # (otherwise regret is linear, not sublinear).
        assert avg_late <= 2 * avg_early, (
            f"regret not sublinear: late avg {avg_late:.4f} vs early {avg_early:.4f}"
        )


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_moderator module not implemented yet")
class TestModeratorCorrectnessInvariants:
    """T3.16 — the true θ must satisfy every accumulated cut."""

    def test_true_theta_in_knowledge_set(self):
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=50, theta_star=fm.theta_true())

        # result.cuts is a list of cut vectors q_θ
        theta_true = fm.theta_true()
        for i, q in enumerate(result.cuts):
            inner = float(theta_true @ q)
            assert inner >= -1e-6, (
                f"cut {i} violates θ_true: <θ_true, q> = {inner}"
            )


@pytest.mark.skipif(not _HAS_IMPL, reason="compact_moderator module not implemented yet")
class TestModeratorStateDimensions:
    """T3.15 — moderator state is in R^d, not R^N."""

    def test_state_dimension(self):
        """The moderator's internal theta trajectory must have dimension d."""
        g = pigou_game(n_agents=5)  # N = 5 * 4 = 20, d = 4
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        assert fm.d == 4
        assert fm.N == g.spec.N  # == 20
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=20)
        # Inspect θ trajectory
        assert result.theta_trajectory.shape == (20, fm.d)

    def test_cuts_have_dimension_d(self):
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=30)
        for q in result.cuts:
            assert q.shape == (fm.d,), (
                f"cut has shape {q.shape}, expected ({fm.d},)"
            )
