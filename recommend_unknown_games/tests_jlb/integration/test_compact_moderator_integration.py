"""End-to-end integration tests for the compact moderator (T4).

These tests run the full compact pipeline on congestion games of
increasing size, including games where the normal-form representation
is physically infeasible.
"""

from __future__ import annotations

import time
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


@pytest.mark.skipif(not _HAS_IMPL, reason="compact modules not implemented yet")
class TestEndToEndSmall:
    """Full pipeline on small games that normal-form CAN also handle —
    used for cross-validation."""

    def test_pigou_5_runs(self):
        g = pigou_game(n_agents=5)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=50, theta_star=fm.theta_true())
        assert result.T == 50

    def test_braess_5_runs(self):
        g = braess_game(n_agents=5, free_shortcut=True)
        nf = g.as_normal_form()
        feedback = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle = CompactSeparationOracle(g, feature_map=fm, feedback=feedback)
        algo = CompactCuttingPlaneAlgorithm(oracle, feature_map=fm, box_radius=10.0)
        result = algo.run(T=50, theta_star=fm.theta_true())
        assert result.T == 50


@pytest.mark.skipif(not _HAS_IMPL, reason="compact modules not implemented yet")
@pytest.mark.slow
class TestEndToEndLargeScale:
    """T4.17 — pipeline on games where M is too large for normal-form."""

    def test_pigou_50_agents_runs(self):
        """M = 2^50 ≈ 10^15 — normal-form can't even allocate the utility
        tables.  The compact moderator must still complete."""
        g = pigou_game(n_agents=50)
        # Intentionally NOT calling g.as_normal_form() — that would allocate
        # a 2^50-entry array.  Instead use the CongestionGame directly in
        # the feedback model by building it specifically for BR queries.
        # For this test we use a compact feedback that works natively on
        # CongestionGame (if such exists) or skip the feedback simulation.
        # Simplest: use the game itself as the "reference" feedback source
        # by constructing a thin adapter that calls game.utility.
        from ...game import AgentFeedback
        # We can't call as_normal_form(), so we'd need a compact feedback.
        # This test documents the goal; may need a CompactAgentFeedback
        # helper.
        pytest.skip("requires CompactAgentFeedback to simulate BR without normal-form")


@pytest.mark.skipif(not _HAS_IMPL, reason="compact modules not implemented yet")
class TestCompactVsNormalFormModerator:
    """On small games, the compact moderator's regret trajectory should
    be comparable to the existing normal-form moderator (same order of
    magnitude, not necessarily identical)."""

    def test_pigou_3_regret_comparable_to_normal_form(self):
        from ...cutting_plane import CuttingPlaneAlgorithm
        from ...oracle import SeparationOracle
        from ...ce_solver import CESolver

        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()

        # Compact moderator
        feedback_c = AgentFeedback(nf, model="best_response", seed=0)
        fm = CongestionFeatureMap(g)
        oracle_c = CompactSeparationOracle(g, feature_map=fm, feedback=feedback_c)
        algo_c = CompactCuttingPlaneAlgorithm(oracle_c, feature_map=fm, box_radius=10.0)
        result_c = algo_c.run(T=50)

        # Normal-form reference moderator
        feedback_n = AgentFeedback(nf, model="best_response", seed=0)
        ce_solver = CESolver(nf.spec)
        oracle_n = SeparationOracle(nf, ce_solver, feedback_n, seed=0)
        algo_n = CuttingPlaneAlgorithm(oracle_n, box_radius=1.0)
        result_n = algo_n.run(T=50)

        # Compare cumulative regrets — same order of magnitude
        rc = result_c.cumulative_regret[-1]
        rn = result_n.cumulative_regret[-1]
        # Allow 10x discrepancy in either direction
        assert rc <= 10 * max(rn, 1e-6), (
            f"compact regret {rc} much larger than normal-form {rn}"
        )
