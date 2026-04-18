"""Tests for the FeatureMap abstraction (T0 and T1 in the design doc).

The feature map converts between the compact parameter θ ∈ R^d and the
full w-vector in R^N.  For a CongestionGame with affine costs,
d = 2 * |E| and θ = (a_0, b_0, a_1, b_1, …, a_{|E|-1}, b_{|E|-1}).

These tests are deliberately written BEFORE the implementation exists so
they define the required API.  They will fail at import until
`compact_moderator/feature_map.py` is created with the correct classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from ...congestion_game import pigou_game, braess_game

# The implementation module (does not exist yet; tests will fail until
# `CongestionFeatureMap` is created).
try:
    from ...feature_map import FeatureMap, CongestionFeatureMap
    _HAS_IMPL = True
except ImportError:
    _HAS_IMPL = False


@pytest.mark.skipif(not _HAS_IMPL, reason="feature_map module not implemented yet")
class TestFeatureMapDimensions:
    """T0.1 and T0.4 – basic sanity checks on d, N, and zero-theta."""

    def test_pigou_d_is_2_times_n_edges(self):
        g = pigou_game(n_agents=3)
        fm = CongestionFeatureMap(g)
        assert fm.d == 2 * g.n_edges  # 2 * 2 = 4

    def test_braess_d_is_2_times_n_edges(self):
        g = braess_game(n_agents=3, free_shortcut=True)
        fm = CongestionFeatureMap(g)
        assert fm.d == 2 * g.n_edges  # 2 * 5 = 10

    def test_N_matches_game_spec(self):
        g = pigou_game(n_agents=3)
        fm = CongestionFeatureMap(g)
        assert fm.N == g.spec.N

    def test_zero_theta_gives_zero_w(self):
        g = pigou_game(n_agents=3)
        fm = CongestionFeatureMap(g)
        w = fm.w_from_theta(np.zeros(fm.d))
        assert np.all(w == 0.0)


@pytest.mark.skipif(not _HAS_IMPL, reason="feature_map module not implemented yet")
class TestFeatureMapWIdentity:
    """T0.2 – Φ(θ_true) must equal the game's own pack_w()."""

    def test_pigou_3_agents(self):
        g = pigou_game(n_agents=3)
        fm = CongestionFeatureMap(g)
        theta = fm.theta_true()
        w_from_map = fm.w_from_theta(theta)
        w_from_nf = g.as_normal_form().pack_w()
        np.testing.assert_allclose(
            w_from_map, w_from_nf, atol=1e-12,
            err_msg="Φ(θ_true) ≠ pack_w() for Pigou 3 agents"
        )

    def test_braess_no_shortcut_2_agents(self):
        g = braess_game(n_agents=2, free_shortcut=False)
        fm = CongestionFeatureMap(g)
        theta = fm.theta_true()
        w_from_map = fm.w_from_theta(theta)
        w_from_nf = g.as_normal_form().pack_w()
        np.testing.assert_allclose(w_from_map, w_from_nf, atol=1e-12)

    def test_braess_with_shortcut_3_agents(self):
        g = braess_game(n_agents=3, free_shortcut=True)
        fm = CongestionFeatureMap(g)
        theta = fm.theta_true()
        w_from_map = fm.w_from_theta(theta)
        w_from_nf = g.as_normal_form().pack_w()
        np.testing.assert_allclose(w_from_map, w_from_nf, atol=1e-12)


@pytest.mark.skipif(not _HAS_IMPL, reason="feature_map module not implemented yet")
class TestFeatureMapLinearity:
    """T0.3 – Φ must be linear in θ."""

    def test_linearity_on_pigou(self):
        g = pigou_game(n_agents=3)
        fm = CongestionFeatureMap(g)
        rng = np.random.default_rng(0)
        theta1 = rng.standard_normal(fm.d)
        theta2 = rng.standard_normal(fm.d)
        a, b = 0.3, -1.7
        w_lhs = fm.w_from_theta(a * theta1 + b * theta2)
        w_rhs = a * fm.w_from_theta(theta1) + b * fm.w_from_theta(theta2)
        np.testing.assert_allclose(w_lhs, w_rhs, atol=1e-12)

    def test_linearity_on_braess(self):
        g = braess_game(n_agents=3, free_shortcut=True)
        fm = CongestionFeatureMap(g)
        rng = np.random.default_rng(1)
        theta1 = rng.standard_normal(fm.d)
        theta2 = rng.standard_normal(fm.d)
        w_lhs = fm.w_from_theta(theta1 + theta2)
        w_rhs = fm.w_from_theta(theta1) + fm.w_from_theta(theta2)
        np.testing.assert_allclose(w_lhs, w_rhs, atol=1e-12)


@pytest.mark.skipif(not _HAS_IMPL, reason="feature_map module not implemented yet")
class TestCutProjectionIdentity:
    """T1.5 and T1.6 – <θ, Φ^T q> = <Φ θ, q> for all θ, q."""

    @pytest.mark.parametrize(
        "game_fn",
        [
            lambda: pigou_game(n_agents=2),
            lambda: pigou_game(n_agents=3),
            lambda: braess_game(n_agents=2, free_shortcut=False),
            lambda: braess_game(n_agents=3, free_shortcut=True),
        ],
    )
    def test_inner_product_identity_true_theta(self, game_fn):
        g = game_fn()
        fm = CongestionFeatureMap(g)
        rng = np.random.default_rng(42)
        # Random cut vector (doesn't need to be physically meaningful)
        q = rng.standard_normal(fm.N)
        q_theta = fm.project_cut_dense(q)
        theta = fm.theta_true()
        lhs = float(theta @ q_theta)
        rhs = float(fm.w_from_theta(theta) @ q)
        assert np.isclose(lhs, rhs, atol=1e-10), (
            f"<θ, Φ^T q> = {lhs}, <Φθ, q> = {rhs}"
        )

    @pytest.mark.parametrize(
        "game_fn",
        [
            lambda: pigou_game(n_agents=3),
            lambda: braess_game(n_agents=3, free_shortcut=True),
        ],
    )
    def test_inner_product_identity_random_theta(self, game_fn):
        g = game_fn()
        fm = CongestionFeatureMap(g)
        rng = np.random.default_rng(7)
        for _ in range(10):
            theta = rng.standard_normal(fm.d)
            q = rng.standard_normal(fm.N)
            q_theta = fm.project_cut_dense(q)
            lhs = float(theta @ q_theta)
            rhs = float(fm.w_from_theta(theta) @ q)
            assert np.isclose(lhs, rhs, atol=1e-10)


@pytest.mark.skipif(not _HAS_IMPL, reason="feature_map module not implemented yet")
class TestSparseProjectionMatchesDense:
    """T1.7 – the sparse projection (only touching nonzero blocks) must
    agree with a dense Φ^T q."""

    def test_single_block_matches_dense(self):
        """A cut with one nonzero block at (agent=0, action_k=1) and
        explicit values should project the same way sparse vs dense."""
        g = pigou_game(n_agents=3)
        fm = CongestionFeatureMap(g)
        # Build a sparse cut: only one block nonzero
        agent = 0
        k = 1  # refer to action 1 in agent 0's block
        d_i = g.spec.d[agent]
        rng = np.random.default_rng(0)
        block = rng.standard_normal(d_i)

        # Dense version
        q_dense = np.zeros(fm.N)
        offset = 0  # agent 0 starts at offset 0
        q_dense[offset + (k - 1) * d_i : offset + k * d_i] = block
        q_theta_dense = fm.project_cut_dense(q_dense)

        # Sparse version
        cut_blocks = {(agent, k): block}
        q_theta_sparse = fm.project_cut_sparse(cut_blocks)

        np.testing.assert_allclose(q_theta_sparse, q_theta_dense, atol=1e-12)

    def test_multiple_blocks_match_dense(self):
        """Multiple deviating-agent blocks; sparse == dense projection."""
        g = braess_game(n_agents=3, free_shortcut=True)
        fm = CongestionFeatureMap(g)
        rng = np.random.default_rng(2)

        # Simulate what the oracle produces: a few blocks
        cut_blocks = {}
        q_dense = np.zeros(fm.N)
        # Need to compute offsets to rebuild dense version
        from ...utils import IndexCache
        cache = IndexCache(g.spec)
        for (agent, k), (_, _) in [((0, 1), (0, 0)), ((1, 2), (0, 0)),
                                    ((2, 1), (0, 0))]:
            d_i = g.spec.d[agent]
            block = rng.standard_normal(d_i)
            cut_blocks[(agent, k)] = block
            off = int(cache.offsets[agent])
            q_dense[off + (k - 1) * d_i : off + k * d_i] += block

        q_theta_dense = fm.project_cut_dense(q_dense)
        q_theta_sparse = fm.project_cut_sparse(cut_blocks)
        np.testing.assert_allclose(q_theta_sparse, q_theta_dense, atol=1e-12)


@pytest.mark.skipif(not _HAS_IMPL, reason="feature_map module not implemented yet")
class TestProjectionStructure:
    """T1.8 – hand-checkable: for a Pigou cut touching only the first
    agent's k=1 block, the projection's nonzero θ entries correspond to
    edges in routes 0 and 1 of agent 0."""

    def test_pigou_single_agent_block_touches_correct_edges(self):
        g = pigou_game(n_agents=2, a=0.0, b=1.0, const_route_cost=1.0)
        fm = CongestionFeatureMap(g)
        d_0 = g.spec.d[0]
        # Block for agent 0, k=1 (i.e., w_0(0, 1))
        # w_0(0, 1)[a_{-0}] = u_0(1, a_{-0}) - u_0(0, a_{-0})
        # Put +1 at all a_{-0} entries.
        cut_blocks = {(0, 1): np.ones(d_0)}
        q_theta = fm.project_cut_sparse(cut_blocks)
        # θ layout: (a_0, b_0, a_1, b_1).
        # Agent 0 has routes 0={edge 0} and 1={edge 1}.
        # Edges touched: 0 (in route 0) and 1 (in route 1).
        # Every entry should be nonzero.
        assert np.any(q_theta[:2] != 0.0), "edge 0 (a_0 or b_0) not touched"
        assert np.any(q_theta[2:4] != 0.0), "edge 1 (a_1 or b_1) not touched"
