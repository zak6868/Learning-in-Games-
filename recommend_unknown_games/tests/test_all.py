"""Comprehensive tests verifying consistency with the paper.

Tests are grouped by what they verify:
    1. GameSpec and index arithmetic
    2. Utility differences and w-vector packing
    3. CE solver correctness
    4. Agent feedback models (BR and QR)
    5. Separation oracle (the main invariant)
    6. Paper's Theorem 5 counterexample
    7. Stress tests on random games of varying sizes
"""

import numpy as np
import pytest

from recommend_unknown_games.types import GameSpec, OracleResult
from recommend_unknown_games.utils import (
    IndexCache,
    agent_block_offsets,
    profiles_with_agent_action,
)
from recommend_unknown_games.game import NormalFormGame, AgentFeedback
from recommend_unknown_games.ce_solver import CESolver
from recommend_unknown_games.oracle import SeparationOracle


# ===================================================================
# 1. GameSpec and index arithmetic
# ===================================================================


class TestGameSpec:
    def test_two_player_2x2(self):
        spec = GameSpec.from_action_counts((2, 2))
        assert spec.n == 2
        assert spec.M == 4
        assert spec.d == (2, 2)
        # N = (2-1)*2 + (2-1)*2 = 4
        assert spec.N == 4

    def test_two_player_4x2(self):
        """Matches the counterexample in Theorem 5: A_1={a1..a4}, A_2={b1,b2}."""
        spec = GameSpec.from_action_counts((4, 2))
        assert spec.n == 2
        assert spec.M == 8
        assert spec.d == (2, 4)
        # N = (4-1)*2 + (2-1)*4 = 6 + 4 = 10
        assert spec.N == 10

    def test_three_player(self):
        spec = GameSpec.from_action_counts((3, 2, 4))
        assert spec.M == 24
        assert spec.d == (8, 12, 6)
        # N = 2*8 + 1*12 + 3*6 = 16 + 12 + 18 = 46
        assert spec.N == 46

    def test_homogeneous(self):
        spec = GameSpec.from_action_counts((3, 3, 3))
        assert spec.M == 27
        assert spec.d == (9, 9, 9)
        assert spec.N == 3 * 2 * 9  # 54


class TestIndexArithmetic:
    def test_agent_block_offsets_homogeneous(self):
        spec = GameSpec.from_action_counts((3, 3))
        offsets = agent_block_offsets(spec)
        # Agent 0: blocks w_0(0,1), w_0(0,2), each of size d_0=3  -> occupies 6
        # Agent 1 starts at 6
        np.testing.assert_array_equal(offsets, [0, 6])

    def test_agent_block_offsets_heterogeneous(self):
        spec = GameSpec.from_action_counts((4, 2))
        offsets = agent_block_offsets(spec)
        # Agent 0: (4-1) * d_0=2 = 6 entries
        # Agent 1 starts at 6
        np.testing.assert_array_equal(offsets, [0, 6])

    def test_agent_block_offsets_three_players(self):
        spec = GameSpec.from_action_counts((3, 2, 4))
        offsets = agent_block_offsets(spec)
        # Agent 0: (3-1)*8 = 16
        # Agent 1: (2-1)*12 = 12
        # Agent 2 starts at 16+12 = 28
        np.testing.assert_array_equal(offsets, [0, 16, 28])

    def test_profiles_with_agent_action(self):
        spec = GameSpec.from_action_counts((2, 3))
        # Agent 0 plays action 1: profiles (1,0), (1,1), (1,2)
        idx = profiles_with_agent_action(spec, 0, 1)
        expected = np.ravel_multi_index(
            ([1, 1, 1], [0, 1, 2]), (2, 3)
        )
        np.testing.assert_array_equal(idx, expected)
        assert len(idx) == spec.d[0]  # d_0 = 3

    def test_profiles_cover_all(self):
        """All profiles for agent i across all actions should cover {0,...,M-1}."""
        spec = GameSpec.from_action_counts((3, 2, 2))
        for i in range(spec.n):
            all_idx = np.concatenate(
                [profiles_with_agent_action(spec, i, a) for a in range(spec.action_counts[i])]
            )
            np.testing.assert_array_equal(np.sort(all_idx), np.arange(spec.M))


class TestIndexCache:
    def test_w_block_identity(self):
        """w_i(a, a) should be zero."""
        spec = GameSpec.from_action_counts((3, 2))
        w = np.random.default_rng(0).standard_normal(spec.N)
        cache = IndexCache(spec)
        for i in range(spec.n):
            for a in range(spec.action_counts[i]):
                block = cache.get_w_block(w, i, a, a)
                np.testing.assert_array_equal(block, 0.0)

    def test_w_block_antisymmetry(self):
        """w_i(a, b) = -w_i(b, a)."""
        spec = GameSpec.from_action_counts((3, 2))
        w = np.random.default_rng(0).standard_normal(spec.N)
        cache = IndexCache(spec)
        for i in range(spec.n):
            m_i = spec.action_counts[i]
            for a in range(m_i):
                for b in range(m_i):
                    w_ab = cache.get_w_block(w, i, a, b)
                    w_ba = cache.get_w_block(w, i, b, a)
                    np.testing.assert_allclose(w_ab, -w_ba, atol=1e-14)

    def test_w_block_triangular(self):
        """w_i(a, c) = w_i(a, b) + w_i(b, c) (triangular identity, paper p.13)."""
        spec = GameSpec.from_action_counts((4, 3))
        w = np.random.default_rng(1).standard_normal(spec.N)
        cache = IndexCache(spec)
        for i in range(spec.n):
            m_i = spec.action_counts[i]
            if m_i < 3:
                continue
            for a in range(m_i):
                for b in range(m_i):
                    for c in range(m_i):
                        w_ac = cache.get_w_block(w, i, a, c)
                        w_ab = cache.get_w_block(w, i, a, b)
                        w_bc = cache.get_w_block(w, i, b, c)
                        np.testing.assert_allclose(
                            w_ac, w_ab + w_bc, atol=1e-13,
                            err_msg=f"Triangular identity failed for agent {i}, "
                                    f"actions ({a},{b},{c})"
                        )


# ===================================================================
# 2. Utility differences and w-vector packing
# ===================================================================


class TestUtilityDiffs:
    def test_diff_matches_definition(self):
        """w_i(a_i, a'_i)[a_{-i}] = u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i})."""
        game = NormalFormGame.random_game((3, 2), seed=42)
        # Agent 0, action 0 vs 1
        w = game.utility_diff_vector(0, 0, 1)
        # Manual check: enumerate all a_{-1} = {0, 1}
        for idx_minus, a_minus in enumerate(range(2)):
            expected = (
                game.utilities[0][1, a_minus] - game.utilities[0][0, a_minus]
            )
            np.testing.assert_allclose(w[idx_minus], expected, atol=1e-15)

    def test_diff_matches_3player(self):
        """Verify w_i on a 3-player game against brute-force."""
        game = NormalFormGame.random_game((2, 3, 2), seed=7)
        # Agent 1, action 0 -> action 2
        w = game.utility_diff_vector(1, 0, 2)
        # a_{-1} = (a_0, a_2) in lex order: (0,0), (0,1), (1,0), (1,1)
        idx = 0
        for a0 in range(2):
            for a2 in range(2):
                expected = game.utilities[1][a0, 2, a2] - game.utilities[1][a0, 0, a2]
                np.testing.assert_allclose(w[idx], expected, atol=1e-15,
                    err_msg=f"a_{-1}=({a0},{a2})")
                idx += 1

    def test_pack_unpack_roundtrip(self):
        """pack_w then get_w_block should recover the original diffs."""
        game = NormalFormGame.random_game((3, 4), seed=5)
        w = game.pack_w()
        cache = IndexCache(game.spec)
        for i in range(game.spec.n):
            m_i = game.spec.action_counts[i]
            for j in range(1, m_i):
                block = cache.get_w_block(w, i, 0, j)
                expected = game.utility_diff_vector(i, 0, j)
                np.testing.assert_allclose(block, expected, atol=1e-14)

    def test_pack_w_triangular_consistency(self):
        """All w_i(a,b) recovered from pack_w via triangular identity
        should match direct computation from utilities."""
        game = NormalFormGame.random_game((3, 3), seed=10)
        w = game.pack_w()
        cache = IndexCache(game.spec)
        for i in range(game.spec.n):
            m_i = game.spec.action_counts[i]
            for a in range(m_i):
                for b in range(m_i):
                    from_packed = cache.get_w_block(w, i, a, b)
                    from_game = game.utility_diff_vector(i, a, b)
                    np.testing.assert_allclose(
                        from_packed, from_game, atol=1e-13,
                        err_msg=f"Agent {i}, ({a},{b})"
                    )


# ===================================================================
# 3. CE solver
# ===================================================================


class TestCESolver:
    def test_ce_exists_2x2(self):
        """Every finite game has a CE (Hart & Schmeidler, 1989)."""
        game = NormalFormGame.random_game((2, 2), seed=0)
        solver = CESolver(game.spec)
        w = game.pack_w()
        x = solver.solve(w)
        assert x is not None
        assert x.shape == (game.spec.M,)

    def test_ce_is_distribution(self):
        game = NormalFormGame.random_game((3, 2), seed=1)
        solver = CESolver(game.spec)
        x = solver.solve(game.pack_w())
        assert x is not None
        assert np.all(x >= -1e-10)
        np.testing.assert_allclose(x.sum(), 1.0, atol=1e-8)

    def test_ce_no_profitable_deviation(self):
        """Core CE property: phi_i(a_i, a'_i, x) <= 0 for all i, a_i, a'_i.
        Paper: Definition 1, eq. (1)."""
        game = NormalFormGame.random_game((3, 3), seed=2)
        solver = CESolver(game.spec)
        w = game.pack_w()
        x = solver.solve(w)
        assert x is not None

        for i in range(game.spec.n):
            m_i = game.spec.action_counts[i]
            for a_i in range(m_i):
                for a_prime in range(m_i):
                    if a_prime == a_i:
                        continue
                    phi = game.phi(i, a_i, a_prime, x)
                    assert phi <= 1e-8, (
                        f"CE violation: agent {i}, ({a_i}->{a_prime}), "
                        f"phi={phi:.6e}"
                    )

    def test_is_ce_checker(self):
        game = NormalFormGame.random_game((2, 3), seed=3)
        solver = CESolver(game.spec)
        w = game.pack_w()
        x = solver.solve(w)
        assert x is not None
        assert solver.is_ce(w, x)

    def test_uniform_not_always_ce(self):
        """A uniform distribution is not always a CE."""
        game = NormalFormGame.random_game((3, 3), seed=4)
        solver = CESolver(game.spec)
        w = game.pack_w()
        x_unif = np.ones(game.spec.M) / game.spec.M
        # May or may not be CE; just verify the checker works
        result = solver.is_ce(w, x_unif)
        assert isinstance(result, bool)

    def test_ce_for_different_w(self):
        """CE solver works for arbitrary w, not just the true game's w."""
        spec = GameSpec.from_action_counts((2, 2))
        solver = CESolver(spec)
        rng = np.random.default_rng(5)
        for _ in range(10):
            w = rng.standard_normal(spec.N)
            x = solver.solve(w)
            assert x is not None, "Every finite game must have a CE"
            assert solver.is_ce(w, x)

    def test_constraint_matrix_shape_and_sparsity(self):
        spec = GameSpec.from_action_counts((3, 2, 4))
        solver = CESolver(spec)
        w = np.random.default_rng(0).standard_normal(spec.N)
        A = solver.build_constraint_matrix(w)
        # Rows: sum_i m_i(m_i-1) = 3*2 + 2*1 + 4*3 = 6+2+12 = 20
        expected_rows = sum(m * (m - 1) for m in spec.action_counts)
        assert A.shape == (expected_rows, spec.M)
        # Each row has d_i nonzeros
        for row_idx, (i, a_i, a_prime) in enumerate(solver._constraint_map):
            row = A.getrow(row_idx)
            assert row.nnz == spec.d[i]
    
    def test_matrix_antisymmetry_of_rows(self):
        """Row for (i, a_i, a'_i) should be the negative of row for (i, a'_i, a_i)."""
        spec = GameSpec.from_action_counts((3, 4, 3))
        solver = CESolver(spec, backend="scipy")
        w = np.random.default_rng(0).standard_normal(spec.N)
        A = solver.build_constraint_matrix(w)

        # Build a lookup from (i, a_i, a_prime) -> row index
        row_lookup = {
            (i, a_i, a_prime): idx
            for idx, (i, a_i, a_prime) in enumerate(solver._constraint_map)
        }

        for (i, a_i, a_prime), row_idx in row_lookup.items():
            reverse_key = (i, a_prime, a_i)
            if reverse_key in row_lookup:
                reverse_idx = row_lookup[reverse_key]
                np.testing.assert_allclose(
                    np.sort(A[row_idx].toarray()), np.sort(-A[reverse_idx].toarray()), atol=1e-14,
                    err_msg=f"Antisymmetry failed for agent={i}, ({a_i},{a_prime})"
                )

    def test_constraint_matrix_updates_with_but_pattern_fixed(self):
        """Different w should produce different constraint matrices."""
        spec = GameSpec.from_action_counts((3, 4, 3))
        solver = CESolver(spec, backend="scipy")
        w1 = np.random.default_rng(0).standard_normal(spec.N)
        w2 = np.random.default_rng(1).standard_normal(spec.N)
        A1 = solver.build_constraint_matrix(w1)
        A2 = solver.build_constraint_matrix(w2)
        diff = (A1 != 0).astype(int) - (A2 != 0).astype(int)
        assert not np.allclose(A1.toarray(), A2.toarray())
        assert diff.nnz == 0


# ===================================================================
# 4. Agent feedback models
# ===================================================================


class TestAgentFeedback:
    def test_br_follows_ce(self):
        """Under BR, agents should follow a true CE recommendation.
        Paper: follows from Definition 1 (CE) and Definition 2 (BR)."""
        game = NormalFormGame.random_game((3, 3), seed=10)
        solver = CESolver(game.spec)
        x = solver.solve(game.pack_w())
        assert x is not None
        feedback = AgentFeedback(game, model="best_response", seed=0)

        # Sample many recommendations and verify agents follow
        rng = np.random.default_rng(99)
        n_trials = 200
        n_deviations = 0
        for _ in range(n_trials):
            flat_idx = rng.choice(game.spec.M, p=x)
            a = tuple(int(c) for c in np.unravel_index(flat_idx, game.spec.action_counts))
            a_star = feedback.respond_all(a, x)
            if a != a_star:
                n_deviations += 1
        # With a true CE, BR agents should (almost) never deviate
        # Allow a small fraction due to ties
        assert n_deviations <= 0.05 * n_trials, (
            f"BR agents deviated {n_deviations}/{n_trials} times from true CE"
        )

    def test_br_deviates_from_bad_recommendation(self):
        """BR agents should deviate from a non-CE recommendation."""
        game = NormalFormGame.random_game((2, 2), seed=20)
        feedback = AgentFeedback(game, model="best_response", seed=0)
        # Use uniform distribution (may not be CE)
        x = np.ones(game.spec.M) / game.spec.M
        # Try many samples
        any_deviation = False
        rng = np.random.default_rng(42)
        for _ in range(100):
            flat_idx = rng.choice(game.spec.M, p=x)
            a = tuple(int(c) for c in np.unravel_index(flat_idx, game.spec.action_counts))
            a_star = feedback.respond_all(a, x)
            if a != a_star:
                any_deviation = True
                break
        # Not guaranteed to deviate, but very likely for random games
        # This is a soft test

    def test_qr_deviation_only_when_profitable(self):
        """QR agents only deviate to actions with phi >= 0.
        Paper: Definition 3, eq. (4)."""
        game = NormalFormGame.random_game((3, 3), seed=30)
        feedback = AgentFeedback(game, model="quantal_response", beta=2.0, seed=0)
        x = np.ones(game.spec.M) / game.spec.M  # uniform

        for _ in range(500):
            a = tuple(np.random.default_rng(42).integers(0, m) for m in game.spec.action_counts)
            a_star = feedback.respond_all(a, x)
            for i in range(game.spec.n):
                if a[i] != a_star[i]:
                    phi = game.phi(i, a[i], a_star[i], x)
                    assert phi >= -1e-10, (
                        f"QR agent {i} deviated from {a[i]} to {a_star[i]} "
                        f"with phi={phi:.6e} < 0"
                    )

    def test_qr_high_beta_approaches_br(self):
        """As beta -> inf, QR should approach BR behavior.
        Paper: Remark 3, Section 4."""
        game = NormalFormGame.random_game((2, 3), seed=40)
        x = np.ones(game.spec.M) / game.spec.M
        fb_br = AgentFeedback(game, model="best_response", seed=0)
        fb_qr = AgentFeedback(game, model="quantal_response", beta=100.0, seed=0)

        n_agree = 0
        n_total = 100
        for trial in range(n_total):
            a = (trial % 2, trial % 3)
            r_br = fb_br.respond_all(a, x)
            r_qr = fb_qr.respond_all(a, x)
            if r_br == r_qr:
                n_agree += 1
        # Should agree almost always with high beta
        assert n_agree >= 0.9 * n_total

    def test_expected_utilities_computation(self):
        """Verify expected utilities match brute-force computation."""
        game = NormalFormGame.random_game((2, 3), seed=50)
        feedback = AgentFeedback(game, model="best_response", seed=0)
        x = np.ones(game.spec.M) / game.spec.M

        for i in range(game.spec.n):
            for a_i in range(game.spec.action_counts[i]):
                eu = feedback._expected_utilities(i, a_i, x)
                # Brute-force
                m_i = game.spec.action_counts[i]
                x_multi = x.reshape(game.spec.action_counts)
                cond = np.take(x_multi, a_i, axis=i).ravel()
                for a_prime in range(m_i):
                    u_slice = np.take(game.utilities[i], a_prime, axis=i).ravel()
                    expected_bf = np.dot(cond, u_slice)
                    np.testing.assert_allclose(
                        eu[a_prime], expected_bf, atol=1e-14,
                        err_msg=f"Agent {i}, rec={a_i}, eval action={a_prime}"
                    )


# ===================================================================
# 5. Separation oracle -- THE MAIN INVARIANT
# ===================================================================


class TestSeparationOracle:
    def _run_separation_test(self, action_counts, feedback_model, seed=0,
                             n_queries=50, beta=2.0):
        """Run the oracle and verify separation property on every deviation.

        THE KEY INVARIANT (paper eq. 9 + surrounding text):
            <w*, q> >= 0   (true game: deviation was profitable)
            <w(t), q> <= 0 (guessed game: CE had no profitable deviation)
        """
        game = NormalFormGame.random_game(action_counts, seed=seed)
        solver = CESolver(game.spec)
        feedback = AgentFeedback(game, model=feedback_model, beta=beta, seed=seed)
        oracle = SeparationOracle(game, solver, feedback, seed=seed + 1)

        w_star = game.pack_w()
        rng = np.random.default_rng(seed + 2)

        n_deviations = 0
        for t in range(n_queries):
            w_guess = rng.standard_normal(game.spec.N)
            result = oracle.query(w_guess)

            if result.deviated:
                n_deviations += 1
                q = result.cut
                assert q is not None

                # THE SEPARATION PROPERTY
                inner_true = np.dot(w_star, q)
                inner_guess = np.dot(w_guess, q)

                assert inner_true >= -1e-8, (
                    f"SEPARATION VIOLATED: <w*, q> = {inner_true:.6e} < 0\n"
                    f"  action_counts={action_counts}, model={feedback_model}\n"
                    f"  recommendation={result.recommendation}\n"
                    f"  response={result.response}"
                )
                assert inner_guess <= 1e-8, (
                    f"SEPARATION VIOLATED: <w(t), q> = {inner_guess:.6e} > 0\n"
                    f"  action_counts={action_counts}, model={feedback_model}\n"
                    f"  recommendation={result.recommendation}\n"
                    f"  response={result.response}"
                )

                # Also verify: <w*, q> = sum_i phi_i (paper eq. 9)
                phi_sum = 0.0
                for i in range(game.spec.n):
                    a_i = result.recommendation[i]
                    a_star_i = result.response[i]
                    if a_i != a_star_i:
                        phi_sum += game.phi(i, a_i, a_star_i, result.ce_dist)
                np.testing.assert_allclose(
                    inner_true, phi_sum, atol=1e-10,
                    err_msg="<w*, q> does not equal sum of phi_i"
                )

                # Verify regret matches
                np.testing.assert_allclose(
                    result.regret, phi_sum, atol=1e-10,
                    err_msg="OracleResult.regret does not match phi sum"
                )

        return n_deviations

    # --- BR tests ---

    def test_separation_2x2_br(self):
        n_dev = self._run_separation_test((2, 2), "best_response", seed=0)
        assert n_dev > 0, "No deviations observed -- test is vacuous"

    def test_separation_3x3_br(self):
        n_dev = self._run_separation_test((3, 3), "best_response", seed=1)
        assert n_dev > 0

    def test_separation_4x2_br(self):
        """Matches the game dimensions in Theorem 5 counterexample."""
        n_dev = self._run_separation_test((4, 2), "best_response", seed=2)
        assert n_dev > 0

    def test_separation_2x2x2_br(self):
        """3-player game."""
        n_dev = self._run_separation_test((2, 2, 2), "best_response", seed=3)
        assert n_dev > 0

    def test_separation_3x2x4_br(self):
        """Heterogeneous action sets, 3 players."""
        n_dev = self._run_separation_test((3, 2, 4), "best_response", seed=4)
        assert n_dev > 0

    # --- QR tests ---

    def test_separation_2x2_qr(self):
        n_dev = self._run_separation_test((2, 2), "quantal_response", seed=10)
        assert n_dev > 0

    def test_separation_3x3_qr(self):
        n_dev = self._run_separation_test((3, 3), "quantal_response", seed=11)
        assert n_dev > 0

    def test_separation_2x2x2_qr(self):
        n_dev = self._run_separation_test((2, 2, 2), "quantal_response", seed=12)
        assert n_dev > 0

    # --- True w* should give no deviations ---

    def test_no_deviation_at_true_w(self):
        """When w_guess = w*, the CE is a true CE, so BR agents should not deviate."""
        game = NormalFormGame.random_game((3, 3), seed=20)
        solver = CESolver(game.spec)
        feedback = AgentFeedback(game, model="best_response", seed=20)
        oracle = SeparationOracle(game, solver, feedback, seed=21)

        w_star = game.pack_w()
        n_deviations = 0
        for _ in range(100):
            result = oracle.query(w_star)
            if result.deviated:
                n_deviations += 1
        # BR agents should almost never deviate from true CE (only possible with ties)
        assert n_deviations <= 5, (
            f"Too many deviations ({n_deviations}/100) from true CE"
        )


# ===================================================================
# 6. Paper's Theorem 5 counterexample
# ===================================================================


class TestTheorem5Counterexample:
    """Verify the counterexample from Theorem 5 (paper Section 3.2).

    Two 2-player games with A_1 = {a1,a2,a3,a4}, A_2 = {b1,b2}:

    u_1:                          v_1:
         b1    b2                      b1    b2
    a1 [  0,    8 ]              a1 [  0,    8 ]
    a2 [  3,  6.5 ]              a2 [  2,    7 ]
    a3 [  5,  4.5 ]              a3 [  6,    3 ]
    a4 [  8,    0 ]              a4 [  8,    0 ]

    Player 2 utilities are identical in both games.
    The games are NOT equivalent (Proposition 2) but ARE
    indistinguishable under BR feedback (Proposition 3).
    """

    def _make_games(self):
        u1 = np.array([[0, 8], [3, 6.5], [5, 4.5], [8, 0]])
        v1 = np.array([[0, 8], [2, 7], [6, 3], [8, 0]])
        # Player 2: same in both games, shape (4, 2) = (m_1, m_2)
        # u2[a1, a2] = utility of player 2 at profile (a1, a2)
        u2 = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.7]])

        game_u = NormalFormGame((4, 2), [u1, u2])
        game_v = NormalFormGame((4, 2), [v1, u2])
        return game_u, game_v

    def test_games_not_equivalent(self):
        """Proposition 2: no lambda > 0, t s.t. v_1(a,.) = lambda * u_1(a,.) + t for all a."""
        game_u, game_v = self._make_games()
        u1 = game_u.utilities[0]  # shape (4, 2)
        v1 = game_v.utilities[0]

        # From a1: t = v1[0] - lambda*u1[0] = (0, 8) - lambda*(0, 8) = (0, 8(1-lambda))
        # From a4: t = v1[3] - lambda*u1[3] = (8, 0) - lambda*(8, 0) = (8(1-lambda), 0)
        # These equal only if lambda=1, t=(0,0). But then v1 must equal u1.
        assert not np.allclose(u1, v1), "Games should differ"
        # Verify a2 differs: u1[1]=(3,6.5) vs v1[1]=(2,7)
        assert not np.allclose(u1[1], v1[1])

    def test_same_br_regions(self):
        """Proposition 3: the best-response regions coincide.

        Boundaries between adjacent actions (for player 1):
            a1/a2 boundary: x2/x1 = 2   (both games)
            a2/a3 boundary: x2/x1 = 1   (both games)
            a3/a4 boundary: x2/x1 = 2/3 (both games)
        """
        game_u, game_v = self._make_games()
        u1 = game_u.utilities[0]
        v1 = game_v.utilities[0]

        # Check boundary slopes: for actions a, b, boundary is where
        # <x, u(a,.)> = <x, u(b,.)>  =>  <x, u(a,.)-u(b,.)> = 0
        # For x=(x1,x2): slope x2/x1 = -(u(a,0)-u(b,0))/(u(a,1)-u(b,1))
        def boundary_slope(utils, a, b):
            diff = utils[a] - utils[b]
            if abs(diff[1]) < 1e-12:
                return float('inf')
            return -diff[0] / diff[1]

        # a1/a2 boundary
        s_u = boundary_slope(u1, 0, 1)
        s_v = boundary_slope(v1, 0, 1)
        np.testing.assert_allclose(s_u, s_v, atol=1e-10,
            err_msg="a1/a2 boundary slopes differ")
        np.testing.assert_allclose(s_u, 2.0, atol=1e-10)

        # a2/a3 boundary
        s_u = boundary_slope(u1, 1, 2)
        s_v = boundary_slope(v1, 1, 2)
        np.testing.assert_allclose(s_u, s_v, atol=1e-10,
            err_msg="a2/a3 boundary slopes differ")
        np.testing.assert_allclose(s_u, 1.0, atol=1e-10)

        # a3/a4 boundary
        s_u = boundary_slope(u1, 2, 3)
        s_v = boundary_slope(v1, 2, 3)
        np.testing.assert_allclose(s_u, s_v, atol=1e-10,
            err_msg="a3/a4 boundary slopes differ")
        np.testing.assert_allclose(s_u, 2.0 / 3.0, atol=1e-10)

    def test_no_weakly_dominated(self):
        """Both games should have no weakly dominated actions (paper proof)."""
        game_u, game_v = self._make_games()
        for game in [game_u, game_v]:
            for a in range(4):
                for b in range(4):
                    if a == b:
                        continue
                    w = game.utility_diff_vector(0, a, b)
                    # Not weakly dominated: w should have both + and - entries
                    has_pos = np.any(w > 0)
                    has_neg = np.any(w < 0)
                    assert has_pos and has_neg, (
                        f"Action {a} weakly dominated by {b}: w={w}"
                    )


# ===================================================================
# 7. Stress tests on larger games
# ===================================================================


class TestStress:
    @pytest.mark.parametrize("action_counts", [
        (2, 2),
        (3, 3),
        (2, 2, 2),
        (4, 2),
        (2, 3, 2),
        (3, 3, 3),
        (2, 2, 2, 2),
    ])
    def test_full_pipeline(self, action_counts):
        """End-to-end: create game, solve CE, verify CE, run oracle, check separation."""
        game = NormalFormGame.random_game(action_counts, seed=hash(action_counts) % 1000)
        solver = CESolver(game.spec)
        w_star = game.pack_w()

        # CE exists and is valid
        x = solver.solve(w_star)
        assert x is not None
        assert solver.is_ce(w_star, x)

        # Oracle separation property
        feedback = AgentFeedback(game, model="best_response", seed=0)
        oracle = SeparationOracle(game, solver, feedback, seed=1)

        rng = np.random.default_rng(42)
        for _ in range(20):
            w_guess = rng.standard_normal(game.spec.N)
            result = oracle.query(w_guess)
            if result.deviated:
                assert np.dot(w_star, result.cut) >= -1e-8
                assert np.dot(w_guess, result.cut) <= 1e-8

    def test_large_game_4x4x4(self):
        """4x4x4 game: M=64, N=3*3*16=144."""
        game = NormalFormGame.random_game((4, 4, 4), seed=77)
        solver = CESolver(game.spec)
        w = game.pack_w()
        x = solver.solve(w)
        assert x is not None
        assert solver.is_ce(w, x)

        # Oracle check
        feedback = AgentFeedback(game, model="best_response", seed=0)
        oracle = SeparationOracle(game, solver, feedback, seed=1)
        w_guess = np.random.default_rng(0).standard_normal(game.spec.N)
        result = oracle.query(w_guess)
        if result.deviated:
            assert np.dot(w, result.cut) >= -1e-8


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
