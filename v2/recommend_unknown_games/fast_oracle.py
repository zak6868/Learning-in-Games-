"""Fast compact separation oracle for *congestion games*.

Replaces the JLB inner solve with Rosenthal's potential-guided
best-response dynamics. Since every pure Nash of a congestion game is
a (degenerate) CE, using the Nash as the recommendation distribution
is valid and much cheaper than running JLB.

Typical per-query cost:
  - Nash via BR dynamics : O(n_iters · n · |route|) ≈ ms at n ≈ 10 000
  - Feedback response    : O(n · m · |route|)
  - Cut construction     : O(n · |route|)

Drop-in replacement for CompactSeparationOracle on *congestion games
only*. For general succinct games where no potential function exists
(graphical, anonymous, AGG), JLB is still required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .compact_feedback import CompactAgentFeedback
from .compact_oracle import CompactOracleResult
from .congestion_game import CongestionGame
from .feature_map import CongestionFeatureMap
from .game import AgentFeedback


class FastCompactOracle:
    """Oracle that uses Nash (via BR dynamics) instead of JLB."""

    def __init__(
        self,
        game: CongestionGame,
        feature_map: CongestionFeatureMap,
        feedback,
        nash_max_iters: int = 50_000,
        n_nashes: int = 100,
        seed: int = 0,
    ) -> None:
        self.game = game
        self.spec = game.spec
        self.feature_map = feature_map
        self.feedback = feedback
        self.d = feature_map.d
        self.rng = np.random.default_rng(seed)
        self.nash_max_iters = nash_max_iters
        self.n_nashes = n_nashes
        self._cache_theta: Optional[NDArray[np.float64]] = None
        self._cache_profiles: Optional[list[tuple[int, ...]]] = None
        self._cache_alpha: Optional[NDArray[np.float64]] = None

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def query(self, theta_guess: NDArray[np.float64]) -> CompactOracleResult:
        theta_guess = np.asarray(theta_guess, dtype=np.float64)
        assert theta_guess.shape == (self.d,), (
            f"theta_guess must have shape ({self.d},)"
        )

        t_total = perf_counter()

        # 1. Build game at θ_guess
        game_at_guess = self._game_at_theta(theta_guess)

        # 2. CE mixture = uniform over a pool of distinct Nashes
        #    (each Nash is a CE; convex combinations of CEs are CEs).
        cached = (
            self._cache_theta is not None
            and self._cache_profiles is not None
            and np.array_equal(self._cache_theta, theta_guess)
        )
        t0 = perf_counter()
        if cached:
            profiles = self._cache_profiles
            alpha = self._cache_alpha
        else:
            seen: set[tuple[int, ...]] = set()
            profiles = []
            for _ in range(self.n_nashes):
                s = int(self.rng.integers(0, 10_000_000))
                nash, _ = game_at_guess.find_pure_nash(
                    max_iters=self.nash_max_iters, seed=s,
                )
                if nash not in seen:
                    seen.add(nash)
                    profiles.append(nash)
            if not profiles:
                # Fallback: at least one Nash
                nash, _ = game_at_guess.find_pure_nash(
                    max_iters=self.nash_max_iters,
                    seed=int(self.rng.integers(0, 10_000_000)),
                )
                profiles = [nash]
            alpha = np.full(len(profiles), 1.0 / len(profiles))
            self._cache_theta = theta_guess.copy()
            self._cache_profiles = profiles
            self._cache_alpha = alpha
        t_ce = perf_counter() - t0

        # 3. Sample recommendation from the Nash mixture
        k_sampled = int(self.rng.choice(len(profiles), p=alpha))
        recommendation = profiles[k_sampled]

        # 4. True-agent response
        t0 = perf_counter()
        response = self._get_response(recommendation, profiles, alpha)
        t_feedback = perf_counter() - t0

        deviated = tuple(recommendation) != tuple(response)
        if not deviated:
            timings = {
                "ce_solve_s": t_ce,
                "feedback_s": t_feedback,
                "cut_build_s": 0.0,
                "total_s": perf_counter() - t_total,
            }
            return CompactOracleResult(
                recommendation=tuple(recommendation),
                response=tuple(response),
                deviated=False,
                cut_theta=None,
                regret=0.0,
                mixture_size=len(profiles),
                timings=timings,
                ce_cached=cached,
            )

        # 5. Build cut directly in θ-space
        deviating = [i for i in range(self.spec.n)
                     if recommendation[i] != response[i]]
        t0 = perf_counter()
        cut_theta = self._build_cut_theta(
            recommendation, response, profiles, alpha
        )
        regret = self._compute_regret(
            recommendation, response, profiles, alpha
        )
        t_cut = perf_counter() - t0

        timings = {
            "ce_solve_s": t_ce,
            "feedback_s": t_feedback,
            "cut_build_s": t_cut,
            "total_s": perf_counter() - t_total,
        }
        return CompactOracleResult(
            recommendation=tuple(recommendation),
            response=tuple(response),
            deviated=True,
            cut_theta=cut_theta,
            regret=regret,
            mixture_size=1,
            deviating_agents=deviating,
            timings=timings,
            ce_cached=cached,
        )

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _game_at_theta(
        self, theta: NDArray[np.float64]
    ) -> CongestionGame:
        theta_mat = theta.reshape(self.game.n_edges, 2).astype(
            np.float64, copy=True
        )
        return CongestionGame(
            n_edges=self.game.n_edges,
            routes=self.game.routes,
            theta=theta_mat,
        )

    def _get_response(
        self,
        recommendation: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> tuple[int, ...]:
        if isinstance(self.feedback, CompactAgentFeedback):
            return self.feedback.respond_all(
                tuple(recommendation), profiles, alpha
            )
        # Fallback: dense (only for small games)
        dense = np.zeros(self.spec.M, dtype=np.float64)
        for profile, w in zip(profiles, alpha):
            flat = int(np.ravel_multi_index(profile, self.spec.action_counts))
            dense[flat] += float(w)
        return self.feedback.respond_all(tuple(recommendation), dense)

    def _build_cut_theta(
        self,
        recommendation: tuple[int, ...],
        response: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build the θ-space cut from a sparse CE mixture.

        Identical formula to CompactSeparationOracle._build_cut_theta,
        but here the mixture is a Nash pool rather than a JLB-derived
        one.  Separation property <θ*, q> ≥ 0 ≥ <θ_guess, q> holds
        because each support profile is a Nash of the guessed game.
        """
        fm = self.feature_map
        q_theta = np.zeros(self.d, dtype=np.float64)

        for i in range(self.spec.n):
            if recommendation[i] == response[i]:
                continue
            j = recommendation[i]
            j_star = response[i]
            r_j = self.game.routes[i][j]
            r_jstar = self.game.routes[i][j_star]
            edges_plus = r_j - r_jstar
            edges_minus = r_jstar - r_j
            for pk, ak in zip(profiles, alpha):
                if pk[i] != j:
                    continue
                a_minus_i = tuple(
                    pk[kk] for kk in range(self.spec.n) if kk != i
                )
                for e in edges_plus:
                    L = fm._load_from_others(i, a_minus_i, e)
                    q_theta[e * 2 + 0] += ak
                    q_theta[e * 2 + 1] += ak * (1.0 + L)
                for e in edges_minus:
                    L = fm._load_from_others(i, a_minus_i, e)
                    q_theta[e * 2 + 0] -= ak
                    q_theta[e * 2 + 1] -= ak * (1.0 + L)
        return q_theta

    def _compute_regret(
        self,
        recommendation: tuple[int, ...],
        response: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> float:
        total = 0.0
        for i in range(self.spec.n):
            if recommendation[i] == response[i]:
                continue
            j = recommendation[i]
            j_star = response[i]
            for pk, ak in zip(profiles, alpha):
                if pk[i] != j:
                    continue
                p_dev = list(pk)
                p_dev[i] = j_star
                u_here = self.game.utility(i, tuple(pk))
                u_dev = self.game.utility(i, tuple(p_dev))
                total += float(ak) * (u_dev - u_here)
        return total


__all__ = ["FastCompactOracle"]
