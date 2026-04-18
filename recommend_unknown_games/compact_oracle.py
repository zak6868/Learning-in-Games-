"""Compact separation oracle for the paper's moderator loop.

Given a parameter guess θ ∈ R^d, this oracle:
  1. Builds an implicit CongestionGame with theta = θ.
  2. Runs the JLB ellipsoid to get a sparse CE mixture of that game.
  3. Samples a recommendation a(t) from the mixture.
  4. Queries the true agents (who operate with θ*) for their response a*(t).
  5. If any agent deviated, constructs a cut q_θ ∈ R^d directly from
     the sparse mixture's support — no dense w-vector, no d_i-length
     block ever materialized.

Cut construction (for each deviating agent i with recommendation j and
response j*):
    q_θ contribution from i =
        Σ_{k : profile_k[i] = j} α_k · Σ_e [1[e ∈ r_j] − 1[e ∈ r_{j*}]]
                                           · (θ_a-slot + θ_b-slot · (1 + L_e^{-i}(profile_k)))

The separation property <θ*, q_θ> ≥ 0 ≥ <θ_guess, q_θ> follows from
  <θ, q_θ> = <Φ θ, q> = Σ_i φ_i(j, j*, x)
(phi here = gain from deviating = u_i(j*, ·) − u_i(j, ·)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .congestion_game import CongestionGame
from .feature_map import CongestionFeatureMap
from .game import AgentFeedback
from .jlb_driver import run_jlb
from .jlb_ellipsoid import run_jlb_ellipsoid


@dataclass
class CompactOracleResult:
    """One oracle query."""

    recommendation: tuple[int, ...]                    # a(t) sampled from mixture
    response: tuple[int, ...]                          # a*(t) from the true agents
    deviated: bool                                     # any agent deviated?
    cut_theta: Optional[NDArray[np.float64]]           # q_θ ∈ R^d
    regret: float                                      # Σ_i φ_i
    mixture_size: int                                  # |support of x(t)|
    deviating_agents: list[int] = field(default_factory=list)


class CompactSeparationOracle:
    """Oracle that operates entirely in θ-space."""

    def __init__(
        self,
        game: CongestionGame,
        feature_map: CongestionFeatureMap,
        feedback: AgentFeedback,
        jlb_kwargs: Optional[dict] = None,
        seed: int = 0,
    ):
        self.game = game
        self.spec = game.spec
        self.feature_map = feature_map
        self.feedback = feedback
        self.d = feature_map.d
        self.rng = np.random.default_rng(seed)
        self.jlb_kwargs = jlb_kwargs or {
            "max_iterations": 2000,
            "initial_radius": 10.0,
            "reconstruct_every": 10,
        }

    # ---------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------

    def query(self, theta_guess: NDArray[np.float64]) -> CompactOracleResult:
        theta_guess = np.asarray(theta_guess, dtype=np.float64)
        assert theta_guess.shape == (self.d,), (
            f"theta_guess must have shape ({self.d},), got {theta_guess.shape}"
        )

        # 1. Build CongestionGame at theta_guess
        game_at_guess = self._game_at_theta(theta_guess)

        # 2. JLB ellipsoid → sparse CE mixture of game_at_guess
        profiles, alpha = self._solve_ce(game_at_guess)

        # 3. Sample recommendation
        k_sampled = int(self.rng.choice(len(profiles), p=alpha))
        recommendation = profiles[k_sampled]

        # 4. True agents respond (using the TRUE game via feedback)
        response = self._get_response(recommendation, profiles, alpha)

        # 5. If deviation, build cut directly in θ-space
        deviated = tuple(recommendation) != tuple(response)
        if not deviated:
            return CompactOracleResult(
                recommendation=tuple(recommendation),
                response=tuple(response),
                deviated=False,
                cut_theta=None,
                regret=0.0,
                mixture_size=len(profiles),
            )

        deviating = [
            i for i in range(self.spec.n)
            if recommendation[i] != response[i]
        ]
        cut_theta = self._build_cut_theta(
            recommendation, response, profiles, alpha
        )
        regret = self._compute_regret(
            recommendation, response, profiles, alpha
        )
        return CompactOracleResult(
            recommendation=tuple(recommendation),
            response=tuple(response),
            deviated=True,
            cut_theta=cut_theta,
            regret=regret,
            mixture_size=len(profiles),
            deviating_agents=deviating,
        )

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _game_at_theta(self, theta: NDArray[np.float64]) -> CongestionGame:
        theta_mat = theta.reshape(self.game.n_edges, 2).astype(np.float64, copy=True)
        return CongestionGame(
            n_edges=self.game.n_edges,
            routes=self.game.routes,
            theta=theta_mat,
        )

    def _solve_ce(
        self,
        game_at_guess: CongestionGame,
    ) -> tuple[list[tuple[int, ...]], NDArray[np.float64]]:
        """Run JLB on the guessed game; return (profiles, alpha) sparse mixture."""
        result = run_jlb_ellipsoid(game_at_guess, **self.jlb_kwargs)
        profiles = result.profiles
        alpha = result.alpha

        if not result.reconstructed:
            # Fall back to structured-y
            fallback = run_jlb(game_at_guess, max_iterations=500,
                                reconstruct_every=10, seed=0)
            if fallback.alpha is not None:
                profiles = fallback.profiles
                alpha = fallback.alpha

        if alpha is None:
            raise RuntimeError(
                "JLB inner solver failed to find a CE for the guessed game"
            )

        alpha = np.asarray(alpha, dtype=np.float64)
        alpha = np.maximum(alpha, 0.0)
        s = alpha.sum()
        if s <= 0:
            raise RuntimeError("JLB returned all-zero alpha")
        alpha = alpha / s
        return profiles, alpha

    def _get_response(
        self,
        recommendation: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> tuple[int, ...]:
        """Ask the true agents to respond.  Uses the existing AgentFeedback
        which expects a dense distribution; we materialize one from the
        sparse mixture (feasible for small games)."""
        dense_x = np.zeros(self.spec.M, dtype=np.float64)
        for profile, weight in zip(profiles, alpha):
            flat = int(np.ravel_multi_index(profile, self.spec.action_counts))
            dense_x[flat] += float(weight)
        return self.feedback.respond_all(tuple(recommendation), dense_x)

    def _build_cut_theta(
        self,
        recommendation: tuple[int, ...],
        response: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute q_θ = Φ^T q directly from sparse mixture support."""
        fm = self.feature_map
        q_theta = np.zeros(self.d, dtype=np.float64)

        for i in range(self.spec.n):
            if recommendation[i] == response[i]:
                continue
            j = recommendation[i]
            j_star = response[i]
            r_j = self.game.routes[i][j]
            r_jstar = self.game.routes[i][j_star]
            edges_plus = r_j - r_jstar    # coefficient +1
            edges_minus = r_jstar - r_j   # coefficient -1

            # Iterate only over mixture profiles with profile_k[i] == j.
            for pk, ak in zip(profiles, alpha):
                if pk[i] != j:
                    continue
                a_minus_i_tuple = tuple(pk[kk] for kk in range(self.spec.n) if kk != i)
                for e in edges_plus:
                    L = fm._load_from_others(i, a_minus_i_tuple, e)
                    q_theta[e * 2 + 0] += ak
                    q_theta[e * 2 + 1] += ak * (1.0 + L)
                for e in edges_minus:
                    L = fm._load_from_others(i, a_minus_i_tuple, e)
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
        """Regret = Σ_i φ_i(rec_i, resp_i, x) in our sign convention."""
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


__all__ = ["CompactSeparationOracle", "CompactOracleResult"]
