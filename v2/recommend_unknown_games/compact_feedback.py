"""Compact agent-feedback model for CongestionGame.

Mirrors `AgentFeedback` (BR and QR) but operates on:
  - a `CongestionGame` (polynomial-time utility evaluator)
  - a sparse mixture {(profile, weight)} of a CE

without ever materializing the normal-form utility tensor.  This is the
last piece needed to scale the compact moderator to games where
`game.as_normal_form()` is infeasible (e.g., Pigou n=50, M=2^50).

BR (Definition 2 of the paper):
    argmax_{a'_i} E_{a_{-i} ~ x(·|a_i)}[u_i(a'_i, a_{-i})]
    With tie-break toward a_i.

QR (Definition 3):
    Deviate to a'_i with probability proportional to
    exp(β · φ_i(a_i, a'_i, x))  when φ >= 0, else 0.

Both reduce to sums over the mixture's support.  For a mixture of size
K and an agent with m_i actions, each BR/QR call costs O(K · m_i · |E|).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .congestion_game import CongestionGame


class CompactAgentFeedback:
    """BR/QR feedback model that works with sparse CE mixtures."""

    def __init__(
        self,
        game: CongestionGame,
        model: str = "best_response",
        beta: float = 1.0,
        gamma: float = 0.0,
        seed: int = 42,
    ):
        assert model in ("best_response", "quantal_response")
        self.game = game
        self.spec = game.spec
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # Core: expected utility under a conditional distribution
    # ---------------------------------------------------------------

    def _compute_expected_utility(
        self,
        agent: int,
        target_action: int,
        profiles: Sequence[tuple[int, ...]],
        alpha: NDArray[np.float64],
        recommendation_action: int | None = None,
    ) -> float:
        """Compute E_{a_{-i} ~ x(·|a_i)}[u_i(target_action, a_{-i})].

        If recommendation_action is None we condition on nothing (treat
        the mixture as the full joint).  If given, we condition on
        profile_k[agent] == recommendation_action.
        """
        total_weight = 0.0
        total_util = 0.0
        for pk, ak in zip(profiles, alpha):
            if recommendation_action is not None and pk[agent] != recommendation_action:
                continue
            # Replace agent's action with target_action
            profile_candidate = list(pk)
            profile_candidate[agent] = target_action
            u = self.game.utility(agent, tuple(profile_candidate))
            total_util += float(ak) * u
            total_weight += float(ak)
        if total_weight <= 0:
            return 0.0
        return total_util / total_weight

    # ---------------------------------------------------------------
    # Response (BR or QR)
    # ---------------------------------------------------------------

    def respond(
        self,
        agent: int,
        a_i: int,
        profiles: Sequence[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> int:
        """Return the action agent plays given recommendation a_i under x."""
        m_i = self.spec.action_counts[agent]
        # Expected utilities under the conditional given a_i
        expected = np.array([
            self._compute_expected_utility(agent, a_prime, profiles, alpha,
                                            recommendation_action=a_i)
            for a_prime in range(m_i)
        ])

        if self.model == "best_response":
            # γ-BR: follow if within γ of the best
            best_val = float(np.max(expected))
            if expected[a_i] >= best_val - self.gamma - 1e-12:
                return a_i
            return int(np.argmax(expected))

        # Quantal response: paper eq. (3), only positive-gain deviations
        gains = expected - expected[a_i]
        mask = gains >= 0.0
        log_probs = np.full_like(gains, -np.inf)
        log_probs[mask] = self.beta * gains[mask]
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()
        return int(self.rng.choice(len(probs), p=probs))

    def respond_all(
        self,
        recommendation: tuple[int, ...],
        profiles: Sequence[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> tuple[int, ...]:
        # Fast vectorized path is congestion-specific (uses n_edges,
        # routes, theta). Fall back to the per-agent respond loop for
        # any other CompactGame.
        if (
            hasattr(self.game, "n_edges")
            and hasattr(self.game, "routes")
            and hasattr(self.game, "theta")
        ):
            return self._respond_all_fast(
                recommendation, list(profiles),
                np.asarray(alpha, dtype=np.float64),
            )
        return tuple(
            self.respond(i, recommendation[i], profiles, alpha)
            for i in range(self.spec.n)
        )

    # ---------------------------------------------------------------
    # Fast batch path
    # ---------------------------------------------------------------

    def _respond_all_fast(
        self,
        recommendation: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> tuple[int, ...]:
        n = self.spec.n
        K = len(profiles)
        m_max = max(self.spec.action_counts)
        E = self.game.n_edges

        # route_masks[i, a, e] ∈ {0, 1}
        route_masks = np.zeros((n, m_max, E), dtype=np.float64)
        for i in range(n):
            for k, route in enumerate(self.game.routes[i]):
                for e in route:
                    route_masks[i, k, e] = 1.0

        # profiles_arr[k, i] = which action agent i plays in profile k
        profiles_arr = np.asarray(profiles, dtype=np.int64)  # (K, n)

        a_coef = self.game.theta[:, 0].astype(np.float64)
        b_coef = self.game.theta[:, 1].astype(np.float64)

        # Per-profile per-edge load and per-agent-own-contribution.
        #   L[k, e] = Σ_i route_masks[i, profile_k[i], e]
        #   Own[k, i, e] = route_masks[i, profile_k[i], e] ∈ {0,1}
        # Own built via np.take_along_axis.
        agent_idx = np.arange(n)
        # route_masks has shape (n, m_max, E); we want Own[k, i, e] =
        # route_masks[i, profiles_arr[k, i], e] → shape (K, n, E).
        Own = route_masks[agent_idx[None, :], profiles_arr, :]        # (K, n, E)
        L = Own.sum(axis=1)                                           # (K, E)

        # Edge-cost per (k, e) under profile k (full load, all agents in):
        #   cost_full[k, e] = a_e + b_e * L[k, e]
        cost_full = a_coef[None, :] + b_coef[None, :] * L             # (K, E)

        # If agent i plays a' (removing their current contribution),
        # the hypothetical new load on edge e is
        #   L[k, e] - Own[k, i, e] + route_masks[i, a', e].
        # Per (k, i, a'): resulting cost edge-wise is
        #   base[k, i, e] = cost_full[k, e] - b_e * Own[k, i, e]
        # and
        #   u_i(a', p_k[-i]) = -Σ_e route_masks[i, a', e] *
        #                       (a_e + b_e * (base[k, i, e]/b_e_or_const +
        #                                     route_masks[i, a', e])).
        # Simpler: build per-(k, i) "a_{-i} edge load" = L[k] - Own[k, i] and
        # cost of playing a' = Σ_e route_masks[i, a', e] * (a_e + b_e * (L_-i + 1)).
        # Wait: if agent i plays a', they use r_{i,a'}. For each edge in r_{i,a'},
        # load is L_-i[e] + 1. So:
        #   u_i(a', p_k[-i]) = -Σ_e 1[e∈r_{i,a'}] * (a_e + b_e * (L_-i[k,i,e] + 1))
        # where L_-i[k,i,e] = L[k, e] - Own[k, i, e].

        L_minus_i = L[:, None, :] - Own                               # (K, n, E)
        # effective cost if any edge is "used" by i at load L_-i+1:
        cost_if_used = a_coef[None, None, :] + b_coef[None, None, :] * (L_minus_i + 1.0)
        # shape (K, n, E)

        # u_i(a', p_k[-i]) = -Σ_e route_masks[i, a', e] * cost_if_used[k, i, e]
        # Using einsum with route_masks shape (n, m_max, E) and cost shape
        # (K, n, E): u[k, i, a'] = -sum_e route_masks[i, a', e] * cost_if_used[k, i, e]
        u = -np.einsum("iae,kie->kia", route_masks, cost_if_used)     # (K, n, m_max)

        # Expected utility per (i, a') conditional on profile[i] =
        # recommendation[i]: average over k where profile_k[i] = rec_i.
        rec_arr = np.asarray(recommendation, dtype=np.int64)           # (n,)
        # mask[k, i] = 1 if profiles_arr[k, i] == rec_arr[i]
        mask = (profiles_arr == rec_arr[None, :]).astype(np.float64)   # (K, n)
        weights = mask * alpha[:, None]                                # (K, n)
        denom = weights.sum(axis=0)                                    # (n,), per-agent total
        # Weighted sum of u over K, per (i, a')
        num = np.einsum("ki,kia->ia", weights, u)                      # (n, m_max)
        expected = num / np.maximum(denom[:, None], 1e-15)              # (n, m_max)

        # Mask out actions with zero conditional weight (action not in
        # support) — set them to -inf so argmax ignores them.
        # zero-denom rows have no support for that agent; we fall back to
        # the recommendation.
        # Compute BR per agent.
        response = np.zeros(n, dtype=np.int64)
        if self.model == "best_response":
            for i in range(n):
                m_i = self.spec.action_counts[i]
                exp_i = expected[i, :m_i]
                rec_i = int(rec_arr[i])
                best_val = float(np.max(exp_i))
                if exp_i[rec_i] >= best_val - self.gamma - 1e-12:
                    response[i] = rec_i
                else:
                    response[i] = int(np.argmax(exp_i))
        else:  # quantal_response
            for i in range(n):
                m_i = self.spec.action_counts[i]
                exp_i = expected[i, :m_i]
                rec_i = int(rec_arr[i])
                gains = exp_i - exp_i[rec_i]
                mask_pos = gains >= 0.0
                log_probs = np.full_like(gains, -np.inf)
                log_probs[mask_pos] = self.beta * gains[mask_pos]
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                probs /= probs.sum()
                response[i] = int(self.rng.choice(len(probs), p=probs))

        return tuple(int(x) for x in response)


__all__ = ["CompactAgentFeedback"]
