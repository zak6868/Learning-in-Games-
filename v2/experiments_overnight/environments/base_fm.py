"""Minimal FeatureMap interface used by the overnight moderator harness.

Every game family we plug into the cutting-plane moderator needs a way to
answer three questions:

1. Given θ ∈ R^d, construct the corresponding `CompactGame` instance.
2. Given the TRUE θ*, return it (for ground-truth / separation checks).
3. Given a pure profile `a` and an agent i, return the gradient
   ∇_θ u_i(a) in R^d.  We *assume utility is linear in θ* — this is what
   makes the cutting-plane halfspace a true separating hyperplane.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from recommend_unknown_games.compact_game import CompactGame
from recommend_unknown_games.types import GameSpec


class FeatureMap(ABC):
    """Generic linear-in-θ feature map.  See module docstring for contract."""

    @property
    @abstractmethod
    def d(self) -> int:
        """Dimension of the θ vector."""

    @property
    @abstractmethod
    def spec(self) -> GameSpec:
        """Immutable game spec (n, action_counts, M, …)."""

    @abstractmethod
    def theta_true(self) -> NDArray[np.float64]:
        """True θ* of the unknown game, as a length-d float64 vector."""

    @abstractmethod
    def game_at_theta(self, theta: NDArray[np.float64]) -> CompactGame:
        """Build the compact game when θ_t is the moderator's current guess."""

    @abstractmethod
    def utility_gradient(
        self, agent: int, profile: tuple[int, ...]
    ) -> NDArray[np.float64]:
        """∇_θ u_i(profile) in R^d, under the linearity assumption."""

    # -----------------------------------------------------------------
    # Generic cut builder: (moderator cut in θ-space)
    # -----------------------------------------------------------------

    def build_cut_theta(
        self,
        recommendation: tuple[int, ...],
        response: tuple[int, ...],
        profiles: list[tuple[int, ...]],
        alpha: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """q_θ = Σ_k α_k · 1[profile_k[i] = rec_i]
                 · [∇_θ u_i(resp_i, a_{-i}^k) − ∇_θ u_i(rec_i, a_{-i}^k)]
        summed over deviating agents i.

        Separation invariant (holds when utilities are linear in θ):
            <θ*, q_θ>  ≥  0  ≥  <θ_guess, q_θ>.
        """
        q = np.zeros(self.d, dtype=np.float64)
        n = self.spec.n
        for i in range(n):
            if recommendation[i] == response[i]:
                continue
            j = recommendation[i]
            j_star = response[i]
            for pk, ak in zip(profiles, alpha):
                if pk[i] != j:
                    continue
                p_rec = tuple(pk)
                p_resp = list(pk)
                p_resp[i] = j_star
                p_resp = tuple(p_resp)
                grad_rec = self.utility_gradient(i, p_rec)
                grad_resp = self.utility_gradient(i, p_resp)
                q += float(ak) * (grad_resp - grad_rec)
        return q


__all__ = ["FeatureMap"]
