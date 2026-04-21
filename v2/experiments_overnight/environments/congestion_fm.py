"""CongestionFeatureMap adapter: bridges the existing CongestionGame and
the overnight harness's base FeatureMap (with `utility_gradient`).

θ = (a_0, b_0, a_1, b_1, …, a_{|E|-1}, b_{|E|-1}) ∈ R^{2|E|}.

∂u_i/∂a_e = −1[e ∈ r_{i, a_i}]
∂u_i/∂b_e = −1[e ∈ r_{i, a_i}] · ℓ_e(profile)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from recommend_unknown_games.compact_game import CompactGame
from recommend_unknown_games.congestion_game import CongestionGame
from recommend_unknown_games.types import GameSpec

from .base_fm import FeatureMap


class CongestionFM(FeatureMap):
    """Wraps a CongestionGame template (routes + true θ) into a FeatureMap
    compatible with the overnight harness.
    """

    def __init__(self, template: CongestionGame) -> None:
        self._template = template
        self._spec = template.spec
        self._d = 2 * template.n_edges
        self._theta_true = template.theta.ravel().astype(np.float64, copy=True)

    @property
    def d(self) -> int:
        return self._d

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def theta_true(self) -> NDArray[np.float64]:
        return self._theta_true.copy()

    def game_at_theta(self, theta: NDArray[np.float64]) -> CompactGame:
        theta_mat = theta.reshape(self._template.n_edges, 2).astype(
            np.float64, copy=True
        )
        return CongestionGame(
            n_edges=self._template.n_edges,
            routes=self._template.routes,
            theta=theta_mat,
        )

    def utility_gradient(
        self, agent: int, profile: tuple[int, ...]
    ) -> NDArray[np.float64]:
        g = np.zeros(self._d, dtype=np.float64)
        r_i = self._template.routes[agent][profile[agent]]
        if not r_i:
            return g
        # Edge loads under `profile`.
        loads = np.zeros(self._template.n_edges, dtype=np.int64)
        for j, a_j in enumerate(profile):
            for e in self._template.routes[j][a_j]:
                loads[e] += 1
        for e in r_i:
            g[e * 2 + 0] = -1.0
            g[e * 2 + 1] = -float(loads[e])
        return g

    def true_game(self) -> CompactGame:
        return self._template


__all__ = ["CongestionFM"]
