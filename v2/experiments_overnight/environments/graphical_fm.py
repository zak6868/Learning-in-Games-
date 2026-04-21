"""GraphicalFeatureMap variants — coloring and best-shot public goods.

Two concrete FeatureMaps are provided, both keeping θ-dim low:

1. `GraphColoringFeatureMap`
       u_i(a) = (# different-color neighbours) − θ_collision · (# same-color neighbours)
       θ ∈ R — single collision-cost scalar.

2. `BestShotPGFeatureMap`
       u_i(a) = θ_benefit · 1[∃ j ∈ N(i) ∪ {i} with a_j = 1]
                − θ_cost · 1[a_i = 1]
       θ = (benefit, cost) ∈ R² — two global scalars.

Both are linear in θ, so `utility_gradient` is trivial.  We build the
CompactGame freshly at each θ_t by plugging the current scalars into
the local utility tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from recommend_unknown_games.compact_game import CompactGame
from recommend_unknown_games.types import GameSpec

from .base_fm import FeatureMap


# ---------------------------------------------------------------------
# Internal CompactGame built from the current θ
# ---------------------------------------------------------------------


@dataclass
class _ColoringGame(CompactGame):
    n: int
    m: int
    neighbours: list[tuple[int, ...]]
    collision_cost: float
    spec: GameSpec = None

    def __post_init__(self) -> None:
        self.spec = GameSpec.from_action_counts(tuple([self.m] * self.n))

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        a_i = profile[agent]
        diff = 0
        same = 0
        for j in self.neighbours[agent]:
            if profile[j] == a_i:
                same += 1
            else:
                diff += 1
        return float(diff - self.collision_cost * same)

    def polynomial_expectation(
        self, agent: int, a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        # E[diff − κ·same] = Σ_j [P(a_j ≠ a_i) − κ · P(a_j = a_i)].
        total = 0.0
        for j in self.neighbours[agent]:
            p_same = float(product_dist[j][a_i])
            p_diff = 1.0 - p_same
            total += p_diff - self.collision_cost * p_same
        return total


@dataclass
class _BestShotPGGame(CompactGame):
    n: int
    neighbours: list[tuple[int, ...]]
    benefit: float
    cost: float
    spec: GameSpec = None

    def __post_init__(self) -> None:
        self.spec = GameSpec.from_action_counts(tuple([2] * self.n))

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        any_contrib = profile[agent] == 1 or any(
            profile[j] == 1 for j in self.neighbours[agent]
        )
        u = self.benefit if any_contrib else 0.0
        if profile[agent] == 1:
            u -= self.cost
        return float(u)

    def polynomial_expectation(
        self, agent: int, a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        # P[no neighbour contributes] = Π_{j ∈ N(i)} x_j[0]
        prob_none_nbr = 1.0
        for j in self.neighbours[agent]:
            prob_none_nbr *= float(product_dist[j][0])
        if a_i == 1:
            # I contribute → definitely "any_contrib", gain benefit, pay cost.
            return float(self.benefit - self.cost)
        # a_i = 0:
        # any_contrib iff at least one neighbour contributes.
        p_any = 1.0 - prob_none_nbr
        return float(p_any * self.benefit)


# ---------------------------------------------------------------------
# Graph coloring FeatureMap
# ---------------------------------------------------------------------


class GraphColoringFeatureMap(FeatureMap):
    """θ ∈ R — collision cost scalar.  Only θ[0] is meaningful; d=1."""

    def __init__(
        self,
        neighbours: list[tuple[int, ...]],
        n_colors: int = 3,
        true_collision_cost: float = 1.0,
    ) -> None:
        self._n = len(neighbours)
        self._m = n_colors
        self._nbrs = [tuple(sorted(nb)) for nb in neighbours]
        self._true_cost = float(true_collision_cost)
        self._spec = GameSpec.from_action_counts(tuple([n_colors] * self._n))

    @property
    def d(self) -> int:
        return 1

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def theta_true(self) -> NDArray[np.float64]:
        return np.array([self._true_cost], dtype=np.float64)

    def game_at_theta(self, theta: NDArray[np.float64]) -> CompactGame:
        return _ColoringGame(
            n=self._n, m=self._m, neighbours=self._nbrs,
            collision_cost=float(theta[0]),
        )

    def utility_gradient(
        self, agent: int, profile: tuple[int, ...]
    ) -> NDArray[np.float64]:
        # ∂u_i/∂κ = −(# same-color neighbours)
        a_i = profile[agent]
        same = sum(1 for j in self._nbrs[agent] if profile[j] == a_i)
        return np.array([-float(same)], dtype=np.float64)

    def true_game(self) -> CompactGame:
        return self.game_at_theta(self.theta_true())


# ---------------------------------------------------------------------
# Best-shot public goods FeatureMap
# ---------------------------------------------------------------------


class BestShotPGFeatureMap(FeatureMap):
    """θ = (benefit, cost) ∈ R², d = 2."""

    def __init__(
        self,
        neighbours: list[tuple[int, ...]],
        true_benefit: float = 1.0,
        true_cost: float = 0.4,
    ) -> None:
        self._n = len(neighbours)
        self._nbrs = [tuple(sorted(nb)) for nb in neighbours]
        self._true_benefit = float(true_benefit)
        self._true_cost = float(true_cost)
        self._spec = GameSpec.from_action_counts(tuple([2] * self._n))

    @property
    def d(self) -> int:
        return 2

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def theta_true(self) -> NDArray[np.float64]:
        return np.array([self._true_benefit, self._true_cost], dtype=np.float64)

    def game_at_theta(self, theta: NDArray[np.float64]) -> CompactGame:
        return _BestShotPGGame(
            n=self._n, neighbours=self._nbrs,
            benefit=float(theta[0]), cost=float(theta[1]),
        )

    def utility_gradient(
        self, agent: int, profile: tuple[int, ...]
    ) -> NDArray[np.float64]:
        # ∂u_i/∂benefit = 1[any_contrib];  ∂u_i/∂cost = −1[a_i = 1]
        any_contrib = profile[agent] == 1 or any(
            profile[j] == 1 for j in self._nbrs[agent]
        )
        g = np.zeros(2, dtype=np.float64)
        g[0] = 1.0 if any_contrib else 0.0
        g[1] = -1.0 if profile[agent] == 1 else 0.0
        return g

    def true_game(self) -> CompactGame:
        return self.game_at_theta(self.theta_true())


__all__ = [
    "GraphColoringFeatureMap",
    "BestShotPGFeatureMap",
    "_ColoringGame",
    "_BestShotPGGame",
]
