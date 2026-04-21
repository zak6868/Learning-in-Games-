"""PolymatrixFeatureMap — platform pricing game with linear demand.

Game:
    u_i(p) = b_i(p_i) + Σ_{j ∈ N(i)} M_{ij}(p_i, p_j)
with linear-demand specification
    b_i(p_i)       = (1 − τ_i)(p_i − c_i)(α_i − β_i p_i)
    M_{ij}(p_i, p_j) = (1 − τ_i)(p_i − c_i) · γ_{ij} · p_j.

Unknown θ:  per-seller (α_i, β_i).  d = 2 n.
Fixed per-instance: τ_i, c_i, γ_{ij}, price grid, graph.

This keeps the θ-dim small while exercising a non-congestion, non-
potential succinct game in the moderator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from recommend_unknown_games.compact_game import CompactGame
from recommend_unknown_games.types import GameSpec

from .base_fm import FeatureMap


# ---------------------------------------------------------------------
# The game (concrete CompactGame at a given θ = (α, β))
# ---------------------------------------------------------------------


@dataclass
class _PolymatrixPricingGame(CompactGame):
    """Internal CompactGame instance — built fresh at each θ_t.

    All fields other than `alpha`, `beta` are shared with the true
    instance; only (α, β) come from the current θ guess.
    """

    n: int
    k: int
    neighbours: list[tuple[int, ...]]
    price_grid: NDArray[np.float64]          # shape (k,)
    tau: NDArray[np.float64]                  # shape (n,)
    c: NDArray[np.float64]                    # shape (n,)
    gamma: dict[tuple[int, int], float]      # γ_{ij} for i, j∈N(i)
    alpha_params: NDArray[np.float64]         # shape (n,) - θ-derived
    beta_params: NDArray[np.float64]          # shape (n,) - θ-derived
    spec: GameSpec = None

    def __post_init__(self) -> None:
        self.spec = GameSpec.from_action_counts(tuple([self.k] * self.n))

    def _b_i(self, i: int, a_i: int) -> float:
        p = float(self.price_grid[a_i])
        return float(
            (1.0 - self.tau[i]) * (p - self.c[i])
            * (self.alpha_params[i] - self.beta_params[i] * p)
        )

    def _M_ij(self, i: int, j: int, a_i: int, a_j: int) -> float:
        p_i = float(self.price_grid[a_i])
        p_j = float(self.price_grid[a_j])
        return float(
            (1.0 - self.tau[i]) * (p_i - self.c[i])
            * self.gamma[(i, j)] * p_j
        )

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        u = self._b_i(agent, profile[agent])
        for j in self.neighbours[agent]:
            u += self._M_ij(agent, j, profile[agent], profile[j])
        return u

    def polynomial_expectation(
        self,
        agent: int,
        a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        u = self._b_i(agent, a_i)
        p_i = float(self.price_grid[a_i])
        coef_i = (1.0 - self.tau[agent]) * (p_i - self.c[agent])
        for j in self.neighbours[agent]:
            x_j = product_dist[j]
            g = self.gamma[(agent, j)]
            # E[γ · (p_i − c_i) · p_j] = coef_i · γ · E_{a_j}[p_j]
            u += coef_i * g * float((x_j * self.price_grid).sum())
        return u


# ---------------------------------------------------------------------
# FeatureMap
# ---------------------------------------------------------------------


class PolymatrixPricingFeatureMap(FeatureMap):
    """θ = [α_1, β_1, α_2, β_2, …, α_n, β_n] of length 2n."""

    def __init__(
        self,
        n: int,
        k: int = 5,
        degree: int = 3,
        price_low: float = 5.0,
        price_high: float = 15.0,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._n = n
        self._k = k
        self._d = 2 * n
        self._price_grid = np.linspace(
            price_low, price_high, k, dtype=np.float64
        )
        self._neighbours = _random_graph(n, degree, seed=seed)
        self._tau = rng.uniform(0.05, 0.15, size=n)
        self._c = rng.uniform(3.0, 5.0, size=n)
        self._gamma = {
            (i, j): float(rng.uniform(0.0, 2.0))
            for i in range(n) for j in self._neighbours[i]
        }
        # True θ
        self._alpha_true = rng.uniform(80.0, 120.0, size=n)
        self._beta_true = rng.uniform(4.0, 8.0, size=n)

        self._spec = GameSpec.from_action_counts(tuple([k] * n))

    # ----- FeatureMap abstract methods -----

    @property
    def d(self) -> int:
        return self._d

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def theta_true(self) -> NDArray[np.float64]:
        theta = np.empty(self._d, dtype=np.float64)
        theta[0::2] = self._alpha_true
        theta[1::2] = self._beta_true
        return theta

    def _split_theta(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        alpha = np.asarray(theta[0::2], dtype=np.float64)
        beta = np.asarray(theta[1::2], dtype=np.float64)
        return alpha, beta

    def game_at_theta(
        self, theta: NDArray[np.float64]
    ) -> CompactGame:
        alpha, beta = self._split_theta(theta)
        return _PolymatrixPricingGame(
            n=self._n, k=self._k, neighbours=self._neighbours,
            price_grid=self._price_grid, tau=self._tau, c=self._c,
            gamma=self._gamma,
            alpha_params=alpha, beta_params=beta,
        )

    def utility_gradient(
        self, agent: int, profile: tuple[int, ...]
    ) -> NDArray[np.float64]:
        """∂u_i / ∂(α_j, β_j) is nonzero only for j = i:
               ∂u_i/∂α_i = (1 − τ_i)(p_i − c_i)
               ∂u_i/∂β_i = −(1 − τ_i)(p_i − c_i) p_i
        All other entries are zero.
        """
        g = np.zeros(self._d, dtype=np.float64)
        p_i = float(self._price_grid[profile[agent]])
        coef = (1.0 - self._tau[agent]) * (p_i - self._c[agent])
        g[2 * agent + 0] = coef
        g[2 * agent + 1] = -coef * p_i
        return g

    # ---- convenience for the harness ----

    def true_game(self) -> CompactGame:
        return self.game_at_theta(self.theta_true())


# ---------------------------------------------------------------------
# Random graph helper (deterministic, shared with pricing factories)
# ---------------------------------------------------------------------


def _random_graph(
    n: int, degree: int, seed: int = 0
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    nbrs: list[set] = [set() for _ in range(n)]
    for i in range(n):
        available = [j for j in range(n) if j != i and j not in nbrs[i]]
        if len(available) <= degree:
            for j in available:
                nbrs[i].add(j); nbrs[j].add(i)
            continue
        chosen = rng.choice(available, size=degree, replace=False)
        for j in chosen:
            nbrs[i].add(int(j)); nbrs[int(j)].add(i)
    return [tuple(sorted(s)) for s in nbrs]


__all__ = ["PolymatrixPricingFeatureMap", "_PolymatrixPricingGame"]
