"""Congestion game as a CompactGame.

A congestion game is defined by:
  - a set of `n_edges` resources (edges)
  - a per-agent set of routes (each route is a subset of edges)
  - per-edge affine cost functions c_e(l) = theta[e, 0] + theta[e, 1] * l

Player i's utility under a pure profile a is
    u_i(a) = -sum_{e in routes[i][a_i]} c_e(l_e(a))
where l_e(a) = #{ j : e in routes[j][a_j] } is the edge load.

Under a product distribution x = (x_1, ..., x_n), the polynomial expectation
property holds: E_{a ~ x | a_i = r_i}[u_i(a)] can be computed in
O(n * n_edges) time via the closed-form expectation of edge loads as sums
of independent Bernoullis.

For affine costs we only need E[l_e], which is the first moment of the
Poisson-binomial distribution of edge-e use by agents other than i.  If
this module is extended to polynomial costs of degree p, we must compute
E[l_e^k] for k = 0, ..., p; this requires either factorial-moment
recursions or the DFT method from `scipy.stats.poisson_binom`.

The compact representation here has size O(n_edges + sum_i m_i * max_route_length)
which is polynomial; the normal-form representation has size n * prod_i m_i
which is exponential in n.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .compact_game import CompactGame
from .game import NormalFormGame
from .types import GameSpec


@dataclass
class CongestionGame(CompactGame):
    """Finite congestion game with affine (linear) edge costs.

    Attributes
    ----------
    n_edges : int
        Total number of edges (resources).
    routes : list of list of frozenset[int]
        routes[i][k] is the set of edges used by agent i when playing her k-th route.
    theta : ndarray of shape (n_edges, 2)
        Affine cost coefficients: c_e(l) = theta[e, 0] + theta[e, 1] * l.
    """

    n_edges: int
    routes: list[list[frozenset[int]]]
    theta: NDArray[np.float64]
    spec: GameSpec = None  # populated in __post_init__

    def __post_init__(self):
        # Derive spec from route counts
        action_counts = tuple(len(r) for r in self.routes)
        self.spec = GameSpec.from_action_counts(action_counts)
        # Shape checks
        assert self.theta.shape == (self.n_edges, 2), (
            f"theta must have shape (n_edges, 2), got {self.theta.shape}"
        )
        # Route validity
        for i, agent_routes in enumerate(self.routes):
            for k, route in enumerate(agent_routes):
                for e in route:
                    if not (0 <= e < self.n_edges):
                        raise ValueError(
                            f"routes[{i}][{k}] refers to edge {e} "
                            f"outside [0, {self.n_edges})"
                        )

    # ---------------------------------------------------------------
    # Core utilities
    # ---------------------------------------------------------------

    def edge_load(self, profile: tuple[int, ...], e: int) -> int:
        """Number of agents using edge e under the pure profile."""
        return sum(
            1 for i, a_i in enumerate(profile) if e in self.routes[i][a_i]
        )

    def edge_cost(self, e: int, load: int) -> float:
        """Affine cost c_e(load) = theta[e, 0] + theta[e, 1] * load."""
        return float(self.theta[e, 0] + self.theta[e, 1] * load)

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        """u_i(a) = - sum_{e in route} c_e(l_e(a)).  Negative sum of delays."""
        i_route = self.routes[agent][profile[agent]]
        total = 0.0
        for e in i_route:
            total += self.edge_cost(e, self.edge_load(profile, e))
        return -total

    # ---------------------------------------------------------------
    # Polynomial expectation oracle
    # ---------------------------------------------------------------

    def edge_use_probability(
        self, agent: int, product_dist: list[NDArray[np.float64]], e: int
    ) -> float:
        """P[agent plays a route containing edge e] under its marginal distribution.

        Equals sum over routes r of agent that contain e, weighted by x_agent[r].
        """
        x_agent = product_dist[agent]
        p = 0.0
        for r, route in enumerate(self.routes[agent]):
            if e in route:
                p += float(x_agent[r])
        return p

    def polynomial_expectation(
        self,
        agent: int,
        a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        """E_{a_{-i} ~ x_{-i}}[u_i(a_i, a_{-i})] for affine-cost congestion.

        We condition on agent playing route `a_i`; then l_e = 1[e in a_i's route]
        + sum_{j != i} 1[e in j's route], where the latter is a sum of independent
        Bernoullis under the product distribution.  For affine costs we only need
        the first moment of l_e.

        Complexity: O(n * n_edges_in_route) which is O(n * n_edges) worst case.
        Does NOT enumerate joint profiles.
        """
        i_route = self.routes[agent][a_i]
        total_cost = 0.0
        for e in i_route:
            # Agent's contribution to load is 1 (since e in agent's route).
            expected_others_load = 0.0
            for j in range(self.spec.n):
                if j == agent:
                    continue
                expected_others_load += self.edge_use_probability(j, product_dist, e)
            expected_load = 1.0 + expected_others_load
            total_cost += self.edge_cost(e, expected_load)
            # Note: edge_cost is linear in load, so E[c_e(l)] = c_e(E[l])
            # for affine costs only.
        return -total_cost

    # ---------------------------------------------------------------
    # Cross-validation helper: convert to NormalFormGame
    # ---------------------------------------------------------------

    def as_normal_form(self) -> NormalFormGame:
        """Build the explicit normal-form representation.

        Only feasible for small games: allocates n arrays of shape
        action_counts which has size M = prod(m_i).  Use only for tests.
        """
        utilities: list[NDArray[np.float64]] = []
        for i in range(self.spec.n):
            U_i = np.zeros(self.spec.action_counts, dtype=np.float64)
            for profile in np.ndindex(*self.spec.action_counts):
                U_i[profile] = self.utility(i, profile)
            utilities.append(U_i)
        return NormalFormGame(self.spec.action_counts, utilities)


# -----------------------------------------------------------------------
# Small benchmark network constructors
# -----------------------------------------------------------------------


def pigou_game(n_agents: int, a: float = 0.0, b: float = 1.0,
               const_route_cost: float = 1.0) -> CongestionGame:
    """The Pigou network: 2 parallel edges between two nodes.

    Edge 0 has variable cost c_0(l) = a + b * l.
    Edge 1 has constant cost c_1(l) = const_route_cost.

    All agents have 2 routes (route 0 = use edge 0, route 1 = use edge 1).
    """
    n_edges = 2
    routes = [[frozenset({0}), frozenset({1})] for _ in range(n_agents)]
    theta = np.array([[a, b], [const_route_cost, 0.0]], dtype=np.float64)
    return CongestionGame(n_edges=n_edges, routes=routes, theta=theta)


def braess_game(n_agents: int, free_shortcut: bool = True) -> CongestionGame:
    """The Braess network: 4 nodes S, A, B, T and 5 edges.

        S -> A  (congestion-sensitive: c(l) = l)
        S -> B  (constant cost 1)
        A -> T  (constant cost 1)
        B -> T  (congestion-sensitive: c(l) = l)
        A -> B  (free shortcut) if free_shortcut else disabled

    We encode edges 0..4 respectively:
        e=0: S-A   c(l) = l               theta = (0, 1)
        e=1: S-B   c(l) = 1               theta = (1, 0)
        e=2: A-T   c(l) = 1               theta = (1, 0)
        e=3: B-T   c(l) = l               theta = (0, 1)
        e=4: A-B   c(l) = 0               theta = (0, 0)  (free shortcut)

    Routes for each agent (source S, sink T):
        r0: S -> A -> T      {0, 2}
        r1: S -> B -> T      {1, 3}
        r2: S -> A -> B -> T (uses shortcut) {0, 4, 3}
    """
    n_edges = 5
    theta = np.array([
        [0.0, 1.0],   # S-A
        [1.0, 0.0],   # S-B
        [1.0, 0.0],   # A-T
        [0.0, 1.0],   # B-T
        [0.0, 0.0],   # A-B shortcut
    ], dtype=np.float64)
    if free_shortcut:
        route_sets = [frozenset({0, 2}), frozenset({1, 3}), frozenset({0, 4, 3})]
    else:
        route_sets = [frozenset({0, 2}), frozenset({1, 3})]
    routes = [list(route_sets) for _ in range(n_agents)]
    return CongestionGame(n_edges=n_edges, routes=routes, theta=theta)


__all__ = ["CongestionGame", "pigou_game", "braess_game"]
