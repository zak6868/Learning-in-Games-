"""Feature maps Φ: R^d → R^N linking a compact parameter space to the
w-vector space used by the paper's moderator.

The abstract `FeatureMap` exposes:
  - `.d`, `.N`, `.spec`
  - `theta_true()`        : true θ of the underlying game
  - `w_from_theta(θ)`     : apply Φ; produces a full w-vector (O(N) time,
                            for test/validation only)
  - `project_cut_dense(q)`: compute Φ^T q from a full (dense) q ∈ R^N
  - `project_cut_sparse(blocks)`: same but from a sparse representation
                            {(agent_i, action_k): block} — one d_i-long
                            block per (deviating) agent / alternative action

Concrete subclass:
  - `CongestionFeatureMap` — affine-cost congestion games. The θ layout
    is [a_0, b_0, a_1, b_1, …, a_{K-1}, b_{K-1}]  (length 2|E|).

The fundamental identities we guarantee:
  Φ(θ_true) == game.as_normal_form().pack_w()          (tested)
  <θ, Φ^T q> == <Φ θ, q>  for all θ, q                 (tested)

The implementation never materializes Φ as a matrix — everything is
computed by iterating over the combinatorial structure of the game.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product as iter_product
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .congestion_game import CongestionGame
from .types import GameSpec


class FeatureMap(ABC):
    """Abstract linear map Φ: R^d → R^N plus its transpose Φ^T."""

    @property
    @abstractmethod
    def d(self) -> int:
        ...

    @property
    @abstractmethod
    def N(self) -> int:
        ...

    @property
    @abstractmethod
    def spec(self) -> GameSpec:
        ...

    @abstractmethod
    def theta_true(self) -> NDArray[np.float64]:
        ...

    @abstractmethod
    def w_from_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    @abstractmethod
    def project_cut_dense(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    @abstractmethod
    def project_cut_sparse(
        self,
        cut_blocks: Mapping[tuple[int, int], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        ...


class CongestionFeatureMap(FeatureMap):
    """Affine-cost congestion-game feature map.

    θ = (a_0, b_0, a_1, b_1, …, a_{K-1}, b_{K-1}) of length d = 2|E|.
    c_e(ℓ) = a_e + b_e * ℓ;  u_i(a) = −Σ_{e ∈ r_{a_i}^{(i)}} c_e(ℓ_e(a)).

    w_i(0, j)[a_{-i}] = u_i(j, a_{-i}) − u_i(0, a_{-i})
                     = Σ_e [1[e ∈ r_0] − 1[e ∈ r_j]]
                          · (a_e + b_e (1 + L_e^{−i}(a_{-i})))
    where L_e^{−i}(a_{-i}) = Σ_{j ≠ i} 1[e ∈ r_{a_j}^{(j)}].
    """

    def __init__(self, game: CongestionGame):
        self._game = game
        self._spec = game.spec
        self._N = game.spec.N
        self._d = 2 * game.n_edges

        # Precompute the agent-block offsets in the w-vector (same
        # convention as utils.IndexCache).  Agent i's block starts at
        # sum_{k < i} (m_k - 1) * d_k.  Use Python ints (arbitrary
        # precision) because d_k can exceed int64 for large games — even
        # though the compact path does not use these offsets, the
        # constructor shouldn't crash when the game is large.
        offsets = [0]
        for k in range(1, self._spec.n):
            m_k = self._spec.action_counts[k - 1]
            d_k = self._spec.d[k - 1]
            offsets.append(offsets[-1] + (m_k - 1) * d_k)
        self._agent_offsets = offsets

    # ----- abstract method implementations -----

    @property
    def d(self) -> int:
        return self._d

    @property
    def N(self) -> int:
        return self._N

    @property
    def spec(self) -> GameSpec:
        return self._spec

    @property
    def game(self) -> CongestionGame:
        return self._game

    def theta_true(self) -> NDArray[np.float64]:
        """Flatten the game's theta matrix into a d-vector."""
        return self._game.theta.ravel().astype(np.float64, copy=True)

    # ----- internal load computation -----

    def _load_from_others(self, i: int, a_minus_i: tuple[int, ...], e: int) -> int:
        """L_e^{-i}(a_{-i}) — number of agents j ≠ i using edge e."""
        L = 0
        idx_in_tuple = 0
        for j in range(self._spec.n):
            if j == i:
                continue
            a_j = a_minus_i[idx_in_tuple]
            idx_in_tuple += 1
            if e in self._game.routes[j][a_j]:
                L += 1
        return L

    def _other_shape(self, i: int) -> tuple[int, ...]:
        return tuple(
            self._spec.action_counts[k]
            for k in range(self._spec.n)
            if k != i
        )

    # ----- w_from_theta -----

    def w_from_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        assert theta.shape == (self._d,), f"theta must have shape ({self._d},)"
        theta_mat = theta.reshape(self._game.n_edges, 2)  # (E, 2): a_e, b_e
        w = np.zeros(self._N, dtype=np.float64)

        for i in range(self._spec.n):
            m_i = self._spec.action_counts[i]
            d_i = self._spec.d[i]
            other_shape = self._other_shape(i)
            agent_off = self._agent_offsets[i]

            r_0 = self._game.routes[i][0]

            for j in range(1, m_i):
                r_j = self._game.routes[i][j]
                diff_plus = r_0 - r_j   # edges in r_0 but not r_j: contribute +
                diff_minus = r_j - r_0  # edges in r_j but not r_0: contribute -

                block_start = agent_off + (j - 1) * d_i
                flat_idx = 0
                for a_minus_i in iter_product(*[range(s) for s in other_shape]):
                    val = 0.0
                    for e in diff_plus:
                        L = self._load_from_others(i, a_minus_i, e)
                        val += theta_mat[e, 0] + theta_mat[e, 1] * (1.0 + L)
                    for e in diff_minus:
                        L = self._load_from_others(i, a_minus_i, e)
                        val -= theta_mat[e, 0] + theta_mat[e, 1] * (1.0 + L)
                    w[block_start + flat_idx] = val
                    flat_idx += 1

        return w

    # ----- dense projection Φ^T q -----

    def project_cut_dense(
        self, q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        assert q.shape == (self._N,), f"q must have shape ({self._N},)"
        q_theta = np.zeros(self._d, dtype=np.float64)

        for i in range(self._spec.n):
            m_i = self._spec.action_counts[i]
            d_i = self._spec.d[i]
            other_shape = self._other_shape(i)
            agent_off = self._agent_offsets[i]

            r_0 = self._game.routes[i][0]

            for j in range(1, m_i):
                r_j = self._game.routes[i][j]
                diff_plus = r_0 - r_j
                diff_minus = r_j - r_0

                block_start = agent_off + (j - 1) * d_i
                flat_idx = 0
                for a_minus_i in iter_product(*[range(s) for s in other_shape]):
                    val_q = float(q[block_start + flat_idx])
                    flat_idx += 1
                    if val_q == 0.0:
                        continue
                    for e in diff_plus:
                        L = self._load_from_others(i, a_minus_i, e)
                        q_theta[e * 2 + 0] += val_q
                        q_theta[e * 2 + 1] += val_q * (1.0 + L)
                    for e in diff_minus:
                        L = self._load_from_others(i, a_minus_i, e)
                        q_theta[e * 2 + 0] -= val_q
                        q_theta[e * 2 + 1] -= val_q * (1.0 + L)

        return q_theta

    # ----- sparse projection Φ^T q from nonzero blocks -----

    def project_cut_sparse(
        self,
        cut_blocks: Mapping[tuple[int, int], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        q_theta = np.zeros(self._d, dtype=np.float64)

        for (i, k), block in cut_blocks.items():
            d_i = self._spec.d[i]
            assert block.shape == (d_i,), (
                f"block for (i={i}, k={k}) has shape {block.shape}, expected ({d_i},)"
            )
            other_shape = self._other_shape(i)
            r_0 = self._game.routes[i][0]
            r_k = self._game.routes[i][k]
            diff_plus = r_0 - r_k
            diff_minus = r_k - r_0

            flat_idx = 0
            for a_minus_i in iter_product(*[range(s) for s in other_shape]):
                val_q = float(block[flat_idx])
                flat_idx += 1
                if val_q == 0.0:
                    continue
                for e in diff_plus:
                    L = self._load_from_others(i, a_minus_i, e)
                    q_theta[e * 2 + 0] += val_q
                    q_theta[e * 2 + 1] += val_q * (1.0 + L)
                for e in diff_minus:
                    L = self._load_from_others(i, a_minus_i, e)
                    q_theta[e * 2 + 0] -= val_q
                    q_theta[e * 2 + 1] -= val_q * (1.0 + L)

        return q_theta


__all__ = ["FeatureMap", "CongestionFeatureMap"]
