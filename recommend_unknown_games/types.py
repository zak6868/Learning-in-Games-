"""Core data structures for the recommendation framework.

Matches the parameterization in:
    Alanqary, Baba, Wu, Bayen. "Learning to Recommend in Unknown Games."
    arXiv:2602.16998v2, March 2026.

Notation (paper Section 2):
    n       number of agents
    m_i     number of actions for agent i
    A_i     action set for agent i, {0, ..., m_i - 1}
    A       joint action set, |A| = M = prod(m_i)
    d_i     |A_{-i}| = M / m_i
    N       dimension of packed w-vector = sum_i (m_i - 1) * d_i
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GameSpec:
    """Immutable specification of game dimensions."""

    n: int                              # number of agents
    action_counts: tuple[int, ...]      # (m_1, ..., m_n)
    M: int                              # prod(m_i) = |A|
    d: tuple[int, ...]                  # (d_1, ..., d_n), d_i = M / m_i
    N: int                              # dim of w-vector

    @staticmethod
    def from_action_counts(action_counts: tuple[int, ...]) -> GameSpec:
        n = len(action_counts)
        M = math.prod(action_counts)
        d = tuple(M // m_i for m_i in action_counts)
        N = sum((m_i - 1) * d_i for m_i, d_i in zip(action_counts, d))
        return GameSpec(n=n, action_counts=action_counts, M=M, d=d, N=N)


@dataclass
class OracleResult:
    """Output of one separation-oracle query (Algorithm 4 in the paper)."""

    recommendation: tuple[int, ...]             # a(t) sampled from x(t)
    response: tuple[int, ...]                   # a*(t) observed from agents
    deviated: bool                              # any agent deviated?
    cut: Optional[NDArray[np.float64]]          # q(t) in R^N, None if no deviation
    ce_dist: NDArray[np.float64]                # x(t), flat array of shape (M,)
    regret: float                               # r(a, a*, x) = sum_i phi_i
