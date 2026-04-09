"""Index arithmetic and w-vector packing utilities.

The w-vector in R^N stores pairwise utility differences using action 0
as the reference for each agent (paper Section 5):

    w = [ w_1(0,1) | w_1(0,2) | ... | w_1(0, m_1-1) |
          w_2(0,1) | ...                               |
          w_n(0,1) | ... | w_n(0, m_n-1) ]

Each block w_i(0, j) has d_i = M / m_i entries, indexed by a_{-i}
in lexicographic order (numpy ravel_multi_index with C order).

Other differences are recovered via the triangular identity (paper p.13):
    w_i(a, b) = w_i(0, b) - w_i(0, a)     for a, b != 0
    w_i(a, 0) = -w_i(0, a)
    w_i(0, b) = stored directly
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .types import GameSpec


def agent_block_offsets(spec: GameSpec) -> NDArray[np.int64]:
    """Starting index in R^N for each agent's block of w.

    offset[i] = sum_{k < i} (m_k - 1) * d_k

    NOTE: The paper's Algorithm 4 uses (i-1)(m_i-1)d_i which is only
    correct for homogeneous action sets. We use the correct cumulative sum.
    """
    offsets = np.zeros(spec.n, dtype=np.int64)
    for k in range(1, spec.n):
        m_k = spec.action_counts[k - 1]
        d_k = spec.d[k - 1]
        offsets[k] = offsets[k - 1] + (m_k - 1) * d_k
    return offsets


def profiles_with_agent_action(
    spec: GameSpec, agent: int, action: int
) -> NDArray[np.int64]:
    """Flat indices of all joint profiles where agent plays the given action.

    Returns array of shape (d_i,) = (M / m_i,) in lexicographic order
    over the OTHER agents' actions.  This matches the ordering used by
    np.take(utilities[i], action, axis=agent).ravel().
    """
    # Build ranges for each agent; fix the target agent to `action`
    ranges = [np.arange(m) for m in spec.action_counts]
    ranges[agent] = np.array([action])
    grids = np.meshgrid(*ranges, indexing="ij")
    multi_idx = tuple(g.ravel() for g in grids)
    return np.ravel_multi_index(multi_idx, spec.action_counts)


class IndexCache:
    """Pre-computed index arrays for a given GameSpec.

    Avoids recomputing meshgrids on every LP build or oracle call.
    """

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.offsets = agent_block_offsets(spec)
        # profile_idx[i][a_i] = flat indices where agent i plays a_i
        self.profile_idx: list[list[NDArray[np.int64]]] = []
        for i in range(spec.n):
            agent_list = []
            for a_i in range(spec.action_counts[i]):
                agent_list.append(profiles_with_agent_action(spec, i, a_i))
            self.profile_idx.append(agent_list)

    def get_w_block(
        self, w: NDArray[np.float64], agent: int, a_i: int, a_prime_i: int
    ) -> NDArray[np.float64]:
        """Extract w_i(a_i, a'_i) from packed w-vector.

        Uses the triangular identity (paper Section 5):
            w_i(a, b) = w_i(0, b) - w_i(0, a)

        Parameters
        ----------
        w : shape (N,)
        agent : agent index i
        a_i, a_prime_i : action pair (a_i, a'_i), 0-indexed

        Returns
        -------
        ndarray of shape (d_i,)
        """
        d_i = self.spec.d[agent]
        offset = int(self.offsets[agent])

        def ref_block(j: int) -> NDArray[np.float64]:
            """w_i(0, j) for j >= 1."""
            start = offset + (j - 1) * d_i
            return w[start : start + d_i]

        if a_i == 0 and a_prime_i == 0:
            return np.zeros(d_i)
        elif a_i == 0:
            return ref_block(a_prime_i).copy()
        elif a_prime_i == 0:
            return -ref_block(a_i)
        else:
            return ref_block(a_prime_i) - ref_block(a_i)
