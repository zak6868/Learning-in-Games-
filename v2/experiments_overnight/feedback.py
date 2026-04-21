"""BR and γ-BR feedback wrappers, plus per-agent φ computation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from recommend_unknown_games.compact_feedback import CompactAgentFeedback
from recommend_unknown_games.compact_game import CompactGame


def make_feedback(
    game: CompactGame,
    kind: str = "BR",
    gamma: float = 0.0,
    seed: int = 0,
) -> CompactAgentFeedback:
    """Build a CompactAgentFeedback with the right γ.

    `kind` ∈ {'BR', 'gamma_BR'}.  For BR we force γ=0.  For gamma_BR
    the supplied `gamma` is used: an agent deviates only if the gain
    strictly exceeds γ.
    """
    assert kind in ("BR", "gamma_BR")
    g = 0.0 if kind == "BR" else float(gamma)
    return CompactAgentFeedback(
        game, model="best_response", gamma=g, seed=seed,
    )


def per_agent_phi(
    game: CompactGame,
    recommendation: tuple[int, ...],
    profiles: Sequence[tuple[int, ...]],
    alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute φ_i(a_i, BR_i, x) for every agent i, where
        φ_i = E_{a_{-i} ~ x(·|a_i)}[u_i(BR_i, a_{-i}) − u_i(a_i, a_{-i})]
    i.e. how much the agent would gain by switching to their best
    alternative under the conditional.

    Returned as a length-n float64 vector (≥ 0).  φ_i = 0 means agent
    is already BR-compliant with the recommendation.
    """
    spec = game.spec
    n = spec.n
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m_i = spec.action_counts[i]
        a_i = recommendation[i]
        # E[u_i(a', a_{-i})] conditional on profile_k[i] = a_i
        expected = np.zeros(m_i)
        total_w = 0.0
        for pk, ak in zip(profiles, alpha):
            if pk[i] != a_i:
                continue
            for a_prime in range(m_i):
                prof_cand = list(pk)
                prof_cand[i] = a_prime
                expected[a_prime] += float(ak) * game.utility(i, tuple(prof_cand))
            total_w += float(ak)
        if total_w <= 0:
            continue
        expected /= total_w
        out[i] = float(max(0.0, expected.max() - expected[a_i]))
    return out


__all__ = ["make_feedback", "per_agent_phi"]
