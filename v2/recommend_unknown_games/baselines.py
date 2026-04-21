"""Baseline: each player runs multiplicative weights independently (no moderator).

Classical no-swap-regret via Blum-Mansour (2007):
Each player i maintains m_i copies of Hedge (one per action).
Swap regret = O(m_i * sqrt(T * log(m_i))).

This serves as comparison to the moderator's O(nM log T).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .game import NormalFormGame


@dataclass
class BaselineResult:
    T: int
    player_swap_regret: NDArray[np.float64]   # (T, n)
    cumulative_welfare: NDArray[np.float64]    # (T,) sum of utilities


def run_multiplicative_weights(
    game: NormalFormGame,
    T: int,
    eta: float | None = None,
    seed: int = 42,
) -> BaselineResult:
    """Each player runs multiplicative weights independently.

    No moderator. Players play against each other, observe their own
    utility, and update weights. External regret O(sqrt(T log m)).

    Parameters
    ----------
    game : the true game
    T : number of rounds
    eta : learning rate (default: sqrt(log(m) / T))
    seed : RNG seed
    """
    rng = np.random.default_rng(seed)
    n = game.spec.n
    action_counts = game.spec.action_counts

    # Per-player weights
    weights = [np.ones(m_i) for m_i in action_counts]

    # Tracking for swap regret computation
    # swap_gains[i][a][b] = sum_{t: a_i(t)=a} u_i(b, a_{-i}(t))
    swap_gains = [[np.zeros(m_i) for _ in range(m_i)] for m_i in action_counts]
    actual_utility = [0.0 for _ in range(n)]
    player_swap_regret = np.zeros((T, n))
    cumulative_welfare = np.zeros(T)
    total_welfare = 0.0

    for t in range(T):
        # Each player samples action from their distribution
        actions = []
        for i in range(n):
            probs = weights[i] / weights[i].sum()
            a_i = rng.choice(action_counts[i], p=probs)
            actions.append(a_i)
        actions = tuple(actions)

        # Observe utilities
        for i in range(n):
            a_i = actions[i]
            u_i = game.utilities[i][actions]
            actual_utility[i] += u_i
            total_welfare += u_i

            # Swap gains: for each alternative action b
            for b in range(action_counts[i]):
                alt_actions = list(actions)
                alt_actions[i] = b
                u_b = game.utilities[i][tuple(alt_actions)]
                swap_gains[i][a_i][b] += u_b

            # Update weights (Hedge / multiplicative weights)
            m_i = action_counts[i]
            lr = eta if eta is not None else np.sqrt(np.log(m_i) / max(t + 1, 1))
            for b in range(m_i):
                alt_actions = list(actions)
                alt_actions[i] = b
                u_b = game.utilities[i][tuple(alt_actions)]
                weights[i][b] *= np.exp(lr * u_b)
            # Normalize to prevent overflow
            weights[i] /= weights[i].sum()

        # Compute swap regret at time t
        for i in range(n):
            sr = sum(np.max(swap_gains[i][a]) for a in range(action_counts[i])) \
                 - actual_utility[i]
            player_swap_regret[t, i] = max(sr, 0.0)

        cumulative_welfare[t] = total_welfare

    return BaselineResult(
        T=T,
        player_swap_regret=player_swap_regret,
        cumulative_welfare=cumulative_welfare,
    )
