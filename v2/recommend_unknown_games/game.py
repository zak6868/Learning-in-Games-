"""Normal-form game environment and agent feedback models.

Paper reference: Section 2 (Model and Preliminaries).

Utility storage
---------------
utilities[i] is an ndarray of shape (m_1, m_2, ..., m_n).
utilities[i][a_1, ..., a_n] = u_i(a_1, ..., a_n).

Feedback models (paper Definitions 2 & 3)
------------------------------------------
Best Response (BR):
    Agent i plays argmax_{a'_i} u^x_i(a'_i | a_i)
    where u^x_i(a'_i | a_i) = <x(a_i, .), u_i(a'_i, .)>

Quantal Response (QR):
    Agent i deviates to a'_i with probability proportional to
    exp(beta * phi_i(a_i, a'_i, x))  if phi_i >= 0, else 0
    where phi_i(a_i, a'_i, x) = <x(a_i, .), w_i(a_i, a'_i)>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import GameSpec


class NormalFormGame:
    """A finite normal-form game with n agents."""

    def __init__(
        self,
        action_counts: tuple[int, ...],
        utilities: list[NDArray[np.float64]] | None = None,
    ):
        """
        Parameters
        ----------
        action_counts : (m_1, ..., m_n)
        utilities : list of n arrays, each of shape (m_1, ..., m_n).
                    If None, all utilities are zero.
        """
        self.spec = GameSpec.from_action_counts(action_counts)
        if utilities is not None:
            assert len(utilities) == self.spec.n
            self.utilities = [
                np.asarray(u, dtype=np.float64) for u in utilities
            ]
            for i, u in enumerate(self.utilities):
                assert u.shape == self.spec.action_counts, (
                    f"Agent {i}: expected shape {self.spec.action_counts}, "
                    f"got {u.shape}"
                )
        else:
            self.utilities = [
                np.zeros(self.spec.action_counts, dtype=np.float64)
                for _ in range(self.spec.n)
            ]

    # ----- utility differences (paper notation w_i(a_i, a'_i)) -----

    def utility_diff_vector(
        self, agent: int, a_i: int, a_prime_i: int
    ) -> NDArray[np.float64]:
        """w_i(a_i, a'_i) = u_i(a'_i, .) - u_i(a_i, .) in R^{d_i}.

        The entries are ordered by a_{-i} in lexicographic (C) order,
        matching np.take(...).ravel().

        Paper: defined above Definition 1, page 3-4.
        """
        if a_i == a_prime_i:
            return np.zeros(self.spec.d[agent], dtype=np.float64)
        u = self.utilities[agent]
        # np.take selects along agent's axis, leaving shape
        # (m_1, ..., m_{i-1}, m_{i+1}, ..., m_n) with d_i entries
        u_prime = np.take(u, a_prime_i, axis=agent)
        u_curr = np.take(u, a_i, axis=agent)
        return (u_prime - u_curr).ravel()

    def pack_w(self) -> NDArray[np.float64]:
        """Pack the true utility differences into R^N.

        Layout: for each agent i, blocks w_i(0, j) for j = 1, ..., m_i-1.
        Each block has d_i entries.

        Paper: Section 5, page 13 -- the w* vector.
        """
        blocks = []
        for i in range(self.spec.n):
            m_i = self.spec.action_counts[i]
            for j in range(1, m_i):
                blocks.append(self.utility_diff_vector(i, 0, j))
        return np.concatenate(blocks)

    # ----- incentive to deviate (paper eq. 1) -----

    def phi(
        self,
        agent: int,
        a_i: int,
        a_prime_i: int,
        x: NDArray[np.float64],
    ) -> float:
        """Incentive of agent i to deviate from a_i to a'_i under x.

        phi_i(a_i, a'_i, x) = <x(a_i, .), w_i(a_i, a'_i)>

        Paper: Definition 1, eq. (1).
        """
        x_multi = x.reshape(self.spec.action_counts)
        cond = np.take(x_multi, a_i, axis=agent).ravel()
        w = self.utility_diff_vector(agent, a_i, a_prime_i)
        return float(cond @ w)

    # ----- factory -----

    @staticmethod
    def random_game(
        action_counts: tuple[int, ...],
        seed: int = 0,
        no_weakly_dominated: bool = False,
    ) -> NormalFormGame:
        """Generate a random game with utilities in [0, 1].

        If no_weakly_dominated=True, reject games with weakly dominated
        actions (Definition 7). This may loop for a while on large games.
        """
        rng = np.random.default_rng(seed)
        n = len(action_counts)

        while True:
            utilities = [
                rng.uniform(0, 1, size=action_counts) for _ in range(n)
            ]
            game = NormalFormGame(action_counts, utilities)
            if not no_weakly_dominated:
                return game
            # Check no weakly dominated actions (Def. 7):
            # a_i is weakly dominated if exists a'_i s.t.
            # w_i(a_i, a'_i) >= 0 everywhere and > 0 somewhere
            ok = True
            for i in range(n):
                m_i = action_counts[i]
                for a_i in range(m_i):
                    for a_prime in range(m_i):
                        if a_prime == a_i:
                            continue
                        w = game.utility_diff_vector(i, a_i, a_prime)
                        if np.all(w >= 0) and np.any(w > 0):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return game


class AgentFeedback:
    """Simulates agent responses to recommendations.

    Paper: Definitions 2 (BR) and 3 (QR), Section 2.
    """

    def __init__(
        self,
        game: NormalFormGame,
        model: str = "best_response",
        beta: float = 1.0,
        gamma: float = 0.0,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        game : the true underlying game
        model : "best_response" or "quantal_response"
        beta : rationality parameter for QR (paper eq. 3)
        gamma : threshold for gamma-BR. Agent deviates only if gain > gamma.
                gamma=0 is standard BR. gamma>0 is needed for swap regret bound.
        seed : RNG seed
        """
        assert model in ("best_response", "quantal_response")
        self.game = game
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    def _expected_utilities(
        self, agent: int, a_i: int, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Expected utility of each action for agent i, given recommendation a_i.

        u^x_i(a'_i | a_i) = <x(a_i, .), u_i(a'_i, .)>  for each a'_i.

        Returns shape (m_i,).

        Paper: footnote 1, page 4.
        """
        spec = self.game.spec
        m_i = spec.action_counts[agent]
        x_multi = x.reshape(spec.action_counts)
        # Conditional: x(a_i, .) has shape (m_1,...,m_{i-1}, m_{i+1},...,m_n)
        cond = np.take(x_multi, a_i, axis=agent).ravel()  # (d_i,)
        # Utility matrix: U_i[j, a_{-i}] = u_i(j, a_{-i}), shape (m_i, d_i)
        U_i = np.moveaxis(self.game.utilities[agent], agent, 0)
        U_i = U_i.reshape(m_i, -1)
        return U_i @ cond  # (m_i,)

    def respond(
        self, agent: int, a_i: int, x: NDArray[np.float64]
    ) -> int:
        """Agent i's chosen action given recommendation a_i under x.

        BR (Def. 2): argmax_{a'_i} u^x_i(a'_i | a_i)
        QR (Def. 3): sample from {a'_i : phi >= 0} with prob ~ exp(beta * phi)

        Paper: Definitions 2 and 3, eqs. (2)-(4).
        """
        expected = self._expected_utilities(agent, a_i, x)

        if self.model == "best_response":
            # Paper Def. 2: BR_i(x, a_i) = argmax_{a'_i} u^x_i(a'_i | a_i)
            # gamma-BR: agent follows if gain over recommended action <= gamma
            best_val = np.max(expected)
            if expected[a_i] >= best_val - self.gamma - 1e-12:
                return a_i
            return int(np.argmax(expected))

        # Quantal response: paper eq. (3)
        # phi_i(a_i, a'_i, x) = expected[a'_i] - expected[a_i]
        # Only deviations with phi >= 0 get positive probability
        gains = expected - expected[a_i]
        # Include the recommended action (phi = 0 for a_i -> a_i)
        mask = gains >= 0.0
        log_probs = np.full_like(gains, -np.inf)
        log_probs[mask] = self.beta * gains[mask]
        # Stable softmax
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()
        return int(self.rng.choice(len(probs), p=probs))

    def respond_all(
        self,
        recommendation: tuple[int, ...],
        x: NDArray[np.float64],
    ) -> tuple[int, ...]:
        """All agents' responses to a joint recommendation."""
        return tuple(
            self.respond(i, recommendation[i], x)
            for i in range(self.game.spec.n)
        )
