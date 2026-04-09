"""Separation oracle for the cutting-plane algorithm.

Paper reference: Algorithm 4 (Separation oracle), Section 5.1.

Pipeline per query:
    1. Receive guess w(t) of utility differences
    2. Compute x(t) in CE(w(t))
    3. Sample recommendation a(t) ~ x(t)
    4. Observe agents' responses a*(t) via BR or QR
    5. If any agent deviated, construct separating hyperplane q(t) in R^N

Separation property (paper eq. 9 and surrounding text):
    <w*, q(t)> = sum_i phi_i(a_i(t), a*_i(t), x(t))  >= 0
    <w(t), q(t)>                                        <= 0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import GameSpec, OracleResult
from .utils import IndexCache
from .game import NormalFormGame, AgentFeedback
from .ce_solver import CESolver


class SeparationOracle:
    """Implements Algorithm 4: guess -> CE -> interact -> cut."""

    def __init__(
        self,
        game: NormalFormGame,
        ce_solver: CESolver,
        feedback: AgentFeedback,
        seed: int = 0,
    ):
        self.game = game
        self.spec = game.spec
        self.ce_solver = ce_solver
        self.feedback = feedback
        self.cache = IndexCache(self.spec)
        self.rng = np.random.default_rng(seed)

    def query(self, w_guess: NDArray[np.float64]) -> OracleResult:
        """Execute one oracle query.

        Parameters
        ----------
        w_guess : shape (N,), guessed utility-difference vector

        Returns
        -------
        OracleResult with cut if deviation occurred
        """
        # Step 1: Solve CE for the guessed game
        x = self.ce_solver.solve(w_guess)
        if x is None:
            raise RuntimeError(
                "CE LP infeasible for w_guess. This should not happen "
                "for any w since every finite game has a CE."
            )

        # Step 2: Sample recommendation
        a_t = self._sample_profile(x)

        # Step 3: Observe agents' responses
        a_star = self.feedback.respond_all(a_t, x)

        # Step 4: Construct cut if any agent deviated
        deviated = a_t != a_star
        cut = None
        regret = 0.0

        if deviated:
            cut = self._construct_cut(a_t, a_star, x)
            # Compute per-round regret: r(a, a*, x) = sum_i phi_i
            regret = sum(
                self.game.phi(i, a_t[i], a_star[i], x)
                for i in range(self.spec.n)
                if a_t[i] != a_star[i]
            )

        return OracleResult(
            recommendation=a_t,
            response=a_star,
            deviated=deviated,
            cut=cut,
            ce_dist=x,
            regret=regret,
        )

    def _sample_profile(self, x: NDArray[np.float64]) -> tuple[int, ...]:
        """Sample a joint action profile from distribution x."""
        flat_idx = self.rng.choice(self.spec.M, p=x)
        return tuple(
            int(c) for c in np.unravel_index(flat_idx, self.spec.action_counts)
        )

    def _construct_cut(
        self,
        a_t: tuple[int, ...],
        a_star: tuple[int, ...],
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Construct separating hyperplane q(t) in R^N (Algorithm 4).

        For each deviating agent i:
            j     = a_t[i]    (recommended action)
            j*    = a_star[i] (realized action)
            cond  = x(j, .)   (conditional over a_{-i}, shape d_i)

        The cut q places:
            +cond in block (i, j*)  if j* != 0
            -cond in block (i, j)   if j  != 0

        This encodes the profitable deviation from j to j* via the
        triangular identity w_i(j, j*) = w_i(0, j*) - w_i(0, j).

        Separation property:
            <w*, q> = sum_i <cond, w_i(j, j*)>
                    = sum_i phi_i(j, j*, x)    >= 0   (agents deviate profitably)
            <w(t), q> <= 0                             (x is CE of w(t))
        """
        q = np.zeros(self.spec.N, dtype=np.float64)
        x_multi = x.reshape(self.spec.action_counts)

        for i in range(self.spec.n):
            j = a_t[i]
            j_star = a_star[i]
            if j == j_star:
                continue

            d_i = self.spec.d[i]
            offset = int(self.cache.offsets[i])

            # Conditional: x(a_i = j, a_{-i}) for all a_{-i}
            cond = np.take(x_multi, j, axis=i).ravel()  # (d_i,)

            # +cond in block for w_i(0, j*)
            if j_star != 0:
                start = offset + (j_star - 1) * d_i
                q[start : start + d_i] += cond

            # -cond in block for w_i(0, j)
            if j != 0:
                start = offset + (j - 1) * d_i
                q[start : start + d_i] -= cond

        return q
