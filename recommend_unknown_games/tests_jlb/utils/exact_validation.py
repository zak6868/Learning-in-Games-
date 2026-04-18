"""Exact rational-arithmetic validation of correlated equilibria.

A CE is verified by literally checking every incentive constraint
    sum_{a_{-i}} x(a_i, a_{-i}) * [u_i(a_i, a_{-i}) - u_i(a'_i, a_{-i})] >= 0
in rational arithmetic via gmpy2.mpq.  If every utility entry is an integer
or small rational (what our benchmark games provide) and the CE probabilities
are rational, this check gives literal equality to zero for exact CE.

Mixture representation
----------------------
A CE is stored as a `Mixture`: a list of `(profile, weight)` pairs where
`profile` is a tuple of per-agent actions and `weight` is a gmpy2.mpq.
Profiles with zero weight are dropped.  The weights sum to exactly 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from gmpy2 import mpq

from ...game import NormalFormGame


@dataclass(frozen=True)
class Mixture:
    """A CE represented as a sparse mixture of pure profiles."""

    spec_action_counts: tuple[int, ...]
    support: tuple[tuple[int, ...], ...]  # tuple of profiles (tuples of ints)
    weights: tuple  # tuple of gmpy2.mpq, same length as support

    def __post_init__(self):
        assert len(self.support) == len(self.weights), \
            "support and weights must have same length"
        # Verify weights are non-negative rationals summing to 1
        total = mpq(0)
        for w in self.weights:
            assert w >= 0, f"negative weight {w}"
            total = total + mpq(w)
        assert total == mpq(1), f"weights sum to {total}, expected 1"

    def probability_of(self, profile: tuple[int, ...]) -> object:
        """Return P[profile] as an mpq.  Zero if profile not in support."""
        result = mpq(0)
        for a, w in zip(self.support, self.weights):
            if a == profile:
                result = result + mpq(w)
        return result

    def marginal(self, agent: int, action: int) -> object:
        """Return the marginal probability P[agent plays action] as an mpq."""
        result = mpq(0)
        for profile, w in zip(self.support, self.weights):
            if profile[agent] == action:
                result = result + mpq(w)
        return result

    @staticmethod
    def from_dense(spec_action_counts: tuple[int, ...],
                   x: NDArray[np.float64],
                   tol: float = 0.0) -> "Mixture":
        """Build a Mixture from a dense float distribution over profiles.

        Entries with |x(a)| <= tol are treated as zero.  If tol == 0 we
        keep every non-zero entry exactly.  Floats are converted to mpq
        via exact binary representation (gmpy2.mpq(float)).
        """
        support: list[tuple[int, ...]] = []
        weights: list = []
        total = mpq(0)
        for flat_idx in range(x.size):
            val = float(x.flat[flat_idx])
            if abs(val) <= tol:
                continue
            if val < 0:
                raise ValueError(f"negative probability {val} at index {flat_idx}")
            profile = tuple(int(c) for c in np.unravel_index(flat_idx, spec_action_counts))
            w = mpq(val)
            support.append(profile)
            weights.append(w)
            total = total + w
        # Normalize to sum exactly to 1 if needed (float rounding may give slightly off)
        if total != mpq(1):
            weights = [mpq(w) / total for w in weights]
        return Mixture(spec_action_counts, tuple(support), tuple(weights))


def _utility_mpq(game, agent: int, profile: tuple[int, ...]) -> object:
    """Return u_agent(profile) as an mpq.  Works for integer utilities exactly.

    Accepts either a NormalFormGame (with `.utilities` attribute) or any
    object exposing a `.utility(agent, profile)` method (e.g.,
    `CompactGame`).
    """
    if hasattr(game, "utility"):
        val = float(game.utility(agent, profile))
    else:
        val = float(game.utilities[agent][profile])
    return mpq(val)


def incentive_violation(game: NormalFormGame, mixture: Mixture,
                        agent: int, a_i: int, a_prime_i: int) -> object:
    """Return the value
        sum_{a_{-i}} P[a_i, a_{-i}] * [u_i(a_prime_i, a_{-i}) - u_i(a_i, a_{-i})]
    as an mpq.  A CE requires this to be <= 0.

    Note the sign: Jiang-Leyton-Brown use
        sum_{a_{-i}} [u_i(a_i, a_{-i}) - u_i(a_prime_i, a_{-i})] * x(a_i, a_{-i}) >= 0
    which equals -incentive_violation(). We return the "gain from deviating"
    convention (positive = bad).
    """
    total = mpq(0)
    for profile, weight in zip(mixture.support, mixture.weights):
        if profile[agent] != a_i:
            continue
        profile_dev = list(profile)
        profile_dev[agent] = a_prime_i
        u_here = _utility_mpq(game, agent, profile)
        u_dev = _utility_mpq(game, agent, tuple(profile_dev))
        total = total + mpq(weight) * (u_dev - u_here)
    return total


def max_incentive_violation(game: NormalFormGame, mixture: Mixture) -> object:
    """Return the maximum incentive-to-deviate gain across all (i, a_i, a'_i).

    If this is <= 0, the mixture is a CE.  Value is an mpq.
    """
    best = mpq(0)
    n = game.spec.n
    for i in range(n):
        m_i = game.spec.action_counts[i]
        for a_i in range(m_i):
            for a_prime in range(m_i):
                if a_prime == a_i:
                    continue
                v = incentive_violation(game, mixture, i, a_i, a_prime)
                if v > best:
                    best = v
    return best


def is_valid_ce_exact(game: NormalFormGame, mixture: Mixture) -> bool:
    """Return True iff the mixture is a CE in exact rational arithmetic.

    Uses gmpy2.mpq throughout so the result is not affected by float rounding.
    """
    return max_incentive_violation(game, mixture) <= mpq(0)
