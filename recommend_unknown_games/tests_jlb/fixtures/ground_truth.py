"""Ground-truth computations used to validate JLB outputs.

For small games (M <= a few hundred profiles) we can:
  - solve the full normal-form CE LP with the existing `CESolver`
  - enumerate all profiles to compute expected utilities
  - verify any proposed CE by checking every constraint literally.

These routines are intentionally slow and simple.  They are our "reference
truth" for debugging any faster implementation.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from gmpy2 import mpq

from ...game import NormalFormGame
from ...ce_solver import CESolver
from ..utils.exact_validation import Mixture, max_incentive_violation


def brute_force_ce(game: NormalFormGame, backend: str = "auto") -> Mixture:
    """Compute a CE by solving the full normal-form LP, return as Mixture.

    Uses the existing CESolver (Gurobi or scipy) and converts the dense
    solution into a sparse Mixture with rational probabilities.
    """
    solver = CESolver(game.spec, backend=backend)
    w = game.pack_w()
    x = solver.solve(w)
    if x is None:
        raise RuntimeError("CE LP infeasible; this should never happen for finite games")
    return Mixture.from_dense(game.spec.action_counts, x)


def enumerate_profiles(spec_action_counts: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """Iterate over all joint pure profiles.  Only feasible for small M."""
    for profile in product(*[range(m) for m in spec_action_counts]):
        yield profile


def expected_utility_brute_force(game: NormalFormGame, agent: int,
                                 x_product: list[NDArray[np.float64]]) -> float:
    """Compute E_{a ~ x_product}[u_agent(a)] by enumerating all profiles.

    x_product[i] is a probability vector over agent i's actions.
    Returns a float; intended for small games only.
    """
    assert len(x_product) == game.spec.n
    total = 0.0
    for profile in enumerate_profiles(game.spec.action_counts):
        prob = 1.0
        for i, a_i in enumerate(profile):
            prob *= float(x_product[i][a_i])
        if prob == 0.0:
            continue
        total += prob * float(game.utilities[agent][profile])
    return total


def ce_constraints_dense(game: NormalFormGame) -> NDArray[np.float64]:
    """Return the full CE constraint matrix U with shape (N, M).

    Row (i, a_i, a_prime) corresponds to the incentive constraint for agent
    i's pair (a_i, a_prime), and entry (row, flat_index(a)) equals
        u_i(a_prime, a_{-i}) - u_i(a_i, a_{-i})   if a's i-th coord is a_i
    and 0 otherwise.  (Sign convention: "gain from deviating" >= 0 would
    violate CE, so a valid CE satisfies U x <= 0.)
    """
    spec = game.spec
    M = spec.M
    rows: list[NDArray[np.float64]] = []
    action_counts = spec.action_counts
    for i in range(spec.n):
        m_i = action_counts[i]
        for a_i in range(m_i):
            for a_prime in range(m_i):
                if a_prime == a_i:
                    continue
                row = np.zeros(M, dtype=np.float64)
                for profile in enumerate_profiles(action_counts):
                    if profile[i] != a_i:
                        continue
                    profile_dev = list(profile)
                    profile_dev[i] = a_prime
                    flat = int(np.ravel_multi_index(profile, action_counts))
                    u_here = float(game.utilities[i][profile])
                    u_dev = float(game.utilities[i][tuple(profile_dev)])
                    row[flat] = u_dev - u_here
                rows.append(row)
    return np.stack(rows, axis=0)


def verify_ce_dense(game: NormalFormGame, x: NDArray[np.float64],
                    tol: float = 1e-10) -> tuple[bool, float]:
    """Return (is_ce, max_violation) for a dense distribution x over profiles.

    Floating-point version; for rational verification use Mixture +
    max_incentive_violation in utils.exact_validation.
    """
    U = ce_constraints_dense(game)
    violations = U @ x
    max_v = float(np.max(violations))
    return bool(max_v <= tol), max_v


__all__ = [
    "brute_force_ce",
    "enumerate_profiles",
    "expected_utility_brute_force",
    "ce_constraints_dense",
    "verify_ce_dense",
]
