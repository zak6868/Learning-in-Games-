"""Drop-in CE solver using the JLB pipeline instead of a normal-form LP.

Same signature as `CESolver`: takes a utility-difference vector w, returns
a dense probability distribution over joint action profiles.

Internally:
  1. Reconstruct a NormalFormGame from w by setting u_i(0, a_{-i}) = 0
     and u_i(j, a_{-i}) = w_i(0, j)[a_{-i}].  (CE depends only on utility
     differences, so this preserves the CE set.)
  2. Wrap as NormalFormCompactGame and run JLB ellipsoid.
  3. Convert the returned sparse mixture to a dense M-vector.

For large games the dense conversion is wasteful; the purpose of this
class is comparison with the existing normal-form `CESolver`.  A future
`CompactSeparationOracle` would avoid ever materializing the dense vector.
"""

from __future__ import annotations

from itertools import product as iter_product
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .compact_game import NormalFormCompactGame
from .game import NormalFormGame
from .jlb_driver import run_jlb
from .jlb_ellipsoid import run_jlb_ellipsoid
from .types import GameSpec
from .utils import IndexCache


def normal_form_from_w(spec: GameSpec, w: NDArray[np.float64]) -> NormalFormGame:
    """Build a NormalFormGame whose w-vector is exactly `w`.

    Sets u_i(0, a_{-i}) = 0 for all i and all a_{-i}, and
    u_i(j, a_{-i}) = w_i(0, j)[a_{-i}] for j = 1, ..., m_i - 1.

    Because the CE LP depends only on utility differences, this
    game has the same CE set as any other game with w-vector equal to w.
    """
    cache = IndexCache(spec)
    utilities: list[NDArray[np.float64]] = []
    for i in range(spec.n):
        U_i = np.zeros(spec.action_counts, dtype=np.float64)
        m_i = spec.action_counts[i]
        for j in range(1, m_i):
            # block is a flat array of length d_i, indexed by a_{-i} in
            # ravel order (C order over the other agents' action counts)
            block = cache.get_w_block(w, i, 0, j)
            # Fill U_i[a_i = j, a_{-i}] = block[flat(a_{-i})]
            other_shape = tuple(
                spec.action_counts[k] for k in range(spec.n) if k != i
            )
            for a_minus_flat, a_minus in enumerate(iter_product(*[range(k) for k in other_shape])):
                profile = list(a_minus)
                profile.insert(i, j)
                U_i[tuple(profile)] = float(block[a_minus_flat])
        utilities.append(U_i)
    return NormalFormGame(spec.action_counts, utilities)


class JLBCESolver:
    """Drop-in replacement for CESolver using the JLB pipeline internally.

    Parameters
    ----------
    spec : GameSpec
    outer_loop : "ellipsoid" | "structured" | "random"
        Which outer loop to use for the JLB inner solve.
    **jlb_kwargs : passed to run_jlb_ellipsoid or run_jlb.
    """

    def __init__(self, spec: GameSpec, outer_loop: str = "ellipsoid",
                 **jlb_kwargs):
        self.spec = spec
        self.outer_loop = outer_loop
        self.jlb_kwargs = jlb_kwargs

    def solve(self, w: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Return a dense distribution over M profiles that is a CE of the
        game with utility-difference vector w.  Returns None on failure."""
        nf = normal_form_from_w(self.spec, w)
        cg = NormalFormCompactGame(nf)

        if self.outer_loop == "ellipsoid":
            result = run_jlb_ellipsoid(cg, **self.jlb_kwargs)
            alpha = result.alpha
            profiles = result.profiles
        else:
            result = run_jlb(cg, **self.jlb_kwargs)
            alpha = result.alpha
            profiles = result.profiles

        if alpha is None:
            return None

        # Build dense distribution by merging duplicates and scattering
        dense = np.zeros(self.spec.M, dtype=np.float64)
        total = float(np.asarray(alpha).sum())
        if total <= 0:
            return None
        for profile, a in zip(profiles, alpha):
            if a <= 0:
                continue
            flat = int(np.ravel_multi_index(profile, self.spec.action_counts))
            dense[flat] += float(a) / total
        return dense


__all__ = ["normal_form_from_w", "JLBCESolver"]
