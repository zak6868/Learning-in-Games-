"""Correlated Equilibrium solver via LP.

Paper reference: Definition 1 (CE), page 4.

A distribution x in Delta(A) is a CE if for all agents i,
all a_i in A_i, all a'_i in A_i:

    sum_{a_{-i}} x(a_i, a_{-i}) * [u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i})] <= 0

Equivalently:   A @ x <= 0,  1^T x = 1,  x >= 0

where row (i, a_i, a'_i) of A is the vector
    A[row, flat(a_i, a_{-i})] = w_i(a_i, a'_i)[a_{-i}]  for all a_{-i}
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .types import GameSpec
from .utils import IndexCache


class CESolver:
    """LP-based solver for Correlated Equilibrium.

    The sparsity pattern (row/col indices) is precomputed once at init.
    The data values are rebuilt per-solve from the w-vector -- fully
    vectorized, no Python loops.
    """

    def __init__(self, spec: GameSpec, backend: str = "auto"):
        self.spec = spec
        self.cache = IndexCache(spec)

        if backend == "auto":
            try:
                import gurobipy  # noqa: F401
                self.backend = "gurobi"
            except ImportError:
                self.backend = "scipy"
        else:
            self.backend = backend

        # Precompute sparsity pattern and data-fetch indices
        self._num_constraints = 0
        self._coo_rows = np.array([], dtype=np.int64)
        self._coo_cols = np.array([], dtype=np.int64)
        # For vectorized _build_data: each constraint's data is a slice
        # of w or a difference of two slices (triangular identity).
        # We store the fetch instructions as arrays of indices into w.
        self._data_src_a: NDArray[np.int64] = np.array([], dtype=np.int64)
        self._data_src_b: NDArray[np.int64] = np.array([], dtype=np.int64)
        self._data_sign_a: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._data_sign_b: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._constraint_map: list[tuple[int, int, int]] = []
        self._precompute()

    def _precompute(self) -> None:
        """Build COO row/col arrays AND vectorized data-fetch arrays.

        For each constraint (i, a_i, a'_i), the data is w_i(a_i, a'_i)
        which, via the triangular identity, equals:

            a_i == 0:       w[offset + (a'_i-1)*d_i : ...]        (direct)
            a'_i == 0:     -w[offset + (a_i-1)*d_i : ...]         (negate)
            else:           w[block for a'_i] - w[block for a_i]   (difference)

        We encode this as:  data = sign_a * w[src_a] + sign_b * w[src_b]
        where src_a, src_b are index arrays into a padded w-vector
        (w_padded = [w..., 0...0] with d_max zeros appended as a "zero block").
        """
        spec = self.spec
        total_nnz = sum(
            spec.action_counts[i] * (spec.action_counts[i] - 1) * spec.d[i]
            for i in range(spec.n)
        )
        N = spec.N
        d_max = max(spec.d) if spec.n > 0 else 0
        # Index of the zero-block in padded w (for a_i==0 or a'_i==0 cases)
        zero_block_start = N  # w_padded[N : N+d_max] = 0

        coo_rows = np.empty(total_nnz, dtype=np.int64)
        coo_cols = np.empty(total_nnz, dtype=np.int64)
        src_a = np.empty(total_nnz, dtype=np.int64)
        src_b = np.empty(total_nnz, dtype=np.int64)
        sign_a = np.empty(total_nnz, dtype=np.float64)
        sign_b = np.empty(total_nnz, dtype=np.float64)

        row = 0
        pos = 0
        offsets = self.cache.offsets

        for i in range(spec.n):
            m_i = spec.action_counts[i]
            d_i = spec.d[i]
            offset_i = int(offsets[i])

            for a_i in range(m_i):
                col_indices = self.cache.profile_idx[i][a_i]  # (d_i,)
                for a_prime in range(m_i):
                    if a_prime == a_i:
                        continue
                    self._constraint_map.append((i, a_i, a_prime))

                    sl = slice(pos, pos + d_i)
                    coo_rows[sl] = row
                    coo_cols[sl] = col_indices

                    # w_i(a_i, a'_i) = w_i(0, a'_i) - w_i(0, a_i)
                    # block for action j (j>=1): offset_i + (j-1)*d_i
                    if a_i == 0 and a_prime == 0:
                        # impossible (a_i != a_prime)
                        pass
                    elif a_i == 0:
                        # w_i(0, a'_i) = direct block
                        block = offset_i + (a_prime - 1) * d_i
                        src_a[sl] = np.arange(block, block + d_i)
                        sign_a[sl] = 1.0
                        src_b[sl] = np.arange(zero_block_start, zero_block_start + d_i)
                        sign_b[sl] = 0.0
                    elif a_prime == 0:
                        # w_i(a_i, 0) = -w_i(0, a_i)
                        block = offset_i + (a_i - 1) * d_i
                        src_a[sl] = np.arange(block, block + d_i)
                        sign_a[sl] = -1.0
                        src_b[sl] = np.arange(zero_block_start, zero_block_start + d_i)
                        sign_b[sl] = 0.0
                    else:
                        # w_i(a_i, a'_i) = w_i(0, a'_i) - w_i(0, a_i)
                        block_a = offset_i + (a_prime - 1) * d_i
                        block_b = offset_i + (a_i - 1) * d_i
                        src_a[sl] = np.arange(block_a, block_a + d_i)
                        sign_a[sl] = 1.0
                        src_b[sl] = np.arange(block_b, block_b + d_i)
                        sign_b[sl] = -1.0

                    row += 1
                    pos += d_i

        self._num_constraints = row
        self._coo_rows = coo_rows
        self._coo_cols = coo_cols
        self._data_src_a = src_a
        self._data_src_b = src_b
        self._data_sign_a = sign_a
        self._data_sign_b = sign_b

    def _build_data(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build data array -- single vectorized operation, no Python loop."""
        # Pad w with zeros for the "zero block" trick
        d_max = max(self.spec.d) if self.spec.n > 0 else 0
        w_pad = np.concatenate([w, np.zeros(d_max)])
        return self._data_sign_a * w_pad[self._data_src_a] + \
               self._data_sign_b * w_pad[self._data_src_b]

    def build_constraint_matrix(self, w: NDArray[np.float64]) -> csr_matrix:
        """Build sparse CE constraint matrix A such that A @ x <= 0."""
        data = self._build_data(w)
        A_coo = coo_matrix(
            (data, (self._coo_rows, self._coo_cols)),
            shape=(self._num_constraints, self.spec.M),
        )
        return A_coo.tocsr()

    def solve(self, w: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Compute a CE for the game defined by utility-difference vector w."""
        A = self.build_constraint_matrix(w)
        if self.backend == "gurobi":
            return self._solve_gurobi(A)
        return self._solve_scipy(A)

    def _solve_gurobi(self, A: csr_matrix) -> NDArray[np.float64] | None:
        import gurobipy as gp

        M = self.spec.M
        C = self._num_constraints

        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.Method = 1

        x = model.addMVar(M, lb=0.0, name="x")
        model.addMConstr(A, x, "<", np.zeros(C))
        model.addConstr(x.sum() == 1.0)
        model.setObjective(x @ x, gp.GRB.MINIMIZE)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            sol = np.array(x.X, dtype=np.float64)
            sol = np.maximum(sol, 0.0)
            sol /= sol.sum()
            return sol
        return None

    def _solve_scipy(self, A: csr_matrix) -> NDArray[np.float64] | None:
        from scipy.optimize import linprog
        from scipy.sparse import csr_array

        M = self.spec.M
        C = self._num_constraints

        c = np.zeros(M)
        b_ub = np.zeros(C)
        A_eq = csr_array(np.ones((1, M)))
        b_eq = np.array([1.0])

        result = linprog(
            c, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=[(0, None)] * M, method="highs",
        )
        if result.success:
            sol = np.array(result.x, dtype=np.float64)
            sol = np.maximum(sol, 0.0)
            sol /= sol.sum()
            return sol
        return None

    def is_ce(self, w: NDArray[np.float64], x: NDArray[np.float64],
              tol: float = 1e-8) -> bool:
        """Check whether x is a CE for the game defined by w."""
        A = self.build_constraint_matrix(w)
        violations = A @ x
        return bool(np.all(violations <= tol))
