"""Cutting-plane algorithm for low-regret recommendations (Algorithm 5).

Paper reference: Section 5 of "Learning to Recommend in Unknown Games."

Implementation uses the Analytic Center Cutting-Plane Method (ACCPM)
following Boyd, Vandenberghe, Skaf (Stanford EE364b notes).

The knowledge set after t cuts is:
    C_t = { w in [-1,1]^N : <w, q_s> >= 0  for s = 1..t }

The analytic center minimizes the log-barrier:
    phi(w) = -sum_s log(<w, q_s>) - sum_j [log(1+w_j) + log(1-w_j)]

Newton's method (infeasible start) computes this center.
Constraint pruning keeps at most max_cuts active constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from .types import GameSpec, OracleResult
from .oracle import SeparationOracle


# =====================================================================
# ACCPM: Analytic center of the knowledge set
# =====================================================================


def analytic_center(
    cuts: NDArray[np.float64],
    w_init: NDArray[np.float64],
    box_radius: float = 1.0,
    max_newton: int = 50,
    tol: float = 1e-8,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> NDArray[np.float64]:
    """Compute the analytic center of the knowledge set via Newton's method.

    Knowledge set:
        { w in [-R, R]^N : Q^T w >= 0 }  (cuts stored as columns of Q)

    Minimizes the log-barrier:
        phi(w) = -sum_s log(q_s^T w) - sum_j [log(R + w_j) + log(R - w_j)]

    Uses damped Newton with backtracking line search.
    Warm-started from w_init (previous analytic center).

    Parameters
    ----------
    cuts : shape (N, t), each column is a cut normal q_s
    w_init : shape (N,), starting point (must be strictly feasible)
    box_radius : R, the box half-width
    max_newton : max Newton iterations
    tol : convergence tolerance on Newton decrement
    alpha, beta : backtracking line search parameters

    Returns
    -------
    w : shape (N,), the analytic center

    Reference: Boyd, Vandenberghe, Skaf, "ACCPM Notes", Section 2.
    """
    N = len(w_init)
    t = cuts.shape[1] if cuts.ndim == 2 else 0
    R = box_radius
    w = w_init.copy()

    # Ensure strict feasibility: all slacks > 0
    if t > 0:
        slacks = cuts.T @ w
        if np.min(slacks) <= 1e-10:
            w = _find_strictly_feasible(cuts, R)

    for iteration in range(max_newton):
        # Slacks for cut constraints: s_k = q_k^T w  (must be > 0)
        if t > 0:
            cut_slacks = cuts.T @ w                     # (t,)
        # Slacks for box constraints
        upper_slack = R - w                             # (N,)
        lower_slack = R + w                             # (N,)

        # --- Gradient ---
        # g = -sum_s q_s / (q_s^T w)  +  1/(R-w) - 1/(R+w)
        g = 1.0 / upper_slack - 1.0 / lower_slack      # box part
        if t > 0:
            g -= cuts @ (1.0 / cut_slacks)              # cut part

        # --- Hessian ---
        # H = sum_s q_s q_s^T / (q_s^T w)^2  +  diag(box)
        # Box diagonal: 1/(R+w_j)^2 + 1/(R-w_j)^2
        H_diag = 1.0 / lower_slack**2 + 1.0 / upper_slack**2  # (N,)

        # --- Newton step via Woodbury ---
        # H = diag(D) + Q_s @ Q_s^T  where D = H_diag, Q_s = cuts/slacks
        # H^{-1} g = D^{-1} g - D^{-1} Q_s (I + Q_s^T D^{-1} Q_s)^{-1} Q_s^T D^{-1} g
        # Cost: O(N t^2) instead of O(N^3) when t << N
        if not np.all(np.isfinite(g)):
            break
        D_inv = 1.0 / np.maximum(H_diag, 1e-15)  # (N,)
        D_inv_g = D_inv * g                        # (N,)

        if t > 0:
            Q_scaled = cuts / cut_slacks[np.newaxis, :]  # (N, t)
            if not np.all(np.isfinite(Q_scaled)):
                break
            D_inv_Q = D_inv[:, np.newaxis] * Q_scaled      # (N, t)
            # M_small = I_t + Q_s^T D^{-1} Q_s  -- (t, t) matrix
            M_small = np.eye(t) + Q_scaled.T @ D_inv_Q     # (t, t)
            try:
                L_small, low = cho_factor(M_small, check_finite=False)
                # Woodbury: H^{-1} g = D^{-1} g - D^{-1} Q (M^{-1}) Q^T D^{-1} g
                v = Q_scaled.T @ D_inv_g                    # (t,)
                v = cho_solve((L_small, low), v, check_finite=False)  # (t,)
                dw = -(D_inv_g - D_inv_Q @ v)
            except np.linalg.LinAlgError:
                dw = -D_inv_g
        else:
            dw = -D_inv_g

        # Newton decrement: lambda^2 = -g^T dw
        newton_dec_sq = float(-g @ dw)
        if newton_dec_sq < 0:
            # Numerical issue -- g and dw not aligned
            break
        if newton_dec_sq / 2 < tol:
            break

        # --- Backtracking line search ---
        step = 1.0
        phi_0 = _barrier_value(w, cuts, t, R)
        accepted = False
        for _ in range(50):
            w_new = w + step * dw
            # Check strict feasibility
            if np.all(w_new > -R + 1e-12) and np.all(w_new < R - 1e-12):
                if t == 0 or np.all(cuts.T @ w_new > 1e-12):
                    phi_new = _barrier_value(w_new, cuts, t, R)
                    if np.isfinite(phi_new) and phi_new <= phi_0 + alpha * step * (g @ dw):
                        accepted = True
                        break
            step *= beta
        if not accepted:
            break  # Newton stalled, return current w

        w = w + step * dw

    return w


def _find_strictly_feasible(
    cuts: NDArray[np.float64],
    R: float,
) -> NDArray[np.float64]:
    """Find a strictly interior point by averaging cut normals.

    Fast O(N*t) heuristic: move from the origin along the average
    direction of all cuts, scaled to stay inside the box.
    Falls back to solving a Chebyshev LP if the heuristic fails.
    """
    N, t = cuts.shape
    # Heuristic: average cut direction, scaled
    direction = cuts.sum(axis=1)  # (N,)
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-15:
        w = direction / d_norm * R * 0.5
        w = np.clip(w, -R * 0.9, R * 0.9)
        slacks = cuts.T @ w
        if np.min(slacks) > 1e-10:
            return w

    # Iterative feasibility push (cheap, no LP)
    w = np.zeros(N)
    for _ in range(200):
        slacks = cuts.T @ w
        min_s = np.min(slacks)
        if min_s > 1e-8:
            return w
        worst = int(np.argmin(slacks))
        q = cuts[:, worst]
        step = (0.01 * R - slacks[worst]) / (q @ q + 1e-15)
        w = w + step * q
        w = np.clip(w, -R * 0.95, R * 0.95)

    return w


def _barrier_value(
    w: NDArray[np.float64],
    cuts: NDArray[np.float64],
    t: int,
    R: float,
) -> float:
    """Evaluate the log-barrier phi(w)."""
    val = -np.sum(np.log(R + w)) - np.sum(np.log(R - w))
    if t > 0:
        cut_slacks = cuts.T @ w
        if np.any(cut_slacks <= 0):
            return np.inf
        val -= np.sum(np.log(cut_slacks))
    return val


def compute_relevance(
    w: NDArray[np.float64],
    cuts: NDArray[np.float64],
    H_inv_diag: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Relevance score for each cut (lower = more relevant).

    Boyd's irrelevance measure (Section 3):
        rho_s = (q_s^T w) / ||H^{-1/2} q_s||

    Cuts with large rho are redundant and can be pruned.
    Simplified: use slack as the relevance score (cheap, effective).
    """
    slacks = cuts.T @ w  # (t,)
    return slacks


# =====================================================================
# Cutting-plane algorithm (Algorithm 5)
# =====================================================================


@dataclass
class CuttingPlaneResult:
    """Results of running the cutting-plane algorithm."""

    T: int
    cumulative_regret: NDArray[np.float64]      # shape (T,) -- platform compliance regret
    player_swap_regret: NDArray[np.float64]     # shape (T, n) -- per-player swap regret
    w_error: NDArray[np.float64]                # ||w(t)-w*||/||w*||, shape (T,)
    n_deviations: int                           # total deviations observed
    n_cuts: int                                 # total cuts accumulated
    timings: dict = field(default_factory=dict)


class CuttingPlaneAlgorithm:
    """Algorithm 5: Low-regret recommendations via cutting-plane.

    At each round:
        1. Compute w(t) = analytic center of knowledge set
        2. Query oracle: CE(w(t)) -> recommend -> observe -> cut
        3. If deviation: add cut, update knowledge set
        4. If no deviation: reuse x(t), zero regret

    Paper: Section 5, Algorithm 5.
    """

    def __init__(
        self,
        oracle: SeparationOracle,
        box_radius: float = 1.0,
        max_cuts: int | None = None,
        newton_tol: float = 1e-8,
        max_newton: int = 50,
    ):
        """
        Parameters
        ----------
        oracle : the separation oracle (guess -> CE -> interact -> cut)
        box_radius : R, half-width of the box constraint
        max_cuts : max constraints to keep (pruning). None = 5*N.
        newton_tol : convergence tolerance for Newton
        max_newton : max Newton iterations per center computation
        """
        self.oracle = oracle
        self.spec = oracle.spec
        self.N = self.spec.N
        self.R = box_radius
        self.max_cuts = max_cuts if max_cuts is not None else 5 * self.N
        self.newton_tol = newton_tol
        self.max_newton = max_newton

    def run(
        self,
        T: int,
        w_star: NDArray[np.float64] | None = None,
        verbose: bool = False,
    ) -> CuttingPlaneResult:
        """Run the cutting-plane algorithm for T rounds.

        Parameters
        ----------
        T : number of rounds
        w_star : true w-vector (for tracking convergence, optional)
        verbose : print progress

        Returns
        -------
        CuttingPlaneResult
        """
        N = self.N
        n = self.spec.n
        action_counts = self.spec.action_counts

        # Storage
        cum_regret = np.zeros(T)
        w_error = np.zeros(T)
        cuts_matrix = np.empty((N, 0), dtype=np.float64)
        w = np.zeros(N)
        w_norm = float(np.linalg.norm(w_star)) if w_star is not None else 1.0

        # Swap regret tracking: for each player i, record
        # u_i(a_i(t), a_{-i}(t)) for each action a_i(t) actually played,
        # and accumulate utility per (recommended -> played) swap pair
        # swap_utility[i][a, b] = sum_t u_i(b, a_{-i}(t)) * 1[a_i(t) == a]
        # actual_utility[i] = sum_t u_i(a_i(t), a_{-i}(t))
        swap_gains = [[np.zeros(action_counts[i]) for _ in range(action_counts[i])]
                      for i in range(n)]
        actual_utility = [0.0 for _ in range(n)]
        player_swap_regret = np.zeros((T, n))

        n_deviations = 0
        update = True

        t_center = 0.0
        t_oracle = 0.0

        for t in range(T):
            # --- Compute query point ---
            if update and cuts_matrix.shape[1] > 0:
                t0 = perf_counter()
                w = analytic_center(
                    cuts_matrix, w,
                    box_radius=self.R,
                    max_newton=self.max_newton,
                    tol=self.newton_tol,
                )
                t_center += perf_counter() - t0

            w = np.clip(w, -self.R * 0.99, self.R * 0.99)

            # --- Query oracle ---
            t0 = perf_counter()
            try:
                result = self.oracle.query(w)
            except RuntimeError:
                w = np.zeros(N)
                result = self.oracle.query(w)
            t_oracle += perf_counter() - t0

            # --- Swap regret tracking ---
            # a(t) = recommendation played by compliant agents
            # a*(t) = what agents actually did (= a(t) under compliance,
            #         but a*(t) if deviated)
            # Under γ-BR compliance: agents follow, so played = recommended
            a_played = result.recommendation
            a_others = result.response  # what actually happened

            game = self.oracle.game
            for i in range(n):
                a_i = a_played[i]
                # Utility of each alternative action given what others played
                for b in range(action_counts[i]):
                    # u_i(b, a_{-i}(t)) where a_{-i} is the actual actions
                    profile = list(a_others)
                    profile[i] = b
                    u_b = game.utilities[i][tuple(profile)]
                    swap_gains[i][a_i][b] += u_b

                u_actual = game.utilities[i][tuple(a_others)]
                actual_utility[i] += u_actual

            # Per-player swap regret at time t:
            # R_i(t) = max_psi sum_{s<=t} [u_i(psi(a_i(s)), a_{-i}(s)) - u_i(a_i(s), a_{-i}(s))]
            # = max over all swap functions psi : sum_a sum_{s: a_i(s)=a} u_i(psi(a), a_{-i}(s)) - actual
            # = sum_a max_b swap_gains[i][a][b] - actual_utility[i]
            for i in range(n):
                sr = sum(np.max(swap_gains[i][a]) for a in range(action_counts[i])) \
                     - actual_utility[i]
                player_swap_regret[t, i] = max(sr, 0.0)

            # --- Process result ---
            if result.deviated:
                n_deviations += 1
                q = result.cut
                q_norm = np.linalg.norm(q)
                if q_norm > 1e-15:
                    q = q / q_norm
                cuts_matrix = np.column_stack([cuts_matrix, q])
                if cuts_matrix.shape[1] > self.max_cuts:
                    cuts_matrix, w = self._prune(cuts_matrix, w)
                update = True
            else:
                update = False

            # --- Log platform regret ---
            regret_t = result.regret if result.deviated else 0.0
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0.0) + regret_t
            if w_star is not None:
                w_error[t] = float(np.linalg.norm(w - w_star)) / max(w_norm, 1e-15)

            if verbose and (t + 1) % max(1, T // 20) == 0:
                avg_sr = np.mean(player_swap_regret[t])
                print(
                    f"  t={t+1:5d}  cuts={cuts_matrix.shape[1]:4d}  "
                    f"cum_regret={cum_regret[t]:.4f}  "
                    f"avg_swap_regret={avg_sr:.4f}",
                    flush=True,
                )

        return CuttingPlaneResult(
            T=T,
            cumulative_regret=cum_regret,
            player_swap_regret=player_swap_regret,
            w_error=w_error,
            n_deviations=n_deviations,
            n_cuts=cuts_matrix.shape[1],
            timings={
                "center_total": t_center,
                "oracle_total": t_oracle,
                "center_per_call": t_center / max(n_deviations, 1),
                "oracle_per_call": t_oracle / T,
            },
        )

    def _prune(
        self,
        cuts_matrix: NDArray[np.float64],
        w: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Drop the least relevant cuts (largest slack = most redundant).

        Boyd, Section 3: keep constraints with smallest slack at the
        analytic center. Drop the rest.
        """
        slacks = cuts_matrix.T @ w  # (t,)
        # Keep the max_cuts cuts with smallest slack (most active)
        keep_idx = np.argsort(slacks)[: self.max_cuts]
        return cuts_matrix[:, keep_idx], w
