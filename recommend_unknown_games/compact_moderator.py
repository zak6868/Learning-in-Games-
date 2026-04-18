"""Compact cutting-plane moderator.

Mirrors the paper's Algorithm 5 but in the compact parameter space R^d:
  - State: θ(t) ∈ R^d and a knowledge set K_t = { θ : q_θ^(s) · θ ≥ 0 ∀s } ∩ [-R, R]^d.
  - Each round:
      1. θ(t) = analytic center of K_t.
      2. Query the compact oracle at θ(t).
      3. If deviation, add q_θ to K_t.

Regret, cut history, and θ trajectory are tracked for inspection.

We deliberately REUSE `analytic_center` from cutting_plane.py — the
Newton-with-box-barrier machinery is identical; only the dimension
changes from N to d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .compact_oracle import CompactOracleResult, CompactSeparationOracle
from .cutting_plane import analytic_center
from .feature_map import FeatureMap


@dataclass
class CompactModeratorResult:
    T: int
    cumulative_regret: NDArray[np.float64]          # shape (T,)
    theta_trajectory: NDArray[np.float64]           # shape (T, d)
    cuts: list[NDArray[np.float64]] = field(default_factory=list)
    n_deviations: int = 0
    timings: dict = field(default_factory=dict)


class CompactCuttingPlaneAlgorithm:
    """Paper's Algorithm 5, adapted for the compact parameter space."""

    def __init__(
        self,
        oracle: CompactSeparationOracle,
        feature_map: FeatureMap,
        box_radius: float = 1.0,
        max_cuts: Optional[int] = None,
        newton_tol: float = 1e-8,
        max_newton: int = 50,
    ):
        self.oracle = oracle
        self.fm = feature_map
        self.d = feature_map.d
        self.R = box_radius
        self.max_cuts = max_cuts if max_cuts is not None else 5 * self.d
        self.newton_tol = newton_tol
        self.max_newton = max_newton

    def run(
        self,
        T: int,
        theta_star: Optional[NDArray[np.float64]] = None,
        verbose: bool = False,
    ) -> CompactModeratorResult:
        d = self.d
        cum_regret = np.zeros(T, dtype=np.float64)
        theta_trajectory = np.zeros((T, d), dtype=np.float64)
        cuts: list[NDArray[np.float64]] = []
        cuts_matrix = np.empty((d, 0), dtype=np.float64)
        theta_current = np.zeros(d, dtype=np.float64)
        n_deviations = 0
        update_theta = True

        t_center = 0.0
        t_oracle = 0.0

        for t in range(T):
            # Update the query point θ(t) at the analytic center of the
            # knowledge set.  On the first iteration (no cuts) the center
            # is the origin.
            if update_theta and cuts_matrix.shape[1] > 0:
                c0 = perf_counter()
                theta_current = analytic_center(
                    cuts_matrix, theta_current,
                    box_radius=self.R,
                    max_newton=self.max_newton,
                    tol=self.newton_tol,
                )
                t_center += perf_counter() - c0

            theta_current = np.clip(theta_current, -self.R * 0.99, self.R * 0.99)
            theta_trajectory[t] = theta_current

            # Query the oracle
            o0 = perf_counter()
            try:
                result: CompactOracleResult = self.oracle.query(theta_current)
            except RuntimeError:
                # If the inner CE solver fails, reset θ to origin and retry.
                theta_current = np.zeros(d, dtype=np.float64)
                result = self.oracle.query(theta_current)
            t_oracle += perf_counter() - o0

            if result.deviated:
                n_deviations += 1
                q = result.cut_theta
                q_norm = float(np.linalg.norm(q))
                if q_norm > 1e-15:
                    q = q / q_norm
                cuts.append(q)
                cuts_matrix = np.column_stack([cuts_matrix, q])
                if cuts_matrix.shape[1] > self.max_cuts:
                    cuts_matrix, theta_current = self._prune(
                        cuts_matrix, theta_current
                    )
                update_theta = True
            else:
                update_theta = False

            regret_t = result.regret if result.deviated else 0.0
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0.0) + regret_t

            if verbose and (t + 1) % max(1, T // 20) == 0:
                err = (
                    float(np.linalg.norm(theta_current - theta_star))
                    if theta_star is not None else -1.0
                )
                print(
                    f"  t={t+1:4d}  cuts={cuts_matrix.shape[1]:3d}  "
                    f"cum_regret={cum_regret[t]:.3f}  "
                    f"|θ(t)-θ*|={err:.3f}",
                    flush=True,
                )

        return CompactModeratorResult(
            T=T,
            cumulative_regret=cum_regret,
            theta_trajectory=theta_trajectory,
            cuts=cuts,
            n_deviations=n_deviations,
            timings={
                "center_s": t_center,
                "oracle_s": t_oracle,
                "center_per_cut": t_center / max(n_deviations, 1),
                "oracle_per_call": t_oracle / T,
            },
        )

    def _prune(
        self,
        cuts_matrix: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Drop the cuts with largest slack at the current center."""
        slacks = cuts_matrix.T @ theta
        keep_idx = np.argsort(slacks)[: self.max_cuts]
        return cuts_matrix[:, keep_idx], theta


__all__ = ["CompactCuttingPlaneAlgorithm", "CompactModeratorResult"]
