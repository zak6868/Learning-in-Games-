"""Moderator end-to-end harness: outer ellipsoid cutting-plane in θ-space,
inner JLB oracle for CE of the guessed game, BR / γ-BR feedback on the
true game.

Per-round metrics logged (no cherry-picking):
    theta_t                         (d,)
    recommendation_t                (n,)
    response_t                      (n,)
    n_deviated_t                    int
    per_agent_phi_t                 (n,)
    max_phi_t, mean_phi_t           floats
    regret_t                        float  (Σ φ_i · 1[deviated_i])
    cut_norm_t                      float  (‖q‖₂, 0 if no dev)
    inner_solver_time_t             float
    ce_cached_t                     bool
    mixture_size_t                  int
    round_wall_time_t               float

Termination:
    - max_rounds reached, or
    - `stop_after_no_dev` consecutive rounds with zero deviations (default 10).

Final metrics:
    final_theta
    verify_ce: {is_ce, max_violation, n_violated, worst_violation}
    total_rounds, last_dev_round, total_wall_time

Saved as a single .npz file per run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from recommend_unknown_games.compact_feedback import CompactAgentFeedback
from recommend_unknown_games.compact_game import CompactGame
from recommend_unknown_games.ellipsoid import Ellipsoid
from recommend_unknown_games.jlb_ellipsoid import run_jlb_ellipsoid
from recommend_unknown_games.jlb_driver import run_jlb

from .environments.base_fm import FeatureMap
from .feedback import make_feedback, per_agent_phi


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------


def verify_ce(
    game: CompactGame,
    profiles: list[tuple[int, ...]],
    alpha: NDArray[np.float64],
    tol: float = 1e-6,
) -> dict:
    """Check every (i, a_i, a'_i) CE constraint on the given mixture."""
    spec = game.spec
    alpha = np.asarray(alpha, dtype=np.float64)
    max_v = -np.inf
    worst = (-1, -1, -1)
    n_violated = 0
    for i in range(spec.n):
        m_i = spec.action_counts[i]
        for a_i in range(m_i):
            for a_prime in range(m_i):
                if a_prime == a_i:
                    continue
                gain = 0.0
                for profile, w in zip(profiles, alpha):
                    if w <= 0 or profile[i] != a_i:
                        continue
                    prof_dev = list(profile)
                    prof_dev[i] = a_prime
                    u_h = game.utility(i, tuple(profile))
                    u_d = game.utility(i, tuple(prof_dev))
                    gain += float(w) * (u_d - u_h)
                if gain > max_v:
                    max_v = gain
                    worst = (i, a_i, a_prime)
                if gain > tol:
                    n_violated += 1
    return {
        "is_ce": bool(max_v <= tol),
        "max_violation": float(max_v),
        "n_violated": int(n_violated),
        "worst_violation": tuple(int(x) for x in worst),
    }


# ---------------------------------------------------------------------
# Harness config
# ---------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    # Identifiers
    label: str                                  # e.g. "pigou_n50_BR_seed0"
    # Problem
    feature_map: FeatureMap
    # Moderator box (θ-space): [box_lower, box_upper]^d
    box_radius: float = 10.0                    # upper bound
    box_lower: float = 0.0                      # lower bound (θ ≥ 0)
    # Outer loop
    outer: str = "ellipsoid"                    # "ellipsoid" | "accpm"
    max_rounds: int = 120
    stop_after_no_dev: int = 10
    # Feedback
    feedback_kind: str = "BR"                   # "BR" | "gamma_BR"
    gamma: float = 0.0
    # Inner JLB
    jlb_max_iters: int = 100_000
    jlb_initial_radius: float = 10.0
    jlb_reconstruct_every: int = 100
    use_structured_y_fallback: bool = True
    # Determinism
    seed: int = 0


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    per_round: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)

    def to_npz(self, path: str) -> None:
        per_round_keys = list(self.per_round[0].keys()) if self.per_round else []
        arrays = {}
        for k in per_round_keys:
            vals = [r[k] for r in self.per_round]
            try:
                arrays[f"pr_{k}"] = np.array(vals, dtype=float)
            except (TypeError, ValueError):
                # Store object arrays (e.g. tuples) flat
                arrays[f"pr_{k}"] = np.array(vals, dtype=object)
        for k, v in self.final.items():
            arrays[f"final_{k}"] = np.array(v) if not isinstance(v, dict) \
                else np.array(list(v.items()), dtype=object)
        cfg_items = {
            "label": self.config.label,
            "box_radius": self.config.box_radius,
            "box_lower": self.config.box_lower,
            "outer": self.config.outer,
            "max_rounds": self.config.max_rounds,
            "stop_after_no_dev": self.config.stop_after_no_dev,
            "feedback_kind": self.config.feedback_kind,
            "gamma": self.config.gamma,
            "seed": self.config.seed,
            "d": self.config.feature_map.d,
            "n": self.config.feature_map.spec.n,
            "action_counts": self.config.feature_map.spec.action_counts,
        }
        arrays["cfg"] = np.array(list(cfg_items.items()), dtype=object)
        np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------


def _solve_inner_ce(
    game: CompactGame,
    jlb_max_iters: int,
    jlb_initial_radius: float,
    jlb_reconstruct_every: int,
    seed: int,
    use_structured_y_fallback: bool,
) -> tuple[Optional[list[tuple[int, ...]]], Optional[NDArray[np.float64]]]:
    res = run_jlb_ellipsoid(
        game,
        max_iterations=jlb_max_iters,
        initial_radius=jlb_initial_radius,
        reconstruct_every=jlb_reconstruct_every,
        seed=seed,
    )
    profiles = res.profiles
    alpha = res.alpha
    if alpha is None and use_structured_y_fallback:
        fb = run_jlb(
            game, max_iterations=min(500, jlb_max_iters),
            reconstruct_every=jlb_reconstruct_every, seed=seed,
        )
        if fb.alpha is not None:
            profiles, alpha = fb.profiles, fb.alpha
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=np.float64)
        alpha = np.maximum(alpha, 0.0)
        s = alpha.sum()
        if s > 0:
            alpha = alpha / s
        else:
            alpha = None
    return (profiles, alpha)


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    fm = cfg.feature_map
    d = fm.d
    n = fm.spec.n
    rng = np.random.default_rng(cfg.seed)

    true_game = fm.game_at_theta(fm.theta_true())
    feedback = make_feedback(
        true_game, kind=cfg.feedback_kind,
        gamma=cfg.gamma, seed=cfg.seed,
    )

    # Outer state
    center = np.full(d, 0.5 * (cfg.box_lower + cfg.box_radius),
                     dtype=np.float64)
    if cfg.outer == "ellipsoid":
        half = 0.5 * (cfg.box_radius - cfg.box_lower)
        ellipsoid = Ellipsoid.ball(d, half, center=center)
    elif cfg.outer == "accpm":
        cuts_matrix = np.empty((d, 0), dtype=np.float64)
        theta_cur = center.copy()
    else:
        raise ValueError(cfg.outer)

    # Caches
    last_theta: Optional[NDArray[np.float64]] = None
    last_profiles: Optional[list[tuple[int, ...]]] = None
    last_alpha: Optional[NDArray[np.float64]] = None

    result = ExperimentResult(config=cfg)
    rounds_no_dev = 0
    cum_regret = 0.0
    t_wall0 = time.time()
    last_dev_round = -1

    for t in range(cfg.max_rounds):
        t_round_start = time.time()
        # 1) θ_t = centre of current feasible set
        if cfg.outer == "ellipsoid":
            theta_t = ellipsoid.c.copy()
        else:
            if cuts_matrix.shape[1] > 0:
                from recommend_unknown_games.cutting_plane import analytic_center
                theta_cur = analytic_center(
                    cuts_matrix, theta_cur,
                    box_radius=cfg.box_radius, box_lower=cfg.box_lower,
                    max_newton=50, tol=1e-8,
                )
            theta_t = theta_cur.copy()
        theta_t = np.clip(theta_t, cfg.box_lower + 1e-9, cfg.box_radius - 1e-9)

        # 2) Build game at θ_t and solve inner CE (or reuse cache)
        cached = (
            last_theta is not None
            and last_profiles is not None
            and last_alpha is not None
            and np.array_equal(last_theta, theta_t)
        )
        t0 = time.time()
        if cached:
            profiles, alpha = last_profiles, last_alpha
        else:
            game_at_t = fm.game_at_theta(theta_t)
            profiles, alpha = _solve_inner_ce(
                game_at_t, cfg.jlb_max_iters,
                cfg.jlb_initial_radius, cfg.jlb_reconstruct_every,
                cfg.seed + t,
                cfg.use_structured_y_fallback,
            )
            last_theta = theta_t.copy()
            last_profiles, last_alpha = profiles, alpha
        t_inner = time.time() - t0

        if alpha is None:
            # Oracle failure: reset θ to box center + perturbation, skip cut.
            center = np.linspace(cfg.box_lower + 0.1,
                                 cfg.box_radius - 0.1, d)
            if cfg.outer == "ellipsoid":
                ellipsoid = Ellipsoid.ball(
                    d, 0.5 * (cfg.box_radius - cfg.box_lower),
                    center=center)
            else:
                theta_cur = center.copy()
            last_theta = None
            continue

        # 3) Sample recommendation from mixture
        k_idx = int(rng.choice(len(profiles), p=alpha))
        recommendation = tuple(int(x) for x in profiles[k_idx])

        # 4) Query true-game BR/γBR feedback
        response = feedback.respond_all(recommendation, profiles, alpha)
        response = tuple(int(x) for x in response)

        # 5) Per-agent φ under TRUE game
        phi = per_agent_phi(true_game, recommendation, profiles, alpha)
        n_deviated = int(sum(
            1 for i in range(n) if recommendation[i] != response[i]
        ))
        max_phi = float(phi.max()) if n > 0 else 0.0
        mean_phi = float(phi.mean()) if n > 0 else 0.0

        # 6) Build cut and apply (only if a deviation)
        cut_norm = 0.0
        if n_deviated > 0:
            q = fm.build_cut_theta(
                recommendation, response, profiles, alpha
            )
            cut_norm = float(np.linalg.norm(q))
            # Separation: want θ* in {θ : <θ, q> ≥ 0} so remove the
            # halfspace {θ : <θ, q> < 0}, equivalently <-q, θ> > 0.
            if cut_norm > 1e-15:
                q_norm = q / cut_norm
                if cfg.outer == "ellipsoid":
                    ellipsoid.apply_cut(-q_norm, 0.0)
                else:
                    cuts_matrix = np.column_stack([cuts_matrix, q_norm])

        # 7) Regret: sum of φ over deviating agents
        regret_t = float(sum(
            phi[i] for i in range(n)
            if recommendation[i] != response[i]
        ))
        cum_regret += regret_t

        # 8) Record
        result.per_round.append({
            "theta": theta_t,
            "recommendation": recommendation,
            "response": response,
            "n_deviated": n_deviated,
            "per_agent_phi": phi,
            "max_phi": max_phi,
            "mean_phi": mean_phi,
            "regret": regret_t,
            "cum_regret": cum_regret,
            "cut_norm": cut_norm,
            "inner_solver_time": t_inner,
            "ce_cached": cached,
            "mixture_size": len(profiles),
            "round_wall_time": time.time() - t_round_start,
        })

        if n_deviated > 0:
            last_dev_round = t
            rounds_no_dev = 0
        else:
            rounds_no_dev += 1
            if rounds_no_dev >= cfg.stop_after_no_dev:
                break

    # Final verify_ce on the most recent CE mixture
    final_verify = None
    if last_profiles is not None and last_alpha is not None:
        # Check the CE on the TRUE game (the game that matters).
        final_verify = verify_ce(
            true_game, last_profiles, last_alpha, tol=1e-6,
        )

    result.final = {
        "final_theta": theta_t,
        "verify_ce": final_verify,
        "total_rounds": len(result.per_round),
        "last_dev_round": last_dev_round,
        "total_wall_time": time.time() - t_wall0,
    }
    return result


__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment", "verify_ce"]
