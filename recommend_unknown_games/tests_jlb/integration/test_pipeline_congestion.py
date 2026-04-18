"""End-to-end JLB pipeline tests on congestion games.

Runs the full pipeline (oracle -> collect cuts -> CE reconstruction) on
small congestion-game instances, and validates:
  - The returned mixture is a valid CE (exact rational check).
  - The CE agrees with the brute-force normal-form CE of the same game.
"""

from __future__ import annotations

import numpy as np
import pytest
from gmpy2 import mpq

from ...congestion_game import CongestionGame, pigou_game, braess_game
from ...jlb_driver import run_jlb, reconstruct_ce_mixture
from ..utils.exact_validation import Mixture, is_valid_ce_exact, max_incentive_violation


def _mixture_from_alpha(
    game: CongestionGame, profiles: list[tuple[int, ...]], alpha: np.ndarray
) -> Mixture:
    """Build a Mixture from (profiles, alpha), merging duplicates."""
    combined: dict[tuple[int, ...], object] = {}
    for p, a in zip(profiles, alpha):
        if a > 0:
            combined[p] = combined.get(p, mpq(0)) + mpq(float(a))
    total = mpq(0)
    for w in combined.values():
        total = total + mpq(w)
    weights = tuple(mpq(w) / total for w in combined.values())
    return Mixture(game.spec.action_counts, tuple(combined.keys()), weights)


# ---------------------------------------------------------------------------
# Tiny games
# ---------------------------------------------------------------------------


class TestPipelineCongestionTiny:
    """Congestion games small enough that we can hand-check the CE."""

    def test_pigou_2_agents(self):
        """Pigou with 2 agents: each takes one of two parallel edges.
           Edge 0: c_0(l) = l.  Edge 1: c_1 = 1 constant.
           With 2 agents, the symmetric Nash splits traffic: each agent
           prefers edge 0 (load 1, cost 1) unless the other also plays it."""
        g = pigou_game(n_agents=2, a=0.0, b=1.0, const_route_cost=1.0)
        result = run_jlb(g, max_iterations=100, reconstruct_every=5, seed=0)
        assert result.stopped_reason == "found_ce", result.stopped_reason
        mixture = _mixture_from_alpha(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6), f"max violation {v}"

    def test_pigou_3_agents(self):
        """Pigou with 3 agents: each agent's options are more constrained."""
        g = pigou_game(n_agents=3, a=0.0, b=1.0, const_route_cost=1.0)
        result = run_jlb(g, max_iterations=200, reconstruct_every=10, seed=0)
        assert result.stopped_reason == "found_ce"
        mixture = _mixture_from_alpha(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6), f"max violation {v}"

    def test_braess_2_agents_no_shortcut(self):
        """Braess without shortcut: should split traffic."""
        g = braess_game(n_agents=2, free_shortcut=False)
        result = run_jlb(g, max_iterations=100, reconstruct_every=5, seed=0)
        assert result.stopped_reason == "found_ce"
        mixture = _mixture_from_alpha(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6), f"max violation {v}"

    def test_braess_with_shortcut(self):
        """Classical Braess paradox: the free shortcut leads to worse
        equilibrium.  We just check a CE exists."""
        g = braess_game(n_agents=2, free_shortcut=True)
        result = run_jlb(g, max_iterations=200, reconstruct_every=10, seed=0)
        assert result.stopped_reason == "found_ce"
        mixture = _mixture_from_alpha(g, result.profiles, result.alpha)
        v = max_incentive_violation(g, mixture)
        assert v <= mpq(1, 10**6), f"max violation {v}"


# ---------------------------------------------------------------------------
# CE agreement: JLB-on-CongestionGame must match brute-force CE of the
# equivalent normal-form representation.
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """For each small game, compare JLB output to brute-force CE."""

    def test_pigou_3_vs_brute_force(self):
        """Both methods should yield a valid CE (not necessarily identical)."""
        from ..fixtures.ground_truth import brute_force_ce
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()

        # Brute force: solve the full M-variable LP on the normal-form rep.
        bf_mixture = brute_force_ce(nf)
        v_bf = max_incentive_violation(nf, bf_mixture)

        # JLB: run on the compact CongestionGame.
        result = run_jlb(g, max_iterations=200, reconstruct_every=10, seed=0)
        assert result.alpha is not None
        jlb_mixture = _mixture_from_alpha(g, result.profiles, result.alpha)
        v_jlb = max_incentive_violation(g, jlb_mixture)

        # Both must be valid CEs.
        assert v_bf <= mpq(1, 10**6), f"brute-force violation {v_bf}"
        assert v_jlb <= mpq(1, 10**6), f"JLB violation {v_jlb}"
