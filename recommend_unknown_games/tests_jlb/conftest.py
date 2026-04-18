"""Shared fixtures and configuration for the JLB test suite."""

from __future__ import annotations

import pytest

from .fixtures.benchmark_games import (
    prisoners_dilemma,
    matching_pennies,
    shapley_game,
    chicken_game,
    theorem5_counterexample_u,
    theorem5_counterexample_v,
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: scaling or long-running tests")
    config.addinivalue_line("markers", "requires_scip: tests needing PySCIPOpt / SCIP 10")
    config.addinivalue_line("markers", "exact: tests using rational arithmetic (gmpy2)")


@pytest.fixture
def game_pd():
    return prisoners_dilemma()


@pytest.fixture
def game_mp():
    return matching_pennies()


@pytest.fixture
def game_shapley():
    return shapley_game()


@pytest.fixture
def game_chicken():
    return chicken_game()


@pytest.fixture
def game_thm5_u():
    return theorem5_counterexample_u()


@pytest.fixture
def game_thm5_v():
    return theorem5_counterexample_v()
