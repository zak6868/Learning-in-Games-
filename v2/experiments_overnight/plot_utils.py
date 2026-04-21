"""Plot utilities for ExperimentResult.

Four plots per run:
  1. cumulative regret vs round t
  2. number of deviating agents per round t
  3. max single-agent φ per round t (ε-CE witness)
  4. inner-solver wall time per round t
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .harness import ExperimentResult


def _arr(result: ExperimentResult, key: str) -> np.ndarray:
    return np.array([r[key] for r in result.per_round])


def make_plots(result: ExperimentResult, outdir: str) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    T = len(result.per_round)
    if T == 0:
        return
    t_axis = np.arange(T)
    label = result.config.label

    # 1. cumulative regret
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cum = _arr(result, "cum_regret")
    ax.plot(t_axis, cum, "-", linewidth=1.8, color="tab:blue", marker="o",
            markersize=3)
    ax.set_xlabel("round t")
    ax.set_ylabel(r"cumulative regret  $\sum_{s \leq t}\, r_s$")
    ax.set_title(f"{label} — cumulative regret")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_regret.png", dpi=130)
    plt.close(fig)

    # 2. number of deviating agents
    fig, ax = plt.subplots(figsize=(8, 4))
    n_dev = _arr(result, "n_deviated")
    ax.bar(t_axis, n_dev, color="tab:red", alpha=0.75, width=0.9)
    ax.set_xlabel("round t")
    ax.set_ylabel("# deviating agents")
    ax.set_title(f"{label} — deviating agents per round")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_deviations.png", dpi=130)
    plt.close(fig)

    # 3. max single-agent φ
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_phi = _arr(result, "max_phi")
    ax.plot(t_axis, max_phi, "-", linewidth=1.8, color="tab:orange",
            marker="o", markersize=3)
    ax.set_xlabel("round t")
    ax.set_ylabel(r"$\max_i \varphi_i(t)$")
    ax.set_title(f"{label} — largest single-agent deviation incentive")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_maxphi.png", dpi=130)
    plt.close(fig)

    # 4. inner solver time
    fig, ax = plt.subplots(figsize=(8, 4))
    t_inner = _arr(result, "inner_solver_time")
    cached = _arr(result, "ce_cached").astype(bool)
    ax.plot(t_axis[~cached], t_inner[~cached], "o", markersize=3,
            color="tab:green", label="JLB call")
    ax.plot(t_axis[cached], t_inner[cached], "x", markersize=4,
            color="gray", label="cache hit")
    ax.set_xlabel("round t")
    ax.set_ylabel("inner-solver wall time (s)")
    ax.set_title(f"{label} — per-round inner cost")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_inner_time.png", dpi=130)
    plt.close(fig)


__all__ = ["make_plots"]
