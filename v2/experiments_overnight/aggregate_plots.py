"""Aggregate plots across seeds.

For each (game_family, n, m, feedback, gamma) configuration, collect
all matching .npz runs and plot mean ± 1 std (shaded) vs round t for
the four core metrics:

  - cumulative regret
  - number of deviating agents per round
  - max single-agent φ_i
  - inner-solver wall time

Handles ragged trajectories (different runs may terminate at different
T) by padding to the maximum T observed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_SEED_RE = re.compile(r"_s\d+$")


def _config_key(label: str) -> str:
    """Strip the `_s{seed}` suffix to identify seed-equivalent runs."""
    return _SEED_RE.sub("", label)


def _load_pr_arrays(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for key in ("pr_cum_regret", "pr_n_deviated", "pr_max_phi",
                "pr_inner_solver_time", "pr_mean_phi"):
        if key in data.files:
            out[key] = np.asarray(data[key], dtype=np.float64)
    # cfg
    cfg = dict(data["cfg"].tolist()) if "cfg" in data.files else {}
    out["cfg"] = cfg
    return out


def _pad(arrays: list[np.ndarray]) -> np.ndarray:
    """Pad 1-D arrays with their last value to the max length, stack."""
    if not arrays:
        return np.zeros((0, 0))
    T_max = max(len(a) for a in arrays)
    out = np.zeros((len(arrays), T_max), dtype=np.float64)
    for i, a in enumerate(arrays):
        L = len(a)
        out[i, :L] = a
        if L < T_max:
            out[i, L:] = a[-1] if L > 0 else 0.0
    return out


def _plot_one(
    ax: plt.Axes,
    data_per_seed: np.ndarray,   # shape (n_seeds, T)
    ylabel: str,
    title: str,
    log_y: bool = False,
    kind: str = "line",          # 'line' or 'bar-mean'
) -> None:
    if data_per_seed.size == 0:
        return
    T = data_per_seed.shape[1]
    t_axis = np.arange(T)
    mean = data_per_seed.mean(axis=0)
    std = data_per_seed.std(axis=0)
    if kind == "line":
        ax.plot(t_axis, mean, "-", linewidth=1.8, color="tab:blue")
        ax.fill_between(t_axis, mean - std, mean + std,
                        color="tab:blue", alpha=0.22)
    else:
        ax.bar(t_axis, mean, yerr=std, capsize=2,
               color="tab:red", alpha=0.75, width=0.9)
    ax.set_xlabel("round t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def aggregate_and_plot(
    npz_dir: str = "experiments_overnight/out",
    outdir: str = "experiments_overnight/out/aggregated",
) -> None:
    npz_paths = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_paths:
        print(f"no .npz files in {npz_dir}")
        return
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Group by (stripped label, i.e. config signature).
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in npz_paths:
        label = p.stem
        key = _config_key(label)
        try:
            arrs = _load_pr_arrays(str(p))
        except Exception:
            continue
        groups[key].append(arrs)

    for key, runs in sorted(groups.items()):
        n_seeds = len(runs)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        cum_regret = _pad([r["pr_cum_regret"] for r in runs
                            if "pr_cum_regret" in r])
        n_dev = _pad([r["pr_n_deviated"] for r in runs
                       if "pr_n_deviated" in r])
        max_phi = _pad([r["pr_max_phi"] for r in runs
                         if "pr_max_phi" in r])
        inner_t = _pad([r["pr_inner_solver_time"] for r in runs
                         if "pr_inner_solver_time" in r])

        _plot_one(axes[0, 0], cum_regret,
                  ylabel=r"cumulative regret $\sum_{s\leq t} r_s$",
                  title="cumulative regret  (mean ± std)", kind="line")
        _plot_one(axes[0, 1], n_dev,
                  ylabel="# deviating agents",
                  title="deviations per round  (mean ± std)", kind="bar-mean")
        _plot_one(axes[1, 0], max_phi,
                  ylabel=r"$\max_i \varphi_i(t)$",
                  title=r"max single-agent $\varphi$  (mean ± std)",
                  kind="line")
        _plot_one(axes[1, 1], inner_t,
                  ylabel="inner-solver time (s)",
                  title="per-round inner JLB time  (mean ± std)",
                  kind="line", log_y=True)

        fig.suptitle(f"{key}   [{n_seeds} seeds]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(outdir / f"{key}_agg.png", dpi=130)
        plt.close(fig)
        print(f"wrote {outdir / (key + '_agg.png')}  ({n_seeds} seeds)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="experiments_overnight/out")
    ap.add_argument("--outdir", default="experiments_overnight/out/aggregated")
    args = ap.parse_args()
    aggregate_and_plot(args.npz_dir, args.outdir)
