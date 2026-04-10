"""Experiments for NeurIPS paper: regret scaling plots.

Generates 3 figures:
1. Cumulative regret vs T  (shows O(log T) scaling)
2. Cumulative regret vs nM (shows O(nM) scaling at fixed T)
3. BR vs QR comparison
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import perf_counter

from recommend_unknown_games.game import NormalFormGame, AgentFeedback
from recommend_unknown_games.ce_solver import CESolver
from recommend_unknown_games.oracle import SeparationOracle
from recommend_unknown_games.cutting_plane import CuttingPlaneAlgorithm

# ── Style ──
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "figure.figsize": (7, 5),
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

OUT = "/Users/owhadigroup/Desktop/Recommend in Unkown Games/figures"


def run_experiment(action_counts, T, model="best_response", beta=2.0,
                   gamma=0.0, seed=42):
    """Run cutting-plane and return result."""
    game = NormalFormGame.random_game(action_counts, seed=seed)
    solver = CESolver(game.spec, backend="gurobi")
    feedback = AgentFeedback(game, model=model, beta=beta, gamma=gamma, seed=seed)
    oracle = SeparationOracle(game, solver, feedback, seed=seed + 1)
    w_star = game.pack_w()
    algo = CuttingPlaneAlgorithm(oracle, box_radius=1.0)
    t0 = perf_counter()
    result = algo.run(T=T, w_star=w_star)
    elapsed = perf_counter() - t0
    return result, game.spec, elapsed


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Cumulative regret vs T (log scaling)
# ═══════════════════════════════════════════════════════════════

def exp1_regret_vs_T():
    print("=== Exp 1: Cumulative regret vs T ===")

    games = {
        "(2,2)": (2, 2),
        "(3,3)": (3, 3),
        "(3,3,3)": (3, 3, 3),
        "(2,)×5": (2,) * 5,
        "(2,)×7": (2,) * 7,
    }
    T = 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, ac in games.items():
        print(f"  {label} ...", end=" ", flush=True)
        result, spec, elapsed = run_experiment(ac, T)
        nM = spec.n * spec.M
        print(f"done ({elapsed:.1f}s, devs={result.n_deviations})")

        # Left: cumulative regret vs t
        axes[0].plot(np.arange(1, T + 1), result.cumulative_regret,
                     label=f"{label} (nM={nM})")

        # Right: cumulative regret vs log(t)
        axes[1].plot(np.arange(1, T + 1), result.cumulative_regret,
                     label=f"{label} (nM={nM})")

    axes[0].set_xlabel("Round $t$")
    axes[0].set_ylabel("Cumulative regret")
    axes[0].set_title("Cumulative regret vs rounds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Round $t$ (log scale)")
    axes[1].set_ylabel("Cumulative regret")
    axes[1].set_title("Cumulative regret vs rounds (log scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp1_regret_vs_T.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp1_regret_vs_T.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp1_regret_vs_T.pdf\n")


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Final regret vs nM (dimension scaling)
# ═══════════════════════════════════════════════════════════════

def exp2_regret_vs_nM():
    print("=== Exp 2: Regret scaling with game dimension nM ===")

    # Use uniform structure: all (2,)*n for clean scaling
    games = [
        (2,) * 2,   (2,) * 3,   (2,) * 4,   (2,) * 5,
        (2,) * 6,   (2,) * 7,   (2,) * 8,   (2,) * 9,
        (2,) * 10,
    ]
    T = 500

    nMs = []
    final_regrets = []
    labels = []
    n_seeds = 5  # average over seeds for less noise

    for ac in games:
        M = int(np.prod(ac))
        n = len(ac)
        nM = n * M
        print(f"  {str(ac):20s} nM={nM:6d} ...", end=" ", flush=True)
        regs = []
        for s in range(n_seeds):
            result, spec, elapsed = run_experiment(ac, T, seed=42 + s)
            regs.append(result.cumulative_regret[-1])
        avg_reg = np.mean(regs)
        print(f"done (avg_reg={avg_reg:.2f})")
        nMs.append(nM)
        final_regrets.append(avg_reg)
        labels.append(str(ac))

    nMs = np.array(nMs)
    final_regrets = np.array(final_regrets)

    # Sort by nM
    order = np.argsort(nMs)
    nMs = nMs[order]
    final_regrets = final_regrets[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nMs, final_regrets, "o-", color="C0", label="Cumulative regret at $T=500$")

    # Fit: regret ~ c * nM * log(T)
    # Since T is fixed, regret should be ~ c * nM
    # Linear fit on log-log scale
    mask = final_regrets > 0
    if np.sum(mask) > 2:
        log_nM = np.log(nMs[mask])
        log_reg = np.log(final_regrets[mask])
        slope, intercept = np.polyfit(log_nM, log_reg, 1)
        nM_fit = np.linspace(nMs[mask].min(), nMs[mask].max(), 100)
        reg_fit = np.exp(intercept) * nM_fit ** slope
        ax.plot(nM_fit, reg_fit, "--", color="C1",
                label=f"Fit: $O(nM^{{{slope:.2f}}})$")

    ax.set_xlabel("Game dimension $nM$")
    ax.set_ylabel(f"Cumulative regret at $T={T}$")
    ax.set_title("Regret scaling with game dimension")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp2_regret_vs_nM.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp2_regret_vs_nM.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp2_regret_vs_nM.pdf\n")


# ═══════════════════════════════════════════════════════════════
# Experiment 3: BR vs QR comparison
# ═══════════════════════════════════════════════════════════════

def exp3_br_vs_qr():
    print("=== Exp 3: BR vs QR feedback comparison ===")

    games = {
        "(3,3)": (3, 3),
        "(3,3,3)": (3, 3, 3),
        "(2,)×7": (2,) * 7,
    }
    T = 500

    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 5))

    for idx, (label, ac) in enumerate(games.items()):
        ax = axes[idx]

        for model, lstyle, clr in [
            ("best_response", "-", "C0"),
            ("quantal_response", "--", "C1"),
        ]:
            mname = "BR" if model == "best_response" else "QR"
            print(f"  {label} {mname} ...", end=" ", flush=True)
            result, spec, elapsed = run_experiment(ac, T, model=model, beta=2.0)
            print(f"done ({elapsed:.1f}s, devs={result.n_deviations})")

            ax.plot(np.arange(1, T + 1), result.cumulative_regret,
                    linestyle=lstyle, color=clr,
                    label=f"{mname} (devs={result.n_deviations})")

        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Cumulative regret")
        ax.set_title(f"Game {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Best Response vs Quantal Response feedback", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp3_br_vs_qr.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp3_br_vs_qr.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp3_br_vs_qr.pdf\n")


# ═══════════════════════════════════════════════════════════════
# Experiment 4: w-convergence
# ═══════════════════════════════════════════════════════════════

def exp4_w_convergence():
    print("=== Exp 4: Convergence of w(t) to equivalence class ===")

    games = {
        "(3,3)": (3, 3),
        "(2,)×5": (2,) * 5,
        "(2,)×7": (2,) * 7,
    }
    T = 500

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, ac in games.items():
        # BR
        print(f"  {label} BR ...", end=" ", flush=True)
        result_br, spec, elapsed = run_experiment(ac, T, model="best_response")
        print(f"done ({elapsed:.1f}s)")
        axes[0].plot(np.arange(1, T + 1), result_br.w_error,
                     label=f"{label} (N={spec.N})")

        # QR
        print(f"  {label} QR ...", end=" ", flush=True)
        result_qr, spec, elapsed = run_experiment(ac, T, model="quantal_response", beta=2.0)
        print(f"done ({elapsed:.1f}s)")
        axes[1].plot(np.arange(1, T + 1), result_qr.w_error,
                     label=f"{label} (N={spec.N})")

    axes[0].set_xlabel("Round $t$")
    axes[0].set_ylabel("$d(w(t), [w^*])$ / $\\|w^*\\|$")
    axes[0].set_title("BR feedback (not learnable — Thm 5)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Round $t$")
    axes[1].set_ylabel("$d(w(t), [w^*])$ / $\\|w^*\\|$")
    axes[1].set_title("QR feedback (learnable — Thm 4)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Distance to equivalence class: BR vs QR", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp4_w_convergence.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp4_w_convergence.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp4_w_convergence.pdf\n")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Experiment 5: Platform regret vs Player swap regret
# ═══════════════════════════════════════════════════════════════

def exp5_platform_vs_player():
    print("=== Exp 5: Platform compliance regret vs Player swap regret ===")

    games = {
        "(3,3)": (3, 3),
        "(2,)×5": (2,) * 5,
        "(2,)×7": (2,) * 7,
    }
    T = 500
    gamma = 0.01  # γ-BR threshold

    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 5))

    for idx, (label, ac) in enumerate(games.items()):
        ax = axes[idx]
        print(f"  {label} (γ={gamma}) ...", end=" ", flush=True)
        result, spec, elapsed = run_experiment(ac, T, gamma=gamma)
        print(f"done ({elapsed:.1f}s, devs={result.n_deviations})")

        ts = np.arange(1, T + 1)

        # Platform compliance regret
        ax.plot(ts, result.cumulative_regret, "-", color="C0",
                label="Platform regret")

        # Average player swap regret
        avg_swap = np.mean(result.player_swap_regret, axis=1)
        ax.plot(ts, avg_swap, "--", color="C1",
                label="Avg player swap regret")

        # Max player swap regret
        max_swap = np.max(result.player_swap_regret, axis=1)
        ax.plot(ts, max_swap, ":", color="C2",
                label="Max player swap regret")

        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Regret")
        ax.set_title(f"Game {label} ($\\gamma$={gamma})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Platform compliance regret vs Player swap regret ($\\gamma$-BR)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp5_platform_vs_player.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp5_platform_vs_player.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp5_platform_vs_player.pdf\n")


# ═══════════════════════════════════════════════════════════════
# Experiment 6: Effect of γ on swap regret
# ═══════════════════════════════════════════════════════════════

def exp6_gamma_effect():
    print("=== Exp 6: Effect of γ on player swap regret ===")

    ac = (3, 3)
    T = 500
    gammas = [0.0, 0.001, 0.01, 0.05, 0.1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for gamma in gammas:
        print(f"  γ={gamma} ...", end=" ", flush=True)
        result, spec, elapsed = run_experiment(ac, T, gamma=gamma)
        print(f"done (devs={result.n_deviations})")

        ts = np.arange(1, T + 1)
        axes[0].plot(ts, result.cumulative_regret,
                     label=f"$\\gamma$={gamma}")
        axes[1].plot(ts, np.mean(result.player_swap_regret, axis=1),
                     label=f"$\\gamma$={gamma}")

    axes[0].set_xlabel("Round $t$")
    axes[0].set_ylabel("Cumulative regret")
    axes[0].set_title("Platform compliance regret")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Round $t$")
    axes[1].set_ylabel("Average swap regret")
    axes[1].set_title("Player swap regret")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Effect of $\\gamma$ on regrets — Game (3,3)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp6_gamma_effect.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp6_gamma_effect.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp6_gamma_effect.pdf\n")


# ═══════════════════════════════════════════════════════════════
# Experiment 7: Moderator vs No-Moderator (MW baseline)
# ═══════════════════════════════════════════════════════════════

def exp7_moderator_vs_baseline():
    print("=== Exp 7: Moderator (γ-BR) vs No-Moderator (MW) ===")

    from recommend_unknown_games.baselines import run_multiplicative_weights

    games = {
        "(3,3)": (3, 3),
        "(2,)×5": (2,) * 5,
        "(2,)×7": (2,) * 7,
    }
    T = 1000
    gamma = 0.01

    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 5))

    for idx, (label, ac) in enumerate(games.items()):
        ax = axes[idx]

        # --- Moderator ---
        print(f"  {label} Moderator ...", end=" ", flush=True)
        result_mod, spec, elapsed = run_experiment(ac, T, gamma=gamma)
        print(f"done ({elapsed:.1f}s)")

        # --- No moderator (MW) ---
        print(f"  {label} MW baseline ...", end=" ", flush=True)
        game = NormalFormGame.random_game(ac, seed=42)
        result_mw = run_multiplicative_weights(game, T, seed=42)
        print("done")

        ts = np.arange(1, T + 1)

        # Moderator: avg swap regret across players
        mod_sr = np.mean(result_mod.player_swap_regret, axis=1)
        ax.plot(ts, mod_sr, "-", color="C0", linewidth=2,
                label="Moderator (γ-BR)")

        # MW: avg swap regret across players
        mw_sr = np.mean(result_mw.player_swap_regret, axis=1)
        ax.plot(ts, mw_sr, "--", color="C1", linewidth=2,
                label="No moderator (MW)")

        # Reference: sqrt(T)
        ax.plot(ts, np.sqrt(ts) * mw_sr[-1] / np.sqrt(T), ":",
                color="gray", alpha=0.5, label="$O(\\sqrt{T})$ ref")

        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Avg player swap regret")
        ax.set_title(f"Game {label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Player swap regret: Moderator ($\\gamma$-BR) vs Independent MW",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp7_moderator_vs_mw.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/exp7_moderator_vs_mw.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to {OUT}/exp7_moderator_vs_mw.pdf\n")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    print(f"Saving figures to {OUT}/\n")

    exp1_regret_vs_T()
    exp2_regret_vs_nM()
    exp3_br_vs_qr()
    exp4_w_convergence()
    exp5_platform_vs_player()
    exp6_gamma_effect()
    exp7_moderator_vs_baseline()

    print("All experiments done.")
