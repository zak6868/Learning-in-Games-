"""Generate result figures from experiment .npz outputs.

Produces:
  - convergence_main.png : per-round regret across game families on the
    largest instance per family, mean +/- 1 sigma over CE-converged seeds,
    x-axis truncated to the first 20 rounds.
  - scaling.png : rounds-to-convergence T as a function of latent dimension nK.
  - paper_aggregated/<cell>_agg.png : per-cell 4-panel plots (cumulative regret,
    deviations per round, max single-agent phi, per-round inner-solver time).
  - paper_aggregated/routing_polymatrix_combined.png : overlay of routing +
    polymatrix cells across the four metrics.

Usage:
    python3 plot_results.py

Style: legend fontsize 16, axis-label fontsize 20, no per-axes titles.
"""

import os
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "experiments_overnight", "out")
OUT_DIR = os.path.join(HERE, "figures")
AGG_DIR = os.path.join(OUT_DIR, "paper_aggregated")
os.makedirs(AGG_DIR, exist_ok=True)

XLIM = 20

LEGEND_FS = 16
AXIS_FS = 20
TICK_FS = 14


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def is_ce(npz):
    vce = npz["final_verify_ce"]
    try:
        if vce.shape == ():
            v = vce.item()
            return v.get("is_ce", None) if isinstance(v, dict) else None
        for row in vce:
            if row[0] == "is_ce":
                return row[1]
    except Exception:
        return None
    return None


def pad(arrays, target_T=None):
    if not arrays:
        return np.zeros((0, 0))
    T = target_T if target_T else max(len(a) for a in arrays)
    out = np.zeros((len(arrays), T), dtype=np.float64)
    for i, a in enumerate(arrays):
        L = min(len(a), T)
        out[i, :L] = a[:L]
        if L < T:
            out[i, L:] = a[L - 1] if L > 0 else 0.0
    return out


def load_runs(patterns):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    runs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if is_ce(d) is not True:
            continue
        runs.append(
            {
                "pr_regret": np.asarray(d["pr_regret"], dtype=np.float64),
                "pr_cum_regret": np.asarray(d["pr_cum_regret"], dtype=np.float64),
                "pr_n_deviated": np.asarray(d["pr_n_deviated"], dtype=np.float64),
                "pr_max_phi": np.asarray(d["pr_max_phi"], dtype=np.float64),
                "pr_inner_solver_time": np.asarray(
                    d["pr_inner_solver_time"], dtype=np.float64
                ),
                "final_total_rounds": int(d["final_total_rounds"]),
            }
        )
    return runs


def normalize(arrays):
    return [a / max(a.max(), 1e-12) for a in arrays]


def style_axis(ax, log_y=False):
    ax.tick_params(axis="both", labelsize=TICK_FS)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------
# convergence_main.png
# ---------------------------------------------------------------------
print("Building convergence_main.png ...")

CELLS_MAIN = [
    ("Pigou $n{=}500$", [f"{DATA_DIR}/pigou_n500_BR_s*.npz"], "#1f77b4"),
    ("SFgrid3 $n{=}50$", [f"{DATA_DIR}/sfgrid3_n50_k3_BR_s*.npz"], "#6baed6"),
    ("Polymatrix $n{=}100$", [f"{DATA_DIR}/n100/poly100_k3_BR_s*.npz"], "#fdae61"),
    ("Polymatrix $n{=}150$", [f"{DATA_DIR}/poly150/poly150_k3_BR_s*.npz"], "#d73027"),
    ("Cournot AVG $n{=}50$", [f"{DATA_DIR}/medium_n/cournot_avg50_k3_BR_s*.npz"], "#41ab5d"),
    ("Cournot AVG $n{=}100$", [f"{DATA_DIR}/medium_n/cournot_avg100_k3_BR_s*.npz"], "#006d2c"),
    ("GNF cycle $n{=}20$", [f"{DATA_DIR}/n100/gnf20_m3_BR_s*.npz"], "#bcbddc"),
    ("GNF cycle $n{=}40$", [f"{DATA_DIR}/gnf_fixed/gnf40_r500_m3_BR_s*.npz"], "#9e9ac8"),
    ("GNF cycle $n{=}50$", [f"{DATA_DIR}/gnf_fixed/gnf50_r500_m3_BR_s*.npz"], "#6a51a3"),
]

fig, ax = plt.subplots(figsize=(9, 5.5))
for label, patterns, color in CELLS_MAIN:
    runs = load_runs(patterns)
    if not runs:
        continue
    series = [r["pr_regret"] / max(r["pr_regret"].max(), 1e-12) for r in runs]
    M = pad(series, XLIM)
    t = np.arange(M.shape[1])
    mu = M.mean(0)
    sd = M.std(0)
    ax.plot(t, mu, "-", lw=1.7, color=color, label=label)
    ax.fill_between(t, np.maximum(mu - sd, 0), mu + sd, color=color, alpha=0.12)

ax.set_xlabel("Round $t$", fontsize=AXIS_FS)
ax.set_ylabel("Per-round regret", fontsize=AXIS_FS)
ax.set_xlim(0, XLIM)
ax.set_ylim(bottom=0)
ax.legend(fontsize=LEGEND_FS, framealpha=0.95, ncol=2, loc="upper right")
style_axis(ax)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "convergence_main.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT_DIR}/convergence_main.png")


# ---------------------------------------------------------------------
# scaling.png
# ---------------------------------------------------------------------
print("Building scaling.png ...")


def get_T_stats(patterns):
    Ts = []
    for p in patterns:
        for f in sorted(glob.glob(p)):
            d = np.load(f, allow_pickle=True)
            if is_ce(d) is not True:
                continue
            Ts.append(int(d["final_total_rounds"]))
    if not Ts:
        return (0, 0)
    return (float(np.mean(Ts)), float(np.std(Ts)))


families = {
    "Routing": [
        (2, *get_T_stats([f"{DATA_DIR}/pigou_n500_BR_s*.npz"])),
        (9, *get_T_stats([f"{DATA_DIR}/braess_n50_BR_s*.npz"])),
        (9, *get_T_stats([f"{DATA_DIR}/sfgrid3_n50_k3_BR_s*.npz"])),
    ],
    "Polymatrix": [
        (100, *get_T_stats([f"{DATA_DIR}/n100/poly50_k3_BR_s*.npz"])),
        (200, *get_T_stats([f"{DATA_DIR}/n100/poly100_k3_BR_s*.npz"])),
        (300, *get_T_stats([f"{DATA_DIR}/poly150/poly150_k3_BR_s*.npz"])),
    ],
    "Cournot AVG": [
        (22, *get_T_stats([f"{DATA_DIR}/n100/cournot20_k3_BR_s*.npz"])),
        (52, *get_T_stats([f"{DATA_DIR}/medium_n/cournot_avg50_k3_BR_s*.npz"])),
        (102, *get_T_stats([f"{DATA_DIR}/medium_n/cournot_avg100_k3_BR_s*.npz"])),
    ],
    "GNF cycle": [
        (180, *get_T_stats([f"{DATA_DIR}/n100/gnf20_m3_BR_s*.npz"])),
        (360, *get_T_stats([f"{DATA_DIR}/gnf_fixed/gnf40_r500_m3_BR_s*.npz"])),
        (450, *get_T_stats([f"{DATA_DIR}/gnf_fixed/gnf50_r500_m3_BR_s*.npz"])),
    ],
}

colors = {
    "Routing": "#1f77b4",
    "Polymatrix": "#d73027",
    "Cournot AVG": "#006d2c",
    "GNF cycle": "#6a51a3",
}
markers = {
    "Routing": "o",
    "Polymatrix": "s",
    "Cournot AVG": "^",
    "GNF cycle": "D",
}

fig, ax = plt.subplots(figsize=(7.5, 5.0))
for fam, points in families.items():
    nKs = [p[0] for p in points]
    Ts = [p[1] for p in points]
    sds = [p[2] for p in points]
    ax.errorbar(
        nKs,
        Ts,
        yerr=sds,
        marker=markers[fam],
        markersize=7,
        lw=1.5,
        capsize=3,
        color=colors[fam],
        label=fam,
    )

xref = np.linspace(1, 500, 100)
ax.plot(xref, 0.1 * xref, "--", color="gray", lw=1, alpha=0.6, label="$T = d/10$ (reference)")

ax.set_xlabel("Latent dimension $nK$", fontsize=AXIS_FS)
ax.set_ylabel("Rounds to convergence $T$", fontsize=AXIS_FS)
ax.legend(fontsize=LEGEND_FS, framealpha=0.95)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
style_axis(ax)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scaling.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT_DIR}/scaling.png")


# ---------------------------------------------------------------------
# Per-cell aggregated 4-panel plots
# ---------------------------------------------------------------------
print("Building per-cell aggregated plots ...")

CELLS_AGG = [
    ("pigou_n500_BR", [f"{DATA_DIR}/pigou_n500_BR_s*.npz"]),
    ("sfgrid3_n50_k3_BR", [f"{DATA_DIR}/sfgrid3_n50_k3_BR_s*.npz"]),
    ("poly100_k3_BR", [f"{DATA_DIR}/n100/poly100_k3_BR_s*.npz"]),
    ("poly150_k3_BR", [f"{DATA_DIR}/poly150/poly150_k3_BR_s*.npz"]),
    ("gnf20_m3_BR", [f"{DATA_DIR}/n100/gnf20_m3_BR_s*.npz"]),
    ("gnf40_r500_m3_BR", [f"{DATA_DIR}/gnf_fixed/gnf40_r500_m3_BR_s*.npz"]),
    ("gnf50_r500_m3_BR", [f"{DATA_DIR}/gnf_fixed/gnf50_r500_m3_BR_s*.npz"]),
]


def plot_one(ax, data, ylabel, log_y=False, kind="line"):
    if data.size == 0:
        return
    T = data.shape[1]
    t_axis = np.arange(T)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if kind == "line":
        ax.plot(t_axis, mean, "-", linewidth=1.8, color="tab:blue")
        ax.fill_between(t_axis, mean - std, mean + std, color="tab:blue", alpha=0.22)
    else:
        ax.bar(t_axis, mean, yerr=std, capsize=2, color="tab:red", alpha=0.75, width=0.9)
    ax.set_xlabel("round $t$", fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_xlim(0, XLIM)
    style_axis(ax, log_y=log_y)


for name, patterns in CELLS_AGG:
    runs = load_runs(patterns)
    if not runs:
        print(f"  SKIP {name} (no CE-converged seeds)")
        continue

    cum_regret_norm = pad(normalize([r["pr_cum_regret"] for r in runs]), XLIM)
    max_phi_norm = pad(normalize([r["pr_max_phi"] for r in runs]), XLIM)
    n_dev = pad([r["pr_n_deviated"] for r in runs], XLIM)
    inner_t = pad([r["pr_inner_solver_time"] for r in runs], XLIM)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_one(axes[0, 0], cum_regret_norm, ylabel="cumulative regret", kind="line")
    plot_one(axes[0, 1], n_dev, ylabel="# deviating agents", kind="bar")
    plot_one(axes[1, 0], max_phi_norm, ylabel=r"$\max_i \varphi_i(t)$", kind="line")
    plot_one(axes[1, 1], inner_t, ylabel="inner-solver time (s)", kind="line", log_y=True)
    fig.tight_layout()
    out = os.path.join(AGG_DIR, f"{name}_agg.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}  ({len(runs)} seeds)")


# ---------------------------------------------------------------------
# routing_polymatrix_combined.png
# ---------------------------------------------------------------------
print("Building routing_polymatrix_combined.png ...")

CELLS_RP = [
    ("Pigou $n{=}500$", [f"{DATA_DIR}/pigou_n500_BR_s*.npz"], "#1f77b4"),
    ("SFgrid3 $n{=}50$", [f"{DATA_DIR}/sfgrid3_n50_k3_BR_s*.npz"], "#6baed6"),
    ("Polymatrix $n{=}100$", [f"{DATA_DIR}/n100/poly100_k3_BR_s*.npz"], "#f46d43"),
    ("Polymatrix $n{=}150$", [f"{DATA_DIR}/poly150/poly150_k3_BR_s*.npz"], "#d73027"),
]

data_rp = {}
for label, patterns, color in CELLS_RP:
    runs = load_runs(patterns)
    if not runs:
        continue
    data_rp[label] = {
        "color": color,
        "n_seeds": len(runs),
        "cum_regret_norm": pad(normalize([r["pr_cum_regret"] for r in runs]), XLIM),
        "max_phi_norm": pad(normalize([r["pr_max_phi"] for r in runs]), XLIM),
        "n_dev": pad([r["pr_n_deviated"] for r in runs], XLIM),
        "inner_t": pad([r["pr_inner_solver_time"] for r in runs], XLIM),
    }


def plot_metric(ax, key, ylabel, log_y=False):
    for label, d in data_rp.items():
        arr = d[key]
        if arr.size == 0:
            continue
        T = arr.shape[1]
        t = np.arange(T)
        mu = arr.mean(0)
        sd = arr.std(0)
        ax.plot(t, mu, "-", lw=1.8, color=d["color"], label=label)
        ax.fill_between(t, mu - sd, mu + sd, color=d["color"], alpha=0.15)
    ax.set_xlabel("round $t$", fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_xlim(0, XLIM)
    ax.legend(fontsize=LEGEND_FS, framealpha=0.9)
    style_axis(ax, log_y=log_y)


def plot_deviations(ax):
    series = list(data_rp.items())
    n_series = len(series)
    width = 0.8 / n_series
    for i, (label, d) in enumerate(series):
        arr = d["n_dev"]
        if arr.size == 0:
            continue
        T = arr.shape[1]
        t = np.arange(T)
        mu = arr.mean(0)
        sd = arr.std(0)
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(
            t + offset,
            mu,
            yerr=sd,
            capsize=2,
            color=d["color"],
            alpha=0.85,
            width=width,
            label=label,
        )
    ax.set_xlabel("round $t$", fontsize=AXIS_FS)
    ax.set_ylabel("# deviating agents", fontsize=AXIS_FS)
    ax.set_xlim(-0.5, XLIM)
    ax.legend(fontsize=LEGEND_FS, framealpha=0.9)
    style_axis(ax)


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
plot_metric(axes[0, 0], "cum_regret_norm", ylabel="cumulative regret")
plot_deviations(axes[0, 1])
plot_metric(axes[1, 0], "max_phi_norm", ylabel=r"$\max_i \varphi_i(t)$")
plot_metric(axes[1, 1], "inner_t", ylabel="inner-solver time (s)", log_y=True)
fig.tight_layout()
out = os.path.join(AGG_DIR, "routing_polymatrix_combined.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"  saved {out}")


print()
print("All output figures:")
for root, _, files in os.walk(OUT_DIR):
    for f in sorted(files):
        if f.endswith(".png"):
            p = os.path.join(root, f)
            kb = os.path.getsize(p) // 1024
            print(f"  {p}  ({kb} KB)")
