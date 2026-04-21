"""List of all overnight cells (game, n, m, seed, feedback, γ).

Imported by runner.py — NOT auto-launched.  To actually start a batch:

    python -m experiments_overnight.runner \
        --config experiments_overnight/configs/overnight_cells.py \
        --workers 6 --outdir experiments_overnight/out/

Each entry is a dict with keys the runner understands.  Scope is kept
to what the overnight plan committed to.  Three seeds each; BR and
γ-BR (γ=0.5) variants for a subset.
"""

from __future__ import annotations


SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)  # 8 seeds for main batch
GAMMA_SEEDS = (0, 1, 2, 3)         # 4 seeds for γ-BR variants


def _congestion_cells() -> list[dict]:
    cells: list[dict] = []
    # Pigou
    for n in [10, 50, 100, 200, 500, 1000, 2000, 5000]:
        for seed in SEEDS:
            cells.append(dict(
                game_family="pigou", n=n, m=2, seed=seed,
                feedback="BR", gamma=0.0,
                label=f"pigou_n{n}_BR_s{seed}",
                box_radius=10.0, box_lower=0.0,
                max_rounds=80, stop_after_no_dev=10,
            ))
    # Braess
    for n in [10, 20, 50, 100, 500, 1000]:
        for seed in SEEDS:
            cells.append(dict(
                game_family="braess", n=n, m=3, seed=seed,
                feedback="BR", gamma=0.0,
                label=f"braess_n{n}_BR_s{seed}",
                box_radius=10.0, box_lower=0.0,
                max_rounds=80, stop_after_no_dev=10,
            ))
    # SFgrid3
    for (n, k) in [(10, 3), (20, 3), (50, 3), (100, 3), (500, 3),
                    (20, 5), (50, 5), (100, 5),
                    (20, 7), (30, 7),
                    (15, 10)]:
        for seed in SEEDS:
            cells.append(dict(
                game_family="sfgrid3", n=n, m=k, seed=seed,
                feedback="BR", gamma=0.0,
                label=f"sfgrid3_n{n}_k{k}_BR_s{seed}",
                box_radius=10.0, box_lower=0.0,
                max_rounds=120, stop_after_no_dev=10,
            ))
    # SFgrid4
    for (n, k) in [(20, 3), (50, 3), (100, 3), (200, 3), (30, 5)]:
        for seed in SEEDS:
            cells.append(dict(
                game_family="sfgrid4", n=n, m=k, seed=seed,
                feedback="BR", gamma=0.0,
                label=f"sfgrid4_n{n}_k{k}_BR_s{seed}",
                box_radius=10.0, box_lower=0.0,
                max_rounds=140, stop_after_no_dev=10,
            ))
    return cells


def _polymatrix_cells() -> list[dict]:
    cells: list[dict] = []
    for (n, k) in [(10, 3), (20, 3), (50, 3), (100, 3), (500, 3),
                    (20, 5), (50, 5), (200, 5),
                    (20, 10), (100, 10),
                    (20, 20), (50, 20),
                    (10, 50), (20, 50),
                    (10, 100)]:
        for seed in SEEDS:
            cells.append(dict(
                game_family="polymatrix_pricing", n=n, m=k, seed=seed,
                feedback="BR", gamma=0.0,
                label=f"polymatrix_n{n}_k{k}_BR_s{seed}",
                box_radius=150.0, box_lower=0.0,
                max_rounds=80, stop_after_no_dev=10,
                degree=3,
            ))
    return cells


def _graphical_cells() -> list[dict]:
    cells: list[dict] = []
    # Graph coloring on grids
    for (rows, cols, k) in [(3, 3, 3), (4, 4, 3), (5, 5, 3),
                             (5, 5, 5), (7, 7, 3)]:
        n = rows * cols
        for seed in SEEDS:
            cells.append(dict(
                game_family="graph_coloring", n=n, m=k, seed=seed,
                grid_size=rows,  # assumes square
                feedback="BR", gamma=0.0,
                label=f"coloring_{rows}x{cols}_k{k}_BR_s{seed}",
                box_radius=3.0, box_lower=0.0,
                max_rounds=60, stop_after_no_dev=10,
                true_collision_cost=1.0,
            ))
    # Best-shot PG on cycles
    for n in [10, 20, 50, 100, 200]:
        for seed in SEEDS:
            cells.append(dict(
                game_family="bestshot_pg", n=n, m=2, seed=seed,
                topology="cycle",
                feedback="BR", gamma=0.0,
                label=f"bestshot_cycle{n}_BR_s{seed}",
                box_radius=3.0, box_lower=0.0,
                max_rounds=60, stop_after_no_dev=10,
                true_benefit=1.0, true_cost=0.4,
            ))
    return cells


def _gamma_br_cells() -> list[dict]:
    """A γ-BR variant for a representative subset."""
    cells: list[dict] = []
    for (game, n, m) in [
        ("pigou", 50, 2), ("pigou", 500, 2),
        ("braess", 50, 3),
        ("sfgrid3", 50, 3), ("sfgrid3", 20, 5),
        ("polymatrix_pricing", 50, 5),
        ("graph_coloring", 9, 3),
        ("bestshot_pg", 20, 2),
    ]:
        for gamma in (0.5, 2.0):
            for seed in GAMMA_SEEDS:
                cell = dict(
                    game_family=game, n=n, m=m, seed=seed,
                    feedback="gamma_BR", gamma=float(gamma),
                    label=f"{game}_n{n}_m{m}_gBR{gamma}_s{seed}",
                    box_radius=10.0, box_lower=0.0,
                    max_rounds=80, stop_after_no_dev=10,
                )
                if game in ("graph_coloring",):
                    import math
                    cell["grid_size"] = int(math.isqrt(n))
                    cell["true_collision_cost"] = 1.0
                    cell["box_radius"] = 3.0
                if game == "bestshot_pg":
                    cell["topology"] = "cycle"
                    cell["true_benefit"] = 1.0
                    cell["true_cost"] = 0.4
                    cell["box_radius"] = 3.0
                if game == "polymatrix_pricing":
                    cell["box_radius"] = 150.0
                    cell["degree"] = 3
                cells.append(cell)
    return cells


def all_cells() -> list[dict]:
    return (
        _congestion_cells()
        + _polymatrix_cells()
        + _graphical_cells()
        + _gamma_br_cells()
    )


if __name__ == "__main__":
    cells = all_cells()
    by_family = {}
    for c in cells:
        by_family.setdefault(c["game_family"], 0)
        by_family[c["game_family"]] += 1
    print(f"Total cells: {len(cells)}")
    for k, v in sorted(by_family.items()):
        print(f"  {k:25s}: {v}")
