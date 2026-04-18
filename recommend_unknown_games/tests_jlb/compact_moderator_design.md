# Compact Moderator — Design and Test Spec

## Goal

Implement a moderator that operates entirely in the compact parameter space `θ ∈ R^d` (e.g., `d = 2|E|` for affine congestion games), never materializing the exponentially-large w-vector in `R^N` (where `N = Σ m_i²`).

This replaces the **inner** CE solver of the paper's moderator (which was Gurobi on the normal-form LP — infeasible for compact games with M > 10^8 profiles) with the **JLB** pipeline that consumes θ and returns a sparse CE mixture.

The existing normal-form moderator (`cutting_plane.py`, `oracle.py`, `ce_solver.py`) is **kept intact** as a ground-truth reference. The new compact moderator lives in parallel modules.

## Architecture

```
[true game with theta*] ──── (hidden from moderator) ────┐
                                                         │
[CompactCuttingPlaneAlgorithm]                           │
  state: θ(t), knowledge set K_t ⊂ R^d                  │
  picks θ(t) = analytic center of K_t                   │
                      │                                  │
                      ▼                                  │
[CompactSeparationOracle(game_skeleton, θ(t))]           │
  1. Build CongestionGame with current θ(t)             │
  2. JLB inner solver → sparse CE mixture x(t)           │
  3. Sample recommendation a(t) ~ x(t)                   │
  4. Query agents (they use θ*) ─────────────────────────┘
  5. Agents return response a*(t)
  6. If deviation: build cut in θ-space directly
       (no materialization of w or full x)
                      │
                      ▼
[CompactCuttingPlaneAlgorithm]
  accumulate cut q_θ ∈ R^d in K_t
  update θ-estimate
```

## API Surfaces (tests reference these)

### `FeatureMap` (abstract)

```python
class FeatureMap(ABC):
    @property
    @abstractmethod
    def d(self) -> int: "Parameter dimension"

    @property
    @abstractmethod
    def N(self) -> int: "Induced w-vector dimension"

    @property
    @abstractmethod
    def spec(self) -> GameSpec: ...

    @abstractmethod
    def theta_true(self) -> NDArray: "The true parameter of the underlying game"

    @abstractmethod
    def w_from_theta(self, theta: NDArray) -> NDArray:
        "Apply Φ: build the full w-vector from θ (for validation only)"

    @abstractmethod
    def project_cut_sparse(self,
                           cut_agents: list[int],
                           cut_blocks: dict[tuple[int, int], NDArray],
                           mixture: SparseMixture,
                           ) -> NDArray:
        """Compute q_θ = Φ^T q for a sparse cut given by its nonzero blocks.

        cut_blocks maps (agent_i, action_k) to a length-d_i array holding
        q_w[i, k, :] (the block of q at agent i, alternative action k).
        """
```

### `CongestionFeatureMap(FeatureMap)`

Concrete implementation for `CongestionGame` with affine costs.
- `d = 2 * n_edges`
- θ = (a_0, b_0, a_1, b_1, …, a_{|E|-1}, b_{|E|-1}).

### `SparseMixture` (new dataclass)

Just a thin wrapper over (profiles, weights) — the output of JLB.

### `CompactSeparationOracle`

```python
class CompactSeparationOracle:
    def __init__(self, game: CongestionGame,
                       feedback: AgentFeedback,
                       jlb_kwargs: dict):
        ...
    def query(self, theta_guess: NDArray) -> CompactOracleResult: ...
```

Where `CompactOracleResult` carries:
- recommendation a(t)
- response a*(t)
- deviated flag
- cut_theta ∈ R^d (None if no deviation)
- regret scalar
- mixture used (for debugging)

### `CompactCuttingPlaneAlgorithm`

```python
class CompactCuttingPlaneAlgorithm:
    def __init__(self, oracle: CompactSeparationOracle,
                       feature_map: FeatureMap,
                       box_radius: float = 1.0,
                       max_cuts: int | None = None):
        ...
    def run(self, T: int, theta_star: Optional[NDArray] = None) -> CompactCuttingPlaneResult: ...
```

Reuses existing `analytic_center` from `cutting_plane.py` but in dimension `d` instead of `N`. Cuts are added as `<q_θ, θ> >= 0` to the knowledge set `{θ ∈ [-R, R]^d}`.

## Test Groups

### T0 — Feature map correctness (hand-checkable invariants)

1. `test_theta_dim_matches_2_edges`: for Pigou (2 edges), d=4.
2. `test_w_from_theta_equals_pack_w`: w_from_theta(theta_true) agrees element-wise with game.as_normal_form().pack_w(). This is the Φ fundamental identity.
3. `test_w_from_theta_is_linear`: Φ(a·θ1 + b·θ2) = a·Φ(θ1) + b·Φ(θ2).
4. `test_zero_theta_gives_zero_w`: Φ(0) = 0.

### T1 — Cut projection correctness (the crucial identity)

5. `test_projection_matches_inner_product_on_true_theta`:
   for arbitrary dense q ∈ R^N, the explicit projection q_θ = Φ^T q
   satisfies <θ_true, q_θ> = <w_true, q>. Use random q and verify on
   small games.
6. `test_projection_matches_inner_product_on_random_theta`: same as (5)
   for θ ≠ θ_true.
7. `test_sparse_projection_matches_dense`: our sparse projection (which
   only takes the nonzero blocks) agrees with a dense Φ^T q computed via
   w_from_theta on a set of basis vectors.
8. `test_projection_structure_for_single_deviation_cut`: for a cut
   representing a single agent's deviation (one positive block, one
   negative block), the projection touches only edges in the two
   corresponding routes. Hand-checkable on Pigou.

### T2 — Compact oracle correctness

9. `test_oracle_returns_valid_mixture`: sparse CE output is a valid CE
   of the current guessed game. Verified via `max_incentive_violation`.
10. `test_oracle_at_true_theta_no_deviation`: when θ_guess = θ_true and
    agents are BR, no deviation is observed.
11. `test_oracle_cut_separation_property`:
    when deviation happens, <θ_true, q_θ> >= 0 and <θ_guess, q_θ> <= 0
    (the fundamental separation property in θ-space).
12. `test_oracle_matches_normal_form_cut_on_small_game`: the cut we
    produce in θ-space equals Φ^T (cut from existing normal-form
    SeparationOracle) up to sign/scaling.

### T3 — Compact moderator correctness

13. `test_moderator_runs_without_error_on_pigou_3`: smoke test.
14. `test_moderator_regret_sublinear_on_small_game`: regret / T → 0
    as T grows (empirical).
15. `test_moderator_never_recovers_w`: verify the moderator state stays
    in R^d — we never have a length-N array in the call path.
16. `test_knowledge_set_contains_true_theta`: at every iteration,
    θ_true satisfies all accumulated cuts (because every cut preserves
    the separation property).

### T4 — Scale integration

17. `test_pipeline_runs_on_large_game`: run for T=50 on Pigou n=50
    where normal-form would have M=2^50 ≈ 10^15 profiles. The compact
    moderator should complete in reasonable time (< 60s).
18. `test_normal_form_moderator_cannot_run_large_game`: verify (by
    attempting and expecting failure or timeout) that the existing
    normal-form moderator cannot handle Pigou n=50.

## Expected behavior on canonical test

On Pigou n=3, affine (0 + 1·ℓ) on edge 0, constant 1 on edge 1:
- θ_true = [0, 1, 1, 0]
- The compact moderator, starting from θ = 0, should converge: after
  ~10 rounds, |θ(t) - θ_true| should be shrinking. Exact convergence
  depends on the outer-loop + cuts, but regret must be sublinear.

## Implementation order (after tests are all written)

1. Test Group T0-T1 → `feature_map.py`
2. Test Group T2    → `compact_oracle.py`
3. Test Group T3    → `compact_moderator.py`
4. Test Group T4    → nothing new, just runs

Nothing new depends on fixing JLB reliability on arbitrary normal-form
games. The inner solver only sees CongestionGame instances, where
JLB's performance is strong.
