# Implementation Plan: Simulation Framework for "Learning to Recommend in Unknown Games"

## Scope

We implement the **environment + oracle** pipeline: given a guess w(t), compute a CE, interact with agents, and produce a separating cut. The cutting-plane loop is deferred.

```
recommend_unknown_games/
├── types.py          # Data structures, type aliases
├── utils.py          # Index arithmetic, w-vector packing
├── game.py           # Game environment + BR/QR feedback
├── ce_solver.py      # CE LP solver (Gurobi / scipy fallback)
├── oracle.py         # Separation oracle pipeline
└── tests/
    ├── test_game.py
    ├── test_ce.py
    └── test_oracle.py
```

---

## 1. Data Structures (`types.py`)

```python
@dataclass(frozen=True)
class GameSpec:
    n: int                           # agents
    action_counts: tuple[int, ...]   # (m_1, ..., m_n)
    M: int                           # prod(m_i)
    d: tuple[int, ...]               # d_i = M / m_i
    N: int                           # sum_i (m_i - 1) * d_i

@dataclass
class OracleResult:
    recommendation: tuple[int, ...]  # a(t)
    response: tuple[int, ...]        # a*(t)
    deviated: bool
    cut: ndarray | None              # q(t) in R^N
    ce_dist: ndarray                 # x(t) flat (M,)
```

**Key choice:** CE distribution stored as flat array `(M,)`. Reshape to `(m_1,...,m_n)` via `x.reshape(action_counts)` (free -- numpy view) only when extracting conditionals.

---

## 2. Game Environment (`game.py`)

### Storage

```
utilities: ndarray, shape (n, m_1, m_2, ..., m_n)
utilities[i][a_1, ..., a_n] = u_i(a_1, ..., a_n)
```

### Utility differences (vectorized, no loops)

```python
def utility_diff_vector(self, i, a_i, a_prime_i) -> ndarray:
    # shape (d_i,) = (M/m_i,)
    diff = np.take(self.utilities[i], a_prime_i, axis=i) \
         - np.take(self.utilities[i], a_i, axis=i)
    return diff.ravel()
```

`np.take(arr, k, axis=i)` selects a slice along agent i's axis, yielding shape `(m_1,...,m_{i-1}, m_{i+1},...,m_n)` with d_i entries. Fully vectorized.

### w-vector packing

Layout of w* in R^N:

```
[ w_1(0,1) | w_1(0,2) | ... | w_1(0,m_1-1) | w_2(0,1) | ... | w_n(0,m_n-1) ]
  <-- d_1 -->                                  <-- d_2 -->
```

Reference action = 0 for every agent. Other differences recovered via triangular identity:

```
w_i(a, b) = w_i(0, b) - w_i(0, a)     (a, b != 0)
w_i(a, 0) = -w_i(0, a)
```

### Agent feedback

```python
def respond(self, i, a_i, ce_dist) -> int:
    # Reshape x, extract conditional x(a_i, .)
    x_multi = ce_dist.reshape(self.spec.action_counts)
    cond = np.take(x_multi, a_i, axis=i).ravel()     # (d_i,)
    
    # Utility matrix U_i: shape (m_i, d_i)
    U_i = np.moveaxis(self.utilities[i], i, 0).reshape(m_i, -1)
    expected = U_i @ cond                              # (m_i,)
    
    if self.model == "best_response":
        return int(np.argmax(expected))
    else:  # quantal_response
        gains = expected - expected[a_i]
        gains = np.maximum(gains, 0.0)                 # truncated QR
        probs = np.exp(self.beta * gains)
        probs /= probs.sum()
        return self.rng.choice(m_i, p=probs)
```

Single matrix-vector product per agent. No loops over action profiles.

---

## 3. CE Solver (`ce_solver.py`)

### The LP

```
find   x in R^M
s.t.   A @ x <= 0          (CE incentive constraints)
       1^T x = 1           (probability)
       x >= 0
```

- Variables: M
- Incentive constraints: C = sum_i m_i(m_i - 1)
- Each row of A has d_i = M/m_i nonzeros

### Sparse constraint matrix (build once, update values)

Row for constraint (i, a_i, a'_i):

```
A[row, flat_index(a_i, a_{-i})] = w_i(a_i, a'_i)[a_{-i}]    for all a_{-i}
```

**Critical optimization:** The **sparsity pattern** (row/col indices) depends only on the game spec, not on w. Precompute COO row/col arrays at init. On each solve, only recompute the data array from w.

```python
class CESolver:
    def __init__(self, spec):
        self._precompute_sparsity_pattern()
        # Cache: profile_indices[i][a_i] = flat indices where agent i plays a_i
    
    def _precompute_sparsity_pattern(self):
        rows, cols = [], []
        self._constraint_map = []  # maps each row to (i, a_i, a'_i)
        row = 0
        for i in range(n):
            for a_i in range(m_i):
                for a_prime in range(m_i):
                    if a_prime == a_i: continue
                    col_indices = self.profile_indices[i][a_i]
                    rows.extend([row] * d_i)
                    cols.extend(col_indices)
                    self._constraint_map.append((i, a_i, a_prime))
                    row += 1
        self._coo_rows = np.array(rows)
        self._coo_cols = np.array(cols)
    
    def _build_data(self, w) -> ndarray:
        data = []
        for (i, a_i, a_prime) in self._constraint_map:
            w_block = get_w_block(w, i, a_i, a_prime)  # (d_i,)
            data.append(w_block)
        return np.concatenate(data)
```

### Solver backends

**Gurobi (preferred):**
```python
import gurobipy as gp
model = gp.Model()
model.Params.OutputFlag = 0
x = model.addMVar(M, lb=0.0)
model.addMConstr(A_csr, x, '<', np.zeros(C))
model.addConstr(x.sum() == 1.0)
model.setObjective(0, gp.GRB.MINIMIZE)   # feasibility
model.optimize()
```

Supports warm-starting from previous basis via `model.update()` + setting VBasis/CBasis attributes.

**Scipy fallback (HiGHS):**
```python
from scipy.optimize import linprog
result = linprog(c=np.zeros(M), A_ub=A, b_ub=np.zeros(C),
                 A_eq=np.ones((1,M)), b_eq=[1.0], method='highs')
```

HiGHS is the best open-source LP solver. Adequate for M up to ~10^5.

---

## 4. Separation Oracle (`oracle.py`)

### Pipeline

```
query(w_guess) -> OracleResult:
    1. x = ce_solver.solve(w_guess)           # CE of guessed game
    2. a = sample(x)                          # recommendation
    3. a_star = feedback.respond_all(a, x)    # agents respond
    4. if a != a_star: q = construct_cut(...)  # separating hyperplane
```

### Cut construction (Algorithm 4, corrected)

Agent offsets (fixing paper's error for heterogeneous action sets):

```python
offsets[0] = 0
offsets[i] = offsets[i-1] + (m_{i-1} - 1) * d_{i-1}
```

For each deviating agent i (a_i != a*_i):

```python
cond = np.take(x_multi, a_t[i], axis=i).ravel()   # x(a_i, .) shape (d_i,)

j, j_star = a_t[i], a_star[i]

# Using triangular identity: w_i(j, j*) = w_i(0, j*) - w_i(0, j)
# So <w*, q> picks up +<cond, w_i(0,j*)> and -<cond, w_i(0,j)>

if j_star != 0:   # block (i, j*) exists in w
    q[offset + (j_star-1)*d_i : offset + j_star*d_i] += cond

if j != 0:         # block (i, j) exists in w
    q[offset + (j-1)*d_i : offset + j*d_i] -= cond
```

### Separation property (the invariant)

```
<w*, q> = sum_i phi_i(a_i, a*_i, x) >= 0     (agents deviate to profitable actions)
<w(t), q> <= 0                                 (x is CE of w(t), so no profitable deviation)
```

---

## 5. Optimization Summary

| Optimization | Where | Impact |
|---|---|---|
| Precompute profile index arrays | `CESolver.__init__` | Avoid recomputing meshgrid every LP |
| Separate sparsity pattern from data | `CESolver._build_data` | Only recompute values, not structure |
| `np.take` for slicing utility arrays | `game.py`, `oracle.py` | Vectorized conditional extraction |
| Matrix-vector for expected utilities | `AgentFeedback.respond` | O(m_i * d_i) per agent, no loops |
| Gurobi warm-start | `CESolver.solve` | Reuse basis across similar LPs |
| Sparse CSR for constraint matrix | `CESolver` | Memory and solver efficiency |

---

## 6. Memory Budget

| n agents | m actions | M = m^n | N | Utilities | CE constraint nnz |
|---|---|---|---|---|---|
| 3 | 5 | 125 | 300 | 3 KB | 7,500 |
| 5 | 3 | 243 | 1,215 | 10 KB | 11,664 |
| 4 | 4 | 256 | 768 | 8 KB | 12,288 |
| 10 | 2 | 1,024 | 5,120 | 80 KB | 10,240 |
| 7 | 5 | 78,125 | 2.19M | 4.4 MB | 18.75M |
| 20 | 2 | 1,048,576 | 10.5M | 160 MB | 20.97M |

Practical limit: ~M = 10^6 with dense storage. Beyond that, need structured representations.

---

## 7. Implementation Order

| Phase | Files | Deliverable |
|---|---|---|
| 1 | `types.py`, `utils.py` | GameSpec, index arithmetic, w-packing |
| 2 | `game.py` | NormalFormGame, random_game, AgentFeedback (BR + QR) |
| 3 | `ce_solver.py` | Sparse constraint build, Gurobi + scipy backends |
| 4 | `oracle.py` | Full query pipeline: guess -> CE -> interact -> cut |
| 5 | `tests/` | Separation property test, known-game CE verification |

Phase 1-2 have no external dependencies. Phase 3 requires Gurobi (or scipy fallback). Phase 4 integrates everything.
