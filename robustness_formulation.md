# Robustness of Recommendations Against Non-Compliant Agents

## Setting

Consider the game $G(\{A_i\}_{i=1}^n, \{u_i\}_{i=1}^n)$ with utilities in $[0,1]$. A moderator interacts with agents over $T$ rounds. At each round $t$:

1. The moderator commits to $x(t) \in \Delta(A)$, samples $R(t) \sim x(t)$, and privately sends $R_i(t)$ to each agent $i$ along with $x(t)$.
2. Each agent $i$ chooses an action $A_i(t)$.

Let $S \subseteq [n]$ denote the set of **compliant agents** who follow the recommendation: $A_i(t) = R_i(t)$ for all $i \in S$. Agents $j \notin S$ play **arbitrarily** (adversarially, possibly depending on full history).

## Regret Definitions

**Individual regret** of a compliant agent $i \in S$:
$$\mathcal{R}_i(T) = \max_{a_i' \in A_i} \sum_{t=1}^{T} \left[ u_i(a_i', A_{-i}(t)) - u_i(R_i(t), A_{-i}(t)) \right]$$

where $A_{-i}(t)$ is the realized action profile of all other agents (compliant ones play their recommendations, adversarial ones play arbitrarily).

**Compliance regret** of the moderator (as in [1]):
$$\text{Reg}(\mathcal{A}) = \mathbb{E}\left[\sum_{t=1}^{T} \sum_{i \in [n]} \phi_i(R_i(t), A_i^*(t), x(t))\right]$$

## Goal

**Theorem (desired).** There exists an algorithm for the moderator such that, for any partition of agents into compliant set $S$ and adversarial set $[n] \setminus S$, and for any (adaptive) strategy of the adversarial agents:

1. Every compliant agent $i \in S$ satisfies
$$\mathbb{E}[\mathcal{R}_i(T)] \leq O(\sqrt{T})$$

2. The moderator learns the game:
$$\text{Reg}(\mathcal{A}) \leq O(nM \log T)$$

Both guarantees hold simultaneously with the same algorithm and the same recommendations $x(t)$.

## Key Challenge

The moderator must choose $x(t)$ to simultaneously serve two purposes:

- **Learning:** $x(t)$ must have CE structure so that observed follow/deviate behavior from compliant agents generates valid cutting-plane updates.
- **Protection:** The marginal distribution over each compliant agent $i$'s actions, $x_i(t)(\cdot) = \sum_{a_{-i}} x(t)(\cdot, a_{-i})$, must form a no-regret sequence against the adversarial actions of others.

The moderator has freedom in choosing *which* distribution in $CE(w(t))$ to recommend. The question is whether this freedom is sufficient to track no-regret marginals while maintaining the CE structure needed for learning.

## Reference

[1] Alanqary, Baba, Wu, Bayen. *Learning to Recommend in Unknown Games*. arXiv:2602.16998, 2026.
