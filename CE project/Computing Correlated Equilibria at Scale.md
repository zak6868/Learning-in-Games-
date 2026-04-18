## Introduction

Correlated equilibria of a game can be formulated as probability distributions over pure strategy profiles satisfying certain linear constraints. The resulting linear feasibility program has size polynomial in the size of the normal form representation of the game. However, the size of the normal form representation grows exponentially in the number of players. This is where using the structure of the payoffs may be very helpful. But even when the game itself is compact, the representation of the CE can require exponential space to specify.

Papadimitriou and Roughgarden (2008) and Jiang and Leyton-Brown (2011) showed that we can compute a sample CE in poly-time using the *Ellipsoid Against Hope* method. Stein, Parrilo and Ozdaglar raised concerns about this algorithm, since using the ellipsoid method to certify the (in)feasibility of an LP requires an upper radius $R$ and a minimum volume $v$ which depend on the maximum encoding size of the constraints in the LP. In the original Papadimitriou–Roughgarden construction, the cuts returned by the separation oracle are convex combinations of the dual constraints and can have encoding size strictly larger than any single original constraint. As a consequence, the infeasibility of the LP formed by the generated cuts is not guaranteed to be detectable using the bounds $R, v$ derived from the original constraints, and the algorithm may fail to correctly certify the existence of an exact CE. Jiang and Leyton-Brown (2011) fix this issue by constructing a separation oracle whose cuts correspond directly to *pure strategy profiles* rather than to mixtures, which bounds the encoding size of each cut and restores correctness of the ellipsoid certification.

## Notations

**Setup.** Consider $n$ players with action sets $A_i$, $|A_i|=m_i$, $m=\max_i m_i$. A pure strategy profile is $a=(a_1,\ldots,a_n)\in A=\prod_i A_i$ of size $M=\prod_i m_i$, with $a_{-i}\in A_{-i}$ denoting the partial profile of players other than $i$. Utilities are nonnegative integers $u_i(a)\in\mathbb{Z}_{\geq 0}$. A correlated distribution $x\in\mathbb{R}^M_{\geq 0}$ satisfies $\sum_a x(a)=1$; it is a product distribution (equivalently, a mixed-strategy profile) when $x(a)=\prod_i x_i(a_i)$ for some $x_i\in\Delta(A_i)$.

**Representation assumptions.** Following Papadimitriou–Roughgarden (2008), we assume the game satisfies: (i) *polynomial type* — $n$ and each $m_i$ are polynomial in the representation size; and (ii) *polynomial expectation property* — for any product distribution $x$ and any player $i$, the expected utility $\mathbb{E}_{a\sim x}[u_i(a)]=\sum_a u_i(a)\,x(a)$ is computable in time polynomial in the representation size. Congestion games, graphical games, polymatrix games, and action-graph games all satisfy both.

**Correlated equilibrium.** A distribution $x$ is a correlated equilibrium if, for every player $i$ and every pair $a_i,a'_i\in A_i$,

$$
\sum_{a_{-i}\in A_{-i}}\big[u_i(a_i,a_{-i})-u_i(a'_i,a_{-i})\big]\,x(a_i,a_{-i})\;\geq\;0. \tag{1}
$$

Writing (1) in matrix form gives $Ux\geq 0$, where $U\in\mathbb{R}^{N\times M}$ has $N=\sum_i m_i^2$ rows indexed by triples $(i,a_i,a'_i)$ and columns indexed by profiles; the entry $U_{(i,a_i,a'_i),a}$ equals $u_i(a_i,a_{-i})-u_i(a'_i,a_{-i})$ when $a$'s $i$-th coordinate is $a_i$, and $0$ otherwise. Together with $x\geq 0$ and $\mathbf{1}^\top x=1$, this defines the linear feasibility program for the set of CE.

**Sparsity of CE solutions.** Germano and Lugosi (2007) show that every $n$-player game admits a CE with support size at most $1+\sum_i m_i(m_i-1)=N+1-\sum_i m_i$. Such CE correspond to basic feasible solutions — vertices of the CE polyhedron — and have rational entries whenever utilities are rational.

## The Ellipsoid Against Hope Algorithm 

$$r \geq 2^{-O(n^2 L)}$$


