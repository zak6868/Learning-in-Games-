# Review: "Learning to Recommend in Unknown Games"
### Alanqary, Baba, Wu, Bayen (arXiv:2602.16998v2, March 2026)

---

## Summary

The paper studies a moderator who repeatedly recommends actions to strategic agents in an unknown normal-form game. Two behavioral feedback models are considered: best response (BR) and quantal response (QR). The main contributions are: (1) a learnability separation -- utilities are identifiable (up to positive affine equivalence) under QR but not under BR, with a complete geometric characterization of the BR indistinguishability set via normal fans of polarized polyhedra; (2) a constructive learning algorithm with O(nmM log(1/epsilon)) sample complexity under QR feedback; (3) a cutting-plane-based online algorithm achieving O(nM log T) regret under both feedback models.

---

## Strengths

**S1. Clean problem formulation and a genuine separation result.** The dichotomy between QR and BR learnability (Theorems 4 vs. 5) is the paper's most compelling contribution. The proof of Theorem 4 is elegant: Lemma 1 (sign information over the positive orthant identifies a mixed-sign vector up to scale) is applied to pairwise utility differences, and a triangular identity argument forces uniform scaling across actions. The counterexample in Theorem 5 is concrete and I have verified it independently -- the two games are indeed non-equivalent yet share identical best-response regions (all boundary slopes x_2/x_1 = 2, 1, 2/3 match between the two games).

**S2. Geometric characterization of BR indistinguishability.** The connection between best-response indistinguishability and normal fans of polarized polyhedra (Theorem 6, via Proposition 1 using Minkowski sums P_i + C_i^o) is mathematically substantive and may have independent value in inverse game theory and computational geometry.

**S3. Well-structured paper.** The logical progression -- model, learnability, constructive algorithm, online regret -- is clear. Definitions are precisely stated and the informal theorem statements in the introduction provide a useful roadmap.

---

## Errors

**E1. Theorem 7 -- boundedness assumption is inconsistent with non-dominance (Critical).**
The assumption in Theorem 7 states:
> c <= u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i}) <= C for all agents i, action pairs a_i != a'_i, and profiles a_{-i}

With 0 < c, this forces ALL utility differences to be strictly positive. But the no-weakly-dominated-actions assumption requires w_i(a_i, a'_i) to have both strictly positive AND strictly negative entries. These are contradictory.

The intended assumption is almost certainly on absolute values: c <= |u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i})| <= C. This is what the proof actually uses -- it needs |w_p|, |w_j| >= c for the binary search to work and bounded ratios for error propagation. As stated, the theorem is technically wrong.

**E2. Algorithm 4 -- index formula l(i,j) is incorrect for heterogeneous games.**
The offset formula l(i,j) = (i-1)(m_i - 1)d_i + (j-2)d_i + 1 uses agent i's action count m_i in the cumulative offset for agents 1 through i-1. The correct formula should be l(i,j) = [sum_{k=1}^{i-1} (m_k - 1)d_k] + (j-2)d_i + 1. The current formula is only correct when all agents have identical action sets.

**E3. Lemma 3 (Appendix B.2) -- converse direction is unjustified.**
The proof claims that not observing a'_i in T = m_i * exp(beta*C) * ln(1/delta) samples verifies a'_i not in QR(x, a_i) with probability 1 - delta. This is a one-sided test presented as two-sided verification. When phi_i(a_i, a'_i, x) is positive but very small, the probability P_x(a'_i | a_i) can be near 1/(m_i * exp(beta*C)), making membership vs. non-membership indistinguishable from finite samples. The bounded-away-from-zero assumption creates a gap that rescues this argument, but the proof never computes this gap or invokes the assumption for the converse direction.

---

## Technical Gaps and Concerns

**G1. "Genericity" is never defined.**
The genericity assumption is load-bearing in Theorem 4 (Step 2 requires linear independence of u_i(a,.) - u_i(c,.) and u_i(b,.) - u_i(c,.)) and in the geometric characterization (Theorem 6). The paper never formally specifies what genericity means. Presumably it requires that a certain finite set of polynomial equalities does not hold (full Lebesgue measure). This should be stated precisely, along with a discussion of what happens with approximately degenerate games.

**G2. Lemma 1's application in Theorem 4 -- compressed argument.**
The jump from "the moderator can query any x in Delta(A)" to "the sign feedback is equivalent to querying all y in R^{d_i}_+" (Step 1) relies on: (a) every direction in R^{d_i}_+ being achievable as x(a_i, .) for some x in Delta(A), and (b) the marginal Pr_x(a_i) being positive so the feedback is actually observed. Both are true but the argument is compressed to one sentence. The paper should also note that Lemma 1 should properly state x in R^d_+ \ {0} (x = 0 gives trivial sign information).

**G3. Myopic agent behavior is assumed but not prominently stated.**
The cutting-plane algorithm (Algorithm 5) relies on agents responding myopically to each recommendation: playing a best response or quantal response to the current x(t), not strategically over time. A forward-looking agent could deviate to a non-optimal action to mislead the moderator's learning, breaking the separation oracle. This assumption is implicit in the feedback model definitions (Definitions 2-3) but should be discussed explicitly, especially given the paper's emphasis on strategic environments.

**G4. Computational complexity is entirely absent.**
Algorithm 5 requires computing cg(C^{t-1} + (1/T)B) -- the center of gravity of a polytope in R^N. This is computationally expensive (O*(N^4) oracle calls per iteration via known algorithms, with N potentially in the thousands even for small games). Computing CE(w(t)) requires solving a linear program. The paper contains no discussion of computational feasibility. For practical purposes, one would need to replace the centroid computation with an ellipsoid method or Vaidya-type algorithm, changing the constants.

**G5. Error propagation in Theorem 7 hides condition-number dependence.**
The bound |hat{w}_k - w_k/w_p| = O(epsilon) absorbs the ratio C/c into the constant. Since C/c could be large, the actual error is O((C/c)^2 * epsilon). This matters for the interpretation of the log(1/epsilon) complexity -- the effective precision is worse than epsilon by a factor depending on the condition number.

**G6. Relationship between learning (Section 4) and regret (Section 5) is unclear.**
If the moderator can learn the game (Section 4), it can compute a CE and achieve zero regret thereafter. So why is the separate regret algorithm needed? Presumably because (a) learning requires QR feedback specifically, (b) learning requires many samples per recommendation (Lemma 3: m_i * e^{beta*C} * ln(1/delta) samples each), while Algorithm 5 needs one sample per round and works with both feedback models. This tradeoff should be discussed explicitly.

---

## Missing Related Work

**R1. Information design and Bayesian persuasion.** The paper's model is a special case of information design in games (Bergemann and Morris, AER 2016; Taneva, AEJ Micro 2019). The recommendation mechanism x in Delta(A) is a correlated signal, and the CE constraint is exactly the obedience constraint from the information design literature. The connection to Kamenica and Gentzkow (AER 2011) on Bayesian persuasion is also direct. This entire literature is absent from the paper and should be discussed.

**R2. Multi-agent inverse RL.** While single-agent IRL is cited [19, 11], there is substantial work on multi-agent IRL (e.g., Natarajan et al., 2010; Lin et al., 2019; Yu et al., 2019) that addresses inferring utilities from behavior in games. This omission weakens the positioning.

**R3. No-regret learning in games.** There is a vast literature on agents using no-regret learning to converge to CE (Hart and Mas-Colell, 2000; Blum and Mansour, 2007). The distinction -- here the moderator learns, there the agents learn -- deserves discussion.

**R4. Mechanism design without money.** The moderator has no monetary instruments, only recommendations. The literature on mechanism design without money (matching, allocation) and information as incentive provision is relevant.

---

## Regret Notion -- Limitations

**L1. Deviation incentive != outcome quality.** The regret (Eq. 5-6) measures agents' incentives to deviate, not social welfare, moderator revenue, or any outcome quality metric. An algorithm with zero regret may consistently recommend a Pareto-dominated CE. The moderator has no explicit objective beyond compliance.

**L2. Low average regret does not imply convergence to CE.** Cumulative regret O(nM log T) implies average per-round regret O(nM log(T)/T) -> 0, but this is convergence on average, not pointwise. Individual rounds may have large deviation incentives.

**L3. Strategic manipulation.** The regret bound assumes myopic agents. Strategic agents could inflate or deflate deviation signals to manipulate the moderator's learning trajectory.

---

## Scalability

The complexity results scale with M = prod_i m_i, which is exponential in n. For n = 10 agents with m_i = 2, we have M = 1024 and the parameter dimension N is in the thousands. For the motivating applications (thousands of drivers, millions of marketplace users), the framework is computationally infeasible. The paper does not discuss whether structured game classes (congestion games, potential games, graphical games) admit polynomial-time algorithms or better bounds. This is a significant gap.

---

## Presentation Issues

| Issue | Location |
|-------|----------|
| "qunatal-response" (typo) | Lines 62, 337, 356, 659, 750 |
| "form" should be "from" | "not learnable form the best-response" (Thm 5 proof) |
| "forTrounds" missing space | Section 1 |
| sigma(ext(Q_i)) undefined | Theorem 6 -- labeling bijection never formally defined |
| Definition 4 uses {A} not {A_i} | Inconsistent with notation elsewhere |
| Algorithm 4 notation introduced after the algorithm box | q(t)(i,j) and l(i,j) should be defined first |
| Lemma 5 proof is near-tautological | The real content (oracle validity) is asserted without detailed proof |

---

## Questions for the Authors

1. Can you state the genericity assumption formally? What is the measure of the set of games that fail it?

2. In Theorem 7, did you intend the boundedness assumption to be on absolute values of utility differences? As stated, it contradicts the no-weak-dominance condition.

3. What is the computational cost per round of Algorithm 5? Have you considered computationally efficient alternatives to the center-of-gravity oracle?

4. The regret notion captures deviation incentives but not outcome quality. Have you considered alternative objectives (social welfare regret, best-CE regret)?

5. How does the framework extend to structured games (congestion, potential, graphical) where M is exponential but the game has compact representation?

6. What happens if agents are forward-looking and strategically manipulate their responses to influence the moderator's learning?

7. Could you discuss the connection to information design (Bergemann-Morris) and Bayesian persuasion (Kamenica-Gentzkow)? The recommendation mechanism is a correlated signal and the CE constraint is an obedience constraint.

---

## Overall Assessment

A technically competent paper with clean theoretical results on a well-defined problem. The QR vs. BR learnability separation is the strongest and most novel contribution. The geometric characterization of BR indistinguishability via polarized polyhedra is elegant and potentially of independent interest. The cutting-plane regret algorithm is a competent but incremental application of existing machinery from [9].

The paper is weakened by: (1) several technical errors (the Theorem 7 assumption inconsistency, the Algorithm 4 index formula, the incomplete Lemma 3 proof); (2) unrealistic modeling assumptions (full observation of x(t) by agents, full observation of all a*(t) by the moderator, normal-form games); (3) exponential scaling in the number of agents with no discussion of structured games; (4) a regret notion that captures compliance but not outcome quality; (5) significant omissions in related work (information design, Bayesian persuasion, multi-agent IRL). The practical applicability claims (traffic, marketplaces) are not supported by the model's assumptions and should be tempered.

**Recommendation:** The theoretical contributions merit publication, but the paper would benefit substantially from: fixing the technical errors identified above, formally defining genericity, discussing computational complexity, expanding the related work to include information design, and tempering the practical applicability claims. The paper should be more transparent about being a foundational theoretical contribution rather than a practically applicable framework.
