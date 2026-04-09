# NeurIPS Review: "Learning to Recommend in Unknown Games"

---

## Summary
The paper studies a moderator issuing action recommendations to strategic agents in an unknown normal-form game, observing only behavioral feedback (follow or deviate). Under two feedback models (best-response and quantal-response), the paper establishes: (1) a learnability separation -- QR feedback identifies utilities up to positive affine equivalence while BR does not, with a complete geometric characterization of the BR indistinguishability set; (2) an O(nmM log(1/epsilon)) learning algorithm under QR; (3) an O(nM log T) regret algorithm under both models via cutting-plane methods.

---

## Scores

| Criterion | Score |
|-----------|-------|
| **Soundness** | 3/4 — Several fixable but non-trivial errors (Theorem 7 assumption inconsistency, Algorithm 4 index formula, Lemma 3 converse) |
| **Novelty** | 3/4 — The learnability separation is genuinely new; the regret algorithm is an incremental adaptation of [9] |
| **Significance** | 2/4 — Interesting theoretical contribution but limited by exponential scaling, narrow regret notion, and gap between model and claimed applications |
| **Clarity** | 3/4 — Well-structured but genericity is undefined, computational costs are unaddressed, and several typos |
| **Overall** | 5/10 |
| **Confidence** | 4/5 |

---

## Detailed Assessment

### What the paper does well

The learnability separation (Theorems 4 and 5) is the paper's standout contribution. The insight that sign-based feedback (QR) suffices for identification while argmax-based feedback (BR) does not is clean and surprising. The proof via Lemma 1 is elegant. The counterexample for BR non-learnability is concrete and I verified it independently -- it checks out. The geometric characterization of the BR indistinguishability set through polarized polyhedra (Theorem 6) is mathematically substantive and could have independent value.

### What holds me back

**1. Technical errors that undermine confidence.** Theorem 7's boundedness assumption (c <= u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i}) <= C with c > 0) is inconsistent with the no-weakly-dominated-actions condition, which requires both positive and negative utility differences. Algorithm 4's index formula is wrong for heterogeneous action sets. Lemma 3's converse direction is incomplete. These are all fixable, but three errors in a 24-page theory paper is concerning. A NeurIPS paper should have airtight proofs.

**2. The regret result (Theorem 8) is incremental.** The cutting-plane framework is directly borrowed from Gollapudi et al. [9]. The new ingredient -- the CE-based separation oracle -- is a natural construction: if x(t) is a CE for guessed utilities w(t), any deviation reveals a separating hyperplane. Lemma 4 (the core geometric lemma) is cited from [9] without proof. The bound O(nM log T) then follows in a few lines. This is a competent application of existing machinery, not a methodological advance.

**3. The model assumptions limit significance.** (a) Agents observe the full joint distribution x(t) -- unrealistic for the claimed applications (traffic, marketplaces). (b) The moderator observes all individual actions -- strong in many settings. (c) Normal-form games with M = prod(m_i) exponential in n -- the framework is computationally infeasible for any realistic multi-agent system. (d) Myopic agents -- strategic agents could manipulate the learning process. The paper presents these as a "theoretical foundation" for AI recommendation systems, but the gap between the model and the motivating applications is very large and not acknowledged.

**4. Missing connections to information design.** The recommendation mechanism is a correlated signal and the CE constraint is precisely the obedience constraint from Bergemann and Morris (2016). This is not an obscure connection -- it is the same mathematical object. The omission of the entire information design / Bayesian persuasion literature is a significant gap in positioning for a game theory paper at NeurIPS.

**5. The regret notion is narrow.** Measuring deviation incentives captures compliance but says nothing about outcome quality. A moderator achieving zero regret may recommend a Pareto-dominated equilibrium every round. For the claimed applications (traffic, marketplaces), platforms care about efficiency, not just obedience.

**6. No computational discussion.** The center-of-gravity computation in Algorithm 5 is expensive (the standard approach requires O*(N^4) oracle calls per iteration). This is not mentioned. For a paper claiming algorithmic contributions, the absence of any computational complexity analysis is a gap.

### Minor issues
- "Qunatal-response" typo appears 4+ times
- Genericity assumption is never formally defined despite being critical
- The sigma labeling in Theorem 6 is undefined
- The relationship between the learning algorithm (Sec 4) and regret algorithm (Sec 5) is not discussed -- when should a practitioner use which?

---

## Questions for Authors

1. Can you fix the Theorem 7 assumption to be on |u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i})| and propagate the correction?
2. What is the formal genericity condition, and what measure does the non-generic set have?
3. What is the per-round computational cost of Algorithm 5?
4. How does this relate to information design (Bergemann-Morris 2016)? Your recommendation mechanism is a correlated signal with obedience constraints.
5. Can the results be extended to structured (compact) game representations to avoid the exponential M dependence?

---

## Decision

**Score: 5 — Marginally below acceptance threshold**

**Recommendation: Reject (weak)**

### Reasoning

The paper's strongest contribution -- the QR/BR learnability separation -- is novel and technically interesting. If this were a focused paper on learnability in inverse game theory (Sections 3-4 expanded, with the geometric characterization developed more fully), it could be compelling. However, in its current form:

- The technical errors (Theorem 7 assumption, Algorithm 4 index, Lemma 3 proof) need correction. For a theory paper, proof correctness is paramount.
- The regret result (Section 5), which occupies significant space, is an incremental application of [9] and does not meet the NeurIPS novelty bar on its own.
- The scalability limitation (exponential in n) combined with the strong modeling assumptions (full observation, normal-form, myopic agents) limits the significance for the NeurIPS audience, which values both theoretical depth and broader impact.
- The missing connection to information design is a positioning gap that reviewers in the GT community will notice.

A revised version that (a) fixes the errors, (b) formally defines genericity, (c) discusses computational complexity, (d) connects to information design, and (e) either extends to structured games or honestly scopes the contribution as pure inverse game theory would be a stronger submission. The learnability results are the core value -- I would encourage the authors to develop that direction more deeply.
