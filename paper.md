# Calibration Does Not Compose, Types Destroy Vagueness: The Hidden-Markov and Fuzzy Primitives Missing from System-One Decision Models

**Draft v0.3 — math-complete, citations verified 2026-09-17.** Section numbering follows `outline.md` exactly. Every lemma is stated formally with a proof sketch in-line and a full proof in Appendix B. Pseudocode is in §6.6 and Appendix C. Items marked **[PROSE]** are where the completing pass should expand. All external claims about Jev are quoted from TypeSafe's launch post and documentation (15 Sept 2026) and The Register's coverage (16 Sept 2026); see References.

---

## Abstract

System-one decision models — fast, non-generative networks that emit typed, calibrated probabilistic decisions for machine-to-machine pipelines (the public description of TypeSafe AI's Jev, trained by Reinforcement Learning for Calibrated Decisions, RLCD, is the case instance) — are audited by hop-level calibration on stationary held-out data. We show that such models omit two structural primitives: recursive belief over a latent state (the hidden-Markov primitive) and graded predicate membership (the fuzzy primitive). We prove two negative results. Lemma 1: when outcomes are coupled through a latent Markov regime, marginal calibration is not preserved under regime shift, trajectory error counts are overdispersed relative to the binomial baseline implied by hop-level calibration, and any permutation-invariant audit statistic (including expected calibration error) has zero power to detect this. Lemma 2: thresholding a typed score makes the composed pipeline action discontinuous; a single ε-perturbation of evidence produces a full action flip on a set of Θ(ε) measure, every downstream hop then sees an O(1) input change, and under common-mode evidence coupling the all-hops-flip event has Θ(ε) rather than Θ(ε^H) measure — whereas a fuzzy membership composed by a t-norm and defuzzified once by a Lipschitz map keeps the action Lipschitz in the evidence. We propose the Principle of Deferred Crispification and a reference architecture, Belief-State Fuzzy System-One (BSF-S1), whose output object retains a membership vector, a calibrated distribution over the degree, and a scaled-likelihood vector over regimes, collapsing exactly once at the actuator at O(K² + KM) cost per step. We specify five experiments and two new audit metrics (trajectory calibration error and action-mass sensitivity) that convert the analysis into evidence.

---

## 1. Introduction

**[PROSE: expand each bullet to a paragraph; keep the claims below verbatim as the load-bearing sentences.]**

**1.1** System-one decision models are positioned as the cheap, fast layer of machine-to-machine AI: no text generation, a typed output schema, and a calibrated probability attached to each typed value. The case instance is TypeSafe AI's Jev, released in early access on 15 September 2026 and described by its authors as "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out" [1]. Its output schema is defined in advance through three primitives — Choice (an enum with a probability per option), Score (a bounded scalar), and Noul — each returned with a confidence score [2]; the published example is a routing decision `{"billing": 0.08, "technical": 0.85, "sales": 0.07}` with confidence 0.82 [3]. Its training method is named Reinforcement Learning for Calibrated Decisions (RLCD), optimising for "answers with epistemically honest probabilities" [1]. The positioning is quantitative: end-to-end response times of 70–500 ms, "40×–200× faster" than frontier LLMs, input priced at $0.042 per million tokens with output "too cheap to meter" [1], which The Register computes as 238× cheaper than a top-tier model [3]. The analysis in this paper applies to the class; Jev is the announced instance.

**1.2** The field's audit practice is hop-level: expected calibration error (ECE) or a proper score computed on an i.i.d. held-out sample of (score, outcome) pairs, plus per-decision accuracy benchmarks. TypeSafe's own documentation states the scope precisely: "Calibration is measured across groups of predictions; it does not guarantee that an individual answer is correct" [2]. This paper takes that sentence seriously and asks what *groups* the measurement is silent about.

**1.3** Problem statement. Two omissions:
(i) The posterior is memoryless: P(d | o) is calibrated against a frozen marginal prior over the world's latent state; there is no recursion that carries belief about that state from one decision to the next.
(ii) The output type is a crisp enum over predicates that are, for a positive fraction of inputs, intrinsically borderline; the model is forced to express ontological vagueness as epistemic uncertainty.

**1.4** Thesis. Both omissions are structural: they concern what object the model emits and how the harness composes it, not the capacity of the network that produces it. They survive any scaling of the network, and they are repairable at O(K² + KM) serving cost per step, which is negligible against a single forward pass.

**1.5** Contributions.
- **C1** Diagnosis of the temporal hole and Lemma 1 (calibration does not compose): prior-ratio mismatch under regime shift; overdispersion of trajectory errors with a closed-form inflation factor; permutation-invariance blindness of ECE.
- **C2** Diagnosis of the vagueness hole and Lemma 2 (cliff cascade): discontinuity, Θ(ε) flip mass with O(1) jumps, cascade with Θ(1) conditional flip probability, and Θ(ε) all-hops-flip under common-mode coupling; Lipschitz continuity of the fuzzy alternative.
- **C3** The Principle of Deferred Crispification.
- **C4** BSF-S1: output object O(x) = (μ, Q, Δb), composition algebras for each component, an exact forward filter over the scaled-likelihood component, and a single actuator collapse.
- **C5** An experimental program and two audit metrics: trajectory calibration error (TCE) and action-mass sensitivity (AMS).

---

## 2. Background and Related Work

**2.1 System-one decision models and RLCD.** What is public [1, 2, 3]: the model "evaluates a state and returns typed answers and probabilities"; probabilities "are optimized against outcomes to reflect uncertainty"; all outputs of a query are generated in parallel in a single pass rather than autoregressively; the model "does not write replies, produce code, or generate explanations of its reasoning." Two further statements from the launch material are load-bearing for this paper. First, the recommended integration pattern is to "ask independent questions together … combine the answers with deterministic checks in code, then route the case" [2] — that is, the harness thresholds the returned probabilities; §5.5 names this the double collapse. Second, the authors observe that "the most reliable real-world workflows tend to have many independent, decomposed questions, with fine-grained behavior that's dependent on probabilities instead of discrete decisions. The end result is discrete branching" [1] — an accurate description of the pipeline whose composition properties §3 and §4 analyse. Jev's weights, architecture and training data are closed; nothing here is an empirical claim about Jev's internals. Every statement is about the *class* of memoryless typed calibrated heads, of which Jev is the announced instance. The launch material also claims consistency — "returns similar answers for similar inputs" [1] — which is a Lipschitz-type property of the *score*; Lemma 2 concerns what happens to that property once the score is thresholded.

**2.2 Hidden Markov models and Bayesian filtering.** Latent state s_t ∈ {1,…,K}, transition A ∈ ℝ^{K×K} with A_{ij} = P(s_{t+1}=j | s_t=i), stationary distribution π (π^T A = π^T), emission P(o_t | s_t). The forward filter:

  b_t(s) ∝ P(o_t | s) · Σ_{s'} A_{s's} b_{t−1}(s')        (2.1)

is exact inference given (A, B). Regime-switching econometrics [Hamilton 1989] is the same recursion with Gaussian emissions; POMDPs [Kaelbling et al. 1998] add actions. The hybrid HMM/neural-network literature [Bourlard & Morgan 1994, ch. 7] established that a discriminatively trained network emitting P(s | o) can be converted to an emission term via the *scaled likelihood* P(o | s) ∝ P(s | o)/π(s); this is the bridge we use in §3.5.

**2.3 Fuzzy sets.** A type-1 fuzzy set on X is a membership μ: X → [0,1] [Zadeh 1965]. Conjunction/disjunction are truth-functional via a t-norm T and t-conorm S (min/max, product/probabilistic sum, Łukasiewicz). Mamdani control [Mamdani & Assilian 1975] aggregates rule consequents into an output membership and defuzzifies once (centroid) at the plant input. A type-2 fuzzy set [Zadeh 1975] attaches to each x a *distribution over the degree*, separating "how much x is M" from "how sure we are about how much."

**2.4 Calibration and proper scoring.** A score f is (marginally) calibrated under distribution P if E_P[Y | f(X) = p] = p for all p. ECE is the L1 gap between this conditional mean and p, binned [Guo et al. 2017]. A scoring rule is strictly proper if the true distribution uniquely minimises expected score [Gneiting & Raftery 2007]. Calibration of a marginal says nothing about the joint distribution of several decisions; this distinction is the entire content of §3.

**2.5 Scope and non-goals.** No generative models; no claims about network architecture, scale, context, or caching; no claims about cognition (the "system-one" label is Kahneman's framing only). We study the decision *output object* and its *composition* across hops.

---

## 3. Analysis I — The Temporal Hole (Missing Hidden Markov)

### 3.1 The memoryless calibrated posterior

Let the world have latent regime s_t following the Markov chain of §2.2, evidence o_t with P(o_t | s_t), and decision target d_t ∈ D with P(d_t | o_t, s_t). The system-one head is f(o) ≈ P_train(d | o) where the training distribution mixes regimes with the stationary weights:

  P_train(d | o) = Σ_s P(d | o, s) · P_train(s | o),   P_train(s | o) ∝ P(o | s) π(s).      (3.1)

RLCD-style objectives can certify that f is calibrated under P_train. They cannot certify anything about P(d | o) under a distribution in which the regime weights differ, because the objective never sees s and never sees more than one time step at once.

### 3.2 Regime shift as prior death

Suppose at serving time the regime marginal is π′ ≠ π (a sojourn in a rare regime, or a permanent drift). The true conditional is

  P′(d | o) = Σ_s P(d | o, s) · P′(s | o),   P′(s | o) ∝ P(o | s) π′(s).      (3.2)

The head f still outputs (3.1). Its score is "correct" with respect to a prior that no longer holds. A held-out set drawn from P_train has calibration error identically zero for a perfectly trained f, so the audit is structurally blind to (3.2).

### 3.3 Lemma 1 (Calibration does not compose)

**Setting.** Binary target d ∈ {0,1} for clarity (the K-ary case is identical componentwise). Assume the standard HMM conditional independences: o_t ⊥ o_{t′} | s_t, s_{t′}; d_t ⊥ (o_{t′}, d_{t′}) | (s_t, o_t). Let f be perfectly calibrated under P_train. Let E_t = 1{decision at t is wrong} and e_s = P(E_t = 1 | s_t = s) the regime-conditional error rate of the fixed decision rule, ē = Σ_s π(s) e_s.

**Statement.**

(i) *Prior-ratio mismatch.* Under serving marginal π′, the pointwise miscalibration of f is

  Δ(o) := P′(d=1 | o) − f(o) = Σ_s P(d=1 | o, s) · [P′(s | o) − P_train(s | o)],      (3.3)

which is nonzero whenever the regime-conditional answers P(d=1 | o, s) differ across s and π′ ≠ π. The expected calibration error under P′ is bounded below by |E_{P′}[Δ(o)]| and the expected calibration error under P_train is zero; no statistic computed on P_train-samples has power against (3.3).

(ii) *Overdispersion.* For a two-regime chain A = [[1−a, a],[b, 1−b]] with λ := 1 − a − b ∈ (−1, 1), the error indicators satisfy

  Cov(E_t, E_{t+k}) = λ^k · Var_π(e),   Var_π(e) := Σ_s π(s) e_s² − ē².      (3.4)

Hence the window error count N_T = Σ_{t=1}^T E_t has

  Var(N_T) = T ē(1−ē) + 2 Var_π(e) Σ_{k=1}^{T−1} (T−k) λ^k,      (3.5)

and for large T the variance inflation factor over the binomial baseline implied by hop-level calibration is

  φ := lim_{T→∞} Var(N_T) / (T ē(1−ē)) = 1 + (2λ / (1−λ)) · Var_π(e) / (ē(1−ē)).      (3.6)

φ > 1 iff λ > 0 and Var_π(e) > 0. Errors cluster on runs whose lengths are the sojourn times of A, geometric with means 1/a and 1/b. For the K-regime chain with spectral decomposition A^k = 1π^T + Σ_{i=2}^K λ_i^k P_i, (3.4) becomes Cov(E_t, E_{t+k}) = Σ_{i≥2} λ_i^k · e^T diag(π) P_i e.

(iii) *All-correct probability.* P(N_T = 0) = E[∏_t (1 − e_{s_t})] ≥ (1−ē)^T whenever λ ≥ 0, with strict inequality iff λ > 0 and Var_π(e) > 0. The trajectory distribution is heavier in both tails than the independent baseline: more all-correct runs *and* more many-error runs.

(iv) *Permutation-invariance blindness.* ECE, any proper score averaged over hops, and any statistic that is a function of the empirical distribution of (f(o_t), y_t) pairs are invariant under permutation of t. The clustered sequence and its random shuffle have identical such statistics while (3.5) differs. Therefore hop-level audits have exactly zero power to detect (ii)–(iii).

**Proof sketch.** (i) is (3.1) minus (3.2). (ii): condition on (s_t, s_{t+k}); by the independences, E[E_t E_{t+k}] = Σ_{s,s′} π(s) (A^k)_{ss′} e_s e_{s′}; substitute A^k = 1π^T + λ^k (I − 1π^T), valid for any 2×2 stochastic matrix; the 1π^T term gives ē², the remainder gives λ^k Var_π(e). Sum covariances for (3.5); the geometric series gives (3.6). (iii): the map x ↦ x^T is convex on [0,1]; at λ = 1 (frozen regime) Jensen gives Σ_s π(s)(1−e_s)^T ≥ (Σ_s π(s)(1−e_s))^T; for 0 < λ < 1 interpolate via the covariance expansion of the product (Appendix B). (iv): ECE is computed from the multiset of pairs; multisets are permutation-invariant. ∎

### 3.4 Consequence

Risk in a pipeline is a trajectory-level object: the quantity a downstream consumer cares about is the distribution of N_T (how many wrong decisions in this window, and whether they arrive together), not ē. Hop-level calibration certifies ē and is silent on φ. Two models with identical ECE can differ in φ by an arbitrary factor.

### 3.5 Repair primitive

Treat the network as an *emission model* and run the exact filter.

1. Train a regime head g(o) ≈ P_train(s | o) (K-way softmax) alongside per-regime decision heads h_s(o) ≈ P(d | o, s).
2. Convert to a scaled likelihood: Δb_t(s) := g_s(o_t) / π(s), normalised to sum to 1 (the normaliser is P(o_t)/Σ, which cancels in (3.7)).
3. Filter:

  b̃_t(s) = Σ_{s′} A_{s′s} b_{t−1}(s′)  (predict);   b_t(s) = b̃_t(s) Δb_t(s) / Σ_{s″} b̃_t(s″) Δb_t(s″)  (update).      (3.7)

4. Decide: P(d_t | o_{1:t}) = Σ_s h_s(o_t) b_t(s).      (3.8)

Given a correct (A, g, h), (3.8) is the true conditional, hence calibrated on any window, under any π′, and its implied joint over N_T is the true joint — trajectory calibration is a consequence of exact inference, not an additional objective. The cost of (3.7) is O(K²) per step.

`★ Insight ─────────────────────────────────────`
Why the scaled likelihood and not the posterior: two posteriors P(s|o_1) and P(s|o_2) each already contain π. Multiplying them counts π twice. Dividing each by π(s) removes it, leaving objects proportional to P(o_i|s), which *do* multiply under conditional independence. This is the reason the carried object in the output specification (§5.3) is Δb, not b.
`─────────────────────────────────────────────────`

---

## 4. Analysis II — The Vagueness Hole (Missing Fuzzy Logic)

### 4.1 Two distinct "not-sures"

Epistemic uncertainty: the world is in exactly one crisp state; we do not know which; a probability over states sums to 1. Ontological vagueness: the predicate itself has no sharp boundary ("this ticket is *urgent*"); membership is a degree; degrees over different predicates need not sum to anything. A single number in [0,1] attached to an enum value conflates the two.

### 4.2 Typed enums as frozen ontology commitments

A schema `priority ∈ {low, medium, high}` commits at design time to sharp boundaries that the data do not have. An input that is genuinely borderline between medium and high is best described by memberships (0, 0.5, 0.5); a calibrated probabilistic head trained on inconsistent labels for such inputs learns p ≈ (0, 0.5, 0.5) *as a probability*, which downstream harnesses read as "the model does not know" and route to escalation or retry. The information "this is a borderline case and will remain one" is destroyed, and the cost of that destruction is paid on every borderline input forever.

### 4.3 Lemma 2 (Cliff cascade)

**Setting.** Evidence x ∈ X ⊂ ℝⁿ with distribution P_X. H hops; hop h has score f_h: X → [0,1], L-Lipschitz, and threshold τ_h ∈ (0,1). Regularity: the pushforward of P_X under f_h has a density p_h that is positive and finite at τ_h, and ‖∇f_h‖ ≥ c > 0 on f_h^{-1}(τ_h).

Crisp pipeline: a_h(x) = 1{f_h(x) ≥ τ_h}; pipeline outcome G(x) = ∧_h a_h(x) (conjunctive gating; any non-constant Boolean combination gives the same conclusions). *Sequential* crisp pipeline: hop h+1's evidence includes a_h, x_{h+1} = (x, a_h).

Fuzzy pipeline: μ_h(x) = f_h(x) read as a membership; compose by a t-norm T that is 1-Lipschitz in ℓ¹ on [0,1]^H (min and product both are); defuzzify once at the actuator by D: [0,1] → 𝒜, L_D-Lipschitz; action u(x) = D(T(μ_1(x), …, μ_H(x))).

**Statement.**

(i) *Discontinuity vs Lipschitz.* G is discontinuous on ∪_h f_h^{-1}(τ_h), a set of P_X-positive ε-neighbourhood for every ε > 0. The fuzzy action satisfies, for all x, x′,

  |u(x) − u(x′)| ≤ L_D · Σ_h |μ_h(x) − μ_h(x′)| ≤ L_D · H · L · ‖x − x′‖.      (4.1)

(ii) *Θ(ε) flip mass with O(1) jumps.* Define the flip set F_ε = {x : ∃ x′, ‖x′ − x‖ ≤ ε, G(x′) ≠ G(x)}. Under the regularity assumptions, P_X(F_ε) = Θ(ε) as ε → 0 with constants c·p_h(τ_h) ≤ (coefficient) ≤ L·p_h(τ_h) per hop. On F_ε the worst-case action change is 1. For the fuzzy pipeline the worst-case action change is ≤ L_D H L ε *at every x*; there is no set on which it is O(1).

(iii) *Cascade.* In the sequential crisp pipeline, conditional on a_h flipping, hop h+1's evidence changes by a fixed amount δ_h = ‖(x,1) − (x,0)‖ independent of ε, so

  P(a_{h+1} flips | a_h flips) ≥ P_X(|f_{h+1} − τ_{h+1}| ≤ c δ_h) =: q_{h+1} = Θ(1).      (4.2)

The expected number of downstream flips triggered by one ε-perturbation is Σ_{h} ∏_{j≤h} q_j, which does not vanish as ε → 0. In the sequential fuzzy pipeline hop h+1 receives μ_h, whose change is ≤ Lε, so the total action change is ≤ L_D (Σ_h L^h) ε — geometric in L but still O(ε).

(iv) *Common-mode coupling.* If the hops' scores share a latent scalar z (e.g., the regime of §3), f_h(x) = φ_h(z, ξ_h) with each φ_h monotone in z, and the perturbation acts on z, then P(all H hops flip) = Θ(ε), whereas under independent per-hop perturbations P(all H hops flip) = ∏_h Θ(ε) = Θ(ε^H).

**Proof sketch.** (i) Step functions composed with continuous maps are discontinuous on the preimage of the threshold; (4.1) is the chain of Lipschitz bounds, using |∏a_h − ∏b_h| ≤ Σ|a_h − b_h| on [0,1]^H (telescoping) and |min a − min b| ≤ max|a_h − b_h|. (ii) F_ε ⊇ {x : τ_h − cε ≤ f_h(x) < τ_h or τ_h ≤ f_h(x) < τ_h + cε for some h} by moving along ∇f_h; F_ε ⊆ {x : |f_h(x) − τ_h| ≤ Lε for some h} by Lipschitzness; both have mass Θ(ε) by the density assumption. (iii) is (4.2) directly. (iv) The event "z crosses the critical value z* at which all φ_h cross their thresholds" is an interval of z of width Θ(ε); by monotonicity all hops flip together. ∎

**Remark on defuzzifiers.** (4.1) requires D Lipschitz. The centroid D(μ_out) = ∫y μ_out(y)dy / ∫μ_out(y)dy is Lipschitz in μ_out (sup norm) on the set {∫μ_out ≥ m_0 > 0}; the architecture must enforce a floor m_0 (a regularised centroid). Mean-of-maxima and height defuzzifiers are *not* continuous and would reintroduce a cliff; the principle in §5 therefore names the defuzzifier.

### 4.4 The wrong algebra

Probability is not truth-functional: P(A ∧ B) is not determined by P(A) and P(B) without an independence assumption. A t-norm is truth-functional by construction: T(μ_A, μ_B) *is* the degree of "A and B." When a harness multiplies typed probabilities across hops (naive Bayes) it silently asserts independence of predicates that are typically about the same object; when it takes max it asserts nothing coherent. The product t-norm coincides numerically with independent multiplication but is licensed by a different semantics; min is the only idempotent t-norm, and idempotence is exactly the property that prevents the same predicate from being double-counted when it enters two rules (product gives μ², the origin of the "confident-by-repetition" failure).

### 4.5 Repair primitive

Emit a membership vector μ(x) ∈ [0,1]^M over the M linguistic predicates the schema cares about, with no sum constraint, and a type-2 layer Q[μ] — a distribution over [0,1]^M (in practice, per-predicate Beta parameters or quantiles) — carrying calibrated uncertainty about the degree. Calibration of Q is quantile calibration: P(μ_true,m ≤ Q_m^{-1}(α)) = α, audited by CRPS or the pinball loss against elicited degrees. The two axes are never merged.

---

## 5. The Principle of Deferred Crispification

### 5.1 Three structures, three algebras

| Structure | Object | Composition algebra | Audit |
|---|---|---|---|
| Vagueness | μ ∈ [0,1]^M | t-norm / t-conorm, rule base | agreement with elicited degrees |
| Temporal coupling | Δb ∈ Δ^K (scaled likelihood) | pointwise product with predicted belief (Bayes filter) | trajectory calibration (TCE) |
| Uncertainty about degree | Q[μ] | mixture / posterior update | proper scores (CRPS, pinball) |

### 5.2 Principle statement

*No structure may be collapsed before the actuator. Collapse — defuzzify, threshold, sample — happens exactly once, with full decision context (belief, cost matrix, and all memberships in hand).*

Each early collapse is a specific information loss: thresholding μ discards the degree (Lemma 2); marginalising b_t discards the regime (Lemma 1); collapsing Q to its mean discards whether the degree is known.

### 5.3 Output object

  O(x_t) = ( μ_t ∈ [0,1]^{K×M},  Q_t,  Δb_t ∈ Δ^K ).      (5.1)

μ_t is emitted *per regime* (row s is the membership vector the model would report if the regime were s); Q_t likewise. This is the fully deferred form: mixing over regimes is itself a collapse and is left to the actuator (see §6.3 for the open problem). When K×M is too large, the harness may pass the belief-weighted mixture μ̄_t = Σ_s b_t(s) μ_t^{(s)} — a convex combination, hence still a membership vector — at the cost of an early collapse over s, which must be declared.

### 5.4 Interface contract for harnesses

- Downstream hops consume O uncollapsed.
- Rule bases compose μ with t-norms/t-conorms; they never threshold.
- Filters compose Δb with A by (3.7); they never receive a posterior from another hop (a posterior would double-count π).
- Scorers audit Q with proper scores and audit the filter with TCE.
- Only the actuator calls `collapse`.

### 5.5 Precedent audit

**[PROSE]** Fuzzy control: defuzzifies once at the plant input; half a century of industrial deployment. Probabilistic forecasting: keeps full predictive distributions to the decision-maker; proper scoring rules exist precisely to audit the uncollapsed object. Regime-switching econometrics: carries the filtered regime probability, never a point regime. Contemporary agent stacks: argmax per hop, then threshold again in the harness — a *double* collapse (model threshold + harness threshold) that Lemma 2 shows compounds. The Jev integration guidance — combine typed probabilities "with deterministic checks in code, then route" [2] — is this pattern stated as best practice.

---

## 6. Architecture: Belief-State Fuzzy System-One (BSF-S1)

### 6.1 Emission block

One shared encoder φ(o_t) feeding: a regime head g(o_t) ∈ Δ^K; per-regime decision heads h_s(o_t) ∈ Δ^{|D|}; per-regime membership heads μ^{(s)}(o_t) ∈ [0,1]^M (sigmoid, no softmax); per-regime type-2 heads giving Beta parameters (α_m^{(s)}, β_m^{(s)}) so Q_m^{(s)} = Beta(α, β) with mean μ_m^{(s)}.

### 6.2 Transition block

A ∈ ℝ^{K×K}, row-stochastic, parameterised by row-wise softmax of a free matrix. Optionally fuzzy transition degrees: rows need not sum to 1 if the update is read as possibilistic; we keep the probabilistic reading so that (3.7) is exact Bayes.

### 6.3 Update block

Predict/update as (3.7) with Δb_t(s) = g_s(o_t)/π(s), π the stationary distribution of the current A (recomputed when A changes). **Open problem (stated, not solved):** b_t renormalises because it is a probability over a partition; μ_t^{(s)} does not because predicates are not a partition. The belief-weighted mixture μ̄_t is well-defined mathematically (convex), but whether "membership in M under uncertainty about which regime's M applies" should be the mixture, the min over plausible regimes, or something else is a semantic question (Zadeh's probability of a fuzzy event is one candidate). BSF-S1 sidesteps it by passing the per-regime matrix and letting the actuator choose.

### 6.4 Propagation

Pass O(x_t) = (μ_t, Q_t, Δb_t) and, for convenience, b_t itself (flagged as *not composable*).

### 6.5 Actuator

Single collapse with context: belief-weighted mixture over regimes, t-norm aggregation across the rule base, regularised centroid defuzzification, cost-sensitive threshold, optional sampling. See Algorithm 2.

### 6.6 Algorithm boxes

**Algorithm 1 — BSF-S1 step** (per observation; O(K² + KM + cost of encoder))

```
input:  o_t, b_{t-1} ∈ Δ^K, A, π, encoder φ, heads (g, h, μ, Q)
z      ← φ(o_t)
gs     ← g(z)                             # K-vector, P_train(s | o_t)
Δb     ← normalise(gs / π)                # scaled likelihood, K-vector
b_pred ← Aᵀ b_{t-1}                       # predict, O(K²)
b_t    ← normalise(b_pred ⊙ Δb)           # update, O(K)
S_t    ← −log Σ_s b_pred[s]·Δb[s]         # belief surprise (§6.7)
for s in 1..K:                            # O(KM)
    μ_t[s,:]  ← μ^{(s)}(z)
    Q_t[s,:]  ← (α^{(s)}(z), β^{(s)}(z))
    P_t[s,:]  ← h_s(z)                    # per-regime decision posterior
output: O_t = (μ_t, Q_t, Δb), b_t, P_t, S_t
```

**Algorithm 2 — Actuator (the single collapse)**

```
input:  O_t, b_t, P_t, rule base R = {(antecedent_r, consequent set C_r)},
        cost matrix Cst ∈ ℝ^{|D|×|D|}, t-norm T, floor m_0
# (a) belief-weighted mixture over regimes  — the ONLY place s is marginalised
μ̄     ← Σ_s b_t[s] · μ_t[s,:]            # M-vector, convex ⇒ in [0,1]^M
P̄     ← Σ_s b_t[s] · P_t[s,:]            # decision posterior, eq. (3.8)
# (b) fuzzy rule evaluation — truth-functional, no thresholds
for each rule r:
    w_r ← T(μ̄[m] for m in antecedent_r)  # firing degree
μ_out(y) ← max_r min(w_r, C_r(y))         # Mamdani aggregation over output universe Y
# (c) regularised centroid — Lipschitz by the floor
area   ← ∫ μ_out(y) dy
if area < m_0: μ_out ← μ_out + (m_0 − area)/|Y|   # uniform lift to the floor
u      ← ∫ y μ_out(y) dy / ∫ μ_out(y) dy   # continuous actuator signal
# (d) cost-sensitive decision from the (uncollapsed until now) posterior
d*     ← argmin_d Σ_{d'} Cst[d, d'] · P̄[d']
# (e) optional: expose Q̄ for the consumer's own risk policy
Q̄     ← Σ_s b_t[s] · Q_t[s,:]             # mixture of Betas, kept as a mixture
output: (u, d*, Q̄)
```

**Algorithm 3 — Training**

```
# Stage 0: pretrain encoder + regime head g on data with regime labels if any,
#          else initialise K by BIC over an HMM fitted on encoder features.
# Stage 1 (EM-style, fixed heads): Baum–Welch for A using Δb_t as emission
#          terms (scaled-likelihood HMM; Bourlard–Morgan).
# Stage 2 (joint): the forward recursion (3.7) is differentiable; minimise
#          L = −Σ_t log Σ_s h_s(o_t)[d_t] · b_t(s)      # decision NLL through the filter, eq. (3.8)
#            + λ_μ · Σ_t Σ_m pinball/CRPS(Q_t, μ_elicited,t,m)
#            + λ_A · KL(rows of A ‖ prior rows)         # keeps A from absorbing head error
#          Note: P(o_{1:T}) is not available with scaled likelihoods; the
#          decision NLL above is the correct computable surrogate.
# Stage 3: temperature-scale g and each h_s per regime on held-out windows;
#          verify TCE (§7) ≈ 0 before deployment.
```

**Serving-cost argument.** Beyond the encoder, Algorithm 1 costs O(K² + KM) multiply-adds and Algorithm 2 costs O(|R|·M + |Y|). With K ≤ 16 and M ≤ 64 this is under 10⁴ operations — orders of magnitude below one encoder pass — and requires no generative decode.

### 6.7 Failure-mode inventory

- *Filter misspecification.* Wrong A or wrong g ⇒ exact inference over a wrong model. Detection: belief surprise S_t (Algorithm 1) has a known expectation under the model; a CUSUM on S_t − E[S_t] raises a drift alarm.
- *K identifiability.* Regimes are latent; K is a modelling choice (BIC/held-out NLL). Over-specified K degrades gracefully (extra regimes stay near-empty); under-specified K reintroduces Lemma 1 within a merged regime.
- *Membership ground truth.* μ_elicited is elicitation-dependent (§8.2); Q is trained to absorb elicitation disagreement as degree-uncertainty rather than pushing μ toward 0.5.
- *Stale A under concept drift.* Detected by the surprise monitor; repaired by online re-estimation of A (Stage 1 can run incrementally).

---

## 7. Experimental Program

**Common synthetic generator (Appendix D).** K-regime chain with tunable λ (via a, b), per-regime emission N(m_s, Σ_s) in ℝⁿ, per-regime label function d = 1{w_s·o + c_s > 0} with regime-dependent (w_s, c_s) so that P(d | o, s) differs across s; a borderline population generated by drawing elicited degrees from Beta(α(o), β(o)).

**E1 Regime-shift stress.** Train a memoryless head on stationary π; evaluate window-conditional ECE on windows inside each regime and across a forced shift to π′. Prediction (Lemma 1(i)): stationary ECE ≈ 0, within-regime ECE ≈ |Δ(o)| averaged; BSF-S1 ≈ 0 in both.

**E2 Trajectory-level calibration.** Metric **TCE_w**: every model implies a joint predictive distribution for the window error count N_w (memoryless head ⇒ Poisson-binomial with its own p_t; BSF-S1 ⇒ forward-filter/backward-sample the joint). TCE_w is the discrepancy between the implied and realised distribution of N_w, computed as the PIT-histogram L1 distance plus the dispersion ratio φ̂ = Var_realised(N_w)/Var_implied(N_w). Prediction: memoryless head passes hop-ECE and shows φ̂ ≈ (3.6); observed error-run lengths ≈ geometric with means 1/a, 1/b; BSF-S1 φ̂ ≈ 1.

**E3 Cliff cascade.** Metric **AMS(ε)** = E_x[sup_{‖δ‖≤ε} |action(x+δ) − action(x)|] (gradient-ascent inner maximisation) and **flip mass FM(ε)** = P(action changes). Prediction (Lemma 2): crisp FM(ε)/ε → constant ∈ [c p(τ), L p(τ)], AMS = FM; fuzzy FM = 0 for continuous u, AMS(ε)/ε ≤ L_D H L. Sequential variant: measure downstream flips per upstream flip, prediction Θ(1) crisp vs O(ε) fuzzy. Common-mode variant: P(all flip) vs ε, slope 1 crisp-coupled vs slope H crisp-independent.

**E4 Composition-algebra ablation.** Same rule base composed with min, product, and naive-Bayes product of probabilities including one duplicated predicate. Prediction: naive-Bayes and product show the μ² double-count; min is invariant; downstream error tracks the double-count.

**E5 Type-2 separation.** Two borderline populations with identical mean degree 0.5, one with Beta(50,50) elicitation (sharp) and one with Beta(1,1) (diffuse). A single-number output cannot separate them; Q separates them; CRPS of Q on each. Prediction: point head CRPS identical on both, Q CRPS lower on the sharp population.

**Baselines.** Memoryless calibrated head; head + EMA smoothing of scores (a heuristic memory, not a filter — predicted to reduce φ̂ but break hop-ECE); HMM with hand-crafted Gaussian emissions on encoder features; BSF-S1 full; BSF-S1 with A = I (ablate transition), with μ thresholded (ablate deferral), with Q collapsed to its mean.

---

## 8. Threats to Validity and Open Problems

**8.1** Jev is closed. All statements are about the class of memoryless typed calibrated heads as publicly described in [1–3]; nothing here is an empirical refutation of Jev. Two public facts limit what we can assert: the Score primitive already returns a bounded scalar rather than an enum, so some vagueness can be expressed by schema choice (though as a single number, not a type-2 pair); and the State object passed to each call may include history the integrator chooses to serialise, so a harness can approximate memory by prompt construction. Neither changes the structural claim: no exact recursion over a latent regime exists inside the model, and no membership-plus-uncertainty pair exists in the output type.

**8.2** Membership ground truth is elicitation-dependent; type-1/type-2 renormalisation across regimes (§6.3) is an open semantic question. We state this as an open problem, not a defect: the architecture is designed so that the question is deferred to the actuator rather than answered implicitly by the model.

**8.3** Filter misspecification: learned (A, g, h) wrong ⇒ exact inference on a wrong model. Mitigation is the surprise monitor of §6.7, which is a specification, not a guarantee.

**8.4 Honesty anchor.** We make no claim of escaping the approximation regime. Learning A, g, h, μ, Q is statistical estimation and inherits every failure mode of the underlying network. What changes is the division of labour: *inference* — the recursion (3.7), the t-norm composition, the single defuzzification — is exact given the model. The paper's claim is that the approximation should live in the parameters, not in the algebra.

---

## 9. Discussion

**[PROSE]** 9.1 Benchmarks should report TCE and AMS alongside ECE; two models with equal ECE are not equally safe. 9.2 Fuzzy control, regime-switching econometrics and POMDP planners are three independent traditions that converged on deferred crispification; their agreement is evidence for the principle. 9.3 Not anti-neural: the network parameterises A, g, h, μ, Q; the algebra composes them. 9.4 Open problems: type-2 filter semantics; learned parametric t-norms (Frank/Hamacher families) with interpretability preserved; belief-aware thresholds (τ as a function of b_t); a theory of trajectory calibration as a proper-scoring object.

## 10. Conclusion

Two primitives, two lemmas, one principle, one architecture: give system-one models a memory of the world's hidden state and a language for its borderline cases, and collapse nothing until the moment of action.

---

## References

[1] Almeida, D. *Introducing System One Models & Jev.* TypeSafe AI blog, 15 Sept 2026. https://typesafe.ai/blog/introducing-system-one-models-and-jev
[2] TypeSafe AI. *System One* (concepts) and *Primitives* (Choice, Score, Noul). TypeSafe documentation, accessed 17 Sept 2026. https://docs.typesafe.ai/concepts/system-one
[3] Claburn, T. *TypeSafe AI debuts model for machines that plays Doom.* The Register, 16 Sept 2026. https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
[4] Zadeh, L. A. Fuzzy sets. *Information and Control* 8(3):338–353, 1965.
[5] Zadeh, L. A. The concept of a linguistic variable and its application to approximate reasoning—I. *Information Sciences* 8(3):199–249, 1975. (Parts II and III: 8(4):301–357 and 9(1):43–80, 1975.)
[6] Mamdani, E. H., Assilian, S. An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies* 7(1):1–13, 1975.
[7] Rabiner, L. R. A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE* 77(2):257–286, 1989.
[8] Hamilton, J. D. A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2):357–384, 1989.
[9] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On calibration of modern neural networks. *ICML*, PMLR 70:1321–1330, 2017.
[10] Gneiting, T., Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477):359–378, 2007.
[11] Kaelbling, L. P., Littman, M. L., Cassandra, A. R. Planning and acting in partially observable stochastic domains. *Artificial Intelligence* 101(1–2):99–134, 1998.
[12] Kahneman, D. *Thinking, Fast and Slow.* Farrar, Straus and Giroux, 2011.
[13] Bourlard, H., Morgan, N. *Connectionist Speech Recognition: A Hybrid Approach.* Kluwer Academic Publishers, ISBN 0-7923-9396-1, 1994. (Scaled likelihoods: network outputs normalised by class priors to serve as HMM emission terms.)
[14] Klement, E. P., Mesiar, R., Pap, E. *Triangular Norms.* Trends in Logic vol. 8, Kluwer Academic Publishers, Dordrecht, 2000.
[15] Mendel, J. M. Type-2 fuzzy sets and systems: an overview [corrected reprint]. *IEEE Computational Intelligence Magazine* 2(1):20–29, Feb 2007.

## Appendix A — Notation

| Symbol | Meaning |
|---|---|
| o_t | evidence at step t |
| d_t ∈ D | typed decision target |
| s_t ∈ {1..K} | latent regime |
| A | K×K transition matrix, A_{ij} = P(s_{t+1}=j \| s_t=i) |
| π | stationary distribution of A |
| λ, λ_i | second (and further) eigenvalues of A |
| g(o) | regime head ≈ P_train(s \| o) |
| h_s(o) | per-regime decision head ≈ P(d \| o, s) |
| Δb_t | scaled likelihood, ∝ g(o_t)/π (composable) |
| b_t | filtered belief P(s_t \| o_{1:t}) (not composable) |
| μ ∈ [0,1]^{K×M} | per-regime membership over M predicates |
| Q | type-2 layer: distribution over the degree |
| τ | threshold |
| T, S | t-norm, t-conorm |
| D | defuzzifier |
| E_t, N_T, ē, e_s, φ | error indicator, window count, mean error, regime error, inflation factor |

## Appendix B — Full proofs

**B.1 Lemma 1(ii).** Let e = (e_1, e_2)ᵀ, Π = diag(π). By the conditional independences, P(E_t=1, E_{t+k}=1) = Σ_{s,s′} π(s) (A^k)_{ss′} e_s e_{s′} = eᵀ Π A^k e. Any 2×2 row-stochastic A has eigenvalues 1 and λ = 1−a−b with right eigenvector 1 and left eigenvector πᵀ for eigenvalue 1, so A^k = 1πᵀ + λ^k (I − 1πᵀ). Then eᵀ Π 1πᵀ e = (πᵀe)² = ē² and eᵀ Π (I − 1πᵀ) e = Σ_s π(s) e_s² − ē² = Var_π(e). Subtracting E[E_t]E[E_{t+k}] = ē² gives (3.4). For (3.5): Var(N_T) = Σ_t Var(E_t) + 2 Σ_{t<t′} Cov(E_t, E_{t′}) = T ē(1−ē) + 2 Var_π(e) Σ_{k=1}^{T−1} (T−k) λ^k. Divide by T ē(1−ē) and let T → ∞; Σ_{k≥1} λ^k = λ/(1−λ) for |λ|<1 gives (3.6). Sojourn in regime 1 is geometric with success probability a, mean 1/a. For K regimes with A diagonalisable, replace the two-term decomposition by A^k = Σ_i λ_i^k P_i with λ_1 = 1, P_1 = 1πᵀ. ∎

**B.2 Lemma 1(iii).** Write c_s = 1−e_s. P(N_T=0) = E[∏_t c_{s_t}]. Define the "frozen" chain (λ=1): E_frozen = Σ_s π(s) c_s^T ≥ (Σ_s π(s) c_s)^T = (1−ē)^T by Jensen (x ↦ x^T convex on [0,1]), strict iff c not constant on supp π. For general λ ∈ (0,1), the two-state chain is a mixture: with A = (1−λ)·1πᵀ + λ I, a sample path is generated by, at each step, redrawing s from π with probability 1−λ or keeping it with probability λ. Conditioning on the redraw times partitions [1,T] into independent frozen blocks of lengths ℓ_1, …, ℓ_J; within each block, E[∏ c] = Σ_s π(s) c_s^{ℓ_j} ≥ (1−ē)^{ℓ_j}; multiply over blocks and take expectation over the partition to get ≥ (1−ē)^T, strict when some block has length ≥ 2 with positive probability (λ > 0) and Var_π(c) > 0. (The representation A = (1−λ)1πᵀ + λI holds for every 2×2 stochastic matrix with λ ≥ 0.) ∎

**B.3 Lemma 1(iv).** ECE and averaged proper scores are functions of the empirical measure (1/T)Σ_t δ_{(f(o_t), y_t)}; the empirical measure is invariant under permutations of t. ∎

**B.4 Lemma 2(i).** For product: |∏_{h} a_h − ∏_h b_h| = |Σ_h (∏_{j<h} b_j)(a_h − b_h)(∏_{j>h} a_j)| ≤ Σ_h |a_h − b_h| on [0,1]^H. For min: WLOG min a = a_i ≤ min b = b_j; then min b − min a ≤ b_i − a_i ≤ max_h |a_h − b_h|. Combine with Lipschitzness of μ_h and D. ∎

**B.5 Lemma 2(ii).** Lower bound: for x with τ_h − cε ≤ f_h(x) < τ_h, moving distance ε along ∇f_h/‖∇f_h‖ raises f_h by ≥ cε (mean value theorem with ‖∇f_h‖ ≥ c near the boundary), crossing τ_h; so x ∈ F_ε. The P_X-mass of this band is ∫_{τ−cε}^{τ} p_h ≈ c p_h(τ_h) ε. Upper bound: if |f_h(x) − τ_h| > Lε for all h then no ε-ball crosses any threshold; the complement has mass ≤ Σ_h 2L p_h(τ_h) ε. ∎

**B.6 Lemma 2(iii), (iv).** As in §4.3; (iv) uses that the set of z within ε of z* has P_Z-measure ≈ p_Z(z*)·ε and monotonicity forces simultaneous crossing. Under independent perturbations the events are independent and the product of H Θ(ε) masses is Θ(ε^H). ∎

## Appendix C — Pseudocode

See Algorithms 1–3 in §6.6. **Forward-filter / backward-sample for TCE (E2):** run Algorithm 1 to obtain b_{1:T} and b_pred_{1:T}; sample s_T ~ b_T; for t = T−1..1 sample s_t ∝ b_t(s) · A_{s, s_{t+1}}; draw E_t ~ Bernoulli(1 − h_{s_t}(o_t)[d*_t]); repeat R times to obtain the implied distribution of N_w.

## Appendix D — Synthetic generators

```
def regime_stream(T, a, b, n, seed):
    A = [[1-a, a],[b, 1-b]]; π = [b/(a+b), a/(a+b)]
    s[0] ~ π; s[t+1] ~ A[s[t]]
    o[t] ~ N(m[s[t]], Σ[s[t]])                # n-dim
    d[t] = 1{ w[s[t]]·o[t] + c[s[t]] > 0 }      # regime-dependent rule
    return o, d, s

def borderline_population(N, sharp: bool):
    o ~ Uniform on the decision boundary band
    (α, β) = (50, 50) if sharp else (1, 1)
    μ_elicited[m] ~ Beta(α, β) per predicate m
    return o, μ_elicited
```

Choose a, b to sweep λ ∈ {0, 0.5, 0.9, 0.99}; choose (w_s, c_s) so that P(d | o, s) differs across s (otherwise Δ(o) ≡ 0 and E1 is vacuous by design).
