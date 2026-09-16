# What Jev Is Missing: Calibration Does Not Compose, Types Destroy Vagueness

**A position paper on TypeSafe AI's Jev and the "System One" class of decision models — and the two mathematical primitives they leave out.**

> Preprint v0.3 · 17 Sept 2026 · [Read the paper](paper.md) · CC BY 4.0

---

## TL;DR

On 15 September 2026, TypeSafe AI released **Jev**, the first *System One model*: no text generation, typed probabilistic decisions (`Choice`, `Score`, `Noul`), calibrated by a new training method called **Reinforcement Learning for Calibrated Decisions (RLCD)**, at 70–500 ms latency and $0.042 per million input tokens. It is a genuinely new kind of interface: *unstructured state in, typed probabilities out*.

This paper argues that the class Jev belongs to — fast, memoryless, typed, calibrated decision heads — is missing **two structural primitives**, and proves two negative results about what happens when their outputs are composed in a pipeline:

| Missing primitive | What goes wrong | Result |
|---|---|---|
| **Hidden-Markov belief** over a latent regime | Per-call calibration is measured against a frozen prior. When the world's regime shifts, the score is still "calibrated" against a prior that no longer exists; errors arrive in clusters whose length is the regime sojourn time; and **expected calibration error is permutation-invariant, so it has exactly zero power to detect this.** | **Lemma 1** — calibration does not compose. Closed-form variance-inflation factor `φ = 1 + 2λ/(1−λ) · Var_π(e)/(ē(1−ē))`. |
| **Fuzzy membership** for borderline predicates | A typed enum forces ontological vagueness ("is this ticket *urgent*?") to be reported as epistemic uncertainty (p ≈ 0.5), which the harness reads as "I don't know." Thresholding the score makes the pipeline action discontinuous: an ε-perturbation flips a full action on a Θ(ε)-measure set, every downstream hop then sees an O(1) input change, and under common-mode evidence the all-hops-flip event has Θ(ε) rather than Θ(ε^H) measure. | **Lemma 2** — cliff cascade. The fuzzy alternative (t-norm composition, single Lipschitz defuzzification) keeps the action Lipschitz in the evidence. |

Both holes are **structural, not capacity-related**: they survive any scaling of the network. Both are **cheap to repair**: O(K² + KM) per step, no generative decode.

## The fix: Deferred Crispification

> *No structure may be collapsed before the actuator. Collapse — defuzzify, threshold, sample — happens exactly once, with full decision context.*

The paper specifies a reference architecture, **Belief-State Fuzzy System-One (BSF-S1)**, whose output object is

```
O(x) = ( μ ∈ [0,1]^{K×M},   Q[μ],   Δb ∈ Δ^K )
        membership per      type-2   scaled likelihood
        regime & predicate  spread   over regimes
```

- `Δb` is a **scaled likelihood** (posterior ÷ prior — the Bourlard–Morgan hybrid-HMM trick), so it composes across hops by pointwise product in an exact forward filter. Posteriors don't compose; likelihoods do.
- `μ` is emitted **per regime** and mixed only at the actuator — mixing over regimes is itself a collapse.
- `Q` is a calibrated distribution *over the degree* (type-2), audited by proper scoring rules, kept separate from the degree itself.

Full pseudocode for the filter step, the single-collapse actuator, and training is in [§6.6](paper.md#66-algorithm-boxes).

## Why this matters if you are building on Jev today

TypeSafe's own integration guidance is to "ask independent questions together … combine the answers with deterministic checks in code, then route the case." That is the **double collapse** the paper analyses: the model thresholds once, the harness thresholds again. Two models with identical ECE can differ in trajectory risk by an arbitrary factor. The paper proposes two audit metrics you can run on any decision pipeline without access to model weights:

- **TCE** (trajectory calibration error): does the model's implied distribution of *errors per window* match reality? Hop-level ECE cannot tell you.
- **AMS** (action-mass sensitivity): how much action mass moves under an ε-perturbation of evidence?

## Reproduce the math

`experiments/verify_lemma1.py` simulates a two-regime chain and checks the closed forms:

```bash
python3 experiments/verify_lemma1.py
```

Output on the default parameters (a=0.05, b=0.10, e=(0.05, 0.40), T=50):

```
var  N: sim 20.304  pred(3.5) 20.314   binomial 6.944
P(all correct): sim 0.0071  indep (1-ebar)^T 0.0001
```

Simulated variance matches equation (3.5) to three decimals; the all-correct probability is ~70× the independent baseline that hop-level calibration implies.

## Status

- ✅ Lemmas 1–2 stated and proved (Appendix B); closed forms numerically verified
- ✅ BSF-S1 output object, composition algebras, and pseudocode complete
- ✅ All Jev claims quoted from primary sources (TypeSafe launch post, docs, The Register); all academic citations verified
- 🔲 Prose expansion of §1, §5.5, §9
- 🔲 Experiments E1–E5 (generators specified in Appendix D)

## Scope and honesty

Jev is closed. Every claim about it is drawn from public material and applies to the *class* of memoryless typed calibrated heads; nothing here is an empirical refutation of Jev. The paper's honesty anchor (§8.4): learning the transition, emission and membership parameters is approximation like any other neural estimate; what the architecture changes is that *inference* — the filter recursion, the t-norm composition, the single defuzzification — is exact given the model. The approximation should live in the parameters, not in the algebra.

## Citation

See [CITATION.cff](CITATION.cff). Also on ResearchGate (link to follow).

## Sources on Jev

- Almeida, D. [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). TypeSafe AI, 15 Sept 2026.
- TypeSafe AI. [System One — concepts](https://docs.typesafe.ai/concepts/system-one).
- Claburn, T. [TypeSafe AI debuts model for machines that plays Doom](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711). The Register, 16 Sept 2026.
