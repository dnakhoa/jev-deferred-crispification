# What Jev Is Missing: Calibration Does Not Compose, Types Destroy Vagueness

**A position paper on TypeSafe AI's Jev and the "System One" class of decision models — and the two mathematical primitives they leave out.**

> **Anh Khoa Doan Ngoc** · Preprint v1.1 · 17 September 2026 · **[Download the PDF](paper.pdf)** · [LaTeX source](paper.tex) · CC BY 4.0
>
> [![DOI](https://img.shields.io/badge/DOI-pending%20Zenodo%20release-lightgrey)](#citation)

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

Full pseudocode for the filter step, the single-collapse actuator, and training is in §6.5 of the [PDF](paper.pdf).

## Why this matters if you are building on Jev today

TypeSafe's own integration guidance is to "ask independent questions together … combine the answers with deterministic checks in code, then route the case." That is the **double collapse** the paper analyses: the model thresholds once, the harness thresholds again. Two models with identical ECE can differ in trajectory risk by an arbitrary factor. The paper proposes two audit metrics you can run on any decision pipeline without access to model weights:

- **TCE** (trajectory calibration error): does the model's implied distribution of *errors per window* match reality? Hop-level ECE cannot tell you.
- **AMS** (action-mass sensitivity): how much action mass moves under an ε-perturbation of evidence?

## Reproduce everything (no GPU; ~1 minute on a laptop)

```bash
python3 experiments/verify_lemma1.py     # closed forms of Lemma 1 vs simulation
python3 experiments/verify_lemma2.py     # cliff cascade: typed+threshold vs fuzzy+defuzzify, adversarial and Gaussian noise
python3 experiments/run_all.py           # E1–E5 -> results/RESULTS.md and results/results.json
```

Only NumPy and SciPy are required. On Google Colab: `!git clone https://github.com/dnakhoa/jev-deferred-crispification && cd jev-deferred-crispification && python3 experiments/run_all.py`.

### Headline results (seed 0, full tables in [results/RESULTS.md](results/RESULTS.md))

| Experiment | Memoryless typed head | BSF-S1 |
|---|---|---|
| E1 ECE inside the rare regime / after shift | 0.156 / 0.109 | 0.018 / 0.010 |
| E2 TCE: KS p-value of window-error PIT | 3×10⁻³⁶ (fail) | 0.04 (pass) |
| E2 dispersion vs binomial baseline | 1.88 (theory 1.94) | — |
| E3 flip mass under ε-perturbation | Θ(ε), jumps of 1, cascade P=0.48 | 0 flips; signal change ≤ 0.21·ε |
| E4 answer-flip rate when a predicate is asked twice | 6.8% (product / naive Bayes) | 0% (min t-norm) |
| E5 CRPS on a diffuse borderline population | 0.25 (point) | 0.17 (type-2 Q) |

## Status

- ✅ Full paper (LaTeX) — [paper.pdf](paper.pdf)
- ✅ Lemmas 1–2 stated and proved (Appendix B); both numerically verified (`verify_lemma1.py`, `verify_lemma2.py`)
- ✅ BSF-S1 output object, composition algebras, and pseudocode complete
- ✅ Experiments E1–E5 run; TCE and AMS operationally defined (§7.1)
- ✅ All Jev claims quoted from TypeSafe's launch post and documentation; all academic citations verified
- 🔲 Ablations (EMA-smoothed head, hand-crafted HMM, BSF-S1 with A=I / thresholded μ / collapsed Q) — contributions welcome

## Scope and honesty

Jev is closed. Every claim about it is drawn from public material and applies to the *class* of memoryless typed calibrated heads; nothing here is an empirical refutation of Jev. The paper's honesty anchor (§8.4): learning the transition, emission and membership parameters is approximation like any other neural estimate; what the architecture changes is that *inference* — the filter recursion, the t-norm composition, the single defuzzification — is exact given the model. The approximation should live in the parameters, not in the algebra.

## Citation

See [CITATION.cff](CITATION.cff). A Zenodo DOI will be minted from the next GitHub release; until then cite the repository URL and version tag. ResearchGate mirror to follow.

**AI-usage statement.** Claude Fable 5.1 assisted with prose expansion, experiment code and typesetting under the author's direction. All claims, proofs, experimental design and responsibility are the author's. No AI system is an author.

## Sources on Jev

- Almeida, D. [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). TypeSafe AI, 15 Sept 2026.
- TypeSafe AI. [System One — concepts](https://docs.typesafe.ai/concepts/system-one).
- Claburn, T. [TypeSafe AI debuts model for machines that plays Doom](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711). The Register, 16 Sept 2026 (press coverage; all technical claims in the paper come from the two TypeSafe sources above).
