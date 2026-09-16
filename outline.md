# OUTLINE — Beyond the One-Shot Handoff: Diagnosing and Repairing the Encoder–Decoder Boundary of Sparse-Attention Language Models

(v1.0 as supplied on 2026-09-17; see technical-supplement.md for math and pseudocode.)

**Authors:** [N. A. K. Doan et al.] · **Status:** expansion-ready outline v1.0 · **Target:** arXiv cs.LG / position-or-ideas track; workshop variants: NeurIPS ENLSP, EMNLP SystEM
**Alternate titles:** (a) The One-Lane Bridge; (b) CED Is Leaking

## Abstract skeleton
1. Encoder–decoder sparse-attention LLMs amortize prompt compute (YOCO → DeepSeek V4/V4.1).
2. The conduit (CED) is undocumented as a design object and is the binding information/bandwidth constraint.
3. Four pathologies — single-scale, state-only, one-shot, optimization-pathological.
4. Multi-resolution CED + index inheritance + Boundary Working Memory (BWM) with train/serve symmetry.
5. Evidence plan + trajectory claim.

## Table 0 — Claims map
| ID | Type | Statement |
|---|---|---|
| O1 | Observation | V4.1-Flash: 20-layer causal encoder + 20-layer decoder; DeepSeekMoE FFNs; CSA2(ratio, mode) ∈ {Full, Reuse, Reindex}; single conduit (CED); Engram taps early encoder layers; mHC fuses input streams. |
| H1 | Hypothesis | CED transfers top-layer states only; multi-scale features and sparse-index structure are discarded. |
| H2 | Hypothesis | Decoder-side hierarchical indexing rebuilds structure the encoder already computed (double indexing); boundary crosses at ratio 1 while encoder compresses at ratio 2. |
| P1–P3 | Proposals | Multi-resolution CED; index inheritance; BWM. |
| S1 | Speculation | Undocumented acronyms (CED, DSpark) denote assumed components. |

## Sections
1. Introduction (1.1 unit economics; 1.2 return of enc–dec; 1.3 seam as design object; 1.4 contributions C1–C5; 1.5 scope)
2. Background and Notation (2.1 sparse compressed attention; 2.2 YOCO amortization; 2.3 MoE/Engram/mHC; 2.4 layer algebra 2+3×(1+5)=20 / 1+3+4×(1+3)=20)
3. Diagnosis: four pathologies (3.1 single-scale; 3.2 state-only; 3.3 one-shot; 3.4 optimization pathology; 3.5 Table 1)
4. Two necessary repairs (4.1 multi-resolution CED; 4.2 index inheritance; 4.3 necessary-but-insufficient)
5. Boundary Working Memory (5.1 architecture; 5.2 training; 5.3 inference; 5.4 train/serve invariant)
6. Case studies (6.1 direct pass; 6.2 Q-Former; 6.3 RLT; 6.4 workload relativity; 6.5 Table 2)
7. Theoretical framing (7.1 support inheritance; 7.2 rate–distortion; 7.3 memory hierarchy; 7.4 limits)
8. Minimal experimental program E1–E5
9. Threats to validity
10. Discussion: trajectories and ethics
11. Conclusion
References [1]–[15]; Appendices A (figures), B (glossary)
