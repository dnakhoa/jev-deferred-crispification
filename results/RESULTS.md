# Results (seed 0)

All runs CPU-only, numpy/scipy. Regenerate with `python3 experiments/run_all.py`.

## E1 — Regime-shift stress (Lemma 1(i))

| Evaluation | Memoryless head ECE | BSF-S1 ECE |
|---|---|---|
| Stationary stream (same π) | 0.0245 | 0.0095 |
| Time steps inside regime 0 | 0.0655 | 0.0151 |
| Time steps inside regime 1 | 0.1561 | 0.0175 |
| Shifted stream π′ = (0.2, 0.8), stale A | 0.1093 | 0.0097 |

Accuracy on the shifted stream: memoryless 0.6663, BSF-S1 0.8072. Baum–Welch recovered A ≈ [[0.966, 0.034], [0.041, 0.959]] (true [[0.99, 0.01], [0.04, 0.96]]).

## E2 — Trajectory calibration, TCE (Lemma 1(ii)–(iv))

Hop-level ECE of the memoryless head: 0.0212 on the stream, 0.0212 on its random shuffle (Lemma 1(iv): identical by construction).

| Model | φ̂ = Var(realised N_w)/Var(implied N_w) | PIT L1 from uniform | KS p-value (PIT ~ U[0,1]) |
|---|---|---|---|
| Memoryless (implied = independent Poisson-binomial) | 1.7645 | 0.5250 | 3.07e-36 |
| BSF-S1 (implied = FFBS joint) | 1.1192 | 0.1000 | 3.82e-02 |

TCE verdict: a model passes if KS does not reject uniformity of the PIT at α = 0.01 and φ̂ ∈ [0.8, 1.25].

Realised variance of the memoryless head's N_w against the *binomial* baseline w·ē(1−ē) that hop-level calibration implies: 1.8768. Theory for the memoryless head from its measured regime error rates e = (0.177, 0.376): φ_w (eq. 3.5, w=50) = 1.9414, φ_∞ (eq. 3.6) = 2.4924. Direct test of eq. (3.4), Cov(E_t,E_t+k) = λ^k·Var_π(e) for k = 1..20: relative L2 error 0.1111.

## E3 — Cliff cascade (Lemma 2)

| ε | crisp FM/ε | fuzzy FM | fuzzy AMS/ε |
|---|---|---|---|
| 0.002 | 0.7400 | 0.0 | 0.2066 |
| 0.005 | 0.7420 | 0.0 | 0.2066 |
| 0.01 | 0.7390 | 0.0 | 0.2066 |
| 0.02 | 0.7772 | 0.0 | 0.2066 |
| 0.05 | 0.7839 | 0.0 | 0.2066 |

Theory: crisp FM/ε → 2·p(0) = 0.7979; fuzzy AMS/ε ≤ L = 0.25.

| ε | crisp P(hop 2 flips \| hop 1 flipped) | fuzzy sup change at hop 2 |
|---|---|---|
| 0.002 | 0.4899 | 0.0001 |
| 0.005 | 0.4838 | 0.0003 |
| 0.01 | 0.4797 | 0.0006 |
| 0.02 | 0.4709 | 0.0011 |
| 0.05 | 0.4693 | 0.0029 |

Common-mode coupling, H = 3, log–log slope of P(all H hops flip) vs ε: 1.00 when the hops' crossing points coincide (theory 1); 1.79 when crossing points are spread by a continuous density (theory H); 2.97 under independent perturbations (theory H).

## E4 — Composition algebra

| Algebra | idempotence gap E\|T(a,a,b)−T(a,b)\| | answer-flip rate |
|---|---|---|
| min | 0.0000 | 0.0000 |
| product | 0.0835 | 0.0677 |
| naive_bayes | 0.0835 | 0.0677 |

## E5 — Type-2 separation

| Population | point-head output | point CRPS (=MAE) | Q fit (α, β) | Q CRPS |
|---|---|---|---|---|
| sharp | 0.5016 | 0.0388 | (49.5, 49.1) | 0.0273 |
| diffuse | 0.5074 | 0.2505 | (1.0, 1.0) | 0.1671 |

The point head emits the same object for both populations (identical: True); the Q head's concentration differs by a factor of 48. On the diffuse population Q strictly beats the point head (1/6 < 1/4).
