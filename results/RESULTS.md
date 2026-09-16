# Results — round 2 (five seeds; mean [min, max])

Regenerate with `python3 experiments/run_all.py`. CPU only.

## E1 — Regime-shift stress (Lemma 1(i))

| Model | ECE stationary | ECE inside regime 1 | ECE shifted (stale A) | NLL stationary | Acc. stationary | Acc. shifted |
|---|---|---|---|---|---|---|
| Memoryless head (quad. logistic) | 0.022 [0.015, 0.025] | 0.156 [0.140, 0.169] | 0.114 [0.107, 0.121] | 0.475 [0.471, 0.482] | 0.783 [0.780, 0.786] | 0.668 [0.663, 0.675] |
| Bayes-optimal memoryless head | 0.007 [0.005, 0.008] | 0.193 [0.182, 0.202] | 0.146 [0.140, 0.149] | 0.462 [0.458, 0.468] | 0.785 [0.781, 0.790] | 0.650 [0.648, 0.655] |
| History-window head, L=10 | 0.010 [0.009, 0.014] | 0.077 [0.058, 0.087] | 0.038 [0.028, 0.046] | 0.443 [0.440, 0.449] | 0.794 [0.791, 0.796] | 0.737 [0.729, 0.742] |
| Memoryless + online Platt (W=500) | 0.019 [0.017, 0.022] | 0.160 [0.151, 0.169] | 0.027 [0.019, 0.032] | 0.478 [0.475, 0.485] | 0.783 [0.780, 0.786] | 0.671 [0.668, 0.675] |
| BSF-S1, oracle regime labels | 0.011 [0.008, 0.017] | 0.061 [0.017, 0.116] | 0.021 [0.010, 0.040] | 0.440 [0.425, 0.485] | 0.792 [0.758, 0.805] | 0.788 [0.758, 0.807] |
| BSF-S1, unsupervised regimes | 0.016 [0.007, 0.027] | 0.060 [0.022, 0.092] | 0.018 [0.005, 0.038] | 0.431 [0.424, 0.442] | 0.800 [0.793, 0.803] | 0.765 [0.746, 0.780] |
| Exact filter, true (A, g, h) — floor | 0.007 [0.005, 0.008] | 0.066 [0.059, 0.076] | 0.011 [0.008, 0.014] | 0.423 [0.420, 0.426] | 0.803 [0.800, 0.806] | 0.785 [0.781, 0.790] |

Regime-conditional ECE conditions on the true regime, which the filter cannot observe, so the exact filter's row is the floor for that column. Unsupervised regime labels agree with the truth on 92% of steps on average.

## E2 — Trajectory calibration, TCE (Lemma 1(ii)–(iv))

| Model | φ̂ mean [min, max] | passes (φ̂ CI ∋ 1) / 5 | KS rejects at 0.01 / 5 |
|---|---|---|---|
| Memoryless | 1.72 [1.54, 1.95] | 0 | 5 |
| Bayes-optimal memoryless | 2.14 [1.90, 2.41] | 0 | 5 |
| BSF-S1 (oracle labels) | 1.09 [0.88, 1.31] | 2 | 1 |
| Exact filter (floor) | 0.99 [0.96, 1.03] | 5 | 0 |
| Control: memoryless on shuffled stream | 0.94 [0.88, 0.98] | 4 | 5 |

Hop-level ECE identical on stream and shuffle on every seed: True (Lemma 1(iv)). Realised variance of the memoryless head's window count vs the binomial baseline: 1.83 (eq. 3.5 predicts 1.86). Lag-covariance test of eq. 3.4, mean relative L2 error 0.28. The shuffled control shows why φ̂ is the clustering-specific statistic: it returns to ≈1 on the shuffle while KS still rejects (hop-level miscalibration).

## E3 — Cliff cascade (Lemma 2), fair comparison

Base rates: crisp conjunction fires on 13.1% of inputs; t-norm+centroid ≥ ½ fires on 0.4%. Single-collapse pipelines are thresholded at their base-rate-matched quantile. Min t-norm at ½ equals the crisp conjunction bit-for-bit: True.

| ε | crisp, own adversary | t-norm+centroid (matched), own adversary | mean-of-probs (matched), own adversary | crisp, Gaussian | t-norm+centroid (matched), Gaussian | t-norm+centroid at ½, crisp adversary (the unfair number) |
|---|---|---|---|---|---|---|
| 0.005 | 0.0031 | 0.0020 | 0.0022 | 0.0012 | 0.0008 | 0.0001 |
| 0.01 | 0.0062 | 0.0044 | 0.0044 | 0.0025 | 0.0018 | 0.0001 |
| 0.02 | 0.0127 | 0.0090 | 0.0088 | 0.0050 | 0.0036 | 0.0002 |
| 0.05 | 0.0309 | 0.0221 | 0.0215 | 0.0130 | 0.0089 | 0.0005 |

Single gate: crisp FM/ε = 0.742, 0.739, 0.777, 0.784 (limit 2p(0) = 0.798); fuzzy AMS/ε = 0.207 (bound ¼).

| coupling weight v | P(gate 2 flips \| gate 1 flipped) | v·p(0) |
|---|---|---|
| 0.05 | 0.028 | 0.020 |
| 0.1 | 0.051 | 0.040 |
| 0.3 | 0.139 | 0.120 |
| 0.5 | 0.228 | 0.199 |
| 1.0 | 0.401 | 0.399 |
| 1.5 | 0.480 | 0.598 |
| 3.0 | 0.516 | 1.197 |

Coupling (N = 4×10⁶ for the spread case, [22, 389, 3155, 24873] events): log–log slope of P(all 3 flip) vs ε = 1.00 coincident (theory 1), 3.05 spread (theory 3), 2.99 independent (theory 3).

## E4 — Composition algebra (an arithmetic identity, reported for completeness)

| Algebra | idempotence gap | answer-flip rate |
|---|---|---|
| min | 0.000 | 0.000 |
| product | 0.084 | 0.068 |
| naive_bayes | 0.084 | 0.068 |

## E5 — Type-2 separation

| Population | point-head output | point CRPS (=MAE) | Q fit (α, β) | Q CRPS |
|---|---|---|---|---|
| sharp | 0.502 | 0.039 | (49.5, 49.1) | 0.027 |
| diffuse | 0.507 | 0.251 | (1.0, 1.0) | 0.167 |