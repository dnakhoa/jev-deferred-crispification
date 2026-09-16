"""E1 — Regime-shift stress (Lemma 1(i)).

Train a memoryless head on a stationary stream. Evaluate ECE on:
  (a) a fresh stationary stream (same pi)      -> both models ~0
  (b) windows conditioned on the true regime   -> memoryless miscalibrated
  (c) a shifted stream with pi' != pi          -> memoryless miscalibrated
BSF-S1 should be ~calibrated in all three.
"""
import numpy as np
from common import regime_stream, MemorylessHead, BSFS1, ece


def regime_conditional_ece(p, d, s):
    """ECE restricted to the time steps spent in each regime: the calibration a
    consumer experiences *during a sojourn*, which the stationary audit averages away."""
    return {k: ece(p[s == k], d[s == k]) for k in (0, 1)}


def run(seed=0, T_train=30000, T_test=30000):
    rng = np.random.default_rng(seed)
    a, b = 0.01, 0.04                              # pi = (0.8, 0.2), lambda = 0.95
    o, d, s, A, pi = regime_stream(T_train, a, b, rng)
    ml = MemorylessHead().fit(o, d)
    bs = BSFS1().fit(o, d, s)

    res = {"A_true": A.tolist(), "A_hat": bs.A.tolist()}
    # (a) stationary
    o2, d2, s2, _, _ = regime_stream(T_test, a, b, rng)
    res["stationary"] = {"memoryless": ece(ml.predict(o2), d2),
                         "bsf_s1": ece(bs.predict(o2)[0], d2)}
    # (b) regime-conditional on the same stationary stream
    res["window_conditional"] = {"memoryless": regime_conditional_ece(ml.predict(o2), d2, s2),
                                 "bsf_s1": regime_conditional_ece(bs.predict(o2)[0], d2, s2)}
    # (c) shifted marginal: the rare regime becomes dominant, pi' = (0.2, 0.8).
    #     The transition matrix has changed too, so BSF-S1 runs with a STALE A
    #     (the failure mode of Section 6.7); the filter must track from evidence alone.
    o3, d3, s3, _, pi3 = regime_stream(T_test, 0.04, 0.01, rng)
    res["shifted"] = {"pi_prime": pi3.tolist(),
                      "memoryless": ece(ml.predict(o3), d3),
                      "bsf_s1": ece(bs.predict(o3)[0], d3)}
    # accuracy for context
    res["accuracy_shifted"] = {"memoryless": float(((ml.predict(o3) >= .5) == d3).mean()),
                               "bsf_s1": float(((bs.predict(o3)[0] >= .5) == d3).mean())}
    return res


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=2))
