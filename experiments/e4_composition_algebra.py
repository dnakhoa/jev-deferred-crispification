"""E4 — Composition-algebra ablation (Section 4.4).

Rule: "A and B", evaluated once with the predicate A entered once and once
with A entered twice ("A and A and B"), under min, product, and naive-Bayes
product of probabilities. Metrics:
  idempotence gap  = E |T(a,a,b) - T(a,b)|            (min -> 0)
  answer-flip rate = P( 1{T(a,a,b) >= 1/2} != 1{T(a,b) >= 1/2} )
The flip rate is the operational cost: asking the same question twice
changes the downstream decision.
"""
import numpy as np


def run(seed=0, N=200000):
    rng = np.random.default_rng(seed)
    a = rng.random(N); b = rng.random(N)
    algebras = {
        "min":         (np.minimum(a, b),              np.minimum(np.minimum(a, a), b)),
        "product":     (a * b,                         a * a * b),
        "naive_bayes": (a * b,                         a * a * b),   # numerically identical to product
    }
    out = {}
    for name, (once, twice) in algebras.items():
        out[name] = {"idempotence_gap": float(np.abs(twice - once).mean()),
                     "answer_flip_rate": float(((twice >= .5) != (once >= .5)).mean())}
    out["note"] = ("naive_bayes equals product numerically; the difference is semantic: "
                   "product-as-t-norm is truth-functional, product-as-probability asserts independence "
                   "of two copies of the same predicate, which is false by construction.")
    return out


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=2))
