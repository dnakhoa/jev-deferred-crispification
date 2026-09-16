"""E1 — Regime-shift stress (Lemma 1(i)), round-2 version.

Five seeds, mean ± range. Rows:
  memoryless (quad logistic)          the system-one baseline
  oracle memoryless (Bayes-optimal)   infinite-capacity memoryless head: removes the capacity objection
  windowed head, L=10                 memoryless head fed a history window (State-object trick)
  memoryless + online Platt           sliding-window recalibration from realised outcomes
  BSF-S1 (oracle regime labels)       learned filter, labels for g and A-init
  BSF-S1 (unsupervised regimes)       labels replaced by a Gaussian-HMM EM on o alone
  exact filter (true A, g, h)         the achievable floor for every metric
Columns: ECE stationary / regime 1 / shifted; NLL and accuracy on the stationary stream; accuracy shifted.
Regime-conditional ECE conditions on the true regime, which is outside the filter's
sigma-field, so even the exact filter has a nonzero floor; read rows against that floor.
"""
import numpy as np
from common import *


def run(seed=0, seeds=(0, 1, 2, 3, 4), T_train=30000, T_test=30000):
    a, b = 0.01, 0.04
    rows = {}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        o, d, s, A, pi = regime_stream(T_train, a, b, rng)
        o2, d2, s2, _, _ = regime_stream(T_test, a, b, rng)
        o3, d3, s3, A3, pi3 = regime_stream(T_test, 0.04, 0.01, rng)
        ml = MemorylessHead().fit(o, d)
        bs = BSFS1().fit(o, d, s)
        g, Ah, _ = ghmm_em(o, seed=sd); sh = g.argmax(1)
        if (sh == s).mean() < .5: sh = 1 - sh; Ah = Ah[::-1, ::-1]
        bsu = BSFS1().fit(o, d, sh, A_init=Ah)
        models = {
            "memoryless": (ml.predict(o2), ml.predict(o3)),
            "oracle_memoryless": (OracleMemoryless(pi).predict(o2), OracleMemoryless(pi).predict(o3)),
            "windowed_L10": (lambda m: (m.predict(o2), m.predict(o3)))(WindowedHead(10).fit(o, d)),
            "memoryless_online_platt": (online_platt(ml.predict(o2), d2), online_platt(ml.predict(o3), d3)),
            "bsf_s1_oracle_labels": (bs.predict(o2)[0], bs.predict(o3)[0]),
            "bsf_s1_unsupervised": (bsu.predict(o2)[0], bsu.predict(o3)[0]),
            "exact_filter": (ExactFilter(A, pi).predict(o2)[0], ExactFilter(A, pi).predict(o3)[0]),
        }
        for k, (p2, p3) in models.items():
            rows.setdefault(k, []).append({
                "ece_stationary": ece(p2, d2), "ece_regime1": ece(p2[s2 == 1], d2[s2 == 1]),
                "ece_shifted": ece(p3, d3), "nll_stationary": nll(p2, d2),
                "acc_stationary": float(((p2 >= .5) == d2).mean()), "acc_shifted": float(((p3 >= .5) == d3).mean())})
        rows.setdefault("_meta", []).append({"seed": sd, "A_hat_oracle_labels": bs.A.tolist(), "A_hat_unsup": bsu.A.tolist(),
                                             "unsup_label_agreement": float((sh == s).mean())})
    out = {"seeds": list(seeds), "per_seed": rows}
    summ = {}
    for k, lst in rows.items():
        if k.startswith("_"): continue
        summ[k] = {m: {"mean": float(np.mean([r[m] for r in lst])), "min": float(np.min([r[m] for r in lst])),
                       "max": float(np.max([r[m] for r in lst]))} for m in lst[0]}
    out["summary"] = summ
    return out


if __name__ == "__main__":
    import json; print(json.dumps(run()["summary"], indent=1))
