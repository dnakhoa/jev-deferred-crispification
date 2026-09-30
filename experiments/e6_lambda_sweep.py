"""E6 — Lemma 1(ii) across regime persistence lambda (new in v1.3).

The Bayes-optimal memoryless head P_train(d|o) is applied to two-regime streams with
the stationary marginal held at pi = (0.8, 0.2) while lambda = 1 - a - b is swept.
For each lambda the realised variance of the window error count N_w is compared with
(a) the binomial baseline that hop-level calibration implies and (b) eq. (3.5)
evaluated at the head's measured regime error rates. Lemma 1(ii) predicts the ratio
rises from 1 at lambda = 0 along eq. (3.5); the time-shuffled stream must stay at 1.

Also stores: the lag covariance at lambda = 0.95 (empirical vs lambda^k Var_pi(e))
for the figure, and the closed-form check of Lemma 1 from verify_lemma1.py.
"""
import numpy as np
from common import regime_stream, OracleMemoryless, ExactFilter
from e2_trajectory_calibration import implied_ffbs, tce


def lemma1_check(seed=0, R=200000):
    rng = np.random.default_rng(seed)
    a, b = 0.05, 0.10; lam = 1 - a - b; pi = np.array([b, a]) / (a + b)
    e = np.array([0.05, 0.40]); ebar = pi @ e; vpe = pi @ e ** 2 - ebar ** 2; T = 50
    A = np.array([[1 - a, a], [b, 1 - b]])
    s = rng.choice(2, size=R, p=pi); N = np.zeros(R); allc = np.ones(R, bool)
    for _ in range(T):
        E = rng.random(R) < e[s]; N += E; allc &= ~E
        s = np.where(rng.random(R) < A[s, 1], 1, 0)
    k = np.arange(1, T)
    return {"var_sim": float(N.var()), "var_eq35": float(T * ebar * (1 - ebar) + 2 * vpe * np.sum((T - k) * lam ** k)),
            "var_binomial": float(T * ebar * (1 - ebar)), "p_all_correct_sim": float(allc.mean()),
            "p_all_correct_independent": float((1 - ebar) ** T)}


def run(seed=0, seeds=(0, 1, 2), lambdas=(0.0, 0.5, 0.8, 0.9, 0.95, 0.98), T=200000, w=50, T_filter=25000, R=100):
    pi_target = np.array([0.8, 0.2])
    rows = []
    for lam in lambdas:
        a, b = pi_target[1] * (1 - lam), pi_target[0] * (1 - lam)
        per = []
        for sd in seeds:
            rng = np.random.default_rng(1000 * sd + int(100 * lam))
            o, d, s, A, pi = regime_stream(T, a, b, rng)
            p = OracleMemoryless(pi).predict(o); err = ((p >= .5).astype(int) != d).astype(int)
            nwin = T // w; err = err[:nwin * w]; s = s[:nwin * w]
            real = err.reshape(nwin, w).sum(1)
            e_s = np.array([err[s == 0].mean(), err[s == 1].mean()]); pe = np.array([1 - s.mean(), s.mean()])
            ebar = pe @ e_s; vpe = pe @ e_s ** 2 - ebar ** 2; k = np.arange(1, w)
            binom = w * ebar * (1 - ebar)
            theory = 1 + 2 * vpe * np.sum((w - k) * lam ** k) / binom
            shuf = rng.permutation(err).reshape(nwin, w).sum(1)
            # exact filter on a shorter prefix: its own TCE dispersion ratio (implied joint by FFBS)
            Tf = (T_filter // w) * w; ex = ExactFilter(A, pi)
            pf, bel, Hf = ex.predict(o[:Tf]); decf = (pf >= .5).astype(int); errf = (decf != d[:Tf]).astype(int)
            realf = errf.reshape(Tf // w, w).sum(1)
            vf, pitf = implied_ffbs(bel, A, Hf, decf, realf, w, rng, R=R)
            tf = tce(realf, vf, pitf, rng)
            per.append({"realised_over_binomial": float(real.var() / binom), "theory_eq35": float(theory),
                        "shuffled_over_binomial": float(shuf.var() / binom), "e_s": e_s.tolist(),
                        "asym_eq36": float(1 + 2 * lam / (1 - lam) * vpe / (ebar * (1 - ebar))),
                        "c": float(vpe / (ebar * (1 - ebar))),
                        "exact_filter_phi": tf["phi_hat"], "exact_filter_pass": tf["pass"],
                        "exact_filter_err_rate": float(errf.mean()), "memoryless_err_rate": float(ebar)})
        f = lambda key: [r[key] for r in per]
        rows.append({"lambda": lam, "a": a, "b": b,
                     "realised_mean": float(np.mean(f("realised_over_binomial"))),
                     "realised_min": float(np.min(f("realised_over_binomial"))),
                     "realised_max": float(np.max(f("realised_over_binomial"))),
                     "theory_mean": float(np.mean(f("theory_eq35"))),
                     "shuffled_mean": float(np.mean(f("shuffled_over_binomial"))),
                     "asymptotic_eq36": float(np.mean(f("asym_eq36"))), "c_mean": float(np.mean(f("c"))),
                     "exact_filter_phi_mean": float(np.mean(f("exact_filter_phi"))),
                     "exact_filter_phi_min": float(np.min(f("exact_filter_phi"))),
                     "exact_filter_phi_max": float(np.max(f("exact_filter_phi"))),
                     "exact_filter_passes": int(sum(f("exact_filter_pass"))),
                     "exact_filter_err_rate": float(np.mean(f("exact_filter_err_rate"))),
                     "memoryless_err_rate": float(np.mean(f("memoryless_err_rate")))})
    # lag covariance at lambda = 0.95, one long stream
    lam = 0.95; a, b = 0.2 * (1 - lam), 0.8 * (1 - lam)
    rng = np.random.default_rng(seed + 7)
    o, d, s, A, pi = regime_stream(400000, a, b, rng)
    p = OracleMemoryless(pi).predict(o); err = ((p >= .5).astype(int) != d).astype(int)
    e_s = np.array([err[s == 0].mean(), err[s == 1].mean()]); pe = np.array([1 - s.mean(), s.mean()])
    ebar = pe @ e_s; vpe = pe @ e_s ** 2 - ebar ** 2; ks = np.arange(1, 61)
    ce = [float(np.mean((err[:-k] - ebar) * (err[k:] - ebar))) for k in ks]
    lagcov = {"lambda": lam, "k": ks.tolist(), "empirical": ce, "theory": (lam ** ks * vpe).tolist(),
              "rel_l2_error": float(np.linalg.norm(np.array(ce) - lam ** ks * vpe) / np.linalg.norm(lam ** ks * vpe))}
    return {"sweep": rows, "window": w, "seeds": list(seeds), "T": T, "lagcov": lagcov, "lemma1_check": lemma1_check(seed)}


if __name__ == "__main__":
    import json; r = run(); print(json.dumps({"sweep": r["sweep"], "lagcov_err": r["lagcov"]["rel_l2_error"], "lemma1": r["lemma1_check"]}, indent=1))
