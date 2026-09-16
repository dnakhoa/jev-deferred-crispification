"""E2 — Trajectory calibration, TCE (Lemma 1(ii)-(iv)), round-2 version.

Five seeds. Models: memoryless, oracle memoryless, BSF-S1 (oracle labels), exact filter.
TCE_w components: dispersion ratio phi_hat (realised var / model-implied total var) with a
window-bootstrap 95% CI — the clustering-specific statistic — and the KS p-value of the
randomised PIT, which also rejects for plain hop-level miscalibration and is therefore
secondary. Control: the memoryless head on a time-SHUFFLED stream (no clustering) should
give phi_hat ~ 1 even if KS rejects.
Verdict rule (fixed before looking): pass iff the phi_hat 95% CI contains 1.
Also: hop-level ECE identical on stream and shuffle (Lemma 1(iv)); realised variance vs the
binomial baseline against eq. (3.5) at the head's measured regime error rates; lag-covariance
test of eq. (3.4).
"""
import numpy as np
from scipy.stats import kstest
from common import *


def poisson_binomial_cdf(p):
    dist = np.array([1.0])
    for q in p: dist = np.convolve(dist, [1 - q, q])
    return np.cumsum(dist)


def implied_independent(p_err, real, w, rng):
    nwin = len(real)
    wv = np.array([(p_err[i*w:(i+1)*w] * (1 - p_err[i*w:(i+1)*w])).sum() for i in range(nwin)])
    wm = np.array([p_err[i*w:(i+1)*w].sum() for i in range(nwin)])
    var_total = wv.mean() + wm.var()
    pit = np.empty(nwin)
    for i in range(nwin):
        cdf = poisson_binomial_cdf(p_err[i*w:(i+1)*w]); n = real[i]
        lo = cdf[n-1] if n > 0 else 0.0; pit[i] = lo + rng.random() * (cdf[n] - lo)
    return var_total, pit


def implied_ffbs(bel, A, H, dec, real, w, rng, R=100):
    T = len(dec); nwin = len(real); samples = np.empty((R, nwin))
    for r in range(R):
        sp = ffbs(bel, A, rng)
        pe = 1 - np.where(dec == 1, H[np.arange(T), sp], 1 - H[np.arange(T), sp])
        e = (rng.random(T) < pe).astype(int)
        samples[r] = [e[i*w:(i+1)*w].sum() for i in range(nwin)]
    pit = np.array([(np.sum(samples[:, i] < real[i]) + rng.random() * np.sum(samples[:, i] == real[i])) / R for i in range(nwin)])
    return samples.var(), pit


def phi_ci(real, var_impl, rng, B=1000):
    n = len(real); boots = np.array([real[rng.integers(0, n, n)].var() / var_impl for _ in range(B)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def tce(real, var_impl, pit, rng):
    phi = float(real.var() / var_impl); ci = phi_ci(real, var_impl, rng)
    return {"phi_hat": phi, "phi_ci95": ci, "ks_pvalue": float(kstest(pit, "uniform").pvalue), "pass": bool(ci[0] <= 1 <= ci[1])}


def run(seed=0, seeds=(0, 1, 2, 3, 4), T_train=30000, T_test=25000, w=50, R=100):
    a, b = 0.01, 0.04; lam = 1 - a - b
    out = {"seeds": list(seeds), "window": w, "per_seed": []}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        o, d, s, A, pi = regime_stream(T_train, a, b, rng)
        o2, d2, s2, _, _ = regime_stream(T_test, a, b, rng)
        nwin = T_test // w; T = nwin * w; o2, d2, s2 = o2[:T], d2[:T], s2[:T]
        ml = MemorylessHead().fit(o, d); bs = BSFS1().fit(o, d, s); ex = ExactFilter(A, pi); om = OracleMemoryless(pi)
        res = {"seed": sd}
        # memoryless-type heads: implied joint = independence
        for name, p in (("memoryless", ml.predict(o2)), ("oracle_memoryless", om.predict(o2))):
            dec = (p >= .5).astype(int); err = (dec != d2).astype(int)
            real = np.array([err[i*w:(i+1)*w].sum() for i in range(nwin)])
            v, pit = implied_independent(1 - np.maximum(p, 1 - p), real, w, rng)
            res[name] = tce(real, v, pit, rng)
            if name == "memoryless":
                # Lemma 1(iv): hop-ECE on stream and shuffle; (3.5) vs binomial; lag-cov (3.4)
                perm = rng.permutation(T)
                res["hop_ece"] = {"stream": ece(p, d2), "shuffled": ece(p[perm], d2[perm])}
                pi_emp = np.array([1 - s2.mean(), s2.mean()]); e_s = np.array([err[s2 == 0].mean(), err[s2 == 1].mean()])
                ebar = pi_emp @ e_s; vpe = pi_emp @ e_s ** 2 - ebar ** 2; k_ = np.arange(1, w)
                res["binomial_ratio"] = {"realised": float(real.var() / (w * ebar * (1 - ebar))),
                                         "theory_eq35": float(1 + 2 * vpe * np.sum((w - k_) * lam ** k_) / (w * ebar * (1 - ebar))),
                                         "regime_error_rates": e_s.tolist()}
                ks = np.arange(1, 21)
                ce = np.array([np.mean((err[:-k] - ebar) * (err[k:] - ebar)) for k in ks]); ct = lam ** ks * vpe
                res["lagcov_rel_l2_error"] = float(np.linalg.norm(ce - ct) / np.linalg.norm(ct))
                # control: shuffled stream, no clustering by construction
                ps, ds = p[perm], d2[perm]; decs = (ps >= .5).astype(int); errs = (decs != ds).astype(int)
                reals = np.array([errs[i*w:(i+1)*w].sum() for i in range(nwin)])
                vs, pits = implied_independent(1 - np.maximum(ps, 1 - ps), reals, w, rng)
                res["memoryless_shuffled_control"] = tce(reals, vs, pits, rng)
        # filter models: implied joint = FFBS
        for name, mdl in (("bsf_s1_oracle_labels", bs), ("exact_filter", ex)):
            p, bel, H = mdl.predict(o2); dec = (p >= .5).astype(int); err = (dec != d2).astype(int)
            real = np.array([err[i*w:(i+1)*w].sum() for i in range(nwin)])
            v, pit = implied_ffbs(bel, mdl.A, H, dec, real, w, rng, R=R)
            res[name] = tce(real, v, pit, rng)
        out["per_seed"].append(res)
    names = ["memoryless", "oracle_memoryless", "bsf_s1_oracle_labels", "exact_filter", "memoryless_shuffled_control"]
    out["summary"] = {n: {"phi_mean": float(np.mean([r[n]["phi_hat"] for r in out["per_seed"]])),
                          "phi_min": float(np.min([r[n]["phi_hat"] for r in out["per_seed"]])),
                          "phi_max": float(np.max([r[n]["phi_hat"] for r in out["per_seed"]])),
                          "passes": int(sum(r[n]["pass"] for r in out["per_seed"])),
                          "ks_reject_at_0.01": int(sum(r[n]["ks_pvalue"] < 0.01 for r in out["per_seed"]))} for n in names}
    out["summary"]["hop_ece_identical_all_seeds"] = bool(all(abs(r["hop_ece"]["stream"] - r["hop_ece"]["shuffled"]) < 1e-12 for r in out["per_seed"]))
    out["summary"]["binomial_ratio"] = {"realised_mean": float(np.mean([r["binomial_ratio"]["realised"] for r in out["per_seed"]])),
                                        "theory_mean": float(np.mean([r["binomial_ratio"]["theory_eq35"] for r in out["per_seed"]]))}
    out["summary"]["lagcov_rel_l2_error_mean"] = float(np.mean([r["lagcov_rel_l2_error"] for r in out["per_seed"]]))
    return out


if __name__ == "__main__":
    import json; print(json.dumps(run()["summary"], indent=1))
