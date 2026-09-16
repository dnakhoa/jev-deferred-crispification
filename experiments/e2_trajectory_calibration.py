"""E2 — Trajectory-level calibration (Lemma 1(ii)-(iv)); metric TCE_w.

Every model implies a joint predictive distribution for the window error
count N_w. Memoryless head: Poisson-binomial with its own p_t (independence
is the only joint it can imply). BSF-S1: forward-filter / backward-sample.
TCE_w = (dispersion ratio phi_hat, PIT-histogram L1 distance from uniform).
Also: hop-level ECE is identical for a stream and its shuffle (Lemma 1(iv)).
"""
import numpy as np
from scipy.stats import kstest
from common import regime_stream, MemorylessHead, BSFS1, ece, ffbs


def poisson_binomial_cdf(p):
    """Exact distribution of sum of independent Bernoulli(p_i)."""
    dist = np.array([1.0])
    for q in p:
        dist = np.convolve(dist, [1 - q, q])
    return np.cumsum(dist)


def pit_l1(pit, bins=10):
    h, _ = np.histogram(pit, bins=bins, range=(0, 1))
    return float(np.abs(h / h.sum() - 1 / bins).sum())


def run(seed=0, T_train=30000, T_test=40000, w=50, R=200):
    rng = np.random.default_rng(seed)
    a, b = 0.01, 0.04                              # pi=(0.8,0.2), lambda=0.95
    o, d, s, A, pi = regime_stream(T_train, a, b, rng)
    ml = MemorylessHead().fit(o, d)
    bs = BSFS1().fit(o, d, s)

    o2, d2, s2, _, _ = regime_stream(T_test, a, b, rng)
    p_ml = ml.predict(o2)
    p_bs, bel, H = bs.predict(o2)
    dec_ml = (p_ml >= .5).astype(int); err_ml = (dec_ml != d2).astype(int)
    dec_bs = (p_bs >= .5).astype(int); err_bs = (dec_bs != d2).astype(int)

    # ---- Lemma 1(iv): hop-ECE is permutation-invariant
    perm = rng.permutation(T_test)
    hop = {"memoryless_ece": ece(p_ml, d2), "memoryless_ece_shuffled": ece(p_ml[perm], d2[perm])}

    # ---- windows
    nwin = T_test // w
    real_ml = np.array([err_ml[i*w:(i+1)*w].sum() for i in range(nwin)])
    real_bs = np.array([err_bs[i*w:(i+1)*w].sum() for i in range(nwin)])

    # memoryless implied: Poisson-binomial with p_err = 1 - max(p, 1-p).
    # Implied TOTAL variance of N_w = E[within-window var] + Var[within-window mean].
    perr = 1 - np.maximum(p_ml, 1 - p_ml)
    win_var_ml = np.array([(perr[i*w:(i+1)*w] * (1 - perr[i*w:(i+1)*w])).sum() for i in range(nwin)])
    win_mean_ml = np.array([perr[i*w:(i+1)*w].sum() for i in range(nwin)])
    var_impl_ml_total = win_var_ml.mean() + win_mean_ml.var()
    pit_ml = np.empty(nwin)
    for i in range(nwin):
        cdf = poisson_binomial_cdf(perr[i*w:(i+1)*w])
        n = real_ml[i]
        lo = cdf[n-1] if n > 0 else 0.0
        pit_ml[i] = lo + rng.random() * (cdf[n] - lo)          # randomised PIT

    # BSF-S1 implied: FFBS regime paths, then errors from per-regime heads
    samples = np.empty((R, nwin))
    for r in range(R):
        sp = ffbs(bel, bs.A, rng)
        p_err_path = 1 - np.where(dec_bs == 1, H[np.arange(T_test), sp], 1 - H[np.arange(T_test), sp])
        e = (rng.random(T_test) < p_err_path).astype(int)
        samples[r] = [e[i*w:(i+1)*w].sum() for i in range(nwin)]
    var_impl_bs_total = samples.var()                       # pooled over samples and windows
    pit_bs = np.array([(np.sum(samples[:, i] < real_bs[i]) + rng.random() * np.sum(samples[:, i] == real_bs[i])) / R
                       for i in range(nwin)])

    # ---- direct test of eq. (3.4)/(3.5): Cov(E_t, E_{t+k}) = lambda^k Var_pi(e)
    lam = 1 - a - b
    pi_emp = np.array([1 - s2.mean(), s2.mean()])
    e_s = np.array([err_ml[s2 == 0].mean(), err_ml[s2 == 1].mean()])
    ebar = pi_emp @ e_s; var_pi_e = pi_emp @ e_s**2 - ebar**2
    ks = np.arange(1, 21)
    cov_emp = np.array([np.mean((err_ml[:-k] - ebar) * (err_ml[k:] - ebar)) for k in ks])
    cov_thy = lam**ks * var_pi_e
    k_ = np.arange(1, w)
    phi_w_theory = 1 + 2 * var_pi_e * np.sum((w - k_) * lam**k_) / (w * ebar * (1 - ebar))
    phi_inf_theory = 1 + 2 * lam / (1 - lam) * var_pi_e / (ebar * (1 - ebar))
    return {
        "regime_error_rates_memoryless": e_s.tolist(),
        "covariance_test": {"k": ks.tolist(), "empirical": cov_emp.tolist(), "theory_lambda_k_VarPi": cov_thy.tolist(),
                            "relative_l2_error": float(np.linalg.norm(cov_emp - cov_thy) / np.linalg.norm(cov_thy))},
        "phi_theory_window_w": float(phi_w_theory), "phi_theory_asymptotic": float(phi_inf_theory),
        "hop_level": hop,
        "memoryless": {"phi_hat": float(real_ml.var() / var_impl_ml_total),
                       "phi_hat_vs_binomial": float(real_ml.var() / (w * ebar * (1 - ebar))),
                       "pit_l1": pit_l1(pit_ml), "ks_pvalue": float(kstest(pit_ml, "uniform").pvalue),
                       "mean_errors_per_window": float(real_ml.mean())},
        "bsf_s1": {"phi_hat": float(real_bs.var() / var_impl_bs_total),
                   "pit_l1": pit_l1(pit_bs), "ks_pvalue": float(kstest(pit_bs, "uniform").pvalue),
                   "mean_errors_per_window": float(real_bs.mean())},
        "window": w, "n_windows": int(nwin),
    }


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=2))
