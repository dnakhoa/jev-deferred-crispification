"""E3 — Cliff cascade (Lemma 2), round-2 version: a FAIR comparison.

(a) Single gate: crisp FM/eps -> 2 p(0); fuzzy AMS/eps <= L.
(b) Full pipeline, three gates: crisp conjunction vs single-collapse pipelines
    (product t-norm + centroid; plain mean of probabilities) at MATCHED action base
    rate, each perturbed along ITS OWN worst-case direction, plus Gaussian noise.
    Also: min t-norm thresholded at 1/2 is bit-identical to the crisp conjunction.
(c) Cascade P(gate 2 flips | gate 1 flips) as a function of the coupling weight v:
    Theta(1) in eps, magnitude ~ v * p(0) for small v.
(d) Coincident vs spread crossing points vs independent perturbations, N = 4e6 for
    the spread case so that the slope is estimated from thousands of events.
"""
import numpy as np
from scipy.special import expit
from scipy.integrate import trapezoid

Y = np.linspace(0, 1, 201); M0 = 0.05
C_LOW = np.clip(1 - 2 * Y, 0, 1); C_HIGH = np.clip(2 * Y - 1, 0, 1)


def centroid(wh):
    mo = np.maximum(np.minimum(wh[:, None], C_HIGH[None, :]), np.minimum(1 - wh[:, None], C_LOW[None, :]))
    area = trapezoid(mo, Y, axis=1); mo = mo + np.clip(M0 - area, 0, None)[:, None]
    return trapezoid(Y * mo, Y, axis=1) / trapezoid(mo, Y, axis=1)


def run(seed=0, N=200000, n=5, H=3):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((H, n)); W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.standard_normal((N, n)); Z = X @ W.T
    eps_grid = [0.005, 0.01, 0.02, 0.05]
    # (a) single gate
    single = [{"eps": e, "crisp_FM_over_eps": float((np.abs(Z[:, 0]) <= e).mean() / e),
               "fuzzy_AMS_over_eps": float(np.abs(expit(Z[:, 0] + e) - expit(Z[:, 0])).mean() / e)} for e in eps_grid]
    # (b) fair pipeline comparison
    crisp = lambda Xp: np.all(Xp @ W.T >= 0, axis=1).astype(int)
    u_tn = lambda Xp: centroid(np.prod(expit(Xp @ W.T), axis=1))
    u_mean = lambda Xp: expit(Xp @ W.T).mean(1)
    u_min = lambda Xp: expit(Xp @ W.T).min(1)
    a0 = crisp(X); base = a0.mean(); u0 = u_tn(X); m0 = u_mean(X); mn0 = u_min(X)
    tau_u = float(np.quantile(u0, 1 - base)); tau_m = float(np.quantile(m0, 1 - base))
    def grad(f, Xp, h=1e-4):
        G = np.zeros_like(Xp)
        for j in range(n):
            E = np.zeros(n); E[j] = h; G[:, j] = (f(Xp + E) - f(Xp - E)) / (2 * h)
        return G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    Gu = grad(u_tn, X); Gm = grad(u_mean, X)
    hh = np.argmin(np.abs(Z), axis=1)
    fair = []
    for e in eps_grid:
        Dc = -np.sign(Z[np.arange(N), hh])[:, None] * e * W[hh]
        Du = -np.sign(u0 - tau_u)[:, None] * e * Gu; Dm = -np.sign(m0 - tau_m)[:, None] * e * Gm
        Dg = e * rng.standard_normal((N, n))
        fair.append({"eps": e,
            "crisp_own_adv": float(np.mean(crisp(X + Dc) != a0)),
            "tnorm_centroid_matched_own_adv": float(np.mean((u_tn(X + Du) >= tau_u) != (u0 >= tau_u))),
            "mean_prob_matched_own_adv": float(np.mean((u_mean(X + Dm) >= tau_m) != (m0 >= tau_m))),
            "tnorm_centroid_tau_half_crisp_adv": float(np.mean((u_tn(X + Dc) >= .5) != (u0 >= .5))),
            "crisp_gauss": float(np.mean(crisp(X + Dg) != a0)),
            "tnorm_centroid_matched_gauss": float(np.mean((u_tn(X + Dg) >= tau_u) != (u0 >= tau_u))),
            "mean_prob_matched_gauss": float(np.mean((u_mean(X + Dg) >= tau_m) != (m0 >= tau_m))),
            "min_tnorm_tau_half_crisp_adv": float(np.mean((u_min(X + Dc) >= .5) != (mn0 >= .5)))})
    base_rates = {"crisp": float(base), "tnorm_centroid_at_tau_half": float(np.mean(u0 >= .5)), "matched_tau_u": tau_u, "matched_tau_mean": tau_m}
    min_identity = bool(np.array_equal((mn0 >= .5).astype(int), a0))
    # (c) cascade vs v
    near = np.abs(Z[:, 0]) <= 0.01; g1 = (Z[:, 0] >= 0).astype(float)
    cascade = [{"v": v, "P_hop2_flips_given_hop1": float(np.mean((((Z[:, 1] + v * g1) >= 0) != ((Z[:, 1] + v * (1 - g1)) >= 0))[near])),
                "v_times_p0": v * 0.39894} for v in (0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0)]
    # (d) coupling slopes
    N2 = 4_000_000; z = rng.standard_normal(N2); xi = 0.3 * rng.standard_normal((N2, H)); lB = z[:, None] + xi
    eg = np.array([0.02, 0.05, 0.1, 0.2])
    coin = np.array([np.mean(np.abs(z) <= e) for e in eg])
    spread = np.array([np.mean(np.all(np.abs(lB) <= e, axis=1) & (np.ptp(np.sign(lB), axis=1) == 0)) for e in eg])
    indep = np.array([np.prod((np.abs(lB) <= e).mean(0)) for e in eg])
    sl = lambda v: float(np.polyfit(np.log(eg), np.log(v), 1)[0])
    coupling = {"eps": eg.tolist(), "P_all_flip_coinciding": coin.tolist(), "P_all_flip_spread": spread.tolist(),
                "spread_event_counts": (spread * N2).astype(int).tolist(), "P_all_flip_independent": indep.tolist(),
                "slope_coinciding": sl(coin), "slope_spread": sl(spread), "slope_independent": sl(indep), "H": H}
    return {"single_gate": single, "fair_pipeline": fair, "base_rates": base_rates, "min_tnorm_equals_crisp": min_identity,
            "cascade_vs_v": cascade, "coupling": coupling, "theory": {"crisp_FM_over_eps_limit": 0.79788, "fuzzy_AMS_over_eps_bound": 0.25}}


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1))
