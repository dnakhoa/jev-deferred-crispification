"""E3 — Cliff cascade (Lemma 2); metrics FM(eps) and AMS(eps).

Scores are logistic in the evidence so the sup over an eps-ball is exact
(move along the weight vector). D = identity (L_D = 1), T = product t-norm.

(i)/(ii)  FM(eps)/eps -> constant for crisp, FM = 0 for fuzzy; AMS/eps bounded.
(iii)     sequential: P(hop2 flips | hop1 flips) = Theta(1) crisp vs O(eps) fuzzy.
(iv)      common-mode: log P(all H flip) vs log eps has slope ~1 when the
          perturbation acts on the shared latent, ~H when independent.
"""
import numpy as np
from scipy.special import expit


def run(seed=0, N=200000, n=5, H=3):
    rng = np.random.default_rng(seed)
    eps_grid = np.array([0.002, 0.005, 0.01, 0.02, 0.05])
    W = rng.standard_normal((H, n)); W /= np.linalg.norm(W, axis=1, keepdims=True)
    c = np.zeros(H); tau = 0.5
    X = rng.standard_normal((N, n))
    Z = X @ W.T + c                                     # logits, (N,H)
    F = expit(Z)

    # ---- (i)/(ii): single hop, crisp vs fuzzy ----
    out_single = []
    for eps in eps_grid:
        # crisp: flip iff |logit| <= eps * ||w|| (=eps, unit w)
        fm = float((np.abs(Z[:, 0]) <= eps).mean())
        # fuzzy: sup |sigma(z+eps) - sigma(z)| over the ball = along w
        ams = float(np.abs(expit(Z[:, 0] + eps) - expit(Z[:, 0])).mean())
        out_single.append({"eps": eps, "crisp_FM": fm, "crisp_FM_over_eps": fm / eps,
                           "fuzzy_FM": 0.0, "fuzzy_AMS_over_eps": ams / eps})
    # density of the logit at 0 ~ N(0,1): p(0)=0.3989 -> FM/eps -> 2*p = 0.798

    # ---- (iii): sequential cascade ----
    v = 1.5                                             # weight hop 2 places on hop 1's output
    a1 = (Z[:, 0] >= 0).astype(float)
    z2_crisp = Z[:, 1] + v * a1
    z2_fuzzy = Z[:, 1] + v * F[:, 0]
    out_seq = []
    for eps in eps_grid:
        flip1 = np.abs(Z[:, 0]) <= eps
        # crisp: after hop-1 flip, hop-2 logit moves by +-v, an O(1) change
        a1p = 1 - a1
        z2p = Z[:, 1] + v * a1p
        flip2_given_1 = float((((z2_crisp >= 0) != (z2p >= 0))[flip1]).mean()) if flip1.any() else 0.0
        # fuzzy: hop-2 membership moves by v * (sigma(z1 +- eps) - sigma(z1)) <= v*eps/4
        dmu2 = np.abs(expit(z2_fuzzy + v * (expit(Z[:, 0] + eps) - F[:, 0])) - expit(z2_fuzzy))
        out_seq.append({"eps": eps, "crisp_P_hop2_flips_given_hop1_flip": flip2_given_1,
                        "fuzzy_sup_change_hop2": float(dmu2.mean())})

    # ---- (iv): common-mode coupling. f_h = sigma(z + xi_h), shared z. ----
    # Case A: crossing points coincide (xi_h = 0): all H hops threshold the same
    #         quantity (model + harness thresholding one score; duplicate predicates).
    # Case B: crossing points spread by a continuous density (xi_h ~ N(0, 0.3^2)).
    # Case C: independent per-hop perturbations (product of marginals).
    z = rng.standard_normal(N); xi = 0.3 * rng.standard_normal((N, H))
    out_cm = []
    for eps in eps_grid:
        lA = np.repeat(z[:, None], H, axis=1); lB = z[:, None] + xi
        allA = float(np.all(np.abs(lA) <= eps, axis=1).mean())
        allB = float((np.all(np.abs(lB) <= eps, axis=1) & (np.ptp(np.sign(lB), axis=1) == 0)).mean())
        allC = float(np.prod((np.abs(lB) <= eps).mean(axis=0)))
        out_cm.append({"eps": eps, "P_all_flip_coinciding": allA, "P_all_flip_spread": allB, "P_all_flip_independent": allC})
    le = np.log(eps_grid)
    def slope(key):
        v = np.array([r[key] for r in out_cm]); ok = v > 0
        return float(np.polyfit(le[ok], np.log(v[ok]), 1)[0]) if ok.sum() > 1 else None
    return {"single_hop": out_single, "sequential": out_seq, "common_mode": out_cm,
            "loglog_slope_coinciding": slope("P_all_flip_coinciding"),
            "loglog_slope_spread": slope("P_all_flip_spread"),
            "loglog_slope_independent": slope("P_all_flip_independent"), "H": H,
            "theory": {"crisp_FM_over_eps_limit": 2 * 0.39894, "fuzzy_AMS_over_eps_bound": 0.25}}


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=2))
