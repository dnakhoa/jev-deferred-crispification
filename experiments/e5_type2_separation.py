"""E5 — Type-2 separation (Section 4.5).

Two borderline populations with identical mean degree 0.5:
  sharp   : elicited degree ~ Beta(50,50)
  diffuse : elicited degree ~ Beta(1,1)
Point head: emits the MLE mean (0.5) for both — the SAME output object.
Type-2 head Q: fits a Beta by MLE — distinguishable objects.
CRPS of a point forecast is the MAE; CRPS of Q computed by quadrature.
Prediction: on the sharp population Q ~ point; on the diffuse population
Q strictly beats the point head (uniform CRPS = 1/6 < MAE = 1/4).
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import beta as Beta


def crps_beta(alpha, b_, y, grid=2001):
    ygrid = np.linspace(0, 1, grid)
    F = Beta.cdf(ygrid, alpha, b_)
    out = np.empty(len(y))
    for i, yi in enumerate(y):
        out[i] = trapezoid((F - (ygrid >= yi)) ** 2, ygrid)
    return out


def run(seed=0, N=4000):
    rng = np.random.default_rng(seed)
    res = {}
    for name, (al, be) in {"sharp": (50, 50), "diffuse": (1, 1)}.items():
        mu = rng.beta(al, be, N)
        train, test = mu[: N // 2], mu[N // 2:]
        point = train.mean()
        a_hat, b_hat, _, _ = Beta.fit(train, floc=0, fscale=1)
        res[name] = {
            "point_head_output": float(point),
            "point_head_CRPS(=MAE)": float(np.abs(test - point).mean()),
            "Q_head_fit_(alpha,beta)": [float(a_hat), float(b_hat)],
            "Q_head_CRPS": float(crps_beta(a_hat, b_hat, test).mean()),
        }
    res["separation"] = {
        "point_head_outputs_identical": abs(res["sharp"]["point_head_output"] - res["diffuse"]["point_head_output"]) < 0.02,
        "Q_head_concentration_ratio": (res["sharp"]["Q_head_fit_(alpha,beta)"][0] + res["sharp"]["Q_head_fit_(alpha,beta)"][1])
                                      / (res["diffuse"]["Q_head_fit_(alpha,beta)"][0] + res["diffuse"]["Q_head_fit_(alpha,beta)"][1]),
    }
    return res


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=2))
