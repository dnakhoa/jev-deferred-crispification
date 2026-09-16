"""Numerical check of Lemma 2 (cliff cascade), symmetric to verify_lemma1.py.

Typed + threshold pipeline vs fuzzy + single centroid defuzzification, same
evidence, same scores. Two perturbation models (Appendix / Section 7, AMS):
  adversarial : delta = argmax over the eps-ball (along the score gradient)
  gaussian    : delta ~ N(0, eps^2 I)
Quantities (all "action mass"):
  TV_action  : total variation between the distribution of the FINAL discrete
               action before and after perturbation. For the crisp pipeline the
               action is the threshold gate itself; for the fuzzy pipeline it is
               the actuator's cost-sensitive threshold on the centroid u.
  flip_mass  : P(final action changes) for the same x  (= per-point TV).
  L1_signal  : E|u(x+delta) - u(x)| for the continuous actuator signal.
Lemma 2 predicts: crisp flip_mass = Theta(eps) with jumps of size 1 and
Theta(1) cascade; fuzzy signal change <= L_D*H*L*eps everywhere; and the fuzzy
pipeline's *final* discrete action, taken once at the actuator, flips on a set
of the same order eps but only ONCE, not once per hop.
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import expit

rng = np.random.default_rng(0)
N, n, H = 200000, 5, 3
W = rng.standard_normal((H, n)); W /= np.linalg.norm(W, axis=1, keepdims=True)
X = rng.standard_normal((N, n))
eps_grid = [0.005, 0.01, 0.02, 0.05]

# --- fuzzy actuator: Mamdani with two consequents on Y=[0,1], regularised centroid
Y = np.linspace(0, 1, 201); m0 = 0.05
C_low = np.clip(1 - 2 * Y, 0, 1); C_high = np.clip(2 * Y - 1, 0, 1)
def centroid(w_high):                       # w_high: firing degree of "proceed", (N,)
    mu_out = np.maximum(np.minimum(w_high[:, None], C_high[None, :]),
                        np.minimum(1 - w_high[:, None], C_low[None, :]))
    area = trapezoid(mu_out, Y, axis=1)
    lift = np.clip(m0 - area, 0, None)[:, None] / (Y[-1] - Y[0])
    mu_out = mu_out + lift
    return trapezoid(Y * mu_out, Y, axis=1) / trapezoid(mu_out, Y, axis=1)

def crisp_action(Xp):                       # H gates, conjunction
    return np.all((Xp @ W.T) >= 0, axis=1).astype(int)
def fuzzy_signal(Xp):                       # product t-norm of memberships, one centroid
    return centroid(np.prod(expit(Xp @ W.T), axis=1))
def fuzzy_action(Xp, tau=0.5):              # single collapse at the actuator
    return (fuzzy_signal(Xp) >= tau).astype(int)

a0, u0, f0 = crisp_action(X), fuzzy_signal(X), fuzzy_action(X)
print(f"{'eps':>6} {'noise':>11} | {'crisp flip':>10} {'crisp TV':>9} | {'fuzzy L1 sig':>12} {'fuzzy flip':>10} {'fuzzy TV':>9}")
for eps in eps_grid:
    for noise in ("adversarial", "gaussian"):
        if noise == "gaussian":
            D = eps * rng.standard_normal((N, n))
        else:   # move each point along the gradient of the score nearest its threshold
            Z = X @ W.T; h = np.argmin(np.abs(Z), axis=1)
            D = -np.sign(Z[np.arange(N), h])[:, None] * eps * W[h]
        Xp = X + D
        a1, u1, f1 = crisp_action(Xp), fuzzy_signal(Xp), fuzzy_action(Xp)
        tv_c = abs(a1.mean() - a0.mean()); tv_f = abs(f1.mean() - f0.mean())
        print(f"{eps:>6} {noise:>11} | {np.mean(a1 != a0):>10.4f} {tv_c:>9.4f} | "
              f"{np.mean(np.abs(u1 - u0)):>12.5f} {np.mean(f1 != f0):>10.4f} {tv_f:>9.4f}")
print("\nLipschitz bound on the fuzzy signal per unit eps: L_D*H*L with L=1/4 (sigmoid), H=3, L_D from the centroid;")
print("crisp flip mass ~ sum_h 2 p_h(0) rho_h eps, rho_h = P(other gates pass | gate h at threshold) ~ 1/4 here (Lemma 2(ii)); the single-gate limit 0.798*eps applies to E3, not to this 3-gate conjunction.")
# cascade: after a crisp gate flips, a downstream gate that consumes it sees an O(1) change
v = 1.5; Z = X @ W.T; g1 = (Z[:, 0] >= 0).astype(float)
z2 = Z[:, 1] + v * g1; z2p = Z[:, 1] + v * (1 - g1)
near = np.abs(Z[:, 0]) <= 0.01
print(f"\ncascade: P(gate 2 flips | gate 1 flipped) = {np.mean(((z2 >= 0) != (z2p >= 0))[near]):.3f}  (Theta(1), independent of eps)")
mu2 = expit(Z[:, 1] + v * expit(Z[:, 0])); mu2p = expit(Z[:, 1] + v * expit(Z[:, 0] + 0.01))
print(f"         fuzzy: sup change at hop 2 for eps=0.01 = {np.mean(np.abs(mu2p - mu2)):.5f}  (O(eps))")
