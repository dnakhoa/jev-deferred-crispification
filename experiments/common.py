"""Shared machinery for E1-E5.

CPU only, numpy + scipy. Every model here is deliberately tiny: the paper's
claims are about the *output object and its composition*, not about capacity,
so logistic heads on quadratic features are sufficient and keep the runs
reproducible in minutes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

# --------------------------------------------------------------------------- #
# Generators (Appendix D)
# --------------------------------------------------------------------------- #

def two_state_A(a: float, b: float) -> np.ndarray:
    return np.array([[1 - a, a], [b, 1 - b]])


def stationary(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    return p / p.sum()


def regime_stream(T: int, a: float, b: float, rng: np.random.Generator,
                  sep: float = 0.5, gain: float = 3.0):
    """Two-regime HMM. Emissions overlap heavily (means `sep` apart, unit
    covariance) so a single observation is only weakly informative about the
    regime; the *sequence* is informative. The label rule differs by regime so
    that P(d | o, s) differs across s (otherwise Lemma 1(i) is vacuous).
    The label link is logistic, P(d=1|o,s) = sigmoid(gain * o_{k(s)}), so that
    a logistic head on (o1, o2) is well-specified per regime and calibration
    differences are attributable to composition, not to head capacity."""
    A = two_state_A(a, b)
    pi = stationary(A)
    means = np.array([[0.0, 0.0], [sep, sep]])
    s = np.empty(T, dtype=int)
    s[0] = rng.choice(2, p=pi)
    u = rng.random(T)
    for t in range(1, T):
        s[t] = 1 if u[t] < A[s[t - 1], 1] else 0
    o = means[s] + rng.standard_normal((T, 2))
    # regime 0 decides on o1, regime 1 decides on o2 (logistic link)
    logit = gain * np.where(s == 0, o[:, 0], o[:, 1])
    d = (rng.random(T) < expit(logit)).astype(int)
    return o, d, s, A, pi


# --------------------------------------------------------------------------- #
# Tiny heads: logistic regression on quadratic features
# --------------------------------------------------------------------------- #

def quad_features(o: np.ndarray) -> np.ndarray:
    o1, o2 = o[:, 0], o[:, 1]
    return np.column_stack([np.ones_like(o1), o1, o2, o1 * o1, o2 * o2, o1 * o2])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    def nll(w):
        z = X @ w
        return np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + l2 * w @ w

    def grad(w):
        p = expit(X @ w)
        return X.T @ (p - y) / len(y) + 2 * l2 * w

    res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B")
    return res.x


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return expit(X @ w)


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    """Stage 3 of Algorithm 3: one-parameter temperature scaling by NLL."""
    def nll(t):
        z = logit / t[0]
        return np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
    return float(minimize(nll, [1.0], bounds=[(0.05, 20)]).x[0])


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #

def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of a binary score p for outcome y (0/1),
    computed on the predicted class as in Guo et al. (2017)."""
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


# --------------------------------------------------------------------------- #
# BSF-S1 filter (eq. 3.7 / Algorithm 1), Baum-Welch for A, FFBS (Appendix C)
# --------------------------------------------------------------------------- #

def scaled_likelihood(g: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Delta b_t(s) = g_s(o_t) / pi(s), row-normalised. g: (T, K)."""
    L = g / pi[None, :]
    return L / L.sum(axis=1, keepdims=True)


def forward_filter(L: np.ndarray, A: np.ndarray, b0: np.ndarray):
    """Exact forward recursion on scaled likelihoods.
    Returns filtered beliefs b (T,K), predicted beliefs b_pred (T,K) and the
    per-step belief surprise S_t = -log sum_s b_pred[s] L[s]."""
    T, K = L.shape
    b = np.empty((T, K))
    bp = np.empty((T, K))
    S = np.empty(T)
    prev = b0
    for t in range(T):
        pred = A.T @ prev
        num = pred * L[t]
        z = num.sum()
        S[t] = -np.log(z + 1e-300)
        b[t] = num / z
        bp[t] = pred
        prev = b[t]
    return b, bp, S


def baum_welch_A(L: np.ndarray, A0: np.ndarray, iters: int = 30) -> np.ndarray:
    """Stage 1 of Algorithm 3: re-estimate A with the emission heads fixed,
    treating the scaled likelihoods as emission terms."""
    T, K = L.shape
    A = A0.copy()
    for _ in range(iters):
        # forward (scaled)
        alpha = np.empty((T, K)); c = np.empty(T)
        alpha[0] = stationary(A) * L[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (A.T @ alpha[t - 1]) * L[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        # backward
        beta = np.empty((T, K)); beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (L[t + 1] * beta[t + 1]) / c[t + 1]
        # expected transitions
        xi = np.zeros((K, K))
        for t in range(T - 1):
            m = np.outer(alpha[t], L[t + 1] * beta[t + 1]) * A / c[t + 1]
            xi += m
        A = xi / xi.sum(axis=1, keepdims=True)
    return A


def ffbs(b: np.ndarray, A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Forward-filter / backward-sample one regime path from the filtered
    beliefs b (T,K) and transition A."""
    T, K = b.shape
    s = np.empty(T, dtype=int)
    s[-1] = rng.choice(K, p=b[-1])
    for t in range(T - 2, -1, -1):
        w = b[t] * A[:, s[t + 1]]
        s[t] = rng.choice(K, p=w / w.sum())
    return s


# --------------------------------------------------------------------------- #
# Model wrappers
# --------------------------------------------------------------------------- #

class MemorylessHead:
    """The system-one baseline: one calibrated head P(d | o), no state.
    Temperature-scaled on a held-out split (same treatment as BSF-S1)."""

    def fit(self, o, d, cal_frac=0.3):
        n = int(len(d) * (1 - cal_frac))
        self.w = fit_logistic(quad_features(o[:n]), d[:n])
        self.T = fit_temperature(quad_features(o[n:]) @ self.w, d[n:])
        return self

    def predict(self, o):
        return expit(quad_features(o) @ self.w / self.T)


class BSFS1:
    """Belief-State head: regime head g, per-regime decision heads h_s,
    transition A learned by Baum-Welch, exact forward filter, decision
    posterior P(d_t | o_{1:t}) = sum_s h_s(o_t) b_t(s)   (eq. 3.8)."""

    def fit(self, o, d, s, A_init=None, bw_iters=30, cal_frac=0.3):
        n = int(len(d) * (1 - cal_frac))
        X, Xc = quad_features(o[:n]), quad_features(o[n:])
        st, sc, dt, dc = s[:n], s[n:], d[:n], d[n:]
        self.wg = fit_logistic(X, st)             # g(o) ≈ P(s=1 | o)
        self.Tg = fit_temperature(Xc @ self.wg, sc)
        self.wh, self.Th = [], []
        for k in (0, 1):                          # h_s(o) ≈ P(d | o, s)
            w = fit_logistic(X[st == k], dt[st == k]); self.wh.append(w)
            self.Th.append(fit_temperature(Xc[sc == k] @ w, dc[sc == k]))
        # pi_train: the prior under which g was fitted. The scaled likelihood
        # must divide by THIS prior (Bourlard-Morgan), never by stationary(A_hat).
        self.pi = np.array([1 - st.mean(), st.mean()])
        # Stage 0/1: initialise A from labelled transition counts (available in
        # this oracle setting), then refine by Baum-Welch on scaled likelihoods.
        if A_init is None:
            cnt = np.zeros((2, 2))
            for u_, v_ in zip(st[:-1], st[1:]):
                cnt[u_, v_] += 1
            A_init = cnt / cnt.sum(axis=1, keepdims=True)
        L = scaled_likelihood(self._g(o[:n]), self.pi)
        self.A = baum_welch_A(L, A_init, iters=bw_iters)
        return self

    def _g(self, o):
        p1 = expit(quad_features(o) @ self.wg / self.Tg)
        return np.column_stack([1 - p1, p1])

    def filter(self, o):
        L = scaled_likelihood(self._g(o), self.pi)
        return forward_filter(L, self.A, stationary(self.A))

    def predict(self, o):
        """Returns decision posterior p (T,), filtered beliefs b (T,2) and
        per-regime decision posteriors H (T,2)."""
        X = quad_features(o)
        H = np.column_stack([expit(X @ w / T) for w, T in zip(self.wh, self.Th)])
        b, _, _ = self.filter(o)
        return (H * b).sum(axis=1), b, H
