"""Figures for the paper, read from results/results.json. Writes figures/*.pdf.

Palette: first three categorical slots of the reference palette (validated for CVD
separation on a light surface). Every series also carries its own marker and line
style so identity survives greyscale printing; every figure has a table in the paper.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
W, H = 3.15, 2.3


def fig_lambda():
    sw = R["E6"]["sweep"]
    lam = np.array([r["lambda"] for r in sw])
    grid = np.linspace(0, 0.98, 200); w = R["E6"]["window"]
    # theory curve from eq. (3.5) using the mean Var_pi(e)/(ebar(1-ebar)) ratio implied by each point
    fig, ax = plt.subplots(figsize=(W, H))
    c = np.mean([r["c_mean"] for r in sw]); k = np.arange(1, w)
    ax.plot(grid, [1 + 2 * c * np.sum((1 - k / w) * g ** k) for g in grid], color=BLUE, ls="-", label="eq. (3.5), window $w=50$")
    ga = grid[grid < 0.975]
    ax.plot(ga, 1 + 2 * c * ga / (1 - ga), color=ORANGE, ls="--", label=r"eq. (3.6), $w\to\infty$")
    m = np.array([r["realised_mean"] for r in sw]); lo = m - np.array([r["realised_min"] for r in sw]); hi = np.array([r["realised_max"] for r in sw]) - m
    ax.errorbar(lam, m, yerr=[lo, hi], fmt="o", ms=4.5, color=BLUE, mfc="white", mew=1.4, capsize=2, lw=1, label="simulated (3 seeds, min–max)")
    ax.plot(lam, [r["shuffled_mean"] for r in sw], color=AQUA, ls=":", marker="s", ms=4, label="time-shuffled stream")
    if "exact_filter_phi_mean" in sw[0]:
        ax.plot(lam, [r["exact_filter_phi_mean"] for r in sw], color=INK, ls="none", marker="D", ms=3.5, label=r"exact filter, own TCE $\hat\varphi$")
    ax.set_xlabel(r"regime persistence $\lambda = 1-a-b$")
    ax.set_ylabel(r"Var$(N_w)$ / binomial baseline")
    ax.set_ylim(0.7, 6.2)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_lambda_sweep.pdf")); plt.close(fig)


def fig_lagcov():
    L = R["E6"]["lagcov"]; k = np.array(L["k"])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(k, L["theory"], color=ORANGE, ls="--", label=r"$\lambda^k\,\mathrm{Var}_\pi(e)$, eq. (3.4)")
    ax.plot(k, L["empirical"], ls="none", marker="o", ms=3, color=BLUE, mfc="white", mew=1.1, label="empirical Cov$(E_t,E_{t+k})$")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("lag $k$"); ax.set_ylabel("error autocovariance")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUT, "fig_lagcov.pdf")); plt.close(fig)


def fig_coupling():
    C = R["E3"]["coupling"]; e = np.array(C["eps"])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.loglog(e, C["P_all_flip_coinciding"], color=BLUE, ls="-", marker="o", ms=4.5, mfc="white", mew=1.3, label=f"coincident thresholds (slope {C['slope_coinciding']:.2f})")
    ax.loglog(e, C["P_all_flip_spread"], color=ORANGE, ls="--", marker="^", ms=4.5, label=f"spread thresholds (slope {C['slope_spread']:.2f})")
    ax.loglog(e, C["P_all_flip_independent"], color=AQUA, ls=":", marker="s", ms=4, label=f"independent perturbations (slope {C['slope_independent']:.2f})")
    ax.set_xticks(e); ax.set_xticklabels([f"{x:g}" for x in e]); ax.minorticks_off()
    ax.set_xlabel(r"perturbation size $\varepsilon$"); ax.set_ylabel(f"P(all {C['H']} hops flip)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.savefig(os.path.join(OUT, "fig_coupling.pdf")); plt.close(fig)


def fig_e1():
    S = R["E1"]["summary"]
    order = [("memoryless", "memoryless head"), ("oracle_memoryless", "Bayes-optimal memoryless"),
             ("memoryless_online_platt", "memoryless + online Platt"), ("windowed_L10", "history window, $L=10$"),
             ("bsf_s1_no_memory", r"ablation: $A=\mathbf{1}\pi^\top$"), ("bsf_s1_oracle_labels", "BSF-S1, oracle labels"),
             ("bsf_s1_unsupervised", "BSF-S1, unsupervised"), ("exact_filter", "exact filter (floor)")]
    order = [o for o in order if o[0] in S]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(W + 0.4, H + 0.15))
    for yi, (k, lab) in zip(y, order):
        d = S[k]["ece_regime1"]
        ax.plot([d["min"], d["max"]], [yi, yi], color=BLUE, lw=1.4, solid_capstyle="round")
        ax.plot(d["mean"], yi, "o", color=BLUE, ms=5, mfc="white" if k != "exact_filter" else BLUE, mew=1.4)
    floor = S["exact_filter"]["ece_regime1"]["mean"]
    ax.axvline(floor, color=MUTED, ls="--", lw=0.9); ax.text(floor, y[-1] - 0.45, " exact-filter floor", color=MUTED, fontsize=7, va="top")
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("ECE inside the rare regime\n(5 seeds: mean and min–max)")
    ax.set_ylim(y[-1] - 1.0, y[0] + 0.6)
    fig.savefig(os.path.join(OUT, "fig_e1_regime1.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_lambda(); fig_lagcov(); fig_coupling(); fig_e1()
    print("wrote", sorted(os.listdir(OUT)))
