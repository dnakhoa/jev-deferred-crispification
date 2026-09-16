"""Run E1-E5 and write results/results.json + results/RESULTS.md.
CPU only. E1 and E2 run five seeds each (~10 min total on a laptop).
Usage: python3 experiments/run_all.py
"""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
import e1_regime_shift, e2_trajectory_calibration, e3_cliff_cascade, e4_composition_algebra, e5_type2_separation

out_dir = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(out_dir, exist_ok=True)
results = {}
for name, mod in [("E1", e1_regime_shift), ("E2", e2_trajectory_calibration), ("E3", e3_cliff_cascade),
                  ("E4", e4_composition_algebra), ("E5", e5_type2_separation)]:
    cached = os.path.join(out_dir, f"{name.lower()}.json")
    t0 = time.time()
    if os.environ.get("USE_CACHED") and os.path.exists(cached):
        results[name] = json.load(open(cached))
    else:
        results[name] = mod.run()
    results[name]["_seconds"] = round(time.time() - t0, 1)
    print(f"{name} done in {results[name]['_seconds']}s", flush=True)
json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)

E1, E2, E3, E4, E5 = (results[k] for k in ("E1", "E2", "E3", "E4", "E5"))
def mr(d): return f"{d['mean']:.3f} [{d['min']:.3f}, {d['max']:.3f}]"
S1 = E1["summary"]; S2 = E2["summary"]; B = E3["base_rates"]; C = E3["coupling"]
names1 = [("memoryless", "Memoryless head (quad. logistic)"), ("oracle_memoryless", "Bayes-optimal memoryless head"),
          ("windowed_L10", "History-window head, L=10"), ("memoryless_online_platt", "Memoryless + online Platt (W=500)"),
          ("bsf_s1_oracle_labels", "BSF-S1, oracle regime labels"), ("bsf_s1_unsupervised", "BSF-S1, unsupervised regimes"),
          ("exact_filter", "Exact filter, true (A, g, h) — floor")]
names2 = [("memoryless", "Memoryless"), ("oracle_memoryless", "Bayes-optimal memoryless"), ("bsf_s1_oracle_labels", "BSF-S1 (oracle labels)"),
          ("exact_filter", "Exact filter (floor)"), ("memoryless_shuffled_control", "Control: memoryless on shuffled stream")]
agree = 100 * sum(m['unsup_label_agreement'] for m in E1['per_seed']['_meta']) / len(E1['per_seed']['_meta'])
md = ["# Results — round 2 (five seeds; mean [min, max])\n", "Regenerate with `python3 experiments/run_all.py`. CPU only.\n",
"## E1 — Regime-shift stress (Lemma 1(i))\n",
"| Model | ECE stationary | ECE inside regime 1 | ECE shifted (stale A) | NLL stationary | Acc. stationary | Acc. shifted |", "|---|---|---|---|---|---|---|",
*[f"| {lab} | {mr(S1[k]['ece_stationary'])} | {mr(S1[k]['ece_regime1'])} | {mr(S1[k]['ece_shifted'])} | {mr(S1[k]['nll_stationary'])} | {mr(S1[k]['acc_stationary'])} | {mr(S1[k]['acc_shifted'])} |" for k, lab in names1],
"\nRegime-conditional ECE conditions on the true regime, which the filter cannot observe, so the exact filter's row is the floor for that column. "
f"Unsupervised regime labels agree with the truth on {agree:.0f}% of steps on average.\n",
"## E2 — Trajectory calibration, TCE (Lemma 1(ii)–(iv))\n",
"| Model | φ̂ mean [min, max] | passes (φ̂ CI ∋ 1) / 5 | KS rejects at 0.01 / 5 |", "|---|---|---|---|",
*[f"| {lab} | {S2[k]['phi_mean']:.2f} [{S2[k]['phi_min']:.2f}, {S2[k]['phi_max']:.2f}] | {S2[k]['passes']} | {S2[k]['ks_reject_at_0.01']} |" for k, lab in names2],
f"\nHop-level ECE identical on stream and shuffle on every seed: {S2['hop_ece_identical_all_seeds']} (Lemma 1(iv)). "
f"Realised variance of the memoryless head's window count vs the binomial baseline: {S2['binomial_ratio']['realised_mean']:.2f} (eq. 3.5 predicts {S2['binomial_ratio']['theory_mean']:.2f}). "
f"Lag-covariance test of eq. 3.4, mean relative L2 error {S2['lagcov_rel_l2_error_mean']:.2f}. "
"The shuffled control shows why φ̂ is the clustering-specific statistic: it returns to ≈1 on the shuffle while KS still rejects (hop-level miscalibration).\n",
"## E3 — Cliff cascade (Lemma 2), fair comparison\n",
f"Base rates: crisp conjunction fires on {100*B['crisp']:.1f}% of inputs; t-norm+centroid ≥ ½ fires on {100*B['tnorm_centroid_at_tau_half']:.1f}%. Single-collapse pipelines are thresholded at their base-rate-matched quantile. Min t-norm at ½ equals the crisp conjunction bit-for-bit: {E3['min_tnorm_equals_crisp']}.\n",
"| ε | crisp, own adversary | t-norm+centroid (matched), own adversary | mean-of-probs (matched), own adversary | crisp, Gaussian | t-norm+centroid (matched), Gaussian | t-norm+centroid at ½, crisp adversary (the unfair number) |", "|---|---|---|---|---|---|---|",
*[f"| {r['eps']} | {r['crisp_own_adv']:.4f} | {r['tnorm_centroid_matched_own_adv']:.4f} | {r['mean_prob_matched_own_adv']:.4f} | {r['crisp_gauss']:.4f} | {r['tnorm_centroid_matched_gauss']:.4f} | {r['tnorm_centroid_tau_half_crisp_adv']:.4f} |" for r in E3['fair_pipeline']],
"\nSingle gate: crisp FM/ε = " + ", ".join(f"{r['crisp_FM_over_eps']:.3f}" for r in E3['single_gate']) + f" (limit 2p(0) = {E3['theory']['crisp_FM_over_eps_limit']:.3f}); fuzzy AMS/ε = {E3['single_gate'][0]['fuzzy_AMS_over_eps']:.3f} (bound ¼).\n",
"| coupling weight v | P(gate 2 flips \\| gate 1 flipped) | v·p(0) |", "|---|---|---|",
*[f"| {r['v']} | {r['P_hop2_flips_given_hop1']:.3f} | {r['v_times_p0']:.3f} |" for r in E3['cascade_vs_v']],
f"\nCoupling (N = 4×10⁶ for the spread case, {C['spread_event_counts']} events): log–log slope of P(all {C['H']} flip) vs ε = {C['slope_coinciding']:.2f} coincident (theory 1), {C['slope_spread']:.2f} spread (theory {C['H']}), {C['slope_independent']:.2f} independent (theory {C['H']}).\n",
"## E4 — Composition algebra (an arithmetic identity, reported for completeness)\n",
"| Algebra | idempotence gap | answer-flip rate |", "|---|---|---|",
*[f"| {k} | {E4[k]['idempotence_gap']:.3f} | {E4[k]['answer_flip_rate']:.3f} |" for k in ('min', 'product', 'naive_bayes')],
"\n## E5 — Type-2 separation\n",
"| Population | point-head output | point CRPS (=MAE) | Q fit (α, β) | Q CRPS |", "|---|---|---|---|---|",
*[f"| {k} | {E5[k]['point_head_output']:.3f} | {E5[k]['point_head_CRPS(=MAE)']:.3f} | ({E5[k]['Q_head_fit_(alpha,beta)'][0]:.1f}, {E5[k]['Q_head_fit_(alpha,beta)'][1]:.1f}) | {E5[k]['Q_head_CRPS']:.3f} |" for k in ('sharp', 'diffuse')],
]
open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md))
print("wrote results/results.json and results/RESULTS.md")
