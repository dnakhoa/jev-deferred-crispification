"""Run E1-E5 and write results/results.json + results/RESULTS.md.
CPU only; ~2-4 minutes on a laptop. Usage: python3 experiments/run_all.py [seed]
"""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
import e1_regime_shift, e2_trajectory_calibration, e3_cliff_cascade, e4_composition_algebra, e5_type2_separation

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out_dir = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(out_dir, exist_ok=True)
results = {}
for name, mod in [("E1", e1_regime_shift), ("E2", e2_trajectory_calibration), ("E3", e3_cliff_cascade),
                  ("E4", e4_composition_algebra), ("E5", e5_type2_separation)]:
    t0 = time.time(); results[name] = mod.run(seed=seed); results[name]["_seconds"] = round(time.time() - t0, 1)
    print(f"{name} done in {results[name]['_seconds']}s", flush=True)
json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=2)

E1, E2, E3, E4, E5 = (results[k] for k in ("E1", "E2", "E3", "E4", "E5"))
f = lambda x: f"{x:.4f}"
md = [f"# Results (seed {seed})\n", "All runs CPU-only, numpy/scipy. Regenerate with `python3 experiments/run_all.py`.\n",
"## E1 — Regime-shift stress (Lemma 1(i))\n",
"| Evaluation | Memoryless head ECE | BSF-S1 ECE |", "|---|---|---|",
f"| Stationary stream (same π) | {f(E1['stationary']['memoryless'])} | {f(E1['stationary']['bsf_s1'])} |",
f"| Time steps inside regime 0 | {f(E1['window_conditional']['memoryless'][0])} | {f(E1['window_conditional']['bsf_s1'][0])} |",
f"| Time steps inside regime 1 | {f(E1['window_conditional']['memoryless'][1])} | {f(E1['window_conditional']['bsf_s1'][1])} |",
f"| Shifted stream π′ = {tuple(round(x,2) for x in E1['shifted']['pi_prime'])}, stale A | {f(E1['shifted']['memoryless'])} | {f(E1['shifted']['bsf_s1'])} |",
f"\nAccuracy on the shifted stream: memoryless {f(E1['accuracy_shifted']['memoryless'])}, BSF-S1 {f(E1['accuracy_shifted']['bsf_s1'])}. "
f"Baum–Welch recovered A ≈ {[[round(x,3) for x in r] for r in E1['A_hat']]} (true {[[round(x,3) for x in r] for r in E1['A_true']]}).\n",
"## E2 — Trajectory calibration, TCE (Lemma 1(ii)–(iv))\n",
f"Hop-level ECE of the memoryless head: {f(E2['hop_level']['memoryless_ece'])} on the stream, {f(E2['hop_level']['memoryless_ece_shuffled'])} on its random shuffle (Lemma 1(iv): identical by construction).\n",
f"| Model | φ̂ = Var(realised N_w)/Var(implied N_w) | PIT L1 from uniform | KS p-value (PIT ~ U[0,1]) |", "|---|---|---|---|",
f"| Memoryless (implied = independent Poisson-binomial) | {f(E2['memoryless']['phi_hat'])} | {f(E2['memoryless']['pit_l1'])} | {E2['memoryless']['ks_pvalue']:.2e} |",
f"| BSF-S1 (implied = FFBS joint) | {f(E2['bsf_s1']['phi_hat'])} | {f(E2['bsf_s1']['pit_l1'])} | {E2['bsf_s1']['ks_pvalue']:.2e} |",
f"\nTCE verdict: a model passes if KS does not reject uniformity of the PIT at α = 0.01 and φ̂ ∈ [0.8, 1.25].\n",
f"Realised variance of the memoryless head's N_w against the *binomial* baseline w·ē(1−ē) that hop-level calibration implies: {f(E2['memoryless']['phi_hat_vs_binomial'])}. "
f"Theory for the memoryless head from its measured regime error rates e = {tuple(round(x,3) for x in E2['regime_error_rates_memoryless'])}: φ_w (eq. 3.5, w={E2['window']}) = {f(E2['phi_theory_window_w'])}, φ_∞ (eq. 3.6) = {f(E2['phi_theory_asymptotic'])}. "
f"Direct test of eq. (3.4), Cov(E_t,E_t+k) = λ^k·Var_π(e) for k = 1..20: relative L2 error {f(E2['covariance_test']['relative_l2_error'])}.\n",
"## E3 — Cliff cascade (Lemma 2)\n",
"| ε | crisp FM/ε | fuzzy FM | fuzzy AMS/ε |", "|---|---|---|---|",
*[f"| {r['eps']} | {f(r['crisp_FM_over_eps'])} | {r['fuzzy_FM']} | {f(r['fuzzy_AMS_over_eps'])} |" for r in E3['single_hop']],
f"\nTheory: crisp FM/ε → 2·p(0) = {f(E3['theory']['crisp_FM_over_eps_limit'])}; fuzzy AMS/ε ≤ L = {E3['theory']['fuzzy_AMS_over_eps_bound']}.\n",
"| ε | crisp P(hop 2 flips \\| hop 1 flipped) | fuzzy sup change at hop 2 |", "|---|---|---|",
*[f"| {r['eps']} | {f(r['crisp_P_hop2_flips_given_hop1_flip'])} | {f(r['fuzzy_sup_change_hop2'])} |" for r in E3['sequential']],
f"\nCommon-mode coupling, H = {E3['H']}, log–log slope of P(all H hops flip) vs ε: {E3['loglog_slope_coinciding']:.2f} when the hops' crossing points coincide (theory 1); {E3['loglog_slope_spread']:.2f} when crossing points are spread by a continuous density (theory H); {E3['loglog_slope_independent']:.2f} under independent perturbations (theory H).\n",
"## E4 — Composition algebra\n",
"| Algebra | idempotence gap E\\|T(a,a,b)−T(a,b)\\| | answer-flip rate |", "|---|---|---|",
*[f"| {k} | {f(E4[k]['idempotence_gap'])} | {f(E4[k]['answer_flip_rate'])} |" for k in ('min','product','naive_bayes')],
"\n## E5 — Type-2 separation\n",
"| Population | point-head output | point CRPS (=MAE) | Q fit (α, β) | Q CRPS |", "|---|---|---|---|---|",
*[f"| {k} | {f(E5[k]['point_head_output'])} | {f(E5[k]['point_head_CRPS(=MAE)'])} | ({E5[k]['Q_head_fit_(alpha,beta)'][0]:.1f}, {E5[k]['Q_head_fit_(alpha,beta)'][1]:.1f}) | {f(E5[k]['Q_head_CRPS'])} |" for k in ('sharp','diffuse')],
f"\nThe point head emits the same object for both populations (identical: {E5['separation']['point_head_outputs_identical']}); the Q head's concentration differs by a factor of {E5['separation']['Q_head_concentration_ratio']:.0f}. On the diffuse population Q strictly beats the point head (1/6 < 1/4).\n",
]
open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md))
print("wrote results/results.json and results/RESULTS.md")
