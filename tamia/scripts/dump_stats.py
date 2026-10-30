import json, sys
p = sys.argv[1]
d = json.load(open(p))
print("micro_batch", d["micro_batch"], "grad_accum", d["grad_accum"])
for r in d["records"]:
    print("step %d lr=%.3e gnorm_postclip=%.4f state_dtype=%s t_shadow=%.1fs t_stats=%.1fs trainer_log=%s" % (
        r["step"], r["lr"], r["grad_norm_postclip"], r["state_dtype_real"], r["t_shadow_step_s"], r["t_stats_s"], r.get("trainer_log")))
    for c, v in r["classes"].items():
        vs = v.get("exp_avg_sq_sample", {})
        print("   %-12s changed=%.5f%% dropped=%.5f%% nz_upd=%.4f%% drop|nz=%s d16=%.3e d32=%.3e cum_changed=%.5f%% cum16=%.3e cum32=%.3e |w|=%.3e v_n=%s v_med=%s v_p10=%s v_p90=%s v_mlog10=%s real0_shadow+=%s shadow0=%s" % (
            c, 100*v["frac_bf16_changed"], 100*v["frac_dropped"], 100*v["frac_update_nonzero"], v["frac_dropped_given_nonzero_update"],
            v["norm_delta_bf16"], v["norm_delta_fp32"], 100*v["frac_bf16_changed_cum"], v["norm_cum_delta_bf16"], v["norm_cum_delta_fp32"], v["norm_weights_bf16"],
            vs.get("n"), vs.get("ratio_median"), vs.get("ratio_p10"), vs.get("ratio_p90"), vs.get("mean_log10_ratio"), vs.get("frac_real_zero_shadow_pos"), vs.get("frac_shadow_zero")))
