# CONTROLS: results table for the evalD control jobs (reads only; prints markdown rows + JSON summary)
import json, os
Q = "/scratch/m/mweiss10/qstar"
R = f"{Q}/results"


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, k=2, pct=True):
    return "-" if x is None else (f"{100 * x:.{k}f}" if pct else f"{x:.{k}f}")


rows, summ = [], {}
for a in ("M1", "M2", "R", "N", "T"):
    st = j(f"{Q}/ckpt/ctrl_{a}/stats/stats.json")
    mk = j(f"{Q}/ckpt/ctrl_{a}/stats/masks.json")
    recs = st["records"] if st else []
    if mk:
        dens = f"{100 * mk['density_over_masked_tensors']:.2f}% of masked matrices ({100 * mk['trainable_frac_all_params']:.2f}% of all params)"
    elif recs:
        ap = [r["classes"]["ALL"]["frac_changed"] for r in recs]
        dens = f"applied/step mean {100 * sum(ap) / len(ap):.2f}% (max {100 * max(ap):.2f}%), cum {100 * recs[-1]['classes']['ALL'].get('frac_bf16_cast_changed_cum', float('nan')):.2f}%"
    else:
        dens = "-"
    h = {s: j(f"{R}/evalD_{a}_harness_step{s}.json") for s in (50, 100)}
    hw0 = {s: j(f"{R}/evalD_{a}_harness_step{s}_w0.json") for s in (50, 100)}
    po = j(f"{R}/evalD_{a}_ppl_off.json")
    pon = j(f"{R}/evalD_{a}_ppl_on_step100.json")
    ppl = {s: (po["models"].get(f"{a}_step{s}", {}).get("off", {}).get("ppl") if po else None) for s in (25, 50, 75, 100)}
    ppl_on = pon["models"].get(f"{a}_step100", {}).get("on", {}).get("ppl") if pon else None

    def hv(s, t, k, src=h):
        d = src[s]
        return d[t][k] if d and t in d else None
    nl = None
    if h[100]:
        nl = (h[100]["gsm8k"]["diag"]["first_pos_newline_wins_frac"], h[100]["csqa"]["diag"]["first_pos_newline_wins_frac"])
    rows.append(f"| {a} | {dens} | {f(hv(50,'gsm8k','accuracy'))}/{f(hv(50,'gsm8k','hard_accuracy'))} | {f(hv(100,'gsm8k','accuracy'))}/{f(hv(100,'gsm8k','hard_accuracy'))} "
                f"| {f(hv(50,'csqa','accuracy'))}/{f(hv(50,'csqa','hard_accuracy'))} | {f(hv(100,'csqa','accuracy'))}/{f(hv(100,'csqa','hard_accuracy'))} "
                f"| {('%.3f / %.3f' % nl) if nl else '-'} | {' / '.join(f(ppl[s], 2, False) for s in (25, 50, 75, 100))} | {f(ppl_on, 2, False)} |")
    if a == "T":
        for s in (50, 100):
            if hw0[s]:
                rows.append(f"| T w0 @{s} | | | GSM8K {f(hv(s,'gsm8k','accuracy',hw0))}/{f(hv(s,'gsm8k','hard_accuracy',hw0))} | | CSQA {f(hv(s,'csqa','accuracy',hw0))}/{f(hv(s,'csqa','hard_accuracy',hw0))} | | | |")
    summ[a] = {"density": dens, "ppl_off": ppl, "ppl_on_100": ppl_on, "n_records": len(recs),
               "applied_frac_steps": {r["step"]: r["classes"]["ALL"]["frac_changed"] for r in recs if r["step"] in (1, 2, 5, 10, 20, 30, 50, 70, 100)},
               "bf16_visible_frac_steps": {r["step"]: r["classes"]["ALL"]["frac_bf16_cast_changed"] for r in recs if r["step"] in (1, 2, 5, 10, 20, 30, 50, 70, 100)},
               "cum_bf16_changed_100": recs[-1]["classes"]["ALL"].get("frac_bf16_cast_changed_cum") if recs else None,
               "cum_norm_100": recs[-1]["classes"]["ALL"].get("norm_cum_delta") if recs else None,
               "per_class_applied_100": {c: v["frac_changed"] for c, v in recs[-1]["classes"].items()} if recs else None,
               "mask_density_by_class": mk.get("density_by_class") if mk else None,
               "nano_recomputes": mk.get("recomputes") if mk else None,
               "model_log_100": recs[-1].get("model_log") if recs else None}
print("| arm | density | GSM8K soft/hard @50 | @100 | CSQA soft/hard @50 | @100 | newline wins @100 (GSM8K / CSQA) | restricted ppl OFF 25/50/75/100 | ppl ON @100 |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print(json.dumps(summ, indent=1, default=str))
