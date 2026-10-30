# CONTROLS 2: results table from the JSONs only (evalE_* for the new runs; evalD_* / eval2_* / evalC for the seed-42 rows),
# per-step stats CSV results/figdata/per_step_stats_ctrl2.csv (same columns as per_step_stats.csv), seed statistics,
# and a JSON summary results/ctrl2_summary.json. Reads only, except those two output files.
import csv, json, math, os, re, statistics as S

Q = "/scratch/m/mweiss10/qstar"
R = f"{Q}/results"
L = f"{Q}/logs"
STEPS_SHOW = (2, 10, 25, 50, 75, 100)


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def first_sum(train_log):
    if not os.path.exists(train_log):
        return None
    for line in open(train_log, errors="replace"):
        if "PILOT first micro-batch" in line:
            m = re.search(r"sum=(\d+)", line)
            return int(m.group(1)) if m else None
    return None


def harness(d):
    if not d:
        return {}
    out = {}
    for t in ("gsm8k", "csqa"):
        if t in d:
            out[t] = {"soft": d[t]["accuracy"], "hard": d[t]["hard_accuracy"], "n": d[t].get("n"),
                      "newline_wins": d[t].get("diag", {}).get("first_pos_newline_wins_frac")}
    return out


def ppl_of(dj, label, mode):
    if not dj:
        return None
    return dj["models"].get(label, {}).get(mode, {}).get("ppl")


def stats_records(p):
    d = j(p)
    return d["records"] if d else []


def changed(recs, armA_shadow_format=False):
    """per-step applied fraction (ALL) and cumulative ever-changed bf16 fraction at each step."""
    out = {}
    for r in recs:
        v = r["classes"]["ALL"]
        if armA_shadow_format:
            out[r["step"]] = (v.get("frac_bf16_changed"), v.get("frac_bf16_changed_cum"), v.get("frac_bf16_changed"))
        else:
            out[r["step"]] = (v.get("frac_changed"), v.get("frac_bf16_cast_changed_cum"), v.get("frac_bf16_cast_changed"))
    return out


runs = []  # dicts: name, arm, seed, harness{50,100}, ppl_off{25..100}, ppl_on100, chg, first_sum, density, source
# --- existing seed-42 rows
A42 = {"name": "A bf16 (648895)", "arm": "A", "seed": 42,
       "h": {50: harness(j(f"{R}/eval2_bf16_step50.json")), 100: harness(j(f"{R}/eval2_bf16_step100.json"))},
       "ppl": {s: ppl_of(j(f"{R}/evalC_ppl_restricted.json"), f"A_bf16_step{s}", "off") for s in (25, 50, 75, 100)},
       "ppl_on": ppl_of(j(f"{R}/evalC_ppl_restricted.json"), "A_bf16_step100", "on"),
       "chg": changed(stats_records(f"{Q}/ckpt/full/stats/stats.json"), True), "first_sum": first_sum(f"{Q}/ckpt/full/train.log"),
       "density": "bf16 (cum 5.65 %)"}
runs.append(A42)
for a in ("M1", "M2", "R", "N", "T"):
    mk = j(f"{Q}/ckpt/ctrl_{a}/stats/masks.json")
    runs.append({"name": f"{a} (evalD)", "arm": a, "seed": 42,
                 "h": {s: harness(j(f"{R}/evalD_{a}_harness_step{s}.json")) for s in (50, 100)},
                 "ppl": {s: ppl_of(j(f"{R}/evalD_{a}_ppl_off.json"), f"{a}_step{s}", "off") for s in (25, 50, 75, 100)},
                 "ppl_on": ppl_of(j(f"{R}/evalD_{a}_ppl_on_step100.json"), f"{a}_step100", "on"),
                 "chg": changed(stats_records(f"{Q}/ckpt/ctrl_{a}/stats/stats.json")), "first_sum": first_sum(f"{Q}/ckpt/ctrl_{a}/train.log"),
                 "density": f"{100 * mk['density_over_masked_tensors']:.4f} % masked" if mk else "-"})
# --- new runs
for a, seeds in (("A", (1, 2)), ("M5", (42, 1, 2)), ("R5", (42, 1, 2))):
    for s in seeds:
        tag = f"{a}_s{s}"
        mk = j(f"{Q}/ckpt/ctrl2_{tag}/stats/masks.json")
        runs.append({"name": tag, "arm": a, "seed": s,
                     "h": {st: harness(j(f"{R}/evalE_{tag}_harness_step{st}.json")) for st in (50, 100)},
                     "ppl": {st: ppl_of(j(f"{R}/evalE_{tag}_ppl_off.json"), f"{tag}_step{st}", "off") for st in (25, 50, 75, 100)},
                     "ppl_on": ppl_of(j(f"{R}/evalE_{tag}_ppl_on_step100.json"), f"{tag}_step100", "on"),
                     "chg": changed(stats_records(f"{Q}/ckpt/ctrl2_{tag}/stats/stats.json")),
                     "first_sum": first_sum(f"{Q}/ckpt/ctrl2_{tag}/train.log"),
                     "density": (f"{100 * mk['density_over_masked_tensors']:.4f} % masked, k={mk['k_total']:,}, mask_seed={mk.get('mask_seed')}" if mk else "bf16")})


def P(x, k=2):
    return "-" if x is None else f"{100 * x:.{k}f}"


def F(x, k=3):
    if x is None:
        return "-"
    return f"{x:.{k}f}" if x < 1e4 else f"{x:.3g}"


print("| run | seed | first-batch sum | GSM8K soft/hard @50 | GSM8K @100 | CSQA soft/hard @50 | CSQA @100 | newline wins @100 G/C | ppl OFF 25 / 50 / 75 / 100 | ppl ON @100 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in runs:
    h = r["h"]

    def hv(st, t):
        x = h[st].get(t)
        return f"{P(x['soft'])}/{P(x['hard'])}" if x else "-"
    nl = h[100].get("gsm8k", {}).get("newline_wins"), h[100].get("csqa", {}).get("newline_wins")
    print(f"| {r['name']} | {r['seed']} | {r['first_sum']} | {hv(50,'gsm8k')} | {hv(100,'gsm8k')} | {hv(50,'csqa')} | {hv(100,'csqa')} | "
          f"{F(nl[0]) if nl[0] is not None else '-'} / {F(nl[1]) if nl[1] is not None else '-'} | {' / '.join(F(r['ppl'][s]) for s in (25, 50, 75, 100))} | {F(r['ppl_on'])} |")
print()
print("| run | applied % per step @" + " / ".join(map(str, STEPS_SHOW)) + " | mean over steps 1-100 | cum ever-changed (bf16) % @25 / 50 / 75 / 100 | bf16-visible % per step @" + " / ".join(map(str, STEPS_SHOW)) + " | density |")
print("|---|---|---|---|---|---|")
for r in runs:
    c = r["chg"]
    ap = [c[s][0] for s in sorted(c) if c[s][0] is not None]
    print(f"| {r['name']} | {' / '.join(P(c.get(s, (None,))[0], 3) for s in STEPS_SHOW)} | {P(sum(ap) / len(ap), 3) if ap else '-'} | "
          f"{' / '.join(P(c.get(s, (None, None))[1], 3) for s in (25, 50, 75, 100))} | {' / '.join(P(c.get(s, (None, None, None))[2], 3) for s in STEPS_SHOW)} | {r['density']} |")

# --- seed statistics
METRICS = [("GSM8K soft @50", lambda r: r["h"][50].get("gsm8k", {}).get("soft"), 100), ("GSM8K hard @50", lambda r: r["h"][50].get("gsm8k", {}).get("hard"), 100),
           ("GSM8K soft @100", lambda r: r["h"][100].get("gsm8k", {}).get("soft"), 100), ("GSM8K hard @100", lambda r: r["h"][100].get("gsm8k", {}).get("hard"), 100),
           ("CSQA soft @50", lambda r: r["h"][50].get("csqa", {}).get("soft"), 100), ("CSQA hard @50", lambda r: r["h"][50].get("csqa", {}).get("hard"), 100),
           ("CSQA soft @100", lambda r: r["h"][100].get("csqa", {}).get("soft"), 100), ("CSQA hard @100", lambda r: r["h"][100].get("csqa", {}).get("hard"), 100),
           ("ppl OFF @25", lambda r: r["ppl"][25], 1), ("ppl OFF @50", lambda r: r["ppl"][50], 1), ("ppl OFF @75", lambda r: r["ppl"][75], 1),
           ("ppl OFF @100", lambda r: r["ppl"][100], 1), ("ppl ON @100", lambda r: r["ppl_on"], 1),
           ("cum changed % @100", lambda r: r["chg"].get(100, (None, None))[1], 100)]
groups = {"A": [r for r in runs if r["arm"] == "A"], "M5": [r for r in runs if r["arm"] == "M5"], "R5": [r for r in runs if r["arm"] == "R5"]}
summary = {"seed_stats": {}, "paired_M5_minus_R5": {}, "runs": []}
print()
print("| metric | " + " | ".join(f"{g}: mean ± sd (min–max, n)" for g in groups) + " | M5 − R5 per seed 42 / 1 / 2 |")
print("|---|---|---|---|---|")
for name, fn, sc in METRICS:
    cells = []
    for g, rs in groups.items():
        xs = [fn(r) * sc for r in rs if fn(r) is not None]
        st = {"n": len(xs), "values": xs}
        if xs:
            st.update(mean=S.mean(xs), sd=S.stdev(xs) if len(xs) > 1 else None, min=min(xs), max=max(xs))
            cells.append(f"{st['mean']:.3f} ± {st['sd']:.3f} ({st['min']:.3f}–{st['max']:.3f}, {len(xs)})" if st["sd"] is not None else f"{st['mean']:.3f} (n=1)")
        else:
            cells.append("-")
        summary["seed_stats"].setdefault(name, {})[g] = st
    pd = []
    for s in (42, 1, 2):
        m = [r for r in groups["M5"] if r["seed"] == s]; q = [r for r in groups["R5"] if r["seed"] == s]
        if m and q and fn(m[0]) is not None and fn(q[0]) is not None:
            pd.append((fn(m[0]) - fn(q[0])) * sc)
        else:
            pd.append(None)
    summary["paired_M5_minus_R5"][name] = pd
    print(f"| {name} | " + " | ".join(cells) + " | " + " / ".join("-" if x is None else f"{x:+.3f}" for x in pd) + " |")
for r in runs:
    summary["runs"].append({k: r[k] for k in ("name", "arm", "seed", "h", "ppl", "ppl_on", "first_sum", "density")} |
                           {"chg_steps": {s: r["chg"].get(s) for s in STEPS_SHOW}})
json.dump(summary, open(f"{R}/ctrl2_summary.json", "w"), indent=1, default=str)

# --- per-step stats CSV (sibling of per_step_stats.csv; same columns)
COLS = ["run", "step", "lr", "grad_norm_postclip", "class", "numel", "applied_frac", "bf16_visible_frac", "dropped_frac",
        "proposed_nonzero_frac", "in_mask_frac", "update_norm", "proposed_norm", "cum_norm", "cum_norm_fp32_shadow",
        "cum_bf16_changed_frac", "weight_norm", "loss_0", "loss_final", "policy_reward"]
os.makedirs(f"{R}/figdata", exist_ok=True)
n = 0
with open(f"{R}/figdata/per_step_stats_ctrl2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, COLS); w.writeheader()
    for a, seeds in (("A", (1, 2)), ("M5", (42, 1, 2)), ("R5", (42, 1, 2))):
        for s in seeds:
            tag = f"{a}_s{s}"
            for r in stats_records(f"{Q}/ckpt/ctrl2_{tag}/stats/stats.json"):
                ml = r.get("model_log") or {}
                for c, v in r["classes"].items():
                    row = {"run": f"ctrl2_{tag}", "step": r["step"], "lr": r["lr"], "grad_norm_postclip": r.get("grad_norm_postclip"), "class": c, "numel": v.get("numel"),
                           "applied_frac": v.get("frac_changed"), "bf16_visible_frac": v.get("frac_bf16_cast_changed"),
                           "dropped_frac": None if a == "A" else v.get("frac_dropped"), "proposed_nonzero_frac": v.get("frac_proposed_nonzero"),
                           "in_mask_frac": v.get("frac_in_mask"), "update_norm": v.get("norm_update"), "proposed_norm": v.get("norm_proposed", v.get("norm_update")),
                           "cum_norm": v.get("norm_cum_delta"), "cum_norm_fp32_shadow": None, "cum_bf16_changed_frac": v.get("frac_bf16_cast_changed_cum"),
                           "weight_norm": v.get("norm_weights")}
                    if c == "ALL":
                        row.update({k: ml.get(k) for k in ("loss_0", "loss_final", "policy_reward")})
                    w.writerow(row); n += 1
print(f"\nper_step_stats_ctrl2.csv rows {n}; summary -> {R}/ctrl2_summary.json")
