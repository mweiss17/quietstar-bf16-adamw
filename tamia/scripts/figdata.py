# CONTROLS / figures: export CSVs to results/figdata/ (re-runnable; overwrites only results/figdata/*).
#  (a) per_step_stats.csv   one row per (run, step, class) for every training run with a stats.json
#  (b) csqa_items.csv       per-item CSQA soft prob (product over letter + "\n") and hard-correct flag, per model;
#      csqa_first_pos_top20.csv  restricted {A..E,\n} distribution at the first answer position (only the first 20 items are stored)
#  (c) armA_cum_scatter.csv Arm A coordinate sample: |w0|, cumulative fp32-shadow delta, bf16-applied flag at steps 50/100
#                            (read from the checkpoints; Arm A's vsample_step*.pt only hold exp_avg_sq pairs)
#      ctrl_wsample.csv     per-step (|w|, |proposed fp32 update|, |applied|, bf16-visible, in-mask) samples of the controls
import csv, glob, json, os, sys
import torch

Q = "/scratch/m/mweiss10/qstar"
OUT = f"{Q}/results/figdata"
os.makedirs(OUT, exist_ok=True)
only = set(sys.argv[1:])  # optional: subset of {a,b,c_armA,c_ctrl}


def want(x):
    return not only or x in only


RUNS = [("A_bf16_lr1e-6", f"{Q}/ckpt/full/stats/stats.json"), ("B_fp32_lr1e-6", f"{Q}/ckpt/fullB/stats/stats.json"),
        ("B_fp32_lr3e-7", f"{Q}/ckpt/fullB_lr3e-7/stats/stats.json"), ("B_fp32_lr1e-7", f"{Q}/ckpt/fullB_lr1e-7/stats/stats.json"),
        ("B_fp32_lr3e-8", f"{Q}/ckpt/fullB_lr3e-8/stats/stats.json")] + \
       [(f"ctrl_{a}", f"{Q}/ckpt/ctrl_{a}/stats/stats.json") for a in ("M1", "M2", "R", "N", "T")]

COLS = ["run", "step", "lr", "grad_norm_postclip", "class", "numel", "applied_frac", "bf16_visible_frac", "dropped_frac",
        "proposed_nonzero_frac", "in_mask_frac", "update_norm", "proposed_norm", "cum_norm", "cum_norm_fp32_shadow",
        "cum_bf16_changed_frac", "weight_norm", "loss_0", "loss_final", "policy_reward"]


def row_armA(run, r, c, v):
    return {"applied_frac": v.get("frac_bf16_changed"), "bf16_visible_frac": v.get("frac_bf16_changed"),
            "dropped_frac": v.get("frac_dropped"), "proposed_nonzero_frac": v.get("frac_update_nonzero"), "in_mask_frac": None,
            "update_norm": v.get("norm_delta_bf16"), "proposed_norm": v.get("norm_delta_fp32"), "cum_norm": v.get("norm_cum_delta_bf16"),
            "cum_norm_fp32_shadow": v.get("norm_cum_delta_fp32"), "cum_bf16_changed_frac": v.get("frac_bf16_changed_cum"),
            "weight_norm": v.get("norm_weights_bf16")}


def row_B(run, r, c, v, is_bf16_path):
    return {"applied_frac": v.get("frac_changed"), "bf16_visible_frac": v.get("frac_bf16_cast_changed"),
            "dropped_frac": None if is_bf16_path else v.get("frac_dropped"), "proposed_nonzero_frac": v.get("frac_proposed_nonzero"),
            "in_mask_frac": v.get("frac_in_mask"), "update_norm": v.get("norm_update"),
            "proposed_norm": v.get("norm_proposed", v.get("norm_update")), "cum_norm": v.get("norm_cum_delta"), "cum_norm_fp32_shadow": None,
            "cum_bf16_changed_frac": v.get("frac_bf16_cast_changed_cum"), "weight_norm": v.get("norm_weights")}


if want("a"):
    n = 0
    with open(f"{OUT}/per_step_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader()
        for run, p in RUNS:
            if not os.path.exists(p):
                print("missing", run, p); continue
            d = json.load(open(p))
            is_A = run.startswith("A_")
            is_bf16_path = is_A or d.get("arm") == "T" or (d.get("records") and d["records"][0].get("arm") == "T")
            for r in d["records"]:
                ml = r.get("model_log") or {}
                for c, v in r["classes"].items():
                    row = {"run": run, "step": r["step"], "lr": r["lr"], "grad_norm_postclip": r.get("grad_norm_postclip"), "class": c, "numel": v.get("numel")}
                    row.update(row_armA(run, r, c, v) if is_A else row_B(run, r, c, v, is_bf16_path))
                    if c == "ALL":
                        row.update({k: ml.get(k) for k in ("loss_0", "loss_final", "policy_reward")})
                    w.writerow(row); n += 1
    print("per_step_stats.csv rows", n)

if want("b"):
    MODELS = [("base", "eval2_base.json"), ("A_bf16_step50", "eval2_bf16_step50.json"), ("A_bf16_step100", "eval2_bf16_step100.json"),
              ("A_shadow_bf16_step50", "eval2_shadow_bf16_step50.json"), ("A_shadow_bf16_step100", "eval2_shadow_bf16_step100.json"),
              ("B_fp32_lr1e-6_step50", "evalB_harness_step50.json"), ("B_fp32_lr1e-6_step100", "evalB_harness_step100.json")] + \
             [(f"B_fp32_lr{lr}_step{s}", f"evalB_lr{lr}_harness_step{s}.json") for lr in ("3e-7", "1e-7", "3e-8") for s in (50, 100)] + \
             [(f"ctrl_{a}_step{s}{w0}", f"evalD_{a}_harness_step{s}{w0}.json") for a in ("M1", "M2", "R", "N", "T") for s in (50, 100) for w0 in ("", "_w0")]
    n1 = n2 = 0
    with open(f"{OUT}/csqa_items.csv", "w", newline="") as f1, open(f"{OUT}/csqa_first_pos_top20.csv", "w", newline="") as f2:
        w1 = csv.writer(f1); w1.writerow(["model", "item", "soft_prob", "hard_correct"])
        w2 = csv.writer(f2); w2.writerow(["model", "item", "pos", "true", "p_A", "p_B", "p_C", "p_D", "p_E", "p_newline", "fullvocab_top1", "fullvocab_top1_p"])
        for label, fn in MODELS:
            p = f"{Q}/results/{fn}"
            if not os.path.exists(p):
                continue
            c = json.load(open(p))["csqa"]
            for i, (s, h) in enumerate(zip(c["per_item"], c["per_item_hard"])):
                w1.writerow([label, i, s, int(bool(h))]); n1 += 1
            for it in c.get("top5_first_items", []):
                rr = it["restricted"]
                w2.writerow([label, it["item"], it["pos"], it["true_str"]] + [rr.get(k) for k in ("A", "B", "C", "D", "E", "\n")] + [it["top"][0][1], it["top"][0][2]]); n2 += 1
    print("csqa_items rows", n1, "top20 rows", n2)

if want("c_armA"):
    from safetensors import safe_open

    def opener(d):
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
        else:
            fs = glob.glob(os.path.join(d, "*.safetensors")); assert len(fs) == 1, d
            with safe_open(fs[0], "pt") as h:
                wm = {k: os.path.basename(fs[0]) for k in h.keys()}
        handles = {}

        def get(name, r):
            fn = wm[name]
            if fn not in handles:
                handles[fn] = safe_open(os.path.join(d, fn), "pt")
            return handles[fn].get_slice(name)[r:r + 1][0]
        return get, wm

    base_dir = glob.glob(f"{Q}/hf_home/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36b*")[0]
    gb, wmb = opener(base_dir)
    ck = {50: (opener(f"{Q}/ckpt/full/step50/bf16_step50")[0], opener(f"{Q}/ckpt/full/step50/shadow_fp32_step50")[0]),
          100: (opener(f"{Q}/ckpt/full/bf16_step100")[0], opener(f"{Q}/ckpt/full/shadow_fp32_step100")[0])}

    def cls_of(nm):
        if "embed_tokens" in nm: return "embed"
        if "lm_head" in nm: return "lm_head"
        if "norm" in nm: return "rmsnorm"
        if ".mlp." in nm: return "mlp"
        if ".self_attn." in nm: return "attn"
        return "other"
    names = sorted(k for k in wmb if k.endswith(".weight"))
    shapes = {}
    for nm in names:
        with safe_open(os.path.join(base_dir, wmb[nm]), "pt") as h:
            shapes[nm] = h.get_slice(nm).get_shape()
    names = [nm for nm in names if len(shapes[nm]) == 2]  # 2-D matrices only (rmsnorm gains skipped: 1-D, 0.004 % of coords)

    def real_rows(nm):
        return min(shapes[nm][0], 32000) if cls_of(nm) in ("embed", "lm_head") else shapes[nm][0]  # no Hub w0 for the 2 added rows
    tot = sum(real_rows(nm) * shapes[nm][1] for nm in names)
    g = torch.Generator().manual_seed(2468)
    ROWS_TOTAL, COLS_PER_ROW = 300, 400
    n = 0
    with open(f"{OUT}/armA_cum_scatter.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tensor", "class", "row", "col", "step", "w0", "w_bf16", "w_shadow_fp32", "abs_w0", "abs_cum_delta_fp32", "abs_cum_delta_bf16", "applied_bf16"])
        for nm in names:
            nr, ncols = real_rows(nm), shapes[nm][1]
            k = max(1, round(ROWS_TOTAL * nr * ncols / tot))
            for r in torch.randint(0, nr, (k,), generator=g).tolist():
                cols = torch.randint(0, ncols, (COLS_PER_ROW,), generator=g)
                w0 = gb(nm, r).float()[cols]
                for step, (g16, g32) in ck.items():
                    a16, a32 = g16(nm, r).float()[cols], g32(nm, r).float()[cols]
                    for j in range(len(cols)):
                        x0, x16, x32 = float(w0[j]), float(a16[j]), float(a32[j])
                        w.writerow([nm, cls_of(nm), r, int(cols[j]), step, f"{x0:.6e}", f"{x16:.6e}", f"{x32:.6e}",
                                    f"{abs(x0):.6e}", f"{abs(x32 - x0):.6e}", f"{abs(x16 - x0):.6e}", int(x16 != x0)])
                        n += 1
    print("armA_cum_scatter rows", n)

if want("c_ctrl"):
    PER_STEP = 40_000
    n = 0
    with open(f"{OUT}/ctrl_wsample.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "step", "lr", "class", "abs_w", "abs_d_proposed", "abs_d_applied", "bf16_visible", "in_mask"])
        for a in ("M1", "M2", "R", "N", "T"):
            files = sorted(glob.glob(f"{Q}/ckpt/ctrl_{a}/stats/wsample_step*.pt"), key=lambda x: int(x.split("step")[-1][:-3]))
            sel = None
            for fp in files:
                d = torch.load(fp)
                m = len(d["absw"])
                if sel is None:
                    sel = torch.randperm(m, generator=torch.Generator().manual_seed(97))[:PER_STEP].sort().values
                cls = [d["tensor_classes"][int(t)] for t in d["tensor"][sel]]
                for j, i in enumerate(sel.tolist()):
                    w.writerow([f"ctrl_{a}", d["step"], d["lr"], cls[j], f"{float(d['absw'][i]):.6e}", f"{float(d['absd_prop'][i]):.6e}",
                                f"{float(d['absd_applied'][i]):.6e}", int(d["bf16_visible"][i]), int(d["in_mask"][i])])
                    n += 1
    print("ctrl_wsample rows", n)
for fn in sorted(os.listdir(OUT)):
    print(fn, os.path.getsize(os.path.join(OUT, fn)) // 1024, "KiB")
