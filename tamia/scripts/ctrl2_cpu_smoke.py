# CONTROLS 2 pre-flight (CPU, login node). Exercises the ACTUAL ArmBStats / classify_param / mask_applies of
# pilot_train_ctrl2.py (extracted with ast; the training script's top-level code does not run) on a tiny model with
# Mistral-style names, CPU AdamW with HF-style wd groups, synthetic gradients, 25 steps. Checks:
#   M5 (static_topk + counts): per-tensor mask count == k from the counts file; the mask is the k smallest |w0| of the
#       tensor (max |w0| inside <= min |w0| outside, checked by comparison); exempt classes update fully; no change outside
#       the mask; in-mask values == an unmasked clone's exact fp32 step.
#   R5 (random + counts): counts == M5's; mask differs from M5; changes only inside; two mask seeds give different masks.
#   Seed: dataset.shuffle(seed) + preprocess crops (python random seeded like the script) differ between seeds 42/1/2,
#       and seed 42 reproduces the cached (existing-run) pipeline. Datasets are kept in memory (no cache files written).
import ast, copy, json, math, os, random, sys, tempfile, types
from collections import defaultdict
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "pilot_train_ctrl2.py")).read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("ArmBStats", "classify_param", "mask_applies")]
assert len(keep) == 3, [n.name for n in keep]
mod = ast.Module(body=keep, type_ignores=[])
import pilot_masks
from ctrl_cpu_smoke import Tiny, make_opt, lr_at, set_grads  # the Controls smoke's tiny model / optimizer / grads (import-safe: main guarded)


def make_ns(mode, counts_path, mask_seed=1234):
    ns = {"torch": torch, "os": os, "json": json, "math": math, "time": __import__("time"), "defaultdict": defaultdict,
          "pilot_gpu_mem": lambda reset=False: {}, "global_gradient_accumulation_steps": 2,
          "_mask_applies_v1": pilot_masks.mask_applies, "MASK_COUNTS": json.load(open(counts_path))["per_tensor"]}
    ns.update({k: getattr(pilot_masks, k) for k in ("bf16_visible_mask", "static_mag_mask", "bottomk_mask", "random_mask", "apply_mask_", "nanoadam_block_mask", "nano_block_counts")})
    ns["PILOT"] = types.SimpleNamespace(mask_mode=mode, arm=mode, fp32_master=True, mag_cut=5.12e-4, mask_seed=mask_seed, mask_interval=10, nano_block=1000,
                                        sample_steps="1,2,10", n_sample=2000, micro_batch=4, steps=25, lr=1e-6, no_thoughts=False,
                                        mask_counts=counts_path, seed=42)
    exec(compile(mod, "pilot_train_ctrl2_extract", "exec"), ns)
    return ns


def make_counts(path):
    """Per-tensor k for the masked tensors of Tiny, with varying densities around 5.65 % (as from ctrl2_counts.py)."""
    m = Tiny(); per = {}
    dens = [0.0565, 0.031, 0.058, 0.057, 0.05, 0.0, 0.2]
    i = 0
    for n, p in m.named_parameters():
        cls = n
        if p.dim() == 2 and not any(x in n for x in ("talk_head", "start_embedding", "end_embedding", "norm")):
            d = dens[i % len(dens)]; i += 1
            per[n] = {"numel": p.numel(), "k": int(round(d * p.numel())), "density": d}
    json.dump({"per_tensor": per}, open(path, "w"))
    return per


def run(mode, counts_path, per, mask_seed=1234, steps=25, ref_masks=None):
    d = tempfile.mkdtemp(prefix=f"ctrl2_{mode}_")
    ns = make_ns(mode, counts_path, mask_seed)
    model = Tiny(); opt = make_opt(model)
    st = ns["ArmBStats"](opt, list(model.named_parameters()), d, model=None)
    opt.register_step_pre_hook(st.pre_step); opt.register_step_post_hook(st.post_step)
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    ok = True
    for t in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(t)
        set_grads(model, t)
        cm = copy.deepcopy(model); co = make_opt(cm); co.load_state_dict(copy.deepcopy(opt.state_dict()))
        for (n, p), (_, q) in zip(model.named_parameters(), cm.named_parameters()):
            q.grad = p.grad.clone()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        co.step(); opt.step()
        for (n, p), (_, q) in zip(model.named_parameters(), cm.named_parameters()):
            changed = p.detach() != before[n]
            m = st.masks[st.name2k[n]]
            if n not in per:
                if m is not None or not torch.equal(p.detach(), q.detach()):
                    ok = False; print(f"  FAIL {mode} step {t} {n}: unmasked tensor not updated fully (mask={m is not None})")
                continue
            if int(m.sum()) != per[n]["k"]:
                ok = False; print(f"  FAIL {mode} step {t} {n}: count {int(m.sum())} != k {per[n]['k']}")
            if bool((changed & ~m).any()):
                ok = False; print(f"  FAIL {mode} step {t} {n}: change outside the mask")
            if not torch.equal(p.detach()[m], q.detach()[m]):
                ok = False; print(f"  FAIL {mode} step {t} {n}: in-mask values != exact fp32 step")
            if mode == "static_topk" and 0 < per[n]["k"] < m.numel():
                a = w0[n].abs()
                if float(a[m].max()) > float(a[~m].min()):
                    ok = False; print(f"  FAIL M5 {n}: mask is not the bottom-k of |w0|")
            if t == steps and per[n]["k"] > 0 and not bool(changed.any()) and not bool((p.detach() != w0[n])[m].any()):
                ok = False; print(f"  FAIL {mode} {n}: nothing inside the mask ever moved")
    rec = st.records[-1]["classes"]["ALL"]
    log = st.mask_log
    print(f"{mode} (mask_seed {mask_seed}): density over masked {log['density_over_masked_tensors']:.4%} k={log['k_total']} "
          f"applied last step={rec['frac_changed']:.4%} in_mask={rec['frac_in_mask']:.4%}; mask_log mode={log['mode']} counts={os.path.basename(log['mask_counts'])} mag_cut={log['mag_cut']}")
    masks = {n: st.masks[st.name2k[n]].clone() for n in per}
    if ref_masks is not None:
        same = all(torch.equal(masks[n], ref_masks[n]) for n in per if per[n]["k"] > 0)
        print(f"  identical to reference masks on every tensor? {same}")
        ok &= not same
    print(f"CTRL2_SMOKE_{'OK' if ok else 'FAIL'} mode={mode} mask_seed={mask_seed}")
    return ok, masks


def seed_check():
    """The data side of --seed, reproduced exactly as pilot_train_ctrl2 does it (random.seed(S) then shuffle(seed=S).map)."""
    from datasets import load_from_disk
    from eval_helpers import preprocess_function
    Q = os.environ.get("QSTAR_ROOT", "/scratch/m/mweiss10/qstar")
    ds = load_from_disk(f"{Q}/data/owm_first1000")
    out = {}
    for S in (42, 1, 2, 42):
        random.seed(S); torch.manual_seed(S)
        tr = ds.shuffle(seed=S, keep_in_memory=True).map(preprocess_function, batched=True, keep_in_memory=True, load_from_cache_file=False)
        ids = torch.tensor(tr["input_ids"][:64])
        key = (len(tr), int(ids.sum()), tr["input_ids"][0][:6])
        print(f"  seed {S}: n_rows={len(tr)} sum(first 64 rows)={key[1]} row0[:6]={key[2]}")
        if S in out:
            assert out[S] == key, "seed 42 not reproducible"
        out[S] = key
    # what the cached pipeline (as used by every existing seed-42 run) produced
    random.seed(42)
    cached = ds.shuffle(seed=42).map(preprocess_function, batched=True)
    ck = (len(cached), int(torch.tensor(cached["input_ids"][:64]).sum()))
    print(f"  cached seed-42 pipeline (existing runs): n_rows={ck[0]} sum={ck[1]} -> {'matches' if ck == out[42][:2] else 'DIFFERS from'} fresh seed 42")
    ok = len({out[42][1], out[1][1], out[2][1]}) == 3
    # sampler side: Trainer.set_seed(args.seed) then RandomSampler (no generator) draws its permutation seed from the
    # CPU default generator, so the order is a function of args.seed (+ RNG consumed during model_init). Shown for no consumption:
    from transformers import set_seed
    perms = {}
    for S in (42, 1, 2):
        set_seed(S)
        perms[S] = list(torch.utils.data.RandomSampler(range(out[42][0])))[:8]
        print(f"  RandomSampler after set_seed({S}) (no model_init RNG use): first indices {perms[S]}")
    ok &= len({tuple(v) for v in perms.values()}) == 3
    print(f"SEED_CHECK_{'OK' if ok else 'FAIL'} (cached-vs-fresh seed 42 equal: {ck == out[42][:2]})")
    return ok


if __name__ == "__main__":
    cp = os.path.join(tempfile.mkdtemp(prefix="ctrl2_counts_"), "counts.json")
    per = make_counts(cp)
    ok1, m5 = run("static_topk", cp, per)
    ok2, r5a = run("random", cp, per, mask_seed=1234, ref_masks=m5)
    ok3, r5b = run("random", cp, per, mask_seed=2234, ref_masks=r5a)
    ok4 = seed_check() if "--no_seed" not in sys.argv else True
    print("ALL_CTRL2_SMOKE_OK" if all((ok1, ok2, ok3, ok4)) else "SOME_CTRL2_SMOKE_FAILED")
