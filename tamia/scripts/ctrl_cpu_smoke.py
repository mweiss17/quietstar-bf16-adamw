# CONTROLS pre-flight (CPU, no GPU): exercise the ACTUAL ArmBStats class and classify_param from pilot_train.py (extracted
# with ast, so the training script's top-level code does not run) on a tiny model with Mistral-style parameter names,
# a CPU AdamW with HF-style wd groups, and synthetic gradients. Checks:
#   M1 (bf16vis): at every step the applied set equals the bf16-visible set of an independent, unmasked clone stepped from
#                 the same state, the applied values equal the clone's exact fp32 values, and the moments equal the clone's.
#   M2 (static_mag): per-tensor density equals |w0| < cut; exempt classes update fully; changes stay inside the mask.
#   R  (random): per-tensor counts equal M2's; changes stay inside the mask; mask differs from M2.
#   N  (nanoadam): authors' block-wise rule (block 1000 here, 29056 on H200): at steps 1, 10, 20 the mask is, in every
#                  contiguous block, exactly the ceil(block*density) smallest-|w| coordinates of the weights before that step
#                  (checked independently by sorting), density = M2's per-tensor density.
import ast, copy, json, math, os, sys, tempfile, types
from collections import defaultdict
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "pilot_train.py")).read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("ArmBStats", "classify_param")]
assert len(keep) == 2
mod = ast.Module(body=keep, type_ignores=[])
import pilot_masks


def make_ns(mode, stats_dir):
    ns = {"torch": torch, "os": os, "json": json, "math": math, "time": __import__("time"), "defaultdict": defaultdict,
          "pilot_gpu_mem": lambda reset=False: {}, "global_gradient_accumulation_steps": 2}
    ns.update({k: getattr(pilot_masks, k) for k in ("mask_applies", "bf16_visible_mask", "static_mag_mask", "bottomk_mask", "random_mask", "apply_mask_", "nanoadam_block_mask", "nano_block_counts")})
    ns["PILOT"] = types.SimpleNamespace(mask_mode=mode, arm=mode, fp32_master=True, mag_cut=5.12e-4, mask_seed=1234, mask_interval=10, nano_block=1000,
                                        sample_steps="1,2,10", n_sample=2000, micro_batch=4, steps=25, lr=1e-6, no_thoughts=False)
    exec(compile(mod, "pilot_train_extract", "exec"), ns)
    return ns


class Tiny(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, V, I = 64, 300, 128

        def P(*shape, std=0.02):  # bf16-representable fp32 init, like Mistral weights cast to fp32
            return torch.nn.Parameter((torch.randn(*shape, generator=g) * std).to(torch.bfloat16).float())
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(V, H); self.model.embed_tokens.weight = P(V, H)
        self.model.layers = torch.nn.ModuleList()
        for _ in range(2):
            L = torch.nn.Module(); L.self_attn = torch.nn.Module(); L.mlp = torch.nn.Module()
            L.self_attn.q_proj = torch.nn.Linear(H, H, bias=False); L.self_attn.q_proj.weight = P(H, H)
            L.mlp.up_proj = torch.nn.Linear(H, I, bias=False); L.mlp.up_proj.weight = P(I, H, std=0.01)
            L.input_layernorm = torch.nn.Module(); L.input_layernorm.weight = torch.nn.Parameter(torch.ones(H))
            self.model.layers.append(L)
        self.model.norm = torch.nn.Module(); self.model.norm.weight = torch.nn.Parameter(torch.ones(H))
        self.lm_head = torch.nn.Linear(H, V, bias=False); self.lm_head.weight = P(V, H)
        self.start_embedding = P(2, H); self.end_embedding = P(2, H)
        self.talk_head = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(2 * H, H), torch.nn.Linear(H, 1))])


def make_opt(model):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "norm" not in n]
    nod = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "norm" not in n)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 1e-3}, {"params": nod, "weight_decay": 0.0}], lr=1e-6, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def lr_at(t):  # warmup 5 then flat, with lr 0 at the first step (as in the HF schedule)
    return 1e-6 * min(1.0, (t - 1) / 5)


def set_grads(model, t):
    g = torch.Generator().manual_seed(1000 + t)
    for n, p in model.named_parameters():
        p.grad = torch.randn(p.shape, generator=g) * (1e-3 if "embed" not in n else 1e-4)


def run(mode, steps=25):
    d = tempfile.mkdtemp(prefix=f"ctrl_{mode}_")
    ns = make_ns(mode, d)
    model = Tiny(); opt = make_opt(model)
    st = ns["ArmBStats"](opt, list(model.named_parameters()), d, model=None)
    opt.register_step_pre_hook(st.pre_step); opt.register_step_post_hook(st.post_step)
    names = {id(p): n for n, p in model.named_parameters()}
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    ok = True
    m2_ref = None
    w_at_recompute = None
    for t in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(t)
        set_grads(model, t)
        # independent unmasked clone stepped from the same state
        cm = copy.deepcopy(model); co = make_opt(cm); co.load_state_dict(copy.deepcopy(opt.state_dict()))
        for (n, p), (n2, q) in zip(model.named_parameters(), cm.named_parameters()):
            q.grad = p.grad.clone()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        if t == 1 or t % 10 == 0:
            w_at_recompute = before
        co.step(); opt.step()
        if t == 1 and mode in ("static_mag", "random", "nanoadam"):
            m2_ref = {n: (w0[n].abs() < 5.12e-4) for n in w0}
        for (n, p), (_, q) in zip(model.named_parameters(), cm.named_parameters()):
            changed = p.detach() != before[n]
            cls = ns["classify_param"](n)
            if mode == "bf16vis":
                vis = q.detach().to(torch.bfloat16) != before[n].to(torch.bfloat16)
                exp = torch.where(vis, q.detach(), before[n])
                if not torch.equal(p.detach(), exp):
                    ok = False; print(f"  FAIL M1 step {t} {n}: masked weights != where(bf16-visible, clone, prev)")
                if not torch.equal(changed, vis & (q.detach() != before[n])):
                    ok = False; print(f"  FAIL M1 step {t} {n}: applied set != bf16-visible set")
                sa, sb = opt.state[p], co.state[q]
                if not (torch.equal(sa["exp_avg"], sb["exp_avg"]) and torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])):
                    ok = False; print(f"  FAIL M1 step {t} {n}: moments differ from the unmasked clone")
            else:
                k = st.name2k[n]
                m = st.masks[k]
                if cls in pilot_masks.EXEMPT_CLASSES or p.dim() != 2:
                    if m is not None or not torch.equal(p.detach(), q.detach()):
                        ok = False; print(f"  FAIL {mode} step {t} {n}: exempt tensor not updated fully")
                else:
                    if bool((changed & ~m).any()):
                        ok = False; print(f"  FAIL {mode} step {t} {n}: change outside the mask")
                    if not torch.equal(p.detach()[m], q.detach()[m]):
                        ok = False; print(f"  FAIL {mode} step {t} {n}: in-mask values != exact fp32 step")
                    if mode != "nanoadam" and int(m.sum()) != int(m2_ref[n].sum()):
                        ok = False; print(f"  FAIL {mode} step {t} {n}: count {int(m.sum())} != M2 count {int(m2_ref[n].sum())}")
                    if mode == "static_mag" and not torch.equal(m, m2_ref[n]):
                        ok = False; print(f"  FAIL M2 {n}: mask != |w0| < cut")
                    if mode == "nanoadam":
                        dens = int(m2_ref[n].sum()) / m.numel()
                        B = 1000; flat_w = w_at_recompute[n].flatten().abs(); flat_m = m.flatten(); nfull = flat_w.numel() // B
                        blocks = [(i * B, (i + 1) * B) for i in range(nfull)] + ([(nfull * B, flat_w.numel())] if flat_w.numel() % B else [])
                        for a, b in blocks:
                            kb = math.ceil((b - a) * dens)
                            wb, mb = flat_w[a:b], flat_m[a:b]
                            if int(mb.sum()) != kb:
                                ok = False; print(f"  FAIL N step {t} {n} block {a}: count {int(mb.sum())} != ceil({b-a}*{dens:.4f}) = {kb}")
                            elif kb and kb < b - a and float(wb[mb].max()) > float(wb[~mb].min()):
                                ok = False; print(f"  FAIL N step {t} {n} block {a}: not the bottom-k of |w| at the last recompute")
    rec = st.records[-1]["classes"]["ALL"]
    dens = {n: v["density"] for n, v in st.mask_log.get("per_tensor", {}).items()}
    print(f"{mode}: steps={steps} last applied/changed={rec['frac_changed']:.4%} proposed_nonzero={rec['frac_proposed_nonzero']:.2%} "
          f"in_mask={rec['frac_in_mask']:.4%}; per-tensor densities {json.dumps({k: round(v, 4) for k, v in dens.items()})}")
    if mode == "random":
        same = all(torch.equal(st.masks[st.name2k[n]], m2_ref[n]) for n in dens)
        print(f"  random mask identical to M2 on every tensor? {same}")
        ok &= not same
    if mode == "nanoadam":
        print("  recomputes:", [(r["before_step"], round(r["overlap_with_previous"], 4)) for r in st.mask_log["recomputes"]])
        ok &= [r["before_step"] for r in st.mask_log["recomputes"]] == [10, 20]
        # the step-21 mask must be bottom-k of |w_20|: check against the weights saved by the stats hook? recompute directly instead
    print(f"  files: {sorted(os.listdir(d))}")
    smp = torch.load(os.path.join(d, "wsample_step10.pt"))
    print(f"  wsample_step10: n={len(smp['absw'])} applied_frac={float((smp['absd_applied'] > 0).float().mean()):.4f} in_mask_frac={float(smp['in_mask'].float().mean()):.4f}")
    print(f"CTRL_SMOKE_{'OK' if ok else 'FAIL'} mode={mode}")
    return ok


if __name__ == "__main__":
    res = [run(m) for m in ("bf16vis", "static_mag", "random", "nanoadam")]
    # N recompute exactness on an isolated tensor: bottom-k == k smallest |w|
    w = torch.randn(1000) * 0.01
    m = pilot_masks.bottomk_mask(w, 37)
    assert int(m.sum()) == 37 and float(w.abs()[m].max()) <= float(w.abs()[~m].min())
    print("ALL_CTRL_SMOKE_OK" if all(res) else "SOME_CTRL_SMOKE_FAILED")
