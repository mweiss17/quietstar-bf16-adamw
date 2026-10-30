"""(3) Fraction of parameters whose Adam update / weight decay is dropped in bf16.

Threshold: a bf16 value w (7 explicit mantissa bits) changes under round-to-nearest only if
|delta| > half-ulp(w) = 2^(floor(log2|w|) - 8). Relative half-ulp is in [2^-9, 2^-8).
Adam update with |m/sqrt(v)| ~= 1 is |delta| = lr. Weight decay delta = |w| * lr * wd.
"""
import os, sys, math
import torch

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_home"))

def half_ulp(w: torch.Tensor) -> torch.Tensor:
    """exact half-ulp of each |w| in bf16 (w assumed bf16-representable), 0 handled as 0."""
    aw = w.abs().double()
    e = torch.floor(torch.log2(aw.clamp_min(1e-300)))
    return torch.where(aw > 0, torch.pow(2.0, e - 8), torch.zeros_like(aw))

def ulp_check():
    # sanity: empirical half-ulp via nextafter in bf16
    x = torch.tensor([0.02, 1.0, 0.001, 3e-5], dtype=torch.bfloat16)
    nxt = torch.nextafter(x.float(), torch.full_like(x.float(), 1e9)).bfloat16()  # not reliable; do direct
    for v in x.tolist():
        b = torch.tensor(v, dtype=torch.bfloat16)
        up = torch.tensor(v, dtype=torch.bfloat16).view(torch.int16) + 1
        up = up.view(torch.bfloat16)
        ulp = (up.double() - b.double()).item()
        print(f"  w={b.item():.6g}  ulp={ulp:.3e}  half-ulp={ulp/2:.3e}  computed={half_ulp(b).item():.3e}  rel={ulp/2/abs(b.item()):.3e}")

def load_weights():
    try:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
        groups = {}
        for n, p in m.named_parameters():
            if "lm_head" in n:  # tied to wte
                continue
            kind = ("ln" if ".ln_" in n or n.endswith("ln_f.weight") or n.endswith("ln_f.bias") else
                    "bias" if n.endswith(".bias") else
                    "embed" if n.startswith("transformer.wte") or n.startswith("transformer.wpe") else
                    "matrix")
            groups.setdefault(kind, []).append(p.detach().flatten())
        groups = {k: torch.cat(v) for k, v in groups.items()}
        allw = torch.cat(list(groups.values()))
        return "gpt2 (124M)", allw, groups
    except Exception as e:
        print("GPT-2 unavailable:", type(e).__name__, str(e)[:120], "-> N(0,0.02)")
        allw = torch.randn(10_000_000) * 0.02
        return "N(0,0.02)", allw, {"all": allw}

def report(name, w, lrs=(1e-5, 2e-5, 1e-4), wds=(0.01, 0.1)):
    wb = w.bfloat16()  # params are bf16 at step time
    hu = half_ulp(wb)
    aw = wb.abs().double()
    n = w.numel()
    q = torch.quantile(aw[torch.randperm(n)[:2_000_000]] if n > 2_000_000 else aw, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    print(f"\n[{name}] n={n:,}  |w| median={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f}  zeros={int((aw==0).sum())}")
    print("  Adam update |delta|=lr (assumes |m/sqrt v|=1):")
    for lr in lrs:
        dropped = (lr <= hu) & (aw > 0)
        print(f"    lr={lr:g}: dropped fraction = {dropped.double().mean().item():.4f}   (|w| > {lr*2**8:.4g} rounds away; |w| >= {lr*2**9:.4g} always)")
    print("  Weight decay |delta| = |w|*lr*wd  (dropped iff lr*wd <= rel-half-ulp(w) in [2^-9, 2^-8) = [1.95e-3, 3.9e-3)):")
    for lr in lrs:
        for wd in wds:
            rel = lr * wd
            dropped = (aw * rel <= hu) & (aw > 0)
            print(f"    lr={lr:g} wd={wd:g}: lr*wd={rel:.1e}  dropped fraction = {dropped.double().mean().item():.4f}")
    # Also: at what lr*wd does decay start to register at all?
    print("  threshold: lr*wd must exceed 2^-9 = %.3e for ANY bf16 weight to decay; > 2^-8 = %.3e for all" % (2**-9, 2**-8))

if __name__ == "__main__":
    torch.manual_seed(0)
    print(torch.__version__)
    print("half-ulp sanity:"); ulp_check()
    name, allw, groups = load_weights()
    report(name, allw)
    if len(groups) > 1:
        for k, v in groups.items():
            report(f"{name} / {k}", v, wds=(0.1,))
