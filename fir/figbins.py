"""Figure data for the article's Figure 2 (|w| histogram) and Figure 4 (changed fraction vs |w|).

Bins are on a log2 scale with 16 bins per octave, so every power of two is a bin edge. That matters because
the real bf16 cliff sits on a power-of-two boundary: at lr 1e-6 a weight in [2^-12, 2^-11) has half-ulp
9.5e-7 (movable by a 1e-6 step), while one in [2^-11, 2^-10) has half-ulp 1.9e-6 (frozen). 512*lr = 5.12e-4
is just above 2^-11 = 4.88e-4.

Output: one long CSV, columns comparator,cls,bin,lo,hi,n,changed. n is the count of base weights in the bin
(identical across comparators); changed counts coordinates whose bf16 value differs from base.
Comparators are optional; any whose path is missing is skipped.

Usage: python figbins.py BASE_DIR OUT_CSV name=DIR [name=DIR ...]
"""
import os, sys, csv, math, torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
PER_OCT, LO_EXP, HI_EXP = 16, -40, 4          # bins cover [2^-40, 2^4); bin 0 also catches exact zeros
NB = (HI_EXP - LO_EXP) * PER_OCT


def index(d):
    idx = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt") as sf:
                for k in sf.keys():
                    idx[k] = f
    return idx


def get(d, idx, k):
    with safe_open(os.path.join(d, idx[k]), "pt") as sf:
        return sf.get_tensor(k)


def cls(n):
    if "norm" in n: return "rmsnorm"
    if "embed_tokens" in n: return "embed"
    if "lm_head" in n: return "lm_head"
    if "self_attn" in n: return "attn"
    if "mlp" in n: return "mlp"
    return "other"


def bins_of(w0b):
    a = w0b.float().abs().clamp_min(2.0 ** LO_EXP)
    b = torch.floor((torch.log2(a) - LO_EXP) * PER_OCT).long()
    return b.clamp_(0, NB - 1)


def main():
    base, out = sys.argv[1], sys.argv[2]
    comps = [tuple(x.split("=", 1)) for x in sys.argv[3:]]
    comps = [(n, p) for n, p in comps if os.path.isdir(p)]
    print("comparators:", comps, flush=True)
    bi = index(base)
    ci = {n: index(p) for n, p in comps}
    hist = {}      # cls -> n per bin
    chg = {}       # (comp, cls) -> changed per bin
    for k in sorted(bi):
        w0b = get(base, bi, k).to(torch.bfloat16).flatten()
        c = cls(k)
        b = bins_of(w0b)
        hist.setdefault(c, torch.zeros(NB, dtype=torch.long)).add_(torch.bincount(b, minlength=NB))
        for name, path in comps:
            if k not in ci[name]:
                continue
            w1b = get(path, ci[name], k).to(torch.bfloat16).flatten()[: w0b.numel()]
            m = w1b != w0b
            chg.setdefault((name, c), torch.zeros(NB, dtype=torch.long)).add_(torch.bincount(b[m], minlength=NB))
        print("done", k, flush=True)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["comparator", "cls", "bin", "lo", "hi", "n", "changed"])
        for c, h in hist.items():
            for i in range(NB):
                if h[i] == 0:
                    continue
                lo = 2.0 ** (LO_EXP + i / PER_OCT); hi = 2.0 ** (LO_EXP + (i + 1) / PER_OCT)
                wr.writerow(["base", c, i, f"{lo:.6e}", f"{hi:.6e}", int(h[i]), ""])
                for name, _ in comps:
                    v = chg.get((name, c))
                    wr.writerow([name, c, i, f"{lo:.6e}", f"{hi:.6e}", int(h[i]), int(v[i]) if v is not None else 0])
    # quick sanity summary: changed share above and below the 2^-11 edge
    edge = (-11 - LO_EXP) * PER_OCT
    for name, _ in comps:
        n_lo = sum(int(hist[c][:edge].sum()) for c in hist); n_hi = sum(int(hist[c][edge:].sum()) for c in hist)
        c_lo = sum(int(chg[(name, c)][:edge].sum()) for c in hist if (name, c) in chg)
        c_hi = sum(int(chg[(name, c)][edge:].sum()) for c in hist if (name, c) in chg)
        print(f"SUMMARY {name}: changed below 2^-11 {c_lo}/{n_lo} = {c_lo/max(n_lo,1):.4%}; "
              f"at or above 2^-11 {c_hi}/{n_hi} = {c_hi/max(n_hi,1):.6%}; total {(c_lo+c_hi)/(n_lo+n_hi):.4%}", flush=True)


if __name__ == "__main__":
    main()
