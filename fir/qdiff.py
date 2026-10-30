import os, json, math, collections, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
torch.set_num_threads(8)
base = snapshot_download("mistralai/Mistral-7B-v0.1", allow_patterns=["*.safetensors","*.json"])
ck   = snapshot_download("ezelikman/quietstar-8-ahead", allow_patterns=["*.safetensors","*.json"])
def load_index(d):
    idx = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d,f),"pt") as sf:
                for k in sf.keys(): idx[k] = f
    return idx
bi, ci = load_index(base), load_index(ck)
print("base tensors", len(bi), "ckpt tensors", len(ci))
new_keys = [k for k in ci if k not in bi]; missing = [k for k in bi if k not in ci]
print("NEW in ckpt:", new_keys[:40]); print("MISSING from ckpt:", missing[:20])
def get(d, idx, k):
    with safe_open(os.path.join(d, idx[k]), "pt") as sf: return sf.get_tensor(k)
def cls(n):
    if "norm" in n: return "rmsnorm"
    if "embed_tokens" in n: return "embed"
    if "lm_head" in n: return "lm_head"
    if "self_attn" in n: return "attn"
    if "mlp" in n: return "mlp"
    return "other"
agg = collections.defaultdict(lambda: collections.Counter())
dtypes = collections.Counter()
for k in bi:
    if k not in ci: continue
    w0 = get(base, bi, k); w1 = get(ck, ci, k); dtypes[str(w1.dtype)] += 1
    if w0.shape != w1.shape: print("SHAPE DIFF", k, w0.shape, w1.shape); 
    n = min(w0.numel(), w1.numel()); w0 = w0.flatten()[:n]; w1 = w1.flatten()[:n]
    w0b = w0.to(torch.bfloat16); w1b = w1.to(torch.bfloat16)
    changed = (w1b != w0b)
    c = cls(k); a = agg[c]; a["n"] += n; a["changed"] += int(changed.sum())
    small = w0b.float().abs() < 5.12e-4   # 512*lr at lr 1e-6
    a["n_small"] += int(small.sum()); a["changed_small"] += int((changed & small).sum()); a["changed_big"] += int((changed & ~small).sum())
    if changed.any():
        x0 = w0b[changed].float(); x1 = w1b[changed].float()
        e = torch.floor(torch.log2(x0.abs().clamp_min(1e-30))); ulp = torch.pow(2.0, e - 7)
        ulps = ((x1 - x0).abs() / ulp).round()
        for lo, hi, name in [(1,1,"1ulp"),(2,2,"2ulp"),(3,4,"3-4ulp"),(5,16,"5-16ulp"),(17,1e9,">16ulp")]:
            a[name] += int(((ulps >= lo) & (ulps <= hi)).sum())
        a["zero_base_changed"] += int((x0 == 0).sum())
        a["signflip"] += int(((x0 > 0) != (x1 > 0)).sum())
        a["sum_rel"] += float(((x1 - x0).abs() / x0.abs().clamp_min(1e-30)).clamp_max(10).sum())
print("ckpt dtypes:", dict(dtypes))
tot = sum(a["n"] for a in agg.values()); ch = sum(a["changed"] for a in agg.values())
print("OVERALL: params=%d changed=%d (%.3f%%)" % (tot, ch, 100*ch/tot))
print("%-8s %13s %9s %9s %9s %9s | %7s %7s %7s %7s %7s | %8s %8s" % ("class","n","changed%","small%","chg|small","chg|big","1ulp","2ulp","3-4","5-16",">16","signflip","meanrel"))
for c, a in sorted(agg.items(), key=lambda x: -x[1]["n"]):
    n=a["n"]; ch=a["changed"] or 1
    print("%-8s %13d %8.3f%% %8.2f%% %8.2f%% %8.3f%% | %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% | %8d %8.4f" % (
        c, n, 100*a["changed"]/n, 100*a["n_small"]/n, 100*a["changed_small"]/max(a["n_small"],1), 100*a["changed_big"]/max(n-a["n_small"],1),
        100*a["1ulp"]/ch, 100*a["2ulp"]/ch, 100*a["3-4ulp"]/ch, 100*a["5-16ulp"]/ch, 100*a[">16ulp"]/ch, a["signflip"], a["sum_rel"]/ch))
print("--- new tensors (talk head etc.)")
for k in new_keys:
    t = get(ck, ci, k).float().flatten()
    print("%-60s shape=%-18s dtype=%s absmean=%.5f std=%.5f zeros=%.3f" % (k, tuple(get(ck,ci,k).shape), get(ck,ci,k).dtype, t.abs().mean(), t.std(), (t==0).float().mean()))
print("--- embeddings of added tokens (last 2 rows of embed_tokens) vs '---' token")
e0 = get(base, bi, "model.embed_tokens.weight"); e1 = get(ck, ci, "model.embed_tokens.weight")
print("embed shapes", tuple(e0.shape), tuple(e1.shape))
if e1.shape[0] > e0.shape[0]:
    for r in range(e0.shape[0], e1.shape[0]):
        row = e1[r].float(); print("added row", r, "absmean=%.5f" % row.abs().mean(), "max|.|=%.4f" % row.abs().max())
