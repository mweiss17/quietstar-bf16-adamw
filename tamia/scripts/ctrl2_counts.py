# CONTROLS 2 (M5 / R5): per-tensor mask counts matched to Arm A's footprint.
# Diffs Arm A's step-100 bf16 weights (job 648895, ckpt/full/bf16_step100) against the Hub base (revision 26bca36b, the
# weights Arm A started from) tensor by tensor (safetensors, streamed; CPU). For every tensor M2 masks (2-D, class not in
# rmsnorm / thought_emb / talk_head) k = number of coordinates whose bf16 value differs at step 100.
# embed_tokens / lm_head: the trained tensors have 32,002 rows (2 rows added by resize_token_embeddings, random init,
# absent from the Hub file); the density is measured on the 32,000 real rows and k = round(density x 32,002 x 4096).
# Note: "differs at step 100" <= "ever changed in steps 1..100" (Arm A's 5.65 %): a coordinate that moved and came back
# is not counted. Both are reported.
import glob, json, os, sys
import torch
from safetensors import safe_open

Q = os.environ.get("QSTAR_ROOT", "/scratch/m/mweiss10/qstar")
TRAINED = f"{Q}/ckpt/full/bf16_step100"
BASE = f"{Q}/hf_home/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{Q}/scripts/ctrl2_counts.json"
EXEMPT = ("rmsnorm", "thought_emb", "talk_head")


def classify_param(name):  # = pilot_train.classify_param
    if "talk_head" in name:
        return "talk_head"
    if name in ("start_embedding", "end_embedding"):
        return "thought_emb"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "rmsnorm"
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name:
        return "attn"
    return "other"


def index(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                m[k] = f
    return m


ti, bi = index(TRAINED), index(BASE)
per_tensor, cls_ch, cls_n, unmasked = {}, {}, {}, {}
for name in sorted(ti):
    cls = classify_param(name)
    with safe_open(ti[name], "pt") as s:
        w1 = s.get_tensor(name)
    if name not in bi:
        unmasked[name] = {"class": cls, "numel": w1.numel(), "note": "not in base (Quiet-STaR addition)"}
        continue
    with safe_open(bi[name], "pt") as s:
        w0 = s.get_tensor(name)
    assert w1.dtype == torch.bfloat16 and w0.dtype == torch.bfloat16, (name, w1.dtype, w0.dtype)
    rows = w0.shape[0]
    assert w1.shape[1:] == w0.shape[1:] and w1.shape[0] >= rows, (name, w1.shape, w0.shape)
    ch = int((w1[:rows] != w0).sum())
    n_real = w0.numel()
    dens = ch / n_real
    cls_ch[cls] = cls_ch.get(cls, 0) + ch; cls_n[cls] = cls_n.get(cls, 0) + n_real
    if w1.dim() == 2 and cls not in EXEMPT:
        k = ch if w1.shape[0] == rows else int(round(dens * w1.numel()))
        per_tensor[name] = {"class": cls, "numel": w1.numel(), "numel_real": n_real, "changed_real": ch, "density": dens, "k": k}
    else:
        unmasked[name] = {"class": cls, "numel": w1.numel(), "changed": ch, "density": dens}

k_tot = sum(v["k"] for v in per_tensor.values()); n_masked = sum(v["numel"] for v in per_tensor.values())
dens_cls = {}
for v in per_tensor.values():
    a = dens_cls.setdefault(v["class"], [0, 0]); a[0] += v["k"]; a[1] += v["numel"]
summary = {"trained": TRAINED, "base": BASE, "k_total": k_tot, "masked_numel": n_masked, "density_over_masked_tensors": k_tot / n_masked,
           "density_by_class_masked": {c: a[0] / a[1] for c, a in dens_cls.items()},
           "frac_differs_at_step100_by_class_real_rows": {c: cls_ch[c] / cls_n[c] for c in cls_n},
           "frac_differs_at_step100_all_base_coords": sum(cls_ch.values()) / sum(cls_n.values()),
           "per_tensor_density_min_median_max": [min(v["density"] for v in per_tensor.values()),
                                                 sorted(v["density"] for v in per_tensor.values())[len(per_tensor) // 2],
                                                 max(v["density"] for v in per_tensor.values())]}
json.dump({"summary": summary, "per_tensor": per_tensor, "unmasked": unmasked}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
print(f"CTRL2_COUNTS_OK n_masked_tensors={len(per_tensor)} -> {OUT}")
