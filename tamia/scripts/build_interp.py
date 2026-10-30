# PILOT: build interpolated checkpoints  w = base + alpha * (shadow - base)  in fp32.
#   --from shadow_bf16_dir  --alpha 0.3 --out_dtype bf16   (the coordinator's spec: built in fp32, cast to bf16)
#   --from shadow_fp32_dir  --alpha 0.3 --out_dtype fp32   (same interpolation on the un-rounded fp32 master)
# talk_head.*, start_embedding, end_embedding and the two resized vocab rows (32000, 32001) are copied VERBATIM from
# the shadow. config / tokenizer files are copied from the shadow dir. Output is a single model.safetensors.
import os, sys, json, glob, shutil, argparse, time
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base_dir", required=True, help="HF snapshot dir of the pinned base revision")
ap.add_argument("--from_dir", required=True)
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--out_dtype", choices=["bf16", "fp32"], default="bf16")
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()
t0 = time.time()
base = {}
for f in sorted(glob.glob(os.path.join(args.base_dir, "*.safetensors"))):
    base.update(load_file(f))
sh = {}
for f in sorted(glob.glob(os.path.join(args.from_dir, "*.safetensors"))):
    sh.update(load_file(f))
print(f"loaded base {len(base)} keys, shadow {len(sh)} keys ({time.time()-t0:.0f}s)", flush=True)
out = {}
verbatim, interp, n_coords, n_changed_vs_base, n_changed_vs_shadow = [], [], 0, 0, 0
odt = torch.bfloat16 if args.out_dtype == "bf16" else torch.float32
for k, t in sh.items():
    if k.startswith("talk_head.") or k in ("start_embedding", "end_embedding") or k not in base:
        out[k] = t.to(odt).contiguous(); verbatim.append(k); continue
    b = base[k].float(); s = t.float()
    if b.shape != s.shape:  # resized embed_tokens / lm_head: 32000 -> 32002 rows
        assert b.shape[0] == 32000 and s.shape[0] == 32002, (k, b.shape, s.shape)
        w = s.clone(); w[:32000] = b + args.alpha * (s[:32000] - b)
    else:
        w = b + args.alpha * (s - b)
    w = w.to(odt)
    out[k] = w.contiguous(); interp.append(k)
    n_coords += w.numel()
    n_changed_vs_base += int((w[:32000].float() != b).sum()) if b.shape != s.shape else int((w.float() != b).sum())
    n_changed_vs_shadow += int((w != t.to(odt)).sum())
os.makedirs(args.out_dir, exist_ok=True)
for f in os.listdir(args.from_dir):
    if f.endswith(".json") or f.endswith(".model") or f == "tokenizer.json":
        if f != "model.safetensors.index.json":
            shutil.copy(os.path.join(args.from_dir, f), os.path.join(args.out_dir, f))
cfg = json.load(open(os.path.join(args.out_dir, "config.json")))
cfg["torch_dtype"] = "bfloat16" if args.out_dtype == "bf16" else "float32"
json.dump(cfg, open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)
save_file(out, os.path.join(args.out_dir, "model.safetensors"), metadata={"format": "pt"})
summ = {"from_dir": args.from_dir, "alpha": args.alpha, "out_dtype": args.out_dtype, "n_interp_keys": len(interp), "verbatim_keys": verbatim,
        "n_coords_interp": n_coords, "frac_out_ne_base": n_changed_vs_base / n_coords, "frac_out_ne_shadow": n_changed_vs_shadow / n_coords,
        "seconds": time.time() - t0}
json.dump(summ, open(os.path.join(args.out_dir, "PILOT_interp_summary.json"), "w"), indent=1)
print("PILOT_INTERP", json.dumps(summ), flush=True)
