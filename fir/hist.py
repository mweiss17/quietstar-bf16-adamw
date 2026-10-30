import os, json, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
torch.set_num_threads(8)
repo = "mistralai/Mistral-7B-v0.1"
d = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
lrs = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 3e-4]
scales = [1.0, 0.3]   # |update| = scale*lr  (Adam step magnitude is <= lr; typically ~0.1-0.5 lr)
tot = 0; moved = {(lr, s): 0 for lr in lrs for s in scales}; per = []
for f in sorted(os.listdir(d)):
    if not f.endswith(".safetensors"): continue
    with safe_open(os.path.join(d, f), "pt") as sf:
        for k in sf.keys():
            w = sf.get_tensor(k).to(torch.bfloat16).flatten()
            n = w.numel(); tot += n
            row = {"name": k, "n": n, "absmean": float(w.float().abs().mean())}
            for lr in lrs:
                for s in scales:
                    delta = s * lr
                    # exact bf16 emulation of p.add_(delta): fp32 add, round to bf16, compare
                    up = ((w.float() + delta).to(torch.bfloat16) != w) | ((w.float() - delta).to(torch.bfloat16) != w)
                    c = int(up.sum()); moved[(lr, s)] += c; row[f"frac_moved_lr{lr:g}_x{s:g}"] = c / n
            per.append(row)
            print(k, n, {f"lr{lr:g}": round(row[f"frac_moved_lr{lr:g}_x1"], 4) for lr in lrs}, flush=True)
summary = {"model": repo, "total_params": tot,
           "frac_of_weights_that_can_move": {f"lr={lr:g} |d|={s:g}*lr": moved[(lr, s)] / tot for lr in lrs for s in scales}}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "per_tensor": per}, open("mistral7b_bf16_movable.json", "w"))
