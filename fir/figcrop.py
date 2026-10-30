"""Figure 3 data: one 256x256 crop of an MLP matrix, base |w| plus which coordinates differ from base at each saved checkpoint.

Per-step change masks were never logged, so this is the coarse version: checkpoints at the steps we saved
(bf16 run: 50, 100; fp32 lr 1e-7: 25, 50, 75, 100; fp32 lr 1e-6: 25, 50, 75, 100).

Usage: python figcrop.py BASE_DIR OUT_CSV TENSOR name=DIR [name=DIR ...]
Output columns: run,row,col,w0,w,changed
"""
import os, sys, csv, torch
from safetensors import safe_open

R0, C0, N = 1024, 2048, 256


def find(d, k):
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt") as sf:
                if k in sf.keys():
                    return sf.get_slice(k)[R0:R0 + N, C0:C0 + N].to(torch.bfloat16)
    raise KeyError(k)


def main():
    base, out, k = sys.argv[1], sys.argv[2], sys.argv[3]
    w0 = find(base, k)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "row", "col", "w0", "w", "changed"])
        for i in range(N):
            for j in range(N):
                wr.writerow(["base", R0 + i, C0 + j, f"{float(w0[i, j]):.6e}", f"{float(w0[i, j]):.6e}", 0])
        for spec in sys.argv[4:]:
            name, d = spec.split("=", 1)
            if not os.path.isdir(d):
                print("skip", name, d, flush=True)
                continue
            w = find(d, k)
            m = w != w0
            print(name, "changed %.4f%%" % (100 * m.float().mean().item()), flush=True)
            for i in range(N):
                for j in range(N):
                    wr.writerow([name, R0 + i, C0 + j, f"{float(w0[i, j]):.6e}", f"{float(w[i, j]):.6e}", int(m[i, j])])


if __name__ == "__main__":
    main()
