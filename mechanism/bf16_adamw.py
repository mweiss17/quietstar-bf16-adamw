"""Does torch.optim.AdamW silently break on bf16 parameters?
(1) weight decay p *= (1 - lr*wd): is it a no-op in bf16?
(2) exp_avg_sq with beta2=0.999: can it decay / update?
(3) small-lr updates below half-ulp dropped?"""
import torch, math
torch.manual_seed(0)
print(torch.__version__)

def run(dtype, lr, wd, betas, steps, grad_fn, fused=False, foreach=None, label=""):
    p = torch.full((1024,), 0.02, dtype=dtype).requires_grad_(True)
    kw = dict(lr=lr, weight_decay=wd, betas=betas)
    if fused: kw["fused"] = True
    if foreach is not None: kw["foreach"] = foreach
    opt = torch.optim.AdamW([p], **kw)
    for t in range(steps):
        p.grad = grad_fn(t, p)
        opt.step()
    st = opt.state[p]
    return p.detach().double(), st["exp_avg"].double(), st["exp_avg_sq"].double()

print("\n== (1) weight decay only (grad = 0): lr=1e-5, wd=0.1, 1000 steps; p0=0.02 ==")
zero = lambda t, p: torch.zeros_like(p)
for dtype in (torch.float32, torch.bfloat16):
    p, m, v = run(dtype, 1e-5, 0.1, (0.9, 0.999), 1000, zero)
    expected = 0.02 * (1 - 1e-5 * 0.1) ** 1000
    print(f"  {str(dtype):16s} p={p[0]:.8f}  expected={expected:.8f}  decayed={p[0] < 0.02}")
for lr, wd in [(1e-4, 0.1), (1e-3, 0.1), (3e-4, 1.0), (1e-2, 0.1)]:
    p, m, v = run(torch.bfloat16, lr, wd, (0.9, 0.999), 100, zero)
    print(f"  bf16 lr={lr:g} wd={wd:g}: lr*wd={lr*wd:.1e} vs half-ulp(0.02)/0.02={2**-9:.1e}  p after 100 steps={p[0]:.6f} (fp32 would be {0.02*(1-lr*wd)**100:.6f})")

print("\n== (2) exp_avg_sq with constant grad g=1e-3 (true steady state v = g^2 = 1e-6) ==")
const = lambda t, p: torch.full_like(p, 1e-3)
for beta2 in (0.999, 0.99, 0.95):
    for dtype in (torch.float32, torch.bfloat16):
        p, m, v = run(dtype, 1e-10, 0.0, (0.9, beta2), 5000, const)
        print(f"  beta2={beta2} {str(dtype):16s} v={v[0]:.4e}  (true 1-{beta2}^5000 ≈ {1e-6*(1-beta2**5000):.4e})  m={m[0]:.4e}")

print("\n== (2b) can v DECAY? grad spike then zero grads, beta2=0.999 ==")
def spike(t, p):
    return torch.full_like(p, 1.0 if t < 50 else 1e-3)
for dtype in (torch.float32, torch.bfloat16):
    p, m, v = run(dtype, 1e-10, 0.0, (0.9, 0.999), 3000, spike)
    print(f"  {str(dtype):16s} v after 50 spike steps + 2950 small steps = {v[0]:.4e}  (fp32 decays toward 1e-6)")

print("\n== (3) weight update dropped: constant grad g=1e-3, lr=1e-5, wd=0, 200 steps; update ≈ lr per step ==")
for dtype in (torch.float32, torch.bfloat16):
    p, m, v = run(dtype, 1e-5, 0.0, (0.9, 0.999), 200, const)
    print(f"  {str(dtype):16s} p={p[0]:.6f}  moved={0.02 - p[0]:.2e}  (fp32 ≈ 200*lr = 2e-3)")

print("\n== fused / foreach variants, bf16, weight decay only lr=1e-4 wd=0.1 ==")
for fused, foreach in ((False, False), (False, True), (True, None)):
    try:
        p, m, v = run(torch.bfloat16, 1e-4, 0.1, (0.9, 0.999), 100, zero, fused=fused, foreach=foreach)
        print(f"  fused={fused} foreach={foreach}: p={p[0]:.6f} (fp32 {0.02*(1-1e-5)**100:.6f})")
    except Exception as e:
        print(f"  fused={fused} foreach={foreach}: {type(e).__name__}: {str(e)[:80]}")

print("\n== has torch.optim.Muon? ==", hasattr(torch.optim, "Muon"))
