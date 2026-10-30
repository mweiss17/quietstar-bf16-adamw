"""Emulate torchao.optim single_param_adam on bf16 params/states (fp32 compute, copy back to bf16,
optional stochastic rounding of the PARAM only) to see which of (a)/(b)/(c) it fixes."""
import torch; torch.manual_seed(0)
def sr(x):  # torchao _fp32_to_bf16_sr: add random low 16 bits then truncate
    r = torch.randint_like(x, 0, 1<<16, dtype=torch.int32)
    return (x.view(torch.int32) + r).bitwise_and_(-65536).view(torch.float32).bfloat16()
def step(p, g, m, v, lr, wd, b1, b2, t, use_sr):
    pf = p.float(); gf = g.float()
    pf = pf - lr*wd*pf                       # decoupled wd in fp32 (torchao line 21)
    mf = m.float().lerp(gf, 1-b1); vf = v.float().lerp(gf*gf, 1-b2)
    m.copy_(mf); v.copy_(vf)                 # states copied back to bf16 (torchao lines 32-33)
    bc1 = 1-b1**t; bc2 = 1-b2**t
    pf = pf - lr*(mf/bc1)/((vf.sqrt()/bc2**0.5)+1e-8)
    p.copy_(sr(pf) if use_sr else pf.bfloat16())
for use_sr in (False, True):
    # (b) steady state v with constant g=1e-3, beta2=0.999 (true 1e-6); (a) wd-only; (c) small-lr update
    p=torch.full((4096,),0.02,dtype=torch.bfloat16); m=torch.zeros_like(p); v=torch.zeros_like(p)
    for t in range(1,5001): step(p, torch.full_like(p,1e-3), m, v, 1e-10, 0.0, 0.9, 0.999, t, use_sr)
    vb = v.double().mean().item()
    p2=torch.full((4096,),0.02,dtype=torch.bfloat16); m2=torch.zeros_like(p2); v2=torch.zeros_like(p2)
    for t in range(1,1001): step(p2, torch.zeros_like(p2), m2, v2, 1e-5, 0.1, 0.9, 0.999, t, use_sr)
    p3=torch.full((4096,),0.02,dtype=torch.bfloat16); m3=torch.zeros_like(p3); v3=torch.zeros_like(p3)
    for t in range(1,201): step(p3, torch.full_like(p3,1e-3), m3, v3, 1e-5, 0.0, 0.9, 0.999, t, use_sr)
    print(f"sr={use_sr}: (b) v after 5000 steps = {vb:.3e} (true ~1e-6, fp32 gives 9.93e-7) | "
          f"(a) wd-only p mean = {p2.double().mean():.6f} (expected {0.02*(1-1e-6)**1000:.6f}) | "
          f"(c) lr=1e-5 200 steps moved = {0.02-p3.double().mean():.2e} (fp32 ~2e-3)")
