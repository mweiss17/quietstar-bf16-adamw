# CONTROLS (evalD): update-mask logic shared by pilot_train.py and the CPU smoke test.
# All masks are applied as a post-step correction on fp32 master weights:
#   prev = w (snapshot before the optimizer step); optimizer.step(); w = where(mask, w_new, prev)
# so the optimizer's moments (exp_avg, exp_avg_sq) are updated on every coordinate as normal; only the weight write is masked.
import math
import torch

EXEMPT_CLASSES = ("rmsnorm", "thought_emb", "talk_head")  # M2 / R / N: these train unmasked


def mask_applies(mode, cls, p):
    """Does `mode` mask this tensor? M1 (bf16vis) masks every parameter (in Arm A every parameter is bf16);
    M2/R/N mask every 2-D matrix (embed, lm_head, attention, MLP) except the exempt classes."""
    if mode == "bf16vis":
        return True
    if mode in ("static_mag", "random", "nanoadam"):
        return p.dim() == 2 and cls not in EXEMPT_CLASSES
    return False


@torch.no_grad()
def bf16_visible_mask(w_new, w_old):
    """M1: coordinates where bf16(w_old + delta) != bf16(w_old), with w_new = w_old + delta the fp32 optimizer result."""
    return w_new.to(torch.bfloat16) != w_old.to(torch.bfloat16)


@torch.no_grad()
def static_mag_mask(w, cut):
    """M2 (PaFi-style): |w0| < cut."""
    return w.abs() < cut


@torch.no_grad()
def bottomk_mask(w, k):
    """N (NanoAdam): exactly the k smallest-|w| coordinates of this tensor (ties broken by torch.topk)."""
    m = torch.zeros(w.numel(), dtype=torch.bool, device=w.device)
    if k > 0:
        idx = torch.topk(w.detach().abs().flatten().float(), k, largest=False, sorted=False).indices
        m[idx] = True
    return m.view_as(w)


NANO_BLOCK_H200 = 29056  # = ((228 KB smem/SM - 1 KB) / 4 B) // 2 on sm90 (H100/H200), as ista_daslab_tools computes it


def nano_block_counts(n, density, block):
    """(full blocks, block size, k per full block, remainder size, k in remainder) exactly as NanoAdam.py
    _initialize_parameter_state: d_block = min(n, block); block_split; k_many = ceil(d_block * density); k_few = ceil(rem * density)."""
    d_block = min(n, block)
    if n < d_block or n == d_block:
        nb, start = 1, n
    else:
        nb, start = n // d_block, n - n % d_block
    rem = n - start
    return nb, d_block, int(math.ceil(d_block * density)), rem, int(math.ceil(rem * density))


@torch.no_grad()
def nanoadam_block_mask(w, density, block=NANO_BLOCK_H200):
    """N, matching the authors' selection rule (RelationalML/nanoadam @ 3f8f916, Nanoadam.py update_step, largest=False,
    mask_criterion='weights'): flatten the tensor, split into contiguous blocks of `block` floats, take the
    ceil(block * density) smallest-|w| coordinates of every full block and ceil(rem * density) of the remainder block."""
    flat = w.detach().flatten()
    n = flat.numel()
    nb, d_block, k_many, rem, k_few = nano_block_counts(n, density, block)
    m = torch.zeros(n, dtype=torch.bool, device=w.device)
    start = nb * d_block
    if k_many > 0:
        idx = torch.topk(flat[:start].abs().float().view(nb, d_block), k_many, dim=1, largest=False, sorted=False).indices
        idx = idx + (torch.arange(nb, device=w.device) * d_block).unsqueeze(1)
        m[idx.flatten()] = True
    if rem > 0 and k_few > 0:
        idx = torch.topk(flat[start:].abs().float(), k_few, largest=False, sorted=False).indices
        m[idx + start] = True
    return m.view_as(w)


@torch.no_grad()
def random_mask(w, k, seed):
    """R: a uniformly random subset of exactly k coordinates (generator on the tensor's device, seeded per tensor)."""
    g = torch.Generator(device=w.device)
    g.manual_seed(int(seed))
    r = torch.rand(w.numel(), generator=g, device=w.device)
    m = torch.zeros(w.numel(), dtype=torch.bool, device=w.device)
    if k > 0:
        m[torch.topk(r, k, largest=False, sorted=False).indices] = True
    return m.view_as(w)


@torch.no_grad()
def apply_mask_(p, prev, mask):
    """w = where(mask, w_new, w_old), in place."""
    p.copy_(torch.where(mask, p, prev))
