# PILOT: derived from quiet-star-train.py (ezelikman/quiet-star @ 892446b).
# PILOT: every change relative to the released script is marked "# PILOT:".
# PILOT: intended behavioural differences vs the released recipe: no eval during training,
# PILOT: clean stop at kill_after steps, data loaded from disk, wandb off, fp32 shadow optimizer + stats.
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset, load_from_disk  # PILOT: load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer, TrainerCallback  # PILOT: TrainerCallback
import os
import time
import wandb
from huggingface_custom_callback import EarlyStoppingCallback  # PILOT: stubbed module (not in the released repo)
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
import argparse, json, gc, math, sys  # PILOT
from collections import defaultdict  # PILOT
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# PILOT: command-line knobs (defaults = released recipe)
QSTAR = os.environ.get("QSTAR_ROOT", "/scratch/mweiss10/qstar")  # PILOT: Vulcan default; tamIA uses /scratch/m/mweiss10/qstar
_ap = argparse.ArgumentParser()
_ap.add_argument("--steps", type=int, default=100, help="optimizer steps to run (= model.kill_after)")
_ap.add_argument("--micro_batch", type=int, default=4, help="per-device batch; grad accum = 8 // micro_batch")
_ap.add_argument("--out_dir", type=str, default=f"{QSTAR}/ckpt/full")
_ap.add_argument("--owm_dir", type=str, default=f"{QSTAR}/data/owm_first1000")
_ap.add_argument("--sample_per_class", type=int, default=150_000, help="fixed exp_avg_sq sample size per tensor class")
_ap.add_argument("--no_save", action="store_true")
_ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "32")))
_ap.add_argument("--layers_per_gpu", type=str, default="", help="e.g. '8,8,8,8': explicit decoder-layer split (placement only); '' = device_map='auto' as released")
_ap.add_argument("--mem_history", action="store_true", help="record CUDA allocator history and print the OOM allocation stack on failure")
_ap.add_argument("--ckpt_mlp", action="store_true", help="activation-checkpoint MistralMLP.forward (non-reentrant recompute; same math, ~0.85 GiB less per layer)")
_ap.add_argument("--profile_probe", action="store_true", help="diagnostics: run under torch.profiler with memory tracking and print the top CUDA-memory ops, then exit")
_ap.add_argument("--save_at", type=int, default=50, help="mid-run safety save (bf16 weights, shadow fp32/bf16, stats) after this optimizer step; 0 = off")
_ap.add_argument("--fp32_master", action="store_true", help="ARM B: fp32 parameters + fused AdamW on them, forward under torch.autocast(bf16) via TrainingArguments(bf16=True); no shadow (ArmBStats instead)")
_ap.add_argument("--save_steps", type=str, default="25,50,75", help="ARM B: optimizer steps after which a bf16-cast checkpoint is written (the final step always gets bf16-cast + fp32)")
_ap.add_argument("--lr", type=float, default=1e-6, help="peak learning rate (released: 1e-6); the linear warmup/decay scales with it")
_ap.add_argument("--no_final_fp32", action="store_true", help="ARM B: skip the fp32 checkpoint at the final step (disk)")
_ap.add_argument("--arm", type=str, default="", help="CONTROLS: label written into the stats records (e.g. M1, M2, R, N, T)")
_ap.add_argument("--mask_mode", type=str, default="none", choices=["none", "bf16vis", "static_mag", "random", "nanoadam"],
                 help="CONTROLS (needs --fp32_master): post-step update mask. bf16vis = M1 (apply the exact fp32 step only where bf16(w+d) != bf16(w)); "
                      "static_mag = M2 (|w0| < --mag_cut, fixed); random = R (fixed random subset, M2's per-tensor count, --mask_seed); "
                      "nanoadam = N (bottom-k |w| per tensor, k = M2's per-tensor count, recomputed every --mask_interval steps)")
_ap.add_argument("--mag_cut", type=float, default=5.12e-4, help="CONTROLS: M2 magnitude cutoff (= 512 x peak lr 1e-6)")
_ap.add_argument("--mask_seed", type=int, default=1234, help="CONTROLS: R mask seed (tensor i uses mask_seed + i)")
_ap.add_argument("--mask_interval", type=int, default=10, help="CONTROLS: N mask recompute interval in optimizer steps (authors' schedule: before step 1 and every step s with s %% interval == 0)")
_ap.add_argument("--nano_block", type=int, default=29056, help="CONTROLS N: NanoAdam block size (authors: get_max_floats_for_shared_memory_per_thread_block()//2 = 29056 on H100/H200)")
_ap.add_argument("--no_thoughts", action="store_true", help="CONTROLS T: train with thoughts disabled (original_mode=True, n_ahead=n_ahead_talk=n_passes=1)")
_ap.add_argument("--gpu_stats", action="store_true", help="CONTROLS T: bf16 (released) optimizer path with the on-GPU ArmBStats statistics instead of the CPU fp32 shadow")
_ap.add_argument("--sample_steps", type=str, default="1,2,5,10,20,50,100", help="CONTROLS: steps at which a fixed coordinate sample of (|w|, |proposed update|, applied) is saved")
_ap.add_argument("--n_sample", type=int, default=200_000, help="CONTROLS: coordinates in that sample")
PILOT = _ap.parse_args()
assert PILOT.mask_mode == "none" or PILOT.fp32_master, "--mask_mode needs --fp32_master"
assert not (PILOT.gpu_stats and PILOT.fp32_master), "--gpu_stats is the bf16 path"
USE_B_PATH = PILOT.fp32_master or PILOT.gpu_stats  # CONTROLS: ArmBStats + bf16-cast saves
from pilot_masks import mask_applies, bf16_visible_mask, static_mag_mask, bottomk_mask, random_mask, apply_mask_, nanoadam_block_mask, nano_block_counts  # CONTROLS
if PILOT.fp32_master:
    # ARM B: torch 2.3 SDPA rejects a floating attn_mask whose dtype differs from the query dtype. Under autocast the
    # released forward builds its 4-D mask from inputs_embeds.dtype (fp32 here: embedding lookups are not autocast ops)
    # while q/k/v come out of autocast Linears in bf16 -> "Expected attn_mask dtype to be bool or to match query dtype".
    # Minimal fix: cast the mask to the query dtype at the call. The clamp keeps the fill value finite (fp32 finfo.min
    # would round to -inf in bf16), i.e. exactly the bf16 finfo.min that Arm A's bf16 mask used.
    _sdpa_orig = torch.nn.functional.scaled_dot_product_attention

    def _sdpa_cast_mask(query, key, value, attn_mask=None, *a, **kw):
        if attn_mask is not None and attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.clamp(min=torch.finfo(query.dtype).min).to(query.dtype)
        return _sdpa_orig(query, key, value, attn_mask, *a, **kw)
    torch.nn.functional.scaled_dot_product_attention = _sdpa_cast_mask
    print("PILOT-B: SDPA attn_mask dtype cast patch installed", flush=True)


def pilot_gpu_mem(reset=False):  # PILOT: per-GPU current / peak allocated (GiB); optionally reset peak stats
    out = {}
    for d in range(torch.cuda.device_count()):
        out[f"gpu{d}"] = {"alloc_gib": torch.cuda.memory_allocated(d) / 2**30, "max_alloc_gib": torch.cuda.max_memory_allocated(d) / 2**30,
                          "reserved_gib": torch.cuda.memory_reserved(d) / 2**30}
        if reset:
            torch.cuda.reset_peak_memory_stats(d)
    return out


def pilot_detach_log_dict(module, inputs, output):  # PILOT: forward hook
    # The released forward accumulates `log_dict[k] += <tensor with autograd graph>` every micro-batch and only resets
    # log_dict inside `if self.wandb_enabled:` (every n_tokens_print forwards = once per optimizer step). With wandb off
    # the dict is never reset and keeps alive graph branches the main backward never traverses (e.g. the CE of every
    # intermediate thought iteration behind rel_loss_i: a 262 MB fp32 log-softmax each, on the head GPU) -> ~6 GB/step
    # leak (job 648841 died at step 15). Detaching keeps the logged values identical and frees the graphs; the dict is
    # reset once per optimizer step in ShadowAdamW.pre_step, i.e. at the released (wandb-on) cadence.
    if module.training:
        for k, v in list(module.log_dict.items()):
            if torch.is_tensor(v):
                module.log_dict[k] = v.detach()
torch.set_num_threads(PILOT.threads)
if PILOT.mem_history:  # PILOT: diagnostics only
    torch.cuda.memory._record_memory_history(max_entries=200000)
if PILOT.ckpt_mlp:
    # PILOT: memory only. 4x L40S-48GB cannot hold the released forward's activations even at micro-batch 1 (2 sequences
    # after n_passes=2, 15 forward iterations): ~2.8 GiB per decoder layer plus a ~14 GiB backward transient. The MLP block
    # is stateless, so recomputing it in backward (torch.utils.checkpoint, use_reentrant=False) gives identical values
    # (same bf16 inputs, same GEMMs). Whole-layer checkpointing is NOT usable here: the thought loop needs the KV cache.
    import torch.utils.checkpoint
    from transformers.models.mistral import modeling_mistral as _mm
    _orig_mlp_forward = _mm.MistralMLP.forward

    def _ckpt_mlp_forward(self, x):
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(_orig_mlp_forward, self, x, use_reentrant=False)
        return _orig_mlp_forward(self, x)

    _mm.MistralMLP.forward = _ckpt_mlp_forward
    print("PILOT: MistralMLP.forward activation checkpointing enabled", flush=True)


def pilot_device_map():  # PILOT: placement only; the math is identical to device_map='auto'
    if not PILOT.layers_per_gpu:
        return "auto"
    counts = [int(x) for x in PILOT.layers_per_gpu.split(",")]
    assert sum(counts) == 32, counts
    dm = {"model.embed_tokens": 0, "start_embedding": 0, "end_embedding": 0}
    li = 0
    for g, c in enumerate(counts):
        for _ in range(c):
            dm[f"model.layers.{li}"] = g
            li += 1
    last = len(counts) - 1
    dm.update({"model.norm": last, "lm_head": last, "talk_head": last})
    return dm


def pilot_oom_report(e):  # PILOT: diagnostics only
    print("PILOT OOM:", str(e)[:500], flush=True)
    for d in range(torch.cuda.device_count()):
        print(f"PILOT OOM gpu{d}: alloc={torch.cuda.memory_allocated(d)/2**30:.2f}G max_alloc={torch.cuda.max_memory_allocated(d)/2**30:.2f}G "
              f"reserved={torch.cuda.memory_reserved(d)/2**30:.2f}G", flush=True)
    if PILOT.mem_history:
        try:
            snap = torch.cuda.memory._snapshot()
            for dev, trace in enumerate(snap.get("device_traces", [])):
                for ev in trace:
                    if ev.get("action") == "oom":
                        print(f"PILOT OOM event on gpu{dev}: requested={ev.get('size', 0)/2**30:.2f}G", flush=True)
                        for fr in ev.get("frames", [])[:40]:
                            print("     ", fr.get("filename", "?").split("site-packages/")[-1], fr.get("line"), fr.get("name"), flush=True)
            # largest live blocks per device
            for seg in sorted(snap.get("segments", []), key=lambda s: -s.get("total_size", 0))[:0]:
                pass
            big = []
            for seg in snap.get("segments", []):
                for b in seg.get("blocks", []):
                    if b.get("state") == "active_allocated" and b.get("size", 0) > 2**28:
                        big.append((b["size"], seg.get("device"), b.get("frames", [])))
            big.sort(key=lambda x: -x[0])
            for size, dev, frames in big[:12]:
                print(f"PILOT live block gpu{dev} {size/2**30:.2f}G:", " <- ".join(f"{fr.get('name')}@{fr.get('filename','?').split('/')[-1]}:{fr.get('line')}" for fr in frames[:6]), flush=True)
        except Exception as ex:
            print("PILOT OOM snapshot failed:", repr(ex), flush=True)
# PILOT: pin Mistral-7B-v0.1 to the last pre-July-2024 Hub revision (2023-12-11). The July-2024 "Align tokenizer with
# PILOT: mistral-common" commit rewrote tokenizer.json in a format tokenizers 0.15 (transformers 4.37) cannot parse;
# PILOT: the weights are byte-identical across revisions. This is the tokenizer the authors had in early 2024.
MISTRAL_REV = "26bca36bde8333b5d7f72e9ed20ccda6a618af24"
os.makedirs(PILOT.out_dir, exist_ok=True)
STATS_DIR = os.path.join(PILOT.out_dir, "stats")
os.makedirs(STATS_DIR, exist_ok=True)
print("PILOT args:", vars(PILOT), flush=True)

# MAIN SETUP
root_prefix = QSTAR + "/"  # PILOT: was "YOUR_CACHE_PATH_HERE"
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'
# dataset_name = 'c4'
project_name = "quiet-star"
os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
os.environ["WANDB_MODE"] = "disabled"  # PILOT
n_ahead_talk_global = 4
n_passes_global = 2
n_ahead_global = 12
n_examples = 1_000
full_batch_size = 8
eval_and_logging_steps = 10
save_steps = 100

def model_init(params):
    gc.collect(); torch.cuda.empty_cache()  # PILOT: Trainer calls model_init twice (init + train); free the first copy
    original = PILOT.no_thoughts  # CONTROLS T: released code hard-codes False; True -> n_ahead = n_ahead_talk = n_passes = 1, original_mode
    if params is None:
        params = {}
    else:
        params = params.params
    # save params to file
    n_ahead = params.get("n_ahead", n_ahead_global if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", n_ahead_talk_global if not original else 1)
    n_passes = params.get("n_passes", n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = "mistralai/Mistral-7B-v0.1"
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=MISTRAL_REV,  # PILOT: see MISTRAL_REV
        torch_dtype=(torch.float32 if PILOT.fp32_master else torch.bfloat16) if torch.cuda.is_available() else torch.float32,  # ARM B: fp32 master weights
        device_map=pilot_device_map(),  # PILOT: released 'auto' (8/9/9/6 layers) unless --layers_per_gpu is given
        # cache_dir=root_prefix + "cache",  # PILOT: use HF_HOME (prefetched) instead of a second cache dir
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
    )
    print("Loaded model")
    print("PILOT hf_device_map:", getattr(model, "hf_device_map", None), flush=True)  # PILOT
    print("PILOT gpu mem after load:", {k: round(v["alloc_gib"], 2) for k, v in pilot_gpu_mem().items()}, flush=True)  # PILOT
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision=MISTRAL_REV)  # PILOT: revision
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens_to_add = []
    if model.use_start_thought_token:
        special_tokens_to_add.append("<|startthought|>")
    if model.use_end_thought_token:
        special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.wandb_enabled = False  # PILOT: was True
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    model.kill_after = PILOT.steps  # PILOT: released value 100 (= default of --steps)
    model.train()
    return model

# Load dataset
# PILOT: released code: load_dataset(dataset_name, "default", split=f"train[:{n_examples}]", ignore_verifications=True, num_proc=16, cache_dir=...)
# PILOT: that downloads 27 GB; prep_data.py streamed the same first 1,000 rows (same shard order) to disk.
dataset = load_from_disk(PILOT.owm_dir)
assert len(dataset) == n_examples, len(dataset)

train_dataset = dataset.shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200)
# PILOT: eval datasets are not built here; evaluation runs separately in pilot_eval.py (same metric).
eval_datasets = None

batch_size = PILOT.micro_batch  # PILOT: released: full_batch_size // n_passes_global (= 4); same effective batch of 8
global_gradient_accumulation_steps = full_batch_size // batch_size
assert batch_size * global_gradient_accumulation_steps == full_batch_size
run_id = int(time.time())
training_args = TrainingArguments(
    output_dir=PILOT.out_dir,  # PILOT
    learning_rate=PILOT.lr,  # PILOT: released 1e-6 (= default of --lr)
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=global_gradient_accumulation_steps,
    max_grad_norm=1.0,
    max_steps=100000,
    warmup_steps=20,
    auto_find_batch_size=False,  # PILOT: was True (could silently halve the batch)
    weight_decay=0.001,
    label_names=["labels"],
    include_inputs_for_metrics=True,
    logging_steps=1,  # PILOT: was eval_and_logging_steps (10); per-step loss in the stats, no effect on training
    eval_steps=eval_and_logging_steps,
    evaluation_strategy="no",  # PILOT: was "steps" (eval is 8x the training compute; run separately)
    save_strategy="no",  # PILOT: checkpoints are written explicitly below (bf16 + shadow)
    save_steps=save_steps,
    run_name=f"n={n_ahead_global}_nt={n_ahead_talk_global}_np={n_passes_global}",
    report_to=[],  # PILOT: wandb off
    disable_tqdm=True,  # PILOT: cleaner batch logs
    bf16=PILOT.fp32_master,  # ARM B: accelerate wraps model.forward in torch.autocast(cuda, bfloat16); fp32 params/grads/optimizer states stay fp32
)


# ----------------------------------------------------------------------------------------------
# PILOT: fp32 shadow optimizer + per-step numerics stats
# ----------------------------------------------------------------------------------------------
def classify_param(name):  # PILOT
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


class ArmBStats:  # ARM B: per-step update statistics for the fp32 master weights (the real optimizer IS the fp32 path; no shadow)
    """pre/post step hooks on the real fused AdamW: snapshot the params on-device before the step, then per class
    ||update||, fraction of coordinates that changed, fraction whose bf16 cast changed (= what a bf16-parameter
    run could have expressed this step), ||w||, post-clip grad norm, and (CONTROLS: every step, w_init kept on GPU)
    ||w - w_init|| and the fraction of coordinates whose bf16 cast differs from the bf16 cast of w_init.
    CONTROLS: with --mask_mode the post-step hook first masks the optimizer's write: w = where(mask, w_new, w_old);
    'proposed' = the unmasked optimizer update (incl. decoupled wd), 'applied'/'changed' = what was kept.
    With --gpu_stats the same statistics run on the released bf16 path (T)."""

    def __init__(self, real_opt, named_params, stats_dir, model=None, cum_every=1, cum_steps=()):
        self.real = real_opt
        self.model = model
        self.model_log = None
        self.id2name = {id(p): n for n, p in named_params}
        self.stats_dir = stats_dir
        self.initialized = False
        self.step = 0
        self.records = []
        self.cur_lr = None
        self.grad_norm_postclip = None
        self.grad_norm_cls = None
        self.cum_every = cum_every
        self.cum_steps = set(cum_steps)
        self.t_pre = None
        self.mode = PILOT.mask_mode
        self.arm = PILOT.arm or ("B_fp32_master" if PILOT.fp32_master else "bf16_gpu_stats")
        self.sample_steps = set(int(x) for x in PILOT.sample_steps.split(",") if x.strip())
        self.mask_log = {}

    @torch.no_grad()
    def _init(self):
        t0 = time.time()
        self.params = []
        for gi, g in enumerate(self.real.param_groups):
            for p in g["params"]:
                name = self.id2name[id(p)]
                self.params.append((gi, p, name, classify_param(name)))
                if PILOT.fp32_master:
                    assert p.dtype == torch.float32, (name, p.dtype)
            print(f"PILOT-B group {gi}: n_params={len(g['params'])} wd={g['weight_decay']} betas={g['betas']} eps={g['eps']} fused={g.get('fused')}", flush=True)
        self.name2k = {name: k for k, (_, _, name, _) in enumerate(self.params)}
        self.prev = [torch.empty_like(p) for _, p, _, _ in self.params]       # on the param's GPU
        self.init_dev = [p.detach().clone() for _, p, _, _ in self.params]     # CONTROLS: w_init on the param's GPU (fp32: +29 GB total, ~3.6 GB/GPU)
        cls_numel = defaultdict(int)
        for _, p, _, cls in self.params:
            cls_numel[cls] += p.numel()
        self.cls_numel = dict(cls_numel)
        classes = {name: {"class": cls, "numel": p.numel(), "shape": list(p.shape), "device": str(p.device), "group": gi}
                   for gi, p, name, cls in self.params}
        with open(os.path.join(self.stats_dir, "param_classes.json"), "w") as f:
            json.dump({"classes": classes, "class_numel": self.cls_numel}, f, indent=1)
        # CONTROLS: masks. M2 counts (|w0| < cut) define the density of M2, R and N.
        self.masks = [None] * len(self.params)
        self.applies = [mask_applies(self.mode, cls, p) for _, p, _, cls in self.params]
        if self.mode in ("static_mag", "random", "nanoadam"):
            per_tensor = {}
            for k, (gi, p, name, cls) in enumerate(self.params):
                if not self.applies[k]:
                    continue
                m2 = static_mag_mask(p, PILOT.mag_cut)
                cnt = int(m2.sum())
                if self.mode == "static_mag":
                    self.masks[k] = m2
                elif self.mode == "random":
                    self.masks[k] = random_mask(p, cnt, PILOT.mask_seed + k); del m2
                    assert int(self.masks[k].sum()) == cnt
                else:  # N: authors' block-wise bottom-k at M2's per-tensor density
                    dens = cnt / p.numel(); del m2
                    self.masks[k] = nanoadam_block_mask(p, dens, PILOT.nano_block)
                    nb, db, km, rem, kf = nano_block_counts(p.numel(), dens, PILOT.nano_block)
                    assert int(self.masks[k].sum()) == nb * km + (kf if rem > 0 else 0)
                    cnt = int(self.masks[k].sum())
                per_tensor[name] = {"class": cls, "numel": p.numel(), "k": cnt, "density": cnt / p.numel()}
                if self.mode == "nanoadam":
                    per_tensor[name]["m2_density"] = dens
            self.mask_counts = {name: v["k"] for name, v in per_tensor.items()}
            self.mask_density = {name: v.get("m2_density", v["density"]) for name, v in per_tensor.items()}
            cls_k, cls_n = defaultdict(int), defaultdict(int)
            for v in per_tensor.values():
                cls_k[v["class"]] += v["k"]; cls_n[v["class"]] += v["numel"]
            n_masked = sum(cls_n.values()); k_tot = sum(cls_k.values()); n_all = sum(self.cls_numel.values())
            exempt = {c: n for c, n in self.cls_numel.items() if c not in cls_n}
            self.mask_log = {"mode": self.mode, "mag_cut": PILOT.mag_cut, "mask_seed": PILOT.mask_seed if self.mode == "random" else None,
                             "mask_interval": PILOT.mask_interval if self.mode == "nanoadam" else None, "nano_block": PILOT.nano_block if self.mode == "nanoadam" else None,
                             "nano_rule": "RelationalML/nanoadam@3f8f916 block-wise bottom-k |w| (largest=False), k_block=ceil(block*density); fp32 moments on all coords (not the k-slot bf16 moments); wd masked like M2" if self.mode == "nanoadam" else None,
                             "density_over_masked_tensors": k_tot / n_masked, "trainable_frac_all_params": (k_tot + n_all - n_masked) / n_all,
                             "masked_numel": n_masked, "k_total": k_tot, "exempt_numel_by_class": exempt,
                             "density_by_class": {c: cls_k[c] / cls_n[c] for c in cls_n}, "per_tensor": per_tensor, "recomputes": []}
            with open(os.path.join(self.stats_dir, "masks.json"), "w") as f:
                json.dump(self.mask_log, f, indent=1)
            print(f"PILOT-C masks ({self.mode}): density over masked tensors {k_tot/n_masked:.4%} (k={k_tot:,} of {n_masked:,}); "
                  f"trainable fraction of all params incl. exempt {(k_tot + n_all - n_masked)/n_all:.4%}; by class "
                  + " ".join(f"{c}={cls_k[c]/cls_n[c]:.4%}" for c in sorted(cls_n)) + f"; exempt (unmasked) {exempt}", flush=True)
        # CONTROLS: fixed coordinate sample for the figure data: proportional to numel, >= 1 per tensor
        gen = torch.Generator().manual_seed(4321)
        n_all = sum(self.cls_numel.values())
        self.sample_idx = []
        for k, (gi, p, name, cls) in enumerate(self.params):
            n_k = max(1, int(round(PILOT.n_sample * p.numel() / n_all)))
            self.sample_idx.append(torch.randint(0, p.numel(), (n_k,), generator=gen).to(p.device))
        self.initialized = True
        mem = {k: round(v["alloc_gib"], 1) for k, v in pilot_gpu_mem().items()}
        print(f"PILOT-B stats init ({self.arm}, mask_mode={self.mode}): {len(self.params)} params, {sum(p.numel() for _, p, _, _ in self.params)/1e9:.3f}B, "
              f"class numel={self.cls_numel}, {time.time()-t0:.1f}s; gpu mem (GiB) {mem}", flush=True)

    @torch.no_grad()
    def _nanoadam_recompute(self):  # CONTROLS N: I <- Bottom_k(|theta_t|) per tensor, k fixed (no density decay)
        t0 = time.time()
        kept, tot = 0, 0
        for k, (gi, p, name, cls) in enumerate(self.params):
            if self.masks[k] is None:
                continue
            new = nanoadam_block_mask(p, self.mask_density[name], PILOT.nano_block)
            kept += int((new & self.masks[k]).sum()); tot += int(new.sum())
            self.masks[k] = new
        rec = {"before_step": self.step + 1, "overlap_with_previous": kept / max(tot, 1), "t_s": time.time() - t0}
        self.mask_log["recomputes"].append(rec)
        with open(os.path.join(self.stats_dir, "masks.json"), "w") as f:
            json.dump(self.mask_log, f, indent=1)
        print(f"PILOT-C nanoadam mask recomputed before step {self.step + 1}: overlap with previous mask {rec['overlap_with_previous']:.4%} ({rec['t_s']:.1f}s)", flush=True)

    @torch.no_grad()
    def pre_step(self, opt, args, kwargs):
        if not self.initialized:
            self._init()
        t0 = time.time()
        if self.mode == "nanoadam" and self.step > 0 and (self.step + 1) % PILOT.mask_interval == 0:  # authors: steps==1 or steps % interval == 0 (steps counted incl. the current one)
            self._nanoadam_recompute()
        if self.model is not None:  # snapshot + reset the model's log_dict once per optimizer step (released wandb cadence)
            self.model_log = {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else (v if not torch.is_tensor(v) else None))
                              for k, v in self.model.log_dict.items()}
            self.model.log_dict = defaultdict(int)
        gsq, gcls = 0.0, defaultdict(float)
        for k, (gi, p, name, cls) in enumerate(self.params):
            if p.grad is not None:
                s = float(p.grad.float().square().sum())
                gsq += s; gcls[cls] += s
            self.prev[k].copy_(p)
        self.grad_norm_postclip = math.sqrt(gsq)
        self.grad_norm_cls = {c: math.sqrt(v) for c, v in gcls.items()}
        self.cur_lr = float(opt.param_groups[0]["lr"])
        self.t_pre = time.time() - t0

    @torch.no_grad()
    def post_step(self, opt, args, kwargs):
        self.step += 1
        t0 = time.time()
        do_cum = (self.step % self.cum_every == 0) or (self.step in self.cum_steps) or self.step == 1
        do_sample = self.step in self.sample_steps
        acc = defaultdict(lambda: defaultdict(float))
        smp = defaultdict(list)
        f64 = torch.float64
        for k, (gi, p, name, cls) in enumerate(self.params):
            prev = self.prev[k]
            prop = p - prev                                  # the optimizer's (unmasked) update, incl. decoupled wd
            mask = None
            if self.mode == "bf16vis":
                mask = bf16_visible_mask(p, prev)
            elif self.masks[k] is not None:
                mask = self.masks[k]
            if mask is not None:
                apply_mask_(p, prev, mask)                   # w = where(mask, w_new, w_old)
            d = p - prev                                     # applied update
            if do_sample:
                idx = self.sample_idx[k]
                smp["absw"].append(prev.flatten()[idx].abs().float().cpu())
                smp["absd_prop"].append(prop.flatten()[idx].abs().float().cpu())
                smp["absd_applied"].append(d.flatten()[idx].abs().float().cpu())
                smp["bf16_visible"].append((p.flatten()[idx].to(torch.bfloat16) != prev.flatten()[idx].to(torch.bfloat16)).cpu())
                smp["in_mask"].append((mask.flatten()[idx] if mask is not None else torch.ones_like(idx, dtype=torch.bool)).cpu())
                smp["tensor"].append(torch.full((len(idx),), k, dtype=torch.int16))
            vals = [d.square().sum(dtype=f64), (d != 0).sum(dtype=f64),
                    (p.to(torch.bfloat16) != prev.to(torch.bfloat16)).sum(dtype=f64), p.float().square().sum(dtype=f64),
                    prop.square().sum(dtype=f64), (prop != 0).sum(dtype=f64),
                    (mask.sum(dtype=f64) if mask is not None else torch.tensor(float(p.numel()), dtype=f64, device=p.device))]
            if do_cum:
                w0 = self.init_dev[k]
                vals += [(p - w0).float().square().sum(dtype=f64), (p.to(torch.bfloat16) != w0.to(torch.bfloat16)).sum(dtype=f64)]
            vals = torch.stack(vals).tolist()
            del prop, d
            for c in (cls, "ALL"):
                a = acc[c]
                a["numel"] += p.numel()
                a["sumsq_update"] += vals[0]; a["n_changed"] += vals[1]; a["n_bf16_cast_changed"] += vals[2]; a["sumsq_w"] += vals[3]
                a["sumsq_proposed"] += vals[4]; a["n_proposed_nonzero"] += vals[5]; a["n_mask"] += vals[6]
                if do_cum:
                    a["sumsq_cum"] += vals[7]; a["n_bf16_cast_changed_cum"] += vals[8]
        classes = {}
        for c, a in acc.items():
            n = a["numel"]
            classes[c] = {"numel": int(n), "norm_update": math.sqrt(a["sumsq_update"]), "frac_changed": a["n_changed"] / n,
                          "frac_bf16_cast_changed": a["n_bf16_cast_changed"] / n, "norm_weights": math.sqrt(a["sumsq_w"]),
                          "norm_proposed": math.sqrt(a["sumsq_proposed"]), "frac_proposed_nonzero": a["n_proposed_nonzero"] / n,
                          "frac_dropped": (a["n_proposed_nonzero"] - a["n_changed"]) / n, "frac_in_mask": a["n_mask"] / n,
                          "grad_norm_postclip": self.grad_norm_cls.get(c, self.grad_norm_postclip if c == "ALL" else 0.0)}
            if do_cum:
                classes[c]["norm_cum_delta"] = math.sqrt(a["sumsq_cum"])
                classes[c]["frac_bf16_cast_changed_cum"] = a["n_bf16_cast_changed_cum"] / n
        if do_sample:
            torch.save({k: torch.cat(v) for k, v in smp.items()} | {"tensor_names": [n for _, _, n, _ in self.params],
                        "tensor_classes": [c for _, _, _, c in self.params], "step": self.step, "lr": self.cur_lr, "arm": self.arm},
                       os.path.join(self.stats_dir, f"wsample_step{self.step}.pt"))
        rec = {"arm": self.arm, "mask_mode": self.mode, "step": self.step, "lr": self.cur_lr, "grad_norm_postclip": self.grad_norm_postclip,
               "classes": classes, "t_pre_s": self.t_pre, "t_stats_s": time.time() - t0, "model_log": self.model_log,
               "gpu_mem": pilot_gpu_mem(reset=True)}
        self.records.append(rec)
        self.write()
        A = classes["ALL"]
        print(f"PILOT-B step {self.step} lr={self.cur_lr:.3e} |g|post-clip={self.grad_norm_postclip:.4f} |upd|={A['norm_update']:.3e} "
              f"|proposed|={A['norm_proposed']:.3e} applied/changed={A['frac_changed']:.4%} proposed-nonzero={A['frac_proposed_nonzero']:.2%} "
              f"in-mask={A['frac_in_mask']:.4%} bf16-visible={A['frac_bf16_cast_changed']:.4%} "
              + (f"cum|d|={A['norm_cum_delta']:.3e} cum-bf16-visible={A['frac_bf16_cast_changed_cum']:.4%} " if do_cum else "")
              + f"stats {rec['t_stats_s']:.1f}s pre {self.t_pre:.1f}s", flush=True)
        print("PILOT-B |upd| per class step %d: " % self.step + " ".join(f"{c}={classes[c]['norm_update']:.2e}/{classes[c]['frac_changed']:.3%}" for c in sorted(classes) if c != "ALL"), flush=True)
        print("PILOT gpu mem step %d (GiB, alloc/peak): " % self.step +
              " ".join(f"{k}={v['alloc_gib']:.1f}/{v['max_alloc_gib']:.1f}" for k, v in rec["gpu_mem"].items()), flush=True)
        if self.model_log:
            ml = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.model_log.items() if k in ("loss_0", "loss_final", "loss_talk", "policy_loss", "policy_reward")}
            print("PILOT model log step %d: %s" % (self.step, ml), flush=True)

    def write(self):
        path = os.path.join(self.stats_dir, "stats.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"arm": self.arm, "mask_mode": self.mode, "micro_batch": PILOT.micro_batch, "grad_accum": global_gradient_accumulation_steps,
                       "steps_requested": PILOT.steps, "lr_peak": PILOT.lr, "no_thoughts": PILOT.no_thoughts, "records": self.records}, f, indent=1)
        os.replace(tmp, path)
        with open(os.path.join(self.stats_dir, "stats.jsonl"), "a") as f:
            f.write(json.dumps(self.records[-1]) + "\n")


class ShadowAdamW:  # PILOT
    """fp32 master copy of every trainable parameter + fp32 torch.optim.AdamW on CPU.

    Hooked as a pre/post step hook on the real (bf16, fused) AdamW, so it is stepped with the same
    already-clipped gradients and the same lr the scheduler set for that step. The master copy is
    created lazily at the first optimizer step so the model's lazy in-forward init (thought
    embeddings, zeroed talk head) is captured.
    """

    def __init__(self, real_opt, named_params, stats_dir, sample_per_class=150_000, seed=1234, model=None):
        self.real = real_opt
        self.model = model
        self.model_log = None
        self.id2name = {id(p): n for n, p in named_params}
        self.stats_dir = stats_dir
        self.sample_per_class = sample_per_class
        self.seed = seed
        self.initialized = False
        self.step = 0
        self.records = []
        self.state_dtype_real = None
        self.t_pre = None
        self.cur_lr = None
        self.grad_norm_postclip = None

    @torch.no_grad()
    def _init(self):
        t0 = time.time()
        self.params = []  # (group_idx, p, name, cls)
        for gi, g in enumerate(self.real.param_groups):
            for p in g["params"]:
                name = self.id2name[id(p)]
                self.params.append((gi, p, name, classify_param(name)))
        self.name2k = {name: k for k, (_, _, name, _) in enumerate(self.params)}
        self.master, self.master_prev, self.prev_bf16, self.cur_bf16, self.init_bf16 = [], [], [], [], []
        for gi, p, name, cls in self.params:
            b = p.detach().to("cpu")  # bf16 copy
            m = b.float().requires_grad_(True)  # exact
            self.master.append(m)
            self.master_prev.append(m.detach().clone())
            self.prev_bf16.append(b)
            self.cur_bf16.append(torch.empty_like(b))
            self.init_bf16.append(b.clone())
        groups = []
        for gi, g in enumerate(self.real.param_groups):
            ks = [k for k, (kg, *_rest) in enumerate(self.params) if kg == gi]
            groups.append({"params": [self.master[k] for k in ks], "lr": g["lr"], "weight_decay": g["weight_decay"],
                           "betas": g["betas"], "eps": g["eps"]})
            print(f"PILOT shadow group {gi}: n_params={len(ks)} wd={g['weight_decay']} betas={g['betas']} eps={g['eps']}", flush=True)
        self.sopt = torch.optim.AdamW(groups, foreach=False)
        # fixed random coordinate sample (stratified per class, proportional within class)
        gen = torch.Generator().manual_seed(self.seed)
        cls_numel = defaultdict(int)
        for _, p, _, cls in self.params:
            cls_numel[cls] += p.numel()
        self.sample_idx = {}
        for k, (_, p, name, cls) in enumerate(self.params):
            n_cls = min(self.sample_per_class, cls_numel[cls])
            n_k = max(1, int(round(n_cls * p.numel() / cls_numel[cls])))
            idx = torch.randint(0, p.numel(), (n_k,), generator=gen)
            self.sample_idx[k] = (idx.to(p.device), idx)
        classes = {name: {"class": cls, "numel": p.numel(), "shape": list(p.shape), "device": str(p.device), "group": gi}
                   for gi, p, name, cls in self.params}
        with open(os.path.join(self.stats_dir, "param_classes.json"), "w") as f:
            json.dump({"classes": classes, "class_numel": dict(cls_numel),
                       "n_sample": {c: sum(len(v[1]) for k, v in self.sample_idx.items() if self.params[k][3] == c) for c in cls_numel}}, f, indent=1)
        self.initialized = True
        print(f"PILOT shadow init: {len(self.params)} params, {sum(p.numel() for _, p, _, _ in self.params)/1e9:.3f}B, "
              f"class numel={dict(cls_numel)}, {time.time()-t0:.1f}s", flush=True)

    @torch.no_grad()
    def pre_step(self, opt, args, kwargs):
        if not self.initialized:
            self._init()
        t0 = time.time()
        if self.model is not None:  # PILOT: snapshot + reset the model's log_dict once per optimizer step (released wandb cadence)
            self.model_log = {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else (v if not torch.is_tensor(v) else None))
                              for k, v in self.model.log_dict.items()}
            self.model.log_dict = defaultdict(int)
        gsq = 0.0
        for k, (gi, p, name, cls) in enumerate(self.params):
            m = self.master[k]
            if p.grad is None:
                m.grad = None
                continue
            g = p.grad.detach()
            gsq += float(g.float().pow(2).sum())
            m.grad = g.to("cpu").float()
        for sg, g in zip(self.sopt.param_groups, opt.param_groups):
            sg["lr"] = g["lr"]
        self.cur_lr = float(opt.param_groups[0]["lr"])
        self.grad_norm_postclip = gsq ** 0.5
        self.sopt.step()
        for m in self.master:
            m.grad = None
        self.t_pre = time.time() - t0

    @staticmethod
    def _summarize(a):
        n = a["numel"]
        nz = a["n_update_nonzero"]
        return {
            "numel": int(n),
            "frac_bf16_changed": a["n_changed"] / n,
            "frac_dropped": a["n_dropped"] / n,
            "frac_update_nonzero": nz / n,
            "frac_dropped_given_nonzero_update": (a["n_dropped_given_nonzero"] / nz) if nz else None,
            "norm_delta_bf16": math.sqrt(a["sumsq_bf16"]),
            "norm_delta_fp32": math.sqrt(a["sumsq_fp32"]),
            "norm_ratio_bf16_over_fp32": (math.sqrt(a["sumsq_bf16"]) / math.sqrt(a["sumsq_fp32"])) if a["sumsq_fp32"] > 0 else None,
            "frac_bf16_changed_cum": a["n_changed_cum"] / n,
            "norm_cum_delta_bf16": math.sqrt(a["sumsq_cum_bf16"]),
            "norm_cum_delta_fp32": math.sqrt(a["sumsq_cum_fp32"]),
            "norm_weights_bf16": math.sqrt(a["sumsq_w"]),
        }

    @staticmethod
    def _vstats(vb, vf):
        out = {"n": int(vb.numel())}
        both = (vb > 0) & (vf > 0)
        out["frac_real_zero_shadow_pos"] = float(((vb == 0) & (vf > 0)).float().mean())
        out["frac_shadow_zero"] = float((vf == 0).float().mean())
        if both.any():
            r = (vb[both].double() / vf[both].double())
            lr_ = torch.log10(r)
            q = torch.quantile(r.float(), torch.tensor([0.1, 0.5, 0.9]))
            out.update({"n_both_pos": int(both.sum()), "ratio_p10": float(q[0]), "ratio_median": float(q[1]), "ratio_p90": float(q[2]),
                        "mean_log10_ratio": float(lr_.mean()), "frac_ratio_lt_0.5": float((r < 0.5).float().mean()),
                        "frac_ratio_gt_2": float((r > 2).float().mean()), "mean_real": float(vb[both].double().mean()),
                        "mean_shadow": float(vf[both].double().mean())})
        return out

    @torch.no_grad()
    def post_step(self, opt, args, kwargs):
        t0 = time.time()
        self.step += 1
        acc = defaultdict(lambda: defaultdict(float))
        vs = defaultdict(lambda: {"bf16": [], "fp32": []})
        for k, (gi, p, name, cls) in enumerate(self.params):
            cur = self.cur_bf16[k]
            cur.copy_(p.detach())
            prev = self.prev_bf16[k]
            m = self.master[k].detach()
            mp = self.master_prev[k]
            init = self.init_bf16[k]
            a = acc[cls]
            a["numel"] += cur.numel()
            a["n_changed"] += int(cur.ne(prev).sum())
            d16 = cur.float().sub_(prev.float())
            a["sumsq_bf16"] += float(d16.pow_(2).sum(dtype=torch.float64))
            d32 = m - mp
            nzmask = d32.ne(0)
            a["n_update_nonzero"] += int(nzmask.sum())
            a["sumsq_fp32"] += float(d32.pow(2).sum(dtype=torch.float64))
            # "dropped": applying the shadow (fp32) update to the bf16 weight rounds back to the same bf16 value
            dropped = prev.float().add_(d32).to(torch.bfloat16).eq(prev)
            a["n_dropped"] += int(dropped.sum())
            a["n_dropped_given_nonzero"] += int((dropped & nzmask).sum())
            a["n_changed_cum"] += int(cur.ne(init).sum())
            a["sumsq_cum_bf16"] += float((cur.float() - init.float()).pow_(2).sum(dtype=torch.float64))
            a["sumsq_cum_fp32"] += float((m - init.float()).pow_(2).sum(dtype=torch.float64))
            a["sumsq_w"] += float(cur.float().pow_(2).sum(dtype=torch.float64))
            st = opt.state.get(p)
            sst = self.sopt.state.get(self.master[k])
            if st is not None and "exp_avg_sq" in st and sst is not None and "exp_avg_sq" in sst:
                idx_dev, idx_cpu = self.sample_idx[k]
                vs[cls]["bf16"].append(st["exp_avg_sq"].detach().flatten()[idx_dev].float().cpu())
                vs[cls]["fp32"].append(sst["exp_avg_sq"].detach().flatten()[idx_cpu].float())
                self.state_dtype_real = str(st["exp_avg_sq"].dtype)
            self.prev_bf16[k], self.cur_bf16[k] = cur, prev
            mp.copy_(m)
        rec = {"step": self.step, "lr": self.cur_lr, "grad_norm_postclip": self.grad_norm_postclip,
               "state_dtype_real": self.state_dtype_real, "t_shadow_step_s": self.t_pre, "classes": {}}
        tot = defaultdict(float)
        for cls, a in acc.items():
            rec["classes"][cls] = self._summarize(a)
            for kk, v in a.items():
                tot[kk] += v
        rec["classes"]["ALL"] = self._summarize(tot)
        vsave = {}
        for cls, d in vs.items():
            vb = torch.cat(d["bf16"]); vf = torch.cat(d["fp32"])
            rec["classes"][cls]["exp_avg_sq_sample"] = self._vstats(vb, vf)
            vsave[cls] = {"real_bf16": vb, "shadow_fp32": vf}
        if vsave:
            vb = torch.cat([v["real_bf16"] for v in vsave.values()]); vf = torch.cat([v["shadow_fp32"] for v in vsave.values()])
            rec["classes"]["ALL"]["exp_avg_sq_sample"] = self._vstats(vb, vf)
        rec["t_stats_s"] = time.time() - t0
        rec["model_log"] = self.model_log  # PILOT: the model's own per-step log_dict (loss_0, loss_final, policy_loss, ...)
        rec["gpu_mem"] = pilot_gpu_mem(reset=True)  # PILOT: per-GPU current/peak since the previous step; peaks reset here
        self.records.append(rec)
        self.write()
        if self.step in (1, 2, 3, 5) or self.step % 10 == 0:
            torch.save(vsave, os.path.join(self.stats_dir, f"vsample_step{self.step}.pt"))
        A = rec["classes"]["ALL"]
        print(f"PILOT step {self.step} lr={self.cur_lr:.3e} |g|post-clip={self.grad_norm_postclip:.4f} "
              f"changed={A['frac_bf16_changed']:.4%} dropped={A['frac_dropped']:.4%} "
              f"|d16|={A['norm_delta_bf16']:.3e} |d32|={A['norm_delta_fp32']:.3e} "
              f"shadow {self.t_pre:.1f}s stats {rec['t_stats_s']:.1f}s", flush=True)
        print("PILOT gpu mem step %d (GiB, alloc/peak): " % self.step +
              " ".join(f"{k}={v['alloc_gib']:.1f}/{v['max_alloc_gib']:.1f}" for k, v in rec["gpu_mem"].items()), flush=True)
        if self.model_log:
            ml = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.model_log.items() if k in ("loss_0", "loss_final", "loss_talk", "policy_loss", "policy_reward")}
            print("PILOT model log step %d: %s" % (self.step, ml), flush=True)
        for cls in sorted(rec["classes"]):
            if cls == "ALL":
                continue
            c = rec["classes"][cls]
            print(f"    {cls:12s} changed={c['frac_bf16_changed']:.4%} dropped={c['frac_dropped']:.4%} "
                  f"nonzero_upd={c['frac_update_nonzero']:.3%} |d16|/|d32|={c['norm_ratio_bf16_over_fp32']}", flush=True)

    def write(self):
        path = os.path.join(self.stats_dir, "stats.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"micro_batch": PILOT.micro_batch, "grad_accum": global_gradient_accumulation_steps,
                       "steps_requested": PILOT.steps, "records": self.records}, f, indent=1)
        os.replace(tmp, path)
        with open(os.path.join(self.stats_dir, "stats.jsonl"), "a") as f:
            f.write(json.dumps(self.records[-1]) + "\n")


class PilotTrainer(Trainer):  # PILOT
    shadow = None

    def create_optimizer(self):
        super().create_optimizer()
        if self.shadow is None:
            if USE_B_PATH:  # ARM B (+ CONTROLS: masks via --mask_mode; T via --gpu_stats on the bf16 path)
                self.shadow = ArmBStats(self.optimizer, list(self.model.named_parameters()), STATS_DIR, model=self.model,
                                        cum_steps=PILOT_SAVE_STEPS | {PILOT.steps})
            else:
                self.shadow = ShadowAdamW(self.optimizer, list(self.model.named_parameters()), STATS_DIR,
                                          sample_per_class=PILOT.sample_per_class, model=self.model)
            self.optimizer.register_step_pre_hook(self.shadow.pre_step)
            self.optimizer.register_step_post_hook(self.shadow.post_step)
            self.model.register_forward_hook(pilot_detach_log_dict)  # PILOT: see pilot_detach_log_dict
            print("PILOT: shadow hooks registered on", type(self.optimizer).__name__, self.optimizer.defaults, flush=True)
        return self.optimizer

    def training_step(self, model, inputs):
        if not getattr(self, "_pilot_first_batch_printed", False):  # PILOT: identify the data order (first micro-batch)
            ids = inputs["input_ids"]
            print(f"PILOT first micro-batch: shape={tuple(ids.shape)} first_tokens={ids[:, :6].tolist()} sum={int(ids.sum())} "
                  f"param_dtype={next(model.parameters()).dtype} autocast_wrapped={hasattr(model, '_original_forward')}", flush=True)
            self._pilot_first_batch_printed = True
        return super().training_step(model, inputs)


PILOT_SAVE_STEPS = set(int(x) for x in PILOT.save_steps.split(",") if x.strip()) if USE_B_PATH else set()


def pilot_save_B(model, out_dir, N, fp32=False):  # ARM B: bf16-cast checkpoint (+ fp32 originals when fp32=True)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        sd = model.state_dict()
        bf16_sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in sd.items()}
        d16 = os.path.join(out_dir, f"bf16cast_step{N}")
        model.save_pretrained(d16, state_dict=bf16_sd, safe_serialization=True)
        model.tokenizer.save_pretrained(d16)
        cfg_path = os.path.join(d16, "config.json")
        cfg = json.load(open(cfg_path)); cfg["torch_dtype"] = "bfloat16"; json.dump(cfg, open(cfg_path, "w"), indent=2)
        print(f"PILOT-B: saved {d16} ({time.time()-t0:.0f}s)", flush=True)
        del bf16_sd
        if fp32:
            d32 = os.path.join(out_dir, f"fp32_step{N}")
            model.save_pretrained(d32, safe_serialization=True)
            model.tokenizer.save_pretrained(d32)
            print(f"PILOT-B: saved {d32} ({time.time()-t0:.0f}s)", flush=True)
        del sd
        gc.collect()


def pilot_save(model, sh, out_dir, N):  # PILOT: bf16 model + shadow fp32 + shadow-as-bf16 (+ tokenizer) into out_dir
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        d_bf16 = os.path.join(out_dir, f"bf16_step{N}")
        model.save_pretrained(d_bf16, safe_serialization=True)
        model.tokenizer.save_pretrained(d_bf16)
        print(f"PILOT: saved {d_bf16} ({time.time()-t0:.0f}s)", flush=True)
        sd = model.state_dict()
        fp32_sd, bf16_sd = {}, {}
        for name, t in sd.items():
            if name in sh.name2k:
                m = sh.master[sh.name2k[name]].detach()
                fp32_sd[name] = m
                bf16_sd[name] = m.to(torch.bfloat16)
            else:
                fp32_sd[name] = t.detach().cpu()
                bf16_sd[name] = t.detach().cpu()
        d_sh32 = os.path.join(out_dir, f"shadow_fp32_step{N}")
        model.save_pretrained(d_sh32, state_dict=fp32_sd, safe_serialization=True)
        model.tokenizer.save_pretrained(d_sh32)
        print(f"PILOT: saved {d_sh32} ({time.time()-t0:.0f}s)", flush=True)
        d_sh16 = os.path.join(out_dir, f"shadow_bf16_step{N}")
        model.save_pretrained(d_sh16, state_dict=bf16_sd, safe_serialization=True)
        model.tokenizer.save_pretrained(d_sh16)
        print(f"PILOT: saved {d_sh16} ({time.time()-t0:.0f}s)", flush=True)
        n_diff = 0; n_tot = 0
        for name in fp32_sd:
            if name in sh.name2k:
                k = sh.name2k[name]
                n_diff += int(sh.prev_bf16[k].ne(bf16_sd[name]).sum()); n_tot += bf16_sd[name].numel()
        summ = {"steps": N, "micro_batch": PILOT.micro_batch, "grad_accum": global_gradient_accumulation_steps,
                "frac_coords_bf16_trained_ne_shadow_bf16": n_diff / n_tot, "n_coords": n_tot,
                "dirs": {"bf16": d_bf16, "shadow_fp32": d_sh32, "shadow_bf16": d_sh16}}
        with open(os.path.join(STATS_DIR, f"final_summary_step{N}.json"), "w") as f:
            json.dump(summ, f, indent=1)
        print(f"PILOT: step {N}: bf16-trained vs shadow-as-bf16 differ on {n_diff/n_tot:.4%} of coordinates", flush=True)
        del sd, fp32_sd, bf16_sd
        gc.collect()
    return summ


class PilotCallback(TrainerCallback):  # PILOT: clean stop at kill_after + attach trainer logs to the step record
    def __init__(self, shadow_getter):
        self.shadow_getter = shadow_getter

    def on_step_end(self, args, state, control, **kwargs):
        if USE_B_PATH:  # ARM B: bf16-cast saves at --save_steps (the final step is saved by the end-of-run code)
            if state.global_step in PILOT_SAVE_STEPS and state.global_step < PILOT.steps and not PILOT.no_save:
                try:
                    pilot_save_B(kwargs["model"] if "model" in kwargs else trainer.model, PILOT.out_dir, state.global_step, fp32=False)
                except Exception as ex:
                    print("PILOT-B: mid-run save FAILED:", repr(ex), flush=True)
            if state.global_step >= PILOT.steps:
                control.should_training_stop = True
            return control
        if PILOT.save_at and state.global_step == PILOT.save_at and not PILOT.no_save:  # PILOT: mid-run safety save
            sh = self.shadow_getter()
            try:
                pilot_save(kwargs["model"] if "model" in kwargs else trainer.model, sh, os.path.join(PILOT.out_dir, f"step{PILOT.save_at}"), sh.step)
            except Exception as ex:
                print("PILOT: mid-run save FAILED:", repr(ex), flush=True)
        if state.global_step >= PILOT.steps:
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        sh = self.shadow_getter()
        if sh is not None and sh.records and logs:
            sh.records[-1].setdefault("trainer_log", {}).update({k: v for k, v in logs.items()})
            sh.records[-1]["global_step"] = state.global_step
            sh.write()
        return control


trainer = PilotTrainer(  # PILOT: was Trainer(...)
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    compute_metrics=compute_metrics,
    model_init=model_init,
)
trainer.add_callback(PilotCallback(lambda: trainer.shadow))  # PILOT

t_train0 = time.time()
if PILOT.profile_probe:  # PILOT: diagnostics only — name the ops behind the memory peak, then exit
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            pilot_oom_report(e)
        except ValueError as e:
            print("PILOT probe ValueError:", str(e)[:200])
    print("PILOT PROFILE (top self CUDA memory, grouped by input shape):", flush=True)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=30,
                                                             max_name_column_width=70, max_shapes_column_width=110), flush=True)
    print("PILOT_PROBE_EXIT", flush=True)
    sys.exit(3)
try:
    trainer.train()
except ValueError as e:  # PILOT: modeling code raises ValueError("Killed after") at kill_after
    if "Killed after" not in str(e):
        raise
    print("PILOT: caught kill_after ValueError", flush=True)
except torch.cuda.OutOfMemoryError as e:  # PILOT: diagnostics, then fail
    pilot_oom_report(e)
    print("PILOT_OOM_EXIT", flush=True)
    sys.exit(3)
print(f"PILOT: training finished in {time.time()-t_train0:.0f}s, shadow steps={trainer.shadow.step if trainer.shadow else None}", flush=True)

# PILOT: save bf16 model, shadow fp32 and shadow-as-bf16  (ARM B: bf16-cast + fp32 originals at the final step)
if not PILOT.no_save:
    if USE_B_PATH:
        pilot_save_B(trainer.model, PILOT.out_dir, trainer.shadow.step, fp32=PILOT.fp32_master and not PILOT.no_final_fp32)
        with open(os.path.join(STATS_DIR, "final_summary.json"), "w") as f:
            json.dump({"arm": trainer.shadow.arm, "mask_mode": PILOT.mask_mode, "mask_summary": {k: v for k, v in trainer.shadow.mask_log.items() if k != "per_tensor"}, "steps": trainer.shadow.step, "micro_batch": PILOT.micro_batch, "grad_accum": global_gradient_accumulation_steps,
                       "last_record": trainer.shadow.records[-1] if trainer.shadow.records else None}, f, indent=1)
    else:
        summ = pilot_save(trainer.model, trainer.shadow, PILOT.out_dir, trainer.shadow.step)
        with open(os.path.join(STATS_DIR, "final_summary.json"), "w") as f:
            json.dump(summ, f, indent=1)
print("PILOT_TRAIN_DONE", flush=True)
