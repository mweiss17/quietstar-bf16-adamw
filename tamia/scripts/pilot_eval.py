# PILOT: zero-shot GSM8K + CSQA evaluation with the paper's own metric (eval_helpers.compute_metrics),
# PILOT: run through Trainer.evaluate() exactly like the released training script, with two memory fixes:
# PILOT:  (1) preprocess_logits_for_metrics reduces each batch's [B,T,32002] logits to the 16 "valid" answer-token
# PILOT:      ids in fp32 (the Trainer would otherwise accumulate 2,540 x 256 x 32002 logits);
# PILOT:  (2) compute_metrics indexes into that reduced tensor. With filter_numbers=True the released metric
# PILOT:      renormalises the softmax over the valid tokens, so softmax over the valid-token logits is identical.
import torch
import random, os, sys, time, json, argparse, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from huggingface_custom_callback import EarlyStoppingCallback  # stub (keeps import parity with the training script)
import eval_helpers
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, initial_tokenizer, eval_answer_marker

QSTAR = os.environ.get("QSTAR_ROOT", "/scratch/mweiss10/qstar")  # PILOT: Vulcan default; tamIA uses /scratch/m/mweiss10/qstar
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="'base' or a checkpoint directory")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0, help="first N items of each eval set (0 = all)")
ap.add_argument("--eval_batch", type=int, default=8)
ap.add_argument("--owm_dir", default=f"{QSTAR}/data/owm_first1000")
ap.add_argument("--tag", default="")
ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "16")))
ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="PILOT: fp32 loads the checkpoint in fp32 and disables TF32")
ap.add_argument("--zero_talk_head", action="store_true", help="PILOT: mixing forced to the base path: rm_initialized=False so the model's own lazy init zeroes the last talk-head layer (w=0) on the first forward; start/end ids stay set so the thought embeddings are NOT re-derived")
ap.add_argument("--label", default="", help="PILOT: name used in the summary line (default: --model)")
args = ap.parse_args()
torch.set_num_threads(args.threads)
if args.dtype == "fp32":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
else:
    torch.backends.cuda.matmul.allow_tf32 = True  # as before (bf16 matmuls are unaffected either way)
os.environ["WANDB_MODE"] = "disabled"
MISTRAL_REV = "26bca36bde8333b5d7f72e9ed20ccda6a618af24"  # PILOT: pre-July-2024 Hub revision (tokenizer.json parsable by tokenizers 0.15; identical weights)

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

n_ahead_talk_global = 4
n_passes_global = 2
n_ahead_global = 12
full_batch_size = 8

# PILOT: replicate the released script's data-prep order so the Python RNG state at eval-set construction matches
# (train crops consume `random` before the eval maps; only eval prompts > 256 tokens are affected).
_train = load_from_disk(args.owm_dir).shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200)
eval_dataset_gsm = load_dataset("gsm8k", "main", split="test", ignore_verifications=True).map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200)
eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split="validation", ignore_verifications=True).map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200)
if args.limit:
    eval_dataset_gsm = eval_dataset_gsm.select(range(args.limit))
    eval_dataset_csqa = eval_dataset_csqa.select(range(args.limit))
eval_datasets = {"gsm8k": eval_dataset_gsm, "csqa": eval_dataset_csqa}

# ---- metric with reduced logits + diagnostics: pilot_metrics.py ----------------------------------------------
from pilot_metrics import reduce_logits, compute_metrics_reduced, REDUCED_IDS  # PILOT


# ---- model ----------------------------------------------------------------------------------------------------
def load_model(which):
    gc.collect(); torch.cuda.empty_cache()
    n_ahead, n_ahead_talk, n_passes = n_ahead_global, n_ahead_talk_global, n_passes_global
    src = "mistralai/Mistral-7B-v0.1" if which == "base" else which
    print("Loading model from", src, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        src,
        revision=MISTRAL_REV if which == "base" else None,  # PILOT
        torch_dtype=(torch.float32 if args.dtype == "fp32" else torch.bfloat16) if torch.cuda.is_available() else torch.float32,  # PILOT: --dtype
        device_map='auto',
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
    )
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
        model.resize_token_embeddings(len(tokenizer))  # no-op for a saved checkpoint (already 32002 rows)
    model.tokenizer = tokenizer
    model.gumbel_detach = True
    model.include_policy_loss = True
    model.use_end_thought_token = True
    model.use_start_thought_token = True
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = full_batch_size // 4
    model.gradient_accumulation_steps = full_batch_size // 4
    model.residual_think_head = False
    model.optimize_lm_head_only_at_start = False
    model.gumbel_temperature = 1
    model.wandb_enabled = False
    model.original_mode = False
    model.config_params = {}
    model.run_start = int(time.time())
    model.kill_after = 100
    if which != "base":
        # PILOT: a trained checkpoint must keep its trained talk head and thought embeddings. On a fresh load the
        # first forward would (a) zero the talk head (rm_initialized=False) and (b) re-derive start/end_embedding
        # from the "---" token row (start_token_id is None). This mirrors the live model state during the released
        # script's in-training evaluation (same model object, already initialised).
        model.rm_initialized = not args.zero_talk_head  # PILOT: --zero_talk_head -> the model zeroes talk_head[0][-1].weight itself
        model.start_token_id = tokenizer.convert_tokens_to_ids("<|startthought|>")
        model.end_token_id = tokenizer.convert_tokens_to_ids("<|endthought|>")
        assert model.start_token_id != 0 and model.end_token_id != 0
    model.eval()
    print(f"PILOT eval load: dtype={next(model.parameters()).dtype} rm_initialized={model.rm_initialized} tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"gpu_mem={torch.cuda.memory_allocated()/2**30:.1f}GiB talk_head_last_absmax={model.talk_head[0][-1].weight.float().abs().max().item():.3e}", flush=True)
    return model


W_ACC = {}  # PILOT: mixing-weight statistics from a hook on talk_head[0] (fires once per forward, in the talk iteration)


def _w_reset():
    W_ACC.update({"sum": 0.0, "sq": 0.0, "n": 0, "min": float("inf"), "max": float("-inf"), "gt1": 0, "lt0": 0})


def _w_hook(module, inputs, output):
    w = output.detach().float()
    W_ACC["sum"] += float(w.sum()); W_ACC["sq"] += float((w * w).sum()); W_ACC["n"] += w.numel()
    W_ACC["min"] = min(W_ACC["min"], float(w.min())); W_ACC["max"] = max(W_ACC["max"], float(w.max()))
    W_ACC["gt1"] += int((w > 1).sum()); W_ACC["lt0"] += int((w < 0).sum())


def _w_summary():
    n = max(W_ACC["n"], 1); m = W_ACC["sum"] / n
    return {"w_mean": m, "w_std": (max(W_ACC["sq"] / n - m * m, 0.0)) ** 0.5, "w_min": W_ACC["min"], "w_max": W_ACC["max"],
            "w_frac_gt1": W_ACC["gt1"] / n, "w_frac_lt0": W_ACC["lt0"] / n, "w_positions": W_ACC["n"]}


model = load_model(args.model)
model.talk_head[0].register_forward_hook(_w_hook)  # PILOT
LABEL = args.label or args.model
targs = TrainingArguments(
    output_dir=f"{QSTAR}/ckpt/_eval_tmp",
    per_device_eval_batch_size=args.eval_batch,
    label_names=["labels"],
    include_inputs_for_metrics=True,
    report_to=[],
    disable_tqdm=True,
    dataloader_pin_memory=False,
)
trainer = Trainer(model=model, args=targs, compute_metrics=compute_metrics_reduced,
                  preprocess_logits_for_metrics=reduce_logits)

results = {"model": args.model, "label": LABEL, "tag": args.tag, "limit": args.limit, "eval_batch": args.eval_batch, "dtype": args.dtype,
           "zero_talk_head": args.zero_talk_head, "n_ahead": n_ahead_global, "n_ahead_talk_eval": 1,
           "metric": "eval_helpers.compute_metrics (filter_numbers=True) via reduced logits; soft = product of renormalised valid-token probs, hard = restricted argmax at every answer token"}
summary = {"model": LABEL, "dtype": args.dtype, "zero_talk_head": args.zero_talk_head, "limit": args.limit}
for name, ds in eval_datasets.items():
    t0 = time.time()
    _w_reset()
    m = trainer.evaluate(eval_dataset=ds, metric_key_prefix=name)
    per_item = m.pop(f"{name}_per_item", None)
    per_item_hard = m.pop(f"{name}_per_item_hard", None)
    top5 = m.pop(f"{name}_top5_first_items", None)
    m["runtime_s"] = time.time() - t0
    diag = {k[len(name) + 1:]: v for k, v in m.items() if k.startswith(name + "_") and not isinstance(v, (list,))}
    diag["mixing_w"] = _w_summary()
    results[name] = {"accuracy": m[f"{name}_accuracy"], "hard_accuracy": m.get(f"{name}_hard_accuracy"), "n": m.get(f"{name}_n_items"),
                     "diag": diag, "top5_first_items": top5, "per_item": per_item, "per_item_hard": per_item_hard}
    summary[name] = m[f"{name}_accuracy"]; summary[name + "_hard"] = m.get(f"{name}_hard_accuracy")
    summary[name + "_newline_wins"] = m.get(f"{name}_first_pos_newline_wins_frac"); summary[name + "_w_mean"] = diag["mixing_w"]["w_mean"]
    print(f"PILOT eval {LABEL} {name}: accuracy={m[f'{name}_accuracy']:.4f} hard={m.get(f'{name}_hard_accuracy'):.4f} n={m.get(f'{name}_n_items')} ({m['runtime_s']:.0f}s)", flush=True)
    print(f"PILOT eval {LABEL} {name} diag:", json.dumps({k: v for k, v in diag.items()}, default=str)[:3000], flush=True)
    if top5:
        for item in top5[:20]:
            print(f"PILOT eval {LABEL} {name} top5 item {item['item']} true={item['true_str']!r}: {item['top']} restricted={item['restricted']}", flush=True)
print("PILOT_EVAL_SUMMARY", json.dumps(summary), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print("PILOT_EVAL_DONE", args.out, flush=True)
