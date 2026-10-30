# PILOT: held-out OpenWebMath next-token perplexity for several checkpoints, in two modes:
#   OFF: thoughts off = model.original_mode=True with n_ahead=n_ahead_talk=n_passes=1 -> fwd_iters=1: a single pass of
#        the decoder on the plain token embeddings, logits = lm_head(hidden); no thought tokens, talk head never called,
#        2-D attention mask (the stock Mistral forward of this modeling file).
#   ON:  the harness path (original_mode=False, n_ahead=12; the forward itself forces n_ahead_talk=1, n_passes=1 in
#        eval): returned logits are the MIXED logits lm_head((1-w)*base_hidden + w*talk_hidden) of the last iteration
#        (output_logits_at_the_end=True, no_residual=True). A hook on talk_head[0] additionally gives, from the same
#        forward, CE of lm_head(base_hidden) and lm_head(talk_hidden) and the statistics of the mixing weight w.
# NLL is the token mean over non-pad target positions (the released code masks pad=eos targets the same way).
import torch
torch.backends.cuda.matmul.allow_tf32 = False  # fp32 checkpoints are evaluated in true fp32 (TF32 would round to 10-bit mantissas)
torch.backends.cudnn.allow_tf32 = False
import os, sys, json, time, argparse, gc, math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from eval_helpers import initial_tokenizer

QSTAR = os.environ.get("QSTAR_ROOT", "/scratch/mweiss10/qstar")
MISTRAL_REV = "26bca36bde8333b5d7f72e9ed20ccda6a618af24"
ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="comma list of label=path[:fp32]; path 'base' = pinned Hub base")
ap.add_argument("--owm_dir", default=f"{QSTAR}/data/owm_heldout_1000_1300")
ap.add_argument("--n_seq", type=int, default=200)
ap.add_argument("--seq_len", type=int, default=256)
ap.add_argument("--min_tokens", type=int, default=64)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--modes", default="off,on")
ap.add_argument("--out", required=True)
ap.add_argument("--vocab_limit", type=int, default=0, help="score only logits[..., :N] (N=32000 drops the two rows added by resize_token_embeddings from the softmax; the released plain path never masks them); 0 = full vocab. Both numbers are always recorded.")
ap.add_argument("--seed_resize", type=int, default=None, help="torch.manual_seed(S) right before resize_token_embeddings, i.e. controls the N(0, 0.02) draw of the two added rows (only matters for 'base'; checkpoints already have 32002 rows)")
args = ap.parse_args()
VL = args.vocab_limit or None
N_REAL = 32000
PAD = initial_tokenizer.pad_token_id  # = eos (2), as in the released preprocessing

# ---- data: first n_seq held-out rows with >= min_tokens tokens, first seq_len tokens (BOS included), right-padded ----
rows = load_from_disk(args.owm_dir)
ids_list, src_idx = [], []
for r in rows:
    ids = initial_tokenizer.encode(r["text"])
    if len(ids) < args.min_tokens:
        continue
    ids = ids[:args.seq_len]
    ids_list.append(ids + [PAD] * (args.seq_len - len(ids)))
    src_idx.append(int(r.get("stream_index", -1)))
    if len(ids_list) >= args.n_seq:
        break
input_ids = torch.tensor(ids_list)
attn = (input_ids != PAD).long()
targets = input_ids[:, 1:].clone()
targets[targets == PAD] = -100
n_tok = int((targets != -100).sum())
print(f"PILOT_PPL data: {len(ids_list)} sequences x {args.seq_len} from {args.owm_dir} (stream rows {min(src_idx)}..{max(src_idx)}), {n_tok} scored tokens", flush=True)


def load_model(path, dtype):
    gc.collect(); torch.cuda.empty_cache()
    src = "mistralai/Mistral-7B-v0.1" if path == "base" else path
    model = AutoModelForCausalLM.from_pretrained(
        src, revision=MISTRAL_REV if path == "base" else None, torch_dtype=dtype, device_map="auto",
        max_thoughts=12 + 4 + 1, merged_talk_heads=True, merged_lm_and_talk_heads=False, merged_lm_and_think_heads=True,
        use_concat_talk_head=True, use_shallow_think=True, use_shallow_talk=False, use_complex_think_head=False,
        use_complex_talk_head=True, use_weighted_talk_head=True)
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision=MISTRAL_REV)
    tok.padding_side = "right"; tok.pad_token_id = tok.eos_token_id
    tok.add_special_tokens({"additional_special_tokens": ["<|startthought|>", "<|endthought|>"]})
    if args.seed_resize is not None:
        torch.manual_seed(args.seed_resize)
    model.resize_token_embeddings(len(tok))
    model.tokenizer = tok
    model.gumbel_detach = True; model.include_policy_loss = True
    model.use_end_thought_token = True; model.use_start_thought_token = True
    model.n_tokens_print = 2; model.gradient_accumulation_steps = 2
    model.residual_think_head = False; model.optimize_lm_head_only_at_start = False
    model.gumbel_temperature = 1; model.wandb_enabled = False; model.config_params = {}
    model.run_start = int(time.time()); model.kill_after = 100
    if path != "base":
        model.rm_initialized = True
        model.start_token_id = tok.convert_tokens_to_ids("<|startthought|>")
        model.end_token_id = tok.convert_tokens_to_ids("<|endthought|>")
    model.eval()
    return model


def set_mode(model, mode):
    if mode == "off":
        model.original_mode = True; model.n_ahead = 1; model.n_ahead_talk = 1; model.n_passes = 1
    else:
        model.original_mode = False; model.n_ahead = 12; model.n_ahead_talk = 4; model.n_passes = 2


def ce_sum(logits, tgt, V=None):
    lg = (logits[:, :-1, :V] if V else logits[:, :-1]).float()
    return float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1).to(lg.device), ignore_index=-100, reduction="sum")), \
        [float(F.cross_entropy(lg[b], tgt[b].to(lg.device), ignore_index=-100, reduction="sum")) for b in range(lg.shape[0])]


def extra_row_stats(logits, mask, acc):  # logits [B,T,V>32000]; mask [B,T-1] = scored positions (next token is a real token)
    if logits.shape[-1] <= N_REAL:
        return
    lg = logits[:, :-1].float()
    valid = mask.bool().to(lg.device)
    real_max = lg[..., :N_REAL].max(-1).values
    for r in range(N_REAL, lg.shape[-1]):
        rel = (lg[..., r] - real_max)[valid]
        a = acc.setdefault(r, {"sum": 0.0, "max": float("-inf"), "n": 0})
        a["sum"] += float(rel.sum()); a["max"] = max(a["max"], float(rel.max())); a["n"] += int(rel.numel())
    mass = torch.softmax(lg, -1)[..., N_REAL:].sum(-1)[valid]
    a = acc.setdefault("mass", {"sum": 0.0, "max": float("-inf"), "n": 0})
    a["sum"] += float(mass.sum()); a["max"] = max(a["max"], float(mass.max())); a["n"] += int(mass.numel())


HOOK = {"tgt": None, "acc": None}


def talk_hook(module, inputs, output):  # inputs[0] = cat(base_hidden, talk_hidden) [B,T,2H]; output = w [B,T,1]
    x = inputs[0]; H = x.shape[-1] // 2
    w = output.detach().float()
    acc = HOOK["acc"]
    acc["w_sum"] += float(w.sum()); acc["w_sq"] += float((w * w).sum()); acc["w_n"] += w.numel()
    acc["w_min"] = min(acc["w_min"], float(w.min())); acc["w_max"] = max(acc["w_max"], float(w.max()))
    acc["w_gt1"] += int((w > 1).sum()); acc["w_lt0"] += int((w < 0).sum())
    lm = module_lm[0]
    with torch.no_grad():
        for name, h in (("base", x[..., :H]), ("talk", x[..., H:])):
            lg = lm(h.to(lm.weight.device).to(lm.weight.dtype))
            s, per = ce_sum(lg, HOOK["tgt"], VL)
            acc[f"ce_{name}"] += s; acc[f"per_{name}"].extend(per)
            if VL:
                acc[f"ce_{name}_full"] += ce_sum(lg, HOOK["tgt"], None)[0]


module_lm = [None]
results = {"owm_dir": args.owm_dir, "n_seq": len(ids_list), "seq_len": args.seq_len, "n_tokens": n_tok, "vocab_limit": VL, "seed_resize": args.seed_resize, "models": {}}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
for spec in args.models.split(","):
    label, path = spec.split("=", 1)
    dtype = torch.float32 if path.endswith(":fp32") else torch.bfloat16
    path = path[:-5] if path.endswith(":fp32") else path
    t0 = time.time()
    model = load_model(path, dtype)
    module_lm[0] = model.lm_head
    dev = model.get_input_embeddings().weight.device
    print(f"PILOT_PPL loaded {label} ({path}, {dtype}) in {time.time()-t0:.0f}s; gpu mem {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    res = {"path": path, "dtype": str(dtype), "vocab_limit": VL, "seed_resize": args.seed_resize}
    with torch.no_grad():  # the two rows added by resize_token_embeddings (eval-time draw for 'base', training-time draw for checkpoints)
        W = model.lm_head.weight.float(); E = model.get_input_embeddings().weight.float()
        res["extra_rows"] = {"lm_head_row_norms": W[N_REAL:].norm(dim=1).tolist(), "lm_head_real_row_norm_mean": float(W[:N_REAL].norm(dim=1).mean()),
                             "lm_head_real_row_norm_max": float(W[:N_REAL].norm(dim=1).max()),
                             "embed_row_norms": E[N_REAL:].norm(dim=1).tolist(), "embed_real_row_norm_mean": float(E[:N_REAL].norm(dim=1).mean()), "vocab": int(W.shape[0])}
        del W, E
    for mode in args.modes.split(","):
        set_mode(model, mode)
        torch.manual_seed(42)
        acc = {"w_sum": 0.0, "w_sq": 0.0, "w_n": 0, "w_min": float("inf"), "w_max": float("-inf"), "w_gt1": 0, "w_lt0": 0,
               "ce_base": 0.0, "ce_talk": 0.0, "ce_base_full": 0.0, "ce_talk_full": 0.0, "per_base": [], "per_talk": []}
        HOOK["acc"] = acc
        handle = model.talk_head[0].register_forward_hook(talk_hook) if mode == "on" else None
        ce_tot, per_seq, model_loss, ce_full, xacc = 0.0, [], [], 0.0, {}
        t1 = time.time()
        with torch.no_grad():
            for b in range(0, len(ids_list), args.batch):
                x = input_ids[b:b + args.batch].to(dev); m = attn[b:b + args.batch].to(dev)
                HOOK["tgt"] = targets[b:b + args.batch]
                if mode == "on":
                    out = model(input_ids=x, attention_mask=m, labels=x.clone())  # labels as in the harness (pads masked inside)
                    model_loss.append(float(out.loss) if out.loss is not None else float("nan"))
                else:
                    out = model(input_ids=x, attention_mask=m)
                s, per = ce_sum(out.logits, targets[b:b + args.batch], VL)
                ce_tot += s; per_seq.extend(per)
                if VL:
                    ce_full += ce_sum(out.logits, targets[b:b + args.batch], None)[0]
                extra_row_stats(out.logits, m[:, :-1] * (targets[b:b + args.batch] != -100).to(m.device), xacc)
        if handle is not None:
            handle.remove()
        r = {"nll": ce_tot / n_tok, "ppl": math.exp(ce_tot / n_tok), "n_tokens": n_tok, "runtime_s": time.time() - t1, "vocab_scored": VL or "full",
             "per_seq_nll_sum": per_seq}
        if VL:
            r["nll_full"] = ce_full / n_tok; r["ppl_full"] = math.exp(ce_full / n_tok)
        if xacc:
            r["extra_rows_logits"] = {("row%d" % k if isinstance(k, int) else k): {"mean": v["sum"] / max(v["n"], 1), "max": v["max"], "n_positions": v["n"]} for k, v in xacc.items()}
            r["extra_rows_logits"]["note"] = "rowN = logit[N] - max real-token logit at scored positions; mass = softmax mass on the added rows"
        if mode == "on":
            n = max(acc["w_n"], 1)
            mean = acc["w_sum"] / n
            r.update({"nll_base_hidden": acc["ce_base"] / n_tok, "ppl_base_hidden": math.exp(acc["ce_base"] / n_tok),
                      "nll_talk_hidden": acc["ce_talk"] / n_tok, "ppl_talk_hidden": math.exp(acc["ce_talk"] / n_tok),
                      "w_mean": mean, "w_std": math.sqrt(max(acc["w_sq"] / n - mean * mean, 0.0)), "w_min": acc["w_min"], "w_max": acc["w_max"],
                      "w_frac_gt1": acc["w_gt1"] / n, "w_frac_lt0": acc["w_lt0"] / n, "w_positions": acc["w_n"],
                      **({"nll_base_hidden_full": acc["ce_base_full"] / n_tok, "nll_talk_hidden_full": acc["ce_talk_full"] / n_tok} if VL else {}),
                      "model_loss_mean": sum(model_loss) / len(model_loss) if model_loss else None,
                      "per_seq_nll_sum_base_hidden": acc["per_base"], "per_seq_nll_sum_talk_hidden": acc["per_talk"]})
        res[mode] = r
        summ = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("per_seq") and k != "extra_rows_logits"}
        if "extra_rows_logits" in r:
            x = r["extra_rows_logits"]
            summ["extra_rows"] = {k: (round(v["mean"], 3), round(v["max"], 3)) for k, v in x.items() if k != "note"}
            summ["extra_rows"]["lm_head_row_norms"] = [round(v, 3) for v in res["extra_rows"]["lm_head_row_norms"]]
        print("PILOT_PPL", json.dumps({"model": label, "mode": mode, "seed_resize": args.seed_resize, **summ}), flush=True)
    results["models"][label] = res
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    del model; module_lm[0] = None
    gc.collect(); torch.cuda.empty_cache()
print("PILOT_PPL_DONE", args.out, flush=True)
