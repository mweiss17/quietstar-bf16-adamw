# ARM B pre-flight (CPU, login node, no GPU): run the released Quiet-STaR forward+backward of a TINY random Mistral
# (2 layers, hidden 256) with fp32 parameters under torch.autocast(bf16), with and without the SDPA attn_mask dtype
# patch, to exercise the dtype interplay (fp32 embeds / bf16 Linear outputs / 4-D mask / torch.cat / eye-mask
# extension / gumbel / talk head). CPU autocast's op coverage differs from CUDA's, so this is a control-flow and
# dtype-compatibility check, not a numerical one.
import os, sys, time, math
import torch
from transformers import AutoConfig, AutoTokenizer, MistralForCausalLM

QSTAR = os.environ.get("QSTAR_ROOT", "/scratch/m/mweiss10/qstar")
REV = "26bca36bde8333b5d7f72e9ed20ccda6a618af24"
PATCH = "--patch" in sys.argv

if PATCH:
    _sdpa_orig = torch.nn.functional.scaled_dot_product_attention

    def _sdpa_cast_mask(query, key, value, attn_mask=None, *a, **kw):
        if attn_mask is not None and attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.clamp(min=torch.finfo(query.dtype).min).to(query.dtype)
        return _sdpa_orig(query, key, value, attn_mask, *a, **kw)
    torch.nn.functional.scaled_dot_product_attention = _sdpa_cast_mask

torch.manual_seed(0)
n_ahead, n_ahead_talk, n_passes = 12, 4, 2
cfg = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", revision=REV,
                                 max_thoughts=n_ahead + n_ahead_talk + 1, merged_talk_heads=True, merged_lm_and_talk_heads=False,
                                 merged_lm_and_think_heads=True, use_concat_talk_head=True, use_shallow_think=True, use_shallow_talk=False,
                                 use_complex_think_head=False, use_complex_talk_head=True, use_weighted_talk_head=True)
cfg.num_hidden_layers = 2; cfg.hidden_size = 256; cfg.intermediate_size = 512; cfg.num_attention_heads = 8; cfg.num_key_value_heads = 2
model = MistralForCausalLM(cfg)  # fp32, random
print("attn implementation:", cfg._attn_implementation, "| param dtype:", next(model.parameters()).dtype, flush=True)
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision=REV)
tok.padding_side = "right"; tok.pad_token_id = tok.eos_token_id
tok.add_special_tokens({"additional_special_tokens": ["<|startthought|>", "<|endthought|>"]})
model.resize_token_embeddings(len(tok))
model.tokenizer = tok
model.gumbel_detach = True; model.include_policy_loss = True; model.use_end_thought_token = True; model.use_start_thought_token = True
model.n_ahead = n_ahead; model.n_ahead_talk = n_ahead_talk; model.n_passes = n_passes
model.n_tokens_print = 2; model.gradient_accumulation_steps = 2; model.residual_think_head = False
model.optimize_lm_head_only_at_start = False; model.gumbel_temperature = 1; model.wandb_enabled = False; model.original_mode = False
model.config_params = {}; model.run_start = int(time.time()); model.kill_after = 100
model.train()
texts = ["The derivative of x^2 is 2x, so the slope at x=3 is 6.", "Let n be an integer. Then n(n+1) is even because"]
enc = tok(texts, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
ids, mask = enc["input_ids"], enc["attention_mask"]
t0 = time.time()
try:
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=mask, labels=ids.clone())
    print(f"forward OK ({time.time()-t0:.1f}s): loss={float(out.loss):.4f} loss.dtype={out.loss.dtype} logits.dtype={out.logits.dtype} finite={bool(torch.isfinite(out.loss))}", flush=True)
    out.loss.backward()
    gn = math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None))
    n_nan = sum(int((~torch.isfinite(p.grad)).sum()) for p in model.parameters() if p.grad is not None)
    print(f"backward OK: grad norm={gn:.4e} non-finite grad entries={n_nan} grad dtype={model.model.layers[0].mlp.up_proj.weight.grad.dtype}", flush=True)
    th = model.talk_head[0][-1].weight.grad
    print("talk_head last-layer grad norm:", float(th.norm()) if th is not None else None, "| start_embedding grad norm:", float(model.start_embedding.grad.norm()) if model.start_embedding.grad is not None else None, flush=True)
    print("log_dict keys:", sorted(model.log_dict.keys())[:12], flush=True)
    print("ARMB_CPU_SMOKE_OK patch=%s" % PATCH, flush=True)
except Exception as e:
    print("ARMB_CPU_SMOKE_FAIL patch=%s: %s: %s" % (PATCH, type(e).__name__, str(e)[:300]), flush=True)
