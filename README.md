# Quiet-STaR and AdamW on bf16 weights: rerun code and results

Code, logs, per-step training statistics and evaluation outputs from rerunning the released [Quiet-STaR](https://github.com/ezelikman/quiet-star) training recipe (Mistral-7B-v0.1, 100 optimizer steps, peak lr 1e-6) with and without fp32 optimizer math, plus mask controls. Runs were done Oct 27–28 2026 on 8×H200 nodes (tamIA, Alliance Canada); CPU analyses ran on Fir.

Model checkpoints are not included.

## Headline numbers

GSM8K and CommonsenseQA use the paper's evaluation (soft accuracy, see notes below). Perplexity is held-out OpenWebMath with thoughts off, over the original 32,000-token vocabulary.

| | GSM8K | CommonsenseQA | perplexity |
|---|---|---|---|
| Mistral-7B base | 4.96 | 35.72 | 6.04 |
| released bf16 recipe, step 100 | 7.75 | 44.25 | 8.37 |
| shadow: same gradients, fp32 AdamW | 0.26 | 0.00 | 1,800,000 |
| fp32 master weights, same recipe | 1.59 | 7.94 | 13.6 |

fp32 master weights, learning-rate sweep:

| peak lr | step 50: GSM8K / CSQA | step 100: GSM8K / CSQA | perplexity at 25 / 50 / 75 / 100 |
|---|---|---|---|
| 1e-6 (paper's) | 1.2 / 0.0 | 1.6 / 7.9 | 11.1 / 5×10¹¹ / 280,000 / 13.6 |
| 3e-7 | 7.9 / 44.3 | 1.7 / 10.5 | 6.1 / 9.7 / 729 / 33,000 |
| 1e-7 | 5.4 / 37.2 | 7.4 / 42.0 | 6.0 / 6.1 / 6.4 / 7.7 |
| 3e-8 | 5.0 / 35.9 | 5.2 / 36.6 | 6.0 / 6.0 / 6.0 / 6.1 |
| bf16 at 1e-6, as released | 6.6 / 40.8 | 7.8 / 44.3 | – / 6.9 / – / 8.4 |

Mask controls (soft / hard accuracy at step 100):

| run | weights updated | GSM8K | CSQA | perplexity at 25 / 50 / 75 / 100 |
|---|---|---|---|---|
| base | none | 4.96 / 8.64 | 35.72 / 58.48 | 6.04 |
| bf16, as released | 0.3 to 2.9% per step, 5.65% ever | 7.75 / 10.46 | 44.25 / 61.92 | – / 6.92 / – / 8.37 |
| fp32, lr 1e-6 | all | 1.59 / – | 7.94 / – | 11.1 / 5×10¹¹ / 280,000 / 13.6 |
| fp32, lr 1e-7 | all | 7.43 / 9.93 | 41.99 / 60.61 | 6.04 / 6.10 / 6.39 / 7.70 |
| M1: bf16's mask, exact values | 1.4% per step, 5.04% ever | 6.39 / 9.93 | 41.07 / 63.06 | 6.06 / 6.23 / 6.55 / 6.61 |
| M2: static \|w\| < 5.12e-4 | 13.8% | 7.88 / 8.87 | 41.82 / 58.31 | 6.18 / 7.90 / 11.48 / 11.67 |
| R: random, same density as M2 | 13.8% | 7.93 / 9.78 | 44.52 / 61.59 | 6.10 / 6.92 / 8.96 / 9.69 |
| N: NanoAdam rule | 13.8%, re-picked every 10 steps | 7.90 / 9.40 | 45.45 / 61.67 | 6.17 / 7.79 / 11.12 / 11.61 |
| T: bf16, thoughts off | 2% per step, 6.06% ever | 5.12 / 8.79 | 34.51 / 54.38 | 6.03 / 6.00 / 5.97 / 5.95 |

Masks at bf16's own density, 3 seeds (42 / 1 / 2):

| run, mean ± sd | GSM8K @50 | GSM8K @100 | CSQA @50 | CSQA @100 | perplexity @100 |
|---|---|---|---|---|---|
| bf16, as released | 6.67 ± 0.04 | 8.23 ± 0.41 | 39.83 ± 0.87 | 41.48 ± 3.44 | 9.15 ± 0.68 |
| M5: smallest 5.69% by \|w\| | 6.04 ± 0.06 | 7.40 ± 0.25 | 38.85 ± 0.45 | 41.01 ± 3.15 | 7.46 ± 0.18 |
| R5: random 5.69% | 5.59 ± 0.07 | 6.52 ± 0.16 | 37.25 ± 0.11 | 39.70 ± 0.72 | 6.70 ± 0.04 |

## Layout

```
mechanism/      CPU checks of AdamW on bf16 params in torch 2.9.1 (weight decay, second moment, dropped updates)
fir/            CPU jobs: how many Mistral-7B weights can move at a given lr; diff of the released checkpoint vs base
tamia/scripts/  training, evaluation, perplexity, mask and figure-data scripts + Slurm job files
tamia/results/  evaluation JSONs (per-item probabilities included), figdata/ CSVs
tamia/ckpt/*/stats/  per-step training statistics (lr, grad norm, changed/dropped fraction by parameter class, update and cumulative norms, GPU memory)
tamia/logs/     Slurm job output
figure-data/    wbins.csv (|w| histogram of changed vs unchanged weights by comparator) and crop_l15_down.csv (layer 15 down_proj crop)
```

## mechanism/

| file | what it does |
|---|---|
| `bf16_adamw.py` | Shows (a) weight decay is a no-op on bf16 params, (b) exp_avg_sq at β₂=0.999 stalls, (c) sub-half-ulp updates are dropped, on all torch AdamW code paths |
| `dropped_fraction.py` / `.out` | Fraction of GPT-2 weights whose update / decay would be dropped at various lrs |
| `torchao_emul.py` | Emulates torchao's bf16 AdamW with stochastic rounding: fixes (a) and (c), not (b) |

## fir/

| file | what it does |
|---|---|
| `hist.py` (via `fir_mistral_hist.sh`) | Fraction of Mistral-7B-v0.1 weights that can change at all for step size 1·lr and 0.3·lr, lr 1e-6 … 3e-4. Output: `hist_760843_output.txt` |
| `qdiff.py` (via `fir_qstar_diff.sh`) | Diffs `ezelikman/quietstar-8-ahead` against Mistral-7B-v0.1: changed fraction by class, by magnitude, ulp-distance histogram. Output: `qdiff_763371_output_from_transcript.txt` |
| `figbins.py`, `figcrop.py` | Figure data: magnitude-binned changed counts (`figbins_651956.out`, `figure-data/wbins.csv`) and a weight crop |

## tamia/scripts/

| file | what it does |
|---|---|
| `bootstrap_tamia.sh`, `setup_env.sh`, `pin_revision.py`, `prep_data.py` | Env (CPython 3.11, torch 2.3.1+cu121, transformers 4.37.0 with Quiet-STaR's patched Mistral files), Mistral Hub revision 26bca36b, OpenWebMath shards |
| `pilot_train.py` | The released `quiet-star-train.py` recipe plus: explicit 8-GPU device map, log_dict leak fix, fp32 CPU shadow AdamW on the same clipped grads (Arm A), `--fp32_master` (Arm B: fp32 params + bf16 autocast + fused AdamW), `--lr`, mask controls, per-step stats, checkpoint saves |
| `pilot_train_ctrl2.py` | Second control round (M5/R5 at bf16's per-tensor density) with `--seed` (the released script hard-codes 42) |
| `pilot_masks.py` | Mask construction for M1/M2/R/N controls (N follows the NanoAdam rule from RelationalML/nanoadam 3f8f916) |
| `eval_helpers.py`, `huggingface_custom_callback.py` | From the Quiet-STaR release (GSM8K/CSQA preprocessing and the paper's soft accuracy); callback is a stub |
| `pilot_eval.py`, `pilot_metrics.py` | Paper harness on full test sets + hard argmax accuracy, newline-wins share, top-5 tokens, mixing-weight stats |
| `pilot_ppl.py` | Held-out OpenWebMath perplexity; `--vocab_limit 32000` restricts to the original vocab |
| `build_interp.py` | base + α·(shadow − base) interpolation checkpoints |
| `full_tamia.sbatch` | Arm A: released bf16 recipe + fp32 shadow (job 648895) |
| `full_tamia_fp32.sbatch`, `full_tamia_fp32_lr.sbatch` | Arm B: fp32 master weights at 1e-6 (650122) and the lr sweep (650671–650673) |
| `eval2_tamia.sbatch`, `evalC_tamia.sbatch` | Diagnostics (649573) and restricted-vocab perplexity for every checkpoint (651011) |
| `ctrl_tamia.sbatch`, `ctrl2_tamia.sbatch` | Control rounds 1 (651425–651429) and 2 (651681–651688) |
| `ctrl_table.py`, `ctrl2_table.py`, `ctrl2_counts.py`, `figdata.py`, `*_cpu_smoke.py` | Tables, per-tensor density counts, figure CSV export, CPU smoke tests |

## tamia/results/

- `eval_*`: first harness run. `eval2_*`: diagnostics. `evalB_*`: fp32-master runs and lr sweep. `evalC_ppl_restricted.json`: the perplexity numbers to use. `evalD_*`: control round 1. `evalE_<arm>_s<seed>_*`: control round 2; `ctrl2_summary.json` has the seed means.
- `figdata/`: `per_step_stats.csv`, `per_step_stats_ctrl2.csv`, `csqa_items.csv`, `csqa_first_pos_top20.csv`.

## Notes

- **Accuracy** in the paper and in these JSONs is soft: the product over answer tokens of the probability renormalised over the valid answer tokens (digits + newline for GSM8K, A–E + newline for CSQA). Hard argmax accuracy is in the `*_hard` fields.
- **Perplexity**: checkpoints have 32,002 vocab rows (two added thought tokens with random rows that take ~10% of softmax mass). Use the restricted numbers (first 32,000 logits) in `evalC_ppl_restricted.json`; earlier full-vocab perplexities in some files are confounded by this.
- **Shadow vs fp32 master**: the shadow (job 648895) is first-order: fp32 AdamW applied to the bf16 model's gradients. Arm B (650122 and the sweep) is a real fp32-master run.
- **Deviations from the released code**: log_dict tensors are detached each step (the released code keeps the graph alive when wandb is off and OOMs); explicit device map instead of `device_map='auto'`; Mistral pinned to an older Hub revision because the current tokenizer.json doesn't parse with tokenizers 0.15.
- Paths in scripts and logs point to the original cluster scratch directories and need editing to rerun.
