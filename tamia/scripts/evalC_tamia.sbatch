#!/bin/bash
#SBATCH --job-name=qstar-evalC
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/evalC_%j.out
# evalC: held-out perplexity RESTRICTED to the 32,000 real vocabulary rows (--vocab_limit 32000), because the two rows
# added by resize_token_embeddings (N(0, 0.02) draw) sit un-masked in the softmax denominator of the plain path:
# 'base' gets a fresh eval-time draw, every checkpoint carries the training-time draw. Both restricted and full numbers
# are recorded for every run, plus the added rows' norms and their logits relative to the max real-token logit.
# Proof: base with torch.manual_seed(1) / (2) before the resize -> full ppl differs, restricted ppl identical.
set -uo pipefail
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$Q/results; L=$Q/logs; A=$Q/ckpt/full; B=$Q/ckpt/fullB
D="--owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --vocab_limit 32000"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
mkdir -p $R $L
rm -f $R/evalC_part_*.json
lane() { local name=$1 gpu=$2; shift 2; ( export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8; echo "--- $name start=$(date -Is)"; "$@"; echo "--- $name rc=$? end=$(date -Is)" ) >> $L/evalC_${name}_$SLURM_JOB_ID.log 2>&1; }
# lane 1 (gpu0): the proof + base thoughts-on + Arm A checkpoints
( lane L1 0 python pilot_ppl.py $D --seed_resize 1 --models base_seed1=base --modes off,on --out $R/evalC_part_base_seed1.json
  lane L1 0 python pilot_ppl.py $D --seed_resize 2 --models base_seed2=base --modes off --out $R/evalC_part_base_seed2.json
  lane L1 0 python pilot_ppl.py $D --models A_bf16_step100=$A/bf16_step100 --modes off,on --out $R/evalC_part_A100.json
  lane L1 0 python pilot_ppl.py $D --models A_bf16_step50=$A/step50/bf16_step50,A_shadow_bf16_step50=$A/step50/shadow_bf16_step50,A_shadow_bf16_step100=$A/shadow_bf16_step100 --modes off --out $R/evalC_part_A.json ) &
sleep 10
# lane 2 (gpu1): Arm B lr 1e-6 (job 650122)
( lane L2 1 python pilot_ppl.py $D --models B_lr1e-6_step25=$B/bf16cast_step25,B_lr1e-6_step50=$B/bf16cast_step50,B_lr1e-6_step75=$B/bf16cast_step75 --modes off --out $R/evalC_part_B1e-6.json
  lane L2 1 python pilot_ppl.py $D --models B_lr1e-6_step100=$B/bf16cast_step100 --modes off,on --out $R/evalC_part_B1e-6_100.json ) &
sleep 10
# lane 3 (gpu2): Arm B lr 3e-7 and 3e-8
( M=""; for lr in 3e-7 3e-8; do for s in 25 50 75 100; do M="${M:+$M,}B_lr${lr}_step$s=${B}_lr$lr/bf16cast_step$s"; done; done
  lane L3 2 python pilot_ppl.py $D --models $M --modes off --out $R/evalC_part_B3e-7_3e-8.json ) &
sleep 10
# lane 4 (gpu3): Arm B lr 1e-7
( lane L4 3 python pilot_ppl.py $D --models B_lr1e-7_step25=${B}_lr1e-7/bf16cast_step25,B_lr1e-7_step50=${B}_lr1e-7/bf16cast_step50,B_lr1e-7_step75=${B}_lr1e-7/bf16cast_step75 --modes off --out $R/evalC_part_B1e-7.json
  lane L4 3 python pilot_ppl.py $D --models B_lr1e-7_step100=${B}_lr1e-7/bf16cast_step100 --modes off,on --out $R/evalC_part_B1e-7_100.json ) &
wait
echo "=== lanes finished $(date -Is) ==="
python - <<'EOF'
import json, glob, os
R = "/scratch/m/mweiss10/qstar/results"
merged = {"parts": {}, "models": {}}
for f in sorted(glob.glob(R + "/evalC_part_*.json")):
    d = json.load(open(f)); merged["parts"][os.path.basename(f)] = {k: v for k, v in d.items() if k != "models"}
    for lab, res in d["models"].items():
        res["seed_resize"] = d.get("seed_resize"); merged["models"][lab] = res
json.dump(merged, open(R + "/evalC_ppl_restricted.json", "w"), indent=1)
order = ["base_seed1", "base_seed2", "A_bf16_step50", "A_bf16_step100", "A_shadow_bf16_step50", "A_shadow_bf16_step100"] + \
        [f"B_lr{lr}_step{s}" for lr in ("1e-6", "3e-7", "1e-7", "3e-8") for s in (25, 50, 75, 100)]
print("EVALC_TABLE model | ppl OFF restricted | ppl OFF full | ppl ON restricted | ppl ON full | lm_head added-row norms | added-row softmax mass mean/max | logit-vs-max-real row32000 mean/max | row32001 mean/max")
for lab in order + [k for k in merged["models"] if k not in order]:
    r = merged["models"].get(lab)
    if not r:
        print("EVALC_TABLE", lab, "| MISSING"); continue
    off = r.get("off", {}); on = r.get("on", {}); x = off.get("extra_rows_logits", {}); er = r.get("extra_rows", {})
    g = lambda d, k: ("%.3f" % d[k]) if k in d and isinstance(d[k], (int, float)) else "-"
    ms = x.get("mass", {}); r0 = x.get("row32000", {}); r1 = x.get("row32001", {})
    print("EVALC_TABLE %s | %s | %s | %s | %s | %s | %s / %s | %s / %s | %s / %s" % (
        lab, g(off, "ppl"), g(off, "ppl_full"), g(on, "ppl"), g(on, "ppl_full"),
        [round(v, 3) for v in er.get("lm_head_row_norms", [])] if er else "-",
        g(ms, "mean"), g(ms, "max"), g(r0, "mean"), g(r0, "max"), g(r1, "mean"), g(r1, "max")))
print("EVALC_TABLE real lm_head row norm mean (base):", merged["models"].get("base_seed1", {}).get("extra_rows", {}).get("lm_head_real_row_norm_mean"))
EOF
for f in $L/evalC_L{1,2,3,4}_$SLURM_JOB_ID.log; do echo "--- $f"; grep -E "PILOT_PPL \{|rc=|Traceback|Error" $f | cut -c1-700; done
echo "EVALC_DONE end=$(date -Is)"
