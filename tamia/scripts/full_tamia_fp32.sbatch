#!/bin/bash
#SBATCH --job-name=qstar-fullB
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/fullB_%j.out
# ARM B: the released recipe with fp32 master weights (fused AdamW on fp32 params, forward under autocast bf16),
# same data order / seed / micro-batch 4 x accum 2 / 100 steps / scheduler / clipping / explicit 8-GPU map as Arm A.
# Writes under ckpt/fullB and results/evalB_*; never touches ckpt/full or results/eval*.
set -uo pipefail
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
MB=${MB:-4}
SPLIT=${SPLIT:-5,5,4,4,4,4,4,2}
EXTRA=${EXTRA:-}
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT=$Q/ckpt/fullB
R=$Q/results
L=$Q/logs
echo "node=$(hostname) job=$SLURM_JOB_ID arm=B micro_batch=$MB split=$SPLIT extra=$EXTRA start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $OUT $R $L
if [ -e $OUT/stats/stats.json ]; then echo "ckpt/fullB already has stats; refusing to overwrite (move it aside first)"; exit 2; fi
echo "=== ARM B train: 100 steps micro_batch=$MB layers=$SPLIT fp32 master + autocast bf16 ($(date -Is)) ==="
python pilot_train.py --fp32_master --steps 100 --micro_batch $MB --layers_per_gpu $SPLIT $EXTRA --save_steps 25,50,75 --out_dir $OUT 2>&1 | tee $OUT/train.log
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] || ! grep -q PILOT_TRAIN_DONE $OUT/train.log; then echo "TRAIN FAILED rc=$rc (evals below run on whatever checkpoints exist)"; fi
echo "=== ARM B evals, 4 lanes on gpu0-3 ($(date -Is)) ==="
lane() { local name=$1 gpu=$2; shift 2; ( export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8; echo "--- $name start=$(date -Is)"; "$@"; echo "--- $name rc=$? end=$(date -Is)" ) > $L/fullB_${name}_$SLURM_JOB_ID.log 2>&1 & }
M=""; for s in 25 50 75 100; do [ -d $OUT/bf16cast_step$s ] && M="${M:+$M,}B_step$s=$OUT/bf16cast_step$s"; done
# (1) thoughts-off (and -on) held-out perplexity, same 200 x 256 slice as eval2, for the 4 bf16-cast Arm B checkpoints
[ -n "$M" ] && lane ppl 0 python pilot_ppl.py --models $M --owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --modes off,on --out $R/evalB_ppl_heldout.json
sleep 15
# (2) harness (soft + hard + newline_wins + w stats) on step 50 and step 100
[ -d $OUT/bf16cast_step50 ] && lane harness50 1 python pilot_eval.py --model $OUT/bf16cast_step50 --label B_step50 --tag evalB --out $R/evalB_harness_step50.json
sleep 15
[ -d $OUT/bf16cast_step100 ] && lane harness100 2 python pilot_eval.py --model $OUT/bf16cast_step100 --label B_step100 --tag evalB --out $R/evalB_harness_step100.json
sleep 15
# (3) debt from eval2: thoughts-off (and -on) ppl for Arm A's step-50 checkpoints (read-only use of ckpt/full)
lane ppl_armA50 3 python pilot_ppl.py --models A_bf16_step50=$Q/ckpt/full/bf16_step50,A_shadow_bf16_step50=$Q/ckpt/full/shadow_bf16_step50 --owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --modes off,on --out $R/evalB_ppl_armA_step50.json
wait
echo "=== evals finished ($(date -Is)) ==="
for f in $L/fullB_ppl_$SLURM_JOB_ID.log $L/fullB_harness50_$SLURM_JOB_ID.log $L/fullB_harness100_$SLURM_JOB_ID.log $L/fullB_ppl_armA50_$SLURM_JOB_ID.log; do
  [ -f $f ] && { echo "--- $f"; grep -E "PILOT_PPL \{|PILOT_PPL data|PILOT_EVAL_SUMMARY|rc=|Traceback|Error" $f | cut -c1-1200; }
done
grep -E "PILOT-B step (1|2|25|50|75|100) |PILOT gpu mem step (1|2|50|100) |PILOT-B: saved|PILOT first micro-batch|PILOT-B group|hf_device_map|gpu mem after load" $OUT/train.log | cut -c1-400
echo "FULLB_DONE end=$(date -Is)"
