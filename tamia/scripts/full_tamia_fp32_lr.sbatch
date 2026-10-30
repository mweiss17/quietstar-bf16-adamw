#!/bin/bash
#SBATCH --job-name=qstar-fullB-lr
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/fullB_lr_%j.out
# ARM B learning-rate sweep: identical to job 650122 (fp32 master weights, bf16 autocast, fused AdamW, seed 42, same
# data order, same 5/5/4/4/4/4/4/2 map, 100 steps, warmup 20, clip 1.0, same wd groups) except the peak lr.
# Usage: sbatch --job-name=qstar-fullB-lr3e-7 --output=/scratch/m/mweiss10/qstar/logs/fullB_lr3e-7_%j.out --export=ALL,LR=3e-7 full_tamia_fp32_lr.sbatch
# Writes ckpt/fullB_lr<LR>, results/evalB_lr<LR>_*, logs/fullB_lr<LR>_*; never touches ckpt/full, ckpt/fullB or existing results.
set -uo pipefail
LR=${LR:?set LR, e.g. --export=ALL,LR=3e-7}
TAG=lr$LR
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
MB=${MB:-4}
SPLIT=${SPLIT:-5,5,4,4,4,4,4,2}
EXTRA=${EXTRA:-}
NO_FP32=${NO_FP32:-}        # set NO_FP32=1 to skip the fp32 checkpoint at step 100 (disk)
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT=$Q/ckpt/fullB_$TAG
R=$Q/results
L=$Q/logs
echo "node=$(hostname) job=$SLURM_JOB_ID arm=B lr=$LR micro_batch=$MB split=$SPLIT extra=$EXTRA no_fp32=$NO_FP32 start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $OUT $R $L
if [ -e $OUT/stats/stats.json ]; then echo "$OUT already has stats; refusing to overwrite (move it aside first)"; exit 2; fi
# the Arm A step-50 ppl debt lane runs ONCE, in whichever sweep job starts first (atomic mkdir claim at job start)
DEBT=0
if mkdir $L/evalB_armA50_debt.lock 2>/dev/null; then DEBT=1; echo "$SLURM_JOB_ID $(date -Is)" > $L/evalB_armA50_debt.lock/owner; echo "claimed the Arm A step-50 ppl debt lane"; else echo "debt lane already claimed by $(cat $L/evalB_armA50_debt.lock/owner 2>/dev/null)"; fi
echo "=== ARM B train lr=$LR: 100 steps micro_batch=$MB layers=$SPLIT fp32 master + autocast bf16 ($(date -Is)) ==="
python pilot_train.py --fp32_master --lr $LR --steps 100 --micro_batch $MB --layers_per_gpu $SPLIT $EXTRA --save_steps 25,50,75 ${NO_FP32:+--no_final_fp32} --out_dir $OUT 2>&1 | tee $OUT/train.log
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] || ! grep -q PILOT_TRAIN_DONE $OUT/train.log; then echo "TRAIN FAILED rc=$rc (evals below run on whatever checkpoints exist)"; fi
echo "=== ARM B lr=$LR evals ($(date -Is)) ==="
lane() { local name=$1 gpu=$2; shift 2; ( export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8; echo "--- $name start=$(date -Is)"; "$@"; echo "--- $name rc=$? end=$(date -Is)" ) > $L/fullB_${TAG}_${name}_$SLURM_JOB_ID.log 2>&1 & }
M=""; for s in 25 50 75 100; do [ -d $OUT/bf16cast_step$s ] && M="${M:+$M,}B_${TAG}_step$s=$OUT/bf16cast_step$s"; done
# (1) thoughts-off and thoughts-on held-out ppl, same 200 x 256 slice, all four bf16-cast checkpoints
[ -n "$M" ] && lane ppl 0 python pilot_ppl.py --models $M --owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --modes off,on --out $R/evalB_${TAG}_ppl_heldout.json
sleep 15
# (2) full harness (soft + hard + newline_wins + w) on step 50 and step 100
[ -d $OUT/bf16cast_step50 ] && lane harness50 1 python pilot_eval.py --model $OUT/bf16cast_step50 --label B_${TAG}_step50 --tag evalB_$TAG --out $R/evalB_${TAG}_harness_step50.json
sleep 15
[ -d $OUT/bf16cast_step100 ] && lane harness100 2 python pilot_eval.py --model $OUT/bf16cast_step100 --label B_${TAG}_step100 --tag evalB_$TAG --out $R/evalB_${TAG}_harness_step100.json
sleep 15
# (3) debt from eval2 (once): thoughts-off/-on ppl for Arm A's step-50 checkpoints (ckpt/full/step50/*, read-only)
if [ $DEBT = 1 ]; then
  lane ppl_armA50 3 python pilot_ppl.py --models A_bf16_step50=$Q/ckpt/full/step50/bf16_step50,A_shadow_bf16_step50=$Q/ckpt/full/step50/shadow_bf16_step50 --owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --modes off,on --out $R/evalB_ppl_armA_step50.json
fi
wait
echo "=== evals finished ($(date -Is)) ==="
for f in $L/fullB_${TAG}_ppl_$SLURM_JOB_ID.log $L/fullB_${TAG}_harness50_$SLURM_JOB_ID.log $L/fullB_${TAG}_harness100_$SLURM_JOB_ID.log $L/fullB_${TAG}_ppl_armA50_$SLURM_JOB_ID.log; do
  [ -f $f ] && { echo "--- $f"; grep -E "PILOT_PPL \{|PILOT_PPL data|PILOT_EVAL_SUMMARY|rc=|Traceback|Error" $f | cut -c1-1200; }
done
grep -E "PILOT-B step (1|2|25|50|75|100) |PILOT gpu mem step (1|2|50|100) |PILOT-B: saved|PILOT first micro-batch|PILOT-B group|PILOT args" $OUT/train.log | cut -c1-400
echo "FULLB_LR_DONE lr=$LR end=$(date -Is)"
