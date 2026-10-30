#!/bin/bash
#SBATCH --job-name=qstar-ctrl
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/ctrl_%j.out
# CONTROLS (evalD). Usage: sbatch --job-name=qstar-ctrl-M1 --output=/scratch/m/mweiss10/qstar/logs/ctrl_M1_%j.out --export=ALL,ARM=M1 ctrl_tamia.sbatch
#   M1 = Arm B + post-step mask "bf16-visible" (exact fp32 step applied only where bf16(w+d) != bf16(w))
#   M2 = Arm B + static magnitude mask |w0| < 5.12e-4 (PaFi-style)      R = Arm B + random mask, M2's per-tensor counts (seed 1234)
#   N  = Arm B + NanoAdam-style bottom-k |w| per tensor, M2's counts, recomputed every 10 steps
#   T  = Arm A recipe exactly (bf16 params, released fused AdamW path) with thoughts disabled; on-GPU stats, no CPU shadow
# Everything else = job 650122 (seed 42, data order, 5/5/4/4/4/4/4/2 map, 100 steps, warmup 20 -> peak 1e-6, clip 1.0, wd groups).
# Writes ckpt/ctrl_$ARM, results/evalD_${ARM}_*, logs/ctrl_${ARM}_*; never touches any other ckpt/results.
set -uo pipefail
ARM=${ARM:?set ARM=M1|M2|R|N|T}
case $ARM in
  M1) FLAGS="--fp32_master --mask_mode bf16vis" ;;
  M2) FLAGS="--fp32_master --mask_mode static_mag --mag_cut 5.12e-4" ;;
  R)  FLAGS="--fp32_master --mask_mode random --mag_cut 5.12e-4 --mask_seed 1234" ;;
  N)  FLAGS="--fp32_master --mask_mode nanoadam --mag_cut 5.12e-4 --mask_interval 10" ;;
  T)  FLAGS="--gpu_stats --no_thoughts" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
SPLIT=${SPLIT:-5,5,4,4,4,4,4,2}
SAVES=${SAVES:-25,50,75}
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT=$Q/ckpt/ctrl_$ARM
R=$Q/results
L=$Q/logs
echo "node=$(hostname) job=$SLURM_JOB_ID arm=$ARM flags=$FLAGS split=$SPLIT saves=$SAVES start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h $Q | tail -1
mkdir -p $OUT $R $L
if [ -e $OUT/stats/stats.json ]; then echo "$OUT already has stats; refusing to overwrite (move it aside first)"; exit 2; fi
echo "=== CTRL $ARM train: 100 steps micro_batch=4 layers=$SPLIT ($(date -Is)) ==="
python pilot_train.py $FLAGS --arm $ARM --lr 1e-6 --steps 100 --micro_batch 4 --layers_per_gpu $SPLIT --save_steps $SAVES --no_final_fp32 --out_dir $OUT 2>&1 | tee $OUT/train.log
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] || ! grep -q PILOT_TRAIN_DONE $OUT/train.log; then echo "TRAIN FAILED rc=$rc (evals below run on whatever checkpoints exist)"; fi
echo "=== CTRL $ARM evals ($(date -Is)) ==="
lane() { local name=$1 gpu=$2; shift 2; ( export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6; echo "--- $name start=$(date -Is)"; "$@"; echo "--- $name rc=$? end=$(date -Is)" ) > $L/ctrl_${ARM}_${name}_$SLURM_JOB_ID.log 2>&1 & }
D="--owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --vocab_limit 32000"
M=""; for s in 25 50 75 100; do [ -d $OUT/bf16cast_step$s ] && M="${M:+$M,}${ARM}_step$s=$OUT/bf16cast_step$s"; done
# (1) thoughts-off RESTRICTED held-out ppl, all four checkpoints; thoughts-on restricted for step 100
[ -n "$M" ] && lane ppl_off 0 python pilot_ppl.py --models $M $D --modes off --out $R/evalD_${ARM}_ppl_off.json
sleep 10
[ -d $OUT/bf16cast_step100 ] && lane ppl_on100 1 python pilot_ppl.py --models ${ARM}_step100=$OUT/bf16cast_step100 $D --modes on --out $R/evalD_${ARM}_ppl_on_step100.json
sleep 10
# (2) harness (soft + hard + newline_wins + w) on step 50 and 100; for T also the w0 variant
g=2
for s in 50 100; do
  [ -d $OUT/bf16cast_step$s ] && lane harness$s $g python pilot_eval.py --model $OUT/bf16cast_step$s --label ${ARM}_step$s --tag evalD_$ARM --out $R/evalD_${ARM}_harness_step$s.json
  g=$((g+1)); sleep 10
  if [ $ARM = T ] && [ -d $OUT/bf16cast_step$s ]; then
    lane harness${s}_w0 $g python pilot_eval.py --model $OUT/bf16cast_step$s --zero_talk_head --label ${ARM}_step${s}_w0 --tag evalD_$ARM --out $R/evalD_${ARM}_harness_step${s}_w0.json
    g=$((g+1)); sleep 10
  fi
done
wait
echo "=== evals finished ($(date -Is)) ==="
for f in $L/ctrl_${ARM}_*_$SLURM_JOB_ID.log; do
  [ -f $f ] && { echo "--- $f"; grep -E "PILOT_PPL \{|PILOT_EVAL_SUMMARY|rc=|Traceback|Error" $f | cut -c1-1500; }
done
grep -E "PILOT-B step (1|2|10|25|50|75|100) |PILOT gpu mem step (1|2|50|100) |PILOT-B: saved|PILOT first micro-batch|PILOT-B group|PILOT-C|PILOT-B stats init|training finished" $OUT/train.log | cut -c1-600
df -h $Q | tail -1
echo "CTRL_DONE arm=$ARM end=$(date -Is)"
