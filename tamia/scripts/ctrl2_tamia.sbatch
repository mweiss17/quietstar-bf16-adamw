#!/bin/bash
#SBATCH --job-name=qstar-ctrl2
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/ctrl2_%j.out
# CONTROLS 2 (evalE). Usage: sbatch --job-name=qstar-ctrl2-M5-s1 --output=/scratch/m/mweiss10/qstar/logs/ctrl2_M5_s1_%j.out --export=ALL,ARM=M5,SEED=1 ctrl2_tamia.sbatch
#   M5 = Arm B code path + static bottom-k |w0| mask, per-tensor k = Arm A's step-100 changed count (ctrl2_counts.json), M2's tensor classes
#   R5 = Arm B code path + random mask with exactly M5's per-tensor counts; mask seed = 1234 (seed 42, = R's), 2234 (seed 1), 3234 (seed 2)
#   A  = Arm A released bf16 recipe (bf16 params, fused AdamW); --gpu_stats = on-GPU read-only stats, NO CPU fp32 shadow
# Everything else = the Controls jobs (5/5/4/4/4/4/4/2 map, 100 steps, warmup 20 -> peak 1e-6, clip 1.0, wd groups); seed via --seed.
# Saves bf16 casts at 25/50/75/100 only (no fp32). Run with CKPT_LOCAL=1: casts go to node-local $SLURM_TMPDIR (--ckpt_dir), evals read
#   them there, nothing is kept and nothing is deleted on /scratch. /scratch gets only the run dir ckpt/ctrl2_<tag> (stats/ + train.log,
#   ~25 MB), results/evalE_<tag>_* (JSONs + a copy of the stats JSONs), logs/ctrl2_<tag>_* and the marker logs/ctrl2_<tag>_STATUS_<job>.txt.
# Writes ckpt/ctrl2_${ARM}_s${SEED}, results/evalE_${ARM}_s${SEED}_*, logs/ctrl2_${ARM}_s${SEED}_*; never touches any other ckpt/results.
set -uo pipefail
ARM=${ARM:?set ARM=M5|R5|A}
SEED=${SEED:?set SEED}
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
COUNTS=$Q/scripts/ctrl2_counts.json
case $SEED in 42) MS=1234 ;; 1) MS=2234 ;; 2) MS=3234 ;; *) MS=$((1234 + 1000 * SEED)) ;; esac
case $ARM in
  M5) FLAGS="--fp32_master --mask_mode static_topk --mask_counts $COUNTS" ;;
  R5) FLAGS="--fp32_master --mask_mode random --mask_counts $COUNTS --mask_seed $MS" ;;
  A)  FLAGS="--gpu_stats" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
TAG=${ARM}_s${SEED}
SPLIT=${SPLIT:-5,5,4,4,4,4,4,2}
SAVES=${SAVES:-25,50,75}
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT=$Q/ckpt/ctrl2_$TAG
R=$Q/results
L=$Q/logs
echo "node=$(hostname) job=$SLURM_JOB_ID arm=$ARM seed=$SEED flags=$FLAGS split=$SPLIT saves=$SAVES start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h $Q | tail -1; diskusage_report 2>/dev/null | grep -i scratch
mkdir -p $OUT $R $L
if [ -e $OUT/stats/stats.json ]; then echo "$OUT already has stats; refusing to overwrite (move it aside first)"; exit 2; fi
# where the bf16 casts go: node-local with CKPT_LOCAL=1 (nothing on /scratch, nothing kept); /scratch only if explicitly allowed
if [ "${CKPT_LOCAL:-0}" = 1 ]; then
  CK=${SLURM_TMPDIR:?CKPT_LOCAL needs SLURM_TMPDIR}/ctrl2_$TAG; mkdir -p $CK; df -h $SLURM_TMPDIR | tail -1
  echo "CKPT_LOCAL: casts -> $CK (node-local; do not survive the job). Stats/train.log -> $OUT, evals -> $R/evalE_${TAG}_*, logs -> $L"
elif [ "${ALLOW_SCRATCH_CKPT:-0}" = 1 ]; then
  CK=$OUT
else
  echo "refusing: set CKPT_LOCAL=1 (or ALLOW_SCRATCH_CKPT=1 to write 58 GB of casts to /scratch)"; exit 2
fi
# status marker on /scratch: rewritten at every stage and on exit / SIGTERM (time limit, scancel)
ST=$L/ctrl2_${TAG}_STATUS_$SLURM_JOB_ID.txt
STAGE=start; TRAIN_RC=NA; ALL_OK=0
status() {
  { echo "tag=$TAG job=$SLURM_JOB_ID node=$(hostname) stage=$STAGE train_rc=$TRAIN_RC all_ok=$ALL_OK time=$(date -Is) ckpt=$CK"
    echo "train_done_marker=$(grep -c PILOT_TRAIN_DONE $OUT/train.log 2>/dev/null)  last_step_line: $(grep -E '^PILOT-B step [0-9]+ ' $OUT/train.log 2>/dev/null | tail -1 | cut -c1-80)"
    echo "first_batch: $(grep -m1 'PILOT first micro-batch' $OUT/train.log 2>/dev/null | grep -o 'sum=[0-9]*')"
    echo "casts_present: $(ls -d $CK/bf16cast_step* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
    for j in ppl_off ppl_on_step100 harness_step50 harness_step100; do [ -s $R/evalE_${TAG}_$j.json ] && echo "result_ok: evalE_${TAG}_$j.json" || echo "result_missing: evalE_${TAG}_$j.json"; done
    for f in $L/ctrl2_${TAG}_*_$SLURM_JOB_ID.log; do [ -f $f ] && echo "lane $(basename $f): $(grep -E -- '^--- .* rc=' $f | tail -1)"; done
  } > $ST.tmp 2>/dev/null && mv -f $ST.tmp $ST
}
trap 'STAGE="killed_during_${STAGE}"; status; exit 143' TERM
trap 'status' EXIT
status
STAGE=train
echo "=== CTRL2 $TAG train: 100 steps micro_batch=4 layers=$SPLIT ($(date -Is)) ==="
python pilot_train_ctrl2.py $FLAGS --seed $SEED --arm $ARM --lr 1e-6 --steps 100 --micro_batch 4 --layers_per_gpu $SPLIT --save_steps $SAVES --no_final_fp32 --out_dir $OUT --ckpt_dir $CK 2>&1 | tee $OUT/train.log &
wait $!   # (backgrounded so the TERM trap can fire while training runs)
TRAIN_RC=$(grep -q PILOT_TRAIN_DONE $OUT/train.log && echo 0 || echo 1)
TRAIN_OK=1
if [ $TRAIN_RC -ne 0 ]; then TRAIN_OK=0; echo "TRAIN FAILED (no PILOT_TRAIN_DONE; evals below run on whatever checkpoints exist)"; fi
STAGE=evals; status
echo "=== CTRL2 $TAG evals ($(date -Is)) ==="
lane() { local name=$1 gpu=$2; shift 2; ( export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6; echo "--- $name start=$(date -Is)"; "$@"; echo "--- $name rc=$? end=$(date -Is)" ) > $L/ctrl2_${TAG}_${name}_$SLURM_JOB_ID.log 2>&1 & }
D="--owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --vocab_limit 32000"
M=""; for s in 25 50 75 100; do [ -f $CK/bf16cast_step$s/config.json ] && M="${M:+$M,}${TAG}_step$s=$CK/bf16cast_step$s"; done
# (1) thoughts-off RESTRICTED held-out ppl, all four checkpoints; thoughts-on restricted for step 100 (as in the Controls)
[ -n "$M" ] && lane ppl_off 0 python pilot_ppl.py --models $M $D --modes off --out $R/evalE_${TAG}_ppl_off.json
sleep 10
[ -f $CK/bf16cast_step100/config.json ] && lane ppl_on100 1 python pilot_ppl.py --models ${TAG}_step100=$CK/bf16cast_step100 $D --modes on --out $R/evalE_${TAG}_ppl_on_step100.json
sleep 10
# (2) harness (soft + hard + newline_wins + w) on step 50 and 100
g=2
for s in 50 100; do
  [ -f $CK/bf16cast_step$s/config.json ] && lane harness$s $g python pilot_eval.py --model $CK/bf16cast_step$s --label ${TAG}_step$s --tag evalE_$TAG --out $R/evalE_${TAG}_harness_step$s.json
  g=$((g+1)); sleep 10
done
wait
STAGE=evals_finished
echo "=== evals finished ($(date -Is)) ==="
ALL_OK=$TRAIN_OK
for f in $L/ctrl2_${TAG}_*_$SLURM_JOB_ID.log; do
  [ -f $f ] && { echo "--- $f"; grep -E "PILOT_PPL \{|PILOT_EVAL_SUMMARY|rc=|Traceback|Error" $f | cut -c1-1500; grep -q -- "--- .* rc=0 end=" $f || ALL_OK=0; }
done
for j in ppl_off ppl_on_step100 harness_step50 harness_step100; do [ -s $R/evalE_${TAG}_$j.json ] || { echo "missing $R/evalE_${TAG}_$j.json"; ALL_OK=0; }; done
grep -E "PILOT-B step (1|2|10|25|50|75|100) |PILOT gpu mem step (1|2|50|100) |PILOT-B: saved|PILOT first micro-batch|PILOT-B group|PILOT-C|PILOT-B stats init|training finished" $OUT/train.log | cut -c1-600
# (3) small copy of the per-step stats next to the eval JSONs (~2 MB; the run dir $OUT keeps the full stats incl. wsample_*.pt)
mkdir -p $R/evalE_${TAG}_stats && cp -p $OUT/stats/*.json $OUT/stats/stats.jsonl $R/evalE_${TAG}_stats/ 2>/dev/null
# (4) checkpoints: node-local ones vanish with the job; on /scratch (ALLOW_SCRATCH_CKPT) drop step 25/75 only if all evals succeeded
if [ "$CK" = "$OUT" ] && [ $ALL_OK = 1 ]; then
  for s in 25 75; do [ -d $OUT/bf16cast_step$s ] && rm -rf -- "$OUT/bf16cast_step$s" && echo "deleted $OUT/bf16cast_step$s"; done
fi
du -sh $OUT $R/evalE_${TAG}_* $L/ctrl2_${TAG}_* 2>/dev/null | sort -h | tail -5
df -h $Q | tail -1; diskusage_report 2>/dev/null | grep -i scratch
STAGE=done; status
echo "CTRL2_DONE tag=$TAG all_ok=$ALL_OK end=$(date -Is)"
