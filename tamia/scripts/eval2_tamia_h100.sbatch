#!/bin/bash
#SBATCH --job-name=qstar-eval2-h100
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/eval2_%j.out
# PILOT: eval-only diagnostics after the shadow collapse. 8 independent lanes, one H200 each (a 7B model fits on one GPU
# in eval), so every lane's partial result lands as soon as it is done; lane logs: logs/eval2_<lane>_<jobid>.log
set -uo pipefail
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$Q/ckpt/full
R=$Q/results
L=$Q/logs
BASE_SNAP=$Q/hf_home/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p $R $L
# the H200 and H100 variants of this job are mutually exclusive: whichever starts first cancels the other
JOBS_FILE=$Q/logs/eval2_jobs.txt
if [ -f $JOBS_FILE ]; then for j in $(cat $JOBS_FILE); do [ "$j" != "$SLURM_JOB_ID" ] && scancel $j 2>/dev/null && echo "cancelled sibling job $j"; done; fi

harness() {  # harness <label> <model> [extra args]  -> results/eval2_<label>.json
  local label=$1 model=$2; shift 2
  echo "=== harness $label ($model $*) start=$(date -Is)"
  python pilot_eval.py --model $model --label $label --out $R/eval2_${label}.json --tag eval2 "$@"
  echo "=== harness $label rc=$? end=$(date -Is)"
}
wait_for() {  # wait_for <file> <max_seconds>
  local f=$1 max=$2 t=0
  while [ ! -f "$f" ] && [ $t -lt $max ]; do sleep 30; t=$((t+30)); done
  [ -f "$f" ]
}

# lane A (gpu0): (1) held-out perplexity, thoughts OFF and ON, 4 models; then the fp32 harness on the fp32-built interpolation
laneA() {
  echo "--- ppl start=$(date -Is)"
  python pilot_ppl.py --models base=base,bf16_step100=$OUT/bf16_step100,shadow_bf16_step100=$OUT/shadow_bf16_step100,shadow_fp32_step100=$OUT/shadow_fp32_step100:fp32 \
    --owm_dir $Q/data/owm_heldout_1000_1300 --n_seq 200 --seq_len 256 --batch 8 --out $R/ppl_heldout.json
  echo "--- ppl rc=$? end=$(date -Is)"
  if wait_for $OUT/shadow_interp0.3_fp32/PILOT_interp_summary.json 1800; then
    harness shadow_interp0.3_fp32 $OUT/shadow_interp0.3_fp32 --dtype fp32
  else
    echo "interp fp32 dir not ready; skipping"
  fi
}
# lanes B-D (gpu1-3): standard harness + diagnostics (a)-(e) on base / bf16_step100 / shadow_bf16_step100
laneB() { harness base base; }
laneC() { harness bf16_step100 $OUT/bf16_step100; }
laneD() { harness shadow_bf16_step100 $OUT/shadow_bf16_step100; }
# lane E (gpu4): (2) the step-50 checkpoints, then the bf16 arm with the mixing forced to the base path (control for lane F)
laneE() { harness bf16_step50 $OUT/step50/bf16_step50; harness shadow_bf16_step50 $OUT/step50/shadow_bf16_step50; harness bf16_step100_w0 $OUT/bf16_step100 --zero_talk_head; }
# lane F (gpu5): (3) build the interpolations (CPU), harness the bf16 one; then (4) shadow with the mixing forced to the base path
laneF() {
  echo "--- build interp start=$(date -Is)"
  python build_interp.py --base_dir $BASE_SNAP --from_dir $OUT/shadow_bf16_step100 --alpha 0.3 --out_dtype bf16 --out_dir $OUT/shadow_interp0.3
  python build_interp.py --base_dir $BASE_SNAP --from_dir $OUT/shadow_fp32_step100 --alpha 0.3 --out_dtype fp32 --out_dir $OUT/shadow_interp0.3_fp32
  echo "--- build interp end=$(date -Is)"
  harness shadow_interp0.3 $OUT/shadow_interp0.3
  harness shadow_bf16_step100_w0 $OUT/shadow_bf16_step100 --zero_talk_head
}
# lane G (gpu6): the un-rounded fp32 shadow, evaluated in true fp32 (TF32 off)
laneG() { harness shadow_fp32_step100_fp32 $OUT/shadow_fp32_step100 --dtype fp32; }
# lane H (gpu7): base in fp32 = control for the fp32 harness path
laneH() { harness base_fp32 base --dtype fp32; }

GPUMAP=(${GPUMAP:-0 0 1 1 2 2 3 3})   # GPU index per lane A..H; "0 0 1 1 2 2 3 3" packs two lanes per GPU on a 4x H100-80GB node
i=0
for lane in A B C D E F G H; do
  ( export CUDA_VISIBLE_DEVICES=${GPUMAP[$i]} OMP_NUM_THREADS=6; lane$lane ) > $L/eval2_${lane}_$SLURM_JOB_ID.log 2>&1 &
  i=$((i+1)); sleep 15
done
wait
echo "=== all lanes finished $(date -Is) ==="
for lane in A B C D E F G H; do
  echo "--- lane $lane"; grep -E "PILOT_PPL |PILOT_PPL data|PILOT_INTERP|PILOT_EVAL_SUMMARY|PILOT eval .* diag:|rc=|Error|error|Traceback" $L/eval2_${lane}_$SLURM_JOB_ID.log | cut -c1-1500
done
echo "EVAL2_DONE end=$(date -Is)"
