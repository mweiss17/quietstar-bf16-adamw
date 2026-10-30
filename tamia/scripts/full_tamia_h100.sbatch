#!/bin/bash
#SBATCH --job-name=qstar-full-h100
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/full_h100_%j.out
set -uo pipefail
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
MB=${MB:-2}
SPLIT=${SPLIT:-auto}
EXTRA=${EXTRA:---ckpt_mlp}
[ "$SPLIT" = "auto" ] && SPLITARG="" || SPLITARG="--layers_per_gpu $SPLIT"
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT=$Q/ckpt/full
echo "node=$(hostname) job=$SLURM_JOB_ID micro_batch=$MB split=$SPLIT extra=$EXTRA start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $OUT $Q/results
echo "=== full train: 100 steps micro_batch=$MB layers=$SPLIT $EXTRA ($(date -Is)) ==="
python pilot_train.py --steps 100 --micro_batch $MB $SPLITARG $EXTRA --out_dir $OUT 2>&1 | tee $OUT/train.log
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] || ! grep -q PILOT_TRAIN_DONE $OUT/train.log; then echo "TRAIN FAILED rc=$rc"; fi
echo "=== eval base ($(date -Is)) ==="
python pilot_eval.py --model base --out $Q/results/eval_base.json --tag full 2>&1 | tee $Q/logs/full_eval_base_$SLURM_JOB_ID.log
for D in bf16_step100 shadow_bf16_step100; do
  if [ -d $OUT/$D ]; then
    echo "=== eval $D ($(date -Is)) ==="
    python pilot_eval.py --model $OUT/$D --out $Q/results/eval_${D}.json --tag full 2>&1 | tee $Q/logs/full_eval_${D}_$SLURM_JOB_ID.log
  else
    echo "MISSING $OUT/$D"
  fi
done
echo "FULL_DONE end=$(date -Is)"
