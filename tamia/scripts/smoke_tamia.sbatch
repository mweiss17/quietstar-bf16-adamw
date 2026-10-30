#!/bin/bash
#SBATCH --job-name=qstar-smoke
#SBATCH --account=aip-pal
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/smoke_h200_%j.out
set -uo pipefail
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
cd $Q/scripts
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONBREAKPOINT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g | head -2
python -c "import torch; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available(), torch.cuda.device_count())" || exit 1
# Released recipe: micro-batch 4 x accum 2, device_map='auto' over 8 GPUs, no checkpointing. Fallback 2 x 4 only on OOM.
OK=""
for MB in 4 2; do
  OUT=$Q/ckpt/smoke_h200_mb$MB
  rm -rf $OUT; mkdir -p $OUT
  echo "=== smoke train: micro_batch=$MB ($(date -Is)) ==="
  python pilot_train.py --steps 3 --micro_batch $MB --out_dir $OUT 2>&1 | tee $OUT/train.log
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ] && grep -q PILOT_TRAIN_DONE $OUT/train.log; then OK=h200_mb$MB; echo "$MB auto" > $Q/ckpt/SMOKE_CONFIG_H200; break; fi
  if grep -qi "out of memory\|OutOfMemoryError\|PILOT_OOM_EXIT" $OUT/train.log; then echo "=== OOM at micro_batch=$MB ($(date -Is)) ==="; continue; fi
  echo "=== non-OOM failure (rc=$rc) at micro_batch=$MB; stopping ==="; break
done
echo "=== smoke eval base (64) ($(date -Is)) ==="
python pilot_eval.py --model base --limit 64 --out $Q/results/smoke_h200_eval_base_64.json --tag smoke_h200 2>&1 | tee $Q/logs/smoke_h200_eval_base_$SLURM_JOB_ID.log
if [ -n "$OK" ]; then
  OUT=$Q/ckpt/smoke_$OK
  for D in bf16_step3 shadow_bf16_step3; do
    if [ -d $OUT/$D ]; then
      echo "=== smoke eval $D (64) ($(date -Is)) ==="
      python pilot_eval.py --model $OUT/$D --limit 64 --out $Q/results/smoke_h200_eval_${D}_64.json --tag smoke_h200 2>&1 | tee $Q/logs/smoke_h200_eval_${D}_$SLURM_JOB_ID.log
    fi
  done
fi
echo "SMOKE_DONE config_ok=$OK end=$(date -Is)"
