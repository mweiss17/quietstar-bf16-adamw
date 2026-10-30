#!/bin/bash
#SBATCH --job-name=qstar-figbins
#SBATCH --account=aip-pal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/m/mweiss10/qstar/logs/figbins_%j.out
set -uo pipefail
Q=/scratch/m/mweiss10/qstar
source $Q/env/bin/activate
export HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
BASE=$Q/hf_home/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24
REL=$(ls -d $Q/hf_home/hub/models--ezelikman--quietstar-8-ahead/snapshots/*/ | head -1)
mkdir -p $Q/results/figdata
python $Q/scripts/figbins.py $BASE $Q/results/figdata/wbins.csv released=$REL bf16_s42=$Q/ckpt/full/bf16_step100 fp32_lr1e-7=$Q/ckpt/fullB_lr1e-7/bf16cast_step100 fp32_lr1e-6=$Q/ckpt/fullB/bf16cast_step100
echo FIGBINS_DONE rc=$?

A=$Q/ckpt; T=model.layers.15.mlp.down_proj.weight
python $Q/scripts/figcrop.py $BASE $Q/results/figdata/crop_l15_down.csv $T bf16_s42_step50=$A/full/step50/bf16_step50 bf16_s42_step100=$A/full/bf16_step100 fp32_lr1e-7_step25=$A/fullB_lr1e-7/bf16cast_step25 fp32_lr1e-7_step50=$A/fullB_lr1e-7/bf16cast_step50 fp32_lr1e-7_step75=$A/fullB_lr1e-7/bf16cast_step75 fp32_lr1e-7_step100=$A/fullB_lr1e-7/bf16cast_step100 fp32_lr1e-6_step25=$A/fullB/bf16cast_step25 fp32_lr1e-6_step50=$A/fullB/bf16cast_step50 fp32_lr1e-6_step75=$A/fullB/bf16cast_step75 fp32_lr1e-6_step100=$A/fullB/bf16cast_step100
echo FIGCROP_DONE rc=$?
