#!/bin/bash
set -uo pipefail
export QSTAR_ROOT=/scratch/m/mweiss10/qstar
Q=$QSTAR_ROOT
cd $Q
bash scripts/setup_env.sh > logs/setup_env.log 2>&1 || { echo SETUP_FAILED; exit 1; }
cp repo/eval_helpers.py scripts/eval_helpers.py
sed -i "s|^initial_tokenizer = AutoTokenizer.from_pretrained(\"mistralai/Mistral-7B-v0.1\")|initial_tokenizer = AutoTokenizer.from_pretrained(\"mistralai/Mistral-7B-v0.1\", revision=\"26bca36bde8333b5d7f72e9ed20ccda6a618af24\")  # PILOT: pre-July-2024 revision (tokenizer.json parsable by tokenizers 0.15)|" scripts/eval_helpers.py
source env/bin/activate
export HF_HOME=$Q/hf_home
python scripts/prep_data.py > logs/prep_data.log 2>&1 || { echo PREP_FAILED; exit 1; }
python scripts/pin_revision.py > logs/pin_revision.log 2>&1
echo 26bca36bde8333b5d7f72e9ed20ccda6a618af24 > hf_home/hub/models--mistralai--Mistral-7B-v0.1/refs/main
HF_HUB_OFFLINE=1 python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(\"mistralai/Mistral-7B-v0.1\"); print(\"offline default tokenizer ok\", len(t), t.convert_ids_to_tokens([28740,13,330]))" >> logs/pin_revision.log 2>&1
cd scripts && python -m py_compile pilot_train.py pilot_eval.py eval_helpers.py && echo COMPILE_OK
echo BOOTSTRAP_DONE
