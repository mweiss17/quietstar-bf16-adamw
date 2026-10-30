#!/bin/bash
# Build the pilot env under /scratch/mweiss10/qstar with a uv-managed Python 3.11
# (the CVMFS python/3.11 module ships a _manylinux shim that rejects PyPI manylinux
# wheels, so PyPI torch cu121 cannot be installed into it).
set -euo pipefail
Q=${QSTAR_ROOT:-/scratch/mweiss10/qstar}
mkdir -p $Q/{logs,results,data,scripts,hf_home,ckpt}
cd $Q
if [ ! -d repo ]; then git clone --depth 1 -q https://github.com/ezelikman/quiet-star repo; fi
(cd repo && git rev-parse --short HEAD)
export UV_INSTALL_DIR=$Q/bin
export UV_PYTHON_INSTALL_DIR=$Q/uv_python
export UV_CACHE_DIR=$Q/uv_cache
export PIP_CONFIG_FILE=/dev/null
if [ ! -x $Q/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$Q/bin:$PATH
uv --version
uv python install 3.11
rm -rf $Q/env
uv venv --python 3.11 $Q/env
source $Q/env/bin/activate
python --version
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install "transformers==4.37.0" "accelerate==0.27.2" "datasets==2.16.1" "huggingface_hub==0.25.2" \
    "pyarrow<15" safetensors sentencepiece "numpy<2" wandb pandas seaborn matplotlib termcolor tqdm reportlab protobuf
# Patch transformers' Mistral exactly as the README describes.
SP=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
for f in modeling_mistral.py configuration_mistral.py; do
  [ -f "$SP/models/mistral/$f.orig" ] || cp "$SP/models/mistral/$f" "$SP/models/mistral/$f.orig"
  cp "repo/$f" "$SP/models/mistral/$f"
done
python - <<'EOF'
import torch, transformers, datasets, accelerate, huggingface_hub, numpy
print("versions", torch.__version__, transformers.__version__, datasets.__version__, accelerate.__version__, huggingface_hub.__version__, numpy.__version__)
from transformers.models.mistral import modeling_mistral as m
print("mistral patched:", hasattr(m.MistralForCausalLM, "infer"))
print("cuda built:", torch.version.cuda)
EOF
echo ENV_SETUP_DONE
