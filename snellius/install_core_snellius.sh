#!/usr/bin/env bash
# Install the portable Python layer only.  CUDA extensions are built later on
# a Snellius GPU node, never copied from Leonardo.
set -Eeuo pipefail

REPO_DIR="/home/geusdd/VLM-3R"
ENV_NAME="vlm3r-snellius"
PIP_CACHE_DIR="/scratch-shared/geusdd/VLM3R/migration/pip_cache"

module purge
module load 2023
module load Miniconda3/23.5.2-0
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh
export PIP_CACHE_DIR

PYTHON=(conda run -n "$ENV_NAME" python)

# Keep the PyTorch ABI and project package pins aligned with Leonardo.  Do
# not put flash-attn here: its CUDA extension is built separately on Snellius.
"${PYTHON[@]}" -m pip install \
  'accelerate==0.29.1' \
  'av==17.0.0' \
  'bitsandbytes==0.41.0' \
  'datasets==2.16.1' \
  'decord==0.6.0' \
  'deepspeed==0.14.4' \
  'einops==0.6.1' \
  'einops-exts==0.0.4' \
  'h5py==3.16.0' \
  'huggingface-hub==0.36.2' \
  'hydra-core==1.3.2' \
  'lpips==0.1.4' \
  'matplotlib==3.10.8' \
  'numpy==1.26.4' \
  'opencv-python-headless==4.9.0.80' \
  'peft==0.4.0' \
  'pillow==10.3.0' \
  'roma==1.5.6' \
  'requests==2.32.3' \
  'safetensors==0.7.0' \
  'scikit-learn==1.2.2' \
  'scipy==1.15.3' \
  'sentencepiece==0.1.99' \
  'tensorboard==2.20.0' \
  'timm==1.0.25' \
  'tokenizers==0.15.2' \
  'trimesh==4.11.4' \
  'viser==1.0.24' \
  'wandb==0.25.1'

# The Leonardo environment froze this commit; it is deliberately preferred to
# a nearby release tag for project-model compatibility.
"${PYTHON[@]}" -m pip install \
  'transformers @ git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4'

# Package the local VLM-3R and evaluator source without asking pip to resolve
# their broad optional dependency sets a second time.
"${PYTHON[@]}" -m pip install -e "$REPO_DIR" --no-deps
"${PYTHON[@]}" -m pip install -e "$REPO_DIR/thinking-in-space" --no-deps

# Install only the evaluator runtime dependencies exercised by the VLM-3R
# VSI-Bench path, avoiding unrelated model families and experiments.
"${PYTHON[@]}" -m pip install \
  'evaluate==0.4.3' \
  'jsonlines==4.0.0' \
  'loguru==0.7.2' \
  'nltk==3.8.1' \
  'numexpr==2.10.0' \
  'openpyxl==3.1.2' \
  'pandas==2.2.2' \
  'protobuf==3.20.3' \
  'pyarrow==16.0.0' \
  'pycocoevalcap==1.2' \
  'pytablewriter==1.2.0' \
  'sacrebleu==2.4.1' \
  'sqlitedict==2.1.0' \
  'tenacity==8.3.0' \
  'tiktoken==0.7.0' \
  'tqdm-multiprocess==0.0.11' \
  'transformers-stream-generator==0.0.5' \
  'zstandard==0.23.0' \
  'zss==1.2.0'

"${PYTHON[@]}" - <<'PY'
import importlib
import sys

for package in ("torch", "torchvision", "transformers", "deepspeed", "llava", "lmms_eval"):
    module = importlib.import_module(package)
    print(f"{package}: {getattr(module, '__version__', 'imported')}")
print(f"python: {sys.version}")
PY
