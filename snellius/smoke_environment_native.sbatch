#!/usr/bin/env bash
# One-node build and environment smoke.  It does not load the 7B model or SFT.
#SBATCH --job-name=SMOKE_vlm3r_env
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_env_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_env_%j.err

set -Eeuo pipefail

REPO_DIR="/home/geusdd/VLM-3R"
VLM3R_ROOT="/scratch-shared/geusdd/VLM3R"
source "$REPO_DIR/snellius/paths.env"

module purge
module load 2023
module load CUDA/12.1.1
VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || { echo "missing environment: $VLM3R_CONDA_PREFIX" >&2; exit 2; }
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR:$REPO_DIR/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=16
VLM3R_JOB_TMPDIR="${TMPDIR:-/scratch-local}/${SLURM_JOB_ID}/vlm3r"
export TMPDIR="$VLM3R_JOB_TMPDIR"
mkdir -p "$VLM3R_JOB_TMPDIR"
trap 'rm -rf -- "$VLM3R_JOB_TMPDIR"' EXIT

"$REPO_DIR/snellius/validate_migration.sh"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.version.cuda)
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

# Build from the checked-out Snellius FlashAttention source, never from a
# Leonardo wheel or compiled extension.
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  "$VLM3R_ROOT/migration/flash-attention-2.7.1.post1"

# Force a local rebuild in the CUT3R checkout.  The prior build/lib artifact
# is neither imported nor trusted; this emits a Python-3.10 Snellius binary.
pushd "$REPO_DIR/third_party/CUT3R/src/croco/models/curope"
python setup.py build_ext --inplace --force
popd

python - <<'PY'
import importlib
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 4, 2, 64, device="cuda", dtype=torch.float16)
output = flash_attn_func(q, q, q, causal=True)
assert output.shape == q.shape and torch.isfinite(output).all()
print("flash_attn", importlib.import_module("flash_attn").__version__, "functional=ok")

curope_dir = Path("/home/geusdd/VLM-3R/third_party/CUT3R/src/croco/models/curope")
sys.path.insert(0, str(curope_dir))
curope = importlib.import_module("curope")
tokens = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float16)
positions = torch.zeros(1, 4, 2, device="cuda", dtype=torch.long)
curope.rope_2d(tokens, positions, 100.0, 1.0)
assert torch.isfinite(tokens).all()
print("curope", curope.__file__, "functional=ok")

import deepspeed
import llava
# Import the evaluator before the CUT3R tower.  The tower intentionally adds
# CUT3R's source root to the front of sys.path; importing the evaluator after
# that would resolve its Hugging Face `datasets` dependency to CUT3R's
# unrelated `croco/datasets` package.
import lmms_eval.models.vlm_3r
from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import Cut3rSpatialTower

print("deepspeed", deepspeed.__version__, "llava", getattr(llava, "__version__", "imported"))
print("CUT3R and VSI-Bench evaluator imports=ok")
PY

torchrun --standalone --nproc_per_node=4 "$REPO_DIR/snellius/nccl_smoke.py"
torchrun --standalone --nproc_per_node=4 "$REPO_DIR/snellius/deepspeed_zero2_smoke.py"

echo "Environment/native smoke passed."
