#!/usr/bin/env bash
# Read-only, single-GPU Snellius preflight. This is deliberately not a
# substitute for smoke_environment_native.sbatch: it neither builds native
# extensions into the shared environment nor tests 4-rank NCCL/ZeRO-2.
#SBATCH --job-name=SMOKE_vlm3r_1gpu_preflight
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_1gpu_preflight_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_1gpu_preflight_%j.err

set -Eeuo pipefail

REPO_DIR="/home/geusdd/VLM-3R"
source "$REPO_DIR/snellius/paths.env"

module purge
module load 2023
module load CUDA/12.1.1
VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || {
  echo "missing environment: $VLM3R_CONDA_PREFIX" >&2
  exit 2
}
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR:$REPO_DIR/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$REPO_DIR/snellius/validate_migration.sh"

python - <<'PY'
import importlib.util

import torch

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
device = torch.device("cuda:0")
value = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
assert torch.isfinite(value @ value.T).all()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "cuda_matmul=ok")

import deepspeed
import llava
import lmms_eval.models.vlm_3r

print("deepspeed", deepspeed.__version__, "llava/lmms_eval_imports=ok")
print("flash_attn_prebuilt=", importlib.util.find_spec("flash_attn") is not None)
print("curope_prebuilt=", importlib.util.find_spec("curope") is not None)
PY

echo "Single-GPU preflight passed; four-GPU native smoke remains required."
