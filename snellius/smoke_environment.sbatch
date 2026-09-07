#!/usr/bin/env bash
# Submit only after snellius/validate_migration.sh passes and dependencies are installed.
# This is an environment/NCCL smoke job; it does not load the 7B model or start SFT.
#SBATCH --job-name=vlm3r-env-smoke
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mem=0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "$SCRIPT_DIR/paths.env"

mkdir -p "$VLM3R_LOG_ROOT"
LOG_FILE="$VLM3R_LOG_ROOT/smoke_environment_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

module purge
module load 2023
module load Miniconda3/23.5.2-0
MINICONDA_ROOT=/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0
# shellcheck disable=SC1091
source "$MINICONDA_ROOT/etc/profile.d/conda.sh"
conda activate vlm3r-snellius

# Do not load a system CUDA module for the runtime. The future PyTorch install
# must provide an explicit CUDA build; load a matching toolkit only to compile
# native extensions after that build is known.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE

printf '%s\n' '== allocation =='
hostname
nvidia-smi
printf 'SLURM_GPUS_ON_NODE=%s\n' "${SLURM_GPUS_ON_NODE:-unset}"
printf '%s\n' '== Python packages and CUDA =='
python - <<'PY'
import importlib
import os
import sys
from pathlib import Path

import torch

print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable to PyTorch')
print('gpu count:', torch.cuda.device_count())
if torch.cuda.device_count() != 4:
    raise SystemExit(f'Expected 4 visible GPUs, got {torch.cuda.device_count()}')
for index in range(torch.cuda.device_count()):
    print(f'gpu {index}: {torch.cuda.get_device_name(index)}')

for package in ('llava', 'transformers', 'deepspeed'):
    module = importlib.import_module(package)
    print(f'{package}: {getattr(module, "__version__", "imported")}')

repo = Path(os.environ['REPO_DIR'])
cut3r_root = repo / 'third_party' / 'CUT3R'
cut3r_src = cut3r_root / 'src'
if cut3r_src.is_dir():
    sys.path.insert(0, str(cut3r_root))
    importlib.import_module('src.dust3r.model')
    print('CUT3R: imported')
    try:
        importlib.import_module('src.croco.models.curope.curope2d')
        print('CUROPE: imported')
    except Exception as exc:
        print(f'CUROPE: unavailable ({exc!r})')
else:
    print(f'CUT3R: unavailable (missing source tree: {cut3r_src})')
PY

printf '%s\n' '== four-rank NCCL test =='
torchrun --standalone --nproc_per_node=4 "$SCRIPT_DIR/nccl_smoke.py"
printf '%s\n' 'Environment smoke test passed.'
