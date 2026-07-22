#!/bin/bash
#SBATCH --job-name=SMOKE_DepthHeadSamples
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
MODEL_PATH="${MODEL_PATH:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_depth_loss_43817021}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_PATH="${SIGLIP_PATH:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
DATA_ROOT="${DATA_ROOT:-$FAST_ROOT/data/vlm3r}"
OUTPUT_DIR="${OUTPUT_DIR:-$FAST_ROOT/eval/depth_head_samples/cut3r_depth_loss_43817021}"

cd "$REPO_DIR"
mkdir -p logs/eval "$OUTPUT_DIR"

echo "==== Depth Head Sample Job ===="
date
echo "HOSTNAME=$(hostname)"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "==============================="

for path in "$MODEL_PATH" "$MODEL_BASE" "$SIGLIP_PATH" "$DATA_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load 2023 CUDA/12.1.1 || module load cuda/12.1 || true
fi

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
fi
set +u
conda activate "$CONDA_ENV"
set -u

export HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/probing/dump_depth_head_samples.py \
  --model-path "$MODEL_PATH" \
  --model-base "$MODEL_BASE" \
  --siglip-path "$SIGLIP_PATH" \
  --data-root "$DATA_ROOT" \
  --spatial-features-root "$DATA_ROOT" \
  --geometry-spatial-features-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples 3
