#!/bin/bash
#SBATCH --job-name=Extract_VSI_CUT3R_Dec6_9
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/preprocess/%x_%j.out
#SBATCH --error=logs/preprocess/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

NOTE="Pre-extract CUT3R decoder-layer 6/9 token sidecars for VSI-Bench eval videos used by CUT3R SpatialStack evaluation."
echo "-------- Note --------"
echo "  note: $NOTE"

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"

FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
INPUT_ROOT="${INPUT_ROOT:-$HF_HOME/vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"

PROCESSOR_CONFIG_PATH="${PROCESSOR_CONFIG_PATH:-$REPO_DIR/processor_config.json}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

SPLITS="${SPLITS:-scannet arkitscenes scannetpp}"
LAYERS="${LAYERS:-6,9}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PRECISION="${PRECISION:-fp16}"
VIDEO_FPS="${VIDEO_FPS:-1}"
FRAMES_UPBOUND="${FRAMES_UPBOUND:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OVERWRITE="${OVERWRITE:-False}"

cd "$REPO_DIR"
mkdir -p logs/preprocess

echo "==== CUT3R SpatialStack decoder-layer extraction config ===="
date
echo "REPO_DIR=$REPO_DIR"
echo "FAST_ROOT=$FAST_ROOT"
echo "HF_HOME=$HF_HOME"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "PROCESSOR_CONFIG_PATH=$PROCESSOR_CONFIG_PATH"
echo "CUT3R_WEIGHTS=$CUT3R_WEIGHTS"
echo "SPLITS=$SPLITS"
echo "LAYERS=$LAYERS"
echo "GPU_IDS=$GPU_IDS"
echo "PRECISION=$PRECISION"
echo "VIDEO_FPS=$VIDEO_FPS"
echo "FRAMES_UPBOUND=$FRAMES_UPBOUND"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "OVERWRITE=$OVERWRITE"
echo "==========================================================="

for path in "$REPO_DIR" "$CUT3R_WEIGHTS" "$PROCESSOR_CONFIG_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done

if command -v module >/dev/null 2>&1; then
  module purge || true
  unset LD_LIBRARY_PATH || true
  module load 2023 CUDA/12.1.1 || echo "[WARN] module load 2023 CUDA/12.1.1 failed; continuing"
fi

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found and conda.sh missing under $CONDA_BASE"
  exit 1
fi

set +u
conda activate "$CONDA_ENV"
set -u

export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLM3R_CODE_ROOT="$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -z "${CUDA_HOME:-}" && -d "/leonardo/prod/opt/compilers/cuda/12.1/none" ]]; then
  export CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.1/none"
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/bin" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

for split in $SPLITS; do
  input_dir="$INPUT_ROOT/$split"
  output_root="$OUTPUT_ROOT/$split"

  if [[ ! -d "$input_dir" ]]; then
    echo "[ERROR] Missing input video directory: $input_dir"
    exit 1
  fi

  mkdir -p "$output_root"

  cmd=(
    python scripts/extraction/extract_cut3r_layer_features.py
    --input-dir "$input_dir"
    --output-root "$output_root"
    --layers "$LAYERS"
    --cut3r-weights-path "$CUT3R_WEIGHTS"
    --processor-config-path "$PROCESSOR_CONFIG_PATH"
    --gpu-ids "$GPU_IDS"
    --batch-size "$BATCH_SIZE"
    --frames-upbound "$FRAMES_UPBOUND"
    --video-fps "$VIDEO_FPS"
    --precision "$PRECISION"
  )

  if [[ "$OVERWRITE" == "True" ]]; then
    cmd+=(--overwrite)
  fi

  printf '[CMD] %q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
done

echo "[DONE] CUT3R decoder-layer sidecars written under $OUTPUT_ROOT/*/spatial_features_dec_{6,9}"
