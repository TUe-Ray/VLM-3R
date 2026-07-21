#!/bin/bash
#SBATCH --job-name=roi_sim_repro2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/diag/%x_%j.out
#SBATCH --error=logs/diag/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"

MODEL_PATH="${MODEL_PATH:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_PATH="${SIGLIP_PATH:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

DATA_PATH="${DATA_PATH:-$REPO_DIR/scripts/VLM_3R/vsibench_data.yaml}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
SPATIAL_FEATURE_DIR="${SPATIAL_FEATURE_DIR:-$DATA_ROOT/spatial_features}"

RUN_NAME="${RUN_NAME:-reproduction2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/leonardo_scratch/fast/EUHPC_D32_006/diag/roi_similarity/$RUN_NAME}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MAX_VISUALIZED_SAMPLES="${MAX_VISUALIZED_SAMPLES:-$NUM_SAMPLES}"
ANCHOR_MODE="${ANCHOR_MODE:-center}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
FRAMES="${FRAMES:-}"
EXCLUDE_REPRESENTATIONS="${EXCLUDE_REPRESENTATIONS:-}"
LLM_LAYERS="${LLM_LAYERS:-1,4}"
FRAMES_UPBOUND="${FRAMES_UPBOUND:-32}"

mkdir -p "$REPO_DIR/logs/diag" "$OUTPUT_DIR"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  unset LD_LIBRARY_PATH || true
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

export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLM3R_CODE_ROOT="$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_BASE=$MODEL_BASE"
echo "SIGLIP_PATH=$SIGLIP_PATH"
echo "DATA_PATH=$DATA_PATH"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPATIAL_FEATURE_DIR=$SPATIAL_FEATURE_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "SAMPLE_OFFSET=$SAMPLE_OFFSET"
echo "MAX_VISUALIZED_SAMPLES=$MAX_VISUALIZED_SAMPLES"
echo "ANCHOR_MODE=$ANCHOR_MODE"
echo "SAMPLE_IDS=$SAMPLE_IDS"
echo "FRAMES=$FRAMES"
echo "EXCLUDE_REPRESENTATIONS=$EXCLUDE_REPRESENTATIONS"
echo "LLM_LAYERS=$LLM_LAYERS"
echo "FRAMES_UPBOUND=$FRAMES_UPBOUND"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/analysis/analyze_roi_similarity_maps.py \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --siglip_path "$SIGLIP_PATH" \
  --cut3r_weights "$CUT3R_WEIGHTS" \
  --data_json "$DATA_PATH" \
  --image_folder "$DATA_ROOT" \
  --video_folder "$DATA_ROOT" \
  --spatial_feature_dir "$SPATIAL_FEATURE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --sample_offset "$SAMPLE_OFFSET" \
  --sample_ids "$SAMPLE_IDS" \
  --frames "$FRAMES" \
  --anchor_mode "$ANCHOR_MODE" \
  --exclude_representations "$EXCLUDE_REPRESENTATIONS" \
  --llm_layers "$LLM_LAYERS" \
  --include_aligned_projection True \
  --save_raw True \
  --max_visualized_samples "$MAX_VISUALIZED_SAMPLES" \
  --normalize_mode global_per_figure \
  --frames_upbound "$FRAMES_UPBOUND" \
  --mm_spatial_pool_stride 2 \
  --pool_mode bilinear \
  --attn_implementation sdpa \
  --device cuda:0 \
  --dtype float16 \
  --seed 42

echo "[DONE] ROI similarity output: $OUTPUT_DIR"
