#!/bin/bash
#SBATCH --job-name=layerwise_spatial_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/diag/%x_%j.out
#SBATCH --error=logs/diag/%x_%j.err
#SBATCH --mem=0

set -eo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a VLM-3R checkpoint directory}"
RUN_NAME="${RUN_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-/leonardo_scratch/fast/EUHPC_D32_006/diag/layerwise_scan_${RUN_NAME}_20pcat}"

MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_PATH="${SIGLIP_PATH:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"
TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-$REPO_DIR/scripts/VLM_3R/vsibench_data.yaml}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
SPATIAL_FEATURE_DIR="${SPATIAL_FEATURE_DIR:-$DATA_ROOT}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"

NUM_PER_CATEGORY="${NUM_PER_CATEGORY:-20}"
CATEGORIES="${CATEGORIES:-absolute_distance,route_planning}"
LAYERS="${LAYERS:-1,8,16,24,final}"
ANCHORS_PER_FRAME="${ANCHORS_PER_FRAME:-64}"

cd "$REPO_DIR"
mkdir -p logs/diag "$OUTPUT_DIR"

echo "==== layerwise spatial scan ===="
date
hostname
echo "MODEL_PATH=$MODEL_PATH"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_PER_CATEGORY=$NUM_PER_CATEGORY"
echo "CATEGORIES=$CATEGORIES"
echo "================================"

module load cuda/12.2

source /leonardo_work/EUHPC_D32_006/miniconda3/etc/profile.d/conda.sh
set +u
conda activate vlm3r
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python scripts/diagnose_layerwise_spatial_hidden_scan.py \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --model_name vlm-3r-llava-qwen2-lora \
  --siglip_path "$SIGLIP_PATH" \
  --cut3r_weights "$CUT3R_WEIGHTS" \
  --train_data_json "$TRAIN_DATA_JSON" \
  --image_folder "$DATA_ROOT" \
  --video_folder "$DATA_ROOT" \
  --spatial_feature_dir "$SPATIAL_FEATURE_DIR" \
  --spatial_features_subdir "$SPATIAL_FEATURES_SUBDIR" \
  --output_dir "$OUTPUT_DIR" \
  --layers "$LAYERS" \
  --anchors_per_frame "$ANCHORS_PER_FRAME" \
  --num_per_category "$NUM_PER_CATEGORY" \
  --categories "$CATEGORIES" \
  --compute_roi_spearman True \
  --roi_anchor_mode grid \
  --roi_grid_size 3 \
  --seed 42 \
  --dtype float16 \
  --attn_implementation sdpa
