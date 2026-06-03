#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vsibench}"

MODEL_PATH="${MODEL_PATH:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/feat_align_cut3r_100p_40723512}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-/leonardo_scratch/fast/EUHPC_D32_006/diag/spatial_perturbation_train_subset/merged_train_for_spatial_diag.json}"
SPATIAL_FEATURE_DIR="${SPATIAL_FEATURE_DIR:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
VIDEO_ROOT="${VIDEO_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
OUTPUT_DIR="${OUTPUT_DIR:-/leonardo_scratch/fast/EUHPC_D32_006/diag/spatial_perturbation_train_subset}"

HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export VLM3R_CODE_ROOT="$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
fi

set +u
conda activate "$CONDA_ENV"
set -u

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

python scripts/diagnose_spatial_perturbation_option_scoring.py \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --model_name vlm-3r-llava-qwen2-lora \
  --train_data_json "$TRAIN_DATA_JSON" \
  --spatial_feature_dir "$SPATIAL_FEATURE_DIR" \
  --spatial_features_subdir spatial_features \
  --video_root "$VIDEO_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --num_per_category "${NUM_PER_CATEGORY:-20}" \
  --categories "${CATEGORIES:-room_size,relative_direction,relative_distance,absolute_distance,route_planning,appearance_order,object_count}" \
  --perturbations "${PERTURBATIONS:-normal,zero_cut3r,shuffle_cut3r_within_frame,replace_cut3r,zero_camera,shuffle_camera}" \
  --seed "${SEED:-42}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --max_frames_num "${MAX_FRAMES_NUM:-32}" \
  --conv_template "${CONV_TEMPLATE:-qwen_1_5}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --torch_dtype "${TORCH_DTYPE:-float16}" \
  --save_logits \
  --no-save_debug_tensors
