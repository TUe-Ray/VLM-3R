#!/bin/bash
set -euo pipefail

# Reproducible launcher for scripts/debug_eomt_alignment.py.
# Override any variable below via environment variables, e.g.:
#   EOMT_CKPT_PATH=/path/to/eomt.bin VIDEO_FOLDER=/data bash scripts/run_debug_eomt_alignment.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-scripts/VLM_3R/vsibench_data.yaml}"
VIDEO_FOLDER="${VIDEO_FOLDER:-$PROJECT_ROOT/VLM-3R-DATA}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$VIDEO_FOLDER}"

EOMT_CONFIG_PATH="${EOMT_CONFIG_PATH:-third_party/EoMT/configs/dinov2/coco/panoptic/eomt_large_640.yaml}"
EOMT_CKPT_PATH="${EOMT_CKPT_PATH:-}"

# Fixed dataset configuration for strict alignment with common training setup.
VIDEO_FPS="${VIDEO_FPS:-1}"
FRAMES_UPBOUND="${FRAMES_UPBOUND:-32}"
FORCE_SAMPLE="${FORCE_SAMPLE:-true}"
IMAGE_ASPECT_RATIO="${IMAGE_ASPECT_RATIO:-anyres_max_9}"
ADD_TIME_INSTRUCTION="${ADD_TIME_INSTRUCTION:-true}"
TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-100}"
TRAIN_DATA_PERCENTAGE_SEED="${TRAIN_DATA_PERCENTAGE_SEED:-42}"
TRAIN_DATA_SHUFFLE="${TRAIN_DATA_SHUFFLE:-false}"

# Sample selection: explicit indices are preferred for reproducibility.
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,2}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
START_INDEX="${START_INDEX:-0}"

TOP_K_MASKS="${TOP_K_MASKS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-debug_eomt_alignment_repro}"
DEVICE="${DEVICE:-cuda}"
PROCESSOR_NAME_OR_PATH="${PROCESSOR_NAME_OR_PATH:-google/siglip-so400m-patch14-384}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen2-0.5B}"

if [[ -z "$EOMT_CKPT_PATH" ]]; then
    echo "[ERROR] EOMT_CKPT_PATH is not set."
    echo "Set it first, for example:"
    echo "  EOMT_CKPT_PATH=/path/to/eomt_weights.bin bash scripts/run_debug_eomt_alignment.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

python scripts/debug_eomt_alignment.py \
    --data_path "$DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --image_folder "$IMAGE_FOLDER" \
    --eomt_config_path "$EOMT_CONFIG_PATH" \
    --eomt_ckpt_path "$EOMT_CKPT_PATH" \
    --video_fps "$VIDEO_FPS" \
    --frames_upbound "$FRAMES_UPBOUND" \
    --force_sample "$FORCE_SAMPLE" \
    --image_aspect_ratio "$IMAGE_ASPECT_RATIO" \
    --add_time_instruction "$ADD_TIME_INSTRUCTION" \
    --train_data_percentage "$TRAIN_DATA_PERCENTAGE" \
    --train_data_percentage_seed "$TRAIN_DATA_PERCENTAGE_SEED" \
    --train_data_shuffle "$TRAIN_DATA_SHUFFLE" \
    --processor_name_or_path "$PROCESSOR_NAME_OR_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
    --sample_indices "$SAMPLE_INDICES" \
    --num_samples "$NUM_SAMPLES" \
    --start_index "$START_INDEX" \
    --top_k_masks "$TOP_K_MASKS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"
