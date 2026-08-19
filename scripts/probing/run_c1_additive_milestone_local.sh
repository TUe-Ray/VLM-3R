#!/usr/bin/env bash
# Run the two-condition C1 milestone on the local TITAN V host.
# This is inference/calibration/probe-only: it never invokes SFT or changes
# probe implementation semantics.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-vlm3r}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL="${MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP="${SIGLIP:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
CUT3R_ROOT="${CUT3R_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
C1_ARTIFACT="${C1_ARTIFACT:-/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/shaoruei/probe_cache/c1_additive_v1/full}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/c1_additive_v1/full}"
GPU_WEIGHT_BUDGET="${GPU_WEIGHT_BUDGET:-4GiB}"
LAYERS="${LAYERS:-0 1 2 3 6 9 15 21 27}"
FEATURE_LEVELS="${FEATURE_LEVELS:-layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT"

extract_base() {
  echo "[RUN] base extraction: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$OUTPUT_ROOT log=$LOG_ROOT/base_extract.log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/c1_mpl conda run -n "$ENV_NAME" python -u \
    scripts/probing/extract_depth_probe_features.py \
    --model-label pre_sft_base_vlm --model-loading-mode pre_sft_base_vlm \
    --model-path "$MODEL" --siglip-path "$SIGLIP" --feature-preset llm_only --layers $LAYERS \
    --sample-indices "$SAMPLE_INDICES" --output-root "$OUTPUT_ROOT" --train-data-json "$DATA_YAML" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$OUTPUT_ROOT/runtime/pre_sft_base_vlm" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" \
    --resume --assert-first-video 2>&1 | tee "$LOG_ROOT/base_extract.log"
}

extract_c1_additive() {
  echo "[RUN] C1 additive extraction: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$OUTPUT_ROOT log=$LOG_ROOT/c1_additive_extract.log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/c1_mpl conda run -n "$ENV_NAME" python -u \
    scripts/probing/extract_depth_probe_features.py \
    --model-label c1_spatialstack_add --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant c1_ss_add \
    --c1-calibration-json "$C1_ARTIFACT" --model-path "$MODEL" --siglip-path "$SIGLIP" \
    --feature-preset llm_only --layers $LAYERS --sample-indices "$SAMPLE_INDICES" \
    --output-root "$OUTPUT_ROOT" --train-data-json "$DATA_YAML" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --feature-root "$CUT3R_ROOT" \
    --spatial-features-subdir '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' \
    --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$OUTPUT_ROOT/runtime/c1_spatialstack_add" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" \
    --resume --assert-first-video 2>&1 | tee "$LOG_ROOT/c1_additive_extract.log"
}

materialize() {
  echo "[RUN] materializing LLM layer cache files"
  conda run -n "$ENV_NAME" python -u scripts/probing/materialize_depth_probe_layers.py \
    --output-root "$OUTPUT_ROOT" --model-labels pre_sft_base_vlm,c1_spatialstack_add \
    --feature-levels "$FEATURE_LEVELS" 2>&1 | tee "$LOG_ROOT/materialize.log"
}

train_probes() {
  echo "[RUN] unchanged depth probe trainer: CUDA_VISIBLE_DEVICES=0 output=$OUTPUT_ROOT log=$LOG_ROOT/depth_probes.log"
  env CUDA_VISIBLE_DEVICES=0 conda run -n "$ENV_NAME" python -u scripts/probing/train_depth_probes.py \
    --output-root "$OUTPUT_ROOT" --sample-indices "$SAMPLE_INDICES" --probe-subdir depth_probes_c1_additive \
    --result-stem c1_additive_depth_probe --model-labels pre_sft_base_vlm,c1_spatialstack_add \
    --feature-levels "$FEATURE_LEVELS" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
    --num-workers 0 --probe-seed 0 --experiment-variant c1_additive_milestone --device cuda:0 \
    2>&1 | tee "$LOG_ROOT/depth_probes.log"
}

extract_base
extract_c1_additive
materialize
train_probes
