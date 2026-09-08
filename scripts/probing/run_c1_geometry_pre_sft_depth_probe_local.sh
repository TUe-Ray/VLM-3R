#!/usr/bin/env bash
# Official local runner for architecture-exposed C1 GeoRoPE pre-SFT probes.
#
# This deliberately retains the historical C1 decoder readout policy of nine
# layers, as specified for the architecture-screening comparison.  The
# --allow-incomplete-pre-sft-features opt-out is therefore explicit below.
set -euo pipefail

MODE="${1:-}"
ARCHITECTURE="${2:-}"
if [[ ! "$MODE" =~ ^(capture|calibrate|smoke|full|summarize)$ ]] || \
   [[ ! "$ARCHITECTURE" =~ ^(geo_rope_fusion|visual_geo_rope)$ ]]; then
  echo "Usage: $0 {capture|calibrate|smoke|full|summarize} {geo_rope_fusion|visual_geo_rope}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
GEOMETRY_ROOT="${GEOMETRY_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
C1_ARTIFACT="${C1_ARTIFACT:-/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json}"
C1_CALIBRATION_MANIFEST="${C1_CALIBRATION_MANIFEST:-/home/shaoruei/probe_outputs/c1_additive_v1/official/calibration_manifest_32.json}"
LOCAL_DATA="${LOCAL_DATA:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
CACHE_BASE="${CACHE_ROOT:-/home/shaoruei/probe_cache/c1_geometry_pre_sft_v1}"
DURABLE_BASE="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/c1_geometry_pre_sft_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/c1_geometry_pre_sft_v1}"
GPU_WEIGHT_BUDGET="${PRE_SFT_GPU_WEIGHT_BUDGET:-4GiB}"
CPU_OFFLOAD_BUDGET="${PRE_SFT_CPU_OFFLOAD_BUDGET:-45GiB}"
# The required historical architecture-screening decoder readouts.
LAYERS=(0 1 2 3 6 9 15 21 27)
PRE_LLM_FEATURES="fusion_output,projected_features"

if [[ "$ARCHITECTURE" == "geo_rope_fusion" ]]; then
  VARIANT="c1_geo_rope_fusion"
  LABEL="c1_geo_rope_fusion"
  CALIBRATOR_ARCH="geo_rope_fusion"
else
  VARIANT="c1_visual_geo_rope"
  LABEL="c1_visual_geo_rope"
  CALIBRATOR_ARCH="visual_geo_rope"
fi
CAPTURE_ROOT="$CACHE_BASE/calibration_captures/$ARCHITECTURE"
CALIBRATION_JSON="$DURABLE_BASE/$ARCHITECTURE/c1_activation.json"
SMOKE_ROOT="$CACHE_BASE/smoke/$ARCHITECTURE"
FULL_ROOT="$CACHE_BASE/full/$ARCHITECTURE"
SMOKE_MARKER="$DURABLE_BASE/$ARCHITECTURE/smoke_verification.json"

mkdir -p "$CACHE_BASE" "$DURABLE_BASE/$ARCHITECTURE" "$LOG_ROOT"

run() {
  printf '[COMMAND] '; printf '%q ' "$@"; printf '\n'
  "$@"
}

require_inputs() {
  local path
  for path in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" \
    "$C1_ARTIFACT" "$C1_CALIBRATION_MANIFEST" "$LOCAL_DATA"; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
  [[ -d "$FORWARD_ROOT" && -d "$TARGET_ROOT" && -d "$FEATURE_ROOT" && -d "$GEOMETRY_ROOT" ]] || {
    echo "Missing required forward/cache input root" >&2; exit 1;
  }
}

require_gpu() {
  local output="$1"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" --physical-gpu-id "$GPU" --output "$output"
}

capture() {
  require_inputs
  require_gpu "$DURABLE_BASE/$ARCHITECTURE/capture_gpu_${GPU}_readiness.json"
  echo "[RUN] forward-only C1 calibration capture: architecture=$ARCHITECTURE GPUs=$CUDA_DEVICES output=$CAPTURE_ROOT"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$VARIANT" \
    --c1-calibration-json "$C1_ARTIFACT" --calibration-capture-pre-llm \
    --model-label "$LABEL" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --feature-levels siglip_output,projected_features --allow-incomplete-pre-sft-features \
    --output-root "$CAPTURE_ROOT" --sample-indices "$C1_CALIBRATION_MANIFEST" \
    --train-data-json "$LOCAL_DATA" --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features \
    --geometry-spatial-features-root "$GEOMETRY_ROOT" --geometry-spatial-features-subdir spatial_features_points \
    --geometry-point-map-key point_maps_ref \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" \
    --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$CAPTURE_ROOT/runtime" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" \
    --resume 2>&1 | tee "$LOG_ROOT/${ARCHITECTURE}_capture.log"
}

calibrate() {
  require_inputs
  [[ -d "$CAPTURE_ROOT/calibration_captures" ]] || { echo "Run capture first: $CAPTURE_ROOT" >&2; exit 1; }
  require_gpu "$DURABLE_BASE/$ARCHITECTURE/calibration_gpu_${GPU}_readiness.json"
  echo "[RUN] unlabeled C1 activation calibration: architecture=$ARCHITECTURE GPU=$GPU output=$CALIBRATION_JSON"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/calibrate_c1_geometry_activation.py" \
    --architecture "$CALIBRATOR_ARCH" --capture-root "$CAPTURE_ROOT/calibration_captures" \
    --c1-reference-json "$C1_ARTIFACT" --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --output-json "$CALIBRATION_JSON" --device cuda:0 --device-map auto --dtype float16 \
    --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" \
    2>&1 | tee "$LOG_ROOT/${ARCHITECTURE}_calibration.log"
}

extract() {
  local output_root="$1" manifest="$2" log="$3" limit="${4:-}"
  local command=(
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py"
    --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$VARIANT"
    --c1-calibration-json "$C1_ARTIFACT" --geometry-c1-calibration-json "$CALIBRATION_JSON"
    --model-label "$LABEL" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL"
    --feature-preset llm_only --layers "${LAYERS[@]}" --pre-llm-features "$PRE_LLM_FEATURES"
    --allow-incomplete-pre-sft-features --output-root "$output_root" --sample-indices "$manifest"
    --train-data-json "$LOCAL_DATA" --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features
    --geometry-spatial-features-root "$GEOMETRY_ROOT" --geometry-spatial-features-subdir spatial_features_points
    --geometry-point-map-key point_maps_ref
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT"
    --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16
    --runtime-root "$output_root/runtime" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" --resume
  )
  if [[ -n "$limit" ]]; then command+=(--limit-videos "$limit" --assert-first-video); fi
  echo "[RUN] C1 architecture-exposed extraction: architecture=$ARCHITECTURE GPUs=$CUDA_DEVICES output=$output_root"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u "${command[@]}" 2>&1 | tee "$log"
}

smoke() {
  require_inputs
  [[ -f "$CALIBRATION_JSON" ]] || { echo "Run calibrate first: $CALIBRATION_JSON" >&2; exit 1; }
  require_gpu "$DURABLE_BASE/$ARCHITECTURE/smoke_gpu_${GPU}_readiness.json"
  extract "$SMOKE_ROOT" "$SAMPLE_INDICES" "$LOG_ROOT/${ARCHITECTURE}_smoke.log" 1
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_c1_geometry_pre_sft_smoke.py" \
    --output-root "$SMOKE_ROOT" --model-label "$LABEL" --architecture "$CALIBRATOR_ARCH" \
    --activation-json "$CALIBRATION_JSON" --c1-reference-json "$C1_ARTIFACT" --output "$SMOKE_MARKER"
  echo "[PASS] smoke marker: $SMOKE_MARKER"
}

train_level() {
  local level="$1" gpu="$2" log="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$FULL_ROOT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes --model-labels "$LABEL" \
    --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 \
    --probe-seed 0 --experiment-variant "$LABEL" --device cuda:0 --no-write-aggregate --skip-existing \
    2>&1 | tee "$log"
}

full() {
  require_inputs
  [[ -f "$SMOKE_MARKER" ]] || { echo "Run '$0 smoke $ARCHITECTURE' successfully before full extraction." >&2; exit 1; }
  require_gpu "$DURABLE_BASE/$ARCHITECTURE/full_gpu_${GPU}_readiness.json"
  extract "$FULL_ROOT" "$SAMPLE_INDICES" "$LOG_ROOT/${ARCHITECTURE}_full_extraction.log"
  local levels=(fusion_output projected_features)
  local layer
  for layer in "${LAYERS[@]}"; do levels+=("layer_$layer"); done
  local index=0 first second first_pid second_pid
  while [[ "$index" -lt "${#levels[@]}" ]]; do
    first="${levels[$index]}"
    train_level "$first" 0 "$LOG_ROOT/${ARCHITECTURE}_${first}_probe.log" & first_pid=$!
    index=$((index + 1))
    if [[ "$index" -lt "${#levels[@]}" ]]; then
      second="${levels[$index]}"
      train_level "$second" 1 "$LOG_ROOT/${ARCHITECTURE}_${second}_probe.log" & second_pid=$!
      wait "$first_pid"; wait "$second_pid"
      index=$((index + 1))
    else
      wait "$first_pid"
    fi
  done
  summarize
}

summarize() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_c1_geometry_pre_sft_depth_probe.py" \
    --output-root "$FULL_ROOT" --model-label "$LABEL" --architecture "$CALIBRATOR_ARCH" \
    --activation-json "$CALIBRATION_JSON" --c1-reference-json "$C1_ARTIFACT" --output-dir "$DURABLE_BASE/$ARCHITECTURE"
}

case "$MODE" in
  capture) capture ;;
  calibrate) calibrate ;;
  smoke) smoke ;;
  full) full ;;
  summarize) summarize ;;
esac
