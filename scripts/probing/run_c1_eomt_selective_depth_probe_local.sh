#!/usr/bin/env bash
# Official local runner: frozen C1 VLM3R full-K/V baseline vs EoMT-selective K/V.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(smoke|full|summarize)$ ]]; then
  echo "Usage: $0 smoke|full|summarize" >&2
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
EOMT_CACHE_ROOT="${EOMT_CACHE_ROOT:-/home/shaoruei/probe_cache/eomt_consumer_grid_v2}"
EOMT_VALIDATION="${EOMT_VALIDATION:-$EOMT_CACHE_ROOT/validation.json}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
C1_ARTIFACT="${C1_ARTIFACT:-/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json}"
PAIRED_CALIBRATION="${PAIRED_CALIBRATION:-/home/shaoruei/probe_outputs/c1_eomt_selective_calibration_v1/full/summary.json}"
CACHE_BASE="${CACHE_ROOT:-/home/shaoruei/probe_cache/c1_eomt_selective_depth_probe_v1}"
DURABLE_BASE="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/c1_eomt_selective_depth_probe_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/c1_eomt_selective_depth_probe_v1}"
GPU_WEIGHT_BUDGET="${PRE_SFT_GPU_WEIGHT_BUDGET:-4GiB}"
CPU_OFFLOAD_BUDGET="${PRE_SFT_CPU_OFFLOAD_BUDGET:-45GiB}"
LABEL="c1_vlm3r_eomt_selective"
LAYERS=(0 1 2 3 6 9 15 21 27)
LOCAL_DATA="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
FULL_ROOT="$CACHE_BASE/full"
SMOKE_ROOT="$CACHE_BASE/smoke"
SMOKE_MARKER="$DURABLE_BASE/smoke/c1_eomt_selective_smoke_report.json"

mkdir -p "$CACHE_BASE" "$DURABLE_BASE" "$LOG_ROOT"

run() {
  printf '[COMMAND] '; printf '%q ' "$@"; printf '\n'
  "$@"
}

require_inputs() {
  local path
  for path in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" "$C1_ARTIFACT" "$EOMT_VALIDATION"; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
  [[ -d "$EOMT_CACHE_ROOT" && -d "$FORWARD_ROOT" && -d "$TARGET_ROOT" && -d "$FEATURE_ROOT" ]] || {
    echo "Missing required cache/input root" >&2; exit 1;
  }
}

require_gpu() {
  local output="$1"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" --physical-gpu-id "$GPU" --output "$output"
}

extract_selective() {
  local output_root="$1" manifest="$2" log="$3" limit="${4:-}"
  local command=(
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py"
    --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant c1_vlm3r
    --c1-calibration-json "$C1_ARTIFACT" --eomt-selective-kv-gate
    --model-label "$LABEL" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL"
    --feature-preset llm_only --layers "${LAYERS[@]}" --output-root "$output_root" --sample-indices "$manifest"
    --train-data-json "$LOCAL_DATA" --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT"
    --eomt-consumer-cache-root "$EOMT_CACHE_ROOT" --eomt-cache-validation "$EOMT_VALIDATION" --verify-eomt-file-checksum
    --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16
    --runtime-root "$output_root/runtime" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" --resume
  )
  if [[ -n "$limit" ]]; then command+=(--limit-videos "$limit" --assert-first-video); fi
  echo "[RUN] frozen C1 selective extraction GPUs=$CUDA_DEVICES output=$output_root log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u "${command[@]}" 2>&1 | tee "$log"
}

run_smoke() {
  require_inputs
  mkdir -p "$DURABLE_BASE/smoke"
  require_gpu "$DURABLE_BASE/smoke/gpu_${GPU}_readiness.json"
  echo "[RUN] paired one-sample residual smoke (C1 lambda is frozen; no probe fitting)"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/diagnose_c1_eomt_selective_calibration.py" \
    --calibration-manifest /home/shaoruei/probe_outputs/c1_additive_v1/official/calibration_manifest_32.json \
    --max-samples 1 --c1-calibration-json "$C1_ARTIFACT" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --forward-frames-root "$FORWARD_ROOT" --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features \
    --train-data-json "$LOCAL_DATA" --eomt-consumer-cache-root "$EOMT_CACHE_ROOT" --eomt-cache-validation "$EOMT_VALIDATION" \
    --output-dir "$DURABLE_BASE/smoke/paired_calibration" --device cuda:0 --device-map auto \
    --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" \
    2>&1 | tee "$LOG_ROOT/smoke_paired_calibration.log"
  extract_selective "$SMOKE_ROOT" "$SAMPLE_INDICES" "$LOG_ROOT/smoke_extraction.log" 1
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_c1_eomt_selective_smoke.py" \
    --output-root "$SMOKE_ROOT" --model-label "$LABEL" \
    --paired-calibration-summary "$DURABLE_BASE/smoke/paired_calibration/summary.json" --output "$SMOKE_MARKER"
  echo "[PASS] Smoke marker: $SMOKE_MARKER"
}

train_level() {
  local level="$1" gpu="$2" log="$3"
  echo "[RUN] frozen-feature MLP probe level=$level physical_gpu=$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$FULL_ROOT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes --model-labels "$LABEL" \
    --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 \
    --probe-seed 0 --experiment-variant c1_vlm3r_eomt_selective --device cuda:0 --no-write-aggregate --skip-existing \
    2>&1 | tee "$log"
}

run_full() {
  require_inputs
  [[ -f "$SMOKE_MARKER" ]] || { echo "Run '$0 smoke' successfully before full extraction." >&2; exit 1; }
  require_gpu "$DURABLE_BASE/full_gpu_${GPU}_readiness.json"
  echo "[RUN] full extraction: GPUs=$CUDA_DEVICES cache=$FULL_ROOT log=$LOG_ROOT/full_extraction.log"
  extract_selective "$FULL_ROOT" "$SAMPLE_INDICES" "$LOG_ROOT/full_extraction.log"
  local index=0 level first_level first_gpu second_level second_gpu
  while [[ "$index" -lt "${#LAYERS[@]}" ]]; do
    first_level="layer_${LAYERS[$index]}"; first_gpu=0
    train_level "$first_level" "$first_gpu" "$LOG_ROOT/probe_${first_level}.log" &
    local first_pid=$!
    index=$((index + 1))
    if [[ "$index" -lt "${#LAYERS[@]}" ]]; then
      second_level="layer_${LAYERS[$index]}"; second_gpu=1
      train_level "$second_level" "$second_gpu" "$LOG_ROOT/probe_${second_level}.log" &
      local second_pid=$!
      wait "$first_pid"; wait "$second_pid"
      index=$((index + 1))
    else
      wait "$first_pid"
    fi
  done
  run_summary
}

run_summary() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_c1_eomt_selective_depth_probe.py" \
    --selective-root "$FULL_ROOT" --model-label "$LABEL" --output-dir "$DURABLE_BASE/full"
}

case "$MODE" in
  smoke) run_smoke ;;
  full) run_full ;;
  summarize) run_summary ;;
esac
