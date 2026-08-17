#!/usr/bin/env bash
# Direct local-server runner for the ScanNet depth-layer completion study.
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  echo "Usage: $0 preflight|smoke|baseline-l6|baseline-missing|zero-missing|summary" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
BASELINE_CKPT="${BASELINE_CKPT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2}"
ZERO_CKPT="${ZERO_CKPT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/zero_spatial_features}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
HISTORICAL_BASELINE="${HISTORICAL_BASELINE:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/aggregate/depth_probe_scannet_baseline_results.json}"
HISTORICAL_ZERO="${HISTORICAL_ZERO:-}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/scannet_depth_layers_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/scannet_depth_layers_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_depth_layers_v1}"
LOCAL_DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"

mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT" "$LOG_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

run() {
  printf '[COMMAND] '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

require_gpu() {
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  conda run -n "$ENV_NAME" python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
}

record_provenance() {
  mkdir -p "$DURABLE_ROOT/provenance"
  cp -a "$SAMPLE_INDICES" "$DURABLE_ROOT/provenance/scannet_sample_indices.json"
  cp -a "$HISTORICAL_BASELINE" "$DURABLE_ROOT/provenance/historical_baseline_results.json"
  if [[ -n "$HISTORICAL_ZERO" ]]; then
    cp -a "$HISTORICAL_ZERO" "$DURABLE_ROOT/provenance/historical_zero_results.json"
  fi
  sha256sum "$SAMPLE_INDICES" "$HISTORICAL_BASELINE" > "$DURABLE_ROOT/provenance/input_sha256.txt"
}

record_target_stats() {
  local label="$1"
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_depth_probe_targets.py" \
    --output-root "$CACHE_ROOT" --sample-indices "$SAMPLE_INDICES" \
    --output "$DURABLE_ROOT/provenance/${label}_target_stats.json"
}

extract_layers() {
  local model_label="$1"
  local checkpoint="$2"
  local preset="$3"
  local layers="$4"
  local manifest="$5"
  local log="$6"
  echo "[RUN] extract model=$model_label layers=$layers gpu=$GPU output=$CACHE_ROOT log=$log"
  run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$model_label" --model-path "$checkpoint" --feature-preset "$preset" \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --output-root "$CACHE_ROOT" --sample-indices "$manifest" --data-yaml "$LOCAL_DATA_YAML" \
    --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" \
    --frames-upbound 32 --dtype float16 --cache-dtype float16 --device cuda:0 \
    --layers $layers --pre-llm-features "" --runtime-root "$CACHE_ROOT/runtime/$model_label" --resume \
    2>&1 | tee "$log"
}

train_layer() {
  local model_label="$1"
  local level="$2"
  local manifest="$3"
  local log="$4"
  local archive_results="${5:-true}"
  echo "[RUN] train model=$model_label level=$level gpu=$GPU output=$CACHE_ROOT log=$log"
  run conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
    --output-root "$CACHE_ROOT" --model-labels "$model_label" --feature-levels "$level" \
    2>&1 | tee -a "$log"
  run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$CACHE_ROOT" --sample-indices "$manifest" --probe-subdir probes \
    --model-labels "$model_label" --feature-levels "$level" --epochs 50 --batch-size 32 \
    --lr 1e-3 --early-stop-patience 10 --num-workers 0 --device cuda:0 --no-write-aggregate \
    2>&1 | tee -a "$log"
  if [[ "$archive_results" == "true" ]]; then
    mkdir -p "$DURABLE_ROOT/probes/$model_label/$level"
    cp -a "$CACHE_ROOT/probes/$model_label/$level/." "$DURABLE_ROOT/probes/$model_label/$level/"
  fi
}

case "$MODE" in
  preflight)
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/validate_scannet_depth_probe.py" \
      --preflight --require-l6 --layers 6 --model-label vlm3r_baseline \
      --sample-indices "$SAMPLE_INDICES" --forward-root "$FORWARD_ROOT" \
      --target-root "$TARGET_ROOT" --sidecar-root "$FEATURE_ROOT" \
      --checkpoint "$BASELINE_CKPT" --cache-root "$CACHE_ROOT" --output-root "$DURABLE_ROOT"
    ;;
  smoke)
    require_gpu
    SMOKE_MANIFEST="$CACHE_ROOT/manifests/scannet_smoke_1train_1val.json"
    run conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
      --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
    extract_layers vlm3r_baseline "$BASELINE_CKPT" original "6" "$SMOKE_MANIFEST" "$LOG_ROOT/baseline_l6_smoke.log"
    train_layer vlm3r_baseline layer_6 "$SMOKE_MANIFEST" "$LOG_ROOT/baseline_l6_smoke.log" false
    ;;
  baseline-l6)
    require_gpu
    record_provenance
    extract_layers vlm3r_baseline "$BASELINE_CKPT" original "6" "$SAMPLE_INDICES" "$LOG_ROOT/baseline_l6_parity.log"
    record_target_stats baseline_l6_parity
    train_layer vlm3r_baseline layer_6 "$SAMPLE_INDICES" "$LOG_ROOT/baseline_l6_parity.log"
    ;;
  baseline-missing)
    require_gpu
    record_provenance
    extract_layers vlm3r_baseline "$BASELINE_CKPT" original "1 2 12 18 24" "$SAMPLE_INDICES" "$LOG_ROOT/baseline_missing_extract.log"
    record_target_stats baseline_missing_layers
    for level in layer_1 layer_2 layer_12 layer_18 layer_24; do
      train_layer vlm3r_baseline "$level" "$SAMPLE_INDICES" "$LOG_ROOT/baseline_${level}.log"
    done
    ;;
  zero-missing)
    require_gpu
    record_provenance
    extract_layers zero_spatial "$ZERO_CKPT" zero_spatial "1 2 12 18 24" "$SAMPLE_INDICES" "$LOG_ROOT/zero_missing_extract.log"
    record_target_stats zero_missing_layers
    for level in layer_1 layer_2 layer_12 layer_18 layer_24; do
      train_layer zero_spatial "$level" "$SAMPLE_INDICES" "$LOG_ROOT/zero_${level}.log"
    done
    ;;
  summary)
    args=(--durable-root "$DURABLE_ROOT" --historical-baseline "$HISTORICAL_BASELINE")
    if [[ -n "$HISTORICAL_ZERO" ]]; then
      args+=(--historical-zero "$HISTORICAL_ZERO")
    fi
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_scannet_depth_layer_completion.py" "${args[@]}"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
