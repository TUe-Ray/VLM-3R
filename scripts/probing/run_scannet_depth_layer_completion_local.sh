#!/usr/bin/env bash
# Direct local-server runner for the ScanNet depth-layer completion study.
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  echo "Usage: $0 preflight|smoke|baseline-l6|baseline-missing|zero-missing|base-smoke|base-full|summary" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-$GPU}"
BASE_ATTN_IMPLEMENTATION="${BASE_ATTN_IMPLEMENTATION:-}"
BASE_GPU_WEIGHT_BUDGET="${BASE_GPU_WEIGHT_BUDGET:-7GiB}"
BASE_CPU_OFFLOAD_BUDGET="${BASE_CPU_OFFLOAD_BUDGET:-45GiB}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
BASELINE_CKPT="${BASELINE_CKPT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2}"
ZERO_CKPT="${ZERO_CKPT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/zero_spatial_features}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
HISTORICAL_BASELINE="${HISTORICAL_BASELINE:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/aggregate/depth_probe_scannet_baseline_results.json}"
CACHE_BASE="${CACHE_ROOT:-/home/shaoruei/probe_cache/scannet_depth_layers_v1}"
DURABLE_BASE="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/scannet_depth_layers_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_depth_layers_v1}"
LOCAL_DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
FULL_CACHE_ROOT="$CACHE_BASE/full"
FULL_DURABLE_ROOT="$DURABLE_BASE/full"
FULL_PROVENANCE_ROOT="$FULL_DURABLE_ROOT/provenance"
PARITY_MARKER="$FULL_PROVENANCE_ROOT/baseline_l6_parity_pass.json"
BASELINE_COMPLETION_MARKER="$FULL_PROVENANCE_ROOT/baseline_missing_layers_complete.json"

mkdir -p "$CACHE_BASE" "$DURABLE_BASE" "$LOG_ROOT"

run() {
  printf '[COMMAND] '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

activate_namespace() {
  local namespace="$1"
  ACTIVE_CACHE_ROOT="$CACHE_BASE/$namespace"
  ACTIVE_DURABLE_ROOT="$DURABLE_BASE/$namespace"
  mkdir -p "$ACTIVE_CACHE_ROOT" "$ACTIVE_DURABLE_ROOT" "$LOG_ROOT"
  export HF_HOME="$ACTIVE_CACHE_ROOT/huggingface"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export TRITON_CACHE_DIR="$ACTIVE_CACHE_ROOT/triton"
  export MPLCONFIGDIR="$ACTIVE_CACHE_ROOT/matplotlib"
}

require_gpu() {
  local readiness_json="$ACTIVE_DURABLE_ROOT/provenance/gpu_${GPU}_readiness.json"
  echo "[RUN] GPU readiness physical_gpu=$GPU namespace=$ACTIVE_CACHE_ROOT output=$readiness_json"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
    --physical-gpu-id "$GPU" --output "$readiness_json"
}

record_provenance() {
  mkdir -p "$ACTIVE_DURABLE_ROOT/provenance"
  cp -a "$SAMPLE_INDICES" "$ACTIVE_DURABLE_ROOT/provenance/scannet_sample_indices.json"
  cp -a "$HISTORICAL_BASELINE" "$ACTIVE_DURABLE_ROOT/provenance/historical_baseline_results.json"
  sha256sum "$SAMPLE_INDICES" "$HISTORICAL_BASELINE" > "$ACTIVE_DURABLE_ROOT/provenance/input_sha256.txt"
}

record_target_stats() {
  local label="$1"
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_depth_probe_targets.py" \
    --output-root "$ACTIVE_CACHE_ROOT" --sample-indices "$SAMPLE_INDICES" \
    --output "$ACTIVE_DURABLE_ROOT/provenance/${label}_target_stats.json"
}

extract_layers() {
  local model_label="$1" checkpoint="$2" preset="$3" layers="$4" manifest="$5" log="$6"
  echo "[RUN] extract model=$model_label layers=$layers gpu=$GPU output=$ACTIVE_CACHE_ROOT log=$log"
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$model_label" --model-path "$checkpoint" --feature-preset "$preset" \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --output-root "$ACTIVE_CACHE_ROOT" --sample-indices "$manifest" --data-yaml "$LOCAL_DATA_YAML" \
    --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" \
    --frames-upbound 32 --dtype float16 --cache-dtype float16 --device cuda:0 \
    --device-map auto \
    --layers $layers --pre-llm-features "" --runtime-root "$ACTIVE_CACHE_ROOT/runtime/$model_label" --resume \
    2>&1 | tee "$log"
}

train_layer() {
  local model_label="$1" level="$2" manifest="$3" log="$4" archive_results="${5:-true}"
  echo "[RUN] train model=$model_label level=$level gpu=$GPU output=$ACTIVE_CACHE_ROOT log=$log"
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
    --output-root "$ACTIVE_CACHE_ROOT" --model-labels "$model_label" --feature-levels "$level" \
    2>&1 | tee -a "$log"
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$ACTIVE_CACHE_ROOT" --sample-indices "$manifest" --probe-subdir probes \
    --model-labels "$model_label" --feature-levels "$level" --epochs 50 --batch-size 32 \
    --lr 1e-3 --early-stop-patience 10 --num-workers 0 --device cuda:0 --no-write-aggregate \
    2>&1 | tee -a "$log"
  if [[ "$archive_results" == "true" ]]; then
    mkdir -p "$ACTIVE_DURABLE_ROOT/probes/$model_label/$level"
    cp -a "$ACTIVE_CACHE_ROOT/probes/$model_label/$level/." "$ACTIVE_DURABLE_ROOT/probes/$model_label/$level/"
  fi
}

run_preflight() {
  local report_json="$1" model_label="$2" checkpoint="$3" layers="$4"
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/validate_scannet_depth_probe.py" \
    --preflight --require-l6 --layers $layers --model-label "$model_label" --sample-indices "$SAMPLE_INDICES" \
    --forward-root "$FORWARD_ROOT" --target-root "$TARGET_ROOT" --sidecar-root "$FEATURE_ROOT" \
    --checkpoint "$checkpoint" --cache-root "$ACTIVE_CACHE_ROOT" --output-root "$ACTIVE_DURABLE_ROOT" \
    --report-json "$report_json" --log-path "$LOG_ROOT/validator.log"
}

require_baseline_parity() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/validate_scannet_depth_probe.py" \
    --verify-parity-marker "$PARITY_MARKER" --model-label vlm3r_baseline --sample-indices "$SAMPLE_INDICES" \
    --checkpoint "$BASELINE_CKPT" --output-root "$FULL_DURABLE_ROOT" \
    --report-json "$FULL_PROVENANCE_ROOT/baseline_l6_parity_marker_verify.json" --log-path "$LOG_ROOT/validator.log"
}

require_baseline_completion() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/validate_scannet_depth_probe.py" \
    --verify-baseline-completion-marker "$BASELINE_COMPLETION_MARKER" --model-label vlm3r_baseline \
    --sample-indices "$SAMPLE_INDICES" --checkpoint "$BASELINE_CKPT" --output-root "$FULL_DURABLE_ROOT" \
    --report-json "$FULL_PROVENANCE_ROOT/baseline_completion_marker_verify.json" --log-path "$LOG_ROOT/validator.log"
}

make_smoke_manifest() {
  SMOKE_MANIFEST="$ACTIVE_CACHE_ROOT/manifests/scannet_smoke_1train_1val.json"
  run conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

run_zero_smoke() {
  activate_namespace smoke/zero_spatial
  make_smoke_manifest
  require_gpu
  extract_layers zero_spatial "$ZERO_CKPT" zero_spatial "6" "$SMOKE_MANIFEST" "$LOG_ROOT/zero_l6_smoke.log"
  train_layer zero_spatial layer_6 "$SMOKE_MANIFEST" "$LOG_ROOT/zero_l6_smoke.log" false
}

base_validator() {
  local mode="$1" output_root="$2" smoke_root="${3:-$2}" smoke_manifest="${4:-}"
  local args=(
    "$REPO_ROOT/scripts/probing/validate_pre_sft_base_depth_probe.py" "$mode"
    --base-model "$BASE_MODEL" --siglip "$SIGLIP_MODEL" --sample-indices "$SAMPLE_INDICES"
    --output-root "$output_root" --smoke-root "$smoke_root" --dtype float16 --device-map auto
    --pre-sft-gpu-weight-budget "$BASE_GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$BASE_CPU_OFFLOAD_BUDGET"
  )
  if [[ -n "$smoke_manifest" ]]; then
    args+=(--smoke-manifest "$smoke_manifest")
  fi
  if [[ -n "$BASE_ATTN_IMPLEMENTATION" ]]; then
    args+=(--attn-implementation "$BASE_ATTN_IMPLEMENTATION")
  fi
  run conda run -n "$ENV_NAME" python -u "${args[@]}"
}

extract_base_features() {
  local layers="$1" pre_llm="$2" manifest="$3" log="$4"
  local args=(
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py"
    --model-label pre_sft_base_vlm --model-loading-mode pre_sft_base_vlm --model-path "$BASE_MODEL"
    --siglip-path "$SIGLIP_MODEL" --output-root "$ACTIVE_CACHE_ROOT" --sample-indices "$manifest"
    --data-yaml "$LOCAL_DATA_YAML" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT"
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32
    --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --layers $layers
    --pre-sft-gpu-weight-budget "$BASE_GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$BASE_CPU_OFFLOAD_BUDGET"
    --pre-llm-features "$pre_llm" --runtime-root "$ACTIVE_CACHE_ROOT/runtime/pre_sft_base_vlm" --resume
  )
  if [[ -n "$BASE_ATTN_IMPLEMENTATION" ]]; then
    args+=(--attn-implementation "$BASE_ATTN_IMPLEMENTATION")
  fi
  echo "[RUN] base extract layers=$layers pre_llm=$pre_llm gpu=$GPU output=$ACTIVE_CACHE_ROOT log=$log"
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u "${args[@]}" 2>&1 | tee "$log"
}

train_base_feature() {
  local level="$1" manifest="$2" log="$3" partial="$4" archive_results="${5:-true}"
  if [[ "$level" == layer_* ]]; then
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
      --output-root "$ACTIVE_CACHE_ROOT" --model-labels pre_sft_base_vlm --feature-levels "$level" \
      2>&1 | tee -a "$log"
  fi
  local args=(
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$ACTIVE_CACHE_ROOT"
    --sample-indices "$manifest" --probe-subdir probes --model-labels pre_sft_base_vlm
    --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10
    --num-workers 0 --device cuda:0 --no-write-aggregate
  )
  if [[ "$partial" == true ]]; then
    args+=(--allow-partial)
  fi
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u "${args[@]}" 2>&1 | tee -a "$log"
  if [[ "$archive_results" == true ]]; then
    mkdir -p "$ACTIVE_DURABLE_ROOT/probes/pre_sft_base_vlm/$level"
    cp -a "$ACTIVE_CACHE_ROOT/probes/pre_sft_base_vlm/$level/." "$ACTIVE_DURABLE_ROOT/probes/pre_sft_base_vlm/$level/"
  fi
}

case "$MODE" in
  preflight)
    activate_namespace full
    run_preflight "$FULL_PROVENANCE_ROOT/baseline_l6_preflight.json" vlm3r_baseline "$BASELINE_CKPT" "6"
    ;;
  smoke)
    activate_namespace smoke/baseline_l6
    make_smoke_manifest
    require_gpu
    extract_layers vlm3r_baseline "$BASELINE_CKPT" original "6" "$SMOKE_MANIFEST" "$LOG_ROOT/baseline_l6_smoke.log"
    train_layer vlm3r_baseline layer_6 "$SMOKE_MANIFEST" "$LOG_ROOT/baseline_l6_smoke.log" false
    ;;
  baseline-l6)
    activate_namespace full
    run_preflight "$FULL_PROVENANCE_ROOT/baseline_l6_preflight.json" vlm3r_baseline "$BASELINE_CKPT" "6"
    require_gpu
    record_provenance
    extract_layers vlm3r_baseline "$BASELINE_CKPT" original "6" "$SAMPLE_INDICES" "$LOG_ROOT/baseline_l6_parity.log"
    record_target_stats baseline_l6_parity
    train_layer vlm3r_baseline layer_6 "$SAMPLE_INDICES" "$LOG_ROOT/baseline_l6_parity.log"
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/validate_scannet_depth_probe.py" \
      --postflight --new-metrics "$ACTIVE_CACHE_ROOT/probes/vlm3r_baseline/layer_6/metrics.json" \
      --preflight-report "$FULL_PROVENANCE_ROOT/baseline_l6_preflight.json" \
      --output-root "$FULL_DURABLE_ROOT" --report-json "$FULL_PROVENANCE_ROOT/baseline_l6_postflight.json" \
      --write-parity-marker "$PARITY_MARKER" --log-path "$LOG_ROOT/validator.log"
    ;;
  baseline-missing)
    activate_namespace full
    require_baseline_parity
    require_gpu
    record_provenance
    extract_layers vlm3r_baseline "$BASELINE_CKPT" original "1 2 12 18 24" "$SAMPLE_INDICES" "$LOG_ROOT/baseline_missing_extract.log"
    record_target_stats baseline_missing_layers
    for level in layer_1 layer_2 layer_12 layer_18 layer_24; do
      train_layer vlm3r_baseline "$level" "$SAMPLE_INDICES" "$LOG_ROOT/baseline_${level}.log"
    done
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/validate_scannet_depth_probe.py" \
      --verify-baseline-completion --layers 1 2 12 18 24 --model-label vlm3r_baseline \
      --sample-indices "$SAMPLE_INDICES" --checkpoint "$BASELINE_CKPT" --cache-root "$ACTIVE_CACHE_ROOT" \
      --output-root "$FULL_DURABLE_ROOT" --report-json "$FULL_PROVENANCE_ROOT/baseline_missing_completion.json" \
      --write-baseline-completion-marker "$BASELINE_COMPLETION_MARKER" --log-path "$LOG_ROOT/validator.log"
    ;;
  zero-missing)
    activate_namespace full
    require_baseline_completion
    run_zero_smoke
    activate_namespace full
    require_gpu
    record_provenance
    extract_layers zero_spatial "$ZERO_CKPT" zero_spatial "1 2 12 18 24" "$SAMPLE_INDICES" "$LOG_ROOT/zero_missing_extract.log"
    record_target_stats zero_missing_layers
    for level in layer_1 layer_2 layer_12 layer_18 layer_24; do
      train_layer zero_spatial "$level" "$SAMPLE_INDICES" "$LOG_ROOT/zero_${level}.log"
    done
    ;;
  base-smoke)
    activate_namespace smoke
    # The selected TITAN V must be proven ready before any smoke data work.
    require_gpu
    make_smoke_manifest
    base_validator --preflight "$ACTIVE_CACHE_ROOT"
    extract_base_features "6" "projected_features" "$SMOKE_MANIFEST" "$LOG_ROOT/pre_sft_base_vlm_smoke.log"
    train_base_feature layer_6 "$SMOKE_MANIFEST" "$LOG_ROOT/pre_sft_base_vlm_smoke.log" true false
    base_validator --write-smoke-attestation "$ACTIVE_CACHE_ROOT" "$ACTIVE_CACHE_ROOT" "$SMOKE_MANIFEST"
    ;;
  base-full)
    activate_namespace full
    # A stale smoke marker is never sufficient: this compares code, configs,
    # split, loading mode, placement, and backend identity before the full run.
    base_validator --verify-smoke-attestation "$ACTIVE_CACHE_ROOT" "$CACHE_BASE/smoke"
    base_validator --preflight "$ACTIVE_CACHE_ROOT" "$CACHE_BASE/smoke"
    require_gpu
    extract_base_features "0 1 2 3 6 9 12 15 18 21 24 27" "projected_features" "$SAMPLE_INDICES" "$LOG_ROOT/pre_sft_base_vlm_full.log"
    for level in projected_features layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12 layer_15 layer_18 layer_21 layer_24 layer_27; do
      train_base_feature "$level" "$SAMPLE_INDICES" "$LOG_ROOT/pre_sft_base_vlm_full.log" false
    done
    base_validator --verify-full "$ACTIVE_CACHE_ROOT" "$CACHE_BASE/smoke"
    ;;
  summary)
    activate_namespace full
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_scannet_depth_layer_completion.py" \
      --durable-root "$FULL_DURABLE_ROOT" --historical-baseline "$HISTORICAL_BASELINE"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
