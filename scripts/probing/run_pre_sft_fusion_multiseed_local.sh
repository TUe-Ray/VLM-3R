#!/usr/bin/env bash
# Direct local-server runner for the pre-SFT SpatialStack vs VLM3R seed study.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "aggregate" ]]; then
  echo "Usage: $0 smoke|full|aggregate" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
FUSION_SEEDS="${FUSION_SEEDS:-0 1}"
PROBE_SEED="${PROBE_SEED:-0}"
COMMON_MODEL_INIT_SEED="${COMMON_MODEL_INIT_SEED:-0}"
VARIANTS="${VARIANTS:-ss_identity vlm3r_native}"
LLM_LAYERS="${LLM_LAYERS:-0 2 9 27}"

BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
LOCAL_DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
CACHE_BASE="${CACHE_ROOT:-/home/shaoruei/probe_cache/pre_sft_fusion_multiseed_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/pre_sft_fusion_multiseed_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/pre_sft_fusion_multiseed_v1}"
RECYCLE_FEATURE_CACHE="${RECYCLE_FEATURE_CACHE:-1}"

mkdir -p "$CACHE_BASE" "$DURABLE_ROOT" "$LOG_ROOT"

run() {
  printf '[COMMAND] '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

label_for() {
  local variant="$1" seed="$2"
  printf 'pre_sft_%s_seed%s' "$variant" "$seed"
}

feature_levels_for() {
  local variant="$1"
  local levels=""
  if [[ "$variant" == "vlm3r_native" ]]; then
    levels="fusion_output,projected_features,"
  fi
  local layer
  for layer in $LLM_LAYERS; do
    levels+="layer_${layer},"
  done
  printf '%s' "${levels%,}"
}

require_gpu() {
  local output="$DURABLE_ROOT/provenance/gpu_${GPU}_readiness.json"
  echo "[RUN] GPU readiness physical_gpu=$GPU visible=$CUDA_DEVICES output=$output"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
    --physical-gpu-id "$GPU" --output "$output"
}

make_smoke_manifest() {
  SMOKE_MANIFEST="$CACHE_BASE/manifests/scannet_smoke_1train_1val.json"
  mkdir -p "$(dirname "$SMOKE_MANIFEST")"
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

marker_path() {
  local variant="$1" seed="$2"
  printf '%s' "$DURABLE_ROOT/smoke_markers/${variant}_seed${seed}.pass"
}

require_smoke_markers() {
  local variant seed marker
  for variant in $VARIANTS; do
    for seed in $FUSION_SEEDS; do
      marker="$(marker_path "$variant" "$seed")"
      [[ -f "$marker" ]] || {
        echo "Missing smoke marker $marker. Run '$0 smoke' before full extraction." >&2
        exit 1
      }
    done
  done
}

recycle_seed_features() {
  local seed_root="$1" label="$2"
  [[ "$RECYCLE_FEATURE_CACHE" == "1" ]] || return 0
  local target="$seed_root/features/$label"
  case "$target" in
    "$CACHE_BASE"/full/*/features/pre_sft_*) ;;
    *) echo "Refusing unexpected cleanup target: $target" >&2; exit 1 ;;
  esac
  [[ -d "$target" ]] || return 0
  echo "[CLEANUP] Recycling verified feature cache $target"
  rm -rf -- "$target"
}

run_seed() {
  local namespace="$1" manifest="$2" variant="$3" seed="$4"
  local label levels seed_root log output_init spatial_subdir
  label="$(label_for "$variant" "$seed")"
  levels="$(feature_levels_for "$variant")"
  seed_root="$CACHE_BASE/$namespace/$variant/seed_${seed}"
  log="$LOG_ROOT/${namespace//\//_}_${variant}_seed${seed}.log"
  output_init=""
  if [[ "$variant" == "ss_identity" ]]; then output_init="identity"; fi
  if [[ "$variant" == "ss_zero" ]]; then output_init="zero"; fi
  spatial_subdir="spatial_features"
  if [[ "$variant" == ss_* ]]; then
    spatial_subdir="6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features"
  fi
  mkdir -p "$seed_root" "$DURABLE_ROOT/probes/$label" "$DURABLE_ROOT/provenance"

  echo "[RUN] namespace=$namespace variant=$variant fusion_seed=$seed probe_seed=$PROBE_SEED layers=$LLM_LAYERS"
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$variant" --fusion-init-seed "$seed" \
    --common-model-init-seed "$COMMON_MODEL_INIT_SEED" \
    --model-label "$label" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --feature-preset llm_only --feature-levels "$levels" \
    --output-root "$seed_root" --sample-indices "$manifest" --data-yaml "$LOCAL_DATA_YAML" \
    --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$spatial_subdir" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto \
    --layers $LLM_LAYERS --runtime-root "$seed_root/runtime/$label" --assert-first-video --resume \
    2>&1 | tee "$log"

  local level
  IFS=',' read -r -a levels_array <<< "$levels"
  for level in "${levels_array[@]}"; do
    run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
      --output-root "$seed_root" --sample-indices "$manifest" --probe-subdir probes \
      --model-labels "$label" --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 \
      --early-stop-patience 10 --num-workers 0 --device cuda:0 --no-write-aggregate \
      --probe-seed "$PROBE_SEED" --experiment-variant "$variant" --fusion-init-seed "$seed" \
      --spatialstack-output-init "$output_init" --shared-llm-layers "0,2,9,27" \
      2>&1 | tee -a "$log"
  done
  cp -a "$seed_root/probes/$label/." "$DURABLE_ROOT/probes/$label/"
  cp -a "$seed_root/features/$label/extraction_provenance.json" \
    "$DURABLE_ROOT/provenance/${label}_extraction_provenance.json"
  if [[ "$namespace" == "smoke" ]]; then
    local marker
    marker="$(marker_path "$variant" "$seed")"
    mkdir -p "$(dirname "$marker")"
    printf 'variant=%s\nfusion_init_seed=%s\nprobe_seed=%s\nlayers=%s\n' \
      "$variant" "$seed" "$PROBE_SEED" "$LLM_LAYERS" > "$marker"
  else
    recycle_seed_features "$seed_root" "$label"
  fi
}

case "$MODE" in
  smoke)
    make_smoke_manifest
    MANIFEST="$SMOKE_MANIFEST"
    require_gpu
    for variant in $VARIANTS; do
      for seed in $FUSION_SEEDS; do run_seed smoke "$MANIFEST" "$variant" "$seed"; done
    done
    ;;
  full)
    require_smoke_markers
    require_gpu
    for variant in $VARIANTS; do
      for seed in $FUSION_SEEDS; do run_seed full "$SAMPLE_INDICES" "$variant" "$seed"; done
    done
    ;;
  aggregate)
    seed_csv="${FUSION_SEEDS// /,}"
    variant_csv="${VARIANTS// /,}"
    run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/aggregate_pre_sft_fusion_multiseed.py" \
      --output-root "$DURABLE_ROOT" --variants "$variant_csv" --fusion-seeds "$seed_csv" --probe-seed "$PROBE_SEED"
    ;;
esac
