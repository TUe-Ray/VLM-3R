#!/usr/bin/env bash
# Sequential local runner for only missing final ScanNet layer-wise probe points.
set -euo pipefail

# systemd user services do not source the interactive shell startup files.
# Load the same Conda installation used by the validated manual commands.
export PATH="/home/shaoruei/miniconda3/bin:${PATH:-}"
if [[ -f /home/shaoruei/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/shaoruei/miniconda3/etc/profile.d/conda.sh
fi

MODE="${1:-}"
LABEL="${2:-}"
if [[ "$MODE" != "preflight" && "$MODE" != "run-one" && "$MODE" != "smoke-one" && "$MODE" != "smoke-all" ]]; then
  echo "Usage: $0 preflight | run-one <model-label> | smoke-one <model-label> | smoke-all" >&2
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
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/scannet_final_layerwise_depth_completion}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/scannet_final_layerwise_depth_completion}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_final_layerwise_depth_completion}"
LOCAL_DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
REPORT="$LOG_ROOT/cpu_preflight.json"

mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT" "$LOG_ROOT"

preflight() {
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/prepare_scannet_final_layerwise_depth_completion.py" \
    --sample-indices "$SAMPLE_INDICES" --durable-root "$DURABLE_ROOT" --cache-root "$CACHE_ROOT" \
    --report "$REPORT" --print-commands --write-summary
}

if [[ "$MODE" == "preflight" ]]; then
  preflight
  exit 0
fi

smoke_manifest() {
  local root="$1"
  local manifest="$root/manifests/scannet_smoke_1train_1val.json"
  if [[ ! -f "$manifest" ]]; then
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
      --sample-indices "$SAMPLE_INDICES" --output "$manifest" --train-videos 1 --val-videos 1 >/dev/null
  fi
  printf '%s\n' "$manifest"
}

configure_model() {
  case "$1" in
    cut3r_spatialstack_44323703)
      CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703"; PRESET=spatialstack
      LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
    cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n)
      CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n"; PRESET=spatialstack
      LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
    cut3r_spatialstack_d2_pointmap_45457911)
      CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_d2_pointmap_45457911"; PRESET=spatialstack
      LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
    cut3r_spatialstack_cross_attn_45303862)
      CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_cross_attn_45303862"; PRESET=spatialstack
      LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
    cut3r_depth_loss_43817021)
      CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_depth_loss_43817021"; PRESET=original
      LEVELS="layer_1,layer_2,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="spatial_features" ;;
    *) echo "Unsupported model label: $1" >&2; return 2 ;;
  esac
}

smoke_one() {
  local smoke_label="$1"
  configure_model "$smoke_label"
  local active_cache="$CACHE_ROOT/smoke/$smoke_label"
  local smoke_log="$LOG_ROOT/smoke/$smoke_label.log"
  local manifest
  mkdir -p "$active_cache" "$LOG_ROOT/smoke"
  manifest="$(smoke_manifest "$CACHE_ROOT/smoke")"
  echo "[SMOKE] model=$smoke_label checkpoint=$CHECKPOINT CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$active_cache log=$smoke_log"
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$smoke_label" --model-path "$CHECKPOINT" --feature-preset "$PRESET" \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$active_cache" \
    --sample-indices "$manifest" --data-yaml "$LOCAL_DATA_YAML" --feature-root "$FEATURE_ROOT" \
    --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 \
    --device cuda:0 --device-map auto --feature-levels "$LEVELS" --runtime-root "$active_cache/runtime" \
    --assert-first-video --resume 2>&1 | tee "$smoke_log"
  IFS=, read -r -a smoke_levels <<< "$LEVELS"
  for level in "${smoke_levels[@]}"; do
    if [[ "$level" == layer_* ]]; then
      conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
        --output-root "$active_cache" --model-labels "$smoke_label" --feature-levels "$level" 2>&1 | tee -a "$smoke_log"
    fi
    env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$active_cache" --sample-indices "$manifest" \
      --probe-subdir probes --model-labels "$smoke_label" --feature-levels "$level" --epochs "${SMOKE_EPOCHS:-2}" \
      --batch-size "${SMOKE_BATCH_SIZE:-2}" --lr 1e-3 --early-stop-patience 1 --num-workers 0 --device cuda:0 \
      --allow-partial --no-write-aggregate 2>&1 | tee -a "$smoke_log"
  done
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_scannet_final_layerwise_smoke.py" \
    --output-root "$active_cache" --model-label "$smoke_label" --feature-levels "$LEVELS" --manifest "$manifest" \
    --report "$active_cache/smoke_verification.json" 2>&1 | tee -a "$smoke_log"
  echo "[SMOKE DONE] $smoke_label; retained isolated smoke artifacts at $active_cache"
}

if [[ "$MODE" == "smoke-one" ]]; then
  smoke_one "$LABEL"
  exit 0
fi
if [[ "$MODE" == "smoke-all" ]]; then
  for smoke_label in \
    cut3r_spatialstack_44323703 \
    cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n \
    cut3r_spatialstack_d2_pointmap_45457911 \
    cut3r_spatialstack_cross_attn_45303862 \
    cut3r_depth_loss_43817021; do
    smoke_one "$smoke_label"
  done
  exit 0
fi

case "$LABEL" in
  cut3r_spatialstack_44323703)
    CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703"; PRESET=spatialstack
    LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
  cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n)
    CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n"; PRESET=spatialstack
    LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
  cut3r_spatialstack_cross_attn_45303862)
    CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_cross_attn_45303862"; PRESET=spatialstack
    LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
  cut3r_depth_loss_43817021)
    CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_depth_loss_43817021"; PRESET=original
    LEVELS="layer_1,layer_2,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="spatial_features" ;;
  cut3r_spatialstack_d2_pointmap_45457911)
    CHECKPOINT="/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_d2_pointmap_45457911"; PRESET=spatialstack
    LEVELS="siglip_output,projected_features,layer_12,layer_18,layer_24"; SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features" ;;
  *) echo "Unsupported or unresolved label: $LABEL" >&2; exit 2 ;;
esac

preflight
if [[ ! -s "$REPORT" ]]; then
  echo "[ERROR] Preflight report missing or empty: $REPORT" >&2
  exit 1
fi
if ! jq -e --arg model_label "$LABEL" '.models[] | select(.label == $model_label)' "$REPORT" >/dev/null; then
  echo "[ERROR] Preflight report has no model entry for $LABEL: $REPORT" >&2
  exit 1
fi
mapfile -t MISSING < <(jq -r --arg model_label "$LABEL" '.models[] | select(.label == $model_label) | (.actually_missing // [])[]' "$REPORT")
if (( ${#MISSING[@]} == 0 )); then
  echo "[INFO] All requested valid points already exist for $LABEL; no CUDA work."
  exit 0
fi

# The user must invoke this only after GPUs are available.  This runner uses a
# single process with model sharding across both exposed TITAN Vs.
echo "[RUN] model=$LABEL checkpoint=$CHECKPOINT CUDA_VISIBLE_DEVICES=$CUDA_DEVICES log=$LOG_ROOT/$LABEL.log"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
ACTIVE_CACHE="$CACHE_ROOT/final_layerwise/$LABEL"
mkdir -p "$ACTIVE_CACHE" "$DURABLE_ROOT/provenance/$LABEL"
IFS=, read -r -a REQUESTED <<< "$LEVELS"
REQUESTED_CSV=$(IFS=,; echo "${MISSING[*]}")

env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n "$ENV_NAME" python -u \
  "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
  --model-label "$LABEL" --model-path "$CHECKPOINT" --feature-preset "$PRESET" \
  --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$ACTIVE_CACHE" \
  --sample-indices "$SAMPLE_INDICES" --data-yaml "$LOCAL_DATA_YAML" --feature-root "$FEATURE_ROOT" \
  --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
  --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 \
  --device cuda:0 --device-map auto --feature-levels "$REQUESTED_CSV" --runtime-root "$ACTIVE_CACHE/runtime" \
  --assert-first-video --resume 2>&1 | tee "$LOG_ROOT/$LABEL.log"

for level in "${MISSING[@]}"; do
  if [[ "$level" == layer_* ]]; then
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
      --output-root "$ACTIVE_CACHE" --model-labels "$LABEL" --feature-levels "$level" 2>&1 | tee -a "$LOG_ROOT/$LABEL.log"
  fi
done

# Extraction deliberately uses both GPUs in one sharded process.  Once it is
# complete, probe jobs are independent: run up to one worker per physical GPU
# while keeping the model variants themselves strictly sequential.
for ((start=0; start<${#MISSING[@]}; start+=2)); do
  train_pids=()
  train_levels=()
  for slot in 0 1; do
    idx=$((start + slot))
    (( idx < ${#MISSING[@]} )) || continue
    level="${MISSING[$idx]}"
    physical_gpu="$slot"
    level_log="$LOG_ROOT/$LABEL.$level.log"
    echo "[TRAIN] model=$LABEL level=$level physical_gpu=$physical_gpu log=$level_log" | tee -a "$LOG_ROOT/$LABEL.log"
    (
      env CUDA_VISIBLE_DEVICES="$physical_gpu" conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
        --output-root "$ACTIVE_CACHE" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes --model-labels "$LABEL" \
        --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 \
        --device cuda:0 --no-write-aggregate 2>&1 | tee "$level_log"
    ) &
    train_pids+=("$!")
    train_levels+=("$level")
  done
  for idx in "${!train_pids[@]}"; do
    if ! wait "${train_pids[$idx]}"; then
      echo "[ERROR] Probe training failed: model=$LABEL level=${train_levels[$idx]}" >&2
      exit 1
    fi
  done
done

for level in "${MISSING[@]}"; do
  mkdir -p "$DURABLE_ROOT/probes/$LABEL/$level"
  cp -a "$ACTIVE_CACHE/probes/$LABEL/$level/." "$DURABLE_ROOT/probes/$LABEL/$level/"
done
cp -a "$ACTIVE_CACHE/features/$LABEL/extraction_provenance.json" "$DURABLE_ROOT/provenance/$LABEL/"

preflight
if jq -e --arg model_label "$LABEL" '.models[] | select(.label == $model_label) | ((.actually_missing // []) | length == 0)' "$REPORT" >/dev/null; then
  # Cleanup is intentionally limited to this newly-created, model-specific
  # cache namespace, after durable probe metrics/checkpoints have validated.
  case "$ACTIVE_CACHE" in "$CACHE_ROOT"/final_layerwise/*) rm -rf -- "$ACTIVE_CACHE" ;; *) exit 1 ;; esac
  echo "[DONE] Durable results verified and temporary cache removed: $ACTIVE_CACHE"
else
  echo "[ERROR] Durable verification failed; retaining cache for diagnosis: $ACTIVE_CACHE" >&2
  exit 1
fi
