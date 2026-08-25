#!/usr/bin/env bash
# Local C1 SpatialStack depth-subspace diagnostic; no SFT or model updates.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-smoke}"
ENV_NAME="${ENV_NAME:-vlm3r}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
BASE="${BASE:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP="${SIGLIP:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
CUT3R_ROOT="${CUT3R_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/depth_subspace_occupancy_v1}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/depth_subspace_occupancy}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/depth_subspace_occupancy}"
C1_012="${C1_012:-/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json}"
C1_123="${C1_123:-/home/shaoruei/probe_outputs/c1_ss_add_123/official/spatialstack_add.json}"
C1_036="${C1_036:-/home/shaoruei/probe_outputs/c1_ss_add_036/official/spatialstack_add.json}"
GPU_WEIGHT_BUDGET="${GPU_WEIGHT_BUDGET:-4GiB}"
SEED="${SEED:-42}"
FROZEN_SELECTION="${FROZEN_SELECTION:-}"
FEATURE_LEVELS="fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
SPATIAL_FEATURE_MAP="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"

mkdir -p "$CACHE_ROOT" "$RESULT_ROOT/manifests" "$LOG_ROOT/$MODE"
cd "$REPO_ROOT"

make_manifests() {
  local exclusion
  exclusion="$(jq -r '.calibration_manifest // empty' "$C1_012")"
  local args=(--source-manifest "$SOURCE_MANIFEST" --output-dir "$RESULT_ROOT/manifests" --seed "$SEED")
  if [[ -n "$exclusion" && -f "$exclusion" ]]; then
    args+=(--exclude-video-manifest "$exclusion")
  fi
  conda run -n "$ENV_NAME" python -u scripts/probing/make_depth_subspace_manifests.py "${args[@]}"
}

extract_model() {
  local label="$1"
  local schedule="$2"
  local artifact="$3"
  local manifest="$4"
  local on_off_split="$5"
  local limit_videos="${6:-}"
  local log="$LOG_ROOT/$MODE/${label}_extract.log"
  echo "[RUN] $label schedule=$schedule CUDA_VISIBLE_DEVICES=$CUDA_DEVICES log=$log"
  local limit_args=()
  if [[ -n "$limit_videos" ]]; then
    limit_args=(--limit-videos "$limit_videos")
  fi
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 MPLCONFIGDIR=/tmp/depth_subspace_mpl \
    conda run -n "$ENV_NAME" python -u scripts/probing/extract_depth_probe_features.py \
      --model-label "$label" --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant c1_ss_add \
      --c1-calibration-json "$artifact" --model-path "$BASE" --siglip-path "$SIGLIP" \
      --spatialstack-cut3r-layers 6,9,12 --spatialstack-llm-layers "$schedule" \
      --feature-preset llm_only --feature-levels "$FEATURE_LEVELS" \
      --sample-indices "$manifest" --output-root "$CACHE_ROOT" --train-data-json "$DATA_YAML" \
      --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
      --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --feature-root "$CUT3R_ROOT" \
      --spatial-features-subdir "$SPATIAL_FEATURE_MAP" --frames-upbound 32 \
      --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
      --runtime-root "$CACHE_ROOT/runtime/$label" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" \
      --geometry-on-off-split "$on_off_split" --assert-first-video --resume "${limit_args[@]}" 2>&1 | tee "$log"
}

run_analysis() {
  local manifest="$1"
  local output="$2"
  shift 2
  echo "[RUN] cache-only analysis output=$output"
  MPLCONFIGDIR=/tmp/depth_subspace_mpl conda run -n "$ENV_NAME" python -u \
    scripts/probing/analyze_depth_subspace_occupancy.py \
    --cache-root "$CACHE_ROOT" --manifest "$manifest" --output-dir "$output" \
    "$@" --seed "$SEED" 2>&1 | tee "$LOG_ROOT/$MODE/analysis.log"
}

summarize_pilot() {
  local output="$1"
  MPLCONFIGDIR=/tmp/depth_subspace_mpl conda run -n "$ENV_NAME" python -u \
    scripts/probing/summarize_depth_subspace_occupancy.py --result-dir "$output" \
    2>&1 | tee "$LOG_ROOT/$MODE/summary.log"
}

make_manifests
case "$MODE" in
  forward-smoke)
    manifest="$RESULT_ROOT/manifests/depth_subspace_smoke_v1.json"
    extract_model SS012 0,1,2 "$C1_012" "$manifest" train 2
    ;;
  smoke)
    manifest="$RESULT_ROOT/manifests/depth_subspace_smoke_v1.json"
    extract_model SS012 0,1,2 "$C1_012" "$manifest" dev_eval
    # The smoke verifies cache loading, ridge/VF numerics, and all feature
    # points for one schedule.  Architecture discrimination requires the
    # three-model pilot, so do not fabricate a one-model profile test.
    run_analysis "$manifest" "$RESULT_ROOT/smoke" --models SS012 --stages v1
    ;;
  pilot)
    manifest="$RESULT_ROOT/manifests/depth_subspace_pilot_v1.json"
    extract_model SS012 0,1,2 "$C1_012" "$manifest" dev_eval
    extract_model SS123 1,2,3 "$C1_123" "$manifest" dev_eval
    extract_model SS036 0,3,6 "$C1_036" "$manifest" dev_eval
    run_analysis "$manifest" "$RESULT_ROOT/development"
    summarize_pilot "$RESULT_ROOT/development"
    ;;
  confirmation)
    if [[ -z "$FROZEN_SELECTION" || ! -f "$FROZEN_SELECTION" ]]; then
      echo "[ERROR] Confirmation is locked until a stable development selection is frozen." >&2
      echo "Set FROZEN_SELECTION=/absolute/path/to/frozen_selection.json; this protects the 12 held-out videos from metric selection." >&2
      exit 2
    fi
    manifest="$RESULT_ROOT/manifests/depth_subspace_confirmation_v1.json"
    extract_model SS012 0,1,2 "$C1_012" "$manifest" confirmation
    extract_model SS123 1,2,3 "$C1_123" "$manifest" confirmation
    extract_model SS036 0,3,6 "$C1_036" "$manifest" confirmation
    confirmation_output="$RESULT_ROOT/confirmation_$(basename "${FROZEN_SELECTION%.json}")"
    echo "[RUN] frozen confirmation output=$confirmation_output"
    MPLCONFIGDIR=/tmp/depth_subspace_mpl conda run -n "$ENV_NAME" python -u \
      scripts/probing/confirm_depth_subspace_occupancy.py \
      --frozen-selection "$FROZEN_SELECTION" --cache-root "$CACHE_ROOT" \
      --output-dir "$confirmation_output" 2>&1 | tee "$LOG_ROOT/$MODE/confirmation.log"
    ;;
  manifests)
    ;;
  *)
    echo "usage: $0 {manifests|forward-smoke|smoke|pilot|confirmation}" >&2
    exit 2
    ;;
esac
