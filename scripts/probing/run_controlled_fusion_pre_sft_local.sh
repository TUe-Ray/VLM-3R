#!/usr/bin/env bash
# C1 calibration and full-policy pre-SFT depth probes for controlled B/C/D/E/H.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(preflight|calibrate|smoke|full|summarize)$ ]]; then
  echo "Usage: $0 {preflight|calibrate|smoke|full|summarize}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CANDIDATES="${CANDIDATES:-B C D E H}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
CALIBRATION_MANIFEST="${CALIBRATION_MANIFEST:-/home/shaoruei/probe_outputs/c1_additive_v1/official/calibration_manifest_32.json}"
BASE_C1="${BASE_C1:-/home/shaoruei/probe_outputs/c1_additive_v1/official/base_r0.json}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/controlled_fusion_pre_sft_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/controlled_fusion_pre_sft_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/controlled_fusion_pre_sft_v1}"
C1_ROOT="$DURABLE_ROOT/c1"
C1_MANIFEST="$C1_ROOT/artifact_manifest.json"
SMOKE_MANIFEST="$DURABLE_ROOT/provenance/smoke_1train_1val.json"
SMOKE_MARKER="$DURABLE_ROOT/provenance/smoke_verification.json"
GPU_WEIGHT_BUDGET="${PRE_SFT_GPU_WEIGHT_BUDGET:-4GiB}"
CPU_OFFLOAD_BUDGET="${PRE_SFT_CPU_OFFLOAD_BUDGET:-45GiB}"
RECYCLE_FEATURE_CACHE="${RECYCLE_FEATURE_CACHE:-1}"

source "$REPO_ROOT/scripts/probing/common_probe_layers.sh"
mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT/provenance" "$LOG_ROOT" "$C1_ROOT"

run() {
  printf '[COMMAND] '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

candidate_field() {
  local id="$1" field="$2"
  case "$id:$field" in
    BASE:label) printf 'pre_sft_base_vlm' ;;
    B:variant) printf 'c1_controlled_b' ;; B:label) printf 'c1_controlled_b' ;;
    B:architecture) printf 'pre_projector_add' ;; B:sources) printf '12' ;; B:layers) printf '' ;;
    C:variant) printf 'c1_controlled_c' ;; C:label) printf 'c1_controlled_c' ;;
    C:architecture) printf 'spatialstack_cross_attn_v1' ;; C:sources) printf '12' ;; C:layers) printf '0' ;;
    D:variant) printf 'c1_controlled_d' ;; D:label) printf 'c1_controlled_d' ;;
    D:architecture) printf 'spatialstack_add' ;; D:sources) printf '12' ;; D:layers) printf '0' ;;
    E:variant) printf 'c1_controlled_e' ;; E:label) printf 'c1_controlled_e' ;;
    E:architecture) printf 'spatialstack_add' ;; E:sources) printf '12,12,12' ;; E:layers) printf '0,1,2' ;;
    H:variant) printf 'c1_controlled_h' ;; H:label) printf 'c1_controlled_h' ;;
    H:architecture) printf 'spatialstack_cross_attn_v1' ;; H:sources) printf '12,12,12' ;; H:layers) printf '0,1,2' ;;
    *) echo "Unsupported controlled candidate/field: $id/$field" >&2; return 2 ;;
  esac
}

validate_candidate_list() {
  local id
  for id in $CANDIDATES; do
    [[ "$id" =~ ^(B|C|D|E|H)$ ]] || { echo "Unsupported CANDIDATES entry: $id" >&2; exit 2; }
  done
}

require_inputs() {
  local path
  for path in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" \
    "$CALIBRATION_MANIFEST" "$BASE_C1" "$DATA_YAML" \
    "$FEATURE_ROOT/scannet/spatial_features/scene0384_00.pt"; do
    [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
  [[ -d "$FORWARD_ROOT" && -d "$TARGET_ROOT" && -d "$FEATURE_ROOT" ]] || {
    echo "Missing required frame, target, or CUT3R root" >&2
    exit 1
  }
  local forbidden_artifact
  forbidden_artifact="$(find "$BASE_MODEL" -type f \( -name adapter_model.bin -o -name non_lora_trainables.bin -o -name adapter_config.json \) -print -quit)"
  if [[ -n "$forbidden_artifact" ]]; then
    echo "Base model contains forbidden post-SFT artifact: $forbidden_artifact" >&2
    exit 1
  fi
}

require_gpu() {
  local purpose="$1"
  local readiness="$DURABLE_ROOT/provenance/gpu_${GPU}_${purpose}_readiness.json"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
    --physical-gpu-id "$GPU" --output "$readiness"
}

preflight() {
  validate_candidate_list
  require_inputs
  require_gpu preflight
  run conda run -n "$ENV_NAME" python -m py_compile \
    "$REPO_ROOT/llava/model/controlled_fusion_pre_sft.py" \
    "$REPO_ROOT/scripts/probing/c1_calibrate_fusion.py" \
    "$REPO_ROOT/scripts/probing/verify_controlled_fusion_c1.py"
  echo "[PASS] controlled-fusion pre-SFT preflight"
}

calibrate_one() {
  local id="$1" architecture output log
  architecture="$(candidate_field "$id" architecture)"
  output="$C1_ROOT/$id/c1.json"
  log="$LOG_ROOT/c1_${id}.log"
  [[ ! -e "$output" ]] || { echo "Refusing to overwrite C1 artifact: $output" >&2; exit 1; }
  mkdir -p "$(dirname "$output")"
  echo "[RUN] C1 calibration candidate=$id architecture=$architecture GPUs=$CUDA_DEVICES output=$output log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/controlled_presft_mpl \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/c1_calibrate_fusion.py" \
    --controlled-fusion-id "$id" --architecture "$architecture" \
    --calibration-manifest "$CALIBRATION_MANIFEST" --output "$output" \
    --base-calibration "$BASE_C1" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --train-data-json "$DATA_YAML" --feature-root "$FEATURE_ROOT" \
    --spatial-features-subdir '12:spatial_features' \
    --forward-frames-root "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" \
    --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 \
    --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" \
    --runtime-root "$CACHE_ROOT/runtime/c1_$id" 2>&1 | tee "$log"
}

lock_c1_artifacts() {
  [[ ! -e "$C1_MANIFEST" ]] || { echo "Refusing to overwrite C1 manifest: $C1_MANIFEST" >&2; exit 1; }
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_controlled_fusion_c1.py" \
    --artifact-root "$C1_ROOT" --base-calibration "$BASE_C1" \
    --calibration-manifest "$CALIBRATION_MANIFEST" --base-model "$BASE_MODEL" \
    --siglip-model "$SIGLIP_MODEL" --output "$C1_MANIFEST"
}

calibrate() {
  preflight
  local id
  for id in B C D E H; do calibrate_one "$id"; done
  lock_c1_artifacts
}

make_smoke_manifest() {
  [[ -f "$SMOKE_MANIFEST" ]] && return 0
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

extract_candidate() {
  local namespace="$1" id="$2" manifest="$3"
  local variant label sources layers output artifact log
  variant="$(candidate_field "$id" variant)"
  label="$(candidate_field "$id" label)"
  sources="$(candidate_field "$id" sources)"
  layers="$(candidate_field "$id" layers)"
  output="$CACHE_ROOT/$namespace/$id"
  artifact="$C1_ROOT/$id/c1.json"
  log="$LOG_ROOT/${namespace}_${id}_extract.log"
  mkdir -p "$output"
  echo "[RUN] pre-SFT extraction candidate=$id GPUs=$CUDA_DEVICES output=$output log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/controlled_presft_mpl \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$label" --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$variant" \
    --c1-calibration-json "$artifact" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$PRE_SFT_FULL_FEATURE_LEVELS_CSV" --sample-indices "$manifest" --output-root "$output" \
    --train-data-json "$DATA_YAML" --feature-root "$FEATURE_ROOT" --spatial-features-subdir '12:spatial_features' \
    --spatialstack-cut3r-layers "$sources" --spatialstack-llm-layers "$layers" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$output/runtime/$label" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" \
    --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" --assert-first-video --resume 2>&1 | tee "$log"
}

extract_baseline() {
  local namespace="$1" manifest="$2"
  local output="$CACHE_ROOT/$namespace/BASE"
  local log="$LOG_ROOT/${namespace}_BASE_extract.log"
  mkdir -p "$output"
  echo "[RUN] pre-SFT Baseline extraction GPUs=$CUDA_DEVICES output=$output log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/controlled_presft_mpl \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label pre_sft_base_vlm --model-loading-mode pre_sft_base_vlm \
    --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$PRE_SFT_FULL_FEATURE_LEVELS_CSV" --sample-indices "$manifest" --output-root "$output" \
    --train-data-json "$DATA_YAML" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$output/runtime/pre_sft_base_vlm" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" \
    --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" --assert-first-video --resume 2>&1 | tee "$log"
}

train_level() {
  local root="$1" id="$2" level="$3" gpu="$4" epochs="$5" namespace="$6"
  local label log
  label="$(candidate_field "$id" label)"
  log="$LOG_ROOT/${namespace}_${id}_${level}_probe.log"
  env CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/controlled_presft_mpl \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$root" --sample-indices "$([[ "$namespace" == smoke ]] && echo "$SMOKE_MANIFEST" || echo "$SAMPLE_INDICES")" \
    --probe-subdir probes --model-labels "$label" --feature-levels "$level" \
    --epochs "$epochs" --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 \
    --probe-seed 0 --experiment-variant "controlled_fusion_${id}_pre_sft" --device cuda:0 \
    --no-write-aggregate --skip-existing 2>&1 | tee "$log"
}

train_candidate_probes() {
  local namespace="$1" id="$2" epochs="$3"
  local root="$CACHE_ROOT/$namespace/$id" levels index first second first_pid second_pid
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
  index=0
  while [[ "$index" -lt "${#levels[@]}" ]]; do
    first="${levels[$index]}"
    train_level "$root" "$id" "$first" 0 "$epochs" "$namespace" & first_pid=$!
    index=$((index + 1))
    if [[ "$index" -lt "${#levels[@]}" ]]; then
      second="${levels[$index]}"
      train_level "$root" "$id" "$second" 1 "$epochs" "$namespace" & second_pid=$!
      wait "$first_pid"
      wait "$second_pid"
      index=$((index + 1))
    else
      wait "$first_pid"
    fi
  done
}

smoke() {
  require_inputs
  [[ -f "$C1_MANIFEST" ]] || { echo "Run '$0 calibrate' first: $C1_MANIFEST" >&2; exit 1; }
  require_gpu smoke
  make_smoke_manifest
  extract_baseline smoke "$SMOKE_MANIFEST"
  train_candidate_probes smoke BASE 2
  local id
  for id in $CANDIDATES; do
    extract_candidate smoke "$id" "$SMOKE_MANIFEST"
    train_candidate_probes smoke "$id" 2
  done
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_controlled_fusion_pre_sft_smoke.py" \
    --cache-root "$CACHE_ROOT/smoke" --artifact-manifest "$C1_MANIFEST" \
    --sample-indices "$SMOKE_MANIFEST" --output "$SMOKE_MARKER"
}

recycle_features() {
  local id="$1" label root target
  label="$(candidate_field "$id" label)"
  root="$CACHE_ROOT/full/$id"
  target="$root/features/$label"
  [[ "$RECYCLE_FEATURE_CACHE" == 1 ]] || return 0
  case "$target" in
    "$CACHE_ROOT"/full/?/features/c1_controlled_*|"$CACHE_ROOT"/full/BASE/features/pre_sft_base_vlm) ;;
    *) echo "Refusing unexpected feature cleanup target: $target" >&2; exit 1 ;;
  esac
  [[ -d "$target" ]] || return 0
  echo "[RECYCLE] removing regenerated feature tensors after verified probe fits: $target"
  rm -rf -- "$target"
}

preserve_candidate_results() {
  local id="$1" label root destination
  label="$(candidate_field "$id" label)"
  root="$CACHE_ROOT/full/$id"
  destination="$DURABLE_ROOT/results/$id"
  [[ ! -e "$destination" ]] || {
    echo "Refusing to overwrite durable controlled-fusion result: $destination" >&2
    exit 1
  }
  mkdir -p "$destination"
  cp -a "$root/probes/$label" "$destination/probes"
  cp -a "$root/features/$label/extraction_provenance.json" "$destination/extraction_provenance.json"
}

full() {
  require_inputs
  [[ -f "$SMOKE_MARKER" ]] || { echo "Run '$0 smoke' successfully before full extraction." >&2; exit 1; }
  require_gpu full
  extract_baseline full "$SAMPLE_INDICES"
  train_candidate_probes full BASE 50
  preserve_candidate_results BASE
  recycle_features BASE
  local id
  for id in $CANDIDATES; do
    extract_candidate full "$id" "$SAMPLE_INDICES"
    train_candidate_probes full "$id" 50
    preserve_candidate_results "$id"
    recycle_features "$id"
  done
  summarize
}

summarize() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_controlled_fusion_pre_sft.py" \
    --results-root "$DURABLE_ROOT/results" --artifact-manifest "$C1_MANIFEST" \
    --sample-indices "$SAMPLE_INDICES" --output-dir "$DURABLE_ROOT/summary"
}

case "$MODE" in
  preflight) preflight ;;
  calibrate) calibrate ;;
  smoke) smoke ;;
  full) full ;;
  summarize) summarize ;;
esac
