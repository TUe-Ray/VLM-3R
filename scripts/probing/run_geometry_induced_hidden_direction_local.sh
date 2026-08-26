#!/usr/bin/env bash
# Probe D_L = H_normal - H_all-geometry-off without changing the probe definition.
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  preflight|smoke|extract|materialize|train|analyze|run) ;;
  *) echo "Usage: $0 preflight|smoke|extract|materialize|train|analyze|run" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
VLM3R_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/geometry_induced_hidden_direction_v1}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/geometry_induced_hidden_direction_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/geometry_induced_hidden_direction_v1}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
SPATIAL_FEATURE_ROOT="${SPATIAL_FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
FEATURE_ROOT="$CACHE_ROOT/full"
PAIR_ROOT="$FEATURE_ROOT/paired_geometry_perturbation"
SMOKE_ROOT="$CACHE_ROOT/smoke"
SMOKE_MANIFEST="$RESULT_ROOT/manifests/canonical_smoke.json"
LAYERS="0,1,2,3,6,9,12,15,18,21,24,27"
SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"

checkpoint_for() {
  case "$1" in
    SS012_old) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703" ;;
    SS012_new) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_45297963" ;;
    SS123) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n" ;;
    SS036) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970" ;;
    *) echo "Unknown model label: $1" >&2; return 2 ;;
  esac
}

injection_layers_for() {
  case "$1" in
    SS012_old|SS012_new) echo "0,1,2" ;;
    SS123) echo "1,2,3" ;;
    SS036) echo "0,3,6" ;;
    *) return 2 ;;
  esac
}

preflight() {
  "$VLM3R_PYTHON" - "$CANONICAL_MANIFEST" "$BASE_MODEL" "$SIGLIP_MODEL" "$FORWARD_ROOT" "$TARGET_ROOT" "$SPATIAL_FEATURE_ROOT" "$DATA_YAML" \
    "$(checkpoint_for SS123)" "$(checkpoint_for SS012_new)" "$(checkpoint_for SS036)" "$(checkpoint_for SS012_old)" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest, base, siglip, forward, target, features, data = map(Path, sys.argv[1:8])
checkpoints = list(map(Path, sys.argv[8:]))
for path in (manifest, base, siglip, forward, target, features, data, *checkpoints):
    if not path.exists():
        raise SystemExit(f"missing required local input: {path}")
payload = json.loads(manifest.read_text())
videos = payload["videos"]
counts = {split: sum(video["split"] == split for video in videos) for split in ("train", "val")}
frame_counts = {split: sum(len(video["frames"]) for video in videos if video["split"] == split) for split in ("train", "val")}
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
if counts != {"train": 1006, "val": 193} or frame_counts != {"train": 2012, "val": 386}:
    raise SystemExit(f"unexpected canonical split coverage: {counts}, {frame_counts}")
if digest != "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e":
    raise SystemExit(f"unexpected canonical manifest SHA-256: {digest}")
print(json.dumps({"status": "PASS", "videos": counts, "frames": frame_counts, "manifest_sha256": digest}))
PY
}

extract_split() {
  local model="$1" manifest="$2" root="$3" pair_root="$4" split="$5" smoke="$6"
  local checkpoint log extra=()
  checkpoint="$(checkpoint_for "$model")"
  log="$LOG_ROOT/${smoke}/${model}_${split}.log"
  mkdir -p "$root" "$pair_root" "$(dirname "$log")"
  if [[ "$smoke" == "smoke" ]]; then extra+=(--geometry-perturbation-verify-normal); fi
  echo "[EXTRACT] model=$model split=$split CUDA_VISIBLE_DEVICES=$CUDA_DEVICES cache=$root pair_cache=$pair_root log=$log"
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$VLM3R_PYTHON" -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$model" --model-loading-mode adapter --model-path "$checkpoint" --feature-preset spatialstack \
    --feature-levels "$(printf 'layer_%s,' ${LAYERS//,/ })" --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --output-root "$root" --sample-indices "$manifest" --data-yaml "$DATA_YAML" \
    --feature-root "$SPATIAL_FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --runtime-root "$root/runtime/$model" \
    --geometry-perturbation-split "$split" --geometry-perturbation-tolerance 1e-6 \
    --geometry-perturbation-feature-cache-root "$pair_root" --assert-first-video --resume "${extra[@]}" 2>&1 | tee "$log"
}

materialize_model() {
  local model="$1" manifest="$2" root="$3" pair_root="$4" allow_partial="$5" delete_source="$6"
  local args=()
  if [[ "$allow_partial" == "1" ]]; then args+=(--allow-partial); fi
  if [[ "$delete_source" == "1" ]]; then args+=(--delete-source); fi
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/materialize_geometry_perturbation_probe_features.py" \
    --source-pair-root "$pair_root" --output-root "$root" --sample-indices "$manifest" --model-label "$model" \
    --layers "$LAYERS" --injection-layers "$(injection_layers_for "$model")" "${args[@]}"
}

extract_one_model() {
  local model="$1"
  extract_split "$model" "$CANONICAL_MANIFEST" "$FEATURE_ROOT" "$PAIR_ROOT" train full
  extract_split "$model" "$CANONICAL_MANIFEST" "$FEATURE_ROOT" "$PAIR_ROOT" val full
}

train_variant_group() {
  local physical_gpu="$1" labels="$2" model="$3"
  echo "[TRAIN] physical_gpu=$physical_gpu models=$labels cache=$FEATURE_ROOT"
  env CUDA_VISIBLE_DEVICES="$physical_gpu" "$VLM3R_PYTHON" -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$FEATURE_ROOT" --sample-indices "$CANONICAL_MANIFEST" --model-labels "$labels" \
    --feature-levels "$(printf 'layer_%s,' ${LAYERS//,/ })" --device cuda:0 --no-write-aggregate \
    2>&1 | tee "$LOG_ROOT/train_gpu${physical_gpu}_${model}.log"
}

train_one_model() {
  local model="$1"
  mkdir -p "$LOG_ROOT"
  # Independent probes share the model's cache but not parameters; keep one
  # worker on each physical GPU while its rolling feature cache is present.
  train_variant_group 0 "$model,${model}__geometry_delta" "$model" &
  local pid0=$!
  train_variant_group 1 "${model}__geometry_off" "$model" &
  local pid1=$!
  wait "$pid0"
  wait "$pid1"
}

recycle_one_model_features() {
  local model="$1" expected=$((12 * 3)) found
  found="$(find "$FEATURE_ROOT/probes" -path "*/layer_*/metrics.json" \( -path "*/$model/*" -o -path "*/${model}__geometry_off/*" -o -path "*/${model}__geometry_delta/*" \) -type f | wc -l)"
  if [[ "$found" -ne "$expected" ]]; then
    echo "Refusing to recycle $model features: found $found/$expected metrics" >&2
    return 1
  fi
  rm -rf "$FEATURE_ROOT/features/$model" "$FEATURE_ROOT/features/${model}__geometry_off" "$FEATURE_ROOT/features/${model}__geometry_delta"
}

run_one_model() {
  local model="$1"
  extract_one_model "$model"
  # A raw pair plus normal/off/delta features is materialized one video at a
  # time; delete the raw payload as soon as its durable variants are written.
  materialize_model "$model" "$CANONICAL_MANIFEST" "$FEATURE_ROOT" "$PAIR_ROOT" 0 1
  train_one_model "$model"
  recycle_one_model_features "$model"
}

analyze() {
  mkdir -p "$RESULT_ROOT/analysis_v1"
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/analyze_geometry_induced_depth_probe.py" \
    --cache-root "$FEATURE_ROOT" --output-dir "$RESULT_ROOT/analysis_v1" \
    --prior-magnitude-summary "/home/shaoruei/probe_outputs/spatialstack_geometry_perturbation_v1/analysis_v1/summary.md"
}

smoke() {
  mkdir -p "$RESULT_ROOT/manifests"
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$CANONICAL_MANIFEST" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
  extract_split SS012_new "$SMOKE_MANIFEST" "$SMOKE_ROOT" "$SMOKE_ROOT/paired_geometry_perturbation" train smoke
  extract_split SS012_new "$SMOKE_MANIFEST" "$SMOKE_ROOT" "$SMOKE_ROOT/paired_geometry_perturbation" val smoke
  materialize_model SS012_new "$SMOKE_MANIFEST" "$SMOKE_ROOT" "$SMOKE_ROOT/paired_geometry_perturbation" 0 0
  env CUDA_VISIBLE_DEVICES=0 "$VLM3R_PYTHON" -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$SMOKE_ROOT" --sample-indices "$SMOKE_MANIFEST" --model-labels "SS012_new__geometry_delta" \
    --feature-levels layer_3 --epochs 1 --early-stop-patience 0 --allow-partial --device cuda:0 --no-write-aggregate \
    2>&1 | tee "$LOG_ROOT/smoke/probe_SS012_new_delta_L3.log"
}

case "$MODE" in
  preflight) preflight ;;
  smoke) preflight; smoke ;;
  extract) preflight; extract_one_model "${MODEL:?Set MODEL to one of SS123, SS012_new, SS036, SS012_old}" ;;
  materialize) materialize_model "${MODEL:?Set MODEL to one model label}" "$CANONICAL_MANIFEST" "$FEATURE_ROOT" "$PAIR_ROOT" 0 0 ;;
  train) train_one_model "${MODEL:?Set MODEL to one model label}" ;;
  analyze) analyze ;;
  run) preflight; for model in SS123 SS012_new SS036 SS012_old; do run_one_model "$model"; done; analyze ;;
esac
