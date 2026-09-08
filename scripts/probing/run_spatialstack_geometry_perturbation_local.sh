#!/usr/bin/env bash
# Paired inference-only SpatialStack residual-mask perturbation experiment.
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  preflight|manifest|smoke|extract|analyze|run) ;;
  *) echo "Usage: $0 preflight | manifest | smoke | extract | analyze | run" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
VLM3R_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/spatialstack_geometry_perturbation_v1}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/spatialstack_geometry_perturbation_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/spatialstack_geometry_perturbation_v1}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
DEVELOPMENT_MANIFEST="${DEVELOPMENT_MANIFEST:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/manifests/depth_subspace_pilot_v1.json}"
EVALUATION_MANIFEST="${EVALUATION_MANIFEST:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/manifests/depth_subspace_confirmation_v1.json}"
MANIFEST="$RESULT_ROOT/manifests/post_sft_geometry_perturbation_v1.json"
SMOKE_MANIFEST="$RESULT_ROOT/manifests/post_sft_geometry_perturbation_v1_smoke.json"
FEATURE_LEVELS="layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"

checkpoint_for() {
  case "$1" in
    SS012_old) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703" ;;
    SS012_new) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_45297963" ;;
    SS123) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n" ;;
    SS036) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970" ;;
    *) echo "unknown model label: $1" >&2; return 2 ;;
  esac
}

expected_layers_for() {
  case "$1" in SS012_old|SS012_new) echo "0,1,2" ;; SS123) echo "1,2,3" ;; SS036) echo "0,3,6" ;; esac
}

make_manifest() {
  mkdir -p "$RESULT_ROOT/manifests" "$LOG_ROOT"
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/make_post_sft_depth_subspace_manifest.py" \
    --development-manifest "$DEVELOPMENT_MANIFEST" --evaluation-manifest "$EVALUATION_MANIFEST" --output "$MANIFEST"
}

make_smoke_manifest() {
  mkdir -p "$RESULT_ROOT/manifests" "$LOG_ROOT"
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/make_post_sft_depth_subspace_manifest.py" \
    --development-manifest "$DEVELOPMENT_MANIFEST" --evaluation-manifest "$EVALUATION_MANIFEST" --output "$SMOKE_MANIFEST" --smoke
}

preflight() {
  "$VLM3R_PYTHON" - "$BASE_MODEL" "$SIGLIP_MODEL" "$FORWARD_ROOT" "$TARGET_ROOT" "$FEATURE_ROOT" "$DATA_YAML" \
    "$(checkpoint_for SS012_old)" "$(checkpoint_for SS012_new)" "$(checkpoint_for SS123)" "$(checkpoint_for SS036)" <<'PY'
import json
import sys
from pathlib import Path

base, siglip, forward, target, features, data = map(Path, sys.argv[1:7])
labels = ("SS012_old", "SS012_new", "SS123", "SS036")
expected = ("0,1,2", "0,1,2", "1,2,3", "0,3,6")
required = ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json", "config.json", "generation_config.json")
for label, schedule, checkpoint in zip(labels, expected, map(Path, sys.argv[7:])):
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise SystemExit(f"{label}: missing checkpoint files {missing}")
    config = json.loads((checkpoint / "config.json").read_text())
    fusion = (config.get("cut3r_spatialstack_fusion_type") or "add").lower()
    if config.get("use_cut3r_spatialstack") is not True or config.get("cut3r_spatialstack_llm_layers") != schedule:
        raise SystemExit(f"{label}: expected active SpatialStack schedule {schedule}")
    if fusion != "add":
        raise SystemExit(f"{label}: this local protocol expects additive fusion, found {fusion}")
for path in (base, siglip, forward, target, features, data):
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
print(json.dumps({"status": "PASS", "models": labels, "mode": "final_residual_mask"}, indent=2))
PY
}

extract_one() {
  local model="$1" manifest="$2" root="$3" log="$4" verify_normal="$5"
  local checkpoint extra=()
  checkpoint="$(checkpoint_for "$model")"
  if [[ "$verify_normal" == "1" ]]; then extra+=(--geometry-perturbation-verify-normal); fi
  mkdir -p "$root" "$(dirname "$log")"
  echo "[EXTRACT] model=$model checkpoint=$checkpoint CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$root log=$log"
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$VLM3R_PYTHON" -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$model" --model-loading-mode adapter --model-path "$checkpoint" --feature-preset spatialstack \
    --feature-levels "$FEATURE_LEVELS" --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --output-root "$root" --sample-indices "$manifest" --data-yaml "$DATA_YAML" \
    --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --runtime-root "$root/runtime/$model" \
    --geometry-perturbation-split dev_eval --geometry-perturbation-tolerance 1e-6 "${extra[@]}" \
    --assert-first-video --resume 2>&1 | tee "$log"
}

smoke() {
  make_smoke_manifest
  local root="$CACHE_ROOT/smoke" log="$LOG_ROOT/smoke/SS012_new.log"
  extract_one SS012_new "$SMOKE_MANIFEST" "$root" "$log" 1
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/analyze_spatialstack_geometry_perturbation.py" \
    --input-root "$root" --output-dir "$RESULT_ROOT/smoke/analysis" --models SS012_new | tee -a "$log"
}

extract_full() {
  make_manifest
  local model
  for model in SS123 SS012_new SS036 SS012_old; do
    extract_one "$model" "$MANIFEST" "$CACHE_ROOT/full" "$LOG_ROOT/full/${model}.log" 0
  done
}

analyze() {
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/analyze_spatialstack_geometry_perturbation.py" \
    --input-root "$CACHE_ROOT/full" --output-dir "$RESULT_ROOT/analysis_v1" \
    --models SS123 SS012_new SS036 SS012_old | tee "$LOG_ROOT/analysis_v1.log"
}

case "$MODE" in
  preflight) preflight ;;
  manifest) make_manifest ;;
  smoke) preflight; smoke ;;
  extract) preflight; extract_full ;;
  analyze) analyze ;;
  run) preflight; extract_full; analyze ;;
esac
