#!/usr/bin/env bash
# Frozen post-SFT all-point ridge/VF extraction and cache-only analysis.
# These are trained adapters, so C1 calibration is neither supplied nor allowed.
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
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/post_sft_depth_subspace_vf_v1}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/post_sft_depth_subspace_vf_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/post_sft_depth_subspace_vf_v1}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
DEVELOPMENT_MANIFEST="${DEVELOPMENT_MANIFEST:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/manifests/depth_subspace_pilot_v1.json}"
EVALUATION_MANIFEST="${EVALUATION_MANIFEST:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/manifests/depth_subspace_confirmation_v1.json}"
TOKEN_SELECTION="${TOKEN_SELECTION:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/development_v1_v4/token_selection.json}"
PRE_SFT_RESULT_DIR="${PRE_SFT_RESULT_DIR:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/confirmation_frozen_v1_vf_enrich_confirmation}"
MANIFEST="$RESULT_ROOT/manifests/post_sft_depth_subspace_frozen_vf_v1.json"
SMOKE_MANIFEST="$RESULT_ROOT/manifests/post_sft_depth_subspace_frozen_vf_v1_smoke.json"
FEATURE_LEVELS="fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"

checkpoint_for() {
  case "$1" in
    SS012) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703" ;;
    SS123) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n" ;;
    SS036) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970" ;;
    *) echo "unknown model label: $1" >&2; return 2 ;;
  esac
}

layers_for() {
  case "$1" in SS012) echo "0,1,2" ;; SS123) echo "1,2,3" ;; SS036) echo "0,3,6" ;; esac
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
  "$VLM3R_PYTHON" - "$BASE_MODEL" "$SIGLIP_MODEL" "$FORWARD_ROOT" "$TARGET_ROOT" "$FEATURE_ROOT" "$DATA_YAML" "$FEATURE_LEVELS" \
    "$(checkpoint_for SS012)" "$(checkpoint_for SS123)" "$(checkpoint_for SS036)" <<'PY'
import json
import sys
from pathlib import Path

base, siglip, forward, target, features, data, levels = sys.argv[1:8]
checkpoints = [Path(value) for value in sys.argv[8:]]
required = ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json", "config.json", "generation_config.json")
expected = (("SS012", "0,1,2"), ("SS123", "1,2,3"), ("SS036", "0,3,6"))
for path, (label, schedule) in zip(checkpoints, expected):
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise SystemExit(f"{label}: missing checkpoint files {missing}")
    cfg = json.loads((path / "config.json").read_text())
    if cfg.get("use_cut3r_spatialstack") is not True or cfg.get("cut3r_spatialstack_llm_layers") != schedule:
        raise SystemExit(f"{label}: unexpected SpatialStack schedule/config")
    if (cfg.get("cut3r_spatialstack_fusion_type") or "add") != "add":
        raise SystemExit(f"{label}: frozen analysis requires additive fusion")
for value in (base, siglip, forward, target, features, data):
    if not Path(value).exists():
        raise SystemExit(f"missing input: {value}")
print(json.dumps({"status": "PASS", "feature_levels": levels, "c1_canonicalization": False, "models": [item[0] for item in expected]}, indent=2))
PY
}

extract_one() {
  local model="$1"
  local manifest="$2"
  local root="$3"
  local log="$4"
  local checkpoint
  checkpoint="$(checkpoint_for "$model")"
  mkdir -p "$root" "$(dirname "$log")"
  echo "[EXTRACT] model=$model checkpoint=$checkpoint manifest=$manifest CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$root log=$log"
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
    --assert-first-video --resume 2>&1 | tee "$log"
}

smoke() {
  make_smoke_manifest
  local root="$CACHE_ROOT/smoke"
  local log="$LOG_ROOT/smoke/SS012.log"
  extract_one SS012 "$SMOKE_MANIFEST" "$root" "$log"
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/verify_post_sft_depth_subspace_smoke.py" \
    --cache-root "$root" --manifest "$SMOKE_MANIFEST" --model SS012 --checkpoint "$(checkpoint_for SS012)" \
    --expected-injection-layers "$(layers_for SS012)" --report "$RESULT_ROOT/smoke/SS012_verification.json" | tee -a "$log"
  echo "[SMOKE DONE] $RESULT_ROOT/smoke/SS012_verification.json"
}

extract_full() {
  make_manifest
  local root="$CACHE_ROOT/full"
  local model
  for model in SS012 SS123 SS036; do
    extract_one "$model" "$MANIFEST" "$root" "$LOG_ROOT/full/${model}.log"
  done
}

analyze() {
  local output="$RESULT_ROOT/analysis_v1_frozen"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite frozen result directory: $output" >&2
    exit 1
  fi
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/analyze_post_sft_depth_subspace_vf.py" \
    --cache-root "$CACHE_ROOT/full" --manifest "$MANIFEST" --token-selection "$TOKEN_SELECTION" \
    --pre-sft-result-dir "$PRE_SFT_RESULT_DIR" --output-dir "$output" \
    --seed 42 --random-directions 64 --label-permutations 32 | tee "$LOG_ROOT/analysis_v1_frozen.log"
}

case "$MODE" in
  preflight) preflight ;;
  manifest) make_manifest ;;
  smoke) preflight; smoke ;;
  extract) preflight; extract_full ;;
  analyze) analyze ;;
  run) preflight; extract_full; analyze ;;
esac
