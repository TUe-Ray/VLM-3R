#!/usr/bin/env bash
# Same-architecture post-SFT SS012 checkpoint comparison using frozen v1 VF.
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  preflight|smoke|extract|analyze|run) ;;
  *) echo "Usage: $0 preflight | smoke | extract | analyze | run" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
VLM3R_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/post_sft_depth_subspace_vf_v1/full}"
SMOKE_CACHE_ROOT="${SMOKE_CACHE_ROOT:-/home/shaoruei/probe_cache/post_sft_ss012_checkpoint_vf_v1/smoke}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/post_sft_ss012_checkpoint_vf_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/post_sft_ss012_checkpoint_vf_v1}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
MANIFEST="${MANIFEST:-/home/shaoruei/probe_outputs/post_sft_depth_subspace_vf_v1/manifests/post_sft_depth_subspace_frozen_vf_v1.json}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-/home/shaoruei/probe_outputs/post_sft_depth_subspace_vf_v1/manifests/post_sft_depth_subspace_frozen_vf_v1_smoke.json}"
TOKEN_SELECTION="${TOKEN_SELECTION:-/home/shaoruei/probe_outputs/depth_subspace_occupancy/development_v1_v4/token_selection.json}"
PRIOR_RESULT_DIR="${PRIOR_RESULT_DIR:-/home/shaoruei/probe_outputs/post_sft_depth_subspace_vf_v1/analysis_v1_frozen}"
CHECKPOINT="${CHECKPOINT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_45297963}"
MODEL_LABEL="SS012_new"
FEATURE_LEVELS="fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"

preflight() {
  "$VLM3R_PYTHON" - "$CHECKPOINT" "$MANIFEST" "$TOKEN_SELECTION" "$PRIOR_RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, manifest, selection, prior = map(Path, sys.argv[1:])
required = ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json", "config.json", "generation_config.json")
missing = [name for name in required if not (checkpoint / name).is_file()]
if missing:
    raise SystemExit(f"missing checkpoint files: {missing}")
config = json.loads((checkpoint / "config.json").read_text())
if config.get("use_cut3r_spatialstack") is not True or config.get("cut3r_spatialstack_llm_layers") != "0,1,2":
    raise SystemExit("checkpoint is not native SS012")
if config.get("cut3r_spatialstack_fusion_type") != "add":
    raise SystemExit("checkpoint is not additive SpatialStack")
payload = json.loads(manifest.read_text())
counts = {split: sum(video["split"] == split for video in payload["videos"]) for split in ("train", "val", "dev_eval")}
if counts != {"train": 6, "val": 2, "dev_eval": 12}:
    raise SystemExit(f"unexpected frozen manifest counts: {counts}")
for path in (selection, prior / "v1_ridge_vf" / "linear_vf_per_video.csv"):
    if not path.is_file():
        raise SystemExit(f"missing prior frozen analysis input: {path}")
print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), "schedule": "0,1,2", "fusion": "add", "c1_canonicalization": False, "manifest_counts": counts}, indent=2))
PY
}

extract_one() {
  local cache_root="$1"
  local manifest="$2"
  local log="$3"
  mkdir -p "$cache_root" "$(dirname "$log")"
  echo "[EXTRACT] model=$MODEL_LABEL checkpoint=$CHECKPOINT manifest=$manifest CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$cache_root log=$log"
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$VLM3R_PYTHON" -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$MODEL_LABEL" --model-loading-mode adapter --model-path "$CHECKPOINT" --feature-preset spatialstack \
    --feature-levels "$FEATURE_LEVELS" --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --output-root "$cache_root" --sample-indices "$manifest" --data-yaml "$DATA_YAML" \
    --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --runtime-root "$cache_root/runtime/$MODEL_LABEL" \
    --assert-first-video --resume 2>&1 | tee "$log"
}

smoke() {
  extract_one "$SMOKE_CACHE_ROOT" "$SMOKE_MANIFEST" "$LOG_ROOT/smoke/SS012_new.log"
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/verify_post_sft_depth_subspace_smoke.py" \
    --cache-root "$SMOKE_CACHE_ROOT" --manifest "$SMOKE_MANIFEST" --model "$MODEL_LABEL" --checkpoint "$CHECKPOINT" \
    --expected-injection-layers "0,1,2" --report "$RESULT_ROOT/smoke/SS012_new_verification.json" | tee -a "$LOG_ROOT/smoke/SS012_new.log"
}

extract_full() {
  extract_one "$CACHE_ROOT" "$MANIFEST" "$LOG_ROOT/full/SS012_new.log"
}

analyze() {
  local output="$RESULT_ROOT/analysis_v1_frozen"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite frozen result directory: $output" >&2
    exit 1
  fi
  "$VLM3R_PYTHON" "$REPO_ROOT/scripts/probing/analyze_ss012_checkpoint_vf.py" \
    --cache-root "$CACHE_ROOT" --manifest "$MANIFEST" --token-selection "$TOKEN_SELECTION" \
    --prior-result-dir "$PRIOR_RESULT_DIR" --new-checkpoint "$CHECKPOINT" --output-dir "$output" \
    --seed 42 --random-directions 64 --label-permutations 32 | tee "$LOG_ROOT/analysis_v1_frozen.log"
}

case "$MODE" in
  preflight) preflight ;;
  smoke) preflight; smoke ;;
  extract) preflight; extract_full ;;
  analyze) analyze ;;
  run) preflight; extract_full; analyze ;;
esac
