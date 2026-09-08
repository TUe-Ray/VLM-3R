#!/usr/bin/env bash
# Local runner for a complete new-model post-SFT SpatialStack probe.
# Missing-layer completion is handled by the dedicated completion runner.
set -euo pipefail

export PATH="/home/shaoruei/miniconda3/bin:${PATH:-}"
if [[ -f /home/shaoruei/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/shaoruei/miniconda3/etc/profile.d/conda.sh
fi

MODE="${1:-}"
if [[ "$MODE" != "preflight" && "$MODE" != "smoke" && "$MODE" != "run" ]]; then
  echo "Usage: $0 preflight | smoke | run" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
VLM3R_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
source "$REPO_ROOT/scripts/probing/common_probe_layers.sh"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CHECKPOINT="${CHECKPOINT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970}"
MODEL_LABEL="${MODEL_LABEL:-cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970}"
SS_LLM_LAYERS="${SS_LLM_LAYERS:-0,3,6}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/scannet_ss_add_036_post_sft_complete_v2}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/scannet_ss_add_036_post_sft_complete_v2}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_ss_add_036_post_sft_complete_v2}"
LAYERS="$COMMON_PROBE_LAYERS_SPACE"
PRE_LLM_FEATURES="fusion_output,projected_features"
FEATURE_LEVELS="fusion_output,projected_features,${COMMON_PROBE_LAYER_LEVELS_CSV}"
RESULT_STEM="${RESULT_STEM:-scannet_ss_post_sft_depth_probe}"
DELETE_FULL_CACHE="${DELETE_FULL_CACHE:-true}"
SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"
LOCAL_DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"

mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT" "$LOG_ROOT"

preflight() {
  "$VLM3R_PYTHON" - "$CHECKPOINT" "$SAMPLE_INDICES" "$FORWARD_ROOT" "$FEATURE_ROOT" \
    "$FEATURE_LEVELS" "$MODEL_LABEL" "$SS_LLM_LAYERS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, split_path, forward_root, feature_root = map(Path, sys.argv[1:5])
feature_levels, model_label, expected_llm_layers = sys.argv[5:]
required = ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json", "config.json", "generation_config.json")
missing = [name for name in required if not (checkpoint / name).is_file()]
if missing:
    raise SystemExit(f"missing checkpoint files: {missing}")

payload = json.loads(split_path.read_text())
videos = payload.get("videos", [])
if len(videos) != 1199 or int(payload.get("train_videos", -1)) != 1006 or int(payload.get("val_videos", -1)) != 193:
    raise SystemExit(f"unexpected split identity: videos={len(videos)} train={payload.get('train_videos')} val={payload.get('val_videos')}")
h = hashlib.sha256(split_path.read_bytes()).hexdigest()
expected = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
if h != expected:
    raise SystemExit(f"split sha256 mismatch: {h}")

def count_pt(root: Path, subdir: str) -> int:
    return sum(1 for _ in (root / subdir).glob("*.pt"))

forward_count = sum(count_pt(Path(forward_root), f"frames/{dataset}") for dataset in ("scannet",))
scene_ids = {str(video["scene_id"]) for video in videos if video.get("source_dataset") == "scannet"}
sidecar_dir = Path(feature_root) / "scannet" / "spatial_features"
missing_sidecars = sorted(scene_id for scene_id in scene_ids if not (sidecar_dir / f"{scene_id}.pt").is_file())
if forward_count != 1199 or len(scene_ids) != 1199 or missing_sidecars:
    raise SystemExit(
        f"incomplete ScanNet inputs: forward_frames={forward_count}, scenes={len(scene_ids)}, "
        f"missing_cut3r_sidecars={missing_sidecars[:5]}"
    )
sidecar_count = len(scene_ids)
config = json.loads((checkpoint / "config.json").read_text())
assert config.get("use_cut3r_spatialstack") is True
assert config.get("cut3r_spatialstack_llm_layers") == expected_llm_layers
assert config.get("cut3r_spatialstack_fusion_type") == "add"
print(json.dumps({
    "checkpoint": str(checkpoint),
    "checkpoint_files": list(required),
    "model_label": model_label,
    "spatialstack_llm_layers": config.get("cut3r_spatialstack_llm_layers"),
    "spatialstack_fusion_type": config.get("cut3r_spatialstack_fusion_type"),
    "split_sha256": h,
    "forward_scannet_files": forward_count,
    "cut3r_scannet_final_files": sidecar_count,
    "feature_levels": feature_levels,
    "assessment": "PASS",
}, indent=2))
PY
}

smoke() {
  local root="$CACHE_ROOT/smoke"
  local manifest="$root/manifests/scannet_smoke_1train_1val.json"
  local log="$LOG_ROOT/smoke.log"
  mkdir -p "$root" "$LOG_ROOT"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$manifest" --train-videos 1 --val-videos 1
  echo "[SMOKE] model=$MODEL_LABEL checkpoint=$CHECKPOINT CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$root log=$log"
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$MODEL_LABEL" --model-path "$CHECKPOINT" --feature-preset spatialstack \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$root" \
    --sample-indices "$manifest" --data-yaml "$LOCAL_DATA_YAML" --feature-root "$FEATURE_ROOT" \
    --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 \
    --device cuda:0 --device-map auto --layers $LAYERS --pre-llm-features "$PRE_LLM_FEATURES" \
    --runtime-root "$root/runtime" \
    --assert-first-video 2>&1 | tee "$log"
  IFS=, read -r -a levels <<< "$FEATURE_LEVELS"
  for level in "${levels[@]}"; do
    if [[ "$level" == layer_* ]]; then
      conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
        --output-root "$root" --model-labels "$MODEL_LABEL" --feature-levels "$level" 2>&1 | tee -a "$log"
    fi
  done
  train_probe_worker() {
    local physical_gpu="$1"
    local worker_log="$2"
    shift 2
    local level
    for level in "$@"; do
      env CUDA_VISIBLE_DEVICES="$physical_gpu" conda run -n "$ENV_NAME" python -u \
        "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$root" --sample-indices "$manifest" \
        --probe-subdir probes --model-labels "$MODEL_LABEL" --feature-levels "$level" \
        --epochs 2 --batch-size 2 --lr 1e-3 --early-stop-patience 1 --num-workers 0 --device cuda:0 \
        --allow-partial --no-write-aggregate >> "$worker_log" 2>&1
    done
  }
  train_probe_worker 0 "$LOG_ROOT/smoke_probe_gpu0.log" \
    fusion_output layer_0 layer_2 layer_6 layer_12 layer_18 layer_24 &
  local worker0=$!
  train_probe_worker 1 "$LOG_ROOT/smoke_probe_gpu1.log" \
    projected_features layer_1 layer_3 layer_9 layer_15 layer_21 layer_27 &
  local worker1=$!
  wait "$worker0"
  wait "$worker1"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_scannet_final_layerwise_smoke.py" \
    --output-root "$root" --model-label "$MODEL_LABEL" --feature-levels "$FEATURE_LEVELS" \
    --manifest "$manifest" --report "$root/smoke_verification.json" 2>&1 | tee -a "$log"
  mkdir -p "$DURABLE_ROOT/provenance/$MODEL_LABEL"
  cp -a "$root/smoke_verification.json" "$DURABLE_ROOT/provenance/$MODEL_LABEL/smoke_verification.json"
  case "$root" in
    "$CACHE_ROOT"/smoke) rm -rf -- "$root" ;;
    *) echo "refusing unexpected smoke cleanup path: $root" >&2; exit 1 ;;
  esac
  echo "[SMOKE DONE] durable verification retained; removed only $root"
}

run_full() {
  local root="$CACHE_ROOT/full"
  local log="$LOG_ROOT/full.log"
  mkdir -p "$root" "$DURABLE_ROOT/provenance/$MODEL_LABEL" "$LOG_ROOT"
  echo "[RUN] model=$MODEL_LABEL checkpoint=$CHECKPOINT CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$root durable=$DURABLE_ROOT log=$log"
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$MODEL_LABEL" --model-path "$CHECKPOINT" --feature-preset spatialstack \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$root" \
    --sample-indices "$SAMPLE_INDICES" --data-yaml "$LOCAL_DATA_YAML" --feature-root "$FEATURE_ROOT" \
    --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 \
    --device cuda:0 --device-map auto --layers $LAYERS --pre-llm-features "$PRE_LLM_FEATURES" \
    --runtime-root "$root/runtime" \
    --assert-first-video --resume 2>&1 | tee "$log"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$root" --model-label "$MODEL_LABEL" --sample-indices "$SAMPLE_INDICES" \
    --output "$DURABLE_ROOT/provenance/$MODEL_LABEL/feature_completeness.json" 2>&1 | tee -a "$log"
  IFS=, read -r -a levels <<< "$FEATURE_LEVELS"
  for level in "${levels[@]}"; do
    if [[ "$level" == layer_* ]]; then
      conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/materialize_depth_probe_layers.py" \
        --output-root "$root" --model-labels "$MODEL_LABEL" --feature-levels "$level" 2>&1 | tee -a "$log"
    fi
  done
  train_probe_worker() {
    local physical_gpu="$1"
    local worker_log="$2"
    shift 2
    local level
    for level in "$@"; do
      env CUDA_VISIBLE_DEVICES="$physical_gpu" conda run -n "$ENV_NAME" python -u \
        "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$root" --sample-indices "$SAMPLE_INDICES" \
        --probe-subdir probes --model-labels "$MODEL_LABEL" --feature-levels "$level" \
        --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 --device cuda:0 \
        --no-write-aggregate >> "$worker_log" 2>&1
    done
  }
  train_probe_worker 0 "$LOG_ROOT/probe_gpu0.log" \
    fusion_output layer_0 layer_2 layer_6 layer_12 layer_18 layer_24 &
  local worker0=$!
  train_probe_worker 1 "$LOG_ROOT/probe_gpu1.log" \
    projected_features layer_1 layer_3 layer_9 layer_15 layer_21 layer_27 &
  local worker1=$!
  wait "$worker0"
  wait "$worker1"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$root" --model-label "$MODEL_LABEL" --sample-indices "$SAMPLE_INDICES" --require-probes \
    --output "$DURABLE_ROOT/provenance/$MODEL_LABEL/probe_completeness.json" 2>&1 | tee -a "$log"
  for level in "${levels[@]}"; do
    mkdir -p "$DURABLE_ROOT/probes/$MODEL_LABEL/$level"
    cp -a "$root/probes/$MODEL_LABEL/$level/." "$DURABLE_ROOT/probes/$MODEL_LABEL/$level/"
  done
  cp -a "$root/features/$MODEL_LABEL/extraction_provenance.json" "$DURABLE_ROOT/provenance/$MODEL_LABEL/"
  sha256sum "$CHECKPOINT"/{adapter_model.bin,non_lora_trainables.bin,adapter_config.json,config.json,generation_config.json} \
    > "$DURABLE_ROOT/provenance/$MODEL_LABEL/checkpoint_sha256.txt"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$DURABLE_ROOT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes \
    --model-labels "$MODEL_LABEL" --feature-levels "$FEATURE_LEVELS" --skip-existing \
    --result-stem "$RESULT_STEM" --device cpu
  "$VLM3R_PYTHON" - "$DURABLE_ROOT" "$MODEL_LABEL" "$FEATURE_LEVELS" \
    "$DURABLE_ROOT/provenance/$MODEL_LABEL/durable_metrics_verification.json" <<'PY'
import json
import sys
from pathlib import Path

root, label, levels_csv, report_path = map(Path, sys.argv[1:5])
levels = levels_csv.name.split(",")
failures = []
for level in levels:
    path = root / "probes" / label.name / level / "metrics.json"
    if not path.is_file():
        failures.append(f"missing:{level}")
        continue
    metrics = json.loads(path.read_text())
    if metrics.get("feature_level") != level or int(metrics.get("num_tokens", -1)) != 75656:
        failures.append(f"invalid:{level}")
report = {
    "model_label": label.name,
    "levels": levels,
    "assessment": "PASS" if not failures else "FAIL",
    "failures": failures,
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps(report, indent=2, sort_keys=True))
if failures:
    raise SystemExit(1)
PY
  if [[ "$DELETE_FULL_CACHE" == true ]]; then
    case "$root" in
      "$CACHE_ROOT"/full) rm -rf -- "$root" ;;
      *) echo "refusing unexpected cleanup path: $root" >&2; exit 1 ;;
    esac
    echo "[DONE] durable results verified; removed only $root"
  else
    echo "[DONE] durable results verified; retained cache at $root"
  fi
}

case "$MODE" in
  preflight) preflight ;;
  smoke) preflight; smoke ;;
  run) preflight; run_full ;;
esac
