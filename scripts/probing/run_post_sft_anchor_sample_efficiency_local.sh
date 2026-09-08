#!/usr/bin/env bash
# Three-anchor formal post-SFT cache completion and absolute-convergence pilot.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in preflight|smoke-all|formal-extract|sweep|analyze|run|schedule) ;; *) echo "Usage: $0 preflight|smoke-all|formal-extract|sweep|analyze|run|schedule" >&2; exit 2;; esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
PYTHON="${PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SPLIT="${SPLIT:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
WORKBOOK="${WORKBOOK:-$REPO_ROOT/post-sft-result-for-codex.xlsx}"
DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
RUNNER="$REPO_ROOT/scripts/probing/run_post_sft_anchor_sample_efficiency.py"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/post_sft_anchor_pilot_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/shaoruei/probe_outputs/post_sft_anchor_pilot_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/post_sft_anchor_pilot_v1}"
SHARED_ROOT="$OUTPUT_ROOT/shared"
SMOKE_ROOT="$CACHE_ROOT/smoke"
SMOKE_MARKER="$OUTPUT_ROOT/smoke/smoke_all_pass.json"
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" "$SHARED_ROOT"

LEVELS_BASELINE="fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
LEVELS_SS="siglip_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
SWEEP_LEVELS="projected_features,layer_0,layer_1,layer_6,layer_27"

configure() {
  case "$1" in
    vlm3r_baseline) LABEL=vlm3r_baseline; CHECKPOINT=/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2; PRESET=original; LEVELS=$LEVELS_BASELINE; SPATIAL_SUBDIR='spatial_features'; ACTIVE_CACHE=/home/shaoruei/probe_cache/scannet_depth_layers_v1/full;;
    cut3r_spatialstack_44323703) LABEL=cut3r_spatialstack_44323703; CHECKPOINT=/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703; PRESET=spatialstack; LEVELS=$LEVELS_SS; SPATIAL_SUBDIR='6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features'; ACTIVE_CACHE=$CACHE_ROOT/full;;
    zero_spatial) LABEL=zero_spatial; CHECKPOINT=/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/zero_spatial_features; PRESET=zero_spatial; LEVELS=$LEVELS_SS; SPATIAL_SUBDIR='spatial_features'; ACTIVE_CACHE=/home/shaoruei/probe_cache/scannet_depth_layers_v1/full;;
    *) echo "Unknown anchor $1" >&2; exit 2;;
  esac
}

prepare() { "$PYTHON" "$RUNNER" --mode prepare --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$OUTPUT_ROOT"; }

preflight() {
  prepare
  "$PYTHON" - "$BASE_MODEL" "$SIGLIP_MODEL" "$FORWARD_ROOT" "$TARGET_ROOT" "$FEATURE_ROOT" "$SPLIT" "$WORKBOOK" <<'PY'
import hashlib, json, sys
from pathlib import Path
paths=[Path(x) for x in sys.argv[1:]]
if any(not x.exists() for x in paths): raise SystemExit('missing required input: '+str([str(x) for x in paths if not x.exists()]))
p=json.loads(paths[5].read_text()); counts={s:sum(v.get('split')==s for v in p['videos']) for s in ('train','val')}
if counts != {'train':1006,'val':193}: raise SystemExit(f'bad split counts {counts}')
print(json.dumps({'status':'PASS','split_counts':counts},indent=2))
PY
  for label in vlm3r_baseline cut3r_spatialstack_44323703 zero_spatial; do configure "$label"; for f in adapter_model.bin non_lora_trainables.bin adapter_config.json config.json generation_config.json; do [[ -f "$CHECKPOINT/$f" ]] || { echo "missing $CHECKPOINT/$f" >&2; exit 1; }; done; done
  "$PYTHON" "$RUNNER" --mode preflight --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$OUTPUT_ROOT/preflight" || true
}

smoke_manifest() {
  local manifest="$OUTPUT_ROOT/manifests/smoke_1train_1val.json"
  if [[ ! -f "$manifest" ]]; then
    "$PYTHON" "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" --sample-indices "$SPLIT" --output "$manifest" --train-videos 1 --val-videos 1 >&2
  fi
  printf '%s\n' "$manifest"
}

smoke_one() {
  configure "$1"; local manifest; manifest="$(smoke_manifest)"; local root="$SMOKE_ROOT/$LABEL"; local log="$LOG_ROOT/smoke/$LABEL.log"; mkdir -p "$root" "$(dirname "$log")"
  echo "[SMOKE] $LABEL checkpoint=$CHECKPOINT cache=$root log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$PYTHON" -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" --model-label "$LABEL" --model-path "$CHECKPOINT" --feature-preset "$PRESET" --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$root" --sample-indices "$manifest" --data-yaml "$DATA_YAML" --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --feature-levels "$LEVELS" --runtime-root "$root/runtime" --assert-first-video --resume 2>&1 | tee "$log"
  IFS=, read -r -a points <<< "$SWEEP_LEVELS"
  for point in "${points[@]}"; do
    env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$root" --sample-indices "$manifest" --probe-subdir probes --model-labels "$LABEL" --feature-levels "$point" --epochs 2 --batch-size 2 --lr 1e-3 --early-stop-patience 1 --num-workers 0 --device cuda:0 --allow-partial --no-write-aggregate 2>&1 | tee -a "$log"
  done
  "$PYTHON" - "$root" "$LABEL" "$manifest" "$LEVELS" <<'PY' | tee -a "$log"
import json, math, sys
from pathlib import Path
import torch
root,label,manifest,levels=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),sys.argv[4].split(',')
p=json.loads(manifest.read_text()); ids=[str(f['frame_sample_id']) for v in p['videos'] for f in v['frames']]
assert len(ids)==4, len(ids)
for level in levels:
 d=root/'features'/label/level; files=[d/f'frame_{x}.pt' for x in ids]; assert all(x.is_file() for x in files), level
 x=torch.load(files[0],map_location='cpu'); assert tuple(x.shape[:2])==(14,14), (level,x.shape)
 assert x.shape[-1] == (1152 if level in ('siglip_output','fusion_output') else 3584), (level,x.shape)
for level in ('projected_features','layer_0','layer_1','layer_6','layer_27'):
 q=json.loads((root/'probes'/label/level/'metrics.json').read_text()); assert all(k in q for k in ('delta125','absrel','mae')); assert all(math.isfinite(float(q[k])) for k in ('delta125','absrel','mae')), q; assert q['num_tokens']==392, q
print(json.dumps({'status':'PASS','model':label,'points':levels,'frames':len(ids)},indent=2))
PY
}

smoke_all() {
  preflight
  for label in vlm3r_baseline cut3r_spatialstack_44323703 zero_spatial; do smoke_one "$label"; done
  "$PYTHON" - "$SMOKE_ROOT" "$SMOKE_MARKER" <<'PY'
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]); models=['vlm3r_baseline','cut3r_spatialstack_44323703','zero_spatial']; points=['projected_features','layer_0','layer_1','layer_6','layer_27']
for m in models:
 for p in points:
  q=root/m/'probes'/m/p/'metrics.json'
  if not q.is_file(): raise SystemExit(f'missing smoke metric {q}')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'status':'PASS','models':models,'points':points},indent=2)+'\n')
print(out)
PY
}

extract_one() {
  configure "$1"; local log="$LOG_ROOT/formal/$LABEL.log"; mkdir -p "$(dirname "$log")" "$ACTIVE_CACHE"
  echo "[FORMAL EXTRACT] $LABEL checkpoint=$CHECKPOINT cache=$ACTIVE_CACHE log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$PYTHON" -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" --model-label "$LABEL" --model-path "$CHECKPOINT" --feature-preset "$PRESET" --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$ACTIVE_CACHE" --sample-indices "$SPLIT" --data-yaml "$DATA_YAML" --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --feature-levels "$LEVELS" --runtime-root "$ACTIVE_CACHE/runtime/$LABEL" --assert-first-video --resume 2>&1 | tee "$log"
  "$PYTHON" "$RUNNER" --mode preflight --model-label "$LABEL" --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$OUTPUT_ROOT/preflight"
}

cache_complete() {
  local label="$1"
  local report="$OUTPUT_ROOT/preflight/cache_check_$label"
  "$PYTHON" "$RUNNER" --mode preflight --model-label "$label" --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$report" >/dev/null
  jq -e --arg model_label "$label" '.[$model_label] | length == 0' "$report/preflight.json" >/dev/null
}

formal_extract() {
  [[ -f "$SMOKE_MARKER" ]] || { echo "Missing PASS smoke marker: $SMOKE_MARKER" >&2; exit 1; }
  preflight
  for label in vlm3r_baseline cut3r_spatialstack_44323703 zero_spatial; do
    if cache_complete "$label"; then
      echo "[FORMAL EXTRACT] $label cache already complete; skipping forward extraction."
    else
      extract_one "$label"
    fi
  done
}

sweep() {
  [[ -f "$SMOKE_MARKER" ]] || { echo "Missing PASS smoke marker: $SMOKE_MARKER" >&2; exit 1; }
  preflight
  local pids=() labels=(); local index=0
  for label in vlm3r_baseline cut3r_spatialstack_44323703 zero_spatial; do
    local gpu=$((index % 2)); index=$((index + 1)); local out="$OUTPUT_ROOT/models/$label"; local log="$LOG_ROOT/sweep/$label.log"; mkdir -p "$out" "$(dirname "$log")"
    (env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$RUNNER" --mode sweep --model-label "$label" --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$out" --device cuda:0 >"$log" 2>&1) & pids+=("$!"); labels+=("$label")
    if (( ${#pids[@]} == 2 )); then for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "sweep failed ${labels[$i]}" >&2; exit 1; }; done; pids=(); labels=(); fi
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "sweep failed ${labels[$i]}" >&2; exit 1; }; done
  "$PYTHON" "$RUNNER" --mode analyze --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$OUTPUT_ROOT"
}

schedule() {
  [[ -f "$SMOKE_MARKER" ]] || { echo "Smoke not PASS; refusing timer." >&2; exit 1; }
  systemd-run --user --unit=spatialfocus-post-sft-anchor-pilot-20260901-2000 --on-calendar='2026-09-01 20:00:00 Europe/Amsterdam' --collect --property=WorkingDirectory="$REPO_ROOT" --setenv=REPO_ROOT="$REPO_ROOT" --setenv=GPU=0 --setenv=CUDA_DEVICES=0,1 --setenv=SPATIALFOCUS_CPU_MERGE_LORA=1 /usr/bin/bash "$REPO_ROOT/scripts/probing/run_post_sft_anchor_sample_efficiency_local.sh" run
}

case "$MODE" in preflight) preflight;; smoke-all) smoke_all;; formal-extract) formal_extract;; sweep) sweep;; analyze) "$PYTHON" "$RUNNER" --mode analyze --split "$SPLIT" --reference-workbook "$WORKBOOK" --shared-root "$SHARED_ROOT" --output-dir "$OUTPUT_ROOT";; run) formal_extract; sweep;; schedule) schedule;; esac
