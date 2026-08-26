#!/usr/bin/env bash
# Frozen normal-probe evaluation under the existing SpatialStack residual-mask OFF intervention.
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|capture|train-missing|evaluate) ;; *) echo "Usage: $0 preflight|smoke|capture|train-missing|evaluate" >&2; exit 2;; esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/spatialstack_geometry_perturbation_v1}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/spatialstack_geometry_causal_probe_drop_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/spatialstack_geometry_causal_probe_drop_v1}"
PAIR_FEATURE_ROOT="${PAIR_FEATURE_ROOT:-$CACHE_ROOT/full/frozen_probe_features}"
MANIFEST="${MANIFEST:-/home/shaoruei/probe_outputs/spatialstack_geometry_perturbation_v1/manifests/post_sft_geometry_perturbation_v1.json}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-/home/shaoruei/probe_outputs/spatialstack_geometry_perturbation_v1/manifests/post_sft_geometry_perturbation_v1_smoke.json}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/leonardo_probe_reference/spatialfocus_probe_reference_20260817/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
LEVELS="layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
MISSING_LEVELS="layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27"
SPATIAL_SUBDIR="6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features"

checkpoint_for() { case "$1" in
  SS012_old) echo /mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703;;
  SS012_new) echo /mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_45297963;;
  SS123) echo /mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n;;
  SS036) echo /mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970;; esac; }
probe_dir_for() { case "$1:$2" in
  SS012_new:*) echo "/home/shaoruei/probe_outputs/scannet_ss012_45297963_v1/probes/cut3r_spatialstack_45297963/layer_$2";;
  SS036:0|SS036:1|SS036:2|SS036:3|SS036:6|SS036:9|SS036:15|SS036:21|SS036:27) echo "/home/shaoruei/probe_outputs/scannet_ss_add_036_post_sft_all_layers_v1/probes/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970/layer_$2";;
  SS036:*) echo "/home/shaoruei/probe_outputs/scannet_ss_add_036_post_sft_v1/probes/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970/layer_$2";;
  SS123:12|SS123:18|SS123:24) echo "/home/shaoruei/probe_outputs/scannet_final_layerwise_depth_completion/probes/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n/layer_$2";;
  SS012_old:12|SS012_old:18|SS012_old:24) echo "/home/shaoruei/probe_outputs/scannet_final_layerwise_depth_completion/probes/cut3r_spatialstack_44323703/layer_$2";;
  *) echo "$RESULT_ROOT/probes/$1/layer_$2";; esac; }

preflight() {
  mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
  "$PYTHON" - "$MANIFEST" "$SAMPLE_INDICES" "$BASE_MODEL" "$SIGLIP_MODEL" "$FORWARD_ROOT" "$TARGET_ROOT" "$FEATURE_ROOT" "$DATA_YAML" "$(checkpoint_for SS012_old)" "$(checkpoint_for SS012_new)" "$(checkpoint_for SS123)" "$(checkpoint_for SS036)" <<'PY'
import json, sys
from pathlib import Path
for value in map(Path, sys.argv[1:]):
    if not value.exists(): raise SystemExit(f"missing required input: {value}")
for checkpoint in map(Path, sys.argv[9:]):
    cfg=json.loads((checkpoint/'config.json').read_text())
    if not cfg.get('use_cut3r_spatialstack') or (cfg.get('cut3r_spatialstack_fusion_type') or 'add').lower() != 'add':
        raise SystemExit(f"not an additive SpatialStack checkpoint: {checkpoint}")
print(json.dumps({'status':'PASS','models':['SS123','SS012_new','SS036','SS012_old'],'layers':list(range(0,28))}, indent=2))
PY
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
}

capture_one() {
  local label="$1" active_manifest="$2" feature_root="$3" log="$4" verify="$5"
  local extra=(); [[ "$verify" == 1 ]] && extra+=(--geometry-perturbation-verify-normal)
  echo "[CAPTURE] model=$label CUDA_VISIBLE_DEVICES=$CUDA_DEVICES cache=$feature_root log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$PYTHON" -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$label" --model-loading-mode adapter --model-path "$(checkpoint_for "$label")" --feature-preset spatialstack --feature-levels "$LEVELS" \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$CACHE_ROOT/full" --sample-indices "$active_manifest" --data-yaml "$DATA_YAML" \
    --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto \
    --runtime-root "$CACHE_ROOT/full/runtime/$label" --geometry-perturbation-split dev_eval --geometry-perturbation-tolerance 1e-6 \
    --geometry-perturbation-feature-cache-root "$feature_root" --assert-first-video --resume "${extra[@]}" 2>&1 | tee "$log"
}

make_tasks() {
  local task_file="$1" models="$2"
  printf 'model\tprobe_dir\tlayer\treference_model\n' > "$task_file"
  local label layer
  for label in $models; do for layer in 0 1 2 3 6 9 12 15 18 21 24 27; do
    printf '%s\t%s\t%s\t\n' "$label" "$(probe_dir_for "$label" "$layer")" "$layer" >> "$task_file"
  done; done
}

smoke() {
  # SS036/L0 has an extant full normal cache and saved probe, so it exercises a real frozen checkpoint.
  capture_one SS036 "$SMOKE_MANIFEST" "$CACHE_ROOT/smoke/frozen_probe_features" "$LOG_ROOT/smoke_SS036.log" 1
  local tasks="$RESULT_ROOT/smoke_tasks.tsv"; printf 'model\tprobe_dir\tlayer\treference_model\nSS036\t%s\t0\tcut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970\n' "$(probe_dir_for SS036 0)" > "$tasks"
  "$PYTHON" "$REPO_ROOT/scripts/probing/evaluate_frozen_geometry_probe_drop.py" --tasks "$tasks" --manifest "$SMOKE_MANIFEST" --cache-root "$CACHE_ROOT/full" --feature-cache-root "$CACHE_ROOT/smoke/frozen_probe_features" --split dev_eval --output-dir "$RESULT_ROOT/smoke" --device cuda:0 --include-delta-diagnostic \
    --normal-reference-cache-root /home/shaoruei/probe_cache/scannet_ss_add_036_post_sft_all_layers_v1/full --normal-reference-manifest "$SAMPLE_INDICES" --normal-reference-split val | tee "$LOG_ROOT/smoke_evaluate.log"
}

train_missing_one() {
  local label="$1" actual_label cache="$RESULT_ROOT/normal_feature_cache/$label" log="$LOG_ROOT/train_missing_$label.log"
  case "$label" in SS123) actual_label=cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n;; SS012_old) actual_label=cut3r_spatialstack_44323703;; esac
  echo "[TRAIN-MISSING] model=$label CUDA_VISIBLE_DEVICES=$CUDA_DEVICES cache=$cache log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 "$PYTHON" -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$actual_label" --model-loading-mode adapter --model-path "$(checkpoint_for "$label")" --feature-preset spatialstack --feature-levels "$MISSING_LEVELS" \
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$cache" --sample-indices "$SAMPLE_INDICES" --data-yaml "$DATA_YAML" --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto --runtime-root "$cache/runtime" --assert-first-video --resume 2>&1 | tee "$log"
  local level
  IFS=, read -r -a levels <<< "$MISSING_LEVELS"
  for level in "${levels[@]}"; do
    env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$cache" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes --model-labels "$actual_label" --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 --device cuda:0 --no-write-aggregate 2>&1 | tee -a "$log"
    mkdir -p "$RESULT_ROOT/probes/$label/$level"; cp -a "$cache/probes/$actual_label/$level/." "$RESULT_ROOT/probes/$label/$level/"
  done
}

evaluate() {
  local tasks="$RESULT_ROOT/tasks.tsv"; make_tasks "$tasks" 'SS123 SS012_new SS036 SS012_old'
  "$PYTHON" "$REPO_ROOT/scripts/probing/evaluate_frozen_geometry_probe_drop.py" --tasks "$tasks" --manifest "$MANIFEST" --cache-root "$CACHE_ROOT/full" --feature-cache-root "$PAIR_FEATURE_ROOT" --split dev_eval --output-dir "$RESULT_ROOT/analysis_v1" --device cuda:0 --include-delta-diagnostic | tee "$LOG_ROOT/evaluate.log"
}

case "$MODE" in
  preflight) preflight;;
  smoke) preflight; smoke;;
  capture) preflight; for label in SS123 SS012_new SS036 SS012_old; do capture_one "$label" "$MANIFEST" "$PAIR_FEATURE_ROOT" "$LOG_ROOT/capture_$label.log" 0; done;;
  train-missing) preflight; train_missing_one SS123; train_missing_one SS012_old;;
  evaluate) evaluate;;
esac
