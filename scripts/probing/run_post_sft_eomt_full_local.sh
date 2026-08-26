#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/shaoruei/SpatialFocus"
ENV_NAME="vlm3r"
SAMPLE_INDICES="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
TRAIN_JSON="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1/manifests/merged_qa_scannet_train.json"
FORWARD_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"
TARGET_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1"
FEATURE_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"
BASE_MODEL="/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_MODEL="/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"
EOMT_CACHE="/home/shaoruei/probe_cache/eomt_consumer_grid_v2"
LAYERS=(0 1 2 3 6 9 12 15 18 21 24 27)
PRE_LLM_CSV="fusion_output,projected_features"
LEVELS=(fusion_output projected_features layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12 layer_15 layer_18 layer_21 layer_24 layer_27)
LOG_ROOT="$REPO_ROOT/logs/post_sft_geometry_probes/full"
mkdir -p "$LOG_ROOT"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

wait_for_idle_and_ready() {
  while true; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1); then
      log "nvidia-smi compute query failed; retrying in 60 seconds: $apps"
      sleep 60
      continue
    fi
    if [[ -n "${apps//[[:space:]]/}" ]]; then
      log "GPU compute jobs are present; not contending with them. Retrying in 60 seconds."
      sleep 60
      continue
    fi
    local gpu
    local passed=1
    for gpu in 0 1; do
      if ! CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
        "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
        --physical-gpu-id "$gpu" --output "$LOG_ROOT/eomt_gpu${gpu}_readiness.json"; then
        passed=0
        break
      fi
    done
    if (( passed )); then
      return
    fi
    log "nvidia-smi/FP16 readiness did not pass; retrying in 60 seconds."
    sleep 60
  done
}

extract() {
  local label="$1" architecture="$2" checkpoint="$3" output_root="$4"
  shift 4
  # A user-owned workload may appear after the preceding stage completed.
  # Recheck both GPUs immediately before every long model process.
  wait_for_idle_and_ready
  mkdir -p "$output_root"
  env CUDA_VISIBLE_DEVICES=0,1 SPATIALFOCUS_CPU_MERGE_LORA=1 \
    SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS=6GiB,10GiB \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
      --model-label "$label" --post-sft-architecture "$architecture" \
      --model-loading-mode adapter --model-path "$checkpoint" \
      --model-base "$BASE_MODEL" --model-name vlm-3r-llava-qwen2-lora \
      --sample-indices "$SAMPLE_INDICES" --train-data-json "$TRAIN_JSON" \
      --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features \
      --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
      --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" \
      --siglip-path "$SIGLIP_MODEL" --skip-spatial-tower-load true \
      --eomt-consumer-cache-root "$EOMT_CACHE" \
      --eomt-cache-validation "$EOMT_CACHE/validation.json" \
      --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
      --layers "${LAYERS[@]}" --pre-llm-features "$PRE_LLM_CSV" \
      "$@" --output-root "$output_root"
}

probe() {
  local label="$1" output_root="$2" log_file="$3"
  # Probe workers are independent GPU jobs; do not launch them into a newly
  # occupied device even though the extraction stage itself has ended.
  wait_for_idle_and_ready
  local gpu=0 pids=() level
  for level in "${LEVELS[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
      --output-root "$output_root" --sample-indices "$SAMPLE_INDICES" \
      --probe-subdir probes --model-labels "$label" --feature-levels "$level" \
      --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
      --num-workers 0 --device cuda:0 --no-write-aggregate --skip-existing >>"$log_file" 2>&1 &
    pids+=("$!")
    gpu=$((1 - gpu))
    if ((${#pids[@]} == 2)); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$output_root" --sample-indices "$SAMPLE_INDICES" \
    --probe-subdir probes --model-labels "$label" --feature-levels "$(IFS=,; echo "${LEVELS[*]}")" \
    --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
    --num-workers 0 --device cpu --skip-existing \
    --result-stem "depth_probe_scannet_${label}" >>"$log_file" 2>&1
}

verify() {
  local label="$1" output_root="$2" require_probes="$3"
  local require=()
  [[ "$require_probes" == true ]] && require=(--require-probes)
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$output_root" --model-label "$label" --sample-indices "$SAMPLE_INDICES" \
    "${require[@]}" --output "$output_root/${label}_$( [[ "$require_probes" == true ]] && echo probe || echo feature )_completeness.json"
}

wait_for_idle_and_ready
SMOKE_VIDEO=$(conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/find_eomt_smoke_video.py" \
  --sample-indices "$SAMPLE_INDICES" --cache-root "$EOMT_CACHE")
log "EoMT smoke scene: $SMOKE_VIDEO"

OBJECT_CKPT="$REPO_ROOT/.offline_runtime/post_sft_geometry_probes/eomt_obj_text_phrase_100p_40403422"
SELECTIVE_CKPT="$REPO_ROOT/.offline_runtime/post_sft_geometry_probes/cut3r_eomt_sel3dr2_wmzero_40416881"
OBJECT_SMOKE="/home/shaoruei/probe_outputs/post_sft_eomt_object_smoke_20260825"
SELECTIVE_SMOKE="/home/shaoruei/probe_outputs/post_sft_eomt_selective_smoke_20260825"

if [[ ! -f "$OBJECT_SMOKE/eomt_vlm_forward_smoke_report.json" ]]; then
  extract eomt_object eomt_object "$OBJECT_CKPT" "$OBJECT_SMOKE" \
    --only-video-path "$SMOKE_VIDEO" --limit-videos 1 --assert-first-video --verify-eomt-file-checksum \
    2>&1 | tee "$LOG_ROOT/eomt_object_smoke.log"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_eomt_vlm_smoke.py" \
    --output-root "$OBJECT_SMOKE" --model-label eomt_object
fi
if [[ ! -f "$SELECTIVE_SMOKE/eomt_vlm_forward_smoke_report.json" ]]; then
  extract eomt_selective eomt_selective "$SELECTIVE_CKPT" "$SELECTIVE_SMOKE" \
    --only-video-path "$SMOKE_VIDEO" --limit-videos 1 --assert-first-video --verify-eomt-file-checksum \
    2>&1 | tee "$LOG_ROOT/eomt_selective_smoke.log"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_eomt_vlm_smoke.py" \
    --output-root "$SELECTIVE_SMOKE" --model-label eomt_selective
fi

OBJECT_FULL="/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825"
SELECTIVE_FULL="/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_20260825"
extract eomt_object eomt_object "$OBJECT_CKPT" "$OBJECT_FULL" --resume --assert-first-video \
  2>&1 | tee "$LOG_ROOT/eomt_object_extraction.log"
verify eomt_object "$OBJECT_FULL" false
probe eomt_object "$OBJECT_FULL" "$LOG_ROOT/eomt_object_probes.log"
verify eomt_object "$OBJECT_FULL" true

extract eomt_selective eomt_selective "$SELECTIVE_CKPT" "$SELECTIVE_FULL" --resume --assert-first-video \
  2>&1 | tee "$LOG_ROOT/eomt_selective_extraction.log"
verify eomt_selective "$SELECTIVE_FULL" false
probe eomt_selective "$SELECTIVE_FULL" "$LOG_ROOT/eomt_selective_probes.log"
verify eomt_selective "$SELECTIVE_FULL" true

conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_post_sft_geometry_probes.py"
log "EoMT post-SFT extraction and probes completed"
