#!/usr/bin/env bash
set -euo pipefail

# Independent, restart-safe EoMT selective post-SFT pipeline.  It deliberately
# does not rely on a parent wrapper PID or a cron SIGCONT handoff: the durable
# object completion and selective smoke reports are its entry gates.

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
OBJECT_OUT="/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825"
SMOKE_OUT="/home/shaoruei/probe_outputs/post_sft_eomt_selective_smoke_20260825"
OUT="/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_20260825"
LABEL="eomt_selective"
ARCHITECTURE="eomt_selective"
CHECKPOINT="$REPO_ROOT/.offline_runtime/post_sft_geometry_probes/cut3r_eomt_sel3dr2_wmzero_40416881"
LAYERS=(0 1 2 3 6 9 12 15 18 21 24 27)
PRE_LLM_CSV="fusion_output,projected_features"
LEVELS=(fusion_output projected_features layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12 layer_15 layer_18 layer_21 layer_24 layer_27)
LOG_ROOT="$REPO_ROOT/logs/post_sft_geometry_probes/full"
LOG="$LOG_ROOT/eomt_selective_full_20260826.log"
LOCK="$LOG_ROOT/eomt_selective_full_20260826.lock"

mkdir -p "$LOG_ROOT" "$OUT"
exec 9>"$LOCK"
flock -n 9 || exit 0

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"; }

require_gate() {
  jq -e '.assessment == "PASS"' "$OBJECT_OUT/eomt_object_probe_completeness.json" >/dev/null
  jq -e '(.status // .overall_status) == "PASS"' "$SMOKE_OUT/eomt_vlm_forward_smoke_report.json" >/dev/null
  jq -e '.status == "PASS"' "$EOMT_CACHE/validation.json" >/dev/null
}

wait_ready() {
  while true; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1); then
      log "nvidia-smi query failed; retrying in 60 seconds: $apps"
      sleep 60
      continue
    fi
    if [[ -n "${apps//[[:space:]]/}" ]]; then
      log "compute jobs are present; waiting without contention"
      sleep 60
      continue
    fi
    local gpu passed=1
    for gpu in 0 1; do
      if ! CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
        "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
        --physical-gpu-id "$gpu" \
        --output "$LOG_ROOT/eomt_selective_gpu${gpu}_readiness_20260826.json" >>"$LOG" 2>&1; then
        passed=0
        break
      fi
    done
    if (( passed )); then return; fi
    log "nvidia-smi/FP16 readiness did not pass; retrying in 60 seconds"
    sleep 60
  done
}

extract() {
  wait_ready
  log "starting/resuming selective full feature extraction across physical GPUs 0,1"
  env CUDA_VISIBLE_DEVICES=0,1 SPATIALFOCUS_CPU_MERGE_LORA=1 \
    SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS=6GiB,10GiB \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
      --model-label "$LABEL" --post-sft-architecture "$ARCHITECTURE" \
      --model-loading-mode adapter --model-path "$CHECKPOINT" \
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
      --resume --assert-first-video --output-root "$OUT" 2>&1 | tee -a "$LOG"
}

verify() {
  local require=()
  [[ "$1" == true ]] && require=(--require-probes)
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$OUT" --model-label "$LABEL" --sample-indices "$SAMPLE_INDICES" \
    "${require[@]}" --output "$OUT/${LABEL}_$( [[ "$1" == true ]] && echo probe || echo feature )_completeness.json" \
    2>&1 | tee -a "$LOG"
}

probe() {
  wait_ready
  log "starting/resuming selective probes on one worker per physical GPU"
  local gpu=0 level pids=()
  for level in "${LEVELS[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
      --output-root "$OUT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes \
      --model-labels "$LABEL" --feature-levels "$level" --epochs 50 --batch-size 32 \
      --lr 1e-3 --early-stop-patience 10 --num-workers 0 --device cuda:0 \
      --no-write-aggregate --skip-existing >>"$LOG" 2>&1 &
    pids+=("$!")
    gpu=$((1 - gpu))
    if ((${#pids[@]} == 2)); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$OUT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes \
    --model-labels "$LABEL" --feature-levels "$(IFS=,; echo "${LEVELS[*]}")" \
    --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 \
    --device cpu --skip-existing --result-stem "depth_probe_scannet_${LABEL}" >>"$LOG" 2>&1
}

if [[ -f "$OUT/${LABEL}_probe_completeness.json" ]] && jq -e '.assessment == "PASS"' "$OUT/${LABEL}_probe_completeness.json" >/dev/null 2>&1; then
  log "selective pipeline is already complete; no action needed"
  exit 0
fi

require_gate
extract
verify false
probe
verify true
conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_post_sft_geometry_probes.py" >>"$LOG" 2>&1
log "EoMT selective post-SFT extraction and probes completed"
