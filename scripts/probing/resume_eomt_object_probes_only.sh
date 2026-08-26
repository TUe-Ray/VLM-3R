#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/shaoruei/SpatialFocus"
ENV_NAME="vlm3r"
OUT="/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825"
SPLIT="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
LOG="$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_object_resume_only_20260825.log"
LABEL="eomt_object"
LEVELS=(fusion_output projected_features layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12 layer_15 layer_18 layer_21 layer_24 layer_27)

exec 9>"$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_object_resume_only.lock"
flock -n 9 || exit 0
if [[ -f "$OUT/eomt_object_probe_completeness.json" ]] && jq -e '.assessment == "PASS"' "$OUT/eomt_object_probe_completeness.json" >/dev/null 2>&1; then
  exit 0
fi

log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG"; }

wait_ready() {
  while true; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1); then
      log "nvidia-smi unavailable; retrying in 60 seconds: $apps"
      sleep 60
      continue
    fi
    if [[ -n "${apps//[[:space:]]/}" ]]; then
      log "compute jobs present; waiting without contention"
      sleep 60
      continue
    fi
    passed=1
    for gpu in 0 1; do
      if ! CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
        "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
        --physical-gpu-id "$gpu" --output "$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_object_gpu${gpu}_readiness.json" \
        >>"$LOG" 2>&1; then
        passed=0
        break
      fi
    done
    if (( passed )); then return; fi
    log "readiness failed; retrying in 60 seconds"
    sleep 60
  done
}

wait_ready
log "readiness passed; resuming object-only probes"
gpu=0
pids=()
for level in "${LEVELS[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$OUT" --sample-indices "$SPLIT" --probe-subdir probes \
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
  --output-root "$OUT" --sample-indices "$SPLIT" --probe-subdir probes \
  --model-labels "$LABEL" --feature-levels "$(IFS=,; echo "${LEVELS[*]}")" \
  --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 \
  --device cpu --skip-existing --result-stem "depth_probe_scannet_${LABEL}" >>"$LOG" 2>&1

conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
  --output-root "$OUT" --model-label "$LABEL" --sample-indices "$SPLIT" --require-probes \
  --output "$OUT/eomt_object_probe_completeness.json" >>"$LOG" 2>&1
log "object-only probes completed; selective was not launched"
crontab -l 2>/dev/null | sed '/spatialfocus-eomt-object-resume/d' | crontab - 2>/dev/null || true
