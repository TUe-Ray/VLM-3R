#!/usr/bin/env bash
# Wait for the requested local time, then retry the final ScanNet completion
# every two hours until all model namespaces finish successfully.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
RUNNER="$REPO_ROOT/scripts/probing/run_scannet_final_layerwise_depth_completion_local.sh"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_final_layerwise_depth_completion}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
TARGET_HM="${TARGET_HM:-23:59}"
RETRY_SECONDS="${RETRY_SECONDS:-7200}"
START_IMMEDIATELY="${START_IMMEDIATELY:-0}"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/watcher.v2.lock"
if ! flock -n 9; then
  echo "[$(date --iso-8601=seconds)] watcher already running; exiting." >&2
  exit 0
fi

WATCH_LOG="$LOG_ROOT/watcher.log"
exec >> "$WATCH_LOG" 2>&1
echo "[$(date --iso-8601=seconds)] watcher started pid=$$ gpu=$GPU cuda_devices=$CUDA_DEVICES target_hm=$TARGET_HM start_immediately=$START_IMMEDIATELY"

if [[ "$START_IMMEDIATELY" == "1" ]]; then
  target_epoch="$(date +%s)"
else
  target_epoch="$(date -d "today $TARGET_HM" +%s)"
  if (( $(date +%s) >= target_epoch )); then
    target_epoch="$(date -d "tomorrow $TARGET_HM" +%s)"
  fi
fi
while (( $(date +%s) < target_epoch )); do
  remaining=$((target_epoch - $(date +%s)))
  sleep_for=$((remaining < 60 ? remaining : 60))
  sleep "$sleep_for"
done
echo "[$(date --iso-8601=seconds)] first GPU availability check"

gpu_free() {
  local usage
  usage="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  [[ -n "$usage" ]] || return 1
  awk '{ if (($1 + 0) > 1024) busy=1 } END { exit busy ? 1 : 0 }' <<< "$usage"
}

labels=(
  cut3r_spatialstack_44323703
  cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n
  cut3r_spatialstack_d2_pointmap_45457911
  cut3r_spatialstack_cross_attn_45303862
  cut3r_depth_loss_43817021
)

while ((${#labels[@]})); do
  if ! gpu_free; then
    echo "[$(date --iso-8601=seconds)] GPUs unavailable or busy; retrying in ${RETRY_SECONDS}s"
    sleep "$RETRY_SECONDS"
    continue
  fi
  label="${labels[0]}"
  echo "[$(date --iso-8601=seconds)] GPUs available; starting $label"
  if GPU="$GPU" CUDA_DEVICES="$CUDA_DEVICES" bash "$RUNNER" run-one "$label"; then
    labels=("${labels[@]:1}")
    echo "[$(date --iso-8601=seconds)] completed $label; remaining=${#labels[@]}"
  else
    echo "[$(date --iso-8601=seconds)] $label failed; retaining its cache and retrying in ${RETRY_SECONDS}s" >&2
    sleep "$RETRY_SECONDS"
  fi
done
echo "[$(date --iso-8601=seconds)] all final ScanNet layer-wise runs complete"
