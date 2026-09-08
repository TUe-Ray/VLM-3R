#!/usr/bin/env bash
# Schedule the independent zero-spatial pre-LLM full run on the local server.
# The watcher waits for the requested wall-clock time, then checks both GPUs.
# If either GPU is occupied, it retries exactly every two hours.  A runner
# failure is reported and exits for manual review; it is not silently retried.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
RUNNER="$REPO_ROOT/scripts/probing/run_scannet_depth_layer_completion_local.sh"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_depth_layers_v1}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
TARGET_HM="${TARGET_HM:-20:00}"
RETRY_SECONDS="${RETRY_SECONDS:-7200}"

mkdir -p "$LOG_ROOT"
LOCK_FILE="$LOG_ROOT/zero_prellm_scheduler.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date --iso-8601=seconds)] zero-preLLM scheduler already running; exiting." >&2
  exit 0
fi

SCHEDULER_LOG="$LOG_ROOT/zero_prellm_scheduler.log"
exec >>"$SCHEDULER_LOG" 2>&1
echo "[$(date --iso-8601=seconds)] scheduler started pid=$$ target_hm=$TARGET_HM gpu=$GPU cuda_devices=$CUDA_DEVICES retry_seconds=$RETRY_SECONDS"

target_epoch="$(date -d "today $TARGET_HM" +%s)"
if (( $(date +%s) >= target_epoch )); then
  target_epoch="$(date -d "tomorrow $TARGET_HM" +%s)"
fi
while (( $(date +%s) < target_epoch )); do
  remaining=$((target_epoch - $(date +%s)))
  sleep_for=$((remaining < 60 ? remaining : 60))
  sleep "$sleep_for"
done
echo "[$(date --iso-8601=seconds)] scheduled time reached; checking GPU availability"

gpu_free() {
  local compute_pids usage
  compute_pids="$(nvidia-smi --id="$CUDA_DEVICES" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" || return 1
  if [[ -n "${compute_pids//[[:space:]]/}" ]]; then
    return 1
  fi
  usage="$(nvidia-smi --id="$CUDA_DEVICES" --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  [[ -n "$usage" ]] || return 1
  awk -F',' '{ if (($2 + 0) > 1024) busy=1 } END { exit busy ? 1 : 0 }' <<<"$usage"
}

while ! gpu_free; do
  echo "[$(date --iso-8601=seconds)] selected GPUs busy/unavailable; retrying in ${RETRY_SECONDS}s"
  sleep "$RETRY_SECONDS"
done

echo "[$(date --iso-8601=seconds)] selected GPUs available; launching zero-prellm-full"
if GPU="$GPU" CUDA_DEVICES="$CUDA_DEVICES" bash "$RUNNER" zero-prellm-full; then
  echo "[$(date --iso-8601=seconds)] zero-prellm-full completed successfully"
else
  rc=$?
  echo "[$(date --iso-8601=seconds)] zero-prellm-full failed rc=$rc; stopping for manual review" >&2
  exit "$rc"
fi
