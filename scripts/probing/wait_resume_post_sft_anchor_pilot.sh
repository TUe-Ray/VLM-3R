#!/usr/bin/env bash
# Resume the anchor pilot only after both local GPUs are free.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/post_sft_anchor_pilot_v1}"
LOG_PATH="$LOG_ROOT/resume_waiter.log"
mkdir -p "$LOG_ROOT"

while true; do
  mapfile -t compute_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
  if (( ${#compute_pids[@]} == 0 )); then
    printf '%s GPUs free; resuming anchor pilot.\n' "$(date --iso-8601=seconds)" | tee -a "$LOG_PATH"
    exec /usr/bin/bash "$REPO_ROOT/scripts/probing/run_post_sft_anchor_sample_efficiency_local.sh" run
  fi
  printf '%s GPUs occupied by compute PIDs: %s; retrying in %ss.\n' "$(date --iso-8601=seconds)" "${compute_pids[*]}" "$INTERVAL_SECONDS" | tee -a "$LOG_PATH"
  sleep "$INTERVAL_SECONDS"
done
