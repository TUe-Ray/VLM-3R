#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/shaoruei/SpatialFocus"
ENV_NAME="vlm3r"
TARGET="$REPO_ROOT/scripts/probing/run_post_sft_geo_rope_full_local.sh"
LOG_ROOT="$REPO_ROOT/logs/post_sft_geometry_probes/full"
INITIAL_DELAY_SECONDS="${INITIAL_DELAY_SECONDS:-3300}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

# Prevent accidentally scheduling the same full pipeline twice.
exec 9>"$LOG_ROOT/delayed_geometry_launch.lock"
if ! flock -n 9; then
  log "Another delayed post-SFT geometry launcher already holds the lock; exiting."
  exit 1
fi

if [[ ! -x "$TARGET" && ! -f "$TARGET" ]]; then
  log "Target wrapper does not exist: $TARGET"
  exit 1
fi

scheduled_epoch=$(( $(date +%s) + INITIAL_DELAY_SECONDS ))
log "Timer armed for $(date -d "@$scheduled_epoch" '+%Y-%m-%d %H:%M:%S %Z') (delay=${INITIAL_DELAY_SECONDS}s)."
sleep "$INITIAL_DELAY_SECONDS"

while true; do
  if ! gpu_state=$(nvidia-smi \
      --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>&1); then
    log "nvidia-smi readiness check failed; retrying in ${POLL_SECONDS}s: $gpu_state"
    sleep "$POLL_SECONDS"
    continue
  fi

  if ! compute_apps=$(nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1); then
    log "Could not query GPU compute jobs; retrying in ${POLL_SECONDS}s: $compute_apps"
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ -n "${compute_apps//[[:space:]]/}" ]]; then
    log "GPU compute jobs are still present; not contending with them. Retrying in ${POLL_SECONDS}s."
    while IFS= read -r row; do
      [[ -n "$row" ]] && log "compute_job: $row"
    done <<< "$compute_apps"
    sleep "$POLL_SECONDS"
    continue
  fi

  log "No GPU compute jobs found. nvidia-smi state:"
  while IFS= read -r row; do
    [[ -n "$row" ]] && log "gpu: $row"
  done <<< "$gpu_state"

  readiness_passed=1
  for gpu in 0 1; do
    readiness_output="$LOG_ROOT/delayed_gpu${gpu}_readiness.json"
    if ! CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
        "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
        --physical-gpu-id "$gpu" --output "$readiness_output"; then
      log "GPU $gpu nvidia-smi/FP16 readiness did not pass; report=$readiness_output"
      readiness_passed=0
      break
    fi
  done

  if (( readiness_passed == 0 )); then
    log "Readiness gate failed; retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  # Close the gap between the readiness tests and launch: if another workload
  # acquired either GPU, return to the non-contention wait loop.
  if ! compute_apps=$(nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1); then
    log "Final compute-job query failed; retrying in ${POLL_SECONDS}s: $compute_apps"
    sleep "$POLL_SECONDS"
    continue
  fi
  if [[ -n "${compute_apps//[[:space:]]/}" ]]; then
    log "A compute job appeared during readiness checks; not contending. Retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  break
done

log "nvidia-smi and FP16 readiness passed on both GPUs with no compute jobs; starting $TARGET"
cd "$REPO_ROOT"
exec bash "$TARGET"
