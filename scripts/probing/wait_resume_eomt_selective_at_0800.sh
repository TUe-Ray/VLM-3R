#!/usr/bin/env bash
set -euo pipefail

# Resume a deliberately stopped EoMT selective process group only after the
# requested local time and only when no *other* GPU compute job is present.
# Keeping the process stopped retains its exact in-flight state, so previously
# saved features are neither overwritten nor extracted again.

REPO_ROOT="/home/shaoruei/SpatialFocus"
WRAPPER_PID="${1:?wrapper PID is required}"
TARGET_LOCAL="${TARGET_LOCAL:-2026-08-26 08:00:00}"
LOG="$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_selective_resume_watch_20260826.log"

mkdir -p "$(dirname "$LOG")"
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"; }

check_owned_wrapper() {
  if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then
    log "wrapper PID $WRAPPER_PID is absent; refusing to resume or restart it."
    return 1
  fi
  local pgid cmd
  pgid=$(ps -o pgid= -p "$WRAPPER_PID" | tr -d ' ')
  cmd=$(tr '\0' ' ' < "/proc/$WRAPPER_PID/cmdline")
  if [[ "$pgid" != "$WRAPPER_PID" || "$cmd" != *"run_post_sft_eomt_selective_local.sh"* ]]; then
    log "PID $WRAPPER_PID no longer identifies the expected selective wrapper; refusing action."
    return 1
  fi
  printf '%s\n' "$pgid"
}

has_external_compute_job() {
  local pgid="$1" apps pid
  if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>&1); then
    log "nvidia-smi compute query failed: $apps"
    return 0
  fi
  while IFS= read -r pid; do
    pid=${pid//[[:space:]]/}
    [[ -z "$pid" ]] && continue
    if [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" != "$pgid" ]]; then
      log "external GPU compute PID $pid is present; keeping selective paused."
      return 0
    fi
  done <<< "$apps"
  return 1
}

target_epoch=$(date -d "$TARGET_LOCAL" +%s)
log "watcher armed for $TARGET_LOCAL; checking every 30 minutes after the target."

while (( $(date +%s) < target_epoch )); do
  sleep 60
done

while true; do
  pgid=$(check_owned_wrapper) || exit 1
  if has_external_compute_job "$pgid"; then
    log "GPU availability check did not pass; retrying in 30 minutes."
    sleep 1800
    continue
  fi
  kill -CONT -- "-$pgid"
  log "GPU availability check passed; resumed owned selective process group $pgid."
  exit 0
done
