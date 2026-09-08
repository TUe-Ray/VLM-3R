#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/shaoruei/SpatialFocus"
LOG_ROOT="$REPO_ROOT/logs/post_sft_geometry_probes/full"
GEO_WRAPPER="$REPO_ROOT/scripts/probing/run_post_sft_geo_rope_full_local.sh"
EOMT_WRAPPER="$REPO_ROOT/scripts/probing/run_post_sft_eomt_full_local.sh"

mkdir -p "$LOG_ROOT"
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

exec 9>"$LOG_ROOT/delayed_eomt_launch.lock"
if ! flock -n 9; then
  log "Another delayed EoMT launcher already holds the lock; exiting."
  exit 1
fi

while pgrep -f -x "bash $GEO_WRAPPER" >/dev/null; do
  log "GeoRoPE wrapper is still active; EoMT will not contend. Retrying in 60 seconds."
  sleep 60
done

log "GeoRoPE wrapper has exited; delegating GPU-idle/readiness checks to $EOMT_WRAPPER"
cd "$REPO_ROOT"
exec bash "$EOMT_WRAPPER"
