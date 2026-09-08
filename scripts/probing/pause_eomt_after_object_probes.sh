#!/usr/bin/env bash
set -euo pipefail

# Pause only the EoMT wrapper started for this post-SFT run, after all fourteen
# object probe metrics are durable.  The wrapper is resumed by the one-shot
# cron continuation script; no user-owned process is ever targeted.
WRAPPER_PID="${1:?wrapper PID is required}"
EXPECTED_METRICS=14
OUTPUT_ROOT="/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825"
LOG="/home/shaoruei/SpatialFocus/logs/post_sft_geometry_probes/full/eomt_pause_after_object_20260825.log"
MARKER="${OUTPUT_ROOT}/eomt_object_probes_paused_for_20260826_0200.json"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"; }

while true; do
  if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then
    log "Wrapper PID $WRAPPER_PID exited before object-probe pause; no signal sent."
    exit 1
  fi
  command_line=$(tr '\0' ' ' < "/proc/$WRAPPER_PID/cmdline")
  if [[ "$command_line" != *"run_post_sft_eomt_full_local.sh"* ]]; then
    log "PID $WRAPPER_PID no longer matches the EoMT wrapper; no signal sent."
    exit 1
  fi
  completed=$(find "$OUTPUT_ROOT/probes/eomt_object" -mindepth 2 -maxdepth 2 -name metrics.json -type f | wc -l)
  if (( completed >= EXPECTED_METRICS )); then
    pgid=$(ps -o pgid= -p "$WRAPPER_PID" | tr -d ' ')
    if [[ -z "$pgid" || "$pgid" != "$WRAPPER_PID" ]]; then
      log "Unexpected wrapper PGID '$pgid'; no signal sent."
      exit 1
    fi
    printf '{\n  "status": "PAUSED",\n  "wrapper_pid": %s,\n  "process_group": %s,\n  "completed_object_probe_metrics": %s,\n  "paused_at": "%s",\n  "resume_at": "2026-08-25T22:30:00+02:00"\n}\n' \
      "$WRAPPER_PID" "$pgid" "$completed" "$(date --iso-8601=seconds)" > "$MARKER"
    kill -STOP -- "-$pgid"
    log "Paused EoMT wrapper process group $pgid after $completed object probe metrics."
    exit 0
  fi
  sleep 10
done
