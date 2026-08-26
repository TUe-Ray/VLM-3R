#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825"
MARKER="${OUTPUT_ROOT}/eomt_object_probes_paused_for_20260826_0200.json"
LOG="/home/shaoruei/SpatialFocus/logs/post_sft_geometry_probes/full/eomt_resume_after_object_20260825_2230.log"
CRON_TAG="# spatialfocus-eomt-resume-20260825-2230"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "$LOG"; }
remove_self() { crontab -l 2>/dev/null | sed "\\|$CRON_TAG|d" | crontab -; }

if [[ ! -f "$MARKER" ]]; then
  log "No pause marker; refusing to resume anything."
  remove_self
  exit 1
fi
pid=$(jq -r '.wrapper_pid // empty' "$MARKER")
pgid=$(jq -r '.process_group // empty' "$MARKER")
if [[ -z "$pid" || -z "$pgid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  log "Paused wrapper PID is absent; no signal sent."
  remove_self
  exit 1
fi
command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
if [[ "$command_line" != *"run_post_sft_eomt_full_local.sh"* ]]; then
  log "PID $pid no longer matches the EoMT wrapper; no signal sent."
  remove_self
  exit 1
fi
kill -CONT -- "-$pgid"
log "Resumed EoMT wrapper process group $pgid."
remove_self
