#!/usr/bin/env bash
set -euo pipefail

# After a deliberate GPU release, restart the durable --resume selective
# pipeline only after the requested time and a clean GPU/FP16 availability
# check.  The cache scrub protects --resume from an interrupted direct save.

REPO_ROOT="/home/shaoruei/SpatialFocus"
ENV_NAME="vlm3r"
TARGET_LOCAL="${TARGET_LOCAL:-2026-08-26 08:00:00}"
OUT="/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_20260825"
LABEL="eomt_selective"
LOG="$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_selective_restart_watch_20260826.log"
LAUNCH_LOG="$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_selective_full_20260826_restart_launcher.log"

mkdir -p "$(dirname "$LOG")"
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"; }

gpu_ready() {
  local apps gpu
  if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>&1); then
    log "nvidia-smi compute query failed: $apps"
    return 1
  fi
  if [[ -n "${apps//[[:space:]]/}" ]]; then
    log "GPU compute jobs are present; not contending with them."
    return 1
  fi
  for gpu in 0 1; do
    if ! CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
      --physical-gpu-id "$gpu" \
      --output "$REPO_ROOT/logs/post_sft_geometry_probes/full/eomt_selective_restart_gpu${gpu}_readiness_20260826.json" \
      >>"$LOG" 2>&1; then
      log "GPU $gpu FP16 readiness did not pass."
      return 1
    fi
  done
  return 0
}

target_epoch=$(date -d "$TARGET_LOCAL" +%s)
log "restart watcher armed for $TARGET_LOCAL; checking every 30 minutes after the target."
while (( $(date +%s) < target_epoch )); do sleep 60; done

while true; do
  if gpu_ready; then
    log "GPU availability and FP16 readiness passed; scrubbing cache then launching --resume pipeline."
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/scrub_depth_probe_cache.py" \
      --output-root "$OUT" --model-label "$LABEL" \
      --report "$OUT/${LABEL}_pre_restart_cache_scrub.json" >>"$LOG" 2>&1
    setsid bash "$REPO_ROOT/scripts/probing/run_post_sft_eomt_selective_local.sh" \
      >"$LAUNCH_LOG" 2>&1 < /dev/null &
    log "launched restart-safe selective pipeline PID $!"
    exit 0
  fi
  log "retrying GPU availability in 30 minutes."
  sleep 1800
done
