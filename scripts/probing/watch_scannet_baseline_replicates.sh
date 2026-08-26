#!/usr/bin/env bash
# Wait for existing post-SFT work, then smoke and run confirmed baseline replications.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
RUNNER="$REPO_ROOT/scripts/probing/run_scannet_baseline_replicates_local.sh"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_baseline_replicates_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/scannet_baseline_replicates_v1}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/scannet_baseline_replicates_v1}"
LABEL="${LABEL:-baseline_apr30_40390735}"
WAIT_FOR_UNIT="${WAIT_FOR_UNIT:-}"
SKIP_SMOKE="${SKIP_SMOKE:-false}"
POLL_SECONDS="${POLL_SECONDS:-60}"

export PATH="/home/shaoruei/miniconda3/bin:${PATH:-}"
mkdir -p "$LOG_ROOT"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

wait_for_turn() {
  local prerequisite_status
  while true; do
    if [[ -n "$WAIT_FOR_UNIT" ]] && systemctl --user is-active --quiet "$WAIT_FOR_UNIT"; then
      log "prerequisite unit $WAIT_FOR_UNIT still active; waiting ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
      continue
    fi
    if [[ -n "$WAIT_FOR_UNIT" ]]; then
      prerequisite_status="$(systemctl --user show "$WAIT_FOR_UNIT" --property=Result,ExecMainStatus --value 2>/dev/null | tr '\n' ' ')"
      if [[ "$prerequisite_status" != "success 0 " ]]; then
        log "prerequisite unit $WAIT_FOR_UNIT did not complete successfully ($prerequisite_status); refusing to start"
        exit 1
      fi
    fi
    # Do not race the already-running post-SFT orchestration between its own
    # extraction and probe stages, even during a momentarily idle GPU gap.
    if ps -eo args= | grep -E '[s]cripts/probing/run_post_sft_(eomt|geo_rope)_full_local\.sh' >/dev/null; then
      log "existing post-SFT runner still active; waiting ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
      continue
    fi
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${apps//[[:space:]]/}" ]]; then
      log "GPU compute process present; waiting ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
      continue
    fi
    return
  done
}

verify_gpus() {
  local gpu
  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n vlm3r python -u \
      "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
      --physical-gpu-id "$gpu" --output "$LOG_ROOT/gpu${gpu}_readiness.json"
  done
}

wait_for_turn
verify_gpus
if [[ "$SKIP_SMOKE" == "true" ]]; then
  log "smoke waived for $LABEL: architecture/config/LoRA recipe match the already-passed April-30 smoke"
else
  log "starting 14-point smoke: $LABEL"
  GPU=0 CUDA_DEVICES=0,1 CACHE_ROOT="$CACHE_ROOT" DURABLE_ROOT="$DURABLE_ROOT" LOG_ROOT="$LOG_ROOT" \
    bash "$RUNNER" smoke-one "$LABEL"

  smoke_root="$CACHE_ROOT/smoke/$LABEL"
  smoke_report="$smoke_root/smoke_verification.json"
  if ! jq -e '.assessment == "PASS"' "$smoke_report" >/dev/null; then
    log "smoke did not pass; retaining cache at $smoke_root"
    exit 1
  fi
  mkdir -p "$DURABLE_ROOT/provenance/$LABEL"
  cp -a "$smoke_report" "$DURABLE_ROOT/provenance/$LABEL/smoke_verification.json"
  case "$smoke_root" in
    "$CACHE_ROOT"/smoke/*) rm -rf -- "$smoke_root" ;;
    *) log "refusing unexpected smoke cleanup path: $smoke_root"; exit 1 ;;
  esac
  log "smoke passed and temporary smoke cache removed"
fi

wait_for_turn
verify_gpus
log "starting full 14-point ScanNet probe: $LABEL"
GPU=0 CUDA_DEVICES=0,1 CACHE_ROOT="$CACHE_ROOT" DURABLE_ROOT="$DURABLE_ROOT" LOG_ROOT="$LOG_ROOT" \
  bash "$RUNNER" run-one "$LABEL"
log "completed and verified: $LABEL"
