#!/bin/bash
#SBATCH --job-name=spatialstack_8n4n_watchdog
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/watchdog/%x_%j.out
#SBATCH --error=logs/watchdog/%x_%j.err

set -euo pipefail

EIGHT_JOB_ID="${EIGHT_JOB_ID:?Set EIGHT_JOB_ID to the 8-node training job id.}"
FOUR_JOB_ID="${FOUR_JOB_ID:?Set FOUR_JOB_ID to the 4-node fallback training job id.}"
RUN_LABEL="${RUN_LABEL:-spatialstack}"
DEADLINE_LABEL="${DEADLINE_LABEL:-01:00}"

mkdir -p logs/watchdog

job_state() {
  local job_id="$1"
  local state
  state="$(squeue -h -j "$job_id" -o "%T" 2>/dev/null | head -n 1 || true)"
  if [[ -n "$state" ]]; then
    printf "%s" "$state"
    return 0
  fi
  state="$(sacct -n -X -j "$job_id" --format=State 2>/dev/null | head -n 1 | awk '{print $1}' || true)"
  printf "%s" "${state:-UNKNOWN}"
}

echo "==== SpatialStack 8n/4n watchdog ===="
date
echo "RUN_LABEL=$RUN_LABEL"
echo "EIGHT_JOB_ID=$EIGHT_JOB_ID"
echo "FOUR_JOB_ID=$FOUR_JOB_ID"
echo "DEADLINE_LABEL=$DEADLINE_LABEL"
echo "======================================"

eight_state="$(job_state "$EIGHT_JOB_ID")"
four_state="$(job_state "$FOUR_JOB_ID")"
echo "8-node state at watchdog time: $eight_state"
echo "4-node state at watchdog time: $four_state"

case "$eight_state" in
  RUNNING|COMPLETING|COMPLETED)
    echo "[WATCHDOG] 8-node job started by $DEADLINE_LABEL; canceling 4-node fallback $FOUR_JOB_ID."
    scancel "$FOUR_JOB_ID" || true
    ;;
  *)
    echo "[WATCHDOG] 8-node job did not start by $DEADLINE_LABEL; canceling 8-node job $EIGHT_JOB_ID and leaving 4-node fallback active."
    scancel "$EIGHT_JOB_ID" || true
    ;;
esac

echo "==== Final states ===="
echo "8-node: $(job_state "$EIGHT_JOB_ID")"
echo "4-node: $(job_state "$FOUR_JOB_ID")"
echo "[DONE]"
