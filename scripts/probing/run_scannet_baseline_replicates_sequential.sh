#!/usr/bin/env bash
# Timer entry point: run the two verified baseline replications one at a time.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
WATCHER="$REPO_ROOT/scripts/probing/watch_scannet_baseline_replicates.sh"

for LABEL in baseline_apr30_40390735 baseline_apr05_reproduction; do
  export LABEL
  unset WAIT_FOR_UNIT
  if [[ "$LABEL" == "baseline_apr05_reproduction" ]]; then
    export SKIP_SMOKE=true
  else
    unset SKIP_SMOKE
  fi
  bash "$WATCHER"
done
