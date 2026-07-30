#!/usr/bin/env bash
# Evaluate the earliest successful member of a 4-node/8-node DualPath pair.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
TRAIN_4_JOB_ID="${TRAIN_4_JOB_ID:?Set TRAIN_4_JOB_ID.}"
TRAIN_8_JOB_ID="${TRAIN_8_JOB_ID:?Set TRAIN_8_JOB_ID.}"
TRAIN_4_CHECKPOINT="${TRAIN_4_CHECKPOINT:?Set TRAIN_4_CHECKPOINT.}"
TRAIN_8_CHECKPOINT="${TRAIN_8_CHECKPOINT:?Set TRAIN_8_CHECKPOINT.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:?Set RUN_NAME_PREFIX.}"
SELECT_TIMEOUT_SECONDS="${SELECT_TIMEOUT_SECONDS:-300}"

completion_time() {
  local job_id="$1"
  sacct -X -n -P -j "$job_id" --format=JobIDRaw,State,End 2>/dev/null | \
    awk -F'|' -v expected="$job_id" '$1 == expected && $2 == "COMPLETED" && $3 != "Unknown" { print $3; exit }'
}

selected_job=""
selected_checkpoint=""
selected_end=""
deadline=$((SECONDS + SELECT_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  for candidate_job in "$TRAIN_4_JOB_ID" "$TRAIN_8_JOB_ID"; do
    candidate_end="$(completion_time "$candidate_job")"
    [[ -n "$candidate_end" ]] || continue
    if [[ -z "$selected_end" || "$candidate_end" < "$selected_end" ]]; then
      selected_job="$candidate_job"
      selected_end="$candidate_end"
      if [[ "$candidate_job" == "$TRAIN_4_JOB_ID" ]]; then
        selected_checkpoint="$TRAIN_4_CHECKPOINT"
      else
        selected_checkpoint="$TRAIN_8_CHECKPOINT"
      fi
    fi
  done
  [[ -n "$selected_job" ]] && break
  sleep 5
done

if [[ -z "$selected_job" ]]; then
  echo "[ERROR] Neither candidate training job completed successfully within ${SELECT_TIMEOUT_SECONDS}s: $TRAIN_4_JOB_ID, $TRAIN_8_JOB_ID"
  exit 1
fi
if [[ ! -d "$selected_checkpoint" ]]; then
  echo "[ERROR] Selected checkpoint is unavailable: $selected_checkpoint"
  exit 1
fi

export PRETRAINED_LOCAL="$selected_checkpoint"
export OUTPUT_PATH="${OUTPUT_ROOT%/}/${RUN_NAME_PREFIX}_from_${selected_job}"
export RUN_NAME="${RUN_NAME_PREFIX}_from_${selected_job}"
echo "[SELECT] earliest successful training job=$selected_job end=$selected_end checkpoint=$selected_checkpoint"
exec bash "$REPO_DIR/eval_cut3r_dualpath_vsibench.sh"
