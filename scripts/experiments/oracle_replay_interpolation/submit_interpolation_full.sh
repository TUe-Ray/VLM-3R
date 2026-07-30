#!/usr/bin/env bash
# Invoke only after the corresponding smoke job has been inspected and promoted.
set -euo pipefail

REPO_DIR="${REPO_DIR:?Set REPO_DIR to the isolated experiment worktree.}"
: "${BETA:?Set BETA to 0.25, 0.50, or 0.75.}"
: "${MODE:?Set MODE to single_gpu or single_node_4gpu.}"
: "${SMOKE_JOB_ID:?Set the successfully monitored smoke job ID.}"
: "${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT.}"
case "$BETA" in 0.25|0.50|0.75) ;; *) echo "[ERROR] Endpoint evaluations are prohibited." >&2; exit 2;; esac
case "$MODE" in single_gpu) wrapper="$REPO_DIR/scripts/experiments/oracle_replay_interpolation/eval_interpolation_single_gpu.sh"; gpus=1;; single_node_4gpu) wrapper="$REPO_DIR/scripts/experiments/oracle_replay_interpolation/eval_interpolation_four_gpu.sh"; gpus=4;; *) echo "[ERROR] Unknown MODE=$MODE" >&2; exit 2;; esac

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/oracle_replay_interpolation_20260730}"
MANIFEST="$OUTPUT_ROOT/manifest/expected_keys.jsonl"
MANIFEST_SHA256="$(awk 'NR==1 {print $1}' "$OUTPUT_ROOT/manifest/expected_keys.sha256")"
[[ -f "$MANIFEST" && -n "$MANIFEST_SHA256" ]] || { echo "[ERROR] Frozen manifest is absent." >&2; exit 2; }
tag="${BETA/./}"
out="$OUTPUT_ROOT/beta_${tag}/$MODE/full"
job="VSIInterp${tag}_${MODE}"
sbatch --dependency="afterok:$SMOKE_JOB_ID" --job-name="$job" --nodes=1 --gpus-per-node="$gpus" \
  --export="ALL,REPO_DIR=$REPO_DIR,BETA=$BETA,OUTPUT_PATH=$out,EXPECTED_KEY_MANIFEST=$MANIFEST,EXPECTED_KEY_MANIFEST_SHA256=$MANIFEST_SHA256,PREDICTOR_CHECKPOINT=$PREDICTOR_CHECKPOINT" \
  "$wrapper"
