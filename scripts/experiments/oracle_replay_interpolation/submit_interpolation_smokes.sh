#!/usr/bin/env bash
# Submit only debug/smoke jobs. Promote each full run manually after its smoke
# has been monitored; this enforces the repository's official-job safety gate.
set -euo pipefail

REPO_DIR="${REPO_DIR:?Set REPO_DIR to the isolated experiment worktree.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/oracle_replay_interpolation_20260730}"
MANIFEST_DIR="$OUTPUT_ROOT/manifest"
MANIFEST="$MANIFEST_DIR/expected_keys.jsonl"
MANIFEST_HASH_FILE="$MANIFEST_DIR/expected_keys.sha256"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT.}"
TASK_YAML="${TASK_YAML:-$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_leonardo_offline/vsibench.yaml}"

if [[ ! -f "$MANIFEST" ]]; then
  python "$REPO_DIR/scripts/experiments/oracle_replay_interpolation/build_expected_key_manifest.py" \
    --task-yaml "$TASK_YAML" --output-dir "$MANIFEST_DIR"
fi
MANIFEST_SHA256="$(awk 'NR==1 {print $1}' "$MANIFEST_HASH_FILE")"
[[ -n "$MANIFEST_SHA256" ]] || { echo "[ERROR] Missing manifest hash." >&2; exit 2; }

for beta in 0.25 0.50 0.75; do
  tag="${beta/./}"
  for mode in single_gpu single_node_4gpu; do
    if [[ "$mode" == single_gpu ]]; then
      wrapper="$REPO_DIR/scripts/experiments/oracle_replay_interpolation/eval_interpolation_single_gpu.sh"
      limit=8
      gpus=1
    else
      wrapper="$REPO_DIR/scripts/experiments/oracle_replay_interpolation/eval_interpolation_four_gpu.sh"
      limit=16
      gpus=4
    fi
    out="$OUTPUT_ROOT/beta_${tag}/$mode/smoke"
    job="SMOKE_Interp${tag}_${mode}"
    sbatch --job-name="$job" --qos=boost_qos_dbg --time=00:30:00 --nodes=1 --gpus-per-node="$gpus" \
      --export="ALL,REPO_DIR=$REPO_DIR,BETA=$beta,OUTPUT_PATH=$out,EXPECTED_KEY_MANIFEST=$MANIFEST,EXPECTED_KEY_MANIFEST_SHA256=$MANIFEST_SHA256,PREDICTOR_CHECKPOINT=$PREDICTOR_CHECKPOINT,LIMIT=$limit" \
      "$wrapper"
  done
done
