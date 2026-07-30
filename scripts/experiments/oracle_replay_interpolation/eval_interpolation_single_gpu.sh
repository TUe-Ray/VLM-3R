#!/usr/bin/env bash
#SBATCH --job-name=VSIInterp1G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --output=logs/oracle_replay_interpolation/%x_%j.out
#SBATCH --error=logs/oracle_replay_interpolation/%x_%j.err
set -euo pipefail

REPO_DIR="${REPO_DIR:?Set REPO_DIR to the isolated experiment worktree.}"
: "${BETA:?Set BETA to 0.25, 0.50, or 0.75.}"
: "${OUTPUT_PATH:?Set a mode-specific output path.}"
: "${EXPECTED_KEY_MANIFEST:?Set the frozen expected_keys.jsonl path.}"
: "${EXPECTED_KEY_MANIFEST_SHA256:?Set the frozen manifest SHA256.}"
: "${PREDICTOR_CHECKPOINT:?Set the Temporal best_validation_relative_l2 checkpoint.}"
: "${PRETRAINED_LOCAL:=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
: "${EXPECTED_TEACHER_CHECKPOINT:=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
[[ "$PRETRAINED_LOCAL" == "$EXPECTED_TEACHER_CHECKPOINT" ]] || { echo "[ERROR] Interpolation must use $EXPECTED_TEACHER_CHECKPOINT, got $PRETRAINED_LOCAL." >&2; exit 2; }
export PRETRAINED_LOCAL
case "$BETA" in 0.25|0.50|0.75) ;; *) echo "[ERROR] Only intermediate beta values are runnable." >&2; exit 2;; esac
[[ -f "$EXPECTED_KEY_MANIFEST" && -f "$PREDICTOR_CHECKPOINT" ]] || { echo "[ERROR] Missing manifest or predictor checkpoint." >&2; exit 2; }

export NUM_PROCESSES=1
export RUN_NAME="${RUN_NAME:-oracle_replay_interp_beta_${BETA//./}_single_gpu}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$OUTPUT_PATH/runtime}"
export CHECK_SPATIAL_SIDECARS=True
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r:spatial_features}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}spatialstack_residual_mode=interpolate,spatialstack_residual_beta=$BETA,use_predicted_spatialstack_residuals=true,residual_predictor_type=auto,residual_predictor_checkpoint=$PREDICTOR_CHECKPOINT,expected_key_manifest=$EXPECTED_KEY_MANIFEST,expected_key_manifest_sha256=$EXPECTED_KEY_MANIFEST_SHA256,evaluation_telemetry_dir=$OUTPUT_PATH/telemetry"
bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
python "$REPO_DIR/scripts/experiments/oracle_replay_interpolation/validate_evaluation_completion.py" --output-path "$OUTPUT_PATH" --expected-samples "${LIMIT:-5130}" --world-size 1
