#!/usr/bin/env bash
#SBATCH --job-name=SMOKEOracleReplayParity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output=logs/oracle_replay_interpolation/%x_%j.out
#SBATCH --error=logs/oracle_replay_interpolation/%x_%j.err
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
: "${PRETRAINED_LOCAL:=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
: "${EXPECTED_TEACHER_CHECKPOINT:=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
[[ "$PRETRAINED_LOCAL" == "$EXPECTED_TEACHER_CHECKPOINT" ]] || { echo "[ERROR] Oracle replay must use $EXPECTED_TEACHER_CHECKPOINT, got $PRETRAINED_LOCAL." >&2; exit 2; }
export PRETRAINED_LOCAL
: "${OUTPUT_PATH:?Set OUTPUT_PATH.}"
: "${EXPECTED_KEY_MANIFEST:?Set the frozen expected_keys.jsonl path.}"
: "${EXPECTED_KEY_MANIFEST_SHA256:?Set the frozen manifest SHA256.}"
: "${PREDICTOR_CHECKPOINT:?Set the Temporal best_validation_relative_l2 checkpoint for teacher-scale provenance.}"
[[ -f "$EXPECTED_KEY_MANIFEST" && -f "$PREDICTOR_CHECKPOINT" ]] || { echo "[ERROR] Missing expected-key manifest or predictor checkpoint." >&2; exit 2; }
export NUM_PROCESSES=1
export RUN_NAME="${RUN_NAME:-oracle_replay_payload_parity}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$OUTPUT_PATH/runtime}"
export CHECK_SPATIAL_SIDECARS=True
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r:spatial_features}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}spatialstack_residual_mode=oracle_replay_parity,residual_predictor_checkpoint=$PREDICTOR_CHECKPOINT,expected_key_manifest=$EXPECTED_KEY_MANIFEST,expected_key_manifest_sha256=$EXPECTED_KEY_MANIFEST_SHA256,evaluation_telemetry_dir=$OUTPUT_PATH/telemetry"
exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
