#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SS_Token036_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

RUN_BASENAME="${RUN_BASENAME:-cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6}"

if [[ -z "${PRETRAINED_LOCAL:-}" ]]; then
  shopt -s nullglob
  checkpoint_candidates=()
  for candidate in "$TRAIN_MODEL_ROOT/$RUN_BASENAME" "$TRAIN_MODEL_ROOT"/"${RUN_BASENAME}_"*; do
    if [[ -f "$candidate/config.json" ]]; then
      checkpoint_candidates+=("$candidate")
    fi
  done
  shopt -u nullglob

  if (( ${#checkpoint_candidates[@]} == 1 )); then
    export PRETRAINED_LOCAL="${checkpoint_candidates[0]}"
  elif (( ${#checkpoint_candidates[@]} == 0 )); then
    echo "[ERROR] No checkpoint found for $RUN_BASENAME under $TRAIN_MODEL_ROOT."
    echo "[ERROR] Set PRETRAINED_LOCAL to the completed training checkpoint."
    exit 1
  else
    echo "[ERROR] Multiple checkpoints found for $RUN_BASENAME:"
    printf '  %s\n' "${checkpoint_candidates[@]}"
    echo "[ERROR] Set PRETRAINED_LOCAL explicitly to avoid evaluating the wrong checkpoint."
    exit 1
  fi
fi

export RUN_NAME="${RUN_NAME:-eval_${RUN_BASENAME}_vsibench}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/${RUN_BASENAME}_vsibench}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-eval_${RUN_BASENAME}_vsibench}}"

# The evaluator validates these values against config.json but never writes them into it.
export PRESERVE_CHECKPOINT_CONFIG="True"
export EXPECTED_CUT3R_SPATIALSTACK_LAYERS="${EXPECTED_CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS="${EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS:-0,3,6}"
export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_TYPE="${EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_TYPE:-token_mlp}"

export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
export CUT3R_TOKEN_FEATURES_ROOT="${CUT3R_TOKEN_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:$CUT3R_TOKEN_FEATURES_ROOT:spatial_features}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"

mkdir -p "$REPO_DIR/logs/eval" "$OUTPUT_PATH"

echo "==== CUT3R SpatialStack token_mlp LLM 0/3/6 VSI-Bench eval ===="
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "PRESERVE_CHECKPOINT_CONFIG=$PRESERVE_CHECKPOINT_CONFIG"
echo "================================================================"

exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
