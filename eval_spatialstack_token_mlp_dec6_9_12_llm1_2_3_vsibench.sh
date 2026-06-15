#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SS_Token123_VSI
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

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

RUN_BASENAME="${RUN_BASENAME:-cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3}"
TRAIN_RUN_NAME_8N="${TRAIN_RUN_NAME_8N:-${RUN_BASENAME}_8n}"
TRAIN_RUN_NAME_4N="${TRAIN_RUN_NAME_4N:-${RUN_BASENAME}_4n}"

if [[ -z "${PRETRAINED_LOCAL:-}" ]] && ! is_true "${RANDOM_WEIGHT_SMOKE:-False}"; then
  candidate_8n="$TRAIN_MODEL_ROOT/$TRAIN_RUN_NAME_8N"
  candidate_4n="$TRAIN_MODEL_ROOT/$TRAIN_RUN_NAME_4N"
  if [[ -f "$candidate_8n/config.json" ]]; then
    export PRETRAINED_LOCAL="$candidate_8n"
  elif [[ -f "$candidate_4n/config.json" ]]; then
    export PRETRAINED_LOCAL="$candidate_4n"
  else
    echo "[ERROR] Could not resolve trained checkpoint. Checked:"
    echo "  $candidate_8n"
    echo "  $candidate_4n"
    exit 1
  fi
fi

export RUN_NAME="${RUN_NAME:-eval_${RUN_BASENAME}_vsibench}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/${RUN_BASENAME}_vsibench}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-eval_${RUN_BASENAME}_vsibench}}"

export CUT3R_SPATIALSTACK_LAYERS="${CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
export CUT3R_SPATIALSTACK_LLM_LAYERS="${CUT3R_SPATIALSTACK_LLM_LAYERS:-1,2,3}"
export CUT3R_SPATIALSTACK_PROJECTOR_TYPE="${CUT3R_SPATIALSTACK_PROJECTOR_TYPE:-token_mlp}"
export CUT3R_SPATIALSTACK_MERGE_SIZE="${CUT3R_SPATIALSTACK_MERGE_SIZE:-2}"
export CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM="${CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM:-4096}"

export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
export CUT3R_TOKEN_FEATURES_ROOT="${CUT3R_TOKEN_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:$CUT3R_TOKEN_FEATURES_ROOT:spatial_features}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"

mkdir -p "$REPO_DIR/logs/eval" "$OUTPUT_PATH"

echo "==== CUT3R SpatialStack token_mlp LLM 1/2/3 VSI-Bench eval ===="
date
echo "PRETRAINED_LOCAL=${PRETRAINED_LOCAL:-<random-weight-smoke>}"
echo "TRAIN_RUN_NAME_8N=$TRAIN_RUN_NAME_8N"
echo "TRAIN_RUN_NAME_4N=$TRAIN_RUN_NAME_4N"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "RANDOM_WEIGHT_SMOKE=${RANDOM_WEIGHT_SMOKE:-False}"
echo "================================================================"

bash "$REPO_DIR/eval_spatialstack_vsibench.sh"

echo "[DONE] token_mlp eval artifacts are under: $OUTPUT_PATH"
