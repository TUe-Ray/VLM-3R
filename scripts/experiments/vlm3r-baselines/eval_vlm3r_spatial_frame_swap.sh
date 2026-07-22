#!/bin/bash
#SBATCH --job-name=SpatialFrameSwap_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/eomt/%x_%j.out
#SBATCH --error=logs/eval/eomt/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"

SPATIAL_FRAME_SWAP="${SPATIAL_FRAME_SWAP:-True}"
SPATIAL_FRAME_SWAP_MODE="${SPATIAL_FRAME_SWAP_MODE:-random_derange}"
SPATIAL_FRAME_SWAP_SEED="${SPATIAL_FRAME_SWAP_SEED:-0}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_eomt/selec_100%_baseline_40390735}"
export RUN_NAME="${RUN_NAME:-selec_100%_baseline_40390735_vsibench_spatial_frame_swap_seed${SPATIAL_FRAME_SWAP_SEED}}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/all_running/selec_100%_baseline_40390735_spatial_frame_swap_seed${SPATIAL_FRAME_SWAP_SEED}}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-spatial_frame_swap_seed${SPATIAL_FRAME_SWAP_SEED}}}"

FRAME_SWAP_MODEL_ARGS="probe_spatial_feature_frame_swap=$SPATIAL_FRAME_SWAP"
FRAME_SWAP_MODEL_ARGS+=",probe_spatial_feature_frame_swap_mode=$SPATIAL_FRAME_SWAP_MODE"
FRAME_SWAP_MODEL_ARGS+=",probe_spatial_feature_frame_swap_seed=$SPATIAL_FRAME_SWAP_SEED"
if [[ -n "${EXTRA_MODEL_ARGS:-}" ]]; then
  export EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS,$FRAME_SWAP_MODEL_ARGS"
else
  export EXTRA_MODEL_ARGS="$FRAME_SWAP_MODEL_ARGS"
fi

mkdir -p "$REPO_DIR/logs/eval/eomt" "$OUTPUT_PATH"

echo "==== Spatial frame-swap VSI-Bench ablation ===="
date
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "RUNTIME_ROOT=$RUNTIME_ROOT"
echo "SPATIAL_FRAME_SWAP=$SPATIAL_FRAME_SWAP"
echo "SPATIAL_FRAME_SWAP_MODE=$SPATIAL_FRAME_SWAP_MODE"
echo "SPATIAL_FRAME_SWAP_SEED=$SPATIAL_FRAME_SWAP_SEED"
echo "EXTRA_MODEL_ARGS=$EXTRA_MODEL_ARGS"
echo "================================================"

bash "$REPO_DIR/eval_vlm3r_orig.sh"

echo "[DONE] Spatial frame-swap eval artifacts are under: $OUTPUT_PATH"
