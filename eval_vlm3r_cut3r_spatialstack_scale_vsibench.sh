#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SpatialStack_Scale_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"

CUT3R_SPATIALSTACK_RESIDUAL_SCALE="${CUT3R_SPATIALSTACK_RESIDUAL_SCALE:-0.5}"
SCALE_TAG="${CUT3R_SPATIALSTACK_RESIDUAL_SCALE//./p}"
SCALE_TAG="${SCALE_TAG//-/m}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_44323703}"
export RUN_NAME="${RUN_NAME:-eval_cut3r_spatialstack_44323703_vsibench_residual_scale_${SCALE_TAG}}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/cut3r_spatialstack_residual_scale_${SCALE_TAG}}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-cut3r_spatialstack_residual_scale_${SCALE_TAG}}}"

ABLATION_MODEL_ARGS="cut3r_spatialstack_residual_scale=$CUT3R_SPATIALSTACK_RESIDUAL_SCALE"
if [[ -n "${EXTRA_MODEL_ARGS:-}" ]]; then
  export EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS,$ABLATION_MODEL_ARGS"
else
  export EXTRA_MODEL_ARGS="$ABLATION_MODEL_ARGS"
fi

mkdir -p "$REPO_DIR/logs/eval" "$OUTPUT_PATH"

echo "==== CUT3R SpatialStack residual-scale VSI-Bench ablation ===="
date
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "RUNTIME_ROOT=$RUNTIME_ROOT"
echo "CUT3R_SPATIALSTACK_RESIDUAL_SCALE=$CUT3R_SPATIALSTACK_RESIDUAL_SCALE"
echo "EXTRA_MODEL_ARGS=$EXTRA_MODEL_ARGS"
echo "============================================================="

bash "$REPO_DIR/eval_vlm3r_cut3r_spatialstack_vsibench.sh"

echo "[DONE] CUT3R SpatialStack residual-scale eval artifacts are under: $OUTPUT_PATH"
