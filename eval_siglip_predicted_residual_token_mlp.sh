#!/usr/bin/env bash
#SBATCH --job-name=EvalSigLIPResidualToken
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-false}" != "true" ]]; then
  echo "Submit this GPU wrapper with: sbatch $0" >&2
  exit 2
fi
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
: "${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT to a token-MlP predictor checkpoint.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for VSI-Bench results.}"
export CHECK_SPATIAL_SIDECARS=False
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/dev/null}"
export RUN_NAME="${RUN_NAME:-siglip_predicted_residual_token_mlp}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}use_predicted_spatialstack_residuals=true,residual_predictor_type=token_mlp,residual_predictor_checkpoint=$PREDICTOR_CHECKPOINT,predicted_residual_gamma_layer0=${GAMMA_LAYER0:-1.0},predicted_residual_gamma_layer1=${GAMMA_LAYER1:-1.0},predicted_residual_gamma_layer2=${GAMMA_LAYER2:-1.0},predicted_residual_control=none"
exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
