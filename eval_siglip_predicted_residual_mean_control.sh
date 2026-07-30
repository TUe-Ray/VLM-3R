#!/usr/bin/env bash
#SBATCH --job-name=EvalSigLIPResidualMean
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
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
: "${MEAN_RESIDUAL_ARTIFACT:?Set MEAN_RESIDUAL_ARTIFACT.}"
[[ -f "$MEAN_RESIDUAL_ARTIFACT" ]] || { echo "[ERROR] Mean residual artifact not found: $MEAN_RESIDUAL_ARTIFACT" >&2; exit 1; }
MEAN_RESIDUAL_ARTIFACT="$(readlink -f "$MEAN_RESIDUAL_ARTIFACT")"
OUTPUT_PATH="$(realpath -m "$OUTPUT_PATH")"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for VSI-Bench results.}"
export CHECK_SPATIAL_SIDECARS=False
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/dev/null}"
export RUN_NAME="${RUN_NAME:-siglip_predicted_residual_mean_control}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}use_predicted_spatialstack_residuals=true,residual_predictor_type=${RESIDUAL_PREDICTOR_TYPE:-token_mlp},predicted_residual_gamma_layer0=${GAMMA_LAYER0:-1.0},predicted_residual_gamma_layer1=${GAMMA_LAYER1:-1.0},predicted_residual_gamma_layer2=${GAMMA_LAYER2:-1.0},predicted_residual_control=mean,mean_residual_artifact=$MEAN_RESIDUAL_ARTIFACT"
exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
