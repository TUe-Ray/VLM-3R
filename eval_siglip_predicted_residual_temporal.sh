#!/usr/bin/env bash
#SBATCH --job-name=EvalSigLIPResidualTemporal
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
: "${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT to a temporal predictor checkpoint.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for VSI-Bench results.}"
[[ -f "$PREDICTOR_CHECKPOINT" ]] || { echo "[ERROR] Predictor checkpoint not found: $PREDICTOR_CHECKPOINT" >&2; exit 1; }
PREDICTOR_CHECKPOINT="$(readlink -f "$PREDICTOR_CHECKPOINT")"
OUTPUT_PATH="$(realpath -m "$OUTPUT_PATH")"
if [[ -n "${DEDUPLICATE_AGAINST:-}" ]]; then
  [[ -f "$DEDUPLICATE_AGAINST" ]] || { echo "[ERROR] DEDUPLICATE_AGAINST checkpoint not found: $DEDUPLICATE_AGAINST" >&2; exit 1; }
  state_hashes="$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c 'import sys, torch; from llava.model.siglip_spatialstack_residual import predictor_state_sha256; [print(predictor_state_sha256(torch.load(path, map_location="cpu", weights_only=False)["predictor"])) for path in sys.argv[1:]]' "$PREDICTOR_CHECKPOINT" "$DEDUPLICATE_AGAINST")"
  read -r checkpoint_hash reference_hash <<<"$state_hashes"
  if [[ "$checkpoint_hash" == "$reference_hash" ]]; then
    mkdir -p "$OUTPUT_PATH"
    printf '{"status":"DEDUPLICATED","predictor_state_sha256":"%s","duplicate_of":"%s"}\n' "$checkpoint_hash" "$DEDUPLICATE_AGAINST" > "$OUTPUT_PATH/results.json"
    echo "DEDUPLICATED: predictor-state SHA256 matches $DEDUPLICATE_AGAINST"
    exit 0
  fi
fi
export CHECK_SPATIAL_SIDECARS=False
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/dev/null}"
export RUN_NAME="${RUN_NAME:-siglip_predicted_residual_temporal}"
control="${PREDICTED_RESIDUAL_CONTROL:-none}"
extra="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}use_predicted_spatialstack_residuals=true,residual_predictor_type=${RESIDUAL_PREDICTOR_TYPE:-auto},residual_predictor_checkpoint=$PREDICTOR_CHECKPOINT,predicted_residual_gamma_layer0=${GAMMA_LAYER0:-1.0},predicted_residual_gamma_layer1=${GAMMA_LAYER1:-1.0},predicted_residual_gamma_layer2=${GAMMA_LAYER2:-1.0},predicted_residual_control=$control"
if [[ "$control" == "calibrated" ]]; then
  : "${RESIDUAL_CALIBRATION_ARTIFACT:?Set RESIDUAL_CALIBRATION_ARTIFACT for calibrated evaluation.}"
  [[ -f "$RESIDUAL_CALIBRATION_ARTIFACT" ]] || { echo "[ERROR] Calibration artifact not found: $RESIDUAL_CALIBRATION_ARTIFACT" >&2; exit 1; }
  extra+=",residual_calibration_artifact=$(readlink -f "$RESIDUAL_CALIBRATION_ARTIFACT")"
fi
export EXTRA_MODEL_ARGS="$extra"
exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
