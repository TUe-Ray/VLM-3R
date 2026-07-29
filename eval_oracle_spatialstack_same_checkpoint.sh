#!/usr/bin/env bash
#SBATCH --job-name=EvalOracleSpatialStack
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
: "${SPATIAL_FEATURES_ROOT:?Set SPATIAL_FEATURES_ROOT for oracle CUT3R sidecars.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for VSI-Bench results.}"
export RUN_NAME="${RUN_NAME:-oracle_spatialstack_same_checkpoint}"
export CHECK_SPATIAL_SIDECARS=True
exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
