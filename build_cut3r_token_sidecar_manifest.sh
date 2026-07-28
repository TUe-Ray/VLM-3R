#!/usr/bin/env bash
#SBATCH --job-name=DBG_CUT3RTokenManifest
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data.yaml}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
OUTPUT="${OUTPUT:-$REPO_DIR/diagnostics/cut3r_token_only/sidecar_manifest_pending.json}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

cd "$REPO_DIR"
mkdir -p logs/cut3r_token_only "$(dirname "$OUTPUT")"
module load profile/deeplrn
export PATH="$WORK/miniconda3/bin:$PATH"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

python scripts/build_cut3r_token_sidecar_manifest.py \
  --data-yaml "$DATA_PATH_YAML" \
  --data-root "$DATA_ROOT" \
  --spatial-features-root "$SPATIAL_FEATURES_ROOT" \
  --spatial-features-subdir spatial_features \
  --frames-upbound 32 --video-fps 1 \
  --cut3r-checkpoint "$CUT3R_WEIGHTS" \
  --extraction-script-commit historical_shared_sampler \
  --output "$OUTPUT" "$@"
