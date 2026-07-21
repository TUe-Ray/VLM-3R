#!/bin/bash
#SBATCH --account=EUHPC_D32_006
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=DBG_cut3r_pointmap_pose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/DBG_cut3r_pointmap_pose_%x_%j.out
#SBATCH --error=logs/DBG_cut3r_pointmap_pose_%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load cuda/12.1

REPO=/leonardo/home/userexternal/shuang00/VLM-3R
SIDECAR_ROOT=${SIDECAR_ROOT:-/leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_pointmaps/scannet/spatial_features_points}
VIDEO_ROOT=${VIDEO_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/videos}
CHECKPOINT=${CHECKPOINT:-$REPO/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/cut3r_pointmap_pose_diagnostic}
SAMPLE_COUNT=${SAMPLE_COUNT:-20}
SEED=${SEED:-17}
SAVED_FRAMES=${SAVED_FRAMES:-3}
PRECISION=${PRECISION:-bf16}
ENRICH_REPRESENTATIVES=${ENRICH_REPRESENTATIVES:-1}

mkdir -p "$REPO/logs" "$OUTPUT_DIR"
cd "$REPO"

ARGS=(
  --sidecar-root "$SIDECAR_ROOT"
  --video-root "$VIDEO_ROOT"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --sample-count "$SAMPLE_COUNT"
  --seed "$SEED"
  --saved-frames "$SAVED_FRAMES"
  --precision "$PRECISION"
)

if [[ "$ENRICH_REPRESENTATIVES" == "1" ]]; then
  ARGS+=(--enrich-representatives)
fi
if [[ -n "${SCENES:-}" ]]; then
  ARGS+=(--scenes "$SCENES")
fi
if [[ -n "${SCANNET_GT_ROOT:-}" ]]; then
  ARGS+=(--scannet-gt-root "$SCANNET_GT_ROOT")
fi

conda run --no-capture-output -n vlm3r python -u scripts/analysis/diagnose_cut3r_pointmap_pose_alignment.py "${ARGS[@]}"
