#!/bin/bash
#SBATCH --job-name=DBG_bev_align_forward
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/probing/%x_%j.out
#SBATCH --error=logs/probing/%x_%j.err

set -euo pipefail

module load cuda/12.1
module load cudnn
module load profile/deeplrn

export PATH="/leonardo_work/EUHPC_D32_006/miniconda3/bin:$PATH"
set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-vlm3r}"
set -u

mkdir -p logs/probing

MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE="${MODEL_BASE:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
MODEL_PATH="${MODEL_PATH:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/geo_rope_spherical_forced_full_rope_resume_fast_workfb_42445435}"
FEATURE_ROOT="${FEATURE_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"
POINT_MAPS_SUBDIR="${POINT_MAPS_SUBDIR:-spatial_features_points}"
TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-scripts/VLM_3R/vsibench_data.yaml}"
BEV_POINT_MAP_KEY="${BEV_POINT_MAP_KEY:-point_maps_ref}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/probing/bev_alignment_forward_${SLURM_JOB_ID}}"

export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
RUNTIME_TMP="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_TMP/matplotlib_${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

echo "[BEV ALIGN] model_path=$MODEL_PATH"
echo "[BEV ALIGN] model_base=$MODEL_BASE"
echo "[BEV ALIGN] feature_root=$FEATURE_ROOT"
echo "[BEV ALIGN] output_dir=$OUTPUT_DIR"

python scripts/probing/debug_bev_alignment.py \
  --train-data-json "$TRAIN_DATA_JSON" \
  --feature-root "$FEATURE_ROOT" \
  --spatial-features-subdir "$SPATIAL_FEATURES_SUBDIR" \
  --point-maps-subdir "$POINT_MAPS_SUBDIR" \
  --bev-point-map-key "$BEV_POINT_MAP_KEY" \
  --sample-index "$SAMPLE_INDEX" \
  --output-dir "$OUTPUT_DIR" \
  --run-model-forward \
  --model-path "$MODEL_PATH" \
  --model-base "$MODEL_BASE" \
  --device cuda:0 \
  --dtype bfloat16 \
  --skip-spatial-tower-load True
