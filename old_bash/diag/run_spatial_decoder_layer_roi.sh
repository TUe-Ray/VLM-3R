#!/bin/bash
#SBATCH --job-name=spatial_layer_roi
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/diag/%x_%j.out
#SBATCH --error=logs/diag/%x_%j.err
#SBATCH --mem=0

set -eo pipefail

REPO_DIR="/leonardo/home/userexternal/shuang00/VLM-3R"
CONDA_BASE="/leonardo_work/EUHPC_D32_006/miniconda3"
CONDA_ENV="vlm3r"

MODEL_BASE="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_PATH="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384"
DATA_PATH="$REPO_DIR/scripts/VLM_3R/vsibench_data.yaml"
DATA_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r"
SPATIAL_FEATURE_DIR="$DATA_ROOT/spatial_features"
CUT3R_WEIGHTS="$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth"
PI3X_WEIGHTS="/leonardo_scratch/fast/EUHPC_D32_006/hf_models/Pi3X"

RUN_NAME="${RUN_NAME:-cut3r_pi3_dec_m3_m2_m1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/leonardo_scratch/fast/EUHPC_D32_006/diag/roi_similarity/$RUN_NAME}"

SAMPLE_IDS="${SAMPLE_IDS:-001bd120-cb6e-4d77-a87e-a21c4ddc00f8,0021c68f-2e4c-4aac-97ef-e58d05bf40f0,0054a922-4345-4264-86e4-b176f2d110b0}"
DECODER_LAYERS="${DECODER_LAYERS:--3,-2,-1}"

mkdir -p "$REPO_DIR/logs/diag" "$OUTPUT_DIR"
cd "$REPO_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "MODEL_BASE=$MODEL_BASE"
echo "SIGLIP_PATH=$SIGLIP_PATH"
echo "DATA_PATH=$DATA_PATH"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUT3R_WEIGHTS=$CUT3R_WEIGHTS"
echo "PI3X_WEIGHTS=$PI3X_WEIGHTS"
echo "SAMPLE_IDS=$SAMPLE_IDS"
echo "DECODER_LAYERS=$DECODER_LAYERS"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/analysis/analyze_spatial_decoder_layer_roi.py \
  --data_json "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 3 \
  --sample_ids "$SAMPLE_IDS" \
  --frames 16 \
  --anchor_mode center \
  --normalize_mode global_per_figure \
  --save_raw True \
  --decoder_layers="$DECODER_LAYERS" \
  --target_grid 14x14 \
  --pool_mode bilinear \
  --model_base "$MODEL_BASE" \
  --siglip_path "$SIGLIP_PATH" \
  --image_folder "$DATA_ROOT" \
  --video_folder "$DATA_ROOT" \
  --spatial_feature_dir "$SPATIAL_FEATURE_DIR" \
  --frames_upbound 32 \
  --video_fps 1 \
  --cut3r_weights "$CUT3R_WEIGHTS" \
  --pi3x_weights "$PI3X_WEIGHTS" \
  --pi3x_input_size 518 \
  --encoder both \
  --device cuda:0 \
  --dtype float16 \
  --seed 42

echo "[DONE] Spatial decoder layer ROI output: $OUTPUT_DIR"
