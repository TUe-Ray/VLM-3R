#!/bin/bash
#SBATCH --job-name=cut3r_minus4layer_ablation
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CUT3R_LAYER="${CUT3R_LAYER:--2}"

WORK_DATA_ROOT="${WORK_DATA_ROOT:-/leonardo_work/EUHPC_D32_006/train_data/vlm3r}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DATA_ROOT="${DATA_ROOT:-$FAST_DATA_ROOT}"
# DATA_ROOT="${DATA_ROOT:-$WORK_DATA_ROOT}"  # WORK mirror fallback if FAST is unstable.
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-cross_attention}"

case "$CUT3R_LAYER" in
    -1|m1|dec_m1|spatial_features)
        CUT3R_LAYER="-1"
        LAYER_TAG="m1"
        # SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$FAST_DATA_ROOT}"  # FAST CUT3R token sidecars.
        SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
        SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"
        REQUIRED_DATASETS=("scannet" "scannetpp")
        ;;
    -2|m2|dec_m2|spatial_features_dec_m2)
        CUT3R_LAYER="-2"
        LAYER_TAG="m2"
        SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
        SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features_dec_m2}"
        REQUIRED_DATASETS=("scannet" "scannetpp" "arkitscenes")
        ;;
    -4|m4|dec_m4|spatial_features_dec_m4)
        CUT3R_LAYER="-4"
        LAYER_TAG="m4"
        SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
        SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features_dec_m4}"
        REQUIRED_DATASETS=("scannet" "scannetpp" "arkitscenes")
        ;;
    *)
        echo "[ERROR] Unsupported CUT3R_LAYER='$CUT3R_LAYER'. Use -1, -2, or -4."
        exit 1
        ;;
esac

echo "==== CUT3R Layer Ablation ===="
echo "CUT3R_LAYER=$CUT3R_LAYER"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "MODEL_FUSION_BLOCK=$MODEL_FUSION_BLOCK"

for dataset in "${REQUIRED_DATASETS[@]}"; do
    feature_dir="$SPATIAL_FEATURES_ROOT/$dataset/$SPATIAL_FEATURES_SUBDIR"
    if [[ ! -d "$feature_dir" ]]; then
        echo "[ERROR] Missing feature directory: $feature_dir"
        echo "[ERROR] For CUT3R_LAYER=$CUT3R_LAYER, submit through submit_cut3r_layer_ablation_after_extract.sh or wait for extraction to finish."
        exit 1
    fi
    if [[ -z "$(find "$feature_dir" -maxdepth 1 -type f -name '*.pt' -print -quit)" ]]; then
        echo "[ERROR] Feature directory exists but contains no .pt files: $feature_dir"
        echo "[ERROR] Extraction may not have completed successfully."
        exit 1
    fi
done

export REPO_DIR
export DATA_ROOT
export MODEL_ROOT
export SPATIAL_FEATURES_ROOT
export SPATIAL_FEATURES_SUBDIR
export MODEL_FUSION_BLOCK
export SUFFIX="${SUFFIX:-vlm_3r_vsibench_cut3r_dec_${LAYER_TAG}_cross_attn_lora}"
export NOTE="${NOTE:-CUT3R decoder-layer ablation: train VLM3R baseline with layer ${CUT3R_LAYER} pre-extracted spatial features and cross-attention fusion.}"

exec bash "$REPO_DIR/train_cut3r_Baseline.sh"
