#!/bin/bash
#SBATCH --job-name=cut3r_spatialstack
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

export NOTE="${NOTE:-CUT3R Semi-SpatialStack dense residual injection: pre-extracted sidecars only, no pre-LLM fusion block.}"
export SUFFIX="${SUFFIX:-vlm_3r_vsibench_cut3r_spatialstack_lora}"

export MODEL_USE_CUT3R_SPATIALSTACK=True
export MODEL_TUNE_CUT3R_SPATIALSTACK=True
export MODEL_CUT3R_SPATIALSTACK_LAYERS="6,9,12"
export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="0,1,2"
export MODEL_CUT3R_SPATIALSTACK_FEATURE_DIM="${MODEL_CUT3R_SPATIALSTACK_FEATURE_DIM:-768}"
export MODEL_CUT3R_SPATIALSTACK_FEATURE_KEY="${MODEL_CUT3R_SPATIALSTACK_FEATURE_KEY:-cut3r_dec_layers}"
export MODEL_CUT3R_SPATIALSTACK_ZERO_INIT="${MODEL_CUT3R_SPATIALSTACK_ZERO_INIT:-True}"

export MODEL_SPATIAL_TOWER="cut3r"
export MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY=True
export MODEL_FUSION_BLOCK=""
export MODEL_TUNE_FUSION_BLOCK=False

export MODEL_USE_GEOMETRY_AWARE_PROJECTION=False
export MODEL_TUNE_GEOMETRY_AWARE_PROJECTION=False
export MODEL_USE_AUXILIARY_GEOMETRY_HEAD=False
export MODEL_USE_AUXILIARY_GEOMETRY_LOSS=False
export MODEL_USE_BEV_SUPERVISION=False
export MODEL_USE_DEPTH_SUPERVISION=False
export MODEL_LLM_VISUAL_3D_ROPE_ENABLE=False

exec bash train_cut3r_Baseline.sh
