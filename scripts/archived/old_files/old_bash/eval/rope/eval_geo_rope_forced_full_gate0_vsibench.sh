#!/bin/bash
#SBATCH --job-name=GeoRoPEForcedFullGate0_VSiBench
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/geo_rope_spherical_forced_full_rope_resume_fast_workfb_42445435}"
export RUN_NAME="${RUN_NAME:-eval_geo_rope_forced_full_42445435_gate0}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/geo_rope_spherical_forced_full_42445435_gate0}"
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_geo_rope_fusion}"
export MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
export FORCE_GEO_ROPE_GATE_ZERO="${FORCE_GEO_ROPE_GATE_ZERO:-True}"
export RUN_RUNTIME_IMPORT_CHECKS="${RUN_RUNTIME_IMPORT_CHECKS:-False}"

export LEARNED_RESULT_DIR="${LEARNED_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/geo_rope_spherical_forced_full_rope_42445435}"
export BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/all_running/selec_100%_baseline_40390735}"

bash "${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}/eval_geo_rope_gate0_vsibench.sh"
