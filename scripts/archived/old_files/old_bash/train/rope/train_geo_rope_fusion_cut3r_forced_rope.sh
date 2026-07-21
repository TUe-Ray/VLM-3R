#!/bin/bash
#SBATCH --job-name=geo_rope_spherical_forced_full_rope
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_geo_rope_fusion_forced}"
export MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-True}"
export MODEL_GEO_ROPE_FUSION_LOG_ATTENTION_STATS="${MODEL_GEO_ROPE_FUSION_LOG_ATTENTION_STATS:-False}"
export NOTE="${NOTE:-CUT3R GeoRoPE Fusion ablation: forced full spherical RoPE Q/K, no gate}"

exec bash "$SCRIPT_DIR/train_geo_rope_fusion_cut3r.sh"
