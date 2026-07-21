#!/bin/bash
#SBATCH --job-name=GeoRoPEGate0_VSiBench
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

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/rope_spherical_100p_40790070}"
export RUN_NAME="${RUN_NAME:-eval_rope_spherical_100p_40790070_gate0}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/eval_rope_spherical_100p_40790070_gate0}"
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_spherical_rope}"
export MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
export FORCE_GEO_ROPE_GATE_ZERO="${FORCE_GEO_ROPE_GATE_ZERO:-True}"
export RUN_RUNTIME_IMPORT_CHECKS="${RUN_RUNTIME_IMPORT_CHECKS:-False}"
export LIMIT="${LIMIT:-0}"

LEARNED_RESULT_DIR="${LEARNED_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/0507_2335_eval_rope_spherical_100p_40790070_vlm_3r_model_args_dd1948}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/all_running/selec_100%_baseline_40390735}"

mkdir -p "$OUTPUT_PATH"
exec > >(tee -a "$OUTPUT_PATH/gate0_eval.log")

echo "==== Gate0 VSiBench ablation ===="
date
echo "FORCE_GEO_ROPE_GATE_ZERO=$FORCE_GEO_ROPE_GATE_ZERO"
echo "checkpoint path=$PRETRAINED_LOCAL"
echo "fusion_block=$MODEL_FUSION_BLOCK"
echo "geometry_rope_mode=$MODEL_GEO_ROPE_FUSION_MODE"
echo "geometry_rope_max_depth=$MODEL_GEO_ROPE_FUSION_MAX_DEPTH"
echo "geo_rope_point_map_key=$MODEL_GEO_ROPE_POINT_MAP_KEY"
echo "RUN_RUNTIME_IMPORT_CHECKS=$RUN_RUNTIME_IMPORT_CHECKS"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "LEARNED_RESULT_DIR=$LEARNED_RESULT_DIR"
echo "BASELINE_RESULT_DIR=$BASELINE_RESULT_DIR"
echo "LIMIT=$LIMIT"
echo "================================="

bash "$REPO_DIR/eval_rope_fusion_cut3r.sh"

summary_cmd=(
  python
  "$REPO_DIR/scripts/summarize_vsibench_gate0_eval.py"
  --gate0-dir "$OUTPUT_PATH"
  --learned-dir "$LEARNED_RESULT_DIR"
  --output-dir "$OUTPUT_PATH"
)

if [[ -n "$BASELINE_RESULT_DIR" && -e "$BASELINE_RESULT_DIR" ]]; then
  summary_cmd+=(--baseline-dir "$BASELINE_RESULT_DIR")
else
  echo "[WARN] Baseline result dir not found; summary will omit original patch-only baseline: $BASELINE_RESULT_DIR"
fi

printf '[SUMMARY CMD] %q ' "${summary_cmd[@]}"
echo
"${summary_cmd[@]}"

echo "[DONE] Gate0 eval artifacts are under: $OUTPUT_PATH"
