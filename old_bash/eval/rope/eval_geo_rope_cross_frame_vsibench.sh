#!/bin/bash
#SBATCH --job-name=GeoRoPEXFrame_VSiBench
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
PROBE_CROSS_FRAME_WINDOW="${PROBE_CROSS_FRAME_WINDOW:-1}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/rope_spherical_100p_40790070}"
export RUN_NAME="${RUN_NAME:-eval_rope_spherical_100p_40790070_xframe_w${PROBE_CROSS_FRAME_WINDOW}}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/eval_rope_spherical_100p_40790070_xframe_w${PROBE_CROSS_FRAME_WINDOW}}"
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_spherical_rope}"
export MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
export FORCE_GEO_ROPE_GATE_ZERO="${FORCE_GEO_ROPE_GATE_ZERO:-False}"
export PROBE_GEOMETRY_SHUFFLE="${PROBE_GEOMETRY_SHUFFLE:-False}"
export PROBE_GEOMETRY_SHUFFLE_MODE="${PROBE_GEOMETRY_SHUFFLE_MODE:-cyclic_shift}"
export PROBE_GEOMETRY_SHUFFLE_SHIFT="${PROBE_GEOMETRY_SHUFFLE_SHIFT:-1}"
export PROBE_GEOMETRY_SHUFFLE_SEED="${PROBE_GEOMETRY_SHUFFLE_SEED:-0}"
export PROBE_CROSS_FRAME_WINDOW
export PROBE_CROSS_FRAME_INCLUDE_SELF="${PROBE_CROSS_FRAME_INCLUDE_SELF:-True}"
export PROBE_CROSS_FRAME_MODE="${PROBE_CROSS_FRAME_MODE:-sliding_window}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"
export RUN_RUNTIME_IMPORT_CHECKS="${RUN_RUNTIME_IMPORT_CHECKS:-False}"
export LIMIT="${LIMIT:-0}"

LEARNED_RESULT_DIR="${LEARNED_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/0507_2335_eval_rope_spherical_100p_40790070_vlm_3r_model_args_dd1948}"
GATE0_RESULT_DIR="${GATE0_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/eval_rope_spherical_100p_40790070_gate0}"
GSHUFFLE_RESULT_DIR="${GSHUFFLE_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/eval_rope_spherical_100p_40790070_gshuffle_shift1}"
XFRAME_W1_RESULT_DIR="${XFRAME_W1_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/eval_rope_spherical_100p_40790070_xframe_w1}"
XFRAME_W2_RESULT_DIR="${XFRAME_W2_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/eval_rope_spherical_100p_40790070_xframe_w2}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$FAST_ROOT/eval/logs/VLM3R/all_running/selec_100%_baseline_40390735}"

mkdir -p "$OUTPUT_PATH" "$REPO_DIR/logs/eval"
exec > >(tee -a "$OUTPUT_PATH/cross_frame_eval.log")

echo "==== Cross-frame VSiBench probe ===="
date
echo "PROBE_CROSS_FRAME_WINDOW=$PROBE_CROSS_FRAME_WINDOW"
echo "PROBE_CROSS_FRAME_INCLUDE_SELF=$PROBE_CROSS_FRAME_INCLUDE_SELF"
echo "PROBE_CROSS_FRAME_MODE=$PROBE_CROSS_FRAME_MODE"
echo "checkpoint path=$PRETRAINED_LOCAL"
echo "fusion_block=$MODEL_FUSION_BLOCK"
echo "geometry_rope_mode=$MODEL_GEO_ROPE_FUSION_MODE"
echo "geometry_rope_max_depth=$MODEL_GEO_ROPE_FUSION_MAX_DEPTH"
echo "geo_rope_point_map_key=$MODEL_GEO_ROPE_POINT_MAP_KEY"
echo "FORCE_GEO_ROPE_GATE_ZERO=$FORCE_GEO_ROPE_GATE_ZERO"
echo "PROBE_GEOMETRY_SHUFFLE=$PROBE_GEOMETRY_SHUFFLE"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "LIMIT=$LIMIT"
echo "===================================="

bash "$REPO_DIR/eval_rope_fusion_cut3r.sh"

primary_label="D2 cross-frame window=${PROBE_CROSS_FRAME_WINDOW}"
summary_cmd=(
  python
  "$REPO_DIR/scripts/summarize_vsibench_probe_eval.py"
  --kind cross_frame
  --learned-dir "$LEARNED_RESULT_DIR"
  --gate0-dir "$GATE0_RESULT_DIR"
  --geometry-shuffle-dir "$GSHUFFLE_RESULT_DIR"
  --cross-frame-w1-dir "$XFRAME_W1_RESULT_DIR"
  --cross-frame-w2-dir "$XFRAME_W2_RESULT_DIR"
  --baseline-dir "$BASELINE_RESULT_DIR"
  --output-dir "$OUTPUT_PATH"
  --primary-label "$primary_label"
  --csv-name cross_frame_eval_summary.csv
  --md-name cross_frame_eval_summary.md
)

printf '[SUMMARY CMD] %q ' "${summary_cmd[@]}"
echo
"${summary_cmd[@]}"

echo "[DONE] Cross-frame eval artifacts are under: $OUTPUT_PATH"
