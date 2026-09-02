#!/usr/bin/env bash
# Read-only architecture proxy comparison for the fixed post-SFT 3D roster.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
GPU="${GPU:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/shaoruei/probe_outputs/post_sft_3d_zero_cost_proxies_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/$MODE}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/post_sft_3d_zero_cost_proxies_v1}"
SMOKE_CANDIDATE="${SMOKE_CANDIDATE:-baseline}"
ALL_CANDIDATES="${ALL_CANDIDATES:-ss_depth,spatial_stack,ss_cross_attn,baseline_depth,baseline,extra_object_token,selective_fusion,zero_spatial}"
# Preserve the known valid whole-vision placement on GPU 0 while reserving
# GPU 1 activation room by CPU-offloading frozen decoder tail blocks.  The
# builder enforces that the nested SigLIP tower itself is never split.
CPU_MERGE_GPU_BUDGETS="${CPU_MERGE_GPU_BUDGETS:-${SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS:-6GiB,8GiB}}"
ATTEMPT_WHOLE_MODEL="${ATTEMPT_WHOLE_MODEL:-1}"
PRIOR_RESULTS="${PRIOR_RESULTS:-}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "[RUN] mode=$MODE CUDA_VISIBLE_DEVICES=$CUDA_DEVICES physical-readiness-gpu=$GPU output=$OUTPUT_ROOT"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

if [[ "$MODE" == "smoke" ]]; then
  CANDIDATES="$SMOKE_CANDIDATE"
  if [[ "$ATTEMPT_WHOLE_MODEL" == "1" ]]; then
    WHOLE_MODEL_ARGS=()
  elif [[ "$ATTEMPT_WHOLE_MODEL" == "0" ]]; then
    WHOLE_MODEL_ARGS=(--no-attempt-whole-model)
  else
    echo "ATTEMPT_WHOLE_MODEL must be 0 or 1, got: $ATTEMPT_WHOLE_MODEL" >&2
    exit 2
  fi
else
  CANDIDATES="$ALL_CANDIDATES"
  # The full command is launched only after a successful smoke; preserve its
  # whole-model OOM finding rather than repeating unsafe allocations.
  WHOLE_MODEL_ARGS=(--no-attempt-whole-model)
fi

PRIOR_RESULTS_ARGS=()
if [[ -n "$PRIOR_RESULTS" ]]; then
  PRIOR_RESULTS_ARGS=(--prior-results "$PRIOR_RESULTS")
fi

env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  SPATIALFOCUS_CPU_MERGE_LORA=1 \
  SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS="$CPU_MERGE_GPU_BUDGETS" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/post_sft_3d_proxy_mpl}" \
  TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/post_sft_3d_proxy_triton}" \
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/evaluate_post_sft_3d_proxy_suite.py" \
  --mode "$MODE" --candidates "$CANDIDATES" --calibration-batches 1 \
  --output-root "$OUTPUT_ROOT" "${WHOLE_MODEL_ARGS[@]}" "${PRIOR_RESULTS_ARGS[@]}" \
  2>&1 | tee "$LOG_ROOT/${MODE}.log"
