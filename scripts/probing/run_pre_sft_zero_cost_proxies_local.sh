#!/usr/bin/env bash
# Local, read-only proxy comparison for the fixed C1 pre-SFT candidate study.
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
OUTPUT_BASE="${OUTPUT_BASE:-/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/$MODE}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/pre_sft_zero_cost_proxies_v1}"
SMOKE_CANDIDATE="${SMOKE_CANDIDATE:-c1_ss_add_012}"
ALL_CANDIDATES="${ALL_CANDIDATES:-c1_ss_add_012,c1_ss_add_036,c1_ss_add_123,c1_ss_cross_attn_012,c1_vlm3r_native}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

echo "[RUN] mode=$MODE CUDA_VISIBLE_DEVICES=$CUDA_DEVICES physical-readiness-gpu=$GPU output=$OUTPUT_ROOT"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

if [[ "$MODE" == "smoke" ]]; then
  CANDIDATES="$SMOKE_CANDIDATE"
  WHOLE_MODEL_ARGS=()
else
  CANDIDATES="$ALL_CANDIDATES"
  # The one-candidate smoke measured a whole-model backward OOM on the local
  # 2x12-GiB setup.  Retain that recorded result and avoid needlessly
  # repeating the unsafe allocation for every full-sweep candidate.
  WHOLE_MODEL_ARGS=(--no-attempt-whole-model)
fi

env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pre_sft_zero_cost_mpl}" \
  TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/pre_sft_zero_cost_triton}" \
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/evaluate_pre_sft_zero_cost_proxies.py" \
  --mode "$MODE" \
  --candidates "$CANDIDATES" \
  --calibration-batches 1 \
  --output-root "$OUTPUT_ROOT" \
  "${WHOLE_MODEL_ARGS[@]}" \
  2>&1 | tee "$LOG_ROOT/${MODE}.log"
