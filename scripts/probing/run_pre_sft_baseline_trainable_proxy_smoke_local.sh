#!/usr/bin/env bash
# One fixed-minibatch pre-SFT Baseline trainable-proxy smoke; no model sweep.
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "preflight" && "$MODE" != "smoke" ]]; then
  echo "Usage: $0 preflight|smoke" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
OUT="${OUT:-/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v2/smoke_baseline_retry6_asymmetric_shards}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/pre_sft_zero_cost_proxies_v2}"
LOG="$LOG_ROOT/baseline_sft_trainable_smoke_retry6_asymmetric_shards.log"

mkdir -p "$LOG_ROOT"

require_inputs() {
  local path
  for path in \
    /mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2/config.json \
    /mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384/config.json \
    /home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json \
    /home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json \
    /home/shaoruei/probe_outputs/post_sft_3d_zero_cost_proxies_v1/complete/results.json; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
}

preflight() {
  require_inputs
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
    --physical-gpu-id "$GPU" \
    --output "$LOG_ROOT/gpu_${GPU}_readiness.json"
}

if [[ "$MODE" == "preflight" ]]; then
  preflight
  exit 0
fi

preflight
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 1
fi
echo "[RUN] pre-SFT Baseline one-minibatch SFT-trainable proxy smoke"
echo "[RUN] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$OUT log=$LOG"
env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
  "$REPO_ROOT/scripts/probing/run_pre_sft_baseline_trainable_proxy_smoke.py" \
  --output-root "$OUT" --device cuda:0 --device-map auto --dtype float16 \
  --pre-sft-gpu-weight-budgets 4GiB,6GiB --pre-sft-cpu-offload-budget 45GiB --rng-seed 42 \
  2>&1 | tee "$LOG"
