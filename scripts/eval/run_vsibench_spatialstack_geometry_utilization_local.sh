#!/usr/bin/env bash
# Paired task-level SpatialStack geometry utilization on native VSiBench.
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  preflight|smoke|full|analyze) ;;
  *) echo "Usage: $0 preflight | smoke | full | analyze" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
CONDA_ENV="${CONDA_ENV:-vsibench}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CPU_MERGE_GPU_BUDGETS="${CPU_MERGE_GPU_BUDGETS:-6GiB,10GiB}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
VSI_VIDEO_ROOT="${VSI_VIDEO_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/vsibench_test}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features}"
RESULT_ROOT="${RESULT_ROOT:-/home/shaoruei/probe_outputs/vsibench_spatialstack_geometry_utilization_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/vsibench_spatialstack_geometry_utilization_v1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-2}"

checkpoint_for() {
  case "$1" in
    SS012_old) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703" ;;
    SS012_new) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_45297963" ;;
    SS123) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n" ;;
    SS036) echo "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970" ;;
    *) echo "Unknown model label: $1" >&2; return 2 ;;
  esac
}

schedule_for() {
  case "$1" in
    SS012_old|SS012_new) echo "0,1,2" ;;
    SS123) echo "1,2,3" ;;
    SS036) echo "0,3,6" ;;
    *) return 2 ;;
  esac
}

preflight() {
  local model checkpoint
  for model in SS123 SS012_new SS036 SS012_old; do
    checkpoint="$(checkpoint_for "$model")"
    [[ -f "$checkpoint/config.json" ]] || { echo "Missing config: $checkpoint/config.json" >&2; exit 1; }
    [[ -f "$checkpoint/adapter_model.bin" || -f "$checkpoint/adapter_model.safetensors" ]] || {
      echo "Missing adapter weights: $checkpoint" >&2; exit 1;
    }
    local config_schedule
    config_schedule="$(conda run --no-capture-output -n "$CONDA_ENV" python -c \
      'import json,sys; c=json.load(open(sys.argv[1])); assert c.get("use_cut3r_spatialstack") is True; assert (c.get("cut3r_spatialstack_fusion_type") or "add").lower() == "add"; print(c["cut3r_spatialstack_llm_layers"])' \
      "$checkpoint/config.json")"
    [[ "$config_schedule" == "$(schedule_for "$model")" ]] || {
      echo "$model has unexpected injection schedule: $config_schedule" >&2; exit 1;
    }
  done
  PRETRAINED_LOCAL="$(checkpoint_for SS012_new)" MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL" SIGLIP_LOCAL="$SIGLIP_LOCAL" \
    VSI_VIDEO_ROOT="$VSI_VIDEO_ROOT" SPATIAL_FEATURES_ROOT="$SPATIAL_FEATURES_ROOT" \
    SPATIAL_FEATURES_SUBDIR="$SPATIAL_FEATURES_SUBDIR" CONDA_ENV="$CONDA_ENV" MODE=preflight \
    RUNTIME_DIR="$RESULT_ROOT/preflight/runtime" OUTPUT_PATH="$LOG_ROOT/preflight" \
    "$REPO_ROOT/scripts/eval/run_vsibench_local_mp4.sh"
  echo "[PREFLIGHT] additive SS checkpoints and layered VSiBench sidecars are ready."
}

run_condition() {
  local model="$1" condition="$2" limit="$3"
  local checkpoint run_name output_path runtime_dir
  checkpoint="$(checkpoint_for "$model")"
  run_name="${model}_${condition}"
  if [[ "$limit" != "0" ]]; then run_name+="_smoke${limit}"; fi
  output_path="$RESULT_ROOT/runs/$run_name"
  runtime_dir="$RESULT_ROOT/runtime/$run_name"
  mkdir -p "$LOG_ROOT" "$RESULT_ROOT/runs" "$RESULT_ROOT/runtime"
  echo "[RUN] model=$model condition=$condition CUDA_VISIBLE_DEVICES=$CUDA_DEVICES output=$output_path"
  PRETRAINED_LOCAL="$checkpoint" MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL" SIGLIP_LOCAL="$SIGLIP_LOCAL" \
    VSI_VIDEO_ROOT="$VSI_VIDEO_ROOT" SPATIAL_FEATURES_ROOT="$SPATIAL_FEATURES_ROOT" \
    SPATIAL_FEATURES_SUBDIR="$SPATIAL_FEATURES_SUBDIR" CUDA_DEVICES="$CUDA_DEVICES" \
    CPU_MERGE_GPU_BUDGETS="$CPU_MERGE_GPU_BUDGETS" CONDA_ENV="$CONDA_ENV" RUN_NAME="$run_name" \
    RUNTIME_DIR="$runtime_dir" OUTPUT_PATH="$output_path" LIMIT="$limit" \
    SPATIALSTACK_PERTURBATION_MODE="$condition" \
    "$REPO_ROOT/scripts/eval/run_vsibench_local_mp4.sh" 2>&1 | tee "$LOG_ROOT/${run_name}.log"
}

analyze_runs() {
  conda run --no-capture-output -n "$CONDA_ENV" python \
    "$REPO_ROOT/scripts/eval/analyze_vsibench_geometry_utilization.py" \
    --run-root "$RESULT_ROOT/runs" --output-dir "$RESULT_ROOT/analysis" \
    --models SS123 SS012_new SS036 SS012_old
}

smoke() {
  # `none` is the established unperturbed generation path.  `normal` uses the
  # same explicit perturbation interface with no disabled residual and must
  # exactly reproduce it before normal/off task scores are interpreted.
  run_condition SS012_new none "$SMOKE_LIMIT"
  run_condition SS012_new normal "$SMOKE_LIMIT"
  run_condition SS012_new geometry_off_all "$SMOKE_LIMIT"
  conda run --no-capture-output -n "$CONDA_ENV" python \
    "$REPO_ROOT/scripts/eval/analyze_vsibench_geometry_utilization.py" \
    --run-root "$RESULT_ROOT/runs" --output-dir "$RESULT_ROOT/smoke/analysis" \
    --models SS012_new --baseline-condition none --require-normal-baseline-match
}

full() {
  local model
  for model in SS123 SS012_new SS036 SS012_old; do
    run_condition "$model" normal 0
    run_condition "$model" geometry_off_all 0
  done
  analyze_runs
}

case "$MODE" in
  preflight) preflight ;;
  smoke) preflight; smoke ;;
  full) preflight; full ;;
  analyze) analyze_runs ;;
esac
