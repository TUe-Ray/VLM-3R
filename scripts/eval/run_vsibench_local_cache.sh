#!/usr/bin/env bash
# Local single-process VSiBench evaluation using migrated 32-frame RGB caches.
# The model is sharded across both TITAN Vs; this is intentionally not DDP.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/shaoruei/SpatialFocus}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_FRAMES_ROOT="${FORWARD_FRAMES_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"
ARROW_DATASET="${ARROW_DATASET:-$REPO_DIR/thinking-in-space/vsibench/parquet/default-f7e40315024f07b0/0.0.0/ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1/parquet-test.arrow}"
TASK_DIR="${TASK_DIR:-$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_local_cache}"
RUN_NAME="${RUN_NAME:-vsibench_reproduction2_local_$(date +%Y%m%d_%H%M%S)}"
RUNTIME_DIR="${RUNTIME_DIR:-/home/shaoruei/probe_outputs/vsibench_runtime/$RUN_NAME}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/logs/vsibench_local/$RUN_NAME}"
LIMIT="${LIMIT:-0}"
MODE="${MODE:-run}" # Set MODE=preflight for a CPU-only input/runtime check.

for required in "$REPO_DIR" "$PRETRAINED_LOCAL" "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL" "$FORWARD_FRAMES_ROOT" "$SPATIAL_FEATURES_ROOT" "$TASK_DIR"; do
    [[ -e "$required" ]] || { echo "[ERROR] Missing required path: $required" >&2; exit 1; }
done
for required in "$PRETRAINED_LOCAL/config.json" "$PRETRAINED_LOCAL/adapter_config.json" "$PRETRAINED_LOCAL/non_lora_trainables.bin" "$MODEL_BASE_LOCAL/config.json" "$SIGLIP_LOCAL/config.json" "$ARROW_DATASET" "$TASK_DIR/vsibench_local_cache.yaml"; do
    [[ -f "$required" ]] || { echo "[ERROR] Missing required file: $required" >&2; exit 1; }
done
if [[ ! -f "$PRETRAINED_LOCAL/adapter_model.bin" && ! -f "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
    echo "[ERROR] Missing LoRA adapter weights under $PRETRAINED_LOCAL" >&2
    exit 1
fi

mkdir -p "$RUNTIME_DIR" "$OUTPUT_PATH"
for item in "$PRETRAINED_LOCAL"/*; do
    name="$(basename "$item")"
    [[ "$name" == "config.json" ]] && continue
    ln -sfn "$item" "$RUNTIME_DIR/$name"
done

# Keep the checkpoint experiment settings, but replace machine-specific paths
# and force its CUT3R cross-attention to consume the migrated token sidecars.
python "$REPO_DIR/scripts/eval/prepare_vsibench_local_runtime.py" \
    --source-config "$PRETRAINED_LOCAL/config.json" \
    --output-config "$RUNTIME_DIR/config.json" \
    --siglip-path "$SIGLIP_LOCAL"

export VLM3R_CODE_ROOT="$REPO_DIR"
export PYTHONPATH="$REPO_DIR/thinking-in-space:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VSI_FORWARD_FRAMES_ROOT="$FORWARD_FRAMES_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER=accelerate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "==== Local VSiBench configuration ===="
echo "CUDA_DEVICES=$CUDA_DEVICES (one sharded process)"
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUNTIME_DIR=$RUNTIME_DIR"
echo "FORWARD_FRAMES_ROOT=$FORWARD_FRAMES_ROOT"
echo "SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "ARROW_DATASET=$ARROW_DATASET"
echo "TASK_DIR=$TASK_DIR"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "LIMIT=$LIMIT"
echo "MODE=$MODE"
echo "======================================"

# Validate every referenced visual cache and CUT3R token sidecar before a
# long generation run.  The Arrow split is the locally cached 5,130-prompt
# union of the original benchmark parquet files.
conda run --no-capture-output -n "$CONDA_ENV" python "$REPO_DIR/scripts/eval/preflight_vsibench_local_cache.py" \
    --arrow-dataset "$ARROW_DATASET" \
    --forward-frames-root "$FORWARD_FRAMES_ROOT" \
    --spatial-features-root "$SPATIAL_FEATURES_ROOT" \
    --spatial-features-subdir "$SPATIAL_FEATURES_SUBDIR"

if [[ "$MODE" == "preflight" ]]; then
    echo "[DONE] CPU-only local VSiBench preflight passed."
    exit 0
fi
if [[ "$MODE" != "run" ]]; then
    echo "[ERROR] MODE must be 'preflight' or 'run', got $MODE" >&2
    exit 2
fi

MODEL_ARGS="pretrained=$RUNTIME_DIR,model_base=$MODEL_BASE_LOCAL,model_name=vlm-3r-llava-qwen2-lora,conv_template=qwen_1_5,max_frames_num=32,attn_implementation=sdpa,device_map=auto,overwrite=False,forward_frames_root=$FORWARD_FRAMES_ROOT,spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR,timing_log_interval=25"
cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
    SPATIALFOCUS_CPU_MERGE_LORA=1
    conda run -n "$CONDA_ENV"
    accelerate launch --num_processes 1
    -m lmms_eval
    --model vlm_3r
    --model_args "$MODEL_ARGS"
    --tasks "$TASK_DIR"
    --batch_size 1
    --log_samples
    --log_samples_suffix "$RUN_NAME"
    --output_path "$OUTPUT_PATH"
)
if [[ "$LIMIT" != "0" ]]; then
    cmd+=(--limit "$LIMIT")
fi

printf '[COMMAND] %q ' "${cmd[@]}"
echo
echo "[INFO] Starting full local VSiBench evaluation; logs: $OUTPUT_PATH/launch.log"
set -o pipefail
"${cmd[@]}" 2>&1 | tee "$OUTPUT_PATH/launch.log"
