#!/usr/bin/env bash
# Local one-process VSiBench evaluation from MP4 media.
# The merged VLM is split across both TITAN Vs; this is intentionally not DDP.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/shaoruei/SpatialFocus}"
CONDA_ENV="${CONDA_ENV:-vsibench}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
# GPU 0 hosts the complete SigLIP tower and therefore needs more free memory
# for its 32-frame activations than GPU 1, which mainly carries decoder layers.
CPU_MERGE_GPU_BUDGETS="${CPU_MERGE_GPU_BUDGETS:-6GiB,10GiB}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
VSI_VIDEO_ROOT="${VSI_VIDEO_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/vsibench_test}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"
ARROW_DATASET="${ARROW_DATASET:-$REPO_DIR/thinking-in-space/vsibench/parquet/default-f7e40315024f07b0/0.0.0/ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1/parquet-test.arrow}"
TASK_DIR="${TASK_DIR:-$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_local_mp4}"
RUN_NAME="${RUN_NAME:-vsibench_reproduction2_local_mp4_$(date +%Y%m%d_%H%M%S)}"
RUNTIME_DIR="${RUNTIME_DIR:-/home/shaoruei/probe_outputs/vsibench_runtime/$RUN_NAME}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/logs/vsibench_local/$RUN_NAME}"
LIMIT="${LIMIT:-0}"
TIMING_LOG_INTERVAL="${TIMING_LOG_INTERVAL:-25}"
SPATIALSTACK_PERTURBATION_MODE="${SPATIALSTACK_PERTURBATION_MODE:-none}"
MODE="${MODE:-run}" # preflight validates all inputs; run launches the evaluator.

for required in "$REPO_DIR" "$PRETRAINED_LOCAL" "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL" "$VSI_VIDEO_ROOT" "$SPATIAL_FEATURES_ROOT" "$TASK_DIR"; do
    [[ -e "$required" ]] || { echo "[ERROR] Missing required path: $required" >&2; exit 1; }
done
for required in "$PRETRAINED_LOCAL/config.json" "$PRETRAINED_LOCAL/adapter_config.json" "$PRETRAINED_LOCAL/non_lora_trainables.bin" "$MODEL_BASE_LOCAL/config.json" "$SIGLIP_LOCAL/config.json" "$ARROW_DATASET" "$TASK_DIR/vsibench_local_mp4.yaml"; do
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

conda run --no-capture-output -n "$CONDA_ENV" python "$REPO_DIR/scripts/eval/prepare_vsibench_local_runtime.py" \
    --source-config "$PRETRAINED_LOCAL/config.json" \
    --output-config "$RUNTIME_DIR/config.json" \
    --siglip-path "$SIGLIP_LOCAL"

export VLM3R_CODE_ROOT="$REPO_DIR"
export PYTHONPATH="$REPO_DIR/thinking-in-space:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VSI_VIDEO_ROOT
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER=accelerate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES (one sharded process)"
echo "[INFO] SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS=$CPU_MERGE_GPU_BUDGETS"
echo "[INFO] VSI_VIDEO_ROOT=$VSI_VIDEO_ROOT"
echo "[INFO] SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "[INFO] SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "[INFO] TIMING_LOG_INTERVAL=$TIMING_LOG_INTERVAL"
echo "[INFO] SPATIALSTACK_PERTURBATION_MODE=$SPATIALSTACK_PERTURBATION_MODE"
echo "[INFO] OUTPUT_PATH=$OUTPUT_PATH"

conda run --no-capture-output -n "$CONDA_ENV" python "$REPO_DIR/scripts/eval/preflight_vsibench_local_mp4.py" \
    --arrow-dataset "$ARROW_DATASET" \
    --video-root "$VSI_VIDEO_ROOT" \
    --spatial-features-root "$SPATIAL_FEATURES_ROOT" \
    --spatial-features-subdir "$SPATIAL_FEATURES_SUBDIR"

if [[ "$MODE" == "preflight" ]]; then
    echo "[DONE] VSiBench MP4 and sidecar preflight passed."
    exit 0
fi
if [[ "$MODE" != "run" ]]; then
    echo "[ERROR] MODE must be 'preflight' or 'run', got $MODE" >&2
    exit 2
fi

MODEL_ARGS="pretrained=$RUNTIME_DIR,model_base=$MODEL_BASE_LOCAL,model_name=vlm-3r-llava-qwen2-lora,conv_template=qwen_1_5,max_frames_num=32,attn_implementation=sdpa,device_map=auto,overwrite=False,video_decode_backend=pyav,spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR,timing_log_interval=$TIMING_LOG_INTERVAL,spatialstack_perturbation_mode=$SPATIALSTACK_PERTURBATION_MODE"
cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
    SPATIALFOCUS_CPU_MERGE_LORA=1
    "SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS=$CPU_MERGE_GPU_BUDGETS"
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
echo "[INFO] Starting VSiBench from MP4; logs: $OUTPUT_PATH/launch.log"
set -o pipefail
"${cmd[@]}" 2>&1 | tee "$OUTPUT_PATH/launch.log"
