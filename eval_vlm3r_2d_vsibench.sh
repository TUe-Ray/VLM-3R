#!/usr/bin/env bash
# Offline VSI-Bench evaluation for the pure canonical 2D LLaVA checkpoint.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
SUBMODULE_DIR="${SUBMODULE_DIR:-$REPO_DIR/thinking-in-space}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vsibench}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
OUTPUT_PATH="${OUTPUT_PATH:?Set OUTPUT_PATH for VSI-Bench results.}"
RUN_NAME="${RUN_NAME:-vlm3r_2d_vsibench}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-0}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/$RUN_NAME-${SLURM_JOB_ID:-manual}}"
PRETRAINED_RUNTIME="$RUNTIME_ROOT/pretrained_2d_local_siglip"

if [[ "$MAX_FRAMES_NUM" != "32" ]]; then
  echo "[ERROR] The controlled 2D evaluation is fixed to the common 32-frame setting."
  exit 2
fi
for path in "$SUBMODULE_DIR" "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL" "$TASK_DIR"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing required path: $path"; exit 2; }
done
[[ -f "$MODEL_BASE_LOCAL/config.json" ]] || { echo "[ERROR] Missing canonical config"; exit 2; }
[[ -f "$SIGLIP_LOCAL/config.json" ]] || { echo "[ERROR] Missing local SigLIP config"; exit 2; }

mkdir -p "$PRETRAINED_RUNTIME"
for source_path in "$MODEL_BASE_LOCAL"/*; do
  base_name="$(basename "$source_path")"
  [[ "$base_name" == "config.json" ]] && continue
  ln -sfn "$source_path" "$PRETRAINED_RUNTIME/$base_name"
done
cp "$MODEL_BASE_LOCAL/config.json" "$PRETRAINED_RUNTIME/config.json"

set +u
source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
set -u
python - "$PRETRAINED_RUNTIME/config.json" "$SIGLIP_LOCAL" <<'PY'
import json
import sys

config_path, siglip_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
config["mm_vision_tower"] = siglip_path
if "vision_tower" in config:
    config["vision_tower"] = siglip_path
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

export HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER=accelerate
cd "$SUBMODULE_DIR"

MODEL_ARGS="pretrained=$PRETRAINED_RUNTIME,model_name=llava-qwen,conv_template=qwen_1_5,max_frames_num=32,attn_implementation=$MODEL_ATTN_IMPLEMENTATION,overwrite=False"
cmd=(accelerate launch --num_processes "$NUM_PROCESSES" -m lmms_eval --model vlm_3r --model_args "$MODEL_ARGS" --tasks "$TASK_DIR" --batch_size "$BATCH_SIZE" --log_samples --log_samples_suffix "$RUN_NAME" --output_path "$OUTPUT_PATH")
if [[ "$LIMIT" != "0" ]]; then
  cmd+=(--limit "$LIMIT")
fi
printf '[CMD] %q ' "${cmd[@]}"
echo
exec "${cmd[@]}"
