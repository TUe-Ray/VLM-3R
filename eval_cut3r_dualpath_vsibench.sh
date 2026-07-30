#!/bin/bash
# Exact-checkpoint VSI-Bench entry point for the opt-in CUT3R dual path.
set -euo pipefail

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
SUBMODULE_DIR="${SUBMODULE_DIR:-$REPO_DIR/thinking-in-space}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
VSI_ROOT="${VSI_ROOT:-$FAST_ROOT/vsibench}"
VSI_MEDIA_ROOT="${VSI_MEDIA_ROOT:-$FAST_ROOT/hf_cache/vsibench}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:?Set PRETRAINED_LOCAL to the exact dual-path checkpoint.}"
OUTPUT_PATH="${OUTPUT_PATH:?Set OUTPUT_PATH for VSI-Bench results.}"
TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-1}"
RUN_NAME="${RUN_NAME:-smoke_cut3r_dualpath_vsibench}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
CUT3R_TOKEN_FEATURES_ROOT="${CUT3R_TOKEN_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:$CUT3R_TOKEN_FEATURES_ROOT:spatial_features}"

if [[ "$MAX_FRAMES_NUM" != "32" ]]; then
  echo "[ERROR] Dual-path evaluation is fixed to the common 32-frame design."
  exit 2
fi
for path in "$REPO_DIR" "$SUBMODULE_DIR" "$PRETRAINED_LOCAL" "$MODEL_BASE_LOCAL" "$TASK_DIR"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing required path: $path"; exit 2; }
done
for file in "$PRETRAINED_LOCAL/config.json" "$PRETRAINED_LOCAL/adapter_config.json" "$PRETRAINED_LOCAL/non_lora_trainables.bin"; do
  [[ -f "$file" ]] || { echo "[ERROR] Missing required checkpoint file: $file"; exit 2; }
done
if [[ ! -f "$PRETRAINED_LOCAL/adapter_model.bin" && ! -f "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
  echo "[ERROR] Missing LoRA adapter weights under $PRETRAINED_LOCAL"
  exit 2
fi

"$CONDA_BASE/envs/$CONDA_ENV/bin/python" - "$PRETRAINED_LOCAL/config.json" "$PRETRAINED_LOCAL/adapter_config.json" <<'PY'
import json
import sys

config_path, adapter_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
with open(adapter_path, encoding="utf-8") as handle:
    adapter = json.load(handle)

if not bool(config.get("enable_dual_path_spatial", False)):
    raise SystemExit("[ERROR] Checkpoint does not enable dual-path spatial execution.")
def layers(key):
    value = config.get(key, [])
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    return [int(layer) for layer in value]

if layers("cut3r_spatialstack_layers") != [6, 9, 12] or layers("spatial_source_layers") != [0, 1, 2]:
    raise SystemExit(
        "[ERROR] Incorrect donor mapping: expected CUT3R [6, 9, 12] -> spatial blocks [0, 1, 2], "
        f"got CUT3R {config.get('cut3r_spatialstack_layers')!r} -> blocks {config.get('spatial_source_layers')!r}."
    )
if "journey9ni" in json.dumps((config, adapter), sort_keys=True).lower():
    raise SystemExit("[ERROR] Journey9ni adapter reference is forbidden.")
print("[CHECKPOINT] dual path enabled; CUT3R mapping 6/9/12 -> blocks 0/1/2; no Journey9ni reference")
PY

source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
export TOKENIZERS_PARALLELISM=false
cd "$SUBMODULE_DIR"

MODEL_ARGS="pretrained=$PRETRAINED_LOCAL,model_base=$MODEL_BASE_LOCAL,model_name=llava-qwen-lora,conv_template=qwen_1_5,max_frames_num=32,attn_implementation=$MODEL_ATTN_IMPLEMENTATION,overwrite=False,spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR"
cmd=(accelerate launch --num_processes "$NUM_PROCESSES" -m lmms_eval --model vlm_3r --model_args "$MODEL_ARGS" --tasks "$TASK_DIR" --batch_size "$BATCH_SIZE" --log_samples --log_samples_suffix "$RUN_NAME" --output_path "$OUTPUT_PATH")
if [[ "$LIMIT" != "0" ]]; then
  cmd+=(--limit "$LIMIT")
fi
printf '[CMD] %q ' "${cmd[@]}"
echo
"${cmd[@]}"
