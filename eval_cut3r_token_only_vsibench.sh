#!/usr/bin/env bash
#SBATCH --job-name=eval_CUT3RTokenOnly_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
SUBMODULE_DIR="${SUBMODULE_DIR:-$REPO_DIR/thinking-in-space}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vsibench}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a CUT3R-token-only smoke checkpoint.}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"
CUT3R_TOKEN_SIDECAR_MANIFEST="${CUT3R_TOKEN_SIDECAR_MANIFEST:-$REPO_DIR/diagnostics/cut3r_token_only/sidecar_manifest_verified.json}"
CUT3R_TOKEN_MANIFEST_POLICY="${CUT3R_TOKEN_MANIFEST_POLICY:-warn}"
OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/cut3r_token_only}"
RUN_NAME="${RUN_NAME:-eval_cut3r_token_only_vsibench}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_PREFLIGHT_ONLY="${EVAL_PREFLIGHT_ONLY:-True}"
LIMIT="${LIMIT:-0}"

cd "$REPO_DIR"
mkdir -p logs/cut3r_token_only "$OUTPUT_PATH"

# All PyTorch imports below must use the evaluation environment, not system Python.
module load cuda/12.1
module load cudnn
module load profile/deeplrn
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
fi
set +u
conda activate "$CONDA_ENV"
set -u
if [[ -v LD_LIBRARY_PATH && -n "$LD_LIBRARY_PATH" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLM3R_CODE_ROOT="$REPO_DIR"
export TOKENIZERS_PARALLELISM=false
if [[ "$EVAL_PREFLIGHT_ONLY" == "True" ]]; then
  # Use the actual task sample. Manifest coverage is advisory: a missing entry
  # must exercise deterministic legacy fallback rather than skip the video.
  export CUT3R_TOKEN_ONLY_EVAL_PREFLIGHT_PATH="$OUTPUT_PATH/cut3r_token_only_preflight.json"
  unset CUT3R_TOKEN_ONLY_EVAL_PREFLIGHT_VIDEO
else
  unset CUT3R_TOKEN_ONLY_EVAL_PREFLIGHT_PATH CUT3R_TOKEN_ONLY_EVAL_PREFLIGHT_VIDEO
fi
for path in "$CHECKPOINT/config.json" "$CHECKPOINT/non_lora_trainables.bin" "$CHECKPOINT/adapter_config.json" "$MODEL_BASE" "$TASK_DIR/vsibench.yaml"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing required path: $path"; exit 1; }
done
if [[ -n "$CUT3R_TOKEN_SIDECAR_MANIFEST" && ! -f "$CUT3R_TOKEN_SIDECAR_MANIFEST" ]]; then
  echo "[CUT3R_TOKEN_ONLY][MANIFEST][WARN] manifest missing; deterministic legacy fallback remains enabled"
fi
python - "$CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path
import torch

checkpoint = Path(sys.argv[1])
config = json.loads((checkpoint / "config.json").read_text())
for key in ("use_cut3r_spatialstack", "use_cut3r_camera_tokens", "use_geometry_aware_projection", "llm_visual_3d_rope_enable", "use_spatial_bridge_tokens", "add_faster_video", "use_bev_supervision", "use_depth_supervision", "use_pointmap_supervision"):
    if bool(config.get(key, False)):
        raise SystemExit(f"CUT3R-only evaluator forbids {key}=True")
if config.get("fusion_block") not in (None, "", "none", "None"):
    raise SystemExit("CUT3R-only evaluator forbids fusion_block")
if config.get("visual_token_source") != "cut3r_only":
    raise SystemExit("checkpoint is not visual_token_source=cut3r_only")
state = torch.load(checkpoint / "non_lora_trainables.bin", map_location="cpu")
projector = [key for key in state if "cut3r_token_projector" in key]
if not projector or any("lora_" in key for key in projector):
    raise SystemExit("projector state missing or incorrectly LoRA-wrapped")
if not ((checkpoint / "adapter_model.bin").is_file() or (checkpoint / "adapter_model.safetensors").is_file()):
    raise SystemExit("checkpoint adapter weights are missing")
print("[CUT3R_TOKEN_ONLY][EVAL_PREFLIGHT] config, adapter, and projector state are present")
PY
export LMMS_EVAL_LAUNCHER=accelerate
cd "$SUBMODULE_DIR"
export PYTHONPATH="$SUBMODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -c "import lmms_eval; print('[CUT3R_TOKEN_ONLY][EVAL] lmms_eval=' + lmms_eval.__file__)"
MODEL_ARGS="pretrained=$CHECKPOINT,model_base=$MODEL_BASE,conv_template=qwen_1_5,max_frames_num=$MAX_FRAMES_NUM,overwrite=False,visual_token_source=cut3r_only,spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR,cut3r_token_sidecar_manifest=$CUT3R_TOKEN_SIDECAR_MANIFEST,cut3r_token_manifest_policy=$CUT3R_TOKEN_MANIFEST_POLICY,video_decode_backend=decord"
cmd=(accelerate launch --num_processes "$NUM_PROCESSES" -m lmms_eval --model vlm_3r --model_args "$MODEL_ARGS" --tasks "$TASK_DIR" --batch_size "$BATCH_SIZE" --log_samples --log_samples_suffix "$RUN_NAME" --output_path "$OUTPUT_PATH")
if [[ "$EVAL_PREFLIGHT_ONLY" == "True" ]]; then
  cmd+=(--limit 1)
elif [[ -n "$LIMIT" && "$LIMIT" != "0" ]]; then
  cmd+=(--limit "$LIMIT")
fi
printf '[CMD] %q ' "${cmd[@]}"; echo
exec "${cmd[@]}"
