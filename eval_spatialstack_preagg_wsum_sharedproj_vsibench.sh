#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SS_PreAgg_WSum_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"

export CUT3R_SPATIALSTACK_PREAGG_ENABLE="${CUT3R_SPATIALSTACK_PREAGG_ENABLE:-True}"
export CUT3R_SPATIALSTACK_PREAGG_LAYERS="${CUT3R_SPATIALSTACK_PREAGG_LAYERS:-6,9,12}"
export CUT3R_SPATIALSTACK_PREAGG_TYPE="${CUT3R_SPATIALSTACK_PREAGG_TYPE:-weighted_sum}"
export CUT3R_SPATIALSTACK_PREAGG_PROJECTOR_SHARING="${CUT3R_SPATIALSTACK_PREAGG_PROJECTOR_SHARING:-shared}"
export CUT3R_SPATIALSTACK_PREAGG_USE_LAYER_GAMMA="${CUT3R_SPATIALSTACK_PREAGG_USE_LAYER_GAMMA:-True}"
export CUT3R_SPATIALSTACK_PREAGG_LAYER_GAMMA_INIT="${CUT3R_SPATIALSTACK_PREAGG_LAYER_GAMMA_INIT:-1.0}"
export CUT3R_SPATIALSTACK_LAYERS="${CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
export CUT3R_SPATIALSTACK_LLM_LAYERS="${CUT3R_SPATIALSTACK_LLM_LAYERS:-1,2,3}"
export CUT3R_SPATIALSTACK_PROJECTOR_TYPE="${CUT3R_SPATIALSTACK_PROJECTOR_TYPE:-token_mlp}"
export CUT3R_SPATIALSTACK_MERGE_SIZE="${CUT3R_SPATIALSTACK_MERGE_SIZE:-2}"
export CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM="${CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM:-4096}"
export CUT3R_SPATIALSTACK_FEATURE_DIM="${CUT3R_SPATIALSTACK_FEATURE_DIM:-768}"

SANITIZED_PREAGG_LAYERS="${CUT3R_SPATIALSTACK_PREAGG_LAYERS//,/_}"
SANITIZED_LLM_LAYERS="${CUT3R_SPATIALSTACK_LLM_LAYERS//,/_}"
RUN_BASENAME="${RUN_BASENAME:-cut3r_spatialstack_preagg_wsum_sharedproj_dec${SANITIZED_PREAGG_LAYERS}_llm${SANITIZED_LLM_LAYERS}}"
TRAIN_RUN_NAME_8N="${TRAIN_RUN_NAME_8N:-${RUN_BASENAME}_8n}"
TRAIN_RUN_NAME_4N="${TRAIN_RUN_NAME_4N:-${RUN_BASENAME}_4n}"

if [[ -z "${PRETRAINED_LOCAL:-}" ]] && ! is_true "${RANDOM_WEIGHT_SMOKE:-False}"; then
  candidate_8n="$TRAIN_MODEL_ROOT/$TRAIN_RUN_NAME_8N"
  candidate_4n="$TRAIN_MODEL_ROOT/$TRAIN_RUN_NAME_4N"
  if [[ -f "$candidate_8n/config.json" ]]; then
    PRETRAINED_LOCAL="$candidate_8n"
  elif [[ -f "$candidate_4n/config.json" ]]; then
    PRETRAINED_LOCAL="$candidate_4n"
  else
    echo "[ERROR] Could not resolve trained checkpoint. Checked:"
    echo "  $candidate_8n"
    echo "  $candidate_4n"
    exit 1
  fi
fi
if is_true "${RANDOM_WEIGHT_SMOKE:-False}"; then
  PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$MODEL_BASE_LOCAL}"
fi

PREAGG_CONFIG_RUNTIME_ROOT="${PREAGG_CONFIG_RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-eval_${RUN_BASENAME}_vsibench}_preagg_source}"
PATCHED_PRETRAINED_LOCAL="$PREAGG_CONFIG_RUNTIME_ROOT/pretrained_preagg_config"
mkdir -p "$PATCHED_PRETRAINED_LOCAL"
for f in "$PRETRAINED_LOCAL"/*; do
  base="$(basename "$f")"
  if [[ "$base" == "config.json" ]]; then
    continue
  fi
  ln -sfn "$f" "$PATCHED_PRETRAINED_LOCAL/$base"
done
cp "$PRETRAINED_LOCAL/config.json" "$PATCHED_PRETRAINED_LOCAL/config.json"
python - "$PATCHED_PRETRAINED_LOCAL/config.json" \
  "$CUT3R_SPATIALSTACK_LAYERS" "$CUT3R_SPATIALSTACK_LLM_LAYERS" "$CUT3R_SPATIALSTACK_FEATURE_DIM" \
  "$CUT3R_SPATIALSTACK_PROJECTOR_TYPE" "$CUT3R_SPATIALSTACK_MERGE_SIZE" "$CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM" \
  "$CUT3R_SPATIALSTACK_PREAGG_ENABLE" "$CUT3R_SPATIALSTACK_PREAGG_LAYERS" "$CUT3R_SPATIALSTACK_PREAGG_TYPE" \
  "$CUT3R_SPATIALSTACK_PREAGG_PROJECTOR_SHARING" "$CUT3R_SPATIALSTACK_PREAGG_USE_LAYER_GAMMA" \
  "$CUT3R_SPATIALSTACK_PREAGG_LAYER_GAMMA_INIT" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
layers = sys.argv[2]
llm_layers = sys.argv[3]
feature_dim = int(sys.argv[4])
projector_type = sys.argv[5]
merge_size = sys.argv[6]
projector_hidden_dim = sys.argv[7]
preagg_enable = sys.argv[8]
preagg_layers = sys.argv[9]
preagg_type = sys.argv[10]
preagg_projector_sharing = sys.argv[11]
preagg_use_layer_gamma = sys.argv[12]
preagg_layer_gamma_init = sys.argv[13]

def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["use_cut3r_spatialstack"] = True
cfg["tune_cut3r_spatialstack"] = False
cfg["cut3r_spatialstack_layers"] = layers
cfg["cut3r_spatialstack_llm_layers"] = llm_layers
cfg["cut3r_spatialstack_feature_dim"] = feature_dim
cfg["cut3r_spatialstack_feature_key"] = "cut3r_dec_layers"
cfg["cut3r_spatialstack_projector_type"] = projector_type
cfg["cut3r_spatialstack_merge_size"] = int(merge_size)
cfg["cut3r_spatialstack_projector_hidden_dim"] = int(projector_hidden_dim)
cfg["cut3r_spatialstack_preagg_enable"] = as_bool(preagg_enable)
cfg["cut3r_spatialstack_preagg_layers"] = preagg_layers
cfg["cut3r_spatialstack_preagg_type"] = preagg_type
cfg["cut3r_spatialstack_preagg_projector_sharing"] = preagg_projector_sharing
cfg["cut3r_spatialstack_preagg_log_weights"] = True
cfg["cut3r_spatialstack_preagg_use_layer_gamma"] = as_bool(preagg_use_layer_gamma)
cfg["cut3r_spatialstack_preagg_layer_gamma_init"] = float(preagg_layer_gamma_init)

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

export PRETRAINED_LOCAL="$PATCHED_PRETRAINED_LOCAL"
export RUN_NAME="${RUN_NAME:-eval_${RUN_BASENAME}_vsibench}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/${RUN_BASENAME}_vsibench}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-eval_${RUN_BASENAME}_vsibench}}"

echo "==== CUT3R SpatialStack preagg VSI-Bench eval ===="
echo "PREAGG_TYPE=$CUT3R_SPATIALSTACK_PREAGG_TYPE"
echo "PREAGG_PROJECTOR_SHARING=$CUT3R_SPATIALSTACK_PREAGG_PROJECTOR_SHARING"
echo "PATCHED_PRETRAINED_LOCAL=${PRETRAINED_LOCAL}"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "CUT3R_SPATIALSTACK_LLM_LAYERS=$CUT3R_SPATIALSTACK_LLM_LAYERS"
echo "===================================================="

bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
