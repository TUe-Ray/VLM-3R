#!/bin/bash
#SBATCH --job-name=Eval_feat_align_frozen_p_geo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=09:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

NOTE="Frozen-P_geo eval wrapper: validates/stages the frozen P_geo dependency, then runs VSI-Bench eval."
echo "-------- Note --------"
echo "  note: $NOTE"

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

# Default to the current frozen-P_geo job. Override TRAIN_RUN_NAME for future runs.
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-feat_align_frozen_p_geo_100p_40855013}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$TRAIN_SAVE_ROOT/$TRAIN_RUN_NAME}"
RUN_NAME="${RUN_NAME:-$TRAIN_RUN_NAME}"

# This is the pretrained P_geo used by train_feat_align_P_geo_freeze.sh.
P_GEO_SOURCE_RUN_DIR="${P_GEO_SOURCE_RUN_DIR:-$TRAIN_SAVE_ROOT/feat_align_cut3r_100p_40723512}"
SPATIAL_RANK_HEAD_PATH="${SPATIAL_RANK_HEAD_PATH:-$P_GEO_SOURCE_RUN_DIR/p_geo.bin}"
P_GEO_EXTRACT_SOURCE="${P_GEO_EXTRACT_SOURCE:-auto}"
P_GEO_TARGET_PATH="${P_GEO_TARGET_PATH:-$PRETRAINED_LOCAL/p_geo.bin}"
STRICT_FROZEN_P_GEO="${STRICT_FROZEN_P_GEO:-True}"

cd "$REPO_DIR"
mkdir -p logs/eval

echo "==== Frozen-P_geo eval config ===="
echo "REPO_DIR=$REPO_DIR"
echo "TRAIN_RUN_NAME=$TRAIN_RUN_NAME"
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUN_NAME=$RUN_NAME"
echo "P_GEO_SOURCE_RUN_DIR=$P_GEO_SOURCE_RUN_DIR"
echo "SPATIAL_RANK_HEAD_PATH=$SPATIAL_RANK_HEAD_PATH"
echo "P_GEO_TARGET_PATH=$P_GEO_TARGET_PATH"
echo "STRICT_FROZEN_P_GEO=$STRICT_FROZEN_P_GEO"
echo "=================================="

if [[ ! -d "$PRETRAINED_LOCAL" ]]; then
  echo "[ERROR] Trained checkpoint directory not found: $PRETRAINED_LOCAL"
  echo "[ERROR] If training is still pending/running, submit this script with --dependency=afterok:<train_job_id>."
  exit 1
fi

if [[ ! -f "$PRETRAINED_LOCAL/config.json" ]]; then
  echo "[ERROR] Missing checkpoint config: $PRETRAINED_LOCAL/config.json"
  exit 1
fi

if [[ "$STRICT_FROZEN_P_GEO" == "True" ]]; then
  python - "$PRETRAINED_LOCAL/config.json" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

if cfg.get("freeze_spatial_rank_head") is not True:
    raise SystemExit(
        "[ERROR] This eval script is only for frozen-P_geo checkpoints, "
        f"but freeze_spatial_rank_head={cfg.get('freeze_spatial_rank_head')!r} in {cfg_path}"
    )

print("[P_GEO] config confirms freeze_spatial_rank_head=True")
PY
fi

if [[ ! -f "$SPATIAL_RANK_HEAD_PATH" ]]; then
  echo "[P_GEO] p_geo.bin not found at $SPATIAL_RANK_HEAD_PATH"
  echo "[P_GEO] Attempting extraction from $P_GEO_SOURCE_RUN_DIR"

  if [[ ! -d "$P_GEO_SOURCE_RUN_DIR" ]]; then
    echo "[ERROR] P_GEO_SOURCE_RUN_DIR not found: $P_GEO_SOURCE_RUN_DIR"
    exit 1
  fi

  if [[ "$P_GEO_EXTRACT_SOURCE" == "auto" ]]; then
    P_GEO_EXTRACT_SOURCE=""
    if [[ -f "$P_GEO_SOURCE_RUN_DIR/non_lora_trainables.bin" ]]; then
      P_GEO_EXTRACT_SOURCE="$P_GEO_SOURCE_RUN_DIR/non_lora_trainables.bin"
    else
      mapfile -t P_GEO_CANDIDATES < <(find "$P_GEO_SOURCE_RUN_DIR" -maxdepth 2 -type f -name "non_lora_trainables.bin" | sort -V)
      if (( ${#P_GEO_CANDIDATES[@]} > 0 )); then
        P_GEO_LAST_INDEX=$((${#P_GEO_CANDIDATES[@]} - 1))
        P_GEO_EXTRACT_SOURCE="${P_GEO_CANDIDATES[$P_GEO_LAST_INDEX]}"
      fi
    fi
  fi

  if [[ -z "$P_GEO_EXTRACT_SOURCE" || ! -f "$P_GEO_EXTRACT_SOURCE" ]]; then
    echo "[ERROR] Could not find non_lora_trainables.bin to extract frozen P_geo."
    echo "[ERROR] Set P_GEO_EXTRACT_SOURCE=/path/to/non_lora_trainables.bin or SPATIAL_RANK_HEAD_PATH=/path/to/p_geo.bin."
    exit 1
  fi

  python scripts/extraction/extract_spatial_rank_head.py \
    --checkpoint "$P_GEO_EXTRACT_SOURCE" \
    --output "$SPATIAL_RANK_HEAD_PATH"
fi

if [[ ! -f "$SPATIAL_RANK_HEAD_PATH" ]]; then
  echo "[ERROR] Frozen P_geo file still missing: $SPATIAL_RANK_HEAD_PATH"
  exit 1
fi

cp -f "$SPATIAL_RANK_HEAD_PATH" "$P_GEO_TARGET_PATH"
echo "[P_GEO] staged frozen P_geo into checkpoint: $P_GEO_TARGET_PATH"

PRETRAINED_LOCAL="$PRETRAINED_LOCAL" \
RUN_NAME="$RUN_NAME" \
bash "$REPO_DIR/eval_feat_align_cut3r.sh"
