#!/usr/bin/env bash
#SBATCH --job-name=cut3r_token_only_vsi_eval
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
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_token_only_vsi_8n_51097720}"
complete_checkpoint() {
  local path="$1"
  [[ -f "$path/config.json" && -f "$path/adapter_config.json" && -f "$path/non_lora_trainables.bin" && -f "$path/trainer_state.json" ]] || return 1
  [[ -f "$path/adapter_model.bin" || -f "$path/adapter_model.safetensors" ]] || return 1
}
if [[ -n "${FINAL_CHECKPOINT:-}" ]]; then
  CHECKPOINT="$FINAL_CHECKPOINT"
  complete_checkpoint "$CHECKPOINT" || { echo "[ERROR] Explicit final checkpoint is incomplete: $CHECKPOINT"; exit 1; }
else
  best_step=-1
  CHECKPOINT=""
  for candidate in "$TRAIN_OUTPUT_DIR"/checkpoint-*; do
    [[ -d "$candidate" ]] || continue
    step="${candidate##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    complete_checkpoint "$candidate" || continue
    if (( step > best_step )); then best_step=$step; CHECKPOINT="$candidate"; fi
  done
  [[ -n "$CHECKPOINT" ]] || { echo "[ERROR] No complete numeric checkpoint-* under $TRAIN_OUTPUT_DIR"; exit 1; }
fi
export CHECKPOINT EVAL_PREFLIGHT_ONLY=False CUT3R_TOKEN_MANIFEST_POLICY=warn
export RUN_NAME="${RUN_NAME:-cut3r_token_only_vsi_eval_51097720}"
export OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/${RUN_NAME}_${SLURM_JOB_ID}}"
echo "[CUT3R_TOKEN_ONLY][EVAL] checkpoint=$CHECKPOINT output=$OUTPUT_PATH policy=$CUT3R_TOKEN_MANIFEST_POLICY"
exec bash "$REPO_DIR/eval_cut3r_token_only_vsibench.sh"
