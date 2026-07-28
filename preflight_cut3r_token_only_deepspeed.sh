#!/usr/bin/env bash
#SBATCH --job-name=SMOKE_CUT3RTokenOnly_DeepSpeedPreflight
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err
#SBATCH --mem=0

# This is an engineering preflight only: it reuses the real Qwen/PEFT/Trainer/
# DeepSpeed command path with a global batch of eight.  It does not change the
# four-node official policy or constitute a scientific smoke result.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
MANIFEST="${CUT3R_TOKEN_SIDECAR_MANIFEST:-$REPO_DIR/diagnostics/cut3r_token_only/sidecar_manifest_verified_full.json}"
[[ -f "$MANIFEST" ]] || { echo "[ERROR] Missing verified full manifest: $MANIFEST"; exit 1; }
export CUT3R_TOKEN_SIDECAR_MANIFEST="$MANIFEST"
export CUT3R_TOKEN_ONLY_PREFLIGHT=True
export SMOKE_MAX_STEPS=2
export SMOKE_SAVE_STEPS=2
export SMOKE_TRAIN_DATA_MAX_SAMPLES=8
export TARGET_GLOBAL_BATCH_SIZE=8
export NOTE="Real two-step DeepSpeed ZeRO-2 engineering preflight; global batch 8 is not the official scientific configuration."
exec bash "$REPO_DIR/smoke_train_cut3r_token_only_vsi.sh"
