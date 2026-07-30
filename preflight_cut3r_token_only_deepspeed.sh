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
MANIFEST="${CUT3R_TOKEN_SIDECAR_MANIFEST:-$REPO_DIR/diagnostics/cut3r_token_only/sidecar_manifest_verified.json}"
export CUT3R_TOKEN_SIDECAR_MANIFEST="$MANIFEST"
export CUT3R_TOKEN_MANIFEST_POLICY="${CUT3R_TOKEN_MANIFEST_POLICY:-warn}"
if [[ -n "$MANIFEST" && ! -f "$MANIFEST" ]]; then echo "[CUT3R_TOKEN_ONLY][MANIFEST][WARN] missing manifest; legacy fallback enabled"; fi
export CUT3R_TOKEN_SIDECAR_MANIFEST="$MANIFEST"
PREFLIGHT_DATA_PATH="${PREFLIGHT_DATA_PATH:-$REPO_DIR/diagnostics/cut3r_token_only/deepspeed_preflight_data/verified_preflight.yaml}"
[[ -f "$PREFLIGHT_DATA_PATH" ]] || { echo "[ERROR] Missing deterministic preflight dataset: $PREFLIGHT_DATA_PATH"; exit 1; }
export DATA_PATH_YAML="$PREFLIGHT_DATA_PATH"
export CUT3R_TOKEN_ONLY_PREFLIGHT=True
# Do not install the experimental live-parameter scan under ZeRO-2; the
# validator compares bounded initial samples with checkpoint-2 instead.
export CUT3R_TOKEN_SMOKE_TELEMETRY=False
export CUT3R_TOKEN_CHECKPOINT_DELTA_VALIDATION=True
export SMOKE_MAX_STEPS=2
export SMOKE_SAVE_STEPS=2
echo "[PREFLIGHT] dataset=$DATA_PATH_YAML manifest=$CUT3R_TOKEN_SIDECAR_MANIFEST model=/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2 sidecar_root=/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r world_size=4 gradient_accumulation=2 max_steps=2"
export SMOKE_TRAIN_DATA_MAX_SAMPLES=8
export TARGET_GLOBAL_BATCH_SIZE=8
echo "[PREFLIGHT] output_dir=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/${SLURM_JOB_NAME:-SMOKE_CUT3RTokenOnly_DeepSpeedPreflight}_${SLURM_JOB_ID:-<allocated-job-id>}"
export NOTE="Real two-step DeepSpeed ZeRO-2 engineering preflight; global batch 8 is not the official scientific configuration."
exec bash "$REPO_DIR/smoke_train_cut3r_token_only_vsi.sh"
