#!/usr/bin/env bash
#SBATCH --job-name=cut3r_token_only_vsi
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --qos=normal
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err

set -euo pipefail
cd "${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
mkdir -p logs/cut3r_token_only

# Promotion gate: the smoke run and exact-frame parity diagnostic must have passed.
: "${SMOKE_GATE_RECORD:?Provide the successful smoke-gate record before submitting this official wrapper.}"
: "${PARITY_SIDECAR:?Set PARITY_SIDECAR.}"
: "${PARITY_RECOMPUTED:?Set PARITY_RECOMPUTED produced from the exact selected 32 frames.}"
python scripts/diagnose_cut3r_token_sidecar_parity.py --sidecar "$PARITY_SIDECAR" --recomputed "$PARITY_RECOMPUTED"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
set +u; eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV_NAME"; set -u
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
OUTPUT_DIR="${OUTPUT_DIR:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_token_only_vsi_${SLURM_JOB_ID}}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

srun --kill-on-bad-exit=1 --wait=30 torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" \
  llava/train/train_mem.py \
  --model_name_or_path "$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2" --vision_tower "$MODEL_ROOT/siglip-so400m-patch14-384" \
  --visual_token_source cut3r_only --cut3r_token_sidecar_key patch_tokens --cut3r_token_feature_dim 768 \
  --cut3r_token_projector_layernorm True --tune_cut3r_token_projector True --cut3r_token_debug_telemetry True \
  --spatial_tower cut3r --spatial_tower_preextracted_only True --spatial_features_root "$DATA_ROOT" --spatial_features_subdir spatial_features \
  --data_path "${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data.yaml}" --video_folder "$DATA_ROOT" --frames_upbound 32 --force_sample True \
  --seed 42 --data_seed 42 --strict_video_loading True --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
  --bf16 True --tf32 True --gradient_checkpointing True --lazy_preprocess True --mm_patch_merge_type spatial_unpad --mm_newline_position grid --mm_spatial_pool_stride 2 \
  --deepspeed scripts/zero2.json --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --save_strategy steps --save_steps 100 --save_total_limit 2 --logging_steps 5 --learning_rate 2e-5 --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --output_dir "$OUTPUT_DIR" --report_to wandb
