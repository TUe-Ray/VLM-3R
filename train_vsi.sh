#!/bin/bash
#SBATCH --job-name=TrainVSI
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4             # 依你的叢集格式：也可能是 --gpus-per-node=1
#SBATCH --ntasks-per-node=1       # 通常 1 個 task，裡面用 torchrun 起多 GPU processes
#SBATCH --cpus-per-task=32
#SBATCH --time=7:00:00
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=normal # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843
#SBATCH --exclusive


NOTE="Test run for VLM-3R 7B Qwen2 LoRA on VSI-Bench, pretrained by Journey9ni/vlm-3r-llava-qwen2-lora, model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2, conv_template=qwen_1_5, max_frames_num=32"

echo "-------- Note --------"
echo "  note: $NOTE"



mkdir -p logs/train

JOB_TIME_LIMIT=$(squeue -j "$SLURM_JOB_ID" -h -o "%l")


echo "=== SLURM Job Specifications ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
echo "Memory per Node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Output: $SLURM_STDOUT"
echo "Error: $SLURM_STDERR"
echo "Job Time Limit: $JOB_TIME_LIMIT"
set -euo pipefail


# Cluster-specific environment setup (overridable via exported env vars)

module load cuda/12.1
module load cudnn
module load profile/deeplrn

echo "[DEBUG] after modules:"
OUT=$(nvidia-smi -L 2>&1) || {
    echo "[ERROR] nvidia-smi failed on $(hostname)"
    echo "$OUT"
    exit 1
}
if echo "$OUT" | grep -q "Driver/library version mismatch"; then
    echo "[ERROR] NVML mismatch on $(hostname)"
    echo "$OUT"
    exit 1
fi
echo "$OUT"

export PATH="$WORK/miniconda3/bin:$PATH"
# Some conda activation hooks assume optional vars exist and can fail under nounset.
set +u
eval "$(conda shell.bash hook)"
conda activate vlm3r
set -u

# Prefer conda runtime libs to avoid system/libstdc++ mismatch.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export WANDB_MODE="offline"
export NCCL_NVLS_ENABLE=0
export WANDB_DIR="${WANDB_DIR:-$WORK/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WORK/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WORK/wandb_config}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Force local/offline Hugging Face resolution on compute nodes.
export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"



# set training parameters
# IMPORTANT: GPU process count is inferred from Slurm allocations.
# set training parameters
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NUM_GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
    NUM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE"
else
    NUM_GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NNODES=${SLURM_JOB_NUM_NODES:-1}
WORLD_SIZE=$((NNODES * NUM_GPUS_PER_NODE))
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export OMP_NUM_THREADS=2
echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"
echo "[DDP] NNODES=$NNODES"
echo "[DDP] NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"

# ============================================================
# Resume Configuration
# ============================================================
# Set to "none" for fresh training (default).
# Set to "auto" to find the latest checkpoint-* in OUTPUT_DIR.
# Set to an explicit path (e.g. "work_dirs.../checkpoint-500") to
#   resume from that exact checkpoint.
RESUME_CHECKPOINT_PATH="none"
export RESUME_CHECKPOINT_PATH  # read by train.py

# Reproducibility seed (used by HF Trainer for data shuffling & RNG).
SEED=42

# Set up training config
SUFFIX="vlm_3r_vsibench_all_tokens_cross_attn_lora"
MID_RUN_NAME="llava_video_7b_qwen2_${SUFFIX}"
OUTPUT_DIR="work_dirs_auto_eval/$MID_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

LOCAL_MODEL_BASE="/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2"
LOCAL_SIGLIP="/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/siglip-so400m-patch14-384"

if [[ ! -d "$LOCAL_MODEL_BASE" ]]; then
    echo "[ERROR] Local model base not found: $LOCAL_MODEL_BASE"
    exit 1
fi
if [[ ! -d "$LOCAL_SIGLIP" ]]; then
    echo "[ERROR] Local SigLIP not found: $LOCAL_SIGLIP"
    exit 1
fi

echo "[LOCAL MODEL] model_name_or_path=$LOCAL_MODEL_BASE"
echo "[LOCAL MODEL] vision_tower=$LOCAL_SIGLIP"

PER_DEVICE_TRAIN_BATCH_SIZE=1
TARGET_GLOBAL_BATCH_SIZE=128
denom=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( TARGET_GLOBAL_BATCH_SIZE % denom != 0 )); then
    echo "[ERROR] TARGET_GLOBAL_BATCH_SIZE($TARGET_GLOBAL_BATCH_SIZE) not divisible by WORLD_SIZE*PER_DEVICE_TRAIN_BATCH_SIZE($denom)"
    echo "Please adjust TARGET_GLOBAL_BATCH_SIZE or per-device batch size."
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / denom))
echo "[BATCH] PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "[BATCH] TARGET_GLOBAL_BATCH_SIZE=$TARGET_GLOBAL_BATCH_SIZE"
echo "[BATCH] GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

declare -A MODEL_ARGS=(
    [model_name_or_path]="$LOCAL_MODEL_BASE"
    [lora_enable]="True"
    [lora_r]="128"
    [lora_alpha]="256"
    [spatial_tower]="cut3r"
    [spatial_tower_select_feature]="all_tokens"
    [spatial_feature_dim]="768"
    [fusion_block]="cross_attention"
    [tune_spatial_tower]="False"
    [tune_fusion_block]="True"
    [tune_mm_mlp_adapter]="True"
    [version]="qwen_1_5"
    [vision_tower]="$LOCAL_SIGLIP"
    [mm_projector_type]="mlp2x_gelu"
    [mm_vision_select_layer]="-2"
    [mm_use_im_start_end]="False"
    [mm_use_im_patch_token]="False"
    [image_aspect_ratio]="anyres_max_9"
    [image_grid_pinpoints]="(1x1),...,(6x6)"
    [mm_patch_merge_type]="spatial_unpad"
    [bf16]="True"
    [tf32]="True"
    [model_max_length]="32768"
    [gradient_checkpointing]="True"
    [lazy_preprocess]="True"
    [torch_compile]="True"
    [torch_compile_backend]="inductor"
    [frames_upbound]="32"
    [mm_newline_position]="grid"
    [add_time_instruction]="True"
    [force_sample]="True"
    [mm_spatial_pool_stride]="2"
)

declare -A DATA_ARGS=(
    [data_path]="scripts/VLM_3R/vsibench_data.yaml"
    [image_folder]="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r"
    [video_folder]="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r"
    [group_by_modality_length]="True"   #控制 dataloader sampler 是否按模態長度分組（
                                        #通常可減少 padding、讓 batch 更穩定）
)


declare -A TRAINING_ARGS=(
    [deepspeed]="scripts/zero2.json"
    [num_train_epochs]="1"  # 1 epoch for ablation, adjust as needed
    [save_total_limit]="3"
    [run_name]="$SUFFIX"
    [output_dir]="$OUTPUT_DIR"
    [per_device_train_batch_size]="$PER_DEVICE_TRAIN_BATCH_SIZE"
    [per_device_eval_batch_size]="4"
    [gradient_accumulation_steps]="$GRADIENT_ACCUMULATION_STEPS"
    [evaluation_strategy]="no"
    #[save_strategy]="epoch"
    [save_strategy]="steps"
    [save_steps]="500"
    [learning_rate]="2e-5"
    [weight_decay]="0."
    [warmup_ratio]="0.03"
    [lr_scheduler_type]="cosine"
    [logging_steps]="5"
    [dataloader_num_workers]="6"
    [report_to]="wandb"
    [dataloader_drop_last]="True"
    [seed]="$SEED"
    [data_seed]="$SEED"
)

echo "========================================"
echo " Training Configuration"
echo "========================================"

echo "--- Resume ---"
echo "  RESUME_CHECKPOINT_PATH:             $RESUME_CHECKPOINT_PATH"
echo "  SEED:                               $SEED"
if [[ "$RESUME_CHECKPOINT_PATH" != "none" ]]; then
    echo "  *** RESUMING TRAINING — weights will be updated in-place ***"
else
    echo "  *** FRESH TRAINING — new run ***"
fi
echo ""

echo "--- ModelArguments ---"
for key in "${!MODEL_ARGS[@]}"; do
    printf "  %-35s %s\n" "$key:" "${MODEL_ARGS[$key]}"
done

echo ""
echo "--- DataArguments ---"
for key in "${!DATA_ARGS[@]}"; do
    printf "  %-35s %s\n" "$key:" "${DATA_ARGS[$key]}"
done

echo ""
echo "--- TrainingArguments ---"
for key in "${!TRAINING_ARGS[@]}"; do
    printf "  %-35s %s\n" "$key:" "${TRAINING_ARGS[$key]}"
done

declare -a TORCHRUN_ARGS=()

for key in "${!MODEL_ARGS[@]}"; do
    TORCHRUN_ARGS+=("--${key}")
    TORCHRUN_ARGS+=("${MODEL_ARGS[$key]}")
done

for key in "${!DATA_ARGS[@]}"; do
    TORCHRUN_ARGS+=("--${key}")
    TORCHRUN_ARGS+=("${DATA_ARGS[$key]}")
done

for key in "${!TRAINING_ARGS[@]}"; do
    TORCHRUN_ARGS+=("--${key}")
    TORCHRUN_ARGS+=("${TRAINING_ARGS[$key]}")
done

srun --export=ALL torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NUM_GPUS_PER_NODE" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        llava/train/train_mem.py \
        "${TORCHRUN_ARGS[@]}"
    2>&1 | tee "$OUTPUT_DIR/train.log"

exit 0