#!/bin/bash
#SBATCH --job-name=RoPE_Spherical_cut3r_100p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

set -euo pipefail
cleanup_on_training_failure() {
    local status=$?
    trap - EXIT TERM INT ERR
    if [[ "$status" -ne 0 ]]; then
        echo "[ERROR] Training script failed with status $status; canceling job ${SLURM_JOB_ID:-unknown} to avoid idle allocation."
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            scancel "$SLURM_JOB_ID" >/dev/null 2>&1 || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_training_failure EXIT TERM INT ERR
SRUN_FAIL_FAST_ARGS=(--kill-on-bad-exit=1 --wait=30)

# Edit this block directly for each experiment.
TRAIN_DATA_PERCENTAGE="100"
SUFFIX="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
NOTE="Spherical MetricGroundedGeometryProjection: 2D visual-token self-attention with Geometry-RoPE from CUT3R point-map sidecars | ${TRAIN_DATA_PERCENTAGE}% training data"
CONDA_ENV_NAME="vlm3r"

MODEL_USE_GEOMETRY_AWARE_PROJECTION="True"
MODEL_SPATIAL_ENCODER_TYPE="cut3r"
MODEL_GEOMETRY_POSITION_MODE="spherical"
MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
# geometry_position_mode:
# depth [1]:      position = log_depth, 1D
# xyz [2, 1, 2]:        position = normalized x,y,z, 3D
# spherical [2,1,2]:  position = azimuth, elevation, log_distance, 3D
#
# Coordinate consistency rule:
# - CUT3R sidecars expose point_maps_ref/pts3d_in_other_view
#   (reference/anchor-frame) and point_maps_cam/pts3d_in_self_view
#   (per-frame camera).
# - Use the same coordinate source in training and evaluation; do not allow
#   eval-only aliases to switch ref<->cam.


MODEL_NUM_GEOMETRY_PROJECTION_LAYERS="1" #after 3D RoPE 有幾層 transformer layer
# visual_tokens
#  -> LayerNorm
#  -> Q/K/V linear
#  -> GeometryRoPE applied to Q and K only
#  -> self-attention over visual tokens
#  -> gated residual: x + gamma_attn * attn_out
#  -> LayerNorm
#  -> FFN: Linear -> GELU -> Linear
#  -> gated residual: x + gamma_ffn * ffn_out
MODEL_GEOMETRY_PROJECTION_NUM_HEADS="16"
#你的 geometry projection block 裡 Q/K/V 都是 hidden_size -> hidden_size
#然後 reshape 成：
#[B, N, hidden_size] -> [B, num_heads, N, head_dim]

# auxiliary geometry head 的設計是為了在中途提供額外的監督信號，幫助模型更好地學習幾何投影分支的表示能力。透過預測幾何屬性（如方位角、仰角、距離等），模型可以在訓練過程中獲得更直接的幾何信息反饋，從而提升整體性能。
MODEL_USE_AUXILIARY_GEOMETRY_HEAD="True"
MODEL_USE_AUXILIARY_GEOMETRY_LOSS="True"
MODEL_AUX_GEOMETRY_TARGETS="azimuth,elevation,log_distance"
# lambda_geo 的設置需要根據 auxiliary geometry loss 的規模和主任務 loss 的規模來調整。一般來說，可以從一個較小的值（如 0.1）開始，觀察訓練過程中 auxiliary geometry loss 和主任務 loss 的變化趨勢。如果 auxiliary geometry loss 過大，可能會對主任務的學習產生干擾，此時可以適當降低 lambda_geo；反之，如果 auxiliary geometry loss 過小，可能無法充分發揮其對幾何投影分支的監督作用，此時可以適當提高 lambda_geo。
MODEL_LAMBDA_GEO="0.1"

MODEL_GEOMETRY_LOSS_TYPE="smooth_l1"
#e = pred - target
#MSE: L_MSE = e²
    #   你非常相信 geometry target，而且想強烈懲罰大誤差。
    # 小誤差：懲罰很小
    # 大誤差：懲罰非常大，因為平方
    # gradient：2e
# L1:Mean Absolute Error, L_L1 = |e|
    #   target 很 noisy，outlier 很多，但梯度會比較不平滑。
    # 小誤差：線性懲罰
    # 大誤差：也是線性懲罰
    # gradient：sign(e)
# Smooth L1:
    # 小誤差時像 MSE
    # 大誤差時像 L1
    # 如果 |e| 很小：
    #   loss ≈ 0.5 * e²

    # 如果 |e| 很大：
    # 小誤差區域：
    #   平滑，方便細緻收斂

    # 大誤差區域：
    #   不會像 MSE 被 outlier 主導
    #   loss ≈ |e| - constant
    #   預設最合理。

MODEL_DETACH_GEOMETRY_TARGETS="True"
MODEL_GEOMETRY_GATE_INIT="0.0" # gate_init=0.0 代表一開始幾乎是 identity mapping, 讓模型有機會先專注於學習融合後的語言/視覺表示，再逐漸學習如何利用幾何投影分支提供的增強表示。
MODEL_USE_GEOMETRY_CONFIDENCE_MASK="True"

#如果 sidecar 有 confidence / conf / depth_conf / pts3d_conf，會只在 confidence > 0 的位置算 geometry attention/loss。
#CUT3R point sidecar 目前沒有 confidence，所以主要還是靠 finite 且 depth > 0 的 mask。
MODEL_ALLOW_MISSING_GEOMETRY_TARGETS="False"
#如果你要求 azimuth,elevation,log_distance，但 sidecar 只能提供 depth，會直接報錯。這是好事，避免你以為有 supervise，其實沒有
MODEL_GEOMETRY_POSITION_MAX_ABS="10.0"
#只影響 xyz mode。xyz 會除以 scene scale 後 clamp 到 [-10, 10]，
#避免超大座標讓 RoPE angle 爆掉。
MODEL_GEOMETRY_FIXED_SCENE_SCALE="5.0"
#只影響 xyz mode 的 fallback。
#如果某個 sample 沒有有效 depth，就用 5.0 當 scene scale。
MODEL_GEOMETRY_PROJECTION_DROPOUT="0.0"
#geometry projection block 裡 attention output / FFN output 的 dropout。現在關掉，讓 ablation 比較乾淨。


# Geometry projection only needs pre-extracted sidecars. This tag tells the data loader
# to load those .pt files without instantiating a runtime CUT3R/PI3/VGGT tower.
DATA_SPATIAL_TOWER_TYPE="cut3r"

# ==============================================================================
# Data source roots. Default to FAST; keep WORK mirror as fallback.
WORK_DATA_ROOT="${WORK_DATA_ROOT:-/leonardo_work/EUHPC_D32_006/train_data/vlm3r}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DATA_ROOT="${DATA_ROOT:-$FAST_DATA_ROOT}"
# DATA_ROOT="${DATA_ROOT:-$WORK_DATA_ROOT}"  # WORK mirror fallback if FAST is unstable.
# SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$FAST_DATA_ROOT}"  # FAST CUT3R point-map sidecars.
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
SPATIAL_FEATURES_SUBDIR="spatial_features_points"

LOCAL_MODEL_BASE="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2"
LOCAL_SIGLIP="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384"

TRAIN_SAVE_ROOT="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R"
LOG_DIR="/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/train_log"

WANDB_DIR="$WORK/wandb"
WANDB_CACHE_DIR="$WORK/wandb_cache"
WANDB_CONFIG_DIR="$WORK/wandb_config"

HF_HOME="/leonardo_scratch/fast/EUHPC_D32_006/hf_cache"
HF_DATASETS_CACHE="$HF_HOME/datasets"
HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

RESUME_MODE="fresh"
RESUME_CHECKPOINT_PATH="none"
ZERO_SPATIAL_FEATURES="False"
SEED="42"

MODEL_LORA_ENABLE="True"
MODEL_LORA_R="128"
MODEL_LORA_ALPHA="256"

# ======= feature alinment losses =======
SPATIAL_RANK_LOSS_ENABLE="False"
LAMBDA_SIM="0.01"
SPATIAL_RANK_MARGIN="0.2"
ANCHORS_PER_FRAME="128"
POSITIVE_TOP_PERCENT="10"
NEGATIVE_BOTTOM_PERCENT="30"
SPATIAL_RANK_DEBUG_CHECKS="False"



MODEL_TUNE_SPATIAL_TOWER="False"
MODEL_TUNE_FUSION_BLOCK="False"
MODEL_TUNE_GEOMETRY_AWARE_PROJECTION="True"
MODEL_TUNE_MM_MLP_ADAPTER="True"

if [[ "$MODEL_TUNE_FUSION_BLOCK" == "True" ]]; then
    echo "[ERROR] This script does not instantiate a fusion block; keep MODEL_TUNE_FUSION_BLOCK=False."
    exit 1
fi

MODEL_VERSION="qwen_1_5"
MODEL_MM_PROJECTOR_TYPE="mlp2x_gelu"
MODEL_MM_VISION_SELECT_LAYER="-2"
MODEL_MM_USE_IM_START_END="False"
MODEL_MM_USE_IM_PATCH_TOKEN="False"
MODEL_IMAGE_ASPECT_RATIO="anyres_max_9"
MODEL_IMAGE_GRID_PINPOINTS="(1x1),...,(6x6)"
MODEL_MM_PATCH_MERGE_TYPE="spatial_unpad"
MODEL_BF16="True"
MODEL_TF32="True"
MODEL_MAX_LENGTH="32768"
MODEL_GRADIENT_CHECKPOINTING="True"
MODEL_LAZY_PREPROCESS="True"
MODEL_TORCH_COMPILE="True"
MODEL_TORCH_COMPILE_BACKEND="inductor"
MODEL_FRAMES_UPBOUND="32"
MODEL_MM_NEWLINE_POSITION="grid"
MODEL_ADD_TIME_INSTRUCTION="True"
MODEL_FORCE_SAMPLE="True"
MODEL_MM_SPATIAL_POOL_STRIDE="2"

DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data.yaml}"  # FAST json_path entries.
# DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data_work.yaml}"  # WORK mirror fallback.
DATA_GROUP_BY_MODALITY_LENGTH="True"
TRAIN_DATA_PERCENTAGE_SEED="$SEED"
TRAIN_DATA_SHUFFLE="True"

PER_DEVICE_TRAIN_BATCH_SIZE="1"
TARGET_GLOBAL_BATCH_SIZE="128"
NUM_TRAIN_EPOCHS="1"
SAVE_TOTAL_LIMIT="2"
SAVE_STRATEGY="steps"
SAVE_STEPS="100"
LEARNING_RATE="2e-5"
WEIGHT_DECAY="0."
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS="5"
DATALOADER_NUM_WORKERS="6"
REPORT_TO="wandb"
DATALOADER_DROP_LAST="True"
DDP_LAUNCH_MODE="elastic"



echo "-------- Note --------"
echo "  note: $NOTE"
mkdir -p logs/train
mkdir -p "$LOG_DIR"

JOB_TIME_LIMIT=$(squeue -j "$SLURM_JOB_ID" -h -o "%l")

echo "=== SLURM Job Specifications ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per Node: ${SLURM_GPUS_PER_NODE:-N/A}"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE:-N/A}"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: ${SLURM_JOB_QOS:-N/A}"
MEMORY_PER_NODE="${SLURM_MEM_PER_NODE:-N/A}"
echo "Memory per Node: $MEMORY_PER_NODE"
echo "Output: ${SLURM_STDOUT:-logs/train/%x_%j.out}"
echo "Error: ${SLURM_STDERR:-logs/train/%x_%j.err}"
echo "Job Time Limit: $JOB_TIME_LIMIT"

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
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

if [[ -v LD_LIBRARY_PATH && -n "$LD_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export WANDB_DIR="$WANDB_DIR"
export WANDB_CACHE_DIR="$WANDB_CACHE_DIR"
export WANDB_CONFIG_DIR="$WANDB_CONFIG_DIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

export HF_HOME="$HF_HOME"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

if [[ -v SLURM_GPUS_ON_NODE && -n "$SLURM_GPUS_ON_NODE" ]]; then
    NUM_GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [[ -v SLURM_GPUS_PER_NODE && -n "$SLURM_GPUS_PER_NODE" ]]; then
    NUM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE"
else
    NUM_GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
if [[ -v SLURM_JOB_NUM_NODES && -n "$SLURM_JOB_NUM_NODES" ]]; then
    NNODES="$SLURM_JOB_NUM_NODES"
else
    NNODES=1
fi
WORLD_SIZE=$((NNODES * NUM_GPUS_PER_NODE))
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export MASTER_ADDR MASTER_PORT NNODES NUM_GPUS_PER_NODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"
echo "[DDP] NNODES=$NNODES"
echo "[DDP] NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "[DDP] DDP_LAUNCH_MODE=$DDP_LAUNCH_MODE NCCL_DEBUG=$NCCL_DEBUG"
if [[ -v NCCL_SOCKET_IFNAME && -n "$NCCL_SOCKET_IFNAME" ]]; then
    echo "[DDP] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
fi

MID_RUN_NAME="$SUFFIX"
OUTPUT_DIR="$TRAIN_SAVE_ROOT/$MID_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

if [[ "$RESUME_CHECKPOINT_PATH" == "none" ]]; then
    if [[ "$RESUME_MODE" == "continue" ]]; then
        RESUME_CHECKPOINT_PATH="auto"
    else
        RESUME_CHECKPOINT_PATH="none"
    fi
fi
export RESUME_CHECKPOINT_PATH

if [[ ! -d "$LOCAL_MODEL_BASE" ]]; then
    echo "[ERROR] Local model base not found: $LOCAL_MODEL_BASE"
    exit 1
fi
if [[ ! -d "$LOCAL_SIGLIP" ]]; then
    echo "[ERROR] Local SigLIP not found: $LOCAL_SIGLIP"
    exit 1
fi
if [[ ! -d "$SPATIAL_FEATURES_ROOT" ]]; then
    echo "[ERROR] SPATIAL_FEATURES_ROOT not found: $SPATIAL_FEATURES_ROOT"
    exit 1
fi
EXPECTED_SPATIAL_DATASETS=("scannet" "scannetpp" "arkitscenes")
MISSING_SPATIAL_SUBDIRS=0
for dataset_name in "${EXPECTED_SPATIAL_DATASETS[@]}"; do
    sidecar_dir="$SPATIAL_FEATURES_ROOT/$dataset_name/$SPATIAL_FEATURES_SUBDIR"
    if [[ ! -d "$sidecar_dir" ]]; then
        echo "[ERROR] Missing geometry sidecar directory: $sidecar_dir"
        MISSING_SPATIAL_SUBDIRS=1
    fi
done
if [[ "$MISSING_SPATIAL_SUBDIRS" != "0" ]]; then
    echo "[ERROR] Pre-extract the requested geometry sidecars before submitting this job."
    exit 1
fi

echo "[LOCAL MODEL] model_name_or_path=$LOCAL_MODEL_BASE"
echo "[LOCAL MODEL] vision_tower=$LOCAL_SIGLIP"
echo "[GEOMETRY] data_spatial_tower_type=$DATA_SPATIAL_TOWER_TYPE"
echo "[GEOMETRY] spatial_features_root=$SPATIAL_FEATURES_ROOT"
echo "[GEOMETRY] spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR"

denom=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( TARGET_GLOBAL_BATCH_SIZE % denom != 0 )); then
    echo "[ERROR] TARGET_GLOBAL_BATCH_SIZE($TARGET_GLOBAL_BATCH_SIZE) not divisible by WORLD_SIZE*PER_DEVICE_TRAIN_BATCH_SIZE($denom)"
    echo "Please adjust TARGET_GLOBAL_BATCH_SIZE or PER_DEVICE_TRAIN_BATCH_SIZE."
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / denom))
echo "[BATCH] PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "[BATCH] TARGET_GLOBAL_BATCH_SIZE=$TARGET_GLOBAL_BATCH_SIZE"
echo "[BATCH] GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

VALID_GEOMETRY_POSITION_MODES=("depth" "xyz" "spherical")
IS_VALID_GEOMETRY_POSITION_MODE="False"
for mode_name in "${VALID_GEOMETRY_POSITION_MODES[@]}"; do
    if [[ "$MODEL_GEOMETRY_POSITION_MODE" == "$mode_name" ]]; then
        IS_VALID_GEOMETRY_POSITION_MODE="True"
        break
    fi
done
if [[ "$IS_VALID_GEOMETRY_POSITION_MODE" != "True" ]]; then
    echo "[ERROR] Unsupported MODEL_GEOMETRY_POSITION_MODE: $MODEL_GEOMETRY_POSITION_MODE"
    echo "[ERROR] Supported values: ${VALID_GEOMETRY_POSITION_MODES[*]}"
    exit 1
fi

echo "[GEOMETRY] use_geometry_aware_projection=$MODEL_USE_GEOMETRY_AWARE_PROJECTION"
echo "[GEOMETRY] position_mode=$MODEL_GEOMETRY_POSITION_MODE layers=$MODEL_NUM_GEOMETRY_PROJECTION_LAYERS heads=$MODEL_GEOMETRY_PROJECTION_NUM_HEADS"
echo "[GEOMETRY] aux_head=$MODEL_USE_AUXILIARY_GEOMETRY_HEAD aux_loss=$MODEL_USE_AUXILIARY_GEOMETRY_LOSS targets=$MODEL_AUX_GEOMETRY_TARGETS lambda_geo=$MODEL_LAMBDA_GEO"
echo "[ABLATION] ZERO_SPATIAL_FEATURES=$ZERO_SPATIAL_FEATURES"
echo "[SPATIAL_RANK] ENABLE=$SPATIAL_RANK_LOSS_ENABLE LAMBDA_SIM=$LAMBDA_SIM MARGIN=$SPATIAL_RANK_MARGIN"

declare -A MODEL_ARGS=(
    [model_name_or_path]="$LOCAL_MODEL_BASE"
    [lora_enable]="$MODEL_LORA_ENABLE"
    [lora_r]="$MODEL_LORA_R"
    [lora_alpha]="$MODEL_LORA_ALPHA"
    [use_geometry_aware_projection]="$MODEL_USE_GEOMETRY_AWARE_PROJECTION"
    [spatial_encoder_type]="$MODEL_SPATIAL_ENCODER_TYPE"
    [geometry_position_mode]="$MODEL_GEOMETRY_POSITION_MODE"
    [geo_rope_point_map_key]="$MODEL_GEO_ROPE_POINT_MAP_KEY"
    [num_geometry_projection_layers]="$MODEL_NUM_GEOMETRY_PROJECTION_LAYERS"
    [geometry_projection_num_heads]="$MODEL_GEOMETRY_PROJECTION_NUM_HEADS"
    [use_auxiliary_geometry_head]="$MODEL_USE_AUXILIARY_GEOMETRY_HEAD"
    [use_auxiliary_geometry_loss]="$MODEL_USE_AUXILIARY_GEOMETRY_LOSS"
    [aux_geometry_targets]="$MODEL_AUX_GEOMETRY_TARGETS"
    [lambda_geo]="$MODEL_LAMBDA_GEO"
    [geometry_loss_type]="$MODEL_GEOMETRY_LOSS_TYPE"
    [detach_geometry_targets]="$MODEL_DETACH_GEOMETRY_TARGETS"
    [geometry_gate_init]="$MODEL_GEOMETRY_GATE_INIT"
    [use_geometry_confidence_mask]="$MODEL_USE_GEOMETRY_CONFIDENCE_MASK"
    [allow_missing_geometry_targets]="$MODEL_ALLOW_MISSING_GEOMETRY_TARGETS"
    [geometry_position_max_abs]="$MODEL_GEOMETRY_POSITION_MAX_ABS"
    [geometry_fixed_scene_scale]="$MODEL_GEOMETRY_FIXED_SCENE_SCALE"
    [geometry_projection_dropout]="$MODEL_GEOMETRY_PROJECTION_DROPOUT"
    [tune_spatial_tower]="$MODEL_TUNE_SPATIAL_TOWER"
    [tune_fusion_block]="$MODEL_TUNE_FUSION_BLOCK"
    [tune_geometry_aware_projection]="$MODEL_TUNE_GEOMETRY_AWARE_PROJECTION"
    [tune_mm_mlp_adapter]="$MODEL_TUNE_MM_MLP_ADAPTER"
    [version]="$MODEL_VERSION"
    [vision_tower]="$LOCAL_SIGLIP"
    [mm_projector_type]="$MODEL_MM_PROJECTOR_TYPE"
    [mm_vision_select_layer]="$MODEL_MM_VISION_SELECT_LAYER"
    [mm_use_im_start_end]="$MODEL_MM_USE_IM_START_END"
    [mm_use_im_patch_token]="$MODEL_MM_USE_IM_PATCH_TOKEN"
    [image_aspect_ratio]="$MODEL_IMAGE_ASPECT_RATIO"
    [image_grid_pinpoints]="$MODEL_IMAGE_GRID_PINPOINTS"
    [mm_patch_merge_type]="$MODEL_MM_PATCH_MERGE_TYPE"
    [bf16]="$MODEL_BF16"
    [tf32]="$MODEL_TF32"
    [model_max_length]="$MODEL_MAX_LENGTH"
    [gradient_checkpointing]="$MODEL_GRADIENT_CHECKPOINTING"
    [lazy_preprocess]="$MODEL_LAZY_PREPROCESS"
    [torch_compile]="$MODEL_TORCH_COMPILE"
    [torch_compile_backend]="$MODEL_TORCH_COMPILE_BACKEND"
    [frames_upbound]="$MODEL_FRAMES_UPBOUND"
    [mm_newline_position]="$MODEL_MM_NEWLINE_POSITION"
    [add_time_instruction]="$MODEL_ADD_TIME_INSTRUCTION"
    [force_sample]="$MODEL_FORCE_SAMPLE"
    [mm_spatial_pool_stride]="$MODEL_MM_SPATIAL_POOL_STRIDE"
)

declare -A DATA_ARGS=(
    [data_path]="$DATA_PATH_YAML"
    [image_folder]="$DATA_ROOT"
    [video_folder]="$DATA_ROOT"
    [zero_spatial_features]="$ZERO_SPATIAL_FEATURES"
    [spatial_tower_type]="$DATA_SPATIAL_TOWER_TYPE"
    [spatial_features_root]="$SPATIAL_FEATURES_ROOT"
    [spatial_features_subdir]="$SPATIAL_FEATURES_SUBDIR"
    [train_data_percentage]="$TRAIN_DATA_PERCENTAGE"
    [train_data_percentage_seed]="$TRAIN_DATA_PERCENTAGE_SEED"
    [train_data_shuffle]="$TRAIN_DATA_SHUFFLE"
    [group_by_modality_length]="$DATA_GROUP_BY_MODALITY_LENGTH"
)

declare -A TRAINING_ARGS=(
    [deepspeed]="scripts/zero2.json"
    [num_train_epochs]="$NUM_TRAIN_EPOCHS"
    [save_total_limit]="$SAVE_TOTAL_LIMIT"
    [run_name]="$SUFFIX"
    [output_dir]="$OUTPUT_DIR"
    [per_device_train_batch_size]="$PER_DEVICE_TRAIN_BATCH_SIZE"
    [per_device_eval_batch_size]="4"
    [gradient_accumulation_steps]="$GRADIENT_ACCUMULATION_STEPS"
    [evaluation_strategy]="no"
    [save_strategy]="$SAVE_STRATEGY"
    [save_steps]="$SAVE_STEPS"
    [learning_rate]="$LEARNING_RATE"
    [weight_decay]="$WEIGHT_DECAY"
    [warmup_ratio]="$WARMUP_RATIO"
    [lr_scheduler_type]="$LR_SCHEDULER_TYPE"
    [logging_steps]="$LOGGING_STEPS"
    [dataloader_num_workers]="$DATALOADER_NUM_WORKERS"
    [report_to]="$REPORT_TO"
    [dataloader_drop_last]="$DATALOADER_DROP_LAST"
    [seed]="$SEED"
    [data_seed]="$SEED"
    [spatial_rank_loss_enable]="$SPATIAL_RANK_LOSS_ENABLE"
    [lambda_sim]="$LAMBDA_SIM"
    [spatial_rank_margin]="$SPATIAL_RANK_MARGIN"
    [anchors_per_frame]="$ANCHORS_PER_FRAME"
    [positive_top_percent]="$POSITIVE_TOP_PERCENT"
    [negative_bottom_percent]="$NEGATIVE_BOTTOM_PERCENT"
    [spatial_rank_debug_checks]="$SPATIAL_RANK_DEBUG_CHECKS"
)

echo "========================================"
echo " Training Configuration"
echo "========================================"

echo "--- Resume ---"
echo "  TRAIN_SAVE_ROOT:                     $TRAIN_SAVE_ROOT"
echo "  OUTPUT_RUN_NAME:                     $MID_RUN_NAME"
echo "  OUTPUT_DIR:                          $OUTPUT_DIR"
echo "  RESUME_MODE:                         $RESUME_MODE"
echo "  RESUME_CHECKPOINT_PATH:              $RESUME_CHECKPOINT_PATH"
echo "  SEED:                                $SEED"
if [[ "$RESUME_CHECKPOINT_PATH" != "none" ]]; then
    echo "  *** RESUMING TRAINING - weights will be updated in-place ***"
else
    echo "  *** FRESH TRAINING - new run ***"
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

case "$DDP_LAUNCH_MODE" in
    elastic)
        srun "${SRUN_FAIL_FAST_ARGS[@]}" --export=ALL torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$NUM_GPUS_PER_NODE" \
            --rdzv_id="$SLURM_JOB_ID" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
            llava/train/train_mem.py \
            "${TORCHRUN_ARGS[@]}" \
            | tee "$LOG_DIR/${SUFFIX}.log"
        ;;
    static)
        srun "${SRUN_FAIL_FAST_ARGS[@]}" --export=ALL bash -c '
            echo "[DDP] host=$(hostname) SLURM_PROCID=${SLURM_PROCID:-NA} node_rank=${SLURM_NODEID:-NA} local_id=${SLURM_LOCALID:-NA}" >&2
            exec torchrun \
                --nnodes="$NNODES" \
                --nproc_per_node="$NUM_GPUS_PER_NODE" \
                --node_rank="$SLURM_NODEID" \
                --master_addr="$MASTER_ADDR" \
                --master_port="$MASTER_PORT" \
                llava/train/train_mem.py \
                "$@"
        ' bash "${TORCHRUN_ARGS[@]}" \
            | tee "$LOG_DIR/${SUFFIX}.log"
        ;;
    *)
        echo "[ERROR] Unsupported DDP_LAUNCH_MODE=$DDP_LAUNCH_MODE (expected elastic or static)"
        exit 1
        ;;
esac

exit 0
