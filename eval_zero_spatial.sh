#!/bin/bash
#SBATCH --job-name=Eval_zero_spatial_feature
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4           # 依你的叢集格式：也可能是 --gpus-per-node=1
#SBATCH --ntasks-per-node=1       # 通常 1 個 task，裡面用 torchrun 起多 GPU processes
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=normal  # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,,lrdn0080,lrdn0843


# ================================================== User-defined variables ==================================================
NOTE="Eval self-trained ablation study of zero spatial feature on VSI-Bench. Model: LLaVA-NeXT-Video-7B-Qwen2 + SigLIP patch14-384. See eval_vsi.sh for details." 
# ================================================== User-defined variables ==================================================



echo "-------- Note --------"
echo "  note: $NOTE"
JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h -o "%l")
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


# ==================================================User-defined variables ==================================================
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_root=/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R
stale_tmp_hours="${STALE_TMP_HOURS:-24}"
stale_tmp_minutes=$((stale_tmp_hours * 60))
job_name="${SLURM_JOB_NAME:-vlm3r_eval_${benchmark}}"
safe_job_name="$(echo "$job_name" | tr -c 'A-Za-z0-9._-' '_')"
output_path="$output_root/$safe_job_name"
mkdir -p "$output_root"

# Best-effort cleanup of abandoned lmms_eval staging directories.
find "$output_root" \
  -mindepth 1 -maxdepth 1 \
  -type d -name ".lmms_eval_tmp_*" \
  -mmin +"$stale_tmp_minutes" \
  -print -exec rm -rf {} + 2>/dev/null || true

staging_output_path="$(mktemp -d "${output_root}/.lmms_eval_tmp_${safe_job_name}.XXXXXX")"
pretrained_local=/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/train/zero_spatial_features
model_base_local=/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2
siglip_local=/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/siglip-so400m-patch14-384
zero_spatial_features_eval=True
# ================================================== User-defined variables ==================================================



echo "=== Evaluation Configuration ==="
echo "Benchmark: $benchmark"
echo "Output Root: $output_root"
echo "Job Name: $job_name"
echo "Output Path (final): $output_path"
echo "Output Path (staging): $staging_output_path"
echo "Pretrained (local): $pretrained_local"
echo "Model base (local): $model_base_local"
echo "SigLIP (local): $siglip_local"
echo "Zero spatial features (eval): $zero_spatial_features_eval"

set -eo pipefail
export HF_HOME="$FAST/hf_cache"
export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_CACHE="$FAST/hf_cache/hub"
export TRANSFORMERS_CACHE="$FAST/hf_cache/transformers"
export HUGGINGFACE_HUB_CACHE="$FAST/hf_cache/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE="$FAST/hf_cache/modules"
# 讓 datasets 不要一直想去網路 check
export HF_UPDATE_DOWNLOAD_COUNTS=0




# Cluster-specific modules (依你的 launch_training.sh 的想法補完整)
HOSTNAME=$(hostname)
which nvidia-smi || true
nvidia-smi -L || true

module load cuda/12.1
# module load cudnn
# module load profile/deeplrn


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
eval "$(conda shell.bash hook)"
conda activate vsibench

# 優先使用 Conda 環境的動態庫，避免系統舊版 libstdc++ 衝突
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0


pretrained_ref="$pretrained_local"
model_base_ref="$model_base_local"

if [[ ! -d "$pretrained_ref" ]]; then
  echo "[ERROR] Offline pretrained path does not exist: $pretrained_ref"
  exit 1
fi
if [[ ! -d "$model_base_ref" ]]; then
  echo "[ERROR] Offline model_base path does not exist: $model_base_ref"
  exit 1
fi
if [[ ! -d "$siglip_local" ]]; then
  echo "[ERROR] Offline SigLIP path does not exist: $siglip_local"
  echo "        Download/copy google/siglip-so400m-patch14-384 to this directory first."
  exit 1
fi

# Build a runtime copy of LoRA config and force mm_vision_tower to local SigLIP directory.
# Keep this outside output_path so eval logs only contain evaluation artifacts.
runtime_pretrained="$(mktemp -d "${TMPDIR:-/tmp}/vlm3r_runtime_pretrained.XXXXXX")"
echo "[INFO] Runtime pretrained temp dir: $runtime_pretrained"
cleanup_runtime_pretrained() {
  rm -rf "$runtime_pretrained"
  rm -rf "$staging_output_path"
}
trap cleanup_runtime_pretrained EXIT

shopt -s dotglob nullglob
for item in "$pretrained_ref"/*; do
  base_name="$(basename "$item")"
  if [[ "$base_name" == checkpoint-* ]]; then
    continue
  fi
  cp -a "$item" "$runtime_pretrained/"
done
shopt -u dotglob nullglob

python - "$runtime_pretrained/config.json" "$siglip_local" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
siglip_local_path = pathlib.Path(sys.argv[2])

if not config_path.exists():
    raise FileNotFoundError(f"Missing config.json in runtime_pretrained: {config_path}")

cfg = json.loads(config_path.read_text())
old = cfg.get("mm_vision_tower")
cfg["mm_vision_tower"] = str(siglip_local_path)
config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print(f"[INFO] Patched mm_vision_tower: {old} -> {cfg['mm_vision_tower']}")
PY

pretrained_ref="$runtime_pretrained"

echo "Resolved pretrained: $pretrained_ref"
echo "Resolved model_base: $model_base_ref"
echo "Resolved SigLIP: $siglip_local"

python - <<'PY'
import os
from datasets import load_dataset

for k in [
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HF_HUB_CACHE",
    "HF_MODULES_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
]:
    print(f"{k}={os.environ.get(k)}")

ds = load_dataset(
    "json",
    data_files={
        "train": "/leonardo_scratch/fast/EUHPC_D32_006/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test.jsonl"
    }
)
print(ds)
print("num_examples =", len(ds["train"]))
PY


# === Start Evaluation ===
log_samples_suffix="$safe_job_name"
accelerate launch --num_processes=4 -m lmms_eval \
    --model vlm_3r \
    --model_args "pretrained=$pretrained_ref,model_base=$model_base_ref,model_name=llava_qwen_lora,conv_template=qwen_1_5,max_frames_num=32,zero_spatial_features=$zero_spatial_features_eval" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "$log_samples_suffix" \
    --output_path "$staging_output_path"

generated_output_dir="$(find "$staging_output_path" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [[ -z "$generated_output_dir" ]]; then
  echo "[ERROR] Cannot find lmms_eval output under staging path: $staging_output_path"
  exit 1
fi

if [[ -d "$output_path" ]]; then
  backup_output_path="${output_path}_backup_$(date "+%Y%m%d_%H%M%S")"
  echo "[WARN] Existing output path found, moving it to: $backup_output_path"
  mv "$output_path" "$backup_output_path"
fi

mv "$generated_output_dir" "$output_path"
echo "[INFO] Final evaluation output: $output_path"