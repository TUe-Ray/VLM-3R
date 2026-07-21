#!/bin/bash
#SBATCH --job-name=Eval_VLM3R_SnelliusParity_Offline_secondtry
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

# Run this on compute nodes WITHOUT internet access.
# It reuses online-prefetched cache and runs using the Snellius lmms_eval files.

SNELLIUS_REPO_DIR="${SNELLIUS_REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R_snellius}"
SUBMODULE_DIR="${SUBMODULE_DIR:-$SNELLIUS_REPO_DIR/thinking-in-space}"

FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
MODEL_ROOT="${MODEL_ROOT:-$FAST_ROOT/hf_models/VLM3R}"

PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$MODEL_ROOT/Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
CUT3R_CKPT_EXPECTED="${CUT3R_CKPT_EXPECTED:-$SNELLIUS_REPO_DIR/CUT3R/src/cut3r_512_dpt_4_64.pth}"
CUT3R_CKPT_FALLBACK="${CUT3R_CKPT_FALLBACK:-$SNELLIUS_REPO_DIR/CUT3R_backup_incomplete/src/cut3r_512_dpt_4_64.pth}"

CONDA_BASE="${CONDA_BASE:-$WORK/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vsibench}"
TASK="${TASK:-vsibench}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"

OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/$(date +%Y%m%d_%H%M%S)_snellius_parity_offline}"

echo "==== Job info ===="
date
echo "HOSTNAME=$(hostname)"
echo "SNELLIUS_REPO_DIR=$SNELLIUS_REPO_DIR"
echo "SUBMODULE_DIR=$SUBMODULE_DIR"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "MODEL_BASE_LOCAL=$MODEL_BASE_LOCAL"
echo "SIGLIP_LOCAL=$SIGLIP_LOCAL"
echo "CUT3R_CKPT_EXPECTED=$CUT3R_CKPT_EXPECTED"
echo "=================="

if [[ ! -d "$PRETRAINED_LOCAL" ]]; then
  echo "[ERROR] Missing PRETRAINED_LOCAL: $PRETRAINED_LOCAL"
  exit 1
fi
if [[ ! -d "$MODEL_BASE_LOCAL" ]]; then
  echo "[ERROR] Missing MODEL_BASE_LOCAL: $MODEL_BASE_LOCAL"
  exit 1
fi
if [[ ! -d "$SIGLIP_LOCAL" ]]; then
  echo "[ERROR] Missing SIGLIP_LOCAL: $SIGLIP_LOCAL"
  exit 1
fi
if [[ ! -d "$SUBMODULE_DIR" ]]; then
  echo "[ERROR] Missing SUBMODULE_DIR: $SUBMODULE_DIR"
  exit 1
fi
if [[ ! -f "$SNELLIUS_REPO_DIR/CUT3R/src/dust3r/model.py" ]]; then
  echo "[ERROR] Incomplete CUT3R source in $SNELLIUS_REPO_DIR/CUT3R"
  echo "        Run /leonardo/home/userexternal/shuang00/VLM-3R/setup_cut3r_snellius_official.sh on login node first."
  exit 1
fi
if [[ ! -f "$CUT3R_CKPT_EXPECTED" ]]; then
  if [[ -f "$CUT3R_CKPT_FALLBACK" ]]; then
    mkdir -p "$(dirname "$CUT3R_CKPT_EXPECTED")"
    ln -sf "$CUT3R_CKPT_FALLBACK" "$CUT3R_CKPT_EXPECTED"
    echo "[INFO] Restored CUT3R checkpoint via symlink: $CUT3R_CKPT_EXPECTED -> $CUT3R_CKPT_FALLBACK"
  else
    echo "[ERROR] Missing CUT3R checkpoint: $CUT3R_CKPT_EXPECTED"
    echo "        Also missing fallback: $CUT3R_CKPT_FALLBACK"
    exit 1
  fi
fi

module purge || true
module load cuda/12.1 || true

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi
echo "CUDA_HOME=${CUDA_HOME:-unset}"

export PATH="$CONDA_BASE/bin:$PATH"
# Conda activate scripts may read unset MKL vars; avoid nounset crash here.
set +u
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export HF_HOME
export MODEL_ROOT
export SIGLIP_LOCAL
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
# Keep transformers and hub on the same cache tree for offline repo-id loads.
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_UPDATE_DOWNLOAD_COUNTS=0
export PYTHONPATH="$SNELLIUS_REPO_DIR/thinking-in-space:$SNELLIUS_REPO_DIR:${PYTHONPATH:-}"

export LMMS_EVAL_LAUNCHER=accelerate
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_PATH"

cd "$SNELLIUS_REPO_DIR"

if ! ls CUT3R/src/croco/models/curope/*.so >/dev/null 2>&1; then
  echo "[INFO] curope extension not found; building in-place..."
  (
    cd CUT3R/src/croco/models/curope
    python setup.py build_ext --inplace
  )
fi

python - <<'PY'
from huggingface_hub import snapshot_download
import os

model_root = os.environ.get("MODEL_ROOT", "")
siglip_local = os.environ.get("SIGLIP_LOCAL") or os.path.join(model_root, "siglip-so400m-patch14-384")
if not siglip_local or not os.path.isfile(os.path.join(siglip_local, "config.json")):
  raise SystemExit(f"[ERROR] SIGLIP_LOCAL missing config.json: {siglip_local}")
print(f"[CACHE OK] local siglip path -> {siglip_local}")

required = [
    ("google/siglip-so400m-patch14-384", "model"),
    ("nyu-visionx/VSI-Bench", "dataset"),
]
for repo_id, repo_type in required:
    path = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_files_only=True)
    print(f"[CACHE OK] {repo_type} {repo_id} -> {path}")
PY

echo "Running Snellius-parity evaluation (offline)"
accelerate launch \
  --num_processes="$NUM_PROCESSES" \
  -m lmms_eval \
  --model vlm_3r \
  --model_args "pretrained=$PRETRAINED_LOCAL,model_base=$MODEL_BASE_LOCAL,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_FRAMES_NUM" \
  --tasks "$TASK" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --log_samples_suffix "vlm_3r_7b_qwen2_lora_snellius_parity_offline" \
  --output_path "$OUTPUT_PATH"

echo "[DONE] Results saved to: $OUTPUT_PATH"
