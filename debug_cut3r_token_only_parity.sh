#!/usr/bin/env bash
#SBATCH --job-name=DBG_CUT3RTokenOnly_Parity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
LOCAL_SIGLIP="${LOCAL_SIGLIP:-$MODEL_ROOT/siglip-so400m-patch14-384}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DATA_ROOT="${DATA_ROOT:-$FAST_DATA_ROOT}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"
DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data.yaml}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/diagnostics/cut3r_token_only/debug_parity_${SLURM_JOB_ID}}"

cd "$REPO_DIR"
mkdir -p logs/cut3r_token_only "$OUTPUT_DIR"

module load cuda/12.1
module load cudnn
module load profile/deeplrn
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
export HF_HOME="/leonardo_scratch/fast/EUHPC_D32_006/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m py_compile \
  llava/model/llava_arch.py \
  llava/model/cut3r_token_only.py \
  llava/train/train.py \
  llava/train/llava_trainer.py \
  scripts/diagnose_cut3r_token_sidecar_parity.py \
  scripts/validate_cut3r_token_only_smoke_gate.py \
  thinking-in-space/lmms_eval/models/vlm_3r.py \
  scripts/compare_cut3r_token_only_wrappers.py
python -c "import llava.model.llava_arch"
python -c "import llava.train.llava_trainer"
python -m unittest discover -s tests -p 'test_cut3r_token_only*.py'

srun --kill-on-bad-exit=1 --wait=30 --ntasks=1 --gpus=1 \
  python scripts/diagnose_cut3r_token_sidecar_parity.py \
    --data-yaml "$DATA_PATH_YAML" \
    --data-root "$DATA_ROOT" \
    --spatial-features-root "$SPATIAL_FEATURES_ROOT" \
    --spatial-features-subdir "$SPATIAL_FEATURES_SUBDIR" \
    --cut3r-weights-path "$CUT3R_WEIGHTS" \
    --processor-config-path "$LOCAL_SIGLIP/preprocessor_config.json" \
    --frames-upbound 32 \
    --video-fps 1 \
    --precision bf16 \
    --num-samples 3 \
    --output-dir "$OUTPUT_DIR"
