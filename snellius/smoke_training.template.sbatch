#!/usr/bin/env bash
# TEMPLATE ONLY. Submit after validate_migration.sh and smoke_environment.sbatch pass.
# It uses one A100 node and <= 1 hour, but intentionally does not encode a
# final SFT command until the Leonardo manifest confirms model, sidecar, and
# data arguments.
#SBATCH --job-name=vlm3r-training-smoke
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mem=0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "$SCRIPT_DIR/paths.env"

mkdir -p "$VLM3R_LOG_ROOT"
LOG_FILE="$VLM3R_LOG_ROOT/smoke_training_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

"$SCRIPT_DIR/validate_migration.sh"

module purge
module load 2023
module load Miniconda3/23.5.2-0
MINICONDA_ROOT=/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0
# shellcheck disable=SC1091
source "$MINICONDA_ROOT/etc/profile.d/conda.sh"
conda activate vlm3r-snellius

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE

# Render the YAML before use; the template deliberately contains ${DATA_ROOT}.
DATA_YAML="$REPO_DIR/snellius/vsibench_data.snellius.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
    printf '%s\n' "Refusing to start: render $REPO_DIR/snellius/vsibench_data.snellius.yaml.template first." >&2
    exit 2
fi

# Replace these after the migration manifest identifies the exact proven
# Leonardo command. Retain the basic one-node torchrun architecture used by
# scripts/VLM_3R/train_vsibench.sh and the current CUT3R wrappers.
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-$REPO_DIR/llava/train/train_mem.py}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$TRAIN_SAVE_ROOT/SMOKE_RUN_NAME_REQUIRED}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-REPLACE_FROM_MANIFEST}"

if [[ "$SMOKE_OUTPUT_DIR" == *SMOKE_RUN_NAME_REQUIRED || "$SMOKE_RUN_NAME" == REPLACE_FROM_MANIFEST ]]; then
    printf '%s\n' 'Refusing to start: set SMOKE_OUTPUT_DIR and SMOKE_RUN_NAME from the migration manifest.' >&2
    exit 2
fi

# Required arguments still depend on the manifest. At minimum it must select
# VLM-3R baseline or SpatialStack and exactly one of dec6, dec9, or dec12;
# do not mix ablation variables in this first smoke run.
printf '%s\n' 'Refusing to start: add the manifest-verified model/data/sidecar arguments below before submitting.' >&2
exit 2

# Example shape only; keep disabled until the exact command is approved:
# torchrun --standalone --nproc_per_node=4 "$TRAIN_ENTRYPOINT" \
#   --model_name_or_path "$LOCAL_MODEL_BASE" \
#   --data_path "$DATA_YAML" \
#   --image_folder "$VSI_BENCH_MEDIA_ROOT" \
#   --output_dir "$SMOKE_OUTPUT_DIR" \
#   --run_name "$SMOKE_RUN_NAME"
