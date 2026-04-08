#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use Python directly from vlm3rEOMT environment
PYTHON_EXEC="/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3rEOMT/bin/python"

OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_ROOT/logs/validate/eomt_pool_validation_report.json}"

HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

cd "$PROJECT_ROOT"

module purge || true
module load cuda/12.1 || true
module load cudnn || true
module load profile/deeplrn || true

if [[ -v LD_LIBRARY_PATH && -n "$LD_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3rEOMT/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3rEOMT/lib"
fi

export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

$PYTHON_EXEC scripts/validate_eomt_pool_alignment_standalone.py --output_json "$OUTPUT_JSON"

echo "Validation report saved to: $OUTPUT_JSON"
