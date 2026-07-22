#!/bin/bash
#SBATCH --account=EUHPC_D32_006
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --job-name=cut3r_gauge_translation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err

set -euo pipefail
module purge
module load profile/deeplrn
module load cuda/12.1
REPO=${REPO:-/leonardo/home/userexternal/shuang00/VLM-3R}
cd "$REPO"
: "${SMOKE_ACCEPTANCE:?Set SMOKE_ACCEPTANCE to a passing smoke_acceptance.json artifact}"
python -c 'import json,sys; payload=json.load(open(sys.argv[1])); assert payload.get("passed") is True, "smoke acceptance did not pass"' "$SMOKE_ACCEPTANCE"
srun --kill-on-bad-exit=1 --wait=30 conda run --no-capture-output -n vlm3r torchrun \
  --standalone --nproc-per-node=4 scripts/gauge_translation/train_cut3r_gauge_translation.py \
  --config configs/gauge_translation/official.json ${RESUME_FROM:+--resume-from "$RESUME_FROM"}
