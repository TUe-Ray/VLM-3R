#!/usr/bin/env bash
#SBATCH --job-name=controlled_D_add_d12_l0
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclusive
set -euo pipefail
export CONTROLLED_FUSION_ID=D
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/train_controlled_fusion_variant.sh"
