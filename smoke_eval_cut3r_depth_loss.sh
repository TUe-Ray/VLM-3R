#!/bin/bash
#SBATCH --job-name=SMOKE_EvalCUT3RDepth
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

export RUN_NAME="${RUN_NAME:-SMOKE_cut3r_depth_loss_43817021}"
export LIMIT="${LIMIT:-20}"
export NOTE="${NOTE:-Smoke eval for the depth-only CUT3R checkpoint before official VSI-Bench eval.}"

exec bash eval_cut3r_depth_loss.sh
