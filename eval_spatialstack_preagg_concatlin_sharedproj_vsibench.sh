#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SS_PreAgg_Concat_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

export CUT3R_SPATIALSTACK_PREAGG_TYPE="${CUT3R_SPATIALSTACK_PREAGG_TYPE:-concat_linear}"
export CUT3R_SPATIALSTACK_PREAGG_PROJECTOR_SHARING="${CUT3R_SPATIALSTACK_PREAGG_PROJECTOR_SHARING:-shared}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SANITIZED_PREAGG_LAYERS="${CUT3R_SPATIALSTACK_PREAGG_LAYERS:-6,9,12}"
SANITIZED_PREAGG_LAYERS="${SANITIZED_PREAGG_LAYERS//,/_}"
SANITIZED_LLM_LAYERS="${CUT3R_SPATIALSTACK_LLM_LAYERS:-1,2,3}"
SANITIZED_LLM_LAYERS="${SANITIZED_LLM_LAYERS//,/_}"
export RUN_BASENAME="${RUN_BASENAME:-cut3r_spatialstack_preagg_concatlin_sharedproj_dec${SANITIZED_PREAGG_LAYERS}_llm${SANITIZED_LLM_LAYERS}}"

exec bash "$SCRIPT_DIR/eval_spatialstack_preagg_wsum_sharedproj_vsibench.sh"
