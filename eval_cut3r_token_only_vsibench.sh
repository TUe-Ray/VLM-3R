#!/usr/bin/env bash
#SBATCH --job-name=SMOKE_CUT3RTokenOnly_EVAL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err

set -euo pipefail
cd "${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
: "${CHECKPOINT:?Set CHECKPOINT to a CUT3R-token-only checkpoint after its smoke checkpoint reload gate passes.}"
: "${PARITY_SIDECAR:?Set PARITY_SIDECAR.}"
: "${PARITY_RECOMPUTED:?Set PARITY_RECOMPUTED on the exact sampled frames.}"
python scripts/diagnose_cut3r_token_sidecar_parity.py --sidecar "$PARITY_SIDECAR" --recomputed "$PARITY_RECOMPUTED"
echo "[CUT3R_TOKEN_ONLY] Evaluation entry point is intentionally gated on a checkpoint reload."
echo "Use CHECKPOINT=$CHECKPOINT with the VSI evaluator once its dataset adapter is configured to load spatial_features sidecars."
