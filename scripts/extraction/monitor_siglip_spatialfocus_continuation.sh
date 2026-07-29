#!/usr/bin/env bash
# Submit this serial controller with --dependency=afterany:<8-node-job>.
# It only continues a cleanly-ended or time-limited extraction; explicit
# extraction failures remain visible for diagnosis instead of looping forever.
#SBATCH --job-name=SIGLIP_sf_controller
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/extraction/%x_%j.out
#SBATCH --error=logs/extraction/%x_%j.err

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs/extraction

: "${PREDECESSOR_JOB:?submit with --export=ALL,PREDECESSOR_JOB=<8-node-job>}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
INDEX="${FAST_DATA_ROOT}/siglip_features_dec_m2_training_index.json"

state="$(sacct -X -n -j "$PREDECESSOR_JOB" --format=State | awk 'NF {print $1; exit}')"
summary_path="${FAST_DATA_ROOT}/summaries/controller-${PREDECESSOR_JOB}.json"
conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py summarize \
  --manifest "$INDEX" --output-root "$FAST_DATA_ROOT" --run-id "controller-${PREDECESSOR_JOB}" >/dev/null

read -r completed expected < <(conda run -n "$CONDA_ENV_NAME" python - "$summary_path" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["completed_sample_count"], value["expected_sample_count"])
PY
)
echo "predecessor=${PREDECESSOR_JOB} state=${state} completed=${completed}/${expected}"

if [[ "$completed" == "$expected" ]]; then
  conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py verify-all \
    --manifest "$INDEX" --output-root "$FAST_DATA_ROOT" --run-id "verify-${PREDECESSOR_JOB}"
  exit 0
fi

case "$state" in
  COMPLETED|TIMEOUT)
    next_job="$(sbatch scripts/extraction/dbg_extract_siglip_spatialfocus_8node.sh | awk '{print $NF}')"
    next_controller="$(sbatch --dependency="afterany:${next_job}" --export="ALL,PREDECESSOR_JOB=${next_job}" "$REPO_DIR/scripts/extraction/monitor_siglip_spatialfocus_continuation.sh" | awk '{print $NF}')"
    echo "submitted_continuation=${next_job} controller=${next_controller}"
    ;;
  *)
    echo "Not continuing failed predecessor ${PREDECESSOR_JOB} (state=${state})." >&2
    exit 1
    ;;
esac
