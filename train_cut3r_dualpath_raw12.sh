#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUFFIX="${SUFFIX:-cut3r_dualpath_raw12_control}"
export MODEL_DUAL_PATH_RAW_LAYER12_CONTROL=True
exec bash "$SCRIPT_DIR/train_cut3r_dualpath.sh"
