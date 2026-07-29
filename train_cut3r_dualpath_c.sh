#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUFFIX="${SUFFIX:-cut3r_dualpath_c_global_global_all}"
export MODEL_SPATIAL_ATTENTION_MODE=global
export MODEL_WRITEBACK_QUERY_SCOPE=all_tokens
export MODEL_WRITEBACK_VISIBILITY=global
exec bash "$SCRIPT_DIR/train_cut3r_dualpath.sh"
