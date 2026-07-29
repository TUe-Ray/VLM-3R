#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUFFIX="${SUFFIX:-cut3r_dualpath_d_global_global_text}"
export MODEL_SPATIAL_ATTENTION_MODE=global
export MODEL_WRITEBACK_QUERY_SCOPE=text_only
export MODEL_WRITEBACK_VISIBILITY=global
exec bash "$SCRIPT_DIR/train_cut3r_dualpath.sh"
