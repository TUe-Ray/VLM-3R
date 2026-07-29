#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUFFIX="${SUFFIX:-cut3r_dualpath_a_global_frame_local_all}"
export MODEL_SPATIAL_ATTENTION_MODE=global
export MODEL_WRITEBACK_QUERY_SCOPE=all_tokens
export MODEL_WRITEBACK_VISIBILITY=frame_local
exec bash "$SCRIPT_DIR/train_cut3r_dualpath.sh"
