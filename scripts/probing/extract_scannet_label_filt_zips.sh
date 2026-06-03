#!/bin/bash
set -euo pipefail

ROOT="${1:?Usage: $0 <scannet_label_root> [parallelism]}"
PARALLELISM="${2:-8}"

extract_one() {
  local zip_path="$1"
  local scene_dir
  scene_dir="$(dirname "$zip_path")"
  if [[ -d "$scene_dir/label-filt" ]]; then
    echo "[SKIP] $(basename "$scene_dir")"
    return 0
  fi
  echo "[UNZIP] $(basename "$scene_dir")"
  unzip -q -n "$zip_path" -d "$scene_dir"
}

export -f extract_one
find "$ROOT/scans" -mindepth 2 -maxdepth 2 -name '*_2d-label-filt.zip' -print0 \
  | xargs -0 -r -n 1 -P "$PARALLELISM" bash -c 'extract_one "$@"' _
