#!/bin/bash
# Download ScanNet v2 2D filtered semantic label zips for a scene list.

set -euo pipefail

SCENE_LIST="${1:?Usage: $0 <scene_list.txt> <output_root> [parallelism]}"
OUT_ROOT="${2:?Usage: $0 <scene_list.txt> <output_root> [parallelism]}"
PARALLELISM="${3:-8}"
BASE_URL="${BASE_URL:-http://kaldir.vc.cit.tum.de/scannet/v2/scans}"

mkdir -p "$OUT_ROOT/scans"

download_one() {
  local scene="$1"
  local scene_dir="$OUT_ROOT/scans/$scene"
  local zip_path="$scene_dir/${scene}_2d-label-filt.zip"
  local tmp_path="$zip_path.tmp.$$"
  local url="$BASE_URL/$scene/${scene}_2d-label-filt.zip"

  mkdir -p "$scene_dir"
  if [[ -s "$zip_path" ]]; then
    echo "[SKIP] $scene"
    return 0
  fi
  echo "[GET] $scene"
  rm -f "$tmp_path"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 --show-error \
    -o "$tmp_path" "$url"
  mv "$tmp_path" "$zip_path"
}

export OUT_ROOT BASE_URL
export -f download_one

xargs -a "$SCENE_LIST" -r -n 1 -P "$PARALLELISM" bash -c 'download_one "$@"' _
