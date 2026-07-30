#!/usr/bin/env bash
# Run inside an interactive 4-GPU Leonardo allocation.  Does not submit Slurm.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_ID="${RUN_ID:-online_latency_$(date -u +%Y%m%dT%H%M%SZ)_$(git -C "$REPO_DIR" rev-parse --short HEAD)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/online_latency/$RUN_ID}"
MANIFEST="$OUTPUT_ROOT/latency_manifest.json"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" scripts/benchmarks/online_latency/build_manifest.py --output "$MANIFEST"

run_mode() {
  local gpu="$1" mode="$2" phase="$3" warmups="$4" measured="$5" reps="$6"
  CUDA_VISIBLE_DEVICES="$gpu" SPATIAL_FEATURES_ROOT=/dev/null \
    "$PYTHON_BIN" scripts/benchmarks/online_latency/worker.py \
      --mode "$mode" --manifest "$MANIFEST" --output-dir "$OUTPUT_ROOT/$phase/$mode" \
      --warmups "$warmups" --measured "$measured" --repetitions "$reps" \
      >"$LOG_ROOT/${phase}_${mode}.out" 2>"$LOG_ROOT/${phase}_${mode}.err"
}

phase_concurrent() {
  local phase="$1" warmups="$2" measured="$3" reps="$4"
  run_mode 0 geometry_off "$phase" "$warmups" "$measured" "$reps" & p0=$!
  run_mode 1 online_spatialstack "$phase" "$warmups" "$measured" "$reps" & p1=$!
  run_mode 2 online_predictor "$phase" "$warmups" "$measured" "$reps" & p2=$!
  set +e
  wait "$p0"; e0=$?
  wait "$p1"; e1=$?
  wait "$p2"; e2=$?
  set -e
  printf '{"geometry_off":%d,"online_spatialstack":%d,"online_predictor":%d}\n' "$e0" "$e1" "$e2" >"$OUTPUT_ROOT/$phase/exit_codes.json"
  (( e0 == 0 && e1 == 0 && e2 == 0 )) || return 1
  "$PYTHON_BIN" scripts/benchmarks/online_latency/summarize.py --input-root "$OUTPUT_ROOT/$phase" --output-root "$OUTPUT_ROOT/$phase" --label "$phase"
}

phase_concurrent smoke 2 4 1
phase_concurrent concurrent 4 16 3

# Fresh sequential processes on physical GPU 0; use the same first four measured samples.
run_mode 0 online_spatialstack sequential 2 4 3
run_mode 0 online_predictor sequential 2 4 3
# The sequential check compares the two spatial branches only; reuse geometry-off
# from the concurrent phase so the strict consolidator can retain three-way input
# parity checks without launching geometry-off a second time.
cp -a "$OUTPUT_ROOT/concurrent/geometry_off" "$OUTPUT_ROOT/sequential/geometry_off"
"$PYTHON_BIN" scripts/benchmarks/online_latency/summarize.py --input-root "$OUTPUT_ROOT/sequential" --output-root "$OUTPUT_ROOT/sequential" --label sequential

# Required top-level canonical artifacts are the concurrent measured artifacts.
for name in latency_manifest.json per_sample_latency.jsonl latency_summary.json latency_summary.csv stage_breakdown.csv runtime_provenance.json latency_report.md; do
  if [[ "$name" == "latency_manifest.json" ]]; then continue; fi
  cp "$OUTPUT_ROOT/concurrent/$name" "$OUTPUT_ROOT/$name"
done
printf '%s\n' "$OUTPUT_ROOT" >"$OUTPUT_ROOT/RESULT_ROOT.txt"
