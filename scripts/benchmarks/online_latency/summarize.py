#!/usr/bin/env python3
"""Consolidate strict worker artifacts and refuse invalid speedup reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmarks.online_latency.common import csv_write, json_dump, summarize

MODES = ("geometry_off", "online_spatialstack", "online_predictor")
TOTAL = "fixed_16_token_total_ms"


def load_worker(path: Path):
    data = json.loads((path / "worker_summary.json").read_text())
    provenance = json.loads((path / "runtime_provenance.json").read_text())
    return data, provenance


def assert_contract(workers):
    errors = []
    reference = None
    for mode, payload in workers.items():
        for record in payload["records"]:
            counts = record.get("assertions", {})
            if any(int(counts.get(key, 0)) != 0 for key in (
                "siglip_feature_cache_reads", "spatialstack_sidecar_reads",
                "projected_visual_token_cache_reads", "residual_tensor_cache_reads",
            )):
                errors.append(f"{mode}:{record['ordinal']}: cache/sidecar read")
            if int(counts.get("siglip_forward_count", 0)) != 1:
                errors.append(f"{mode}:{record['ordinal']}: SigLIP forward count")
            key = (record["ordinal"], record["canonical_key"], tuple(record["frame_ids"]), record["raw_video_path"], record["prompt"])
            if reference is None:
                reference = {}
            prior = reference.setdefault(record["ordinal"], key)
            if prior != key:
                errors.append(f"{mode}:{record['ordinal']}: input parity mismatch")
            if mode == "online_spatialstack" and not (counts.get("cut3r_loaded") and int(counts.get("cut3r_forward_count", 0)) >= 1):
                errors.append(f"{mode}:{record['ordinal']}: CUT3R online contract")
            if mode == "online_predictor" and (counts.get("cut3r_loaded") or int(counts.get("cut3r_forward_count", 0)) or int(counts.get("predictor_forward_count", 0)) != 1):
                errors.append(f"{mode}:{record['ordinal']}: predictor online contract")
    if errors:
        raise RuntimeError("No speedup may be reported; assertions failed: " + "; ".join(errors))


def measured(records):
    return [record for record in records if record.get("split") == "measured"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Contains one directory per mode")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="concurrent")
    args = parser.parse_args()
    root, output = Path(args.input_root), Path(args.output_root)
    workers, provenance = {}, {}
    for mode in MODES:
        workers[mode], provenance[mode] = load_worker(root / mode)
    assert_contract(workers)
    summary = {"label": args.label, "all_assertions_passed": True, "modes": {}, "comparisons": {}}
    all_rows, stage_rows = [], []
    stages = ["video_decode_frame_sampling_ms", "image_preprocess_ms", "siglip_forward_ms", "cut3r_forward_ms",
              "predictor_forward_ms", "spatialstack_projection_ms", "residual_construction_ms",
              "multimodal_prepare_and_qwen_prefill_ms", "qwen_prefill_ms", "token_decode_ms", TOTAL,
              "spatial_branch_ms"]
    paired = {}
    for mode, worker in workers.items():
        values = measured(worker["records"])
        by_ordinal = {row["ordinal"]: row for row in values}
        paired[mode] = by_ordinal
        mode_summary = {"cold_start_model_loading_ms": worker["cold_start_model_loading_ms"], "stages": {}}
        for stage in stages:
            mode_summary["stages"][stage] = summarize([row.get(stage, 0.0) for row in values])
            stage_rows.append({"mode": mode, "stage": stage, **mode_summary["stages"][stage]})
        mode_summary["peak_cuda_allocated_bytes"] = max(row.get("peak_cuda_allocated_bytes", 0) for row in values)
        mode_summary["peak_cuda_reserved_bytes"] = max(row.get("peak_cuda_reserved_bytes", 0) for row in values)
        total_sum = sum(row[TOTAL] for row in values)
        mode_summary["effective_samples_per_second"] = len(values) * 1000.0 / total_sum
        mode_summary["effective_frames_per_second"] = 16.0 * mode_summary["effective_samples_per_second"]
        summary["modes"][mode] = mode_summary
        all_rows.extend([{**row, "mode": mode, "phase": args.label} for row in values])
    ss, pred, geo = paired["online_spatialstack"], paired["online_predictor"], paired["geometry_off"]
    ordinal_set = set(ss) & set(pred) & set(geo)
    expected_pairs = 16 if args.label == "concurrent" else 4
    if len(ordinal_set) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} paired measured samples, got {len(ordinal_set)}")
    ss_total = [ss[i][TOTAL] for i in sorted(ordinal_set)]
    pred_total = [pred[i][TOTAL] for i in sorted(ordinal_set)]
    geo_total = [geo[i][TOTAL] for i in sorted(ordinal_set)]
    ss_branch = [ss[i]["spatial_branch_ms"] for i in sorted(ordinal_set)]
    pred_branch = [pred[i]["spatial_branch_ms"] for i in sorted(ordinal_set)]
    import statistics
    median_ss, median_pred, median_geo = map(statistics.median, (ss_total, pred_total, geo_total))
    ss_alloc = summary["modes"]["online_spatialstack"]["peak_cuda_allocated_bytes"]
    pred_alloc = summary["modes"]["online_predictor"]["peak_cuda_allocated_bytes"]
    summary["comparisons"] = {
        "paired_samples": len(ordinal_set), "total_speedup_spatialstack_over_predictor": median_ss / median_pred,
        "latency_reduction_percent": 100.0 * (1.0 - median_pred / median_ss),
        "spatial_branch_speedup": statistics.median(ss_branch) / statistics.median(pred_branch),
        "peak_allocated_memory_reduction_bytes": ss_alloc - pred_alloc,
        "peak_allocated_memory_reduction_percent": 100.0 * (1.0 - pred_alloc / ss_alloc),
        "overhead_recovery_relative_to_geometry_off_percent": 100.0 * (median_ss - median_pred) / (median_ss - median_geo),
        "note": "p95/bootstrap CI are exploratory at n=16",
    }
    output.mkdir(parents=True, exist_ok=True)
    json_dump(summary, output / "latency_summary.json")
    json_dump(provenance, output / "runtime_provenance.json")
    json_dump({"schema_version": 1, "workers": workers}, output / "worker_artifacts.json")
    csv_write([{ "mode": mode, **summary["modes"][mode]["stages"][TOTAL],
                 "peak_cuda_allocated_bytes": summary["modes"][mode]["peak_cuda_allocated_bytes"],
                 "peak_cuda_reserved_bytes": summary["modes"][mode]["peak_cuda_reserved_bytes"],
                 "effective_samples_per_second": summary["modes"][mode]["effective_samples_per_second"],
                 "effective_frames_per_second": summary["modes"][mode]["effective_frames_per_second"]} for mode in MODES], output / "latency_summary.csv")
    csv_write(stage_rows, output / "stage_breakdown.csv")
    with (output / "per_sample_latency.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    comp = summary["comparisons"]
    (output / "latency_report.md").write_text(
        "# Real-online latency benchmark\n\n"
        f"All raw-video/no-cache assertions passed for {comp['paired_samples']} paired samples.\n\n"
        f"- SpatialStack / predictor total speedup: {comp['total_speedup_spatialstack_over_predictor']:.3f}×\n"
        f"- Latency reduction: {comp['latency_reduction_percent']:.2f}%\n"
        f"- Spatial branch speedup: {comp['spatial_branch_speedup']:.3f}×\n"
        f"- Peak allocated-memory reduction: {comp['peak_allocated_memory_reduction_percent']:.2f}%\n"
        f"- Overhead recovery relative to geometry-off: {comp['overhead_recovery_relative_to_geometry_off_percent']:.2f}%\n\n"
        "p95 and bootstrap CIs are included in JSON/CSV as exploratory statistics for n=16.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
