#!/usr/bin/env python
"""Aggregate official C1 full-K/V versus EoMT-selective depth-probe results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


LAYERS = [0, 1, 2, 3, 6, 9, 15, 21, 27]
EXPECTED_VAL_TOKENS = 75656
EXPECTED_VIDEOS = 1199
EXPECTED_FORWARD_FRAMES = EXPECTED_VIDEOS * 32


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def distribution(values: list[float]) -> dict[str, float | int]:
    require(values, "Cannot summarize an empty distribution")
    return {
        "count": len(values), "min": min(values), "max": max(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "std": statistics.pstdev(values),
    }


def read_metrics(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    require(isinstance(rows, list), f"Expected result list: {path}")
    result = {str(row["feature_level"]): row for row in rows}
    require(set(result) == {f"layer_{layer}" for layer in LAYERS}, f"Unexpected feature levels in {path}: {sorted(result)}")
    for level, row in result.items():
        require(int(row.get("num_tokens", -1)) == EXPECTED_VAL_TOKENS, f"{level} does not have official validation tokens")
        for key in ("mae", "absrel", "delta125"):
            require(math.isfinite(float(row[key])), f"Non-finite {key} for {level}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selective-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="c1_vlm3r_eomt_selective")
    parser.add_argument("--baseline-results", default="/home/shaoruei/probe_cache/c1_vlm3r_v1/full/c1_vlm3r_depth_probe_results.json")
    parser.add_argument("--baseline-provenance", default="/home/shaoruei/probe_cache/c1_vlm3r_v1/full/features/c1_vlm3r/extraction_provenance.json")
    args = parser.parse_args()

    selective_root = Path(args.selective_root)
    baseline_provenance = load_json(Path(args.baseline_provenance))
    selective_provenance_path = selective_root / "features" / args.model_label / "extraction_provenance.json"
    selective_provenance = load_json(selective_provenance_path)
    require(baseline_provenance.get("experiment_variant") == "c1_vlm3r", "Baseline is not C1 VLM3R")
    require(selective_provenance.get("experiment_variant") == "c1_vlm3r", "Selective run is not C1 VLM3R")
    require(selective_provenance.get("eomt_selective_kv_gate") is True, "Selective gate is not enabled")
    require(selective_provenance.get("c1_calibration_sha256") == baseline_provenance.get("c1_calibration_sha256"), "C1 artifacts differ")
    require(selective_provenance.get("sample_indices_sha256") == baseline_provenance.get("sample_indices_sha256"), "Probe split differs")
    require(selective_provenance.get("eomt_lambda_artifact") is not None, "Frozen lambda is missing")
    settings = selective_provenance.get("eomt_selective_settings") or {}
    require(settings.get("mm_eomt_selective_3d_enable") is True, "Gate setting is disabled")
    require(settings.get("mm_eomt_selective_3d_word_match_enable") is True, "Word-match setting is missing")
    require(settings.get("mm_eomt_selective_3d_empty_fallback") == "zero_3d", "Wrong fallback")

    baseline = read_metrics(Path(args.baseline_results))
    selective_paths = [selective_root / "probes" / args.model_label / f"layer_{layer}" / "metrics.json" for layer in LAYERS]
    for path in selective_paths:
        require(path.is_file(), f"Missing selective probe metric: {path}")
    selective = {f"layer_{layer}": load_json(path) for layer, path in zip(LAYERS, selective_paths)}
    for level, row in selective.items():
        require(int(row.get("num_tokens", -1)) == EXPECTED_VAL_TOKENS, f"{level} does not have official validation tokens")

    samples = selective_provenance.get("extraction_samples") or []
    require(len(samples) == EXPECTED_VIDEOS, f"Expected {EXPECTED_VIDEOS} extracted videos, got {len(samples)}")
    debug = [item for sample in samples for item in (sample.get("eomt_selective_debug") or [])]
    require(len(debug) == EXPECTED_FORWARD_FRAMES, f"Expected {EXPECTED_FORWARD_FRAMES} gate frames, got {len(debug)}")
    require(all(item.get("no_words_available") is True for item in debug), "Word metadata unexpectedly appeared")
    require(all(item.get("word_match_applied") is False for item in debug), "Word match unexpectedly ran")
    require(all(item.get("word_match_effective_noop") is True for item in debug), "No-word word-match no-op failed")
    require(all(item.get("camera_tokens_ungated") is True for item in debug), "Camera tokens were not consistently ungated")
    gate_stats = {
        "forward_frames": len(debug),
        "mean_gate_value": distribution([float(item["gate_mean"]) for item in debug]),
        "active_patch_fraction": distribution([float(item["active_patch_fraction"]) for item in debug]),
        "selected_queries_per_frame": distribution([float(item["selected_queries"]) for item in debug]),
        "zero3d_fallback_frame_rate": sum(item.get("fallback") == "zero_3d" for item in debug) / len(debug),
        "no_words_available_frame_rate": 1.0,
        "word_match_applied_frame_rate": 0.0,
        "word_match_effective_noop_frame_rate": 1.0,
        "camera_tokens_ungated_frame_rate": 1.0,
    }

    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        level = f"layer_{layer}"
        full, gated = baseline[level], selective[level]
        rows.append({
            "layer": layer,
            "full_mae": full["mae"], "selective_mae": gated["mae"], "delta_mae": gated["mae"] - full["mae"],
            "full_absrel": full["absrel"], "selective_absrel": gated["absrel"], "delta_absrel": gated["absrel"] - full["absrel"],
            "full_delta125": full["delta125"], "selective_delta125": gated["delta125"], "delta_delta125": gated["delta125"] - full["delta125"],
            "num_tokens": EXPECTED_VAL_TOKENS,
        })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "c1_eomt_selective_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    markdown = ["# C1 full K/V vs EoMT-selective depth probe", "", "| Layer | Full MAE | Selective MAE | Full AbsRel | Selective AbsRel | Full δ<1.25 | Selective δ<1.25 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    markdown += [f"| L{row['layer']} | {row['full_mae']:.6f} | {row['selective_mae']:.6f} | {row['full_absrel']:.6f} | {row['selective_absrel']:.6f} | {row['full_delta125']:.6f} | {row['selective_delta125']:.6f} |" for row in rows]
    (output / "c1_eomt_selective_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "c1_eomt_selective_depth_probe_v1", "architecture": "P_gated=M_EoMT*P_CUT3R; DeltaH=CrossAttention(H,[camera_token,P_gated]); H'=H+DeltaH",
        "eomt_selector_settings": settings, "lambda_used": selective_provenance.get("eomt_lambda_artifact"),
        "lambda_runtime": selective_provenance.get("eomt_lambda_runtime"), "lambda_recalibrated": False,
        "c1_artifact": selective_provenance.get("c1_calibration_json"), "c1_artifact_sha256": selective_provenance.get("c1_calibration_sha256"),
        "eomt_cache_path": selective_provenance.get("eomt_consumer_cache_root"), "eomt_cache_validation_sha256": selective_provenance.get("eomt_cache_validation_sha256"),
        "layer_list": LAYERS, "sample_indices": selective_provenance.get("sample_indices"), "sample_indices_sha256": selective_provenance.get("sample_indices_sha256"),
        "official_validation_tokens_per_layer": EXPECTED_VAL_TOKENS, "gate_statistics": gate_stats,
        "baseline_results": str(Path(args.baseline_results).resolve()), "baseline_provenance": str(Path(args.baseline_provenance).resolve()),
        "selective_provenance": str(selective_provenance_path.resolve()),
    }
    (output / "c1_eomt_selective_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "c1_eomt_selective_comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(markdown))
    print(json.dumps({"gate_statistics": gate_stats, "metadata": str(output / "c1_eomt_selective_metadata.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
