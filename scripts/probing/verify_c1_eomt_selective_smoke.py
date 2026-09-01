#!/usr/bin/env python
"""Fail closed on the C1 pre-SFT EoMT-selective one-video smoke contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-label", default="c1_vlm3r_eomt_selective")
    parser.add_argument("--paired-calibration-summary", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    provenance_path = root / "features" / args.model_label / "extraction_provenance.json"
    provenance = load_json(provenance_path)
    paired = load_json(Path(args.paired_calibration_summary))
    settings = provenance.get("eomt_selective_settings") or {}
    samples = provenance.get("extraction_samples") or []
    require(provenance.get("model_loading_mode") == "pre_sft_fusion", "Smoke is not pre-SFT fusion")
    require(provenance.get("experiment_variant") == "c1_vlm3r", "Smoke is not C1 VLM3R")
    require(provenance.get("eomt_selective_kv_gate") is True, "Selective K/V gate is disabled")
    require(provenance.get("c1_artifact_architecture") == "vlm3r", "Wrong C1 artifact architecture")
    require(settings.get("mm_eomt_selective_3d_enable") is True, "Selective setting was not enabled")
    require(settings.get("mm_eomt_selective_3d_word_match_enable") is True, "Checkpoint word-match setting was not retained")
    require(settings.get("mm_eomt_selective_3d_empty_fallback") == "zero_3d", "zero_3d fallback is not configured")
    require(len(samples) == 1, f"Expected one smoke video, found {len(samples)}")
    sample = samples[0]
    runtime = sample.get("first_video_runtime_assertions") or {}
    require(runtime.get("assessment") == "PASS", "First-forward runtime assertion did not pass")
    require(runtime.get("ordinary_visual_tokens") == 32 * 196, "Ordinary visual token count is not 32x196")
    require(runtime.get("camera_tokens_ungated") is True, "Camera tokens were not attested ungated")
    require(runtime.get("eomt_selective_gate_active") is True, "Selective gate was not active")
    require(runtime.get("no_words_available") is True, "Expected no word metadata in probe forward")
    require(runtime.get("word_match_applied") is False, "Word match unexpectedly ran without word metadata")
    require(runtime.get("word_match_effective_noop") is True, "No-word word-match no-op was not attested")
    debug = sample.get("eomt_selective_debug") or []
    require(len(debug) == 32, "Expected one existing-gate metadata record per forward frame")
    for item in debug:
        require(item.get("camera_tokens_ungated") is True, "Camera token gate invariant failed")
        require(item.get("no_words_available") is True, "Unexpected word metadata")
        require(item.get("word_match_applied") is False, "Word match unexpectedly changed selector path")
        require(item.get("word_match_effective_noop") is True, "Word-match altered a no-word selection")

    frozen = paired.get("frozen_c1") or {}
    conditions = paired.get("conditions") or {}
    require(frozen.get("artifact_sha256") == provenance.get("c1_calibration_sha256"), "Smoke and paired diagnostic use different C1 artifacts")
    require(frozen.get("lambda_recalibrated") is False, "Paired diagnostic recalibrated lambda")
    require(conditions.get("only_condition_difference") == "mm_eomt_selective_3d_enable", "Paired diagnostic is not isolated")
    report = {
        "assessment": "PASS",
        "architecture": "C1 VLM3R + EoMT soft thing-mask CUT3R patch K/V gate",
        "ordinary_visual_tokens": 32 * 196,
        "camera_tokens_ungated": True,
        "gate_active_only_in_selective_condition": True,
        "no_words_available": True,
        "word_match_applied": False,
        "word_match_effective_noop": True,
        "c1_calibration_sha256": provenance.get("c1_calibration_sha256"),
        "lambda_artifact": provenance.get("eomt_lambda_artifact"),
        "lambda_runtime": provenance.get("eomt_lambda_runtime"),
        "eomt_selector_settings": settings,
        "paired_calibration_summary": str(Path(args.paired_calibration_summary).resolve()),
        "extraction_provenance": str(provenance_path.resolve()),
    }
    output = Path(args.output) if args.output else root / "c1_eomt_selective_smoke_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
