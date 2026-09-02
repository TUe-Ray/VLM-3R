#!/usr/bin/env python
"""Fail closed on a one-video C1 GeoRoPE pre-SFT smoke contract."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--architecture", required=True, choices=("geo_rope_fusion", "visual_geo_rope"))
    parser.add_argument("--activation-json", required=True)
    parser.add_argument("--c1-reference-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    provenance_path = root / "features" / args.model_label / "extraction_provenance.json"
    provenance = load_json(provenance_path)
    activation = load_json(Path(args.activation_json))
    reference = load_json(Path(args.c1_reference_json))
    expected_runtime_architecture = "geo_rope_fusion" if args.architecture == "geo_rope_fusion" else "visual_3d_rope"
    expected_variant = "c1_geo_rope_fusion" if args.architecture == "geo_rope_fusion" else "c1_visual_geo_rope"
    expected_inputs = (
        {"spatial_features": True, "point_maps": False, "geometry_spatial_features": True}
        if args.architecture == "geo_rope_fusion"
        else {"spatial_features": False, "point_maps": True, "geometry_spatial_features": True}
    )

    require(provenance.get("model_loading_mode") == "pre_sft_fusion", "Smoke is not a pre-SFT fusion run")
    require(provenance.get("experiment_variant") == expected_variant, "Wrong C1 architecture variant")
    require(provenance.get("active_geometry_architecture") == expected_runtime_architecture, "Wrong active geometry architecture")
    require(provenance.get("c1_artifact_architecture") == "vlm3r", "Dense C1 source is not VLM3R")
    require(provenance.get("geometry_c1_calibration_sha256"), "Missing solved geometry activation artifact")
    require(activation.get("architecture") == args.architecture, "Activation artifact architecture mismatch")
    require(activation.get("reference_c1_artifact_sha256") == provenance.get("c1_calibration_sha256"), "Dense C1 artifact changed")
    require(math.isclose(float(activation.get("r0")), float(reference.get("r0")), rel_tol=0.0, abs_tol=1e-12), "r0 changed")
    achieved = activation.get("achieved_ratio") or {}
    require(abs(float(achieved.get("median")) - float(reference["r0"])) <= 2e-4, "Calibration missed r0")

    samples = provenance.get("extraction_samples") or []
    require(len(samples) == 1, f"Expected exactly one smoke sample, got {len(samples)}")
    sample = samples[0]
    runtime = sample.get("first_video_runtime_assertions") or {}
    require(runtime.get("assessment") == "PASS", "First-forward runtime contract did not pass")
    require(runtime.get("ordinary_visual_tokens") == 32 * 196, "Ordinary visual token count is not 32x196")
    observed_inputs = sample.get("model_forward_inputs") or {}
    for key, expected in expected_inputs.items():
        require(observed_inputs.get(key) is expected, f"Forward input {key} expected {expected}, got {observed_inputs.get(key)}")
    geometry_shape = sample.get("geometry_point_map_shape")
    require(isinstance(geometry_shape, list) and geometry_shape[0] == 32, "Full 32-frame geometry was not consumed")
    runtime_activation = sample.get("geometry_activation_runtime") or {}
    if args.architecture == "geo_rope_fusion":
        require(runtime_activation.get("c1_enabled") is True, "GeoRoPE residual C1 branch is disabled")
        require(math.isclose(float(runtime_activation.get("geo_rope_gate_q")), 1.0, abs_tol=2e-3), "GeoRoPE Q gate is not active")
        require(math.isclose(float(runtime_activation.get("geo_rope_gate_k")), 1.0, abs_tol=2e-3), "GeoRoPE K gate is not active")
        require(math.isclose(float(runtime_activation.get("c1_residual_gain")), float(activation["activation"]["lambda_geo"]), abs_tol=2e-3), "Runtime lambda_geo differs from solved artifact")
        parity = (activation.get("diagnostics") or {}).get("parity") or {}
        require(int((parity.get("projected_delta_rms_error") or {}).get("count", 0)) > 0, "Missing g=0 parity diagnostic")
    else:
        require(runtime_activation.get("shared_gamma") is True, "Visual GeoRoPE gates are not tied")
        gamma = float(activation["activation"]["gamma_c1"])
        require(math.isclose(float(runtime_activation.get("gamma_attn")), gamma, abs_tol=2e-3), "Runtime gamma_attn differs from artifact")
        require(math.isclose(float(runtime_activation.get("gamma_ffn")), gamma, abs_tol=2e-3), "Runtime gamma_ffn differs from artifact")
        require(int(((activation.get("diagnostics") or {}).get("native_zero_identity_delta_rms") or {}).get("count", 0)) > 0, "Missing gamma=0 identity diagnostic")

    report = {
        "assessment": "PASS",
        "architecture": args.architecture,
        "ordinary_visual_tokens": 32 * 196,
        "geometry_source": "predicted CUT3R point_maps_ref",
        "activation": activation["activation"],
        "achieved_ratio": activation["achieved_ratio"],
        "geometry_activation_runtime": runtime_activation,
        "model_forward_inputs": observed_inputs,
        "extraction_provenance": str(provenance_path.resolve()),
        "activation_artifact": str(Path(args.activation_json).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
