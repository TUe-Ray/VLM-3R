#!/usr/bin/env python
"""Turn an EoMT one-video VLM smoke provenance record into a compact PASS gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-label", required=True, choices=("eomt_object", "eomt_selective"))
    args = parser.parse_args()

    # ``provenance.json`` lives in every feature-level directory and only
    # describes that tensor cache.  The extractor writes per-forward runtime
    # attestation to this model-level artifact.
    provenance_path = args.output_root / "features" / args.model_label / "extraction_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    samples = provenance.get("extraction_samples", [])
    if not isinstance(samples, list) or len(samples) != 1:
        raise RuntimeError("EoMT VLM smoke must contain exactly one completed video forward")
    sample = samples[0]
    runtime = sample.get("first_video_runtime_assertions", {})
    if runtime.get("assessment") != "PASS":
        raise RuntimeError("EoMT VLM smoke lacks a PASS first-forward runtime contract")
    if runtime.get("ordinary_visual_tokens") != 32 * 196:
        raise RuntimeError("EoMT VLM smoke did not preserve 32x196 ordinary visual tokens")
    if not runtime.get("primary_probe_excludes_auxiliary_tokens", False):
        raise RuntimeError("EoMT VLM smoke primary representation includes auxiliary tokens")
    if not sample.get("eomt_cache_scene"):
        raise RuntimeError("EoMT VLM smoke did not attest its consumed cache scene")

    consumer_debug = (
        sample.get("eomt_object_debug")
        if args.model_label == "eomt_object"
        else sample.get("eomt_selective_debug")
    )
    if not isinstance(consumer_debug, list):
        raise RuntimeError("EoMT VLM smoke did not execute its cached consumer")
    if args.model_label == "eomt_selective" and len(consumer_debug) != 32:
        raise RuntimeError("Selective EoMT smoke did not produce 32 frame gates")

    report = {
        "status": "PASS",
        "model_label": args.model_label,
        "cache_scene": sample["eomt_cache_scene"],
        "ordinary_visual_tokens": runtime["ordinary_visual_tokens"],
        "primary_probe_excludes_auxiliary_tokens": True,
        "eomt_object_auxiliary_token_count": runtime.get("eomt_object_auxiliary_token_count", 0),
        "consumer_debug": consumer_debug,
        "provenance": str(provenance_path),
    }
    destination = args.output_root / "eomt_vlm_forward_smoke_report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
