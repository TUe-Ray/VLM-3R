#!/usr/bin/env python
"""Fail closed on completeness of one post-SFT ScanNet depth-probe result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.post_sft_geometry_probe_specs import (
    POST_SFT_DEPTH_LAYERS,
    POST_SFT_PRE_LLM_FEATURES,
    SPLIT_SHA256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--require-probes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = load_json(args.sample_indices)
    expected_ids = {
        str(frame["frame_sample_id"])
        for video in split.get("videos", [])
        for frame in video.get("frames", [])
    }
    if len(expected_ids) != 2398 or sha256(args.sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Sample-index identity is not the authoritative 1,199-video/2,398-frame ScanNet split")

    report = {
        "schema_version": "post_sft_depth_probe_completeness_v2",
        "output_root": str(args.output_root),
        "model_label": args.model_label,
        "sample_indices": str(args.sample_indices),
        "sample_indices_sha256": sha256(args.sample_indices),
        "expected_selected_frames": len(expected_ids),
        "required_pre_llm_features": list(POST_SFT_PRE_LLM_FEATURES),
        "required_layers": list(POST_SFT_DEPTH_LAYERS),
        "pre_llm_features": {},
        "layers": {},
    }
    failures: list[str] = []
    for feature_name in POST_SFT_PRE_LLM_FEATURES:
        feature_dir = args.output_root / "features" / args.model_label / feature_name
        actual_ids = {
            path.name.removeprefix("frame_").removesuffix(".pt")
            for path in feature_dir.glob("frame_*.pt")
        }
        missing, extra = sorted(expected_ids - actual_ids), sorted(actual_ids - expected_ids)
        provenance_path = feature_dir / "provenance.json"
        provenance = load_json(provenance_path) if provenance_path.is_file() else {}
        valid_provenance = (
            provenance.get("sample_indices_sha256") == SPLIT_SHA256
            and provenance.get("requested_pre_llm_features") == list(POST_SFT_PRE_LLM_FEATURES)
            and provenance.get("feature_level") == feature_name
        )
        probe_path = args.output_root / "probes" / args.model_label / feature_name / "metrics.json"
        probe = load_json(probe_path) if probe_path.is_file() else None
        probe_ok = (
            isinstance(probe, dict)
            and probe.get("feature_level") == feature_name
            and int(probe.get("num_tokens", -1)) == 75656
        )
        report["pre_llm_features"][feature_name] = {
            "feature_file_count": len(actual_ids),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "first_missing": missing[:5],
            "first_extra": extra[:5],
            "provenance_ok": valid_provenance,
            "probe_metrics_path": str(probe_path),
            "probe_ok": probe_ok,
            "metrics": probe,
        }
        if missing or extra or not valid_provenance or (args.require_probes and not probe_ok):
            failures.append(feature_name)
    for layer in POST_SFT_DEPTH_LAYERS:
        layer_dir = args.output_root / "features" / args.model_label / f"layer_{layer}"
        actual_ids = {
            path.name.removeprefix("frame_").removesuffix(".pt")
            for path in layer_dir.glob("frame_*.pt")
        }
        missing, extra = sorted(expected_ids - actual_ids), sorted(actual_ids - expected_ids)
        provenance_path = layer_dir / "provenance.json"
        provenance = load_json(provenance_path) if provenance_path.is_file() else {}
        valid_provenance = (
            provenance.get("sample_indices_sha256") == SPLIT_SHA256
            and provenance.get("requested_llm_layers") == list(POST_SFT_DEPTH_LAYERS)
            and provenance.get("hidden_state_indexing") == "requested_L -> hidden_states[L + 1]"
        )
        probe_path = args.output_root / "probes" / args.model_label / f"layer_{layer}" / "metrics.json"
        probe = load_json(probe_path) if probe_path.is_file() else None
        probe_ok = (
            isinstance(probe, dict)
            and probe.get("feature_level") == f"layer_{layer}"
            and int(probe.get("num_tokens", -1)) == 75656
        )
        report["layers"][str(layer)] = {
            "feature_file_count": len(actual_ids),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "first_missing": missing[:5],
            "first_extra": extra[:5],
            "provenance_ok": valid_provenance,
            "probe_metrics_path": str(probe_path),
            "probe_ok": probe_ok,
            "metrics": probe,
        }
        if missing or extra or not valid_provenance or (args.require_probes and not probe_ok):
            failures.append(f"layer_{layer}")
    report["assessment"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
