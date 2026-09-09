#!/usr/bin/env python
"""Verify the complete two-video controlled-fusion pre-SFT smoke sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.controlled_fusion_pre_sft import CONTROLLED_FUSION_PRE_SFT_SPECS  # noqa: E402
from scripts.probing.probe_layer_policy import (  # noqa: E402
    COMMON_PROBE_LAYERS,
    PRE_SFT_PRE_LLM_FEATURES,
)
from scripts.probing.verify_controlled_fusion_c1 import SCHEMA_VERSION as C1_MANIFEST_SCHEMA  # noqa: E402


SCHEMA_VERSION = "controlled_fusion_pre_sft_smoke_verification_v1"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke verification: {args.output}")
    artifact_manifest = read_json(args.artifact_manifest)
    if artifact_manifest.get("schema_version") != C1_MANIFEST_SCHEMA:
        raise ValueError("Smoke received an incompatible C1 artifact manifest")
    if artifact_manifest.get("post_sft_state_loaded") is not False:
        raise ValueError("Smoke C1 manifest does not prove absence of post-SFT state")
    samples = read_json(args.sample_indices)
    videos = samples.get("videos")
    if not isinstance(videos, list) or len(videos) != 2:
        raise ValueError("Controlled-fusion smoke must contain exactly one train and one validation video")
    feature_levels = [*PRE_SFT_PRE_LLM_FEATURES, *(f"layer_{layer}" for layer in COMMON_PROBE_LAYERS)]
    expected_frames = 2 * len(videos)
    candidates: dict[str, Any] = {}
    candidate_specs = {"BASE": None, **CONTROLLED_FUSION_PRE_SFT_SPECS}
    for identifier, spec in candidate_specs.items():
        model_label = "pre_sft_base_vlm" if spec is None else spec.pre_sft_variant
        root = args.cache_root / identifier
        feature_root = root / "features" / model_label
        provenance_path = feature_root / "extraction_provenance.json"
        provenance = read_json(provenance_path)
        expected_loading_mode = "pre_sft_base_vlm" if spec is None else "pre_sft_fusion"
        if provenance.get("model_loading_mode") != expected_loading_mode:
            raise ValueError(f"{identifier} smoke loading-mode mismatch")
        if spec is not None and provenance.get("experiment_variant") != spec.pre_sft_variant:
            raise ValueError(f"{identifier} smoke variant mismatch")
        if provenance.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise ValueError(f"{identifier} smoke lacks proof that no SFT adapter was loaded")
        if provenance.get("git_worktree_dirty") is not False:
            raise ValueError(f"{identifier} smoke was executed from a dirty worktree")
        if provenance.get("git_commit") != artifact_manifest.get("git_commit"):
            raise ValueError(f"{identifier} smoke commit differs from its C1 manifest")
        if spec is None:
            if provenance.get("c1_calibration_json") is not None:
                raise ValueError("BASE smoke unexpectedly loaded C1 state")
        else:
            artifact_entry = artifact_manifest["artifacts"][identifier]
            if provenance.get("c1_calibration_sha256") != artifact_entry["sha256"]:
                raise ValueError(f"{identifier} smoke C1 artifact hash mismatch")
        counts: dict[str, int] = {}
        metrics: dict[str, Any] = {}
        for level in feature_levels:
            level_root = feature_root / level
            count = len(list(level_root.glob("frame_*.pt")))
            if count != expected_frames:
                raise ValueError(
                    f"{identifier}/{level} has {count} selected-frame tensors; expected {expected_frames}"
                )
            counts[level] = count
            metric_path = root / "probes" / model_label / level / "metrics.json"
            metric = read_json(metric_path)
            for name in ("mae", "absrel", "delta125"):
                if not math.isfinite(float(metric.get(name, float("nan")))):
                    raise ValueError(f"{identifier}/{level} has non-finite {name}")
            metrics[level] = {
                name: metric[name] for name in ("mae", "absrel", "delta125", "num_tokens")
            }
        candidates[identifier] = {
            "status": "PASS",
            "model_label": model_label,
            "feature_counts": counts,
            "probe_metrics": metrics,
            "extraction_provenance": str(provenance_path),
            "extraction_provenance_sha256": sha256_file(provenance_path),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "artifact_manifest_sha256": sha256_file(args.artifact_manifest),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sha256_file(args.sample_indices),
        "feature_levels": feature_levels,
        "selected_frame_tensors_per_level": expected_frames,
        "candidates": candidates,
        "post_sft_state_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "candidates": list(candidates)}))


if __name__ == "__main__":
    main()
