#!/usr/bin/env python3
"""Evidence-producing preprocessing/patch alignment audit for raw distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llava.model.raw_siglip_cut3r import GRID_SIZE, PatchCoordinateResampler


def centres(image_size: int, patch_size: int):
    return [(patch_size * i + patch_size / 2.0) / image_size for i in range(GRID_SIZE)]


def intervals(image_size: int, patch_size: int):
    return [[patch_size * i / image_size, patch_size * (i + 1) / image_size] for i in range(GRID_SIZE)]


def metadata_probe(root: str | None, limit: int):
    if not root:
        return {"checked": 0, "available": False}
    paths = sorted(Path(root).rglob("*.pt"))[:limit]
    evidence = []
    for path in paths:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
            metadata = value.get("metadata", {}) if isinstance(value, dict) else {}
            evidence.append({"path": str(path), "frame_indices": metadata.get("frame_indices"), "source_video": metadata.get("source_video")})
        except Exception as exc:
            evidence.append({"path": str(path), "error": str(exc)})
    return {"checked": len(paths), "available": bool(paths), "samples": evidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--siglip_image_size", type=int, default=384)
    parser.add_argument("--siglip_patch_size", type=int, default=14)
    parser.add_argument("--cut3r_image_size", type=int, default=432)
    parser.add_argument("--cut3r_patch_size", type=int, default=16)
    parser.add_argument("--cut3r_layer6_cache")
    parser.add_argument("--probe_samples", type=int, default=3)
    args = parser.parse_args()
    siglip_centres = centres(args.siglip_image_size, args.siglip_patch_size)
    cut3r_centres = centres(args.cut3r_image_size, args.cut3r_patch_size)
    siglip_intervals = intervals(args.siglip_image_size, args.siglip_patch_size)
    cut3r_intervals = intervals(args.cut3r_image_size, args.cut3r_patch_size)
    centre_delta = max(abs(a - b) for a, b in zip(siglip_centres, cut3r_centres))
    interval_delta = max(abs(a - b) for pair_a, pair_b in zip(siglip_intervals, cut3r_intervals) for a, b in zip(pair_a, pair_b))
    exact = centre_delta <= 1e-9 and interval_delta <= 1e-9
    status = "EXACT_PATCH_ALIGNMENT" if exact else "ALIGNMENT_WITH_DETERMINISTIC_RESAMPLING"
    resampler = PatchCoordinateResampler(siglip_centres, cut3r_centres, status=status)
    # Coordinate values are row-major landmarks.  Identity preserves them;
    # a non-identity transform is separately recorded rather than assumed.
    landmarks = torch.arange(GRID_SIZE * GRID_SIZE, dtype=torch.float32).reshape(1, 1, GRID_SIZE * GRID_SIZE, 1).expand(-1, -1, -1, 1152)
    transformed = resampler(landmarks)
    probe = {str(index): float(transformed[0, 0, index, 0]) for index in (0, GRID_SIZE // 2 * GRID_SIZE + GRID_SIZE // 2, GRID_SIZE * GRID_SIZE - 1)}
    payload = {
        "status": status,
        "grid": {"shape": [GRID_SIZE, GRID_SIZE], "flattening": "row_major", "index_mapping": "row=index//27,column=index%27", "special_tokens_included": False},
        "siglip": {
            "preprocessing": "SigLipImageProcessor: RGB -> direct resize -> rescale -> normalize; no crop/pad/flip",
            "image_size": args.siglip_image_size, "patch_size": args.siglip_patch_size,
            "normalized_patch_centers": siglip_centres, "normalized_patch_intervals": siglip_intervals,
            "source_reference": "llava/model/multimodal_encoder/siglip_encoder.py:SigLipImageProcessor and SigLipVisionTower.forward",
        },
        "cut3r": {
            "preprocessing": "process_video_with_decord -> SigLipImageProcessor.preprocess -> bilinear resize to CUT3R input; no crop/pad/flip",
            "image_size": args.cut3r_image_size, "patch_size": args.cut3r_patch_size,
            "normalized_patch_centers": cut3r_centres, "normalized_patch_intervals": cut3r_intervals,
            "source_reference": "scripts/extraction/extract_cut3r_layer_features.py:load_and_preprocess_video_frames/run_cut3r_layers",
        },
        "comparison": {"max_center_delta": centre_delta, "max_interval_boundary_delta": interval_delta, "orientation": "same RGB/top-left origin", "frame_sampling": "process_video_with_decord(video_fps=1, frames_upbound=32, force_sample=True)"},
        "deterministic_resampling": resampler.metadata(),
        "synthetic_row_major_landmarks": probe,
        "frame_metadata_probe": metadata_probe(args.cut3r_layer6_cache, args.probe_samples),
    }
    digest_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(digest_payload).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "sha256": payload["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
