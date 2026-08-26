#!/usr/bin/env python
"""Materialize paired SpatialStack residual-mask features for the depth probe.

The perturbation extractor stores one compact tensor bundle per video so the
normal and all-geometry-off states are guaranteed to come from the same model
input, selected frames, and visual-token positions.  This utility validates
those bundles against the authoritative frame manifest and writes only the two
new probe variants in the ordinary layer-per-directory cache layout:

``geometry_off`` and ``geometry_delta = normal - geometry_off``.

Normal features are written by the extractor itself and deliberately remain
under the original model label, avoiding an unnecessary duplicate cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.probing.depth_probe_common import layer_feature_path, read_json
except ModuleNotFoundError:  # Direct execution from scripts/probing.
    from depth_probe_common import layer_feature_path, read_json


SCHEMA_VERSION = "geometry_induced_depth_probe_materialization_v1"


def parse_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers or any(layer < 0 for layer in layers) or len(layers) != len(set(layers)):
        raise ValueError(f"Invalid unique non-negative layer list: {value!r}")
    return layers


def safe_video_id(video: dict[str, Any]) -> str:
    video_id = str(video.get("video_sample_id", video.get("video_path", "")))
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in video_id)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.float().square())).item())


def require_tensor(payload: dict[str, Any], section: str, layer: int, frame: int, path: Path) -> torch.Tensor:
    try:
        value = payload[section][f"layer_{layer}"][str(frame)]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Missing {section}/layer_{layer}/frame_{frame} in {path}") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor at {section}/layer_{layer}/frame_{frame} in {path}")
    return value


def verify_frame_metadata(
    *, output_root: Path, frame_record: dict[str, Any], payload: dict[str, Any], tensor: torch.Tensor
) -> tuple[int, int]:
    fsid = str(frame_record["frame_sample_id"])
    metadata_path = output_root / "metadata" / f"frame_{fsid}.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Normal extraction metadata missing: {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu")
    expected_grid = tuple(int(value) for value in metadata["visual_grid_shape"])
    if tensor.ndim == 2:
        if int(tensor.shape[0]) != expected_grid[0] * expected_grid[1]:
            raise RuntimeError(
                f"Token count mismatch for {fsid}: paired={tuple(tensor.shape)}, normal metadata={expected_grid}"
            )
    elif tensor.ndim == 3:
        if tuple(tensor.shape[:2]) != expected_grid:
            raise RuntimeError(
                f"Token grid mismatch for {fsid}: paired={tuple(tensor.shape)}, normal metadata={expected_grid}"
            )
    else:
        raise RuntimeError(f"Expected 2-D tokens or 3-D grid for {fsid}, got {tuple(tensor.shape)}")
    selected_frames = [int(value) for value in payload.get("selected_frames", [])]
    if int(frame_record["frame_index"]) not in selected_frames:
        raise RuntimeError(f"Frame {frame_record['frame_index']} absent from paired selected_frames for {fsid}")
    return expected_grid


def atomic_save(tensor: torch.Tensor, path: Path, *, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        existing = torch.load(path, map_location="cpu")
        if not isinstance(existing, torch.Tensor) or tuple(existing.shape) != tuple(tensor.shape):
            raise RuntimeError(f"Existing feature is incompatible; use --overwrite after inspection: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pair-root", required=True)
    parser.add_argument("--output-root", required=True, help="Normal extractor cache root with targets/metadata.")
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated historical L indices.")
    parser.add_argument("--injection-layers", required=True, help="Comma-separated active SpatialStack injection L indices.")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each paired raw video payload only after its off/delta features materialize successfully.",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Smoke-only: allow missing manifest videos.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    pair_root = Path(args.source_pair_root) / args.model_label
    layers = parse_layers(args.layers)
    injection_layers = parse_layers(args.injection_layers)
    target_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    manifest_path = Path(args.sample_indices)
    manifest = read_json(manifest_path)
    videos = list(manifest.get("videos", []))
    expected_videos = len(videos)
    seen = 0
    feature_rms_rows: list[dict[str, Any]] = []

    for video in videos:
        pair_path = pair_root / f"video_{safe_video_id(video)}.pt"
        if not pair_path.is_file():
            if args.allow_partial:
                continue
            raise FileNotFoundError(f"Missing paired feature payload: {pair_path}")
        payload = torch.load(pair_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected paired payload dict at {pair_path}")
        if payload.get("schema_version") != "frozen_probe_geometry_perturbation_features_v1":
            raise RuntimeError(f"Unexpected paired feature schema in {pair_path}: {payload.get('schema_version')!r}")
        if str(payload.get("model_label")) != args.model_label:
            raise RuntimeError(f"Model mismatch in {pair_path}: {payload.get('model_label')!r}")
        if str(payload.get("video_id")) != str(video.get("video_sample_id", video.get("video_path", ""))):
            raise RuntimeError(f"Video identity mismatch in {pair_path}")
        if str(payload.get("split")) != str(video.get("split")):
            raise RuntimeError(f"Split mismatch in {pair_path}")
        if payload.get("hidden_state_indexing") != "requested_L -> hidden_states[L + 1] (post-decoder-block L; includes injection at L)":
            raise RuntimeError(f"Unexpected hidden-state convention in {pair_path}")

        for frame_record in video["frames"]:
            frame = int(frame_record["frame_index"])
            for layer in layers:
                normal = require_tensor(payload, "normal_by_layer", layer, frame, pair_path)
                off = require_tensor(payload, "geometry_off_all_by_layer", layer, frame, pair_path)
                if tuple(normal.shape) != tuple(off.shape):
                    raise RuntimeError(f"Normal/off shape mismatch at {pair_path}, L{layer}, frame {frame}")
                if normal.ndim not in {2, 3} or normal.shape[-1] != 3584:
                    raise RuntimeError(f"Unexpected hidden feature shape {tuple(normal.shape)} at {pair_path}, L{layer}")
                if not torch.isfinite(normal).all() or not torch.isfinite(off).all():
                    raise RuntimeError(f"Non-finite paired feature at {pair_path}, L{layer}, frame {frame}")
                grid_shape = verify_frame_metadata(
                    output_root=output_root, frame_record=frame_record, payload=payload, tensor=normal
                )
                delta = normal.float() - off.float()
                if layer < min(injection_layers) and float(delta.abs().max().item()) > 1e-6:
                    raise RuntimeError(f"All-off delta is nonzero before first injection at L{layer}: {pair_path}")
                if layer >= min(injection_layers) and float(delta.abs().max().item()) == 0.0:
                    raise RuntimeError(f"All-off delta is identically zero downstream of injection at L{layer}: {pair_path}")

                fsid = str(frame_record["frame_sample_id"])
                off_grid = (
                    off.reshape(*grid_shape, off.shape[-1]) if off.ndim == 2 else off
                ).to(dtype=target_dtype).contiguous()
                delta_grid = (
                    delta.reshape(*grid_shape, delta.shape[-1]) if delta.ndim == 2 else delta
                ).to(dtype=target_dtype).contiguous()
                atomic_save(
                    off_grid,
                    layer_feature_path(output_root, f"{args.model_label}__geometry_off", layer, fsid),
                    overwrite=args.overwrite,
                )
                atomic_save(
                    delta_grid,
                    layer_feature_path(output_root, f"{args.model_label}__geometry_delta", layer, fsid),
                    overwrite=args.overwrite,
                )
                for feature_type, value in (("normal", normal), ("geometry_off", off), ("geometry_delta", delta)):
                    feature_rms_rows.append(
                        {
                            "model": args.model_label,
                            "video_id": str(payload["video_id"]),
                            "split": str(payload["split"]),
                            "frame_index": frame,
                            "layer": layer,
                            "feature_type": feature_type,
                            "rms": feature_rms(value),
                        }
                    )
        if args.delete_source:
            pair_path.unlink()
        seen += 1

    if seen != expected_videos and not args.allow_partial:
        raise RuntimeError(f"Materialized {seen}/{expected_videos} paired videos")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "model_label": args.model_label,
        "source_pair_root": str(pair_root),
        "sample_indices": str(manifest_path),
        "sample_indices_sha256": sha256(manifest_path),
        "layers": layers,
        "injection_layers": injection_layers,
        "feature_definition": "geometry_delta = paired normal hidden state - paired all-geometry-off hidden state",
        "hidden_state_indexing": "requested_L -> hidden_states[L + 1] (post-decoder-block L; includes injection at L)",
        "videos_materialized": seen,
    }
    provenance_path = output_root / "geometry_induced_probe" / args.model_label / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rms_path = output_root / "geometry_induced_probe" / args.model_label / "feature_rms.json"
    rms_path.write_text(json.dumps(feature_rms_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **provenance, "rms_path": str(rms_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
