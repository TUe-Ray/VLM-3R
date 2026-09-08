#!/usr/bin/env python
"""Verify the post-SFT all-point smoke cache without fitting a new probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge


POINTS = (
    "fusion_output", "projected_features", "L0", "L1", "L2", "L3", "L6", "L9",
    "L12", "L15", "L18", "L21", "L24", "L27",
)


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def level(point: str) -> str:
    return f"layer_{point[1:]}" if point.startswith("L") else point


def valid_arrays(cache_root: Path, model: str, point: str, video: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, masks = [], [], []
    for frame in video["frames"]:
        frame_id = str(frame["frame_sample_id"])
        feature = load(cache_root / "features" / model / level(point) / f"frame_{frame_id}.pt")
        target = load(cache_root / "gt_depth" / f"frame_{frame_id}.pt")
        metadata = load(cache_root / "metadata" / f"frame_{frame_id}.pt")
        if not isinstance(feature, torch.Tensor) or not isinstance(target, torch.Tensor) or not isinstance(metadata, dict):
            raise RuntimeError(f"Cannot prepare ridge smoke arrays for {point}/{frame_id}")
        valid = metadata["gt_valid_mask"].reshape(-1).bool() & torch.isfinite(target.reshape(-1)) & (target.reshape(-1) > 0)
        features.append(feature.reshape(-1, feature.shape[-1]).float().numpy())
        targets.append(target.reshape(-1).float().numpy())
        masks.append(valid.numpy())
    x_all = np.concatenate(features, axis=0)
    y_all = np.concatenate(targets, axis=0)
    valid_all = np.concatenate(masks, axis=0)
    return x_all, x_all[valid_all], y_all[valid_all]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-injection-layers", required=True, help="Comma-separated e.g. 0,1,2")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    cache_root = Path(args.cache_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    videos = list(manifest["videos"])
    expected_layers = [int(value) for value in args.expected_injection_layers.split(",") if value]
    split_counts = tuple(sum(video["split"] == split for video in videos) for split in ("train", "val", "dev_eval"))
    if split_counts != (1, 1, 1):
        raise RuntimeError("Smoke manifest must contain exactly one train, validation, and held-out video")
    config = json.loads((Path(args.checkpoint) / "config.json").read_text(encoding="utf-8"))
    if config.get("use_cut3r_spatialstack") is not True:
        raise RuntimeError("Checkpoint does not enable native CUT3R SpatialStack")
    configured_layers = [int(value) for value in str(config.get("cut3r_spatialstack_llm_layers", "")).split(",") if value]
    if configured_layers != expected_layers:
        raise RuntimeError(f"Checkpoint injection schedule {configured_layers} != requested {expected_layers}")
    checked_frames = []
    for video in videos:
        for frame in video["frames"]:
            frame_id = str(frame["frame_sample_id"])
            target = load(cache_root / "gt_depth" / f"frame_{frame_id}.pt")
            metadata = load(cache_root / "metadata" / f"frame_{frame_id}.pt")
            if not isinstance(target, torch.Tensor) or not isinstance(metadata, dict):
                raise RuntimeError(f"Invalid target or metadata cache for {frame_id}")
            valid = metadata.get("gt_valid_mask")
            if not isinstance(valid, torch.Tensor) or int(valid.sum()) <= 0:
                raise RuntimeError(f"No valid depth targets for {frame_id}")
            point_shapes: dict[str, list[int]] = {}
            for point in POINTS:
                path = cache_root / "features" / args.model / level(point) / f"frame_{frame_id}.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing requested point {point}: {path}")
                feature = load(path)
                if not isinstance(feature, torch.Tensor) or feature.ndim < 2:
                    raise RuntimeError(f"Invalid feature tensor for {point}/{frame_id}")
                if int(feature.reshape(-1, feature.shape[-1]).shape[0]) != int(target.numel()):
                    raise RuntimeError(f"Feature/depth token mismatch for {point}/{frame_id}")
                if not bool(torch.isfinite(feature.float()).all()):
                    raise RuntimeError(f"Non-finite feature values for {point}/{frame_id}")
                point_shapes[point] = [int(value) for value in feature.shape]
            checked_frames.append({"frame_sample_id": frame_id, "point_shapes": point_shapes, "valid_tokens": int(valid.sum())})
    split_to_video = {str(video["split"]): video for video in videos}
    ridge_vf = []
    for point in POINTS:
        x_train_all, x_train, y_train = valid_arrays(cache_root, args.model, point, split_to_video["train"])
        x_eval_all, _x_eval, _y_eval = valid_arrays(cache_root, args.model, point, split_to_video["dev_eval"])
        # A fixed alpha is intentional here: this is only a finite-value
        # smoke of the frozen ridge/VF path. Formal alpha selection remains
        # validation-video-only in the cache-only analysis entry point.
        fit = Ridge(alpha=1.0).fit(x_train, y_train)
        coefficient = np.asarray(fit.coef_, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(coefficient))
        direction = coefficient / norm
        centered = x_eval_all - x_eval_all.mean(axis=0, keepdims=True)
        vf = float(np.square(centered @ direction).sum() / np.square(centered).sum())
        if not (np.isfinite(coefficient).all() and np.isfinite(vf) and norm > 0.0):
            raise RuntimeError(f"Non-finite ridge or VF smoke value at {point}")
        ridge_vf.append({"probe_point": point, "coefficient_norm": norm, "vf_depth": vf, "vf_enrich": float(x_eval_all.shape[1] * vf)})
    provenance_path = cache_root / "features" / args.model / "extraction_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("model_loading_mode") != "adapter":
        raise RuntimeError("Post-SFT smoke did not use the required trained-adapter load path")
    samples = provenance.get("extraction_samples", [])
    if not samples:
        raise RuntimeError("Extraction provenance lacks first-forward runtime evidence")
    stats = samples[0].get("spatialstack_insertion_stats", [])
    observed_layers = [int(row["layer_idx"]) for row in stats if row.get("fusion_type") == "add"]
    if observed_layers != expected_layers:
        raise RuntimeError(f"Observed additive injection layers {observed_layers} != {expected_layers}")
    report = {
        "status": "PASS",
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "c1_canonicalization_active": False,
        "probe_points": list(POINTS),
        "observed_additive_injection_layers": observed_layers,
        "checked_frames": checked_frames,
        "ridge_vf_finite_smoke": ridge_vf,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
