#!/usr/bin/env python
"""Shared utilities for the VLM-3R frame-level depth probing experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_YAML = REPO_ROOT / "scripts" / "VLM_3R" / "vsibench_data.yaml"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("WORK", "/leonardo_work/EUHPC_D32_006")) / "probing_experiment"
DEFAULT_FAST_FEATURE_ROOT = Path("/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
DEFAULT_SPATIAL_FEATURES_SUBDIR = "spatial_features"
DEFAULT_POINT_MAPS_SUBDIR = "spatial_features_points"

MODEL_PRESETS = {
    "zero_spatial": "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/zero_spatial_features",
    "vlm3r_baseline": "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2",
    "geo_rope_fusion": "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/geo_rope_spherical_forced_full_rope_resume_fast_workfb_42445435",
}

LLM_LAYERS = [0, 3, 6, 9, 15, 21, 27]
PRE_LLM_FEATURES = ["fusion_output", "projected_features"]

CAMERA_DEPTH_KEYS = ("point_maps_cam", "pts3d_in_self_view", "point_maps", "point_map", "pts3d")
REFERENCE_DEPTH_KEYS = ("point_maps_ref", "pts3d_in_other_view")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def stable_int_seed(*parts: Any) -> int:
    text = "\0".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_sample(items: list[Any], count: int, seed: int) -> list[Any]:
    if count > len(items):
        raise ValueError(f"Cannot sample {count} items from {len(items)} candidates")
    rng = random.Random(seed)
    indices = rng.sample(range(len(items)), count)
    return [items[idx] for idx in indices]


def frame_sample_indices(num_frames: int, count: int, seed: int, video_key: str) -> list[int]:
    if num_frames < count:
        raise ValueError(f"Need at least {count} frames, got {num_frames} for {video_key}")
    rng = random.Random(stable_int_seed(seed, video_key, "frames"))
    return sorted(rng.sample(range(num_frames), count))


def sidecar_relative_path(video_rel_path: str, subdir: str | None) -> Path:
    video_pt = Path(video_rel_path).with_suffix(".pt")
    parts = list(video_pt.parts)
    if subdir:
        if "videos" in parts:
            parts[parts.index("videos")] = subdir
        else:
            parts.insert(-1, subdir)
    return Path(*parts)


def resolve_sidecar_path(
    video_rel_path: str,
    root: Path | str,
    subdir: str | None,
    *,
    fallback_root: Path | str | None = None,
) -> Path | None:
    roots = [Path(root)]
    if fallback_root is not None:
        roots.append(Path(fallback_root))
    rel_candidates = [sidecar_relative_path(video_rel_path, subdir)]
    if subdir:
        rel_candidates.append(sidecar_relative_path(video_rel_path, None))
    for cur_root in roots:
        for rel in rel_candidates:
            candidate = cur_root / rel
            if candidate.exists():
                return candidate
    return None


def raw_sample_ids(raw_item: dict[str, Any], index: int) -> list[str]:
    ids: list[str] = []
    for key in ("sample_id", "doc_id", "id", "question_id", "uid"):
        if raw_item.get(key) is not None:
            ids.append(str(raw_item[key]))
    ids.append(str(index))
    return list(dict.fromkeys(ids))


def stable_sample_key(item: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    conv0 = ""
    conversations = item.get("conversations", [])
    if isinstance(conversations, list) and conversations:
        first = conversations[0]
        if isinstance(first, dict):
            conv0 = str(first.get("value", ""))
    return (
        str(item.get("_annotation_path", "")),
        str(item.get("id", "")),
        str(item.get("question_id", "")),
        str(item.get("video", "")),
        str(item.get("image", "")),
        str(item.get("data_source", "")),
        conv0[:128],
    )


def scene_id_from_item(item: dict[str, Any]) -> str:
    if item.get("scene_name") is not None:
        return str(item["scene_name"])
    video = str(item.get("video", ""))
    return Path(video).stem


def frame_sample_id(video_sample_id: str, frame_index: int) -> str:
    safe_video = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(video_sample_id))
    return f"{safe_video}_f{int(frame_index):04d}"


def iter_frame_records(sample_index_payload: dict[str, Any], split: str | None = None) -> Iterable[dict[str, Any]]:
    for video in sample_index_payload.get("videos", []):
        if split is not None and video.get("split") != split:
            continue
        for frame in video.get("frames", []):
            record = dict(video)
            record.pop("frames", None)
            record.update(frame)
            yield record


def load_frame_records(sample_indices_path: Path, split: str | None = None) -> list[dict[str, Any]]:
    return list(iter_frame_records(read_json(sample_indices_path), split=split))


def load_yaml_dataset_records(data_yaml: Path) -> list[dict[str, Any]]:
    try:
        import yaml  # type: ignore

        with data_yaml.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        datasets = payload.get("datasets", [])
    except ModuleNotFoundError:
        datasets = []
        current: dict[str, Any] | None = None
        with data_yaml.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].rstrip()
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("- "):
                    if current:
                        datasets.append(current)
                    current = {}
                    stripped = stripped[2:].strip()
                    if stripped:
                        key, sep, value = stripped.partition(":")
                        if sep:
                            current[key.strip()] = value.strip().strip("'\"")
                    continue
                if current is not None and ":" in stripped:
                    key, value = stripped.split(":", 1)
                    current[key.strip()] = value.strip().strip("'\"")
        if current:
            datasets.append(current)
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        json_path = Path(dataset["json_path"])
        with json_path.open("r", encoding="utf-8") as f:
            if json_path.suffix == ".jsonl":
                cur = [json.loads(line) for line in f if line.strip()]
            else:
                cur = json.load(f)
        for item in cur:
            if isinstance(item, dict):
                enriched = dict(item)
                enriched["_annotation_path"] = str(json_path)
                enriched["_with_depth"] = bool(dataset.get("with_depth", False))
                records.append(enriched)
    records.sort(key=stable_sample_key)
    return records


def load_point_map_sidecar(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected point-map sidecar dict at {path}, got {type(payload)}")
    return payload


def sidecar_num_frames(payload: dict[str, Any]) -> int:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("num_frames") is not None:
        return int(metadata["num_frames"])
    for key in CAMERA_DEPTH_KEYS + REFERENCE_DEPTH_KEYS + ("patch_tokens", "camera_tokens"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("Could not infer sidecar frame count")


def select_point_maps(payload: dict[str, Any], *, allow_euclidean_depth: bool = False) -> tuple[torch.Tensor, str, str]:
    for key in CAMERA_DEPTH_KEYS:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value, key, "camera_z"
    for key in REFERENCE_DEPTH_KEYS:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            if allow_euclidean_depth:
                return value, key, "euclidean"
            raise ValueError(
                f"Only reference/world point maps are available ({key}); pass --allow-euclidean-depth to use Euclidean distance."
            )
    raise KeyError(f"None of the supported point-map keys were found: {CAMERA_DEPTH_KEYS + REFERENCE_DEPTH_KEYS}")


def depth_from_point_maps(point_maps: torch.Tensor, mode: str) -> torch.Tensor:
    if point_maps.ndim != 4:
        raise ValueError(f"Expected rank-4 point maps, got {tuple(point_maps.shape)}")
    if point_maps.shape[-1] == 3:
        points = point_maps.float()
    elif point_maps.shape[1] == 3:
        points = point_maps.permute(0, 2, 3, 1).float()
    else:
        raise ValueError(f"Expected [F,H,W,3] or [F,3,H,W] point maps, got {tuple(point_maps.shape)}")
    if mode == "camera_z":
        return points[..., 2]
    if mode == "euclidean":
        return torch.linalg.norm(points, dim=-1)
    raise ValueError(f"Unknown depth mode: {mode}")


def downsample_depth_to_grid(depth: torch.Tensor, grid_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim != 2:
        raise ValueError(f"Expected a single-frame depth map [H,W], got {tuple(depth.shape)}")
    h_tok, w_tok = int(grid_shape[0]), int(grid_shape[1])
    depth = depth.float()
    valid = torch.isfinite(depth) & (depth > 0)
    values = torch.where(valid, depth, torch.zeros_like(depth))
    values = values.unsqueeze(0).unsqueeze(0)
    mask = valid.float().unsqueeze(0).unsqueeze(0)
    pooled_values = F.adaptive_avg_pool2d(values, (h_tok, w_tok))[0, 0]
    pooled_mask = F.adaptive_avg_pool2d(mask, (h_tok, w_tok))[0, 0]
    pooled_depth = pooled_values / pooled_mask.clamp_min(1e-8)
    pooled_depth = torch.where(pooled_mask > 0, pooled_depth, torch.zeros_like(pooled_depth))
    return pooled_depth.float(), (pooled_mask > 0)


def frame_depth_metadata(depth: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    values = depth[valid & torch.isfinite(depth) & (depth > 0)].float()
    if values.numel() == 0:
        return {
            "max_depth": float("nan"),
            "p95_depth": float("nan"),
            "fraction_depth_gt_10m": float("nan"),
            "mean_depth": float("nan"),
        }
    return {
        "max_depth": float(values.max().item()),
        "p95_depth": float(torch.quantile(values, 0.95).item()),
        "fraction_depth_gt_10m": float((values > 10.0).float().mean().item()),
        "mean_depth": float(values.mean().item()),
    }


def grid_shape_for_frame(metadata: dict[str, Any], frame_id: int, token_count: int | None = None) -> tuple[int, int]:
    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    shapes = metadata.get("visual_grid_shapes") or []
    if frame_order and shapes and int(frame_id) in frame_order:
        idx = frame_order.index(int(frame_id))
        shape = shapes[idx]
        return int(shape[0]), int(shape[1])
    if token_count is None:
        tokens_per_frame = metadata.get("tokens_per_frame")
        if isinstance(tokens_per_frame, list) and tokens_per_frame:
            token_count = int(tokens_per_frame[0])
    if token_count is None:
        raise ValueError(f"Cannot infer grid shape for frame {frame_id}")
    side = int(math.isqrt(int(token_count)))
    if side * side != int(token_count):
        raise ValueError(f"Token count is not square: {token_count}")
    return side, side


def reshape_tokens_to_grid(tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    h_tok, w_tok = int(grid_shape[0]), int(grid_shape[1])
    if tokens.ndim != 2:
        raise ValueError(f"Expected token tensor [N,D], got {tuple(tokens.shape)}")
    if tokens.shape[0] != h_tok * w_tok:
        raise ValueError(f"Token count {tokens.shape[0]} does not match grid {grid_shape}")
    return tokens.reshape(h_tok, w_tok, tokens.shape[-1]).contiguous()


def coerce_cache_dtype(value: torch.Tensor, cache_dtype: torch.dtype) -> torch.Tensor:
    if not value.is_floating_point():
        return value.cpu()
    return value.detach().to(device="cpu", dtype=cache_dtype).contiguous()


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def metric_values(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    pred = pred.float().reshape(-1)
    gt = gt.float().reshape(-1)
    if valid is None:
        valid_mask = torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred)
    else:
        valid_mask = valid.reshape(-1).bool() & torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred)
    if valid_mask.sum().item() == 0:
        return {"mae": float("nan"), "absrel": float("nan"), "delta125": float("nan"), "num_tokens": 0}
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    eps = 1e-6
    pred_pos = pred.clamp_min(eps)
    gt_pos = gt.clamp_min(eps)
    ratio = torch.maximum(pred_pos / gt_pos, gt_pos / pred_pos)
    return {
        "mae": float((pred - gt).abs().mean().item()),
        "absrel": float(((pred - gt).abs() / gt_pos).mean().item()),
        "delta125": float((ratio < 1.25).float().mean().item()),
        "num_tokens": int(gt.numel()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
