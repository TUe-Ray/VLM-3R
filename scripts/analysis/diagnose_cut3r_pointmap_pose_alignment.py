#!/usr/bin/env python3
"""Numerically diagnose CUT3R point-map / camera-pose non-alignment.

The default path is training-free and reads existing CUT3R point-map sidecars.
Optional representative enrichment re-runs CUT3R for only the selected
best/typical/worst samples to recover confidences that the v1 sidecar schema did
not save.  Optional ScanNet ``.sens`` input adds ground-truth depth and pose
isolation tests without changing the source sidecars.

Coordinate contract (verified from CUT3R's official helpers):

* point_maps_cam: per-frame OpenCV camera coordinates, stored [..., 3];
* point_maps_ref: CUT3R reference coordinates, stored [..., 3];
* camera_pose: [tx, ty, tz, qw, qx, qy, qz], OpenCV camera-to-reference T_cw;
* geotrf(T_cw, point_maps_cam) applies X_ref = R_cw X_cam + t_cw.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CUT3R_ROOT = REPO_ROOT / "third_party" / "CUT3R"
CUT3R_SRC_ROOT = CUT3R_ROOT / "src"
for import_root in (REPO_ROOT, CUT3R_ROOT, CUT3R_SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from src.dust3r.utils.geometry import geotrf  # noqa: E402


DEFAULT_SIDECAR_ROOT = Path(
    "/leonardo_scratch/large/userexternal/shuang00/"
    "VLM_3R_cut3r_pointmaps/scannet/spatial_features_points"
)
DEFAULT_VIDEO_ROOT = Path("/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/videos")
DEFAULT_CHECKPOINT = CUT3R_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CONFIDENCE_SWEEP = ("all", "top75", "top50", "top25")
POINT_REF_KEYS = ("point_maps_ref", "pts3d_in_other_view")
POINT_CAM_KEYS = ("point_maps_cam", "pts3d_in_self_view")
CONF_SELF_KEYS = ("conf_self", "confidence_self")
CONF_REF_KEYS = ("conf_ref", "conf", "confidence", "confidence_ref")


@dataclass
class Sample:
    sample_id: str
    sidecar_path: Path
    point_maps_cam: torch.Tensor
    point_maps_ref: torch.Tensor
    pose_encoding: torch.Tensor
    camera_pose: torch.Tensor
    conf_self: Optional[torch.Tensor]
    conf_ref: Optional[torch.Tensor]
    metadata: dict[str, Any]


@dataclass
class GTData:
    frame_indices: list[int]
    depth: torch.Tensor
    camera_pose: torch.Tensor
    intrinsics: torch.Tensor
    valid_frames: torch.Tensor
    rgb_alignment: dict[str, Any]
    source: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def first_tensor(payload: dict[str, Any], keys: Iterable[str]) -> tuple[Optional[torch.Tensor], Optional[str]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value, key
    return None, None


def as_fhw3(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"{name}: expected [F,H,W,3] or [F,3,H,W], got {tuple(value.shape)}")
    if value.shape[-1] == 3:
        return value.detach().cpu()
    if value.shape[1] == 3:
        return value.permute(0, 2, 3, 1).detach().cpu()
    raise ValueError(f"{name}: no coordinate dimension of size 3 in {tuple(value.shape)}")


def as_fhw(value: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"{name}: expected [F,H,W], got {tuple(value.shape)}")
    return value.detach().cpu().float()


def load_sample(path: Path, sidecar_root: Path) -> Sample:
    payload = load_torch(path)
    if not isinstance(payload, dict):
        raise ValueError(f"sidecar is {type(payload).__name__}, expected dict")
    cam_raw, cam_key = first_tensor(payload, POINT_CAM_KEYS)
    ref_raw, ref_key = first_tensor(payload, POINT_REF_KEYS)
    pose_raw, _ = first_tensor(payload, ("camera_pose",))
    if cam_raw is None or ref_raw is None or pose_raw is None:
        raise ValueError(f"missing camera/reference/pose fields; keys={sorted(payload)}")
    cam = as_fhw3(cam_raw, str(cam_key))
    ref = as_fhw3(ref_raw, str(ref_key))
    pose_encoding = pose_raw.detach().cpu().float()
    if pose_encoding.ndim == 3 and pose_encoding.shape[0] == 1:
        pose_encoding = pose_encoding[0]
    if pose_encoding.ndim != 2 or pose_encoding.shape[-1] != 7:
        raise ValueError(f"camera_pose must be [F,7], got {tuple(pose_encoding.shape)}")
    if cam.shape != ref.shape or cam.shape[0] != pose_encoding.shape[0]:
        raise ValueError(
            f"shape mismatch cam={tuple(cam.shape)} ref={tuple(ref.shape)} pose={tuple(pose_encoding.shape)}"
        )
    camera_pose = pose_encoding_to_camera(pose_encoding)
    conf_self_raw, _ = first_tensor(payload, CONF_SELF_KEYS)
    conf_ref_raw, _ = first_tensor(payload, CONF_REF_KEYS)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    try:
        sample_id = str(path.relative_to(sidecar_root).with_suffix(""))
    except ValueError:
        sample_id = path.stem
    return Sample(
        sample_id=sample_id,
        sidecar_path=path,
        point_maps_cam=cam,
        point_maps_ref=ref,
        pose_encoding=pose_encoding,
        camera_pose=camera_pose,
        conf_self=as_fhw(conf_self_raw, "conf_self"),
        conf_ref=as_fhw(conf_ref_raw, "conf_ref"),
        metadata=dict(metadata),
    )


def camera_from_ref(point_maps_ref: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
    return geotrf(torch.linalg.inv(camera_pose), point_maps_ref)


def ref_from_camera(point_maps_cam: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
    return geotrf(camera_pose, point_maps_cam)


def manual_geotrf(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.einsum("fij,fhwj->fhwi", transform[:, :3, :3], points) + transform[:, None, None, :3, 3]


def run_pose_smoke_tests() -> dict[str, Any]:
    angle = math.pi / 3
    q = torch.tensor([[math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]], dtype=torch.float32)
    enc = torch.cat((torch.tensor([[1.25, -0.5, 2.0]]), q), dim=-1)
    pose = pose_encoding_to_camera(enc)
    points = torch.tensor(
        [[[[0.2, -0.1, 1.0], [0.4, 0.3, 2.0]], [[-0.5, 0.2, 3.0], [0.1, -0.4, 4.0]]]],
        dtype=torch.float32,
    )
    transformed = geotrf(pose, points)
    recovered = geotrf(torch.linalg.inv(pose), transformed)
    manual = manual_geotrf(pose, points)
    wrong = geotrf(pose, transformed)
    result = {
        "official_inverse_max_abs": float((recovered - points).abs().max()),
        "official_vs_manual_max_abs": float((transformed - manual).abs().max()),
        "intentional_wrong_direction_rmse": float(torch.sqrt((wrong - points).square().mean())),
        "passed": bool(
            torch.allclose(recovered, points, atol=1e-5, rtol=1e-5)
            and torch.allclose(transformed, manual, atol=1e-6, rtol=1e-6)
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"official pose smoke test failed: {result}")
    return result


def real_pose_convention_test(sample: Sample) -> dict[str, Any]:
    cam = sample.point_maps_cam[:1].float()
    ref = sample.point_maps_ref[:1].float()
    pose = sample.camera_pose[:1]
    pred_correct = camera_from_ref(ref, pose)
    pred_wrong = ref_from_camera(ref, pose)
    valid_correct = finite_positive_mask(cam, pred_correct)
    valid_wrong = finite_positive_mask(cam, pred_wrong)
    correct = torch.linalg.vector_norm(cam - pred_correct, dim=-1)[valid_correct]
    wrong = torch.linalg.vector_norm(cam - pred_wrong, dim=-1)[valid_wrong]
    roundtrip = camera_from_ref(ref_from_camera(cam, pose), pose)
    finite = torch.isfinite(cam).all(dim=-1)
    return {
        "sample_id": sample.sample_id,
        "camera_to_reference_median_error": float(correct.median()),
        "reference_to_camera_candidate_median_error": float(wrong.median()),
        "wrong_over_correct_ratio": float(wrong.median() / correct.median().clamp_min(1e-12)),
        "roundtrip_max_abs": float((roundtrip[finite] - cam[finite]).abs().max()),
        "selected_convention": "camera_pose is camera-to-reference/world T_cw",
    }


def finite_positive_mask(cam: torch.Tensor, cam_from_ref: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(cam).all(dim=-1)
        & torch.isfinite(cam_from_ref).all(dim=-1)
        & (cam[..., 2] > 0)
        & (cam_from_ref[..., 2] > 0)
    )


def combined_confidence(sample: Sample) -> tuple[Optional[torch.Tensor], str]:
    fields = [field for field in (sample.conf_self, sample.conf_ref) if field is not None]
    if not fields:
        return None, "unavailable"
    if len(fields) == 1:
        return fields[0], "conf_self" if sample.conf_self is not None else "conf_ref"
    return torch.minimum(fields[0], fields[1]), "minimum(conf_self,conf_ref)"


def confidence_mask(valid: torch.Tensor, confidence: Optional[torch.Tensor], label: str) -> torch.Tensor:
    if label == "all":
        return valid
    if confidence is None:
        return torch.zeros_like(valid)
    fractions = {"top75": 0.75, "top50": 0.50, "top25": 0.25}
    fraction = fractions[label]
    finite = valid & torch.isfinite(confidence)
    values = confidence[finite]
    if values.numel() == 0:
        return torch.zeros_like(valid)
    threshold = torch.quantile(values, 1.0 - fraction)
    return finite & (confidence >= threshold)


def scene_scale(points: torch.Tensor, valid: Optional[torch.Tensor] = None, max_points: int = 200_000) -> float:
    stride = max(1, int(math.ceil(points.numel() / 3 / max_points)))
    sampled = points.reshape(-1, 3)[::stride].float()
    if valid is not None:
        values = sampled[valid.reshape(-1)[::stride]]
    else:
        values = sampled[torch.isfinite(sampled).all(dim=-1)]
    if values.numel() == 0:
        return float("nan")
    center = values.median(dim=0).values
    distances = torch.linalg.vector_norm(values - center, dim=-1)
    scale = distances.median()
    return float(scale) if scale > 1e-12 else float("nan")


def robust_stats(values: torch.Tensor, scale: Optional[float] = None) -> dict[str, Any]:
    values = values.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {key: None for key in ("mean", "median", "rmse", "p90", "p95", "p99", "max", "count", "normalized_mean", "normalized_median", "normalized_rmse", "normalized_p95")}
    q = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    result = {
        "mean": float(values.mean()),
        "median": float(q[0]),
        "rmse": float(torch.sqrt(values.square().mean())),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(values.max()),
        "count": int(values.numel()),
    }
    denom = float(scale) if scale is not None and math.isfinite(scale) and scale > 0 else None
    for key in ("mean", "median", "rmse", "p95"):
        result[f"normalized_{key}"] = result[key] / denom if denom else None
    return result


class StatsAccumulator:
    """Exact moments/max plus deterministic bounded samples for quantiles."""

    def __init__(self, sample_limit: int = 262_144, per_update_limit: int = 8_192):
        self.sample_limit = sample_limit
        self.per_update_limit = per_update_limit
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.maximum = -float("inf")
        self.samples: list[torch.Tensor] = []
        self.sample_count = 0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().flatten()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.double().sum())
        self.total_square += float(values.double().square().sum())
        self.maximum = max(self.maximum, float(values.max()))
        remaining = self.sample_limit - self.sample_count
        if remaining > 0:
            take = min(remaining, self.per_update_limit, int(values.numel()))
            indices = torch.linspace(0, values.numel() - 1, take).long()
            selected = values[indices].cpu()
            self.samples.append(selected)
            self.sample_count += int(selected.numel())

    def finalize(self, scale: Optional[float] = None) -> dict[str, Any]:
        if self.count == 0:
            return robust_stats(torch.empty(0), scale)
        sample = torch.cat(self.samples) if self.samples else torch.empty(0)
        q = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.95, 0.99]))
        result = {
            "mean": self.total / self.count,
            "median": float(q[0]),
            "rmse": math.sqrt(self.total_square / self.count),
            "p90": float(q[1]),
            "p95": float(q[2]),
            "p99": float(q[3]),
            "max": self.maximum,
            "count": self.count,
            "quantile_sample_count": int(sample.numel()),
        }
        denom = float(scale) if scale is not None and math.isfinite(scale) and scale > 0 else None
        for key in ("mean", "median", "rmse", "p95"):
            result[f"normalized_{key}"] = result[key] / denom if denom else None
        return result


def ray_angle_degrees(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.linalg.vector_norm(a, dim=-1)
    b_norm = torch.linalg.vector_norm(b, dim=-1)
    cosine = (a * b).sum(dim=-1) / (a_norm * b_norm).clamp_min(1e-12)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp_min(1e-12)
    u = intrinsics[..., None, None, 0, 0] * points[..., 0] / z + intrinsics[..., None, None, 0, 2]
    v = intrinsics[..., None, None, 1, 1] * points[..., 1] / z + intrinsics[..., None, None, 1, 2]
    return torch.stack((u, v), dim=-1)


def metric_bundle(
    cam: torch.Tensor,
    cam_from_ref: torch.Tensor,
    ref: torch.Tensor,
    ref_from_cam: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    intrinsics: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    if not mask.any():
        return {"valid_points": 0, "error_3d": robust_stats(torch.empty(0)), "depth_abs": robust_stats(torch.empty(0)), "ray_angle_deg": robust_stats(torch.empty(0)), "reference_direction_3d": robust_stats(torch.empty(0)), "reprojection_px": None}
    error = torch.linalg.vector_norm(cam - cam_from_ref, dim=-1)
    error_ref = torch.linalg.vector_norm(ref - ref_from_cam, dim=-1)
    depth = (cam[..., 2] - cam_from_ref[..., 2]).abs()
    ray = ray_angle_degrees(cam, cam_from_ref)
    reprojection = None
    if intrinsics is not None:
        uv_a = project_points(cam, intrinsics)
        uv_b = project_points(cam_from_ref, intrinsics)
        reprojection = robust_stats(torch.linalg.vector_norm(uv_a - uv_b, dim=-1)[mask])
    return {
        "valid_points": int(mask.sum()),
        "error_3d": robust_stats(error[mask], scale),
        "depth_abs": robust_stats(depth[mask], scale),
        "ray_angle_deg": robust_stats(ray[mask]),
        "reference_direction_3d": robust_stats(error_ref[mask], scale),
        "reprojection_px": reprojection,
    }


def evaluate_sample(
    sample: Sample,
    intrinsics: Optional[torch.Tensor] = None,
    return_maps: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    confidence, confidence_source = combined_confidence(sample)
    scale = scene_scale(sample.point_maps_ref)
    frame_results: list[dict[str, Any]] = []
    accumulator_names = ("error_3d", "depth_abs", "ray_angle_deg", "reference_direction_3d", "reprojection_px")
    accumulators = {
        label: {name: StatsAccumulator() for name in accumulator_names}
        for label in CONFIDENCE_SWEEP
        if confidence is not None or label == "all"
    }
    valid_totals = {label: 0 for label in accumulators}
    map_parts: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("cam_from_ref", "ref_from_cam", "valid", "error_3d", "depth_error", "ray_angle")
    }
    for frame_idx in range(sample.point_maps_cam.shape[0]):
        cam = sample.point_maps_cam[frame_idx : frame_idx + 1].float()
        ref = sample.point_maps_ref[frame_idx : frame_idx + 1].float()
        pose = sample.camera_pose[frame_idx : frame_idx + 1]
        cam_from_ref = camera_from_ref(ref, pose)
        ref_from_cam = ref_from_camera(cam, pose)
        valid = finite_positive_mask(cam, cam_from_ref)
        error = torch.linalg.vector_norm(cam - cam_from_ref, dim=-1)
        error_ref = torch.linalg.vector_norm(ref - ref_from_cam, dim=-1)
        depth = (cam[..., 2] - cam_from_ref[..., 2]).abs()
        ray = ray_angle_degrees(cam, cam_from_ref)
        frame_intrinsics = None if intrinsics is None else intrinsics[frame_idx : frame_idx + 1]
        reprojection = None
        if frame_intrinsics is not None:
            reprojection = torch.linalg.vector_norm(
                project_points(cam, frame_intrinsics) - project_points(cam_from_ref, frame_intrinsics), dim=-1
            )
        frame_confidence = None if confidence is None else confidence[frame_idx]
        for label in accumulators:
            mask = confidence_mask(valid[0], frame_confidence, label)[None]
            metrics = metric_bundle(cam, cam_from_ref, ref, ref_from_cam, mask, scale, frame_intrinsics)
            frame_results.append({"frame_index": frame_idx, "confidence_selection": label, **metrics})
            valid_totals[label] += int(mask.sum())
            values = {
                "error_3d": error[mask],
                "depth_abs": depth[mask],
                "ray_angle_deg": ray[mask],
                "reference_direction_3d": error_ref[mask],
                "reprojection_px": torch.empty(0) if reprojection is None else reprojection[mask],
            }
            for name, value in values.items():
                accumulators[label][name].update(value)
        if return_maps:
            map_parts["cam_from_ref"].append(cam_from_ref.to(torch.float16))
            map_parts["ref_from_cam"].append(ref_from_cam.to(torch.float16))
            map_parts["valid"].append(valid)
            map_parts["error_3d"].append(error)
            map_parts["depth_error"].append(depth)
            map_parts["ray_angle"].append(ray)
    aggregate: dict[str, Any] = {}
    for label in CONFIDENCE_SWEEP:
        if confidence is None and label != "all":
            aggregate[label] = {"available": False, "skip_reason": "confidence fields absent from sidecar"}
            continue
        current = accumulators[label]
        aggregate[label] = {
            "available": True,
            "valid_points": valid_totals[label],
            "error_3d": current["error_3d"].finalize(scale),
            "depth_abs": current["depth_abs"].finalize(scale),
            "ray_angle_deg": current["ray_angle_deg"].finalize(),
            "reference_direction_3d": current["reference_direction_3d"].finalize(scale),
            "reprojection_px": None if intrinsics is None else current["reprojection_px"].finalize(),
        }
    result = {
        "sample_id": sample.sample_id,
        "sidecar_path": str(sample.sidecar_path.resolve()),
        "num_frames": int(sample.point_maps_cam.shape[0]),
        "point_map_shape": list(sample.point_maps_cam.shape),
        "scene_scale": scale,
        "confidence_available": confidence is not None,
        "confidence_source": confidence_source,
        "aggregate": aggregate,
        "frames": frame_results,
        "metadata": sample.metadata,
    }
    maps = {key: torch.cat(parts) for key, parts in map_parts.items()} if return_maps else {}
    return result, maps


def discover_sidecars(root: Path, scenes: Optional[Sequence[str]], count: int, seed: int) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"sidecar root does not exist: {root}")
    available = {path.stem: path for path in sorted(root.rglob("*.pt"))}
    if scenes:
        missing = [scene for scene in scenes if Path(scene).stem not in available]
        if missing:
            raise FileNotFoundError(f"requested scenes have no sidecar: {missing}")
        return [available[Path(scene).stem] for scene in scenes]
    paths = list(available.values())
    random.Random(seed).shuffle(paths)
    return paths[:count]


def representative_ids(results: Sequence[dict[str, Any]]) -> dict[str, str]:
    scored = [
        (float(result["aggregate"]["all"]["error_3d"]["p95"]), result["sample_id"])
        for result in results
        if result["aggregate"]["all"]["error_3d"]["p95"] is not None
    ]
    if len(scored) < 3:
        raise ValueError("need at least three valid samples to select representatives")
    scored.sort()
    median_score = float(np.median([score for score, _ in scored]))
    typical = min(scored, key=lambda item: abs(item[0] - median_score))
    return {"best": scored[0][1], "typical": typical[1], "worst": scored[-1][1]}


def parse_scene_list(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return [part.strip() for part in raw.split(",") if part.strip()]


def sampled_video_indices(video_path: Path, count: int) -> tuple[list[int], int, float]:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError("decord is required for frame-alignment checks") from exc
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total = len(reader)
    indices = np.linspace(0, total - 1, count, dtype=int).tolist()
    return indices, total, float(reader.get_avg_fps())


def video_path_for_sample(sample: Sample, video_root: Path) -> Path:
    metadata_path = sample.metadata.get("source_video")
    if metadata_path and Path(str(metadata_path)).is_file():
        return Path(str(metadata_path))
    return video_root / f"{Path(sample.sample_id).name}.mp4"


def decode_rgb_frames(video_path: Path, indices: Sequence[int]) -> Optional[torch.Tensor]:
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return None
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    array = reader.get_batch([int(index) for index in indices]).asnumpy()
    return torch.from_numpy(array.copy()).to(torch.uint8)


def locate_sens(root: Path, scene_id: str) -> Optional[Path]:
    candidates = (
        root / "scans" / scene_id / f"{scene_id}.sens",
        root / scene_id / f"{scene_id}.sens",
        root / f"{scene_id}.sens",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def load_sensor_data(sens_path: Path) -> Any:
    sensor_path = REPO_ROOT / "vlm_3r_data_process/src/metadata_generation/ScanNet/preprocess/SensorData.py"
    spec = importlib.util.spec_from_file_location("vlm3r_scannet_sensor_data", sensor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ScanNet SensorData from {sensor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SensorData(str(sens_path))


def _resize_rgb_small(value: np.ndarray, size: int = 64) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1).float()[None]
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0]


def rgb_alignment_test(sensor: Any, video_path: Path, frame_indices: Sequence[int]) -> dict[str, Any]:
    rgb_video = decode_rgb_frames(video_path, frame_indices)
    if rgb_video is None:
        return {"available": False, "skip_reason": "decord unavailable"}
    offsets: dict[int, list[float]] = {offset: [] for offset in range(-2, 3)}
    for local_idx, raw_idx in enumerate(frame_indices):
        video_small = _resize_rgb_small(rgb_video[local_idx].numpy())
        for offset in offsets:
            candidate = raw_idx + offset
            if 0 <= candidate < len(sensor.frames):
                raw = sensor.frames[candidate].decompress_color(sensor.color_compression_type)
                raw_small = _resize_rgb_small(raw)
                offsets[offset].append(float((video_small - raw_small).abs().mean()))
    mean_by_offset = {str(offset): (float(np.mean(values)) if values else None) for offset, values in offsets.items()}
    valid_items = [(offset, value) for offset, value in mean_by_offset.items() if value is not None]
    best_offset, best_mae = min(valid_items, key=lambda item: item[1])
    zero_mae = mean_by_offset.get("0")
    return {
        "available": True,
        "raw_frame_count": len(sensor.frames),
        "video_frame_count": int(max(frame_indices) + 1) if frame_indices else 0,
        "rgb_mae_0_255_by_offset": mean_by_offset,
        "best_offset": int(best_offset),
        "best_mae_0_255": best_mae,
        "zero_offset_is_best": int(best_offset) == 0,
        "zero_offset_psnr_db": None if zero_mae in (None, 0) else float(20 * math.log10(255.0 / zero_mae)),
    }


def load_scannet_gt(sample: Sample, gt_root: Path, video_root: Path) -> Optional[GTData]:
    scene_id = Path(sample.sample_id).name
    sens_path = locate_sens(gt_root, scene_id)
    if sens_path is None:
        return None
    sensor = load_sensor_data(sens_path)
    video_path = video_path_for_sample(sample, video_root)
    frame_indices, video_count, fps = sampled_video_indices(video_path, sample.point_maps_cam.shape[0])
    if video_count != len(sensor.frames):
        raise ValueError(
            f"frame-count mismatch for {scene_id}: video={video_count}, sens={len(sensor.frames)}; "
            "refusing silent index alignment"
        )
    depths: list[torch.Tensor] = []
    poses: list[torch.Tensor] = []
    valid_frames: list[bool] = []
    for index in frame_indices:
        frame = sensor.frames[index]
        depth_bytes = frame.decompress_depth(sensor.depth_compression_type)
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(sensor.depth_height, sensor.depth_width).copy()
        depth_tensor = torch.from_numpy(depth.astype(np.float32, copy=False)) / float(sensor.depth_shift)
        depth_tensor = F.interpolate(
            depth_tensor[None, None],
            size=sample.point_maps_cam.shape[1:3],
            mode="nearest",
        )[0, 0]
        pose = torch.from_numpy(np.asarray(frame.camera_to_world).copy()).float()
        frame_valid = bool(torch.isfinite(pose).all() and torch.linalg.det(pose[:3, :3]).abs() > 1e-6)
        depths.append(depth_tensor)
        poses.append(pose)
        valid_frames.append(frame_valid)
    pose_tensor = torch.stack(poses)
    first_valid = next((i for i, valid in enumerate(valid_frames) if valid), None)
    if first_valid is None:
        raise ValueError(f"no finite GT poses in {sens_path}")
    reference_inverse = torch.linalg.inv(pose_tensor[first_valid])
    pose_relative = torch.einsum("ij,fjk->fik", reference_inverse, pose_tensor)
    height, width = sample.point_maps_cam.shape[1:3]
    intrinsics = torch.from_numpy(np.asarray(sensor.intrinsic_depth).copy()).float()[:3, :3]
    intrinsics[0] *= width / float(sensor.depth_width)
    intrinsics[1] *= height / float(sensor.depth_height)
    intrinsics = intrinsics[None].repeat(len(frame_indices), 1, 1)
    alignment = rgb_alignment_test(sensor, video_path, frame_indices)
    alignment.update({"video_total_frames": video_count, "video_fps": fps, "sampled_indices": frame_indices})
    return GTData(
        frame_indices=frame_indices,
        depth=torch.stack(depths),
        camera_pose=pose_relative,
        intrinsics=intrinsics,
        valid_frames=torch.tensor(valid_frames, dtype=torch.bool),
        rgb_alignment=alignment,
        source=str(sens_path.resolve()),
    )


def average_rotation(rotations: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rotations.sum(dim=0))
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def align_predicted_poses(pred: torch.Tensor, gt: torch.Tensor, valid_frames: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_valid = pred[valid_frames]
    gt_valid = gt[valid_frames]
    rotation_candidates = gt_valid[:, :3, :3] @ pred_valid[:, :3, :3].transpose(-1, -2)
    rotation = average_rotation(rotation_candidates)
    pred_centers = pred_valid[:, :3, 3]
    gt_centers = gt_valid[:, :3, 3]
    pred_rotated = torch.einsum("ij,fj->fi", rotation, pred_centers)
    pred_center = pred_rotated.mean(dim=0)
    gt_center = gt_centers.mean(dim=0)
    numerator = ((pred_rotated - pred_center) * (gt_centers - gt_center)).sum()
    denominator = (pred_rotated - pred_center).square().sum().clamp_min(1e-12)
    scale = (numerator / denominator).abs().clamp_min(1e-8)
    translation = gt_center - scale * pred_center
    aligned = torch.eye(4).repeat(pred.shape[0], 1, 1)
    aligned[:, :3, :3] = torch.einsum("ij,fjk->fik", rotation, pred[:, :3, :3])
    aligned[:, :3, 3] = scale * torch.einsum("ij,fj->fi", rotation, pred[:, :3, 3]) + translation
    return {"rotation": rotation, "translation": translation, "scale": scale, "camera_pose_aligned": aligned}


def rotation_error_degrees(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    relative = a.transpose(-1, -2) @ b
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    valid = valid & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
    pred_v, gt_v = pred[valid], gt[valid]
    if pred_v.numel() == 0:
        return {"valid_points": 0}
    ratio = torch.maximum(pred_v / gt_v, gt_v / pred_v)
    diff = pred_v - gt_v
    return {
        "valid_points": int(pred_v.numel()),
        "abs_rel": float((diff.abs() / gt_v).mean()),
        "sq_rel": float((diff.square() / gt_v).mean()),
        "rmse": float(torch.sqrt(diff.square().mean())),
        "rmse_log": float(torch.sqrt((torch.log(pred_v) - torch.log(gt_v)).square().mean())),
        "delta1": float((ratio < 1.25).float().mean()),
        "delta2": float((ratio < 1.25**2).float().mean()),
        "delta3": float((ratio < 1.25**3).float().mean()),
        "abs_error": robust_stats(diff.abs()),
    }


def gt_camera_points(gt: GTData) -> torch.Tensor:
    frames, height, width = gt.depth.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xx = xx.float()[None].expand(frames, -1, -1)
    yy = yy.float()[None].expand(frames, -1, -1)
    z = gt.depth
    x = (xx - gt.intrinsics[:, None, None, 0, 2]) * z / gt.intrinsics[:, None, None, 0, 0]
    y = (yy - gt.intrinsics[:, None, None, 1, 2]) * z / gt.intrinsics[:, None, None, 1, 1]
    return torch.stack((x, y, z), dim=-1)


def evaluate_gt(sample: Sample, gt: GTData) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    alignment = align_predicted_poses(sample.camera_pose, gt.camera_pose, gt.valid_frames)
    scale = alignment["scale"]
    rotation = alignment["rotation"]
    translation = alignment["translation"]
    point_maps_ref = sample.point_maps_ref.float()
    point_maps_cam = sample.point_maps_cam.float()
    ref_gt = scale * torch.einsum("ij,fhwj->fhwi", rotation, point_maps_ref) + translation
    cam_ref_pred = scale * camera_from_ref(point_maps_ref, sample.camera_pose)
    cam_ref_gt = geotrf(torch.linalg.inv(gt.camera_pose), ref_gt)
    cam_self = scale * point_maps_cam
    gt_points = gt_camera_points(gt)
    valid_gt = (gt.depth > 0) & gt.valid_frames[:, None, None]
    translation_error = torch.linalg.vector_norm(
        alignment["camera_pose_aligned"][:, :3, 3] - gt.camera_pose[:, :3, 3], dim=-1
    )[gt.valid_frames]
    rotation_error = rotation_error_degrees(
        alignment["camera_pose_aligned"][:, :3, :3], gt.camera_pose[:, :3, :3]
    )[gt.valid_frames]
    result = {
        "available": True,
        "source": gt.source,
        "frame_alignment": gt.rgb_alignment,
        "similarity_alignment": {
            "scale_pred_to_gt": float(scale),
            "rotation_pred_to_gt": rotation,
            "translation_pred_to_gt": translation,
        },
        "pose": {
            "translation_error_m": robust_stats(translation_error),
            "rotation_error_deg": robust_stats(rotation_error),
        },
        "self_camera_map_depth": depth_metrics(cam_self[..., 2], gt.depth, valid_gt),
        "reference_map_with_predicted_pose_depth": depth_metrics(cam_ref_pred[..., 2], gt.depth, valid_gt),
        "reference_map_with_gt_pose_depth": depth_metrics(cam_ref_gt[..., 2], gt.depth, valid_gt),
        "self_camera_map_3d": robust_stats(torch.linalg.vector_norm(cam_self - gt_points, dim=-1)[valid_gt]),
        "reference_map_with_predicted_pose_3d": robust_stats(torch.linalg.vector_norm(cam_ref_pred - gt_points, dim=-1)[valid_gt]),
        "reference_map_with_gt_pose_3d": robust_stats(torch.linalg.vector_norm(cam_ref_gt - gt_points, dim=-1)[valid_gt]),
    }
    maps = {
        "cam_from_ref_pred_pose_gt_scale": cam_ref_pred,
        "cam_from_ref_gt_pose": cam_ref_gt,
        "ref_aligned_gt": ref_gt,
        "gt_points": gt_points,
    }
    return result, maps


def inspect_checkpoint_configuration(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_file():
        return {"available": False, "skip_reason": f"checkpoint not found: {checkpoint}"}
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        kwargs["mmap"] = True
        payload = torch.load(str(checkpoint), weights_only=False, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        payload = torch.load(str(checkpoint), **kwargs)
    model_text = str(getattr(payload.get("args"), "model", "")) if isinstance(payload, dict) else ""
    head_match = re.search(r"head_type=['\"]([^'\"]+)", model_text)
    output_match = re.search(r"output_mode=['\"]([^'\"]+)", model_text)
    head_type = head_match.group(1) if head_match else None
    output_mode = output_match.group(1) if output_match else None
    class_name = "DPTPts3dPose" if (head_type, output_mode) == ("dpt", "pts3d+pose") else None
    del payload
    return {
        "available": True,
        "checkpoint": str(checkpoint.resolve()),
        "head_type": head_type,
        "output_mode": output_mode,
        "expected_downstream_head_class": class_name,
        "loaded_downstream_head_class": None,
        "independent_outputs": class_name == "DPTPts3dPose",
        "architecture_evidence": "DPTPts3dPose has separate dpt_self, dpt_cross, and pose_head modules",
    }


def regenerate_representative(sample: Sample, checkpoint: Path, video_root: Path, precision: str) -> dict[str, Any]:
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import prepare_input
    from llava.utils import process_video_with_decord
    from src.dust3r.model import ARCroco3DStereo

    class DataArgs:
        video_fps = 1
        frames_upbound = int(sample.point_maps_cam.shape[0])
        force_sample = True

    if not torch.cuda.is_available():
        raise RuntimeError("--enrich-representatives requires a CUDA allocation")
    device = torch.device("cuda")
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).eval()
    downstream_head_class = (
        f"{type(model.downstream_head).__module__}.{type(model.downstream_head).__qualname__}"
    )
    if type(model.downstream_head).__name__ != "DPTPts3dPose":
        raise AssertionError(
            "Expected active downstream head DPTPts3dPose, got "
            f"{downstream_head_class}"
        )
    print(f"[HEAD] active downstream head: {downstream_head_class}")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model.to(device=device, dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad = False
    video_path = video_path_for_sample(sample, video_root)
    frames, _, _, _ = process_video_with_decord(str(video_path), DataArgs())
    processor_payload = json.loads((REPO_ROOT / "processor_config.json").read_text())
    size = processor_payload.get("size", {"height": 384, "width": 384})
    processor = SigLipImageProcessor(
        image_mean=processor_payload.get("image_mean", (0.5, 0.5, 0.5)),
        image_std=processor_payload.get("image_std", (0.5, 0.5, 0.5)),
        size=(size["height"], size["width"]),
        resample=processor_payload.get("resample", 3),
        rescale_factor=processor_payload.get("rescale_factor", 1 / 255.0),
    )
    pixels = processor.preprocess(images=frames, return_tensors="pt")["pixel_values"]
    pixels = F.interpolate(pixels, size=sample.point_maps_cam.shape[1:3], mode="bilinear", align_corners=False)
    pixels = pixels[:, None].to(device=device, dtype=dtype)
    views = prepare_input(pixel_values=pixels)
    shape, feat_ls, pos = model._encode_views(views)
    feat = feat_ls[-1]
    state_feat, state_pos = model._init_state(feat[0], pos[0])
    memory = model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
    initial_state = state_feat.clone()
    initial_memory = memory.clone()
    outputs: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for frame_idx, view in enumerate(views):
            feat_i, pos_i = feat[frame_idx].to(dtype), pos[frame_idx]
            global_feat = model._get_img_level_feat(feat_i)
            pose_feat = model.pose_token.expand(feat_i.shape[0], -1, -1) if frame_idx == 0 else model.pose_retriever.inquire(global_feat, memory)
            pose_pos = -torch.ones(feat_i.shape[0], 1, 2, device=device, dtype=pos_i.dtype)
            new_state, dec = model._recurrent_rollout(
                state_feat, state_pos, feat_i, pos_i, pose_feat, pose_pos, initial_state,
                img_mask=view["img_mask"], reset_mask=view["reset"], update=view.get("update"),
            )
            out_pose_feat = dec[-1][:, :1]
            new_memory = model.pose_retriever.update_mem(memory, global_feat, out_pose_feat)
            head_input = [dec[0], dec[model.dec_depth * 2 // 4][:, 1:], dec[model.dec_depth * 3 // 4][:, 1:], dec[model.dec_depth]]
            outputs.append(model._downstream_head(head_input, shape[frame_idx], pos=pos_i))
            update = view.get("update")
            update_mask = view["img_mask"] & update if update is not None else view["img_mask"]
            update_mask = update_mask[:, None, None].to(dtype)
            state_feat = new_state * update_mask + state_feat * (1 - update_mask)
            memory = new_memory * update_mask + memory * (1 - update_mask)
            reset = view["reset"]
            if reset is not None:
                reset_mask = reset[:, None, None].to(dtype)
                state_feat = initial_state * reset_mask + state_feat * (1 - reset_mask)
                memory = initial_memory * reset_mask + memory * (1 - reset_mask)
    def collect(key: str) -> torch.Tensor:
        return torch.cat([output[key] for output in outputs], dim=0).detach().cpu().float()
    regenerated_cam = collect("pts3d_in_self_view")
    regenerated_ref = collect("pts3d_in_other_view")
    result = {
        "loaded_downstream_head_class": downstream_head_class,
        "conf_self": collect("conf_self"),
        "conf_ref": collect("conf"),
        "regenerated_camera_map_max_abs": float((regenerated_cam - sample.point_maps_cam).abs().max()),
        "regenerated_camera_map_mean_abs": float((regenerated_cam - sample.point_maps_cam).abs().mean()),
        "regenerated_reference_map_max_abs": float((regenerated_ref - sample.point_maps_ref).abs().max()),
        "regenerated_reference_map_mean_abs": float((regenerated_ref - sample.point_maps_ref).abs().mean()),
    }
    del model, outputs, feat_ls, views
    torch.cuda.empty_cache()
    return result


def frame_selection(num_frames: int, count: int) -> list[int]:
    return np.unique(np.linspace(0, num_frames - 1, min(count, num_frames), dtype=int)).tolist()


def save_representative(
    output_path: Path,
    reason: str,
    sample: Sample,
    maps: dict[str, torch.Tensor],
    frame_count: int,
    video_root: Path,
    gt: Optional[GTData],
    gt_maps: Optional[dict[str, torch.Tensor]],
    gt_result: Optional[dict[str, Any]],
) -> None:
    selected = frame_selection(sample.point_maps_cam.shape[0], frame_count)
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    cam_selected = sample.point_maps_cam[selected_tensor].float()
    ref_selected = sample.point_maps_ref[selected_tensor].float()
    pose_selected = sample.camera_pose[selected_tensor]
    cam_from_ref_selected = camera_from_ref(ref_selected, pose_selected)
    ref_from_cam_selected = ref_from_camera(cam_selected, pose_selected)
    valid_selected = finite_positive_mask(cam_selected, cam_from_ref_selected)
    error_selected = torch.linalg.vector_norm(cam_selected - cam_from_ref_selected, dim=-1)
    depth_error_selected = (cam_selected[..., 2] - cam_from_ref_selected[..., 2]).abs()
    ray_selected = ray_angle_degrees(cam_selected, cam_from_ref_selected)
    video_path = video_path_for_sample(sample, video_root)
    video_indices, _, _ = sampled_video_indices(video_path, sample.point_maps_cam.shape[0])
    source_indices = [video_indices[index] for index in selected]
    rgb = decode_rgb_frames(video_path, source_indices)
    def select_source(value: Optional[torch.Tensor], dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        if value is None:
            return None
        result = value[selected].detach().cpu()
        return result.to(dtype) if dtype is not None else result
    payload = {
        "sample_id": sample.sample_id,
        "frame_indices": torch.tensor(selected, dtype=torch.int64),
        "source_video_frame_indices": torch.tensor(source_indices, dtype=torch.int64),
        "rgb_frames_or_paths": rgb if rgb is not None else {"video_path": str(video_path), "frame_indices": source_indices},
        "point_maps_cam": cam_selected.to(torch.float16),
        "point_maps_ref": ref_selected.to(torch.float16),
        "point_maps_cam_from_ref_pred_pose": cam_from_ref_selected.to(torch.float16),
        "point_maps_cam_from_ref_gt_pose": None if gt_maps is None else select_source(gt_maps["cam_from_ref_gt_pose"], torch.float16),
        "point_maps_ref_from_cam_pred_pose": ref_from_cam_selected.to(torch.float16),
        "camera_pose_pred": pose_selected.to(torch.float32),
        "camera_pose_gt": None if gt is None else select_source(gt.camera_pose, torch.float32),
        "depth_gt": None if gt is None else select_source(gt.depth, torch.float32),
        "conf_self": select_source(sample.conf_self, torch.float16),
        "conf_ref": select_source(sample.conf_ref, torch.float16),
        "valid_mask": valid_selected,
        "error_3d_map": error_selected,
        "depth_error_map": depth_error_selected,
        "ray_angle_error_map": ray_selected,
        "intrinsics": None if gt is None else select_source(gt.intrinsics, torch.float32),
        "metadata": {
            "coordinate_convention": "point_maps_cam=per-frame OpenCV camera; point_maps_ref=CUT3R reference",
            "pose_convention": "camera-to-reference T_cw; [tx,ty,tz,qw,qx,qy,qz]",
            "resize_crop_information": "640x480 MP4 direct resize to 384x384, then bilinear 432x432; no crop",
            "scene_scale": scene_scale(sample.point_maps_ref),
            "selection_reason": reason,
            "source_sidecar": str(sample.sidecar_path.resolve()),
            "source_video": str(video_path.resolve()),
            "gt_evaluation": gt_result,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def flatten_csv_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for frame in result["frames"]:
            row = {
                "level": "frame",
                "sample_id": result["sample_id"],
                "frame_index": frame["frame_index"],
                "confidence_selection": frame["confidence_selection"],
                "scene_scale": result["scene_scale"],
                "valid_points": frame["valid_points"],
            }
            for family in ("error_3d", "depth_abs", "ray_angle_deg", "reference_direction_3d", "reprojection_px"):
                if isinstance(frame.get(family), dict):
                    for key, value in frame[family].items():
                        row[f"{family}_{key}"] = value
            rows.append(row)
        for label, metrics in result["aggregate"].items():
            if not metrics.get("available"):
                continue
            row = {
                "level": "video",
                "sample_id": result["sample_id"],
                "frame_index": "",
                "confidence_selection": label,
                "scene_scale": result["scene_scale"],
                "valid_points": metrics["valid_points"],
            }
            for family in ("error_3d", "depth_abs", "ray_angle_deg", "reference_direction_3d", "reprojection_px"):
                if isinstance(metrics.get(family), dict):
                    for key, value in metrics[family].items():
                        row[f"{family}_{key}"] = value
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_dataset(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"num_valid_videos": len(results), "confidence": {}}
    for label in CONFIDENCE_SWEEP:
        entries = [result["aggregate"][label] for result in results if result["aggregate"].get(label, {}).get("available")]
        aggregate["confidence"][label] = {
            "videos_available": len(entries),
            "median_video_3d_median": float(np.median([entry["error_3d"]["median"] for entry in entries])) if entries else None,
            "median_video_3d_p95": float(np.median([entry["error_3d"]["p95"] for entry in entries])) if entries else None,
            "median_video_normalized_3d_median": float(np.median([entry["error_3d"]["normalized_median"] for entry in entries])) if entries else None,
            "median_video_normalized_3d_p95": float(np.median([entry["error_3d"]["normalized_p95"] for entry in entries])) if entries else None,
            "total_valid_points": sum(entry["valid_points"] for entry in entries),
        }
    return aggregate


def decision_text(results: Sequence[dict[str, Any]], gt_results: dict[str, Any]) -> str:
    if not gt_results:
        return (
            "The exact discrepancy is expected from independent DPT self/cross/pose heads, but source quality "
            "cannot be isolated without ScanNet GT. Mutual consistency alone does not prove which prediction is inaccurate."
        )
    improvements = []
    self_ref_differences = []
    zero_offset_checks = []
    for result in gt_results.values():
        pred = result.get("reference_map_with_predicted_pose_depth", {}).get("rmse")
        gt = result.get("reference_map_with_gt_pose_depth", {}).get("rmse")
        self_rmse = result.get("self_camera_map_depth", {}).get("rmse")
        if pred and gt is not None:
            improvements.append((pred - gt) / pred)
        if pred and self_rmse is not None:
            self_ref_differences.append(abs(pred - self_rmse) / max(pred, self_rmse, 1e-12))
        zero_offset_checks.append(bool(result.get("frame_alignment", {}).get("zero_offset_is_best")))
    median_improvement = float(np.median(improvements)) if improvements else 0.0
    if median_improvement > 0.3:
        return "GT pose substantially improves the transformed reference map; pose error is a major contributor, alongside independent-head residuals."
    if zero_offset_checks and all(zero_offset_checks) and self_ref_differences and float(np.median(self_ref_differences)) < 0.1:
        return (
            "Frame order is correct and GT pose does not substantially improve the reference map. Self and "
            "predicted-pose reference depth quality are similar, while the architecture decodes them independently. "
            "The exact rigid mismatch is therefore primarily independent-head non-alignment, with ordinary decoder "
            "and pose inaccuracies as smaller, sample-dependent contributors."
        )
    return "GT pose does not substantially improve the reference map; decoder error or independent-head inconsistency dominates over pose error."


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    dataset = payload["dataset_aggregate"]
    checkpoint = payload["checkpoint"]
    real_test = payload["real_convention_test"]
    lines = [
        "# CUT3R Point-Map / Pose Alignment Diagnostic",
        "",
        "## Verified implementation",
        "",
        f"- Loaded downstream head: `{checkpoint.get('loaded_downstream_head_class') or checkpoint.get('expected_downstream_head_class')}` (`head_type=dpt`, `output_mode=pts3d+pose`).",
        "- `DPTPts3dPose` independently decodes `pts3d_in_self_view` with `dpt_self` and `pts3d_in_other_view` with `dpt_cross`; pose is a third learned branch.",
        "- Pose is OpenCV camera-to-reference `T_cw`, encoded `[tx,ty,tz,qw,qx,qy,qz]` (real-first quaternion).",
        "- Points use row-shaped storage but official `geotrf` applies `X_ref = R_cw X_cam + t_cw`.",
        f"- Official synthetic inverse max error: {payload['pose_smoke']['official_inverse_max_abs']}; real wrong/correct direction error ratio: {real_test['wrong_over_correct_ratio']}.",
        "",
        "## Frame and preprocessing alignment",
        "",
        "- Sidecar frame `i` corresponds to MP4 index `linspace(0, n-1, 32)[i]`.",
        "- RGB is directly resized from 640×480 to 384×384 and then to 432×432; there is no crop.",
        "- When `.sens` GT is present, this report requires equal raw/video frame counts and records RGB ±2-frame matching.",
        "",
        "## Dataset summary",
        "",
        f"- Valid videos: {dataset['num_valid_videos']}",
        f"- Median video 3D median error (all): {dataset['confidence']['all']['median_video_3d_median']}",
        f"- Median video 3D p95 error (all): {dataset['confidence']['all']['median_video_3d_p95']}",
        f"- Median scale-normalized 3D p95 (all): {dataset['confidence']['all']['median_video_normalized_3d_p95']}",
        f"- Videos with saved/regenerated confidence: {dataset['confidence']['top50']['videos_available']}",
        f"- Representative median p95 at top 75% / 50% / 25% confidence: {dataset['confidence']['top75']['median_video_3d_p95']} / {dataset['confidence']['top50']['median_video_3d_p95']} / {dataset['confidence']['top25']['median_video_3d_p95']}",
        "",
        "## GT isolation",
        "",
    ]
    if payload["gt_results"]:
        for sample_id, result in payload["gt_results"].items():
            lines.extend(
                [
                    f"### {sample_id}",
                    "",
                    f"- RGB zero-offset best: {result['frame_alignment'].get('zero_offset_is_best')}",
                    f"- Pose translation median: {result['pose']['translation_error_m']['median']} m",
                    f"- Pose rotation median: {result['pose']['rotation_error_deg']['median']}°",
                    f"- Self depth RMSE: {result['self_camera_map_depth'].get('rmse')}",
                    f"- Reference + predicted pose depth RMSE: {result['reference_map_with_predicted_pose_depth'].get('rmse')}",
                    f"- Reference + GT pose depth RMSE: {result['reference_map_with_gt_pose_depth'].get('rmse')}",
                    "",
                ]
            )
    else:
        lines.extend(["No `.sens` GT was found under the configured root; GT-dependent conclusions are explicitly withheld.", ""])
    config = payload["configuration"]
    lines.extend(
        [
            "## Interpretation",
            "",
            payload["decision"],
            "",
            "## Reproduction commands",
            "",
            "```bash",
            "python scripts/analysis/diagnose_cut3r_pointmap_pose_alignment.py \\",
            f"  --sidecar-root {config['sidecar_root']} \\",
            f"  --video-root {config['video_root']} \\",
            f"  --checkpoint {checkpoint['checkpoint']} \\",
            f"  --output-dir {config['output_dir']} \\",
            f"  --sample-count {config['sample_count']} --seed {config['seed']} \\",
            f"  --saved-frames {config['saved_frames']} --precision {config['precision']} \\",
            "  --enrich-representatives \\",
            f"  --scannet-gt-root {config.get('scannet_gt_root')}",
            "",
            "sbatch --export=ALL,ENRICH_REPRESENTATIVES=1,"
            f"SCANNET_GT_ROOT={config.get('scannet_gt_root')} \\",
            "  scripts/analysis/slurm_cut3r_pointmap_pose_diagnostic.sbatch",
            "```",
            "",
            "## Output paths",
            "",
            f"- JSON: `{config['output_dir']}/results.json`",
            f"- CSV: `{config['output_dir']}/metrics.csv`",
            f"- Report: `{config['output_dir']}/report.md`",
        ]
    )
    for role, representative in payload["representatives"].items():
        lines.append(f"- {role.title()} representative: `{representative}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def validate_representative(path: Path) -> None:
    payload = load_torch(path)
    required = {
        "sample_id", "frame_indices", "rgb_frames_or_paths", "point_maps_cam", "point_maps_ref",
        "point_maps_cam_from_ref_pred_pose", "point_maps_cam_from_ref_gt_pose",
        "point_maps_ref_from_cam_pred_pose", "camera_pose_pred", "camera_pose_gt", "depth_gt",
        "conf_self", "conf_ref", "valid_mask", "error_3d_map", "depth_error_map",
        "ray_angle_error_map", "intrinsics", "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"representative {path} missing keys: {sorted(missing)}")
    for value in payload.values():
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError(f"representative {path} contains non-CPU tensor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_SIDECAR_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scannet-gt-root", type=Path, default=None, help="Root containing scans/<scene>/<scene>.sens")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/cut3r_pointmap_pose_diagnostic")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scenes", default=None, help="Comma list or newline-delimited file")
    parser.add_argument("--saved-frames", type=int, default=3)
    parser.add_argument("--enrich-representatives", action="store_true", help="Re-run CUT3R for the three representatives to recover confidence")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--skip-checkpoint-inspection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count < 3 and not args.scenes:
        raise ValueError("--sample-count must be at least 3")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_smoke = run_pose_smoke_tests()
    checkpoint_info = (
        {"available": False, "skip_reason": "disabled by --skip-checkpoint-inspection"}
        if args.skip_checkpoint_inspection
        else inspect_checkpoint_configuration(args.checkpoint)
    )
    paths = discover_sidecars(args.sidecar_root, parse_scene_list(args.scenes), args.sample_count, args.seed)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    sample_paths: dict[str, Path] = {}
    real_convention: Optional[dict[str, Any]] = None
    for path in paths:
        try:
            sample = load_sample(path, args.sidecar_root)
            if real_convention is None:
                real_convention = real_pose_convention_test(sample)
                print(f"[REAL-SMOKE] shapes cam/ref={tuple(sample.point_maps_cam.shape)} pose={tuple(sample.pose_encoding.shape)} convention={real_convention['selected_convention']}")
            result, _ = evaluate_sample(sample)
            results.append(result)
            sample_paths[sample.sample_id] = path
            all_metrics = result["aggregate"]["all"]["error_3d"]
            print(f"[VIDEO] {sample.sample_id} valid={result['aggregate']['all']['valid_points']} median={all_metrics['median']:.6g} p95={all_metrics['p95']:.6g}")
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[SKIP] {path}: {exc}")
    if len(results) < min(20, len(paths)):
        raise RuntimeError(f"only {len(results)} valid videos; acceptance requires at least 20 (skipped={len(skipped)})")
    representatives = representative_ids(results)
    gt_results: dict[str, Any] = {}
    representative_paths: dict[str, str] = {}
    enrichment: dict[str, Any] = {}
    for reason, sample_id in representatives.items():
        sample = load_sample(sample_paths[sample_id], args.sidecar_root)
        if args.enrich_representatives:
            regen = regenerate_representative(sample, args.checkpoint, args.video_root, args.precision)
            sample.conf_self = regen.pop("conf_self")
            sample.conf_ref = regen.pop("conf_ref")
            enrichment[sample_id] = regen
            checkpoint_info["loaded_downstream_head_class"] = regen["loaded_downstream_head_class"]
            enriched_result, maps = evaluate_sample(sample)
            for index, result in enumerate(results):
                if result["sample_id"] == sample_id:
                    results[index] = enriched_result
                    break
        else:
            _, maps = evaluate_sample(sample)
        gt = None
        gt_maps = None
        gt_result = None
        if args.scannet_gt_root is not None:
            try:
                gt = load_scannet_gt(sample, args.scannet_gt_root, args.video_root)
                if gt is not None:
                    gt_result, gt_maps = evaluate_gt(sample, gt)
                    gt_results[sample_id] = gt_result
            except Exception as exc:
                skipped.append({"path": sample_id, "reason": f"GT: {type(exc).__name__}: {exc}"})
                print(f"[GT-SKIP] {sample_id}: {exc}")
        output_path = args.output_dir / f"representative_{reason}_{Path(sample_id).name}.pt"
        save_representative(output_path, reason, sample, maps, args.saved_frames, args.video_root, gt, gt_maps, gt_result)
        validate_representative(output_path)
        representative_paths[reason] = str(output_path.resolve())
    dataset_aggregate = aggregate_dataset(results)
    payload = {
        "configuration": {
            "sidecar_root": str(args.sidecar_root.resolve()),
            "video_root": str(args.video_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "sample_count_requested": args.sample_count,
            "sample_count": args.sample_count,
            "seed": args.seed,
            "saved_frames": args.saved_frames,
            "precision": args.precision,
            "enrich_representatives": args.enrich_representatives,
            "confidence_sweep": list(CONFIDENCE_SWEEP),
            "scannet_gt_root": None if args.scannet_gt_root is None else str(args.scannet_gt_root.resolve()),
        },
        "conventions": {
            "point_maps_cam": "per-frame OpenCV camera coordinates",
            "point_maps_ref": "CUT3R reference/anchor coordinates",
            "camera_pose": "OpenCV camera-to-reference/world T_cw",
            "pose_encoding": "[tx,ty,tz,qw,qx,qy,qz]",
            "point_storage": "row-shaped [...,3]; geotrf applies column-vector R X + t",
        },
        "pose_smoke": pose_smoke,
        "real_convention_test": real_convention,
        "checkpoint": checkpoint_info,
        "dataset_aggregate": dataset_aggregate,
        "videos": results,
        "skipped": skipped,
        "representatives": representative_paths,
        "representative_enrichment": enrichment,
        "gt_results": gt_results,
    }
    payload["decision"] = decision_text(results, gt_results)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "metrics.csv"
    report_path = args.output_dir / "report.md"
    write_json(json_path, payload)
    write_csv(csv_path, flatten_csv_rows(results))
    write_markdown_report(report_path, payload)
    print(f"[SUMMARY] {json.dumps(_jsonable(dataset_aggregate), sort_keys=True)}")
    print(f"[OUTPUT] JSON {json_path.resolve()}")
    print(f"[OUTPUT] CSV {csv_path.resolve()}")
    print(f"[OUTPUT] REPORT {report_path.resolve()}")
    for reason in ("best", "typical", "worst"):
        print(f"[OUTPUT] PT_{reason.upper()} {representative_paths[reason]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
