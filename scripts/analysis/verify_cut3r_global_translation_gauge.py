#!/usr/bin/env python3
"""Verify CUT3R's global-translation gauge on extracted point-map sidecars.

CUT3R point-map sidecars written by ``scripts/extraction/extract_cut3r_point_maps.py``
have this explicit coordinate contract:

* ``point_maps_cam``: [F, H, W, 3] per-frame camera points;
* ``point_maps_ref``: [F, H, W, 3] reference/anchor-frame points;
* ``camera_pose``: [F, 7] = (tx, ty, tz, qw, qx, qy, qz).

The CUT3R pose helper defines this seven-value pose as an OpenCV camera-to-world
(here, camera-to-reference) transform.  Points are stored as row vectors in the
last tensor dimension, but are transformed with the usual column-vector relation
``X_ref = R_cw @ X_cam + t_cw``.  Thus the inverse used below is
``X_cam = R_cw.T @ (X_ref - t_cw)``.

This is an opt-in analysis only: it never edits sidecars or runs model inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_TRANSLATIONS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, -0.5, 0.25),
    (-0.75, 0.4, 1.2),
    (0.3, 1.1, -0.6),
)
REFERENCE_KEYS = ("point_maps_ref", "pts3d_in_other_view")
CAMERA_KEYS = ("point_maps_cam", "pts3d_in_self_view")
CONFIDENCE_KEYS = ("confidence", "conf", "depth_conf", "pts3d_conf")
DEPTH_KEYS = ("depth", "depth_map")


@dataclass
class GeometrySample:
    identifier: str
    world: torch.Tensor
    camera: torch.Tensor
    pose: torch.Tensor
    confidence: Optional[torch.Tensor]
    explicit_depth: Optional[torch.Tensor]
    source: str


def _as_fhwc(value: torch.Tensor, name: str) -> torch.Tensor:
    """Return one sample's point map as FP32 [F,H,W,3]."""
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError(f"{name} has batch={value.shape[0]}; expected a single-sample sidecar")
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"{name} must be [F,H,W,3] or [F,3,H,W], got {tuple(value.shape)}")
    if value.shape[-1] == 3:
        return value.detach().to(dtype=torch.float32, device="cpu")
    if value.shape[1] == 3:
        return value.permute(0, 2, 3, 1).detach().to(dtype=torch.float32, device="cpu")
    raise ValueError(f"{name} has no 3-coordinate channel, got {tuple(value.shape)}")


def _as_pose(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"camera_pose has batch={value.shape[0]}; expected [F,7]")
        value = value[0]
    if value.ndim != 2 or value.shape[-1] != 7:
        raise ValueError(
            "camera_pose must use the CUT3R [F,7] absT_quaR encoding "
            f"(tx,ty,tz,qw,qx,qy,qz), got {tuple(value.shape)}"
        )
    pose = value.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(pose).all():
        raise ValueError("camera_pose contains non-finite values")
    quaternion_norm = pose[:, 3:7].norm(dim=-1)
    if torch.any(quaternion_norm <= 1e-8):
        raise ValueError("camera_pose contains a zero-norm quaternion")
    return pose


def _as_fhw(value: torch.Tensor, name: str) -> Optional[torch.Tensor]:
    if value.ndim == 5:
        if value.shape[0] != 1:
            return None
        value = value[0]
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        return None
    return value.detach().to(dtype=torch.float32, device="cpu")


def _first_tensor(payload: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value, key
    return None, None


def load_sidecar(path: Path, sidecar_root: Optional[Path] = None) -> GeometrySample:
    try:
        # Sidecars are local artifacts produced by this repository, not untrusted downloads.
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"cannot load torch sidecar: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"sidecar must be a dict, got {type(payload).__name__}")

    world_raw, world_key = _first_tensor(payload, REFERENCE_KEYS)
    camera_raw, camera_key = _first_tensor(payload, CAMERA_KEYS)
    pose_raw, pose_key = _first_tensor(payload, ("camera_pose",))
    missing = [
        name
        for name, value in (("reference point map", world_raw), ("camera point map", camera_raw), ("camera_pose", pose_raw))
        if value is None
    ]
    if missing:
        raise ValueError(f"missing required geometry: {', '.join(missing)}; keys={sorted(payload.keys())}")

    world = _as_fhwc(world_raw, str(world_key))
    camera = _as_fhwc(camera_raw, str(camera_key))
    pose = _as_pose(pose_raw)
    if world.shape != camera.shape:
        raise ValueError(f"world/camera point-map shape mismatch: {tuple(world.shape)} vs {tuple(camera.shape)}")
    if world.shape[0] != pose.shape[0]:
        raise ValueError(f"frame mismatch: point maps have F={world.shape[0]}, poses have F={pose.shape[0]}")

    confidence_raw, _ = _first_tensor(payload, CONFIDENCE_KEYS)
    confidence = _as_fhw(confidence_raw, "confidence") if confidence_raw is not None else None
    if confidence is not None and confidence.shape != world.shape[:3]:
        confidence = None  # A non-dense confidence map cannot be safely applied pixelwise.

    depth_raw, _ = _first_tensor(payload, DEPTH_KEYS)
    explicit_depth = _as_fhw(depth_raw, "depth") if depth_raw is not None else None
    if explicit_depth is not None and explicit_depth.shape != world.shape[:3]:
        explicit_depth = None

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    source_video = metadata.get("source_video")
    identifier = Path(str(source_video)).stem if source_video else path.stem
    if sidecar_root is not None:
        try:
            source = str(path.relative_to(sidecar_root))
        except ValueError:
            source = path.name
    else:
        source = path.name
    return GeometrySample(identifier, world, camera, pose, confidence, explicit_depth, source)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions [F,4] to rotation matrices [F,3,3]."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def pose_to_c2w(pose: torch.Tensor) -> torch.Tensor:
    matrices = torch.eye(4, dtype=torch.float32).repeat(pose.shape[0], 1, 1)
    matrices[:, :3, :3] = quaternion_to_matrix(pose[:, 3:7])
    matrices[:, :3, 3] = pose[:, :3]
    return matrices


def reference_to_camera(world: torch.Tensor, pose_cw: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of CUT3R's camera-to-reference pose in FP32."""
    rotation = quaternion_to_matrix(pose_cw[:, 3:7])
    centered = world - pose_cw[:, None, None, :3]
    return torch.einsum("fji,fhwj->fhwi", rotation, centered)


def dense_valid_mask(sample: GeometrySample) -> torch.Tensor:
    valid = torch.isfinite(sample.world).all(dim=-1)
    valid &= torch.isfinite(sample.camera).all(dim=-1)
    valid &= sample.camera[..., 2] > 0  # Repository depth convention: positive camera z.
    if sample.confidence is not None:
        valid &= torch.isfinite(sample.confidence) & (sample.confidence > 0)
    return valid


def error_stats(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    diff = (a - b)[valid]
    if diff.numel() == 0:
        raise ValueError("no valid entries remain for comparison")
    abs_diff = diff.abs()
    return {
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt((diff.square()).mean()).item()),
    }


def scalar_error_stats(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    diff = (a - b)[valid]
    if diff.numel() == 0:
        raise ValueError("no valid depth pixels remain for comparison")
    abs_diff = diff.abs()
    return {"max_abs": float(abs_diff.max().item()), "mean_abs": float(abs_diff.mean().item())}


def evaluate_sample(
    sample: GeometrySample,
    translation: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    delta = torch.tensor(translation, dtype=torch.float32)
    if delta.shape != (3,) or torch.all(delta == 0):
        raise ValueError(f"translation must be a nonzero three-vector, got {list(translation)}")
    valid = dense_valid_mask(sample)
    valid_count = int(valid.sum().item())
    total_count = int(valid.numel())
    if valid_count == 0:
        raise ValueError("no valid finite point-map/depth pixels")

    original_camera = reference_to_camera(sample.world, sample.pose)
    translated_world = sample.world.clone()
    translated_world[valid] += delta
    translated_pose = sample.pose.clone()
    translated_pose[:, :3] += delta
    translated_camera = reference_to_camera(translated_world, translated_pose)
    point_only_camera = reference_to_camera(translated_world, sample.pose)

    original_depth = original_camera[..., 2]
    translated_depth = translated_camera[..., 2]
    point_only_depth = point_only_camera[..., 2]
    camera_errors = error_stats(original_camera, translated_camera, valid)
    depth_errors = scalar_error_stats(original_depth, translated_depth, valid)
    point_only_errors = error_stats(original_camera, point_only_camera, valid)
    point_only_depth_errors = scalar_error_stats(original_depth, point_only_depth, valid)
    sidecar_camera_errors = error_stats(sample.camera, original_camera, valid)

    c2w = pose_to_c2w(sample.pose)
    c2w_prime = pose_to_c2w(translated_pose)
    gauge = torch.eye(4, dtype=torch.float32).repeat(c2w.shape[0], 1, 1)
    gauge[:, :3, 3] = delta
    pose_errors = (c2w_prime - torch.bmm(gauge, c2w)).abs()
    world_delta = translated_world[valid] - sample.world[valid]
    world_residual = world_delta - delta

    equivalent_pass = bool(
        torch.allclose(original_camera[valid], translated_camera[valid], atol=atol, rtol=rtol)
        and torch.allclose(original_depth[valid], translated_depth[valid], atol=atol, rtol=rtol)
        and float(pose_errors.max().item()) <= atol
    )
    # A nonzero camera-point change proves the negative control is detectable.  It
    # is intentionally not required that z changes: a translation parallel to a
    # camera's image plane leaves depth unchanged while still being physically wrong.
    point_only_detected = bool(point_only_errors["max_abs"] > atol)

    return {
        "sample_id": sample.identifier,
        "sidecar": sample.source,
        "num_frames": int(sample.world.shape[0]),
        "point_map_shape_fhw3": list(sample.world.shape),
        "camera_pose_shape_f7": list(sample.pose.shape),
        "coordinate_convention": "point_maps_ref=reference/anchor; point_maps_cam=per-frame camera",
        "pose_convention": "OpenCV camera-to-reference/world (T_cw), absT_quaR [tx,ty,tz,qw,qx,qy,qz]",
        "depth_convention": "camera-space z (point_maps_cam[...,2])",
        "translation": [float(x) for x in delta.tolist()],
        "valid_points": valid_count,
        "total_points": total_count,
        "valid_percent": 100.0 * valid_count / total_count,
        "explicit_depth_available": sample.explicit_depth is not None,
        "confidence_mask_applied": sample.confidence is not None,
        "original_recomputed_vs_sidecar_camera": sidecar_camera_errors,
        "correct_transform_camera_errors": camera_errors,
        "correct_transform_depth_errors": depth_errors,
        "point_only_camera_errors": point_only_errors,
        "point_only_depth_errors": point_only_depth_errors,
        "world_translation_max_abs_change": float(world_delta.abs().max().item()),
        "world_translation_residual_max_abs": float(world_residual.abs().max().item()),
        "pose_gauge_consistency_max_abs": float(pose_errors.max().item()),
        "correct_transform_pass": equivalent_pass,
        "point_only_control_detected": point_only_detected,
        "passed": bool(equivalent_pass and point_only_detected),
    }


def parse_translations(raw: Optional[str], count: int) -> List[Tuple[float, float, float]]:
    if raw is None:
        values = list(DEFAULT_TRANSLATIONS)
    else:
        values = []
        for item in raw.split(";"):
            components = [component.strip() for component in item.split(",") if component.strip()]
            if len(components) != 3:
                raise ValueError("--translations must be semicolon-separated x,y,z triples")
            values.append(tuple(float(component) for component in components))
    if len(values) < count:
        raise ValueError(f"need {count} translation vectors, received {len(values)}")
    selected = values[:count]
    if any(all(component == 0.0 for component in vector) for vector in selected):
        raise ValueError("all selected translations must be nonzero")
    return selected


def discover_sidecars(root: Path, pattern: str, seed: int) -> List[Path]:
    if not root.is_dir():
        raise ValueError(f"--sidecar-root is not a directory: {root}")
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"no sidecars matching {pattern!r} under {root}")
    random.Random(seed).shuffle(paths)
    return paths


def make_synthetic_sample() -> GeometrySample:
    """Known c2w pose and camera cloud for an independent matrix smoke test."""
    camera = torch.tensor(
        [[[[0.2, -0.1, 2.0], [0.5, 0.0, 3.0]], [[-0.4, 0.3, 4.0], [0.1, -0.2, 5.0]]]],
        dtype=torch.float32,
    )
    angle = math.pi / 2.0
    quaternion = torch.tensor([[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]])
    pose = torch.cat((torch.tensor([[2.0, -3.0, 1.5]]), quaternion), dim=-1)
    rotation = quaternion_to_matrix(pose[:, 3:7])
    world = torch.einsum("fij,fhwj->fhwi", rotation, camera) + pose[:, None, None, :3]
    return GeometrySample("synthetic_known_pose", world, camera, pose, None, None, "synthetic")


def aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def maximum(path: Sequence[str]) -> float:
        values = []
        for result in results:
            value: Any = result
            for key in path:
                value = value[key]
            values.append(float(value))
        return max(values)

    return {
        "num_samples": len(results),
        "all_passed": all(result["passed"] for result in results),
        "max_correct_camera_abs_error": maximum(("correct_transform_camera_errors", "max_abs")),
        "max_correct_depth_abs_error": maximum(("correct_transform_depth_errors", "max_abs")),
        "min_point_only_camera_abs_error": min(result["point_only_camera_errors"]["max_abs"] for result in results),
        "max_pose_gauge_consistency_error": maximum(("pose_gauge_consistency_max_abs",)),
    }


def write_report(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gauge_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = payload["aggregate"]
    lines = [
        "# CUT3R Global-Translation Gauge Diagnostic",
        "",
        "## Detected conventions",
        "",
        "- Point maps use row-vector storage `[F,H,W,3]`; calculations follow `X_ref = R_cw X_cam + t_cw`.",
        "- `camera_pose` is CUT3R `absT_quaR` `[F,7]`, an OpenCV camera-to-reference/world transform.",
        "- Depth is recomputed as positive camera-space z, matching `depth_supervision.py`.",
        "- CUT3R sidecars do not declare an independent physical-unit calibration; translations are in native CUT3R reference-coordinate units.",
        "",
        "## Aggregate",
        "",
        f"- Samples: {summary['num_samples']}",
        f"- Physical equivalence verified: {summary['all_passed']}",
        f"- Max correct-transform camera error: {summary['max_correct_camera_abs_error']:.8g}",
        f"- Max correct-transform depth error: {summary['max_correct_depth_abs_error']:.8g}",
        f"- Min point-only camera error: {summary['min_point_only_camera_abs_error']:.8g}",
        "",
        "## Per-sample results",
        "",
        "| Sample | Frames | Valid points | Correct camera max | Correct depth max | Point-only camera max | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["samples"]:
        lines.append(
            "| {sample_id} | {num_frames} | {valid_points} ({valid_percent:.2f}%) | "
            "{camera:.8g} | {depth:.8g} | {control:.8g} | {passed} |".format(
                sample_id=result["sample_id"],
                num_frames=result["num_frames"],
                valid_points=result["valid_points"],
                valid_percent=result["valid_percent"],
                camera=result["correct_transform_camera_errors"]["max_abs"],
                depth=result["correct_transform_depth_errors"]["max_abs"],
                control=result["point_only_camera_errors"]["max_abs"],
                passed=result["passed"],
            )
        )
    (output_dir / "gauge_report.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--sidecar-root", type=Path, help="Root containing CUT3R point-map .pt sidecars")
    source.add_argument("--sidecar-files", type=Path, nargs="+", help="Explicit CUT3R point-map sidecars")
    parser.add_argument("--sidecar-glob", default="*.pt", help="Recursive pattern used with --sidecar-root (default: %(default)s)")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of valid samples to evaluate (default: %(default)s)")
    parser.add_argument("--translations", help="Semicolon-separated x,y,z translation vectors; default has three fixed vectors")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic candidate-order seed (default: %(default)s)")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute comparison tolerance (default: %(default)s)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative comparison tolerance (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cut3r_global_translation_gauge"))
    parser.add_argument("--synthetic-smoke-test", action="store_true", help="Run the known-pose matrix smoke test instead of loading sidecars")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("--atol and --rtol must be nonnegative")
    translations = parse_translations(args.translations, args.num_samples)

    if args.synthetic_smoke_test:
        candidates: List[GeometrySample] = [make_synthetic_sample()]
    else:
        if args.sidecar_files:
            paths = list(args.sidecar_files)
        elif args.sidecar_root:
            paths = discover_sidecars(args.sidecar_root, args.sidecar_glob, args.seed)
        else:
            raise ValueError("provide --sidecar-root or --sidecar-files, or use --synthetic-smoke-test")
        candidates = []
        rejected: List[Dict[str, str]] = []
        for path in paths:
            try:
                candidates.append(load_sidecar(path, args.sidecar_root))
            except ValueError as exc:
                rejected.append({"sidecar": path.name, "reason": str(exc)})
        if rejected:
            print(f"[INFO] Rejected {len(rejected)} malformed/incompatible sidecars before evaluation.")

    results: List[Dict[str, Any]] = []
    rejections: List[Dict[str, str]] = []
    for candidate in candidates:
        if len(results) == args.num_samples:
            break
        try:
            result = evaluate_sample(candidate, translations[len(results)], atol=args.atol, rtol=args.rtol)
        except ValueError as exc:
            rejections.append({"sample": candidate.identifier, "reason": str(exc)})
            print(f"[WARN] Rejecting {candidate.identifier}: {exc}")
            continue
        results.append(result)
        print(
            "[RESULT] sample={sample_id} frames={num_frames} shapes={point_map_shape_fhw3}/{camera_pose_shape_f7} "
            "pose=T_cw correct_cam_max={camera:.8g} correct_depth_max={depth:.8g} "
            "point_only_cam_max={control:.8g} pass={passed}".format(
                **result,
                camera=result["correct_transform_camera_errors"]["max_abs"],
                depth=result["correct_transform_depth_errors"]["max_abs"],
                control=result["point_only_camera_errors"]["max_abs"],
            )
        )

    if len(results) != args.num_samples:
        raise RuntimeError(
            f"needed exactly {args.num_samples} valid samples but evaluated {len(results)}; "
            "provide a sidecar root with more compatible CUT3R point-map sidecars"
        )
    payload = {
        "schema": "cut3r_global_translation_gauge_v1",
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "synthetic_smoke_test": bool(args.synthetic_smoke_test),
        "samples": results,
        "rejections_during_evaluation": rejections,
        "aggregate": aggregate(results),
    }
    write_report(args.output_dir, payload)
    print(f"[SUMMARY] {json.dumps(payload['aggregate'], sort_keys=True)}")
    print(f"[OUTPUT] {args.output_dir / 'gauge_results.json'}")
    return 0 if payload["aggregate"]["all_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
