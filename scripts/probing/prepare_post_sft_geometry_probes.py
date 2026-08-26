#!/usr/bin/env python
"""Fail-closed preflight for the four post-SFT geometry depth probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import resolve_sidecar_path  # noqa: E402
from scripts.probing.local_depth_probe_cache import (  # noqa: E402
    compact_target_path,
    forward_cache_path,
)
from scripts.probing.post_sft_geometry_probe_specs import (  # noqa: E402
    BASE_MODEL,
    POST_SFT_DEPTH_FEATURE_LEVELS,
    POST_SFT_DEPTH_LAYERS,
    POST_SFT_PRE_LLM_FEATURES,
    MODEL_SPECS,
    SIGLIP_MODEL,
    SPLIT_SHA256,
    iter_specs,
    prepare_runtime_overlay,
    sha256_file,
)


DEFAULT_SPLIT = Path(
    "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
    "scannet_baseline_L6_depth_provenance/splits/"
    "semantic_probe_scannet_final_usable_sample_indices.json"
)
DEFAULT_FORWARD_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1")
DEFAULT_TARGET_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1")
DEFAULT_CUT3R_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features")
DEFAULT_GEOMETRY_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1")
DEFAULT_RUNTIME_ROOT = REPO_ROOT / ".offline_runtime/post_sft_geometry_probes"
DEFAULT_REPORT = REPO_ROOT / "logs/post_sft_geometry_probes/preflight.json"


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _check(name: str, ok: bool, *, detail: Any = None, blocker: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "blocker": bool(blocker and not ok), "detail": detail}


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _resolve_all(videos: list[dict[str, Any]], root: Path, subdir: str | None) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    missing: list[str] = []
    for video in videos:
        video_path = str(video["video_path"])
        sidecar = resolve_sidecar_path(video_path, root, subdir)
        if sidecar is None:
            missing.append(video_path)
        else:
            found.append(sidecar)
    return found, missing


def _resolve_local_cache(
    videos: list[dict[str, Any]], root: Path, resolver: Any
) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    missing: list[str] = []
    for video in videos:
        video_path = str(video["video_path"])
        path = resolver(root, video_path)
        if path.is_file():
            found.append(path)
        else:
            missing.append(video_path)
    return found, missing


def _inspect_forward_cache(paths: list[Path]) -> dict[str, Any]:
    inspected = []
    for path in (paths[:1] + paths[-1:] if paths else []):
        payload = _load_pt(path)
        if isinstance(payload, dict):
            candidates = {
                key: _tensor_shape(value)
                for key, value in payload.items()
                if isinstance(value, torch.Tensor)
            }
        else:
            candidates = {"payload": _tensor_shape(payload)}
        inspected.append({"path": str(path), "tensor_shapes": candidates})
    return {"inspected": inspected}


def _inspect_cut3r(paths: list[Path]) -> dict[str, Any]:
    inspected = []
    valid = True
    for path in (paths[:1] + paths[-1:] if paths else []):
        payload = _load_pt(path)
        keys = sorted(payload) if isinstance(payload, dict) else []
        camera_shape = _tensor_shape(payload.get("camera_tokens")) if isinstance(payload, dict) else None
        patch_shape = _tensor_shape(payload.get("patch_tokens")) if isinstance(payload, dict) else None
        sample_ok = (
            camera_shape is not None
            and patch_shape is not None
            and camera_shape[0] == 32
            and patch_shape[0] == 32
            and patch_shape[-2:] == [729, 768]
        )
        valid = valid and sample_ok
        inspected.append(
            {
                "path": str(path),
                "keys": keys,
                "camera_tokens_shape": camera_shape,
                "patch_tokens_shape": patch_shape,
                "valid": sample_ok,
            }
        )
    return {"valid": valid and bool(paths), "inspected": inspected}


def _inspect_geometry(paths: list[Path], key: str) -> dict[str, Any]:
    inspected = []
    valid = True
    compact_hits = 0
    for path in (paths[:2] + paths[-1:] if paths else []):
        payload = _load_pt(path)
        tensor = payload.get(key) if isinstance(payload, dict) else None
        shape = _tensor_shape(tensor)
        frames = int(shape[0]) if shape else None
        sample_ok = bool(
            shape
            and frames == 32
            and len(shape) == 4
            and (shape[-1] == 3 or shape[1] == 3)
        )
        compact_hits += int(frames == 2)
        valid = valid and sample_ok
        inspected.append(
            {
                "path": str(path),
                "keys": sorted(payload) if isinstance(payload, dict) else [],
                "selected_key": key,
                "selected_shape": shape,
                "valid": sample_ok,
            }
        )
    return {
        "valid": valid and bool(paths) and compact_hits == 0,
        "compact_two_frame_payloads_seen": compact_hits,
        "inspected": inspected,
    }


def _gpu_status() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"ok": False, "nvidia_smi": "not found", "torch_cuda_available": torch.cuda.is_available()}
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    device_nodes = sorted(str(path) for path in Path("/dev").glob("nvidia[0-9]*"))
    return {
        "ok": result.returncode == 0 and torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "nvidia_smi_returncode": result.returncode,
        "nvidia_smi_output": result.stdout.strip(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "device_nodes": device_nodes,
    }


def _eomt_status(args: argparse.Namespace, videos: list[dict[str, Any]]) -> dict[str, Any]:
    raw_assets = {
        "source_config": Path(args.eomt_config).is_file(),
        "checkpoint": Path(args.eomt_checkpoint).is_file(),
        "backbone_config": (Path(args.eomt_backbone) / "config.json").is_file(),
    }
    cache_paths, cache_missing = _resolve_all(videos, Path(args.eomt_cache_root), args.eomt_cache_subdir)
    cache_complete = len(cache_paths) == len(videos) and not cache_missing
    return {
        "ok": all(raw_assets.values()) or cache_complete,
        "raw_assets": raw_assets,
        "raw_asset_paths": {
            "source_config": args.eomt_config,
            "checkpoint": args.eomt_checkpoint,
            "backbone": args.eomt_backbone,
        },
        "cache": {
            "root": args.eomt_cache_root,
            "subdir": args.eomt_cache_subdir,
            "found": len(cache_paths),
            "missing": len(cache_missing),
            "first_missing": cache_missing[:5],
            "complete": cache_complete,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=sorted(MODEL_SPECS))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--forward-root", default=str(DEFAULT_FORWARD_ROOT))
    parser.add_argument("--forward-subdir", default="forward_frames_32")
    parser.add_argument("--targets-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--targets-subdir", default="spatial_features_points")
    parser.add_argument("--cut3r-root", default=str(DEFAULT_CUT3R_ROOT))
    parser.add_argument("--cut3r-subdir", default="spatial_features")
    parser.add_argument("--geometry-root", default=str(DEFAULT_GEOMETRY_ROOT))
    parser.add_argument("--geometry-subdir", default="spatial_features_points")
    parser.add_argument("--eomt-config", default="/mnt/DATA_SSD/shaoruei/probing_data/eomt_runtime/configs/eomt_large_640.yaml")
    parser.add_argument("--eomt-checkpoint", default="/mnt/DATA_SSD/shaoruei/probing_data/eomt_runtime/checkpoints/pytorch_model.bin")
    parser.add_argument("--eomt-backbone", default="/mnt/DATA_SSD/shaoruei/probing_data/eomt_runtime/backbone/timm_vit_large_patch14_reg4_dinov2_lvd142m")
    parser.add_argument("--eomt-cache-root", default="/mnt/DATA_SSD/shaoruei/probing_data/eomt_outputs_32_v1")
    parser.add_argument("--eomt-cache-subdir", default="eomt_outputs")
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--report-only", action="store_true", help="Write the report but return success when blockers exist.")
    args = parser.parse_args()

    split_path = Path(args.split).resolve()
    checks: list[dict[str, Any]] = []
    checks.append(_check("split_exists", split_path.is_file(), detail=str(split_path)))
    if not split_path.is_file():
        videos: list[dict[str, Any]] = []
        split = {}
        split_hash = None
    else:
        split = _json(split_path)
        videos = list(split.get("videos", []))
        split_hash = sha256_file(split_path)
    split_detail = {
        "sha256": split_hash,
        "expected_sha256": SPLIT_SHA256,
        "videos": len(videos),
        "train_videos": split.get("train_videos"),
        "val_videos": split.get("val_videos"),
        "frames": split.get("final_usable_frames"),
    }
    checks.append(
        _check(
            "authoritative_split_identity",
            split_hash == SPLIT_SHA256
            and len(videos) == 1199
            and split.get("train_videos") == 1006
            and split.get("val_videos") == 193
            and split.get("final_usable_frames") == 2398,
            detail=split_detail,
        )
    )

    for name, path in (("base_model", BASE_MODEL), ("siglip_model", SIGLIP_MODEL)):
        checks.append(_check(name, (path / "config.json").is_file(), detail=str(path)))

    overlays: dict[str, Any] = {}
    for spec in iter_specs(args.models):
        try:
            runtime = prepare_runtime_overlay(spec, Path(args.runtime_root).resolve())
            runtime_config = _json(runtime / "config.json")
            decoder_depth = int(runtime_config.get("num_hidden_layers", 0))
            fusion_output_source = (
                "geometry_aware_projection_output_before_mm_projector"
                if spec.key == "visual_3d_rope"
                else "explicit_fusion_block_output_before_mm_projector"
            )
            overlays[spec.key] = {
                "ok": True,
                "runtime": str(runtime),
                "effective_config_sha256": sha256_file(runtime / "config.json"),
                "architecture": spec.architecture,
                "token_layout": spec.token_layout,
                "point_map_key": spec.point_map_key,
                "probe_capabilities": {
                    "decoder_depth": decoder_depth,
                    "requested_llm_layers": list(POST_SFT_DEPTH_LAYERS),
                    "all_requested_llm_layers_available": (
                        bool(POST_SFT_DEPTH_LAYERS)
                        and max(POST_SFT_DEPTH_LAYERS) < decoder_depth
                    ),
                    "requested_pre_llm_features": list(POST_SFT_PRE_LLM_FEATURES),
                    "fusion_output_source": fusion_output_source,
                    "projected_features_source": "mm_projector_output",
                    "ordinary_visual_tokens_only": True,
                    "runtime_capture_smoke_required_after_contract_expansion": True,
                    "runtime_capture_verified_by_this_preflight": False,
                },
            }
            checks.append(_check(f"checkpoint_{spec.key}", True, detail=overlays[spec.key]))
        except Exception as exc:
            overlays[spec.key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            checks.append(_check(f"checkpoint_{spec.key}", False, detail=overlays[spec.key]))

    forward_paths, forward_missing = _resolve_local_cache(
        videos, Path(args.forward_root), forward_cache_path
    )
    forward_detail = {
        "root": args.forward_root,
        "subdir": args.forward_subdir,
        "found": len(forward_paths),
        "missing": len(forward_missing),
        "first_missing": forward_missing[:5],
        **_inspect_forward_cache(forward_paths),
    }
    checks.append(_check("forward_frames_32", len(forward_paths) == len(videos) == 1199, detail=forward_detail))

    target_paths, target_missing = _resolve_local_cache(
        videos, Path(args.targets_root), compact_target_path
    )
    target_inspection = _inspect_geometry(target_paths, "point_maps_cam")
    targets_ok = len(target_paths) == len(videos) == 1199 and target_inspection["compact_two_frame_payloads_seen"] > 0
    target_detail = {
        "root": args.targets_root,
        "subdir": args.targets_subdir,
        "found": len(target_paths),
        "missing": len(target_missing),
        "first_missing": target_missing[:5],
        "semantics": "compact two-selected-frame point_maps_cam depth targets only",
        **target_inspection,
    }
    checks.append(_check("compact_probe_targets_2f", targets_ok, detail=target_detail))

    cut3r_paths, cut3r_missing = _resolve_all(videos, Path(args.cut3r_root), args.cut3r_subdir)
    cut3r_inspection = _inspect_cut3r(cut3r_paths)
    cut3r_detail = {
        "root": args.cut3r_root,
        "subdir": args.cut3r_subdir,
        "found": len(cut3r_paths),
        "missing": len(cut3r_missing),
        "first_missing": cut3r_missing[:5],
        **cut3r_inspection,
    }
    needs_cut3r = any(MODEL_SPECS[key].requires_cut3r_tokens for key in args.models)
    checks.append(
        _check(
            "cut3r_token_sidecars_32",
            (not needs_cut3r) or (len(cut3r_paths) == len(videos) == 1199 and cut3r_inspection["valid"]),
            detail=cut3r_detail,
        )
    )

    geometry_paths, geometry_missing = _resolve_all(videos, Path(args.geometry_root), args.geometry_subdir)
    geometry_inspection = _inspect_geometry(geometry_paths, "point_maps_ref")
    geometry_detail = {
        "root": args.geometry_root,
        "subdir": args.geometry_subdir,
        "found": len(geometry_paths),
        "missing": len(geometry_missing),
        "first_missing": geometry_missing[:5],
        "required_shape": "32 x H x W x 3 (or 32 x 3 x H x W)",
        "required_key": "point_maps_ref",
        "compact_probe_targets_are_forbidden": True,
        **geometry_inspection,
    }
    needs_geometry = any(MODEL_SPECS[key].requires_full_point_maps for key in args.models)
    checks.append(
        _check(
            "full_point_map_sidecars_32",
            (not needs_geometry)
            or (len(geometry_paths) == len(videos) == 1199 and geometry_inspection["valid"]),
            detail=geometry_detail,
        )
    )

    eomt = _eomt_status(args, videos)
    needs_eomt = any(MODEL_SPECS[key].requires_eomt for key in args.models)
    checks.append(_check("eomt_inference_inputs", (not needs_eomt) or eomt["ok"], detail=eomt))

    gpu = _gpu_status()
    checks.append(_check("two_titan_v_gpu_readiness", gpu["ok"], detail=gpu))

    disk = shutil.disk_usage("/home/shaoruei")
    disk_detail = {
        "free_bytes": disk.free,
        "free_gib": round(disk.free / (1024**3), 3),
        "minimum_free_gib_before_generation": 300,
        "rolling_cache_policy": True,
        "probe_layer_count": len(POST_SFT_DEPTH_LAYERS),
        "estimated_one_model_complete_layers_gib": round(28.24 * len(POST_SFT_DEPTH_LAYERS) / 9, 2),
    }
    checks.append(_check("nvme_headroom", disk.free >= 300 * 1024**3, detail=disk_detail))

    blockers = [item for item in checks if item["blocker"]]
    report = {
        "schema_version": "post_sft_geometry_probe_preflight_v1",
        "ready": not blockers,
        "blocker_count": len(blockers),
        "blockers": [item["name"] for item in blockers],
        "post_sft_depth_layers": list(POST_SFT_DEPTH_LAYERS),
        "post_sft_depth_feature_levels": list(POST_SFT_DEPTH_FEATURE_LEVELS),
        "primary_probe_tokens": "ordinary frame-aligned visual tokens only",
        "split": split_detail,
        "runtime_overlays": overlays,
        "checks": checks,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[REPORT] {report_path}")
    if blockers and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
