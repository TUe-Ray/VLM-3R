#!/usr/bin/env python
"""CPU-only provenance, input, and future-parity validation for ScanNet depth probes.

This command deliberately does not load a VLM or invoke CUDA.  It validates
the local migrated inputs against the authoritative transferred ScanNet L6
bundle, then can compare a future L6 metrics.json without modifying any
experiment configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from depth_probe_common import depth_from_point_maps, read_json, validate_llm_layers, write_json
from local_depth_probe_cache import compact_target_path, forward_cache_path, load_forward_frames, load_selected_camera_depths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVENANCE_ROOT = (
    Path("/home/shaoruei/probe_provenance/scannet_baseline_L6") / "scannet_baseline_L6_depth_provenance"
)
DEFAULT_FORWARD_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1")
DEFAULT_TARGET_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1")
DEFAULT_SIDECAR_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features")
DEFAULT_CHECKPOINT = Path("/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2")
DEFAULT_OUTPUT_ROOT = Path("/home/shaoruei/probe_outputs/scannet_depth_layers_v1")
EXPECTED_SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
EXPECTED_METRICS = {
    "mae": 0.2548516317768708,
    "absrel": 0.14451827108860016,
    "delta125": 0.8027783632278442,
    "num_tokens": 75656,
    "best_epoch": 31,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def add_check(report: dict[str, Any], name: str, passed: bool | None, detail: Any, *, required: bool = True) -> None:
    status = "PASS" if passed is True else "FAIL" if passed is False else "UNVERIFIED"
    report["checks"].append({"name": name, "status": status, "required": required, "detail": detail})


def historical_reference(provenance_root: Path) -> dict[str, Any]:
    split_path = provenance_root / "splits" / "semantic_probe_scannet_final_usable_sample_indices.json"
    metrics_path = provenance_root / "baseline_L6" / "metrics.json"
    history_path = provenance_root / "baseline_L6" / "history.json"
    task_path = provenance_root / "configs" / "scannet_baseline_probe_tasks.tsv"
    config_paths = [
        provenance_root / "configs" / "depth_probe_common.py",
        provenance_root / "configs" / "train_depth_probes.py",
        provenance_root / "configs" / "README_scannet_probe_pipeline.md",
    ]
    missing = [str(path) for path in [split_path, metrics_path, history_path, task_path, *config_paths] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Historical ScanNet provenance is incomplete: {missing}")
    metrics = read_json(metrics_path)
    return {
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "task_path": str(task_path),
        "config_sources": [{"path": str(path), "sha256": sha256_file(path)} for path in config_paths],
        "metrics": metrics,
    }


def checkpoint_identity(path: Path) -> dict[str, Any]:
    required = ["adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json", "config.json", "generation_config.json"]
    files: dict[str, Any] = {}
    for name in required:
        candidate = path / name
        files[name] = {"path": str(candidate), "exists": candidate.is_file()}
        if candidate.is_file():
            files[name]["sha256"] = sha256_file(candidate)
    config = read_json(path / "config.json") if (path / "config.json").is_file() else {}
    return {
        "path": str(path),
        "all_required_files_present": all(entry["exists"] for entry in files.values()),
        "files": files,
        "config": {
            "fusion_block": config.get("fusion_block"),
            "spatial_tower": config.get("spatial_tower", config.get("mm_spatial_tower")),
            "num_hidden_layers": config.get("num_hidden_layers"),
        },
    }


def split_video_map(split_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    videos = split_payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Split has no videos list")
    result: dict[str, dict[str, Any]] = {}
    for video in videos:
        path = str(video.get("video_path", ""))
        if not path or path in result:
            raise ValueError(f"Split has missing or duplicate video_path {path!r}")
        result[path] = video
    return result


def validate_manifest_mapping(
    split_videos: dict[str, dict[str, Any]], forward_root: Path, target_root: Path, sidecar_root: Path
) -> tuple[list[str], dict[str, Any]]:
    frame_manifest = read_json(forward_root / "manifests" / "frame_cache_manifest.json")
    target_manifest = read_json(target_root / "manifests" / "migration_manifest.json")
    frame_records = {
        str(record.get("source_video_relative_path")): record
        for record in frame_manifest.get("records", [])
        if record.get("dataset") == "scannet"
    }
    target_records = {
        str(record.get("source_video_relative_path")): record
        for record in target_manifest.get("records", [])
        if record.get("dataset") == "scannet"
    }
    errors: list[str] = []
    for video_path, video in split_videos.items():
        selected = [int(frame["frame_index"]) for frame in video.get("frames", [])]
        expected_ids = [str(frame["frame_sample_id"]) for frame in video.get("frames", [])]
        frame_record = frame_records.get(video_path)
        target_record = target_records.get(video_path)
        if frame_record is None:
            errors.append(f"forward manifest missing {video_path}")
            continue
        if target_record is None:
            errors.append(f"target manifest missing {video_path}")
            continue
        observed_positions = [int(item["probe_position"]) for item in frame_record.get("probe_targets", [])]
        observed_ids = [str(item["frame_sample_id"]) for item in frame_record.get("probe_targets", [])]
        if observed_positions != selected or observed_ids != expected_ids:
            errors.append(f"forward selected-frame mapping mismatch for {video_path}")
        if [int(value) for value in target_record.get("selected_indices", [])] != selected:
            errors.append(f"target selected-frame mapping mismatch for {video_path}")
        cache_path = forward_root / str(frame_record.get("cache_relative_path", ""))
        target_path = target_root / str(target_record.get("output_relative_path", ""))
        sidecar_path = sidecar_root / "scannet" / "spatial_features" / f"{video['scene_id']}.pt"
        if not cache_path.is_file():
            errors.append(f"missing forward cache {cache_path}")
        if not target_path.is_file():
            errors.append(f"missing compact target {target_path}")
        if not sidecar_path.is_file():
            errors.append(f"missing CUT3R sidecar {sidecar_path}")
    detail = {
        "frame_cache_schema": frame_manifest.get("schema_version"),
        "target_bundle_version": target_manifest.get("bundle_version"),
        "scannet_forward_records": len(frame_records),
        "scannet_target_records": len(target_records),
        "checked_videos": len(split_videos),
    }
    return errors, detail


def validate_payload_samples(
    split_videos: dict[str, dict[str, Any]], forward_root: Path, target_root: Path, sidecar_root: Path, count: int
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    depths: list[torch.Tensor] = []
    sampled = list(sorted(split_videos))[:count]
    for video_path in sampled:
        video = split_videos[video_path]
        selected = [int(frame["frame_index"]) for frame in video["frames"]]
        try:
            forward = load_forward_frames(forward_root, video_path)
            payload_depths, _target_path, _payload = load_selected_camera_depths(target_root, video, selected)
            if list(payload_depths) != selected:
                errors.append(f"target depth order mismatch for {video_path}")
            for depth in payload_depths.values():
                valid = depth[torch.isfinite(depth) & (depth > 0)]
                if valid.numel():
                    depths.append(valid.reshape(-1))
            sidecar_path = sidecar_root / "scannet" / "spatial_features" / f"{video['scene_id']}.pt"
            sidecar = torch_load(sidecar_path)
            camera, patch = sidecar.get("camera_tokens"), sidecar.get("patch_tokens")
            if not isinstance(camera, torch.Tensor) or not isinstance(patch, torch.Tensor):
                errors.append(f"CUT3R sidecar missing camera_tokens/patch_tokens for {video_path}")
            elif camera.ndim != 3 or patch.ndim != 3 or camera.shape[0] != 32 or patch.shape[0] != 32:
                errors.append(f"CUT3R sidecar has unexpected tensor shapes for {video_path}")
            if forward["frames_rgb_uint8"].shape[0] != 32:
                errors.append(f"forward payload has wrong frame count for {video_path}")
        except Exception as exc:
            errors.append(f"payload validation failed for {video_path}: {exc}")
    stats: dict[str, Any] = {"payload_videos_checked": len(sampled)}
    if depths:
        all_depths = torch.cat(depths)
        stats["camera_depth_sample_stats"] = {
            "valid_values": int(all_depths.numel()),
            "mean": float(all_depths.float().mean().item()),
            "min": float(all_depths.float().min().item()),
            "max": float(all_depths.float().max().item()),
        }
    return errors, stats


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    historical = historical_reference(args.provenance_root)
    split_path = Path(args.sample_indices) if args.sample_indices else Path(historical["split_path"])
    split_payload = read_json(split_path)
    split_videos = split_video_map(split_payload)
    report: dict[str, Any] = {
        "mode": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_model_forward": "NOT RUN",
        "requested_layers": validate_llm_layers(args.layers),
        "historical_reference": historical,
        "checks": [],
    }
    local_sha = sha256_file(split_path)
    add_check(report, "split_sha256", local_sha == historical["split_sha256"] == EXPECTED_SPLIT_SHA256, {"local": local_sha, "historical": historical["split_sha256"]})
    videos = list(split_videos.values())
    train = [video for video in videos if video.get("split") == "train"]
    val = [video for video in videos if video.get("split") == "val"]
    add_check(report, "scannet_video_count", len(videos) == 1199, {"actual": len(videos), "expected": 1199})
    add_check(report, "split_video_counts", len(train) == 1006 and len(val) == 193, {"train": len(train), "val": len(val)})
    add_check(report, "selected_frame_counts", all(len(video.get("frames", [])) == 2 for video in videos), {"frames_per_video": sorted({len(video.get("frames", [])) for video in videos})})
    add_check(report, "selected_frame_indices", all(all(0 <= int(frame["frame_index"]) < 32 for frame in video["frames"]) for video in videos), "all selected indices in [0, 31]")
    add_check(report, "camera_depth_target_semantics", split_payload.get("point_maps_subdir") == "spatial_features_points" and bool(split_payload.get("require_depth_gt")), {"point_maps_subdir": split_payload.get("point_maps_subdir"), "require_depth_gt": split_payload.get("require_depth_gt"), "depth_mode": "camera_z via point_maps_cam"})
    add_check(report, "layer_request", report["requested_layers"] == [6] if args.require_l6 else True, {"requested": report["requested_layers"], "require_l6": args.require_l6})
    identity = checkpoint_identity(args.checkpoint)
    report["checkpoint_identity"] = identity
    add_check(report, "checkpoint_files", identity["all_required_files_present"], identity["files"])
    config = identity["config"]
    sidecar_contract_ok = str(config.get("fusion_block", "")).lower() == "cross_attention" and "cut3r" in str(config.get("spatial_tower", "")).lower()
    add_check(report, "baseline_cut3r_sidecar_contract", sidecar_contract_ok, config)
    # The transferred bundle names the model label but does not provide hashes
    # for the original checkpoint.  Surface this fact instead of inventing a
    # historical fingerprint.
    add_check(report, "historical_checkpoint_hash", None, "No checkpoint hash is present in the authoritative provenance bundle.", required=False)
    manifest_errors, manifest_detail = validate_manifest_mapping(split_videos, args.forward_root, args.target_root, args.sidecar_root)
    add_check(report, "local_manifest_and_file_mapping", not manifest_errors, {**manifest_detail, "errors": manifest_errors[:20], "error_count": len(manifest_errors)})
    payload_count = len(videos) if args.verify_payloads else min(args.payload_samples, len(videos))
    payload_errors, payload_stats = validate_payload_samples(split_videos, args.forward_root, args.target_root, args.sidecar_root, payload_count)
    add_check(report, "payload_schema_and_target_statistics", not payload_errors, {**payload_stats, "errors": payload_errors[:20], "error_count": len(payload_errors), "scope": "all" if args.verify_payloads else "sample"})
    output_paths = [
        args.cache_root / "features" / args.model_label / f"layer_{layer}" / "frame_<frame_sample_id>.pt"
        for layer in report["requested_layers"]
    ]
    add_check(report, "output_path_collision", len({str(path) for path in output_paths}) == len(output_paths), [str(path) for path in output_paths])
    report["assessment"] = "PASS" if all(check["status"] == "PASS" for check in report["checks"] if check["required"]) else "FAIL"
    return report


def postflight(args: argparse.Namespace) -> dict[str, Any]:
    historical = historical_reference(args.provenance_root)
    new_metrics = read_json(args.new_metrics)
    report: dict[str, Any] = {
        "mode": "postflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "historical_reference": historical,
        "new_metrics_path": str(args.new_metrics),
        "checks": [],
    }
    identity = args.preflight_report and read_json(args.preflight_report)
    if identity:
        add_check(report, "preflight_identity", identity.get("assessment") == "PASS", {"preflight_report": str(args.preflight_report), "assessment": identity.get("assessment")})
    else:
        add_check(report, "preflight_identity", None, "No preflight report supplied; identity cannot be re-established.", required=False)
    expected = historical["metrics"]
    add_check(report, "model_label", new_metrics.get("model_label") == expected.get("model_label"), {"historical": expected.get("model_label"), "new": new_metrics.get("model_label")})
    add_check(report, "feature_level", new_metrics.get("feature_level") == "layer_6", {"historical": expected.get("feature_level"), "new": new_metrics.get("feature_level")})
    add_check(report, "validation_tokens", int(new_metrics.get("num_tokens", -1)) == int(EXPECTED_METRICS["num_tokens"]), {"historical": EXPECTED_METRICS["num_tokens"], "new": new_metrics.get("num_tokens")})
    differences: dict[str, Any] = {}
    for key in ("mae", "absrel", "delta125"):
        historical_value = float(expected[key])
        new_value = float(new_metrics[key])
        absolute = new_value - historical_value
        relative = absolute / historical_value if historical_value else float("nan")
        differences[key] = {"historical": historical_value, "new": new_value, "absolute_difference": absolute, "relative_difference": relative}
    report["metric_differences"] = differences
    add_check(report, "mae_within_5_percent", abs(differences["mae"]["relative_difference"]) <= 0.05, differences["mae"])
    add_check(report, "absrel_comparable", abs(differences["absrel"]["relative_difference"]) <= 0.05, differences["absrel"], required=False)
    add_check(report, "delta125_comparable", abs(differences["delta125"]["relative_difference"]) <= 0.05, differences["delta125"], required=False)
    report["assessment"] = "PASS" if all(check["status"] == "PASS" for check in report["checks"] if check["required"]) else "FAIL"
    report["diagnostic_categories_if_failed"] = ["wrong checkpoint", "wrong ScanNet split/sample indices", "32-frame cache mismatch", "selected-frame mismatch", "hidden-state indexing", "dtype/preprocessing", "probe configuration", "stale/mixed cache", "camera-space target mismatch"]
    return report


def write_report(report: dict[str, Any], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{report['mode']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    json_path = output_root / f"{stem}.json"
    markdown_path = output_root / f"{stem}.md"
    write_json(json_path, report)
    lines = [f"# ScanNet depth-probe {report['mode']}", "", f"Assessment: **{report['assessment']}**", "", "| Check | Status |", "| --- | --- |"]
    lines.extend(f"| {check['name']} | {check['status']} |" for check in report["checks"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validate local inputs and exit without loading a model.")
    parser.add_argument("--postflight", action="store_true", help="Compare a future L6 metrics artifact against provenance.")
    parser.add_argument("--layers", nargs="+", type=int, default=[6])
    parser.add_argument("--require-l6", action="store_true", default=False, help="Require exactly L6 (for parity preflight).")
    parser.add_argument("--model-label", default="vlm3r_baseline")
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--provenance-root", type=Path, default=DEFAULT_PROVENANCE_ROOT)
    parser.add_argument("--forward-root", type=Path, default=DEFAULT_FORWARD_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_SIDECAR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, default=Path("/home/shaoruei/probe_cache/scannet_depth_layers_v1"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-path", type=Path, default=REPO_ROOT / "logs" / "scannet_depth_layers_v1" / "validate_scannet_depth_probe.log")
    parser.add_argument("--payload-samples", type=int, default=3)
    parser.add_argument("--verify-payloads", action="store_true", help="Load every cache/target/sidecar payload; CPU I/O intensive.")
    parser.add_argument("--new-metrics", type=Path, default=None)
    parser.add_argument("--preflight-report", type=Path, default=None)
    args = parser.parse_args()
    if args.preflight == args.postflight:
        parser.error("Choose exactly one of --preflight or --postflight")
    if args.preflight:
        args.require_l6 = bool(args.require_l6)
        report = preflight(args)
    else:
        if args.new_metrics is None:
            parser.error("--postflight requires --new-metrics")
        report = postflight(args)
    report_path = write_report(report, args.output_root)
    lines = [f"ScanNet depth probe {report['mode']}: {report['assessment']}"]
    if report["mode"] == "preflight":
        lines.append("layers=" + ",".join(str(layer) for layer in report["requested_layers"]))
        lines.append("CUDA/model forward  NOT RUN")
    for check in report["checks"]:
        lines.append(f"{check['name']:<32} {check['status']}")
    lines.append(f"report={report_path}")
    console = "\n".join(lines)
    print(console)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8") as f:
        f.write(console + "\n")
    if report["assessment"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
