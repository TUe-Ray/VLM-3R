#!/usr/bin/env python
"""Prepare and validate a shared 32-frame EoMT raw-output cache.

This is a migration utility, not a probing/training entry point.  It consumes
the already validated lossless RGB frame cache, runs the historical EoMT
wrapper, and writes raw float32 mask/class outputs plus explicit provenance.
The script is deliberately fail-closed: it will not generate a full cache
unless the one-video smoke report is PASS, and it never overwrites an
existing output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow direct execution from any working directory (including ``conda run``)
# to import the repository's lightweight EoMT wrapper.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
EXPECTED_EOMT_REVISION = "eb13d17ea9ea698baa14fc1632939544e03bceef"
EXPECTED_VIDEOS = 1199
EXPECTED_NUM_Q = 200
EXPECTED_NUM_CLASSES = 133


def nontrivial_variance(value: Any, *, name: str) -> float:
    """Return a finite nonzero population variance or reject a degenerate cache."""

    import torch

    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    variance = float(value.detach().float().var(unbiased=False).item())
    if not torch.isfinite(torch.tensor(variance)) or variance <= 0.0:
        raise RuntimeError(f"{name} has no nontrivial output variance: {variance}")
    return variance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scene_id_from_video(video: dict[str, Any]) -> str:
    if isinstance(video.get("scene_id"), str):
        return str(video["scene_id"])
    raw = str(video.get("video_path", ""))
    return Path(raw).stem


def validate_input_manifests(
    authoritative_manifest_path: Path,
    forward_manifest_path: Path,
    expected_forward_manifest_sha256: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    authoritative_sha256 = sha256_file(authoritative_manifest_path)
    if authoritative_sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"authoritative manifest SHA-256 mismatch: {authoritative_sha256} != {EXPECTED_SPLIT_SHA256}"
        )
    authoritative = load_json(authoritative_manifest_path)
    videos = authoritative.get("videos")
    if not isinstance(videos, list) or len(videos) != EXPECTED_VIDEOS:
        raise RuntimeError(f"Expected {EXPECTED_VIDEOS} authoritative videos, got {len(videos or [])}")
    authoritative_scenes = {scene_id_from_video(video): video for video in videos}
    if len(authoritative_scenes) != EXPECTED_VIDEOS:
        raise RuntimeError("Authoritative manifest contains duplicate scene IDs")

    forward_sha256 = sha256_file(forward_manifest_path)
    if expected_forward_manifest_sha256 and forward_sha256 != expected_forward_manifest_sha256:
        raise RuntimeError(
            f"forward cache manifest SHA-256 mismatch: {forward_sha256} != {expected_forward_manifest_sha256}"
        )
    forward = load_json(forward_manifest_path)
    records = [record for record in forward.get("records", []) if record.get("dataset") == "scannet"]
    if len(records) != EXPECTED_VIDEOS:
        raise RuntimeError(f"Expected {EXPECTED_VIDEOS} ScanNet forward records, got {len(records)}")
    by_scene = {str(record["scene_id"]): record for record in records}
    if len(by_scene) != EXPECTED_VIDEOS:
        raise RuntimeError("Forward cache manifest contains duplicate ScanNet scene IDs")
    missing = sorted(set(authoritative_scenes) - set(by_scene))
    extra = sorted(set(by_scene) - set(authoritative_scenes))
    if missing or extra:
        raise RuntimeError(f"Forward/authoritative scene set mismatch: missing={missing[:5]} extra={extra[:5]}")
    for scene, record in by_scene.items():
        indices = record.get("source_frame_indices")
        if not isinstance(indices, list) or len(indices) != 32:
            raise RuntimeError(f"{scene}: forward manifest does not contain exactly 32 source indices")
        if not isinstance(record.get("cache_relative_path"), str):
            raise RuntimeError(f"{scene}: forward manifest has no cache_relative_path")

    provenance = {
        "authoritative_manifest": str(authoritative_manifest_path),
        "authoritative_manifest_sha256": authoritative_sha256,
        "authoritative_video_count": len(authoritative_scenes),
        "forward_cache_manifest": str(forward_manifest_path),
        "forward_cache_manifest_sha256": forward_sha256,
        "forward_cache_manifest_validation_status": forward.get("aggregate", {}).get("validation_status"),
        "forward_frames_per_video": forward.get("aggregate", {}).get("forward_positions_per_video"),
        "scene_set_exact": True,
    }
    return by_scene, provenance


def load_frame_payload(path: Path, scene: str) -> tuple[Any, list[int], dict[str, Any]]:
    import torch

    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{scene}: frame cache is not a dict: {path}")
    frames = payload.get("frames_rgb_uint8")
    source_indices = payload.get("source_frame_indices")
    if not isinstance(frames, torch.Tensor):
        raise RuntimeError(f"{scene}: missing frames_rgb_uint8")
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[0] != 32 or frames.shape[-1] != 3:
        raise RuntimeError(f"{scene}: invalid frame tensor {tuple(frames.shape)} {frames.dtype}")
    if not isinstance(source_indices, torch.Tensor) or tuple(source_indices.shape) != (32,):
        raise RuntimeError(f"{scene}: invalid source_frame_indices")
    source_indices_list = [int(value) for value in source_indices.tolist()]
    metadata = {
        "schema_version": payload.get("schema_version"),
        "dataset": payload.get("dataset"),
        "scene_id": payload.get("scene_id"),
        "source_video_relative_path": payload.get("source_video_relative_path"),
        "source_video_path": payload.get("source_video_path"),
        "source_frame_count": int(payload.get("source_frame_count", -1)),
        "source_fps": float(payload.get("source_fps", 0.0)),
        "video_time_seconds": float(payload.get("video_time_seconds", 0.0)),
        "frame_time_seconds": [float(value) for value in payload.get("frame_time_seconds", [])],
        "frame_time_string": str(payload.get("frame_time_string", "")),
    }
    if metadata["dataset"] != "scannet" or metadata["scene_id"] != scene:
        raise RuntimeError(f"{scene}: frame cache metadata mismatch: {metadata}")
    return frames, source_indices_list, metadata


def checkpoint_file_provenance(path: Path, filenames: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"directory": str(path), "files": {}}
    for filename in filenames:
        file_path = path / filename
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        result["files"][filename] = {
            "path": str(file_path),
            "bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        }
    return result


def build_provenance(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.eomt_config).resolve()
    eomt_checkpoint = Path(args.eomt_checkpoint).resolve()
    backbone = Path(args.eomt_backbone).resolve()
    if not config_path.is_file() or not eomt_checkpoint.is_file() or not (backbone / "config.json").is_file():
        raise FileNotFoundError("EoMT config/checkpoint/backbone is incomplete")
    backbone_weights = backbone / "pytorch_model.bin"
    if not backbone_weights.is_file():
        backbone_weights = backbone / "model.safetensors"
    if not backbone_weights.is_file():
        raise FileNotFoundError(f"No backbone weights in {backbone}")
    config = load_yaml(config_path)
    stuff_classes = list(config.get("data", {}).get("init_args", {}).get("stuff_classes", []))
    return {
        "eomt_source_revision": args.eomt_source_revision,
        "eomt_runtime": {
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size, "sha256": sha256_file(config_path)},
            "checkpoint": {"path": str(eomt_checkpoint), "bytes": eomt_checkpoint.stat().st_size, "sha256": sha256_file(eomt_checkpoint)},
            "backbone": {
                "directory": str(backbone),
                "config": {"path": str(backbone / "config.json"), "bytes": (backbone / "config.json").stat().st_size, "sha256": sha256_file(backbone / "config.json")},
                "weights": {"path": str(backbone_weights), "bytes": backbone_weights.stat().st_size, "sha256": sha256_file(backbone_weights)},
            },
            "loader": "VLM-3R .codex-work/feat-eomt-runtime llava.model.multimodal_eomt.EoMTExtractor",
        },
        "post_sft_checkpoints": [
            checkpoint_file_provenance(Path(path).resolve(), ("adapter_model.bin", "non_lora_trainables.bin"))
            for path in args.post_sft_checkpoint
        ],
        "taxonomy": {
            "dataset": "COCO panoptic",
            "thing_class_count": 80,
            "stuff_class_ids": [int(value) for value in stuff_classes],
            "semantic_class_count": EXPECTED_NUM_CLASSES,
            "no_object_class_index": EXPECTED_NUM_CLASSES,
            "output_class_count_including_no_object": EXPECTED_NUM_CLASSES + 1,
            "class_logits_semantics": "COCO panoptic semantic class logits plus final no-object class",
        },
        "query_count": EXPECTED_NUM_Q,
        "output_precision": "float32 CPU tensors; raw mask logits and raw class logits",
    }


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def load_model(args: argparse.Namespace) -> Any:
    import torch

    os.environ["EOMT_LOCAL_BACKBONE_PATH"] = str(Path(args.eomt_backbone).resolve())
    os.environ["EOMT_REPO_ROOT"] = str(Path(args.eomt_repo).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.eomt_repo not in sys.path:
        sys.path.insert(0, args.eomt_repo)
    from llava.model.multimodal_eomt import EoMTExtractor

    model = EoMTExtractor(
        {
            "config_path": str(Path(args.eomt_config).resolve()),
            "ckpt_path": str(Path(args.eomt_checkpoint).resolve()),
            "repo_root": str(Path(args.eomt_repo).resolve()),
            "local_backbone_path": str(Path(args.eomt_backbone).resolve()),
            "device": args.device,
            "dtype": torch.float16,
            "img_size": (640, 640),
            "num_classes": EXPECTED_NUM_CLASSES,
        }
    )
    if not model.is_available:
        raise RuntimeError("EoMTExtractor is unavailable; refusing no-op/fallback output")
    if int(model.num_q) != EXPECTED_NUM_Q or int(model.num_classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(f"Unexpected EoMT dimensions: num_q={model.num_q}, num_classes={model.num_classes}")
    model = model.to(args.device)
    model.eval()
    return model


def verify_loaded_checkpoint(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    state = torch_load(Path(args.eomt_checkpoint))
    if not isinstance(state, dict):
        raise RuntimeError("EoMT checkpoint is not a state dict")
    cleaned = {str(key)[len("network."):] if str(key).startswith("network.") else str(key): value for key, value in state.items()}
    # This is a Lightning training-wrapper buffer rather than an EoMT network
    # parameter.  All actual inference tensors must match exactly.
    cleaned.pop("criterion.empty_weight", None)
    current = model.network.state_dict()
    matched = sorted(set(cleaned).intersection(current))
    critical = ["q.weight", "class_head.weight", "mask_head.0.weight", "encoder.backbone"]
    missing_critical = [prefix for prefix in critical if not any(key.startswith(prefix) for key in matched)]
    if missing_critical:
        raise RuntimeError(f"EoMT checkpoint did not load critical tensors: {missing_critical}")
    missing = sorted(set(current) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(current))
    if missing or unexpected or len(matched) != len(current):
        raise RuntimeError(
            "EoMT checkpoint/model topology mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} "
            f"matched={len(matched)} model={len(current)}"
        )
    return {
        "checkpoint_state_tensor_count": len(cleaned),
        "model_state_tensor_count": len(current),
        "matching_tensor_count": len(matched),
        "matching_fraction_of_checkpoint": len(matched) / max(1, len(cleaned)),
        "critical_prefixes_present": [prefix for prefix in critical if prefix not in missing_critical],
        "strict_network_tensor_match": True,
    }


def direct_decode_matches_cache(args: argparse.Namespace, record: dict[str, Any], frames: Any, indices: list[int]) -> dict[str, Any]:
    import numpy as np
    from decord import VideoReader, cpu

    source = Path(args.video_root) / str(record["source_video_relative_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    reader = VideoReader(str(source), ctx=cpu(0), num_threads=1)
    expected_indices = np.linspace(0, len(reader) - 1, 32, dtype=int).tolist()
    if indices != expected_indices:
        raise RuntimeError(f"{record['scene_id']}: cached indices do not equal historical np.linspace indices")
    direct = reader.get_batch(expected_indices).asnumpy()
    cached = frames.numpy()
    if not np.array_equal(direct, cached):
        raise RuntimeError(f"{record['scene_id']}: cached RGB differs from direct historical Decord output")
    return {
        "source_video": str(source),
        "source_frame_count": len(reader),
        "frame_indices": expected_indices,
        "cache_equals_direct_decord": True,
        "frame_count": 32,
    }


def run_one(model: Any, frames: Any, frame_metadata: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from PIL import Image

    images = [Image.fromarray(frame.numpy(), mode="RGB") for frame in frames]
    frame_meta = [{"frame_index": index} for index in range(32)]
    with torch.no_grad():
        output = model(images, frame_meta)
    mask_logits = output.get("mask_logits")
    class_logits = output.get("class_logits")
    if not isinstance(mask_logits, torch.Tensor) or not isinstance(class_logits, torch.Tensor):
        raise RuntimeError("EoMT output missing mask_logits/class_logits")
    if mask_logits.shape[0] != 32 or class_logits.shape[0] != 32:
        raise RuntimeError(f"EoMT output is not 32-frame: mask={tuple(mask_logits.shape)} class={tuple(class_logits.shape)}")
    if mask_logits.shape[1] != EXPECTED_NUM_Q or class_logits.shape[1] != EXPECTED_NUM_Q:
        raise RuntimeError(f"Unexpected query count: mask={tuple(mask_logits.shape)} class={tuple(class_logits.shape)}")
    if class_logits.shape[-1] != EXPECTED_NUM_CLASSES + 1:
        raise RuntimeError(f"Unexpected taxonomy dimension: {tuple(class_logits.shape)}")
    if not torch.isfinite(mask_logits).all() or not torch.isfinite(class_logits).all():
        raise RuntimeError("EoMT outputs contain non-finite values")
    mask_float = mask_logits.detach().float().cpu().contiguous()
    class_float = class_logits.detach().float().cpu().contiguous()
    mask_variance = nontrivial_variance(mask_float, name="mask_logits")
    class_variance = nontrivial_variance(class_float, name="class_logits")
    object_probs = torch.softmax(class_float, dim=-1)[..., :-1]
    max_object_score = float(object_probs.max())
    # Do not reject an otherwise valid tensor merely because a model assigns
    # no foreground confidence for this video.  Forward compatibility is
    # established by the selector/fusion parity gate; this cache records the
    # diagnostic without imposing a semantic "nonzero logits" heuristic.
    if not np.isfinite(max_object_score):
        raise RuntimeError("EoMT object class probability is non-finite")
    outputs = {
        "mask_logits": mask_float,
        "class_logits": class_float,
        "query_count": int(output["query_count"]),
        "mask_logit_shape": list(mask_float.shape),
        "class_logit_shape": list(class_float.shape),
        "mask_logits_variance": mask_variance,
        "class_logits_variance": class_variance,
        "max_object_class_probability": max_object_score,
    }
    frame_provenance = {
        **frame_metadata,
        "frame_count": 32,
        "frame_order": list(range(32)),
        "source_frame_indices": frame_metadata["source_frame_indices"],
        "rgb_dtype": "torch.uint8",
        "rgb_shape": list(frames.shape),
    }
    return outputs, frame_provenance


def cache_file_path(root: Path, scene: str) -> Path:
    return root / "eomt_outputs" / "scannet" / f"{scene}.pt"


def validate_output_file(path: Path, scene: str, expected_record: dict[str, Any]) -> dict[str, Any]:
    import torch

    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{scene}: output is not a dict")
    if payload.get("schema_version") != "eomt_outputs_32_v1":
        raise RuntimeError(f"{scene}: schema version mismatch")
    if payload.get("scene_id") != scene or payload.get("dataset") != "scannet":
        raise RuntimeError(f"{scene}: scene/dataset metadata mismatch")
    frame = payload.get("frame_provenance", {})
    if frame.get("frame_count") != 32 or frame.get("source_frame_indices") != expected_record.get("source_frame_indices"):
        raise RuntimeError(f"{scene}: frame provenance mismatch")
    mask = payload.get("mask_logits")
    cls = payload.get("class_logits")
    if not isinstance(mask, torch.Tensor) or not isinstance(cls, torch.Tensor):
        raise RuntimeError(f"{scene}: raw tensors missing")
    if mask.dtype != torch.float32 or cls.dtype != torch.float32:
        raise RuntimeError(f"{scene}: outputs are not float32: {mask.dtype} {cls.dtype}")
    if tuple(mask.shape[:2]) != (32, EXPECTED_NUM_Q) or tuple(cls.shape[:2]) != (32, EXPECTED_NUM_Q) or cls.shape[-1] != EXPECTED_NUM_CLASSES + 1:
        raise RuntimeError(f"{scene}: invalid output shapes: {tuple(mask.shape)} {tuple(cls.shape)}")
    if not torch.isfinite(mask).all() or not torch.isfinite(cls).all():
        raise RuntimeError(f"{scene}: outputs contain non-finite values")
    return {
        "scene_id": scene,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mask_logits_variance": nontrivial_variance(mask, name=f"{scene}.mask_logits"),
        "class_logits_variance": nontrivial_variance(cls, name=f"{scene}.class_logits"),
    }


def smoke(args: argparse.Namespace, records: dict[str, dict[str, Any]], input_provenance: dict[str, Any], model_provenance: dict[str, Any]) -> None:
    scene = args.smoke_scene or sorted(records)[0]
    record = records[scene]
    cache_path = Path(args.forward_root) / str(record["cache_relative_path"])
    frames, indices, frame_metadata = load_frame_payload(cache_path, scene)
    source_video = Path(args.video_root) / str(record["source_video_relative_path"])
    if args.skip_direct_decode:
        direct = {
            "skipped": True,
            "reason": "validated forward_frames_32_v1 cache is the authoritative local RGB source",
            "source_video_present": source_video.is_file(),
            "frame_indices": indices,
            "frame_count": 32,
        }
    else:
        direct = direct_decode_matches_cache(args, record, frames, indices)
    model = load_model(args)
    loaded = verify_loaded_checkpoint(model, args)
    outputs, frame_provenance = run_one(model, frames, {**frame_metadata, "source_frame_indices": indices}, args)
    payload = {
        "schema_version": "eomt_outputs_32_v1",
        "dataset": "scannet",
        "scene_id": scene,
        "frame_provenance": frame_provenance,
        **outputs,
        "checkpoint_config_provenance": model_provenance,
    }
    smoke_root = Path(args.output_root) / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_output = smoke_root / f"{scene}.pt"
    if smoke_output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {smoke_output}")
    import torch

    torch.save(payload, smoke_output)
    report = {
        "status": "PASS",
        "scene_id": scene,
        "frame_count": 32,
        "frame_indices_order_match": True,
        "direct_decord_match": direct,
        "eomt_model_available": True,
        "eomt_query_count": outputs["query_count"],
        "mask_logits_shape": outputs["mask_logit_shape"],
        "class_logits_shape": outputs["class_logit_shape"],
        "raw_output_validation": {
            "finite_tensors": True,
            "expected_32_frame_order": True,
            "expected_shapes_and_float32_dtype": True,
            "valid_class_dimension": EXPECTED_NUM_CLASSES + 1,
            "mask_logits_variance": outputs["mask_logits_variance"],
            "class_logits_variance": outputs["class_logits_variance"],
            "provenance_and_checksum": True,
        },
        "loaded_checkpoint": loaded,
        "input_provenance": input_provenance,
        "checkpoint_config_provenance": model_provenance,
        "output": {"path": str(smoke_output), "bytes": smoke_output.stat().st_size, "sha256": sha256_file(smoke_output)},
    }
    atomic_json(Path(args.output_root) / "smoke_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def full(args: argparse.Namespace, records: dict[str, dict[str, Any]], input_provenance: dict[str, Any], model_provenance: dict[str, Any]) -> None:
    smoke_report_path = Path(args.output_root) / "smoke_report.json"
    smoke_report = load_json(smoke_report_path) if smoke_report_path.is_file() else {}
    if smoke_report.get("status") != "PASS" or not isinstance(smoke_report.get("raw_output_validation"), dict):
        raise RuntimeError("Full cache is blocked: PASS smoke_report.json is required")
    model = load_model(args)
    loaded = verify_loaded_checkpoint(model, args)
    output_root = Path(args.output_root)
    output_dir = output_root / "eomt_outputs" / "scannet"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    resumed = 0
    for number, scene in enumerate(sorted(records), start=1):
        record = records[scene]
        output_path = cache_file_path(output_root, scene)
        if output_path.exists():
            validate_output_file(output_path, scene, record)
            resumed += 1
            completed += 1
            continue
        frame_path = Path(args.forward_root) / str(record["cache_relative_path"])
        frames, indices, frame_metadata = load_frame_payload(frame_path, scene)
        if indices != [int(value) for value in record["source_frame_indices"]]:
            raise RuntimeError(f"{scene}: frame cache indices differ from forward manifest")
        outputs, frame_provenance = run_one(model, frames, {**frame_metadata, "source_frame_indices": indices}, args)
        payload = {
            "schema_version": "eomt_outputs_32_v1",
            "dataset": "scannet",
            "scene_id": scene,
            "frame_provenance": frame_provenance,
            **outputs,
            "checkpoint_config_provenance": model_provenance,
        }
        partial = output_path.with_suffix(output_path.suffix + ".partial")
        if partial.exists():
            raise RuntimeError(f"Refusing to reuse ambiguous partial output: {partial}")
        import torch

        torch.save(payload, partial)
        os.replace(partial, output_path)
        validate_output_file(output_path, scene, record)
        completed += 1
        if number == 1 or number % 10 == 0 or number == len(records):
            elapsed = time.time() - started
            rate = completed / max(elapsed, 1e-6)
            print(f"[FULL] {completed}/{len(records)} scene={scene} resumed={resumed} rate={rate:.3f} videos/s", flush=True)

    checksum_records = [validate_output_file(cache_file_path(output_root, scene), scene, records[scene]) for scene in sorted(records)]
    checksum_manifest = {
        "schema_version": "eomt_outputs_32_v1_checksum_manifest",
        "status": "PASS",
        "file_count": len(checksum_records),
        "records": checksum_records,
        "input_provenance": input_provenance,
        "checkpoint_config_provenance": model_provenance,
        "loaded_checkpoint": loaded,
    }
    atomic_json(output_root / "checksums.json", checksum_manifest)
    total_bytes = sum(int(item["bytes"]) for item in checksum_records)
    report = {
        "status": "PASS",
        "file_count": len(checksum_records),
        "expected_file_count": EXPECTED_VIDEOS,
        "scene_set_exact": set(records) == {str(item["scene_id"]) for item in checksum_records},
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "checksum_manifest": str(output_root / "checksums.json"),
        "checksum_status": "PASS",
        "full_validation": "PASS",
        "resumed_files": resumed,
        "elapsed_seconds": time.time() - started,
        "input_provenance": input_provenance,
        "checkpoint_config_provenance": model_provenance,
    }
    atomic_json(output_root / "validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--authoritative-manifest", required=True)
    parser.add_argument("--forward-root", required=True)
    parser.add_argument("--forward-manifest", required=True)
    parser.add_argument("--expected-forward-manifest-sha256", default=None)
    parser.add_argument("--video-root", required=True)
    parser.add_argument(
        "--skip-direct-decode",
        action="store_true",
        help="Use the already validated local 32-frame RGB cache when original MP4s are not migrated.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--eomt-repo",
        default="/mnt/DATA_SSD/shaoruei/probing_data/eomt_runtime/third_party/EoMT",
    )
    parser.add_argument("--eomt-config", required=True)
    parser.add_argument("--eomt-checkpoint", required=True)
    parser.add_argument("--eomt-backbone", required=True)
    parser.add_argument("--eomt-source-revision", default=EXPECTED_EOMT_REVISION)
    parser.add_argument("--post-sft-checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-scene", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, input_provenance = validate_input_manifests(
        Path(args.authoritative_manifest),
        Path(args.forward_manifest),
        args.expected_forward_manifest_sha256,
    )
    if args.eomt_source_revision != EXPECTED_EOMT_REVISION:
        raise RuntimeError(f"Unexpected EoMT source revision: {args.eomt_source_revision}")
    model_provenance = build_provenance(args)
    if args.mode == "smoke":
        smoke(args, records, input_provenance, model_provenance)
    else:
        full(args, records, input_provenance, model_provenance)


if __name__ == "__main__":
    main()
