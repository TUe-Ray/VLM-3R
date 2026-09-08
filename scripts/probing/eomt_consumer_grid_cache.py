#!/usr/bin/env python
"""Build an exact-consumer-input EoMT cache for post-SFT probing.

The frozen EoMT network emits dense 160x160 mask logits.  The two historical
VLM consumers do not use those logits directly: they use ``sigmoid`` masks
resized to their token grids, together with the full class-logit tensor.  This
tool materialises that smaller FP32 representation without ever serialising a
full raw-logit corpus.

The cache deliberately keeps every query.  Query selection is prompt- and
configuration-dependent (notably for the selective word-matching branch), so
preselecting top-k queries or serialising a final gate is not forward-safe.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.eomt_raw_cache_migration import (  # noqa: E402
    EXPECTED_EOMT_REVISION,
    EXPECTED_NUM_CLASSES,
    EXPECTED_NUM_Q,
    EXPECTED_VIDEOS,
    atomic_json,
    build_provenance,
    load_frame_payload,
    load_model,
    sha256_file,
    torch_load,
    validate_input_manifests,
    verify_loaded_checkpoint,
)

# v2 is produced by the checkpoint-exact LayerScale EoMT loader.  v1 was
# generated while 48 learned LayerScale tensors were silently discarded and
# must never be consumed by a formal selective-fusion probe.
SCHEMA_VERSION = "eomt_consumer_grid_v2"
CLASS_SCHEMA_VERSION = "eomt_consumer_grid_class_logits_v2"
MASK_SCHEMA_VERSION = "eomt_consumer_grid_masks_v2"
INTERPOLATION = {"mode": "bilinear", "align_corners": False, "input": "sigmoid(mask_logits.float())"}


def parse_hw(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid must be H,W (for example 14,14)")
    try:
        height, width = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid dimensions must be integers") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return height, width


def tensor_summary(value: Any, *, name: str, shape: tuple[int | None, ...]) -> dict[str, Any]:
    """Validate an FP32 consumer tensor and return diagnostics.

    A zero-variance tensor is recorded rather than rejected.  It is unusual,
    but whether it is a valid model outcome must be decided by downstream
    forward parity rather than a heuristic cache writer.
    """

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is not a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if value.ndim != len(shape):
        raise ValueError(f"{name} rank {value.ndim} != expected {len(shape)}")
    for index, expected in enumerate(shape):
        if expected is not None and int(value.shape[index]) != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} violates dimension {index}={expected}")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")
    variance = float(value.var(unbiased=False).item())
    if not torch.isfinite(torch.tensor(variance)):
        raise RuntimeError(f"{name} variance is non-finite")
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": "float32",
        "finite": True,
        "variance": variance,
        "nontrivial_variance": variance > 0.0,
    }


def cache_paths(root: Path, scene: str, *, smoke: bool = False) -> dict[str, Path]:
    base = root / "smoke" if smoke else root
    return {
        "class_logits": base / "class_logits" / "scannet" / f"{scene}.pt",
        "object_masks": base / "object_masks" / "scannet" / f"{scene}.pt",
        "selective_masks": base / "selective_masks" / "scannet" / f"{scene}.pt",
    }


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"Refusing ambiguous partial cache output: {partial}")
    torch.save(payload, partial)
    os.replace(partial, path)


def run_eomt(model: Any, frames: Any) -> tuple[Any, Any, dict[str, Any]]:
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
    mask = mask_logits.detach().float().cpu().contiguous()
    classes = class_logits.detach().float().cpu().contiguous()
    raw = {
        "mask_logits": tensor_summary(mask, name="mask_logits", shape=(32, EXPECTED_NUM_Q, None, None)),
        "class_logits": tensor_summary(
            classes,
            name="class_logits",
            shape=(32, EXPECTED_NUM_Q, EXPECTED_NUM_CLASSES + 1),
        ),
        "query_count": int(output.get("query_count", -1)),
        "stuff_class_ids": sorted(int(item) for item in output.get("stuff_class_ids", ())),
    }
    if raw["query_count"] != EXPECTED_NUM_Q:
        raise RuntimeError(f"Unexpected EoMT query count {raw['query_count']}")
    # A 32-frame, 640px EoMT forward is close to the 12 GiB TITAN V budget.
    # The cache payloads are already CPU-resident, so return unused allocator
    # blocks between videos rather than letting a fragmented reserve grow until
    # the next scene cannot allocate its temporary mask head activation.
    del output, mask_logits, class_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mask, classes, raw


def consumer_masks(mask_logits: Any, target_hw: tuple[int, int]) -> Any:
    import torch.nn.functional as F

    # This is intentionally sigmoid-then-resize: it matches the historical
    # EoMT extractor, MaskGuidedPooler, and Selective3DGate contract.
    soft_masks = mask_logits.sigmoid()
    return F.interpolate(
        soft_masks,
        size=target_hw,
        mode=INTERPOLATION["mode"],
        align_corners=INTERPOLATION["align_corners"],
    ).contiguous()


def frame_provenance(frame_metadata: dict[str, Any], source_indices: list[int], frames: Any) -> dict[str, Any]:
    if len(source_indices) != 32:
        raise RuntimeError("EoMT cache requires exactly 32 source frame indices")
    return {
        **frame_metadata,
        "frame_count": 32,
        "frame_order": list(range(32)),
        "source_frame_indices": [int(item) for item in source_indices],
        "rgb_dtype": "torch.uint8",
        "rgb_shape": [int(item) for item in frames.shape],
    }


def global_provenance(args: argparse.Namespace, input_provenance: dict[str, Any]) -> dict[str, Any]:
    eomt_provenance = build_provenance(args)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "purpose": "post-SFT EoMT consumer-grid cache; no raw mask logits persisted",
        "input_provenance": input_provenance,
        "eomt_provenance": eomt_provenance,
        "consumers": {
            "object_masks": {
                "grid": list(args.object_grid),
                "consumer": "MaskGuidedPooler visual-token grid",
                "interpolation": INTERPOLATION,
            },
            "selective_masks": {
                "grid": list(args.selective_grid),
                "consumer": "Selective3DGate CUT3R patch-token grid",
                "interpolation": INTERPOLATION,
            },
            "class_logits": {
                "shape": [32, EXPECTED_NUM_Q, EXPECTED_NUM_CLASSES + 1],
                "consumer": "all query selection paths",
            },
        },
        "cache_policy": {
            "precision": "float32",
            "query_count": EXPECTED_NUM_Q,
            "preselection": "disabled; all queries retained",
            "raw_mask_logits_persisted": False,
            "actual_vlm_forward_parity": "pending; cache is blocked from formal extraction until recorded separately",
        },
    }


def write_global_provenance(root: Path, provenance: dict[str, Any]) -> str:
    path = root / "provenance.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Existing cache root has incompatible schema: {path}")
        return sha256_file(path)
    atomic_json(path, provenance)
    return sha256_file(path)


def build_payloads(
    *,
    scene: str,
    frame: dict[str, Any],
    classes: Any,
    object_masks: Any,
    selective_masks: Any,
    global_sha256: str,
    raw_diagnostics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    common = {
        "dataset": "scannet",
        "scene_id": scene,
        "frame_provenance": frame,
        "global_provenance_sha256": global_sha256,
        "raw_eomt_diagnostics": raw_diagnostics,
    }
    return {
        "class_logits": {
            "schema_version": CLASS_SCHEMA_VERSION,
            **common,
            "class_logits": classes,
        },
        "object_masks": {
            "schema_version": MASK_SCHEMA_VERSION,
            **common,
            "consumer": "object_masks",
            "interpolation": INTERPOLATION,
            "soft_masks": object_masks,
        },
        "selective_masks": {
            "schema_version": MASK_SCHEMA_VERSION,
            **common,
            "consumer": "selective_masks",
            "interpolation": INTERPOLATION,
            "soft_masks": selective_masks,
        },
    }


def validate_payloads(paths: dict[str, Path], scene: str, expected_record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payloads = {name: torch_load(path) for name, path in paths.items()}
    source_indices = [int(value) for value in expected_record["source_frame_indices"]]
    for name, payload in payloads.items():
        if not isinstance(payload, dict):
            raise TypeError(f"{scene}/{name}: cache payload is not a dict")
        expected_schema = CLASS_SCHEMA_VERSION if name == "class_logits" else MASK_SCHEMA_VERSION
        if payload.get("schema_version") != expected_schema:
            raise RuntimeError(f"{scene}/{name}: schema mismatch")
        if payload.get("dataset") != "scannet" or payload.get("scene_id") != scene:
            raise RuntimeError(f"{scene}/{name}: scene provenance mismatch")
        frame = payload.get("frame_provenance", {})
        if frame.get("frame_count") != 32 or frame.get("source_frame_indices") != source_indices:
            raise RuntimeError(f"{scene}/{name}: 32-frame provenance mismatch")
        if not isinstance(payload.get("global_provenance_sha256"), str):
            raise RuntimeError(f"{scene}/{name}: global provenance checksum missing")
    classes = payloads["class_logits"].get("class_logits")
    object_masks = payloads["object_masks"].get("soft_masks")
    selective_masks = payloads["selective_masks"].get("soft_masks")
    summaries = {
        "class_logits": tensor_summary(
            classes, name=f"{scene}.class_logits", shape=(32, EXPECTED_NUM_Q, EXPECTED_NUM_CLASSES + 1)
        ),
        "object_masks": tensor_summary(
            object_masks, name=f"{scene}.object_masks", shape=(32, EXPECTED_NUM_Q, *args.object_grid)
        ),
        "selective_masks": tensor_summary(
            selective_masks, name=f"{scene}.selective_masks", shape=(32, EXPECTED_NUM_Q, *args.selective_grid)
        ),
    }
    return {
        "scene_id": scene,
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "tensor_validation": summaries,
    }


def write_scene(
    *,
    root: Path,
    scene: str,
    record: dict[str, Any],
    model: Any,
    args: argparse.Namespace,
    global_sha256: str,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    frame_path = Path(args.forward_root) / str(record["cache_relative_path"])
    frames, source_indices, metadata = load_frame_payload(frame_path, scene)
    expected_indices = [int(value) for value in record["source_frame_indices"]]
    if source_indices != expected_indices:
        raise RuntimeError(f"{scene}: frame-cache indices differ from manifest")
    raw_mask_logits, classes, raw_diagnostics = run_eomt(model, frames)
    object_masks = consumer_masks(raw_mask_logits, args.object_grid)
    selective_masks = consumer_masks(raw_mask_logits, args.selective_grid)
    mask_diagnostics = {
        "object_masks": tensor_summary(
            object_masks, name="object_masks", shape=(32, EXPECTED_NUM_Q, *args.object_grid)
        ),
        "selective_masks": tensor_summary(
            selective_masks, name="selective_masks", shape=(32, EXPECTED_NUM_Q, *args.selective_grid)
        ),
    }
    frame = frame_provenance(metadata, source_indices, frames)
    paths = cache_paths(root, scene, smoke=smoke)
    payloads = build_payloads(
        scene=scene,
        frame=frame,
        classes=classes,
        object_masks=object_masks,
        selective_masks=selective_masks,
        global_sha256=global_sha256,
        raw_diagnostics={**raw_diagnostics, "consumer_masks": mask_diagnostics},
    )
    for name, path in paths.items():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        atomic_torch_save(path, payloads[name])

    validation = validate_payloads(paths, scene, record, args)
    reloaded = {name: torch_load(path) for name, path in paths.items()}
    parity = {
        "class_logits_bitwise_equal": bool(torch.equal(classes, reloaded["class_logits"]["class_logits"])),
        "object_masks_bitwise_equal": bool(torch.equal(object_masks, reloaded["object_masks"]["soft_masks"])),
        "selective_masks_bitwise_equal": bool(torch.equal(selective_masks, reloaded["selective_masks"]["soft_masks"])),
    }
    if not all(parity.values()):
        raise RuntimeError(f"{scene}: serialization parity failure: {parity}")
    return validation, parity


def smoke(
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    provenance_sha256: str,
    loaded: dict[str, Any],
    model: Any,
) -> None:
    scene = args.smoke_scene or sorted(records)[0]
    paths = cache_paths(Path(args.output_root), scene, smoke=True)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite existing smoke cache for {scene}")
    validation, parity = write_scene(
        root=Path(args.output_root), scene=scene, record=records[scene], model=model,
        args=args, global_sha256=provenance_sha256, smoke=True,
    )
    report = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene,
        "consumer_input_serialization_parity": parity,
        "validation": validation,
        "loaded_checkpoint": loaded,
        "actual_vlm_forward_parity": "PENDING_EOMT_VLM_FORWARD_INTEGRATION",
    }
    atomic_json(Path(args.output_root) / "smoke_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def full(
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    provenance_sha256: str,
    loaded: dict[str, Any],
    model: Any,
) -> None:
    smoke_report_path = Path(args.output_root) / "smoke_report.json"
    report = json.loads(smoke_report_path.read_text(encoding="utf-8")) if smoke_report_path.is_file() else {}
    if report.get("status") != "PASS" or not all(report.get("consumer_input_serialization_parity", {}).values()):
        raise RuntimeError("Full cache is blocked: PASS smoke_report with serialization parity is required")
    root = Path(args.output_root)
    started = time.time()
    results: list[dict[str, Any]] = []
    resumed = 0
    for number, scene in enumerate(sorted(records), start=1):
        paths = cache_paths(root, scene)
        if all(path.exists() for path in paths.values()):
            result = validate_payloads(paths, scene, records[scene], args)
            resumed += 1
        elif any(path.exists() for path in paths.values()):
            raise RuntimeError(f"{scene}: incomplete three-file consumer cache; resolve manually before resuming")
        else:
            result, parity = write_scene(
                root=root, scene=scene, record=records[scene], model=model, args=args,
                global_sha256=provenance_sha256, smoke=False,
            )
            if not all(parity.values()):
                raise RuntimeError(f"{scene}: write/read serialization parity failed")
        results.append(result)
        if number == 1 or number % 10 == 0 or number == len(records):
            elapsed = time.time() - started
            print(f"[FULL] {number}/{len(records)} scene={scene} resumed={resumed} rate={number / max(elapsed, 1e-6):.3f} videos/s", flush=True)
    checksum_manifest = {
        "schema_version": f"{SCHEMA_VERSION}_checksum_manifest",
        "status": "PASS",
        "file_count": len(results) * 3,
        "scene_count": len(results),
        "expected_scene_count": EXPECTED_VIDEOS,
        "records": results,
        "loaded_checkpoint": loaded,
        "global_provenance_sha256": provenance_sha256,
        "actual_vlm_forward_parity": "PENDING_EOMT_VLM_FORWARD_INTEGRATION",
    }
    atomic_json(root / "checksums.json", checksum_manifest)
    atomic_json(root / "validation.json", {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "scene_count": len(results),
        "file_count": len(results) * 3,
        "resumed_scenes": resumed,
        "elapsed_seconds": time.time() - started,
        "checksums": str(root / "checksums.json"),
        "actual_vlm_forward_parity": "PENDING_EOMT_VLM_FORWARD_INTEGRATION",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--authoritative-manifest", required=True)
    parser.add_argument("--forward-root", required=True)
    parser.add_argument("--forward-manifest", required=True)
    parser.add_argument("--expected-forward-manifest-sha256", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eomt-repo", default="/mnt/DATA_SSD/shaoruei/probing_data/eomt_runtime/third_party/EoMT")
    parser.add_argument("--eomt-config", required=True)
    parser.add_argument("--eomt-checkpoint", required=True)
    parser.add_argument("--eomt-backbone", required=True)
    parser.add_argument("--eomt-source-revision", default=EXPECTED_EOMT_REVISION)
    parser.add_argument("--post-sft-checkpoint", action="append", required=True)
    parser.add_argument("--object-grid", type=parse_hw, default=(14, 14))
    parser.add_argument("--selective-grid", type=parse_hw, default=(27, 27))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-scene", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eomt_source_revision != EXPECTED_EOMT_REVISION:
        raise RuntimeError(f"Unexpected EoMT source revision: {args.eomt_source_revision}")
    records, input_provenance = validate_input_manifests(
        Path(args.authoritative_manifest), Path(args.forward_manifest), args.expected_forward_manifest_sha256,
    )
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    provenance_sha256 = write_global_provenance(root, global_provenance(args, input_provenance))
    # Check state-dict compatibility before creating any per-video output, then
    # reuse the same frozen EoMT instance for the selected mode.
    model = load_model(args)
    loaded = verify_loaded_checkpoint(model, args)
    if args.mode == "smoke":
        smoke(args, records, provenance_sha256, loaded, model)
    else:
        full(args, records, provenance_sha256, loaded, model)


if __name__ == "__main__":
    main()
