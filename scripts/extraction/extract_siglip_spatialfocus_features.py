#!/usr/bin/env python3
"""Extract immutable-alignment SigLIP patch-token sidecars for SpatialFocus.

This program deliberately does *not* use the VLM mm_projector.  It delegates
feature selection to the project's vision-tower object, so its pre-projector
selection is identical to the one used by VLM-3R.

The workflow has separate phases:
  build-manifest  build the immutable CUT3R-aligned sample list once
  extract         run one independent worker per GPU (SLURM rank aware)
  summarize       merge rank-local logs and done markers
  verify-all      deserialize every completed tensor (slow, integrity audit)
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


SCHEMA_VERSION = 1
EXPECTED_SHAPE = (32, 729, 1152)
EXPECTED_DTYPE = "torch.bfloat16"
DEFAULT_FAST = "/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r"
DEFAULT_CUT3R_ROOT = "/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if readonly:
        path.chmod(0o444)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def relative_key(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def torch_load(path: Path) -> Any:
    # weights_only is both safer and sufficient for all supported CUT3R sidecars.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(str(path), map_location="cpu", weights_only=True)


def find_cut3r_files(root: Path, subdir: str | None = None) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"CUT3R root does not exist: {root}")
    if subdir:
        result: dict[str, Path] = {}
        for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
            feature_root = dataset / subdir
            if feature_root.is_dir():
                for path in sorted(feature_root.rglob("*.pt")):
                    result[f"{dataset.name}/{relative_key(feature_root, path)}"] = path
        return result
    return {relative_key(root, path): path for path in sorted(root.rglob("*.pt"))}


def _metadata_value(metadata: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in metadata and metadata[name] is not None:
            return metadata[name]
    return None


def alignment_from_metadata(
    metadata: dict[str, Any], *, dataset_root: Path | None
) -> tuple[list[int] | None, str | None, dict[str, Any]]:
    """Read recorded alignment without ever inventing a new sampler."""
    indices = _metadata_value(metadata, ("frame_indices", "frame_ids", "sampled_frame_indices"))
    indices = list(indices) if isinstance(indices, (list, tuple)) else None
    source_video = _metadata_value(metadata, ("source_video", "video_path", "source_path"))
    if indices is not None:
        if len(indices) != 32:
            raise ValueError("CUT3R metadata frame_indices is not a 32-frame sequence")
        try:
            indices = [int(index) for index in indices]
        except (TypeError, ValueError) as error:
            raise ValueError("frame indices are not integral") from error
    video_path = Path(str(source_video))
    if not video_path.is_absolute() and dataset_root is not None:
        video_path = dataset_root / video_path
    padding = {
        key: metadata[key]
        for key in ("padding", "padded", "padding_mask", "missing_frames", "frame_valid_mask")
        if key in metadata
    }
    return indices, str(video_path) if source_video else None, padding


def formal_training_frame_indices(source_video: str, metadata: dict[str, Any]) -> list[int]:
    """Call the exact VLM-3R video sampler used by formal SpatialStack training."""
    from llava.utils import process_video_with_decord

    video_fps = int(metadata.get("video_fps", 1))
    frames_upbound = int(metadata.get("frames_upbound", 32))
    if video_fps != 1 or frames_upbound != 32:
        raise ValueError(
            "Historical CUT3R metadata does not match formal SpatialStack's "
            f"video_fps=1/frames_upbound=32: {video_fps}/{frames_upbound}"
        )
    if not Path(source_video).is_file():
        raise FileNotFoundError(f"CUT3R source video is missing: {source_video}")
    sampler = SimpleNamespace(video_fps=1, frames_upbound=32, force_sample=True)
    _, _, _, _, indices = process_video_with_decord(source_video, sampler, return_indices=True)
    if len(indices) != 32:
        raise ValueError(f"Formal sampler produced {len(indices)} frames, expected 32")
    return [int(value) for value in indices]


def load_pipeline_alignment(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load an export made by the formal dataset pipeline, keyed by CUT3R key.

    The extractor intentionally accepts this strict interchange format instead
    of reimplementing video sampling.  A pipeline export contains the exact
    data item after VLM-3R's existing sampling/padding logic.
    """
    if path is None:
        return {}
    value = load_json(path)
    entries = value.get("entries", value)
    if not isinstance(entries, list):
        raise ValueError("pipeline alignment JSON must contain an entries list")
    result = {}
    for entry in entries:
        key = entry["key"]
        result[key] = entry
    return result


def metadata_alignment_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(entry["frame_indices"]),
        entry["source_video"],
        canonical_bytes(entry.get("padding", {})),
    )


def patch_shape(sidecar: Any, layer: str) -> tuple[int, int, int]:
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("patch_tokens"), torch.Tensor):
        raise ValueError(f"{layer} is not a CUT3R token sidecar with patch_tokens")
    shape = tuple(int(value) for value in sidecar["patch_tokens"].shape)
    if len(shape) != 3 or shape[1:] != (729, 768):
        raise ValueError(f"{layer} patch_tokens has incompatible shape {shape}")
    return shape


def command_build_manifest(args: argparse.Namespace) -> None:
    destination = Path(args.manifest).resolve()
    if destination.exists():
        existing = load_json(destination)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Existing manifest has unsupported schema: {destination}")
        print(f"Immutable manifest already exists: {destination} ({existing['digest']})")
        return

    roots = parse_mapping(args.cut3r_layer_root)
    subdirs: dict[str, str] = {}
    for raw in args.cut3r_subdir:
        if "=" not in raw:
            raise ValueError(f"Expected layer=subdir, got {raw!r}")
        layer, subdir = raw.split("=", 1)
        subdirs[layer] = subdir
    expected_layers = tuple(sorted(roots))
    if len(roots) < 3:
        raise ValueError("Pass all CUT3R roots, normally 6=..., 9=..., 12=...")
    if args.alignment_layer not in roots:
        raise ValueError(f"--alignment-layer {args.alignment_layer!r} is not one of {sorted(roots)}")
    files_by_layer = {layer: find_cut3r_files(root, subdirs.get(layer)) for layer, root in roots.items()}
    common = set.intersection(*(set(files) for files in files_by_layer.values()))
    union = set.union(*(set(files) for files in files_by_layer.values()))
    pipeline = load_pipeline_alignment(Path(args.pipeline_alignment_json) if args.pipeline_alignment_json else None)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    entries: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}

    for key in sorted(common):
        try:
            # Reading every tensor storage merely to inspect shape would consume
            # the entire debug allocation before extraction starts.  The three
            # directory key sets prove coverage here; each emitted sample then
            # fully validates all three referenced sidecars before its SigLIP
            # output is published.  Layer 6 is the canonical alignment source.
            exported = pipeline.get(key)
            if args.source_video_root:
                parts = Path(key).parts
                source_video = str(Path(args.source_video_root).resolve() / parts[0] / Path(*parts[1:]).with_suffix(".mp4"))
                if not Path(source_video).is_file():
                    raise FileNotFoundError(f"Derived source video is missing: {source_video}")
                frame_indices = None
                padding = {}
                alignment_source = "official_cut3r_relative_video_layout_runtime_sampler"
            else:
                reference_sidecar = torch_load(files_by_layer[args.alignment_layer][key])
                reference_shape = patch_shape(reference_sidecar, args.alignment_layer)
                reference_metadata = reference_sidecar.get("metadata", {}) if isinstance(reference_sidecar, dict) else {}
                indices, source_video, padding = alignment_from_metadata(reference_metadata, dataset_root=dataset_root)
                if source_video:
                    recorded = [(args.alignment_layer, indices, source_video, padding, reference_metadata)]
                else:
                    recorded = []
                if recorded:
                    _, recorded_indices, source_video, padding, source_metadata = recorded[0]
                    if recorded_indices is None:
                        frame_indices = formal_training_frame_indices(source_video, source_metadata)
                        alignment_source = "formal_spatialstack_sampler_reconstructed"
                    else:
                        frame_indices = recorded_indices
                        alignment_source = "cut3r_sidecar_metadata"
                elif exported is not None:
                    source_video = str(exported["source_video"])
                    frame_indices = [int(value) for value in exported["frame_indices"]]
                    padding = exported.get("padding", {})
                    alignment_source = "formal_spatialstack_pipeline_export"
                else:
                    raise ValueError("No CUT3R source_video metadata or formal pipeline export is available")
                if len(frame_indices) != 32 or reference_shape[0] != len(frame_indices):
                    raise ValueError("canonical frame alignment is not 32 frames")
            if frame_indices is not None and len(frame_indices) != 32:
                raise ValueError("canonical frame alignment is not 32 frames")
            parts = Path(key).parts
            if not parts:
                raise ValueError("empty key")
            entries.append(
                {
                    "key": key,
                    "dataset": parts[0],
                    "relative_output": Path(*parts[1:]).as_posix(),
                    "frame_indices": frame_indices,
                    "source_video": source_video,
                    "padding": padding,
                    "alignment_source": alignment_source,
                    "cut3r": {layer: str(files_by_layer[layer][key]) for layer in expected_layers},
                }
            )
        except Exception as error:  # keep auditability: invalid keys never silently disappear
            unavailable[key] = str(error)

    missing_by_layer = {
        key: sorted(layer for layer, files in files_by_layer.items() if key not in files)
        for key in sorted(union - common)
    }
    contract = {
        "siglip_checkpoint": args.siglip_checkpoint,
        "vision_select_layer": -2,
        "vision_select_feature": args.vision_select_feature,
        "frames": 32,
        "shape": list(EXPECTED_SHAPE),
        "dtype": EXPECTED_DTYPE,
        "git_commit": git_commit(),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cut3r_layers": list(expected_layers),
        "contract": contract,
        "entries": entries,
        "unavailable": unavailable,
        "missing_by_layer": missing_by_layer,
    }
    payload["digest"] = digest(payload)
    atomic_json(destination, payload, readonly=True)
    print(json.dumps({"manifest": str(destination), "digest": payload["digest"], "expected": len(entries), "unavailable": len(unavailable), "missing_by_layer": len(missing_by_layer)}, indent=2))


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    claimed = manifest.get("digest")
    base = dict(manifest)
    base.pop("digest", None)
    if claimed != digest(base):
        raise RuntimeError(f"Manifest digest mismatch: {path}")
    if tuple(manifest["contract"]["shape"]) != EXPECTED_SHAPE:
        raise RuntimeError("Manifest does not describe the required [32,729,1152] output")
    return manifest


def output_path(output_root: Path, entry: dict[str, Any]) -> Path:
    return output_root / entry["dataset"] / "siglip_features_dec_m2" / entry["relative_output"]


def selected_entries(manifest: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    entries = manifest["entries"]
    if max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    return entries if max_samples == 0 else entries[:max_samples]


def marker_path(feature: Path) -> Path:
    return feature.with_name(feature.name + ".done.json")


def fast_done(feature: Path, entry: dict[str, Any], manifest: dict[str, Any]) -> bool:
    marker = marker_path(feature)
    if not feature.is_file() or not marker.is_file():
        return False
    try:
        value = load_json(marker)
        return (
            value.get("key") == entry["key"]
            and value.get("manifest_digest") == manifest["digest"]
            and tuple(value.get("shape", ())) == EXPECTED_SHAPE
            and value.get("dtype") == EXPECTED_DTYPE
            and value.get("bytes") == feature.stat().st_size
            and value.get("status") == "complete"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def progress_writer(root: Path, run_id: str, rank: int):
    directory = root / "progress" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{rank:06d}.jsonl"
    handle = path.open("a", encoding="utf-8")

    def write(event: str, entry: dict[str, Any], **extra: Any) -> None:
        line = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "event": event, "key": entry["key"], "rank": rank, **extra}
        handle.write(json.dumps(line, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    return handle, write


def load_official_tower(checkpoint: str, select_feature: str, device: torch.device):
    """Construct the repository's exact pre-projector SigLIP tower path."""
    from llava.model.multimodal_encoder.builder import build_vision_tower

    config = SimpleNamespace(
        mm_vision_tower=checkpoint,
        vision_tower=checkpoint,
        mm_vision_select_layer=-2,
        mm_vision_select_feature=select_feature,
        unfreeze_mm_vision_tower=False,
    )
    tower = build_vision_tower(config)
    if not getattr(tower, "is_loaded", False) and hasattr(tower, "load_model"):
        tower.load_model()
    tower.vision_tower.to(device=device, dtype=torch.bfloat16)
    tower.vision_tower.eval()
    if select_feature != "patch" or not hasattr(tower, "image_processor"):
        raise RuntimeError("SpatialFocus requires the official SigLIP patch-only feature strategy")
    # SigLipVisionTower.load_model() removes the terminal encoder layer and its
    # forward() returns hidden_states[-1].  Therefore this is exactly the
    # full-tower hidden_states[-2] path used by VLM-3R, before mm_projector.
    strategy = f"{type(tower).__module__}.{type(tower).__name__}.forward(hidden_states[-1] after terminal-layer removal = full hidden_states[-2])"
    return tower, strategy


def validate_cut3r_entry(entry: dict[str, Any]) -> None:
    """Validate all aligned CUT3R layer sidecars before publishing a sample."""
    for layer, raw_path in entry["cut3r"].items():
        sidecar = torch_load(Path(raw_path))
        if patch_shape(sidecar, layer)[0] != 32:
            raise RuntimeError(f"CUT3R layer {layer} has a non-32 frame count for {entry['key']}")
        metadata = sidecar.get("metadata", {}) if isinstance(sidecar, dict) else {}
        recorded_indices, recorded_video, _ = alignment_from_metadata(metadata, dataset_root=None)
        if recorded_video and str(Path(recorded_video)) != str(Path(entry["source_video"])):
            raise RuntimeError(f"CUT3R layer {layer} source video differs for {entry['key']}")
        if entry.get("frame_indices") is not None and recorded_indices is not None and recorded_indices != entry["frame_indices"]:
            raise RuntimeError(f"CUT3R layer {layer} frame order differs for {entry['key']}")


def feature_tensor(tower: Any, entry: dict[str, Any], device: torch.device) -> torch.Tensor:
    from llava.utils import process_video_with_decord

    video_path = Path(entry["source_video"])
    if not video_path.is_file():
        raise FileNotFoundError(f"Recorded CUT3R source video is absent: {video_path}")
    sampler = SimpleNamespace(video_fps=1, frames_upbound=32, force_sample=True)
    video, _, _, _, sampled_indices = process_video_with_decord(str(video_path), sampler, return_indices=True)
    if entry.get("frame_indices") is not None and [int(value) for value in sampled_indices] != [int(value) for value in entry["frame_indices"]]:
        raise RuntimeError(
            f"Formal VLM-3R sampler frame order changed for {video_path}: "
            f"{sampled_indices} != {entry['frame_indices']}"
        )
    pixels = tower.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        selected = tower(pixels)
    if tuple(selected.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"Official feature selection returned {tuple(selected.shape)}, expected {EXPECTED_SHAPE}")
    if not torch.isfinite(selected).all().item():
        raise RuntimeError("SigLIP feature selection produced non-finite values")
    return selected.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def publish(feature: Path, tensor: torch.Tensor, entry: dict[str, Any], manifest: dict[str, Any], metadata_digest: str, rank: int) -> None:
    feature.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature.with_name(f".{feature.name}.tmp.rank{rank}.{os.getpid()}.{uuid.uuid4().hex}")
    torch.save(tensor, temporary)
    check = torch_load(temporary)
    if not isinstance(check, torch.Tensor) or tuple(check.shape) != EXPECTED_SHAPE or check.dtype != torch.bfloat16 or not torch.isfinite(check).all().item():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary artifact validation failed: {temporary}")
    os.replace(temporary, feature)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "key": entry["key"],
        "manifest_digest": manifest["digest"],
        "extraction_metadata_digest": metadata_digest,
        "shape": list(EXPECTED_SHAPE),
        "dtype": EXPECTED_DTYPE,
        "bytes": feature.stat().st_size,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(marker_path(feature), marker)


def extraction_metadata(manifest: dict[str, Any], strategy: str) -> dict[str, Any]:
    contract = manifest["contract"]
    return {
        "siglip_checkpoint": contract["siglip_checkpoint"],
        "vision_select_layer": -2,
        "feature_selection_strategy": strategy,
        "preprocessing": "official VLM-3R process_video_with_decord(video_fps=1, frames_upbound=32, force_sample=True) + SigLipImageProcessor.preprocess",
        "frames": 32,
        "dtype": EXPECTED_DTYPE,
        "git_commit": git_commit(),
        "manifest_digest": manifest["digest"],
    }


def command_extract(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    root = Path(args.output_root).resolve()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank/world-size: {rank}/{world_size}")
    run_id = args.run_id or os.environ.get("SLURM_JOB_ID", "manual")
    handle, log = progress_writer(root, run_id, rank)
    device = torch.device(args.device or (f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"))
    tower, strategy = load_official_tower(manifest["contract"]["siglip_checkpoint"], manifest["contract"]["vision_select_feature"], device)
    metadata = extraction_metadata(manifest, strategy)
    metadata_digest = digest(metadata)
    if rank == 0:
        metadata_path = root / "metadata" / f"extraction-{manifest['digest']}.json"
        if not metadata_path.exists():
            atomic_json(metadata_path, {**metadata, "digest": metadata_digest}, readonly=True)
    processed = skipped = failed = 0
    try:
        for index, entry in enumerate(selected_entries(manifest, args.max_samples)):
            if index % world_size != rank:
                continue
            feature = output_path(root, entry)
            if fast_done(feature, entry, manifest):
                skipped += 1
                log("skipped", entry)
                continue
            log("claimed", entry)
            try:
                validate_cut3r_entry(entry)
                tensor = feature_tensor(tower, entry, device)
                publish(feature, tensor, entry, manifest, metadata_digest, rank)
                processed += 1
                log("completed", entry, bytes=feature.stat().st_size)
            except Exception as error:
                failed += 1
                log("error", entry, error=repr(error))
                print(f"[rank {rank}] {entry['key']}: {error}", file=sys.stderr)
        print(json.dumps({"rank": rank, "world_size": world_size, "processed": processed, "skipped": skipped, "failed": failed}))
        if failed and args.fail_on_error:
            raise RuntimeError(f"rank {rank} had {failed} extraction failures")
    finally:
        handle.close()


def scan(manifest: dict[str, Any], output_root: Path, *, verify: bool, max_samples: int) -> dict[str, Any]:
    completed: list[str] = []
    missing: list[str] = []
    corrupted: list[str] = []
    total_bytes = 0
    temporary: list[str] = []
    for entry in selected_entries(manifest, max_samples):
        feature = output_path(output_root, entry)
        temporary.extend(str(path) for path in feature.parent.glob(f".{feature.name}.tmp.*"))
        if not fast_done(feature, entry, manifest):
            (missing if not feature.exists() else corrupted).append(entry["key"])
            continue
        if verify:
            try:
                value = torch_load(feature)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != EXPECTED_SHAPE or value.dtype != torch.bfloat16 or not torch.isfinite(value).all().item():
                    corrupted.append(entry["key"])
                    continue
            except Exception:
                corrupted.append(entry["key"])
                continue
        completed.append(entry["key"])
        total_bytes += feature.stat().st_size
    return {"expected": len(selected_entries(manifest, max_samples)), "completed": completed, "missing": missing, "corrupted": corrupted, "temporary": temporary, "bytes": total_bytes}


def command_summarize(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    root = Path(args.output_root).resolve()
    results = scan(manifest, root, verify=args.verify_all, max_samples=args.max_samples)
    run_id = args.run_id or os.environ.get("SLURM_JOB_ID", "manual")
    claims: dict[str, set[int]] = collections.defaultdict(set)
    durations: list[float] = []
    for path in (root / "progress" / run_id).glob("rank-*.jsonl") if (root / "progress" / run_id).is_dir() else []:
        starts: dict[str, dt.datetime] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                if event.get("event") == "claimed":
                    claims[event["key"]].add(int(event["rank"]))
                    starts[event["key"]] = dt.datetime.fromisoformat(event["timestamp"])
                elif event.get("event") == "completed" and event["key"] in starts:
                    durations.append((dt.datetime.fromisoformat(event["timestamp"]) - starts[event["key"]]).total_seconds())
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue
    duplicates = sorted(key for key, ranks in claims.items() if len(ranks) > 1)
    throughput = (len(durations) / sum(durations)) if durations and sum(durations) else None
    remaining = len(results["missing"]) + len(results["corrupted"])
    eta_seconds_2node = (remaining / (throughput * 8)) if throughput else None
    summary = {
        "manifest_digest": manifest["digest"],
        "expected_sample_count": results["expected"],
        "completed_sample_count": len(results["completed"]),
        "missing_sample_ids": results["missing"],
        "duplicate_sample_ids": duplicates,
        "corrupted_sample_ids": results["corrupted"],
        "temporary_artifacts": results["temporary"],
        "actual_output_bytes": results["bytes"],
        "per_gpu_samples_per_second": throughput,
        "estimated_remaining_seconds_2_nodes": eta_seconds_2node,
        "verify_all": args.verify_all,
    }
    destination = root / "summaries" / f"{run_id}.json"
    atomic_json(destination, summary)
    print(json.dumps(summary, indent=2))


def command_partition(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    keys = [entry["key"] for entry in selected_entries(manifest, args.max_samples)]
    for world_size in (8, 32):
        assigned = [key for rank in range(world_size) for index, key in enumerate(keys) if index % world_size == rank]
        if sorted(assigned) != sorted(keys) or len(assigned) != len(set(assigned)):
            raise RuntimeError(f"Partition failed for world_size={world_size}")
    print("partition validation passed for world_size=8 and 32")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifest")
    build.add_argument("--manifest", required=True)
    build.add_argument("--cut3r-layer-root", action="append", required=True, help="layer=directory; pass 6, 9, and 12")
    build.add_argument("--cut3r-subdir", action="append", default=[], help="layer=feature subdirectory under every dataset")
    build.add_argument("--alignment-layer", default="6", help="CUT3R layer whose recorded metadata defines canonical alignment")
    build.add_argument("--source-video-root", help="verified CUT3R extractor input root; enables metadata-free immutable source mapping")
    build.add_argument("--siglip-checkpoint", required=True)
    build.add_argument("--vision-select-feature", default="patch")
    build.add_argument("--dataset-root")
    build.add_argument("--pipeline-alignment-json", help="strict export from the formal SpatialStack dataset pipeline")
    build.set_defaults(func=command_build_manifest)
    for name, function in (("extract", command_extract), ("summarize", command_summarize), ("verify-all", command_summarize), ("validate-partition", command_partition)):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output-root", required=name in {"extract", "summarize", "verify-all"})
        command.add_argument("--run-id")
        if name == "extract":
            command.add_argument("--device")
            command.add_argument("--fail-on-error", action="store_true")
        command.add_argument("--max-samples", type=int, default=0, help="0 means all canonical entries; used only for smoke jobs")
        if name in {"summarize", "verify-all"}:
            command.add_argument("--verify-all", action="store_true", default=name == "verify-all")
        command.set_defaults(func=function)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
