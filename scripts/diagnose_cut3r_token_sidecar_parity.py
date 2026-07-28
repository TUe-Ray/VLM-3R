#!/usr/bin/env python3
"""Verify CUT3R final patch-token sidecars against a real deterministic recomputation.

The legacy ``--sidecar/--recomputed`` mode remains available.  Real mode selects
records from the training YAML, calls the same ``process_video_with_decord``
sampler used by the dataloader, records source frame indices, and recomputes
final CUT3R patch tokens before any 27x27-to-14x14 pooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.cut3r_token_sidecar_manifest import (
    load_cut3r_token_sidecar_manifest,
    validate_cut3r_token_sidecar_manifest_entry,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(payload: Any) -> dict[str, Any]:
    return payload.get("metadata", {}) if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) else {}


def _frame_order(payload: Any) -> list[int] | None:
    metadata = _metadata(payload)
    value = payload.get("frame_indices", payload.get("frame_order", metadata.get("frame_indices", metadata.get("frame_order")))) if isinstance(payload, dict) else None
    if value is None:
        return None
    return [int(x) for x in torch.as_tensor(value).flatten().tolist()]


def _tokens(payload: Any, label: str) -> torch.Tensor:
    tokens = payload.get("patch_tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, torch.Tensor):
        raise RuntimeError(f"{label} must be a CUT3R sidecar dict containing patch_tokens.")
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (729, 768):
        raise RuntimeError(f"{label} patch_tokens must be [F,729,768], got {tuple(tokens.shape)}.")
    if not torch.isfinite(tokens).all():
        raise RuntimeError(f"{label} patch_tokens contain non-finite values.")
    return tokens


def _compare(saved: dict[str, Any], recomputed: dict[str, Any], *, min_mean_cosine: float, min_frame_cosine: float) -> dict[str, Any]:
    saved_tokens, recomputed_tokens = _tokens(saved, "saved sidecar"), _tokens(recomputed, "recomputed sidecar")
    if tuple(saved_tokens.shape) != tuple(recomputed_tokens.shape):
        raise RuntimeError(f"CUT3R sidecar shape mismatch: saved={tuple(saved_tokens.shape)}, recomputed={tuple(recomputed_tokens.shape)}")
    saved_order, recomputed_order = _frame_order(saved), _frame_order(recomputed)
    if saved_order is None or recomputed_order is None:
        raise RuntimeError("Exact frame-order verification requires non-empty frame-index metadata for both saved and recomputed payloads.")
    if saved_order != recomputed_order:
        raise RuntimeError(f"CUT3R frame-order mismatch: saved={saved_order}, recomputed={recomputed_order}")
    saved_vectors, recomputed_vectors = saved_tokens.float().flatten(1), recomputed_tokens.float().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(saved_vectors, recomputed_vectors, dim=1)
    difference = (saved_vectors - recomputed_vectors).abs()
    report = {
        "frame_order": saved_order,
        "saved_tensor_shape": list(saved_tokens.shape),
        "recomputed_tensor_shape": list(recomputed_tokens.shape),
        "per_frame_cosine_similarity": [float(x) for x in cosine.tolist()],
        "minimum_cosine_similarity": float(cosine.min()),
        "mean_cosine_similarity": float(cosine.mean()),
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
        "thresholds": {"minimum_frame_cosine": min_frame_cosine, "mean_cosine": min_mean_cosine},
    }
    if report["mean_cosine_similarity"] < min_mean_cosine or report["minimum_cosine_similarity"] < min_frame_cosine:
        raise RuntimeError("CUT3R parity threshold failed: " + json.dumps(report, sort_keys=True))
    return report


def _legacy_mode(args: argparse.Namespace) -> dict[str, Any]:
    saved_path, recomputed_path = Path(args.sidecar).resolve(), Path(args.recomputed).resolve()
    if saved_path == recomputed_path:
        raise RuntimeError("Saved and recomputed sidecar paths must differ.")
    if _sha256(saved_path) == _sha256(recomputed_path):
        raise RuntimeError("Saved and recomputed sidecar hashes are identical; this is not independent parity evidence.")
    saved, recomputed = torch.load(saved_path, map_location="cpu"), torch.load(recomputed_path, map_location="cpu")
    report = _compare(saved, recomputed, min_mean_cosine=args.min_mean_cosine, min_frame_cosine=args.min_frame_cosine)
    report.update({"mode": "two_file", "sidecar_file": str(saved_path), "recomputed_file": str(recomputed_path), "sidecar_file_hash": _sha256(saved_path), "recomputed_file_hash": _sha256(recomputed_path)})
    return report


def _load_records(data_yaml: Path) -> list[dict[str, Any]]:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for dataset in payload.get("datasets", []):
        json_path = Path(dataset["json_path"])
        source = dataset.get("name") or json_path.stem
        for record in json.loads(json_path.read_text(encoding="utf-8")):
            item = dict(record)
            item["_annotation_path"] = str(json_path)
            item["_dataset_source"] = str(item.get("data_source") or source)
            if item.get("video"):
                records.append(item)
    if not records:
        raise RuntimeError(f"No video records were found in {data_yaml}.")
    return records


def _video_path(record: dict[str, Any], data_root: Path) -> Path:
    value = Path(str(record["video"]))
    return value if value.is_absolute() else (data_root / value)


def _sidecar_path(record: dict[str, Any], root: Path, subdir: str) -> Path:
    video = Path(str(record["video"]))
    source = str(record.get("data_source") or video.parts[0])
    return root / source / subdir / f"{video.stem}.pt"


def _resolve_saved_indices(payload: dict[str, Any], *, video_path: Path, selected_indices: list[int], frames_upbound: int, video_fps: int) -> tuple[list[int], str]:
    explicit = _frame_order(payload)
    if explicit is not None:
        return explicit, "sidecar_metadata"
    metadata = _metadata(payload)
    source_video = metadata.get("source_video")
    metadata_ok = (
        source_video is not None
        and Path(str(source_video)).name == video_path.name
        and int(metadata.get("num_frames", -1)) == len(selected_indices)
        and int(metadata.get("frames_upbound", -1)) == int(frames_upbound)
        and int(metadata.get("video_fps", -1)) == int(video_fps)
    )
    if not metadata_ok:
        raise RuntimeError(
            "Saved sidecar has no explicit frame_indices and cannot be resolved from deterministic sampler metadata; "
            "regenerate sidecars with frame_indices before smoke."
        )
    return list(selected_indices), "resolved_from_deterministic_sampler_metadata"


def _resolved_saved_indices(args, saved, video_path, sidecar_path, selected_indices):
    tokens = _tokens(saved, "saved sidecar")
    manifest = getattr(args, "_sidecar_manifest", None)
    if manifest is not None:
        indices = validate_cut3r_token_sidecar_manifest_entry(
            manifest, video_path=video_path, sidecar_path=sidecar_path, patch_tokens=tokens,
            selected_frame_indices=selected_indices, video_fps=args.video_fps,
            frames_upbound=args.frames_upbound, force_sample=True, require_verified=False,
        )
        return indices, "external_manifest"
    return _resolve_saved_indices(saved, video_path=video_path, selected_indices=selected_indices,
                                 frames_upbound=args.frames_upbound, video_fps=args.video_fps)


def _preflight_saved_alignments(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Validate exact saved-sidecar frame provenance before allocating CUT3R work."""
    from decord import VideoReader, cpu
    from llava.utils import process_video_with_decord

    data_root = Path(args.data_root).resolve()
    sidecar_root = Path(args.spatial_features_root).resolve()
    alignments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        video_path = _video_path(record, data_root).resolve()
        sidecar_path = _sidecar_path(record, sidecar_root, args.spatial_features_subdir).resolve()
        item: dict[str, Any] = {
            "dataset_source": record.get("_dataset_source"),
            "sample_id": record.get("id"),
            "video_path": str(video_path),
            "sidecar_file": str(sidecar_path),
        }
        selected_indices = None
        saved = None
        try:
            if not video_path.is_file():
                raise RuntimeError("training video is missing")
            if not sidecar_path.is_file():
                raise RuntimeError("training sidecar is missing")
            item["video_frame_count"] = int(len(VideoReader(str(video_path), ctx=cpu(0), num_threads=1)))
            sampler_args = SimpleNamespace(video_fps=args.video_fps, frames_upbound=args.frames_upbound, force_sample=True)
            _, _, _, num_frames, selected_indices = process_video_with_decord(str(video_path), sampler_args, return_indices=True)
            saved = torch.load(sidecar_path, map_location="cpu")
            tokens = _tokens(saved, "saved sidecar")
            item.update({"saved_tensor_shape": list(tokens.shape), "sidecar_file_hash": _sha256(sidecar_path)})
            saved_indices, index_source = _resolved_saved_indices(
                args, saved, video_path, sidecar_path, selected_indices
            )
            if saved_indices != selected_indices:
                raise RuntimeError(f"saved frame order differs from training sampler: sidecar={saved_indices}, sampler={selected_indices}")
            item.update({
                "selected_source_frame_indices": selected_indices,
                "sidecar_frame_index_metadata": saved_indices,
                "sidecar_frame_index_resolution": index_source,
                "saved_tensor_shape": list(tokens.shape),
                "sidecar_file_hash": _sha256(sidecar_path),
            })
            alignments.append(item)
        except Exception as exc:
            if selected_indices is not None:
                item["selected_source_frame_indices"] = selected_indices
            if isinstance(saved, dict):
                item["sidecar_payload_keys"] = sorted(str(key) for key in saved)
                item["sidecar_metadata_keys"] = sorted(str(key) for key in _metadata(saved))
            item["error"] = str(exc)
            failures.append(item)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"passed": not failures, "stage": "saved_sidecar_frame_alignment", "alignments": alignments, "failures": failures}
    report_path = output_dir / "parity_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(
            "Saved CUT3R sidecars cannot be frame-aligned with the active training sampler; "
            f"regenerate sidecars with explicit frame_indices. Report: {report_path}"
        )
    return alignments


def _build_processor(config_path: Path):
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

    config = json.loads(config_path.read_text(encoding="utf-8"))
    size = config.get("size", {"height": 384, "width": 384})
    return SigLipImageProcessor(
        image_mean=config.get("image_mean", (0.5, 0.5, 0.5)),
        image_std=config.get("image_std", (0.5, 0.5, 0.5)),
        size=(size["height"], size["width"]),
        resample=config.get("resample", 3),
        rescale_factor=config.get("rescale_factor", 1 / 255.0),
    ), config


def _load_cut3r(weights_path: Path, device: torch.device, precision: str):
    from scripts.extraction.extract_cut3r_layer_features import ARCroco3DStereo, Cut3rSpatialConfig

    if not weights_path.exists():
        raise RuntimeError(f"CUT3R checkpoint is missing: {weights_path}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    model = ARCroco3DStereo.from_pretrained(Cut3rSpatialConfig(weights_path=str(weights_path)).weights_path)
    model.eval().to(device=device, dtype=dtype)
    model.requires_grad_(False)
    return model, dtype


def _recompute(record: dict[str, Any], args: argparse.Namespace, cut3r, dtype: torch.dtype, processor, processor_config: dict[str, Any]) -> dict[str, Any]:
    from decord import VideoReader, cpu
    from scripts.extraction.extract_cut3r_layer_features import run_cut3r_layers
    from llava.utils import process_video_with_decord

    data_root, sidecar_root = Path(args.data_root).resolve(), Path(args.spatial_features_root).resolve()
    video_path = _video_path(record, data_root).resolve()
    sidecar_path = _sidecar_path(record, sidecar_root, args.spatial_features_subdir).resolve()
    if not video_path.is_file():
        raise RuntimeError(f"Training video does not exist: {video_path}")
    if not sidecar_path.is_file():
        raise RuntimeError(f"Training sidecar does not exist: {sidecar_path}")
    sampler_args = SimpleNamespace(video_fps=args.video_fps, frames_upbound=args.frames_upbound, force_sample=True)
    video, _, _, num_frames, selected_indices = process_video_with_decord(str(video_path), sampler_args, return_indices=True)
    saved = torch.load(sidecar_path, map_location="cpu")
    saved_indices, index_source = _resolved_saved_indices(args, saved, video_path, sidecar_path, selected_indices)
    if saved_indices != selected_indices:
        raise RuntimeError(f"Saved sidecar frame order differs from training sampler: sidecar={saved_indices}, sampler={selected_indices}")
    processed = processor.preprocess(images=video, return_tensors="pt")["pixel_values"]
    if tuple(processed.shape[-2:]) != (432, 432):
        processed = torch.nn.functional.interpolate(processed, size=(432, 432), mode="bilinear", align_corners=False)
    with torch.no_grad():
        output = run_cut3r_layers(cut3r, processed.unsqueeze(1).to(device=next(cut3r.parameters()).device, dtype=dtype), [12])
    patch_tokens = output[12][1][0, :num_frames].detach().to(device="cpu", dtype=torch.float16)
    recomputed = {
        "patch_tokens": patch_tokens,
        "metadata": {
            "frame_indices": selected_indices,
            "source_video": str(video_path),
            "frames_upbound": int(args.frames_upbound),
            "video_fps": int(args.video_fps),
            "num_frames": int(num_frames),
            "cut3r_weights_path": str(args.cut3r_weights_path),
            "preprocessing": processor_config,
        },
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recomputed_path = output_dir / f"recomputed_{record.get('id', video_path.stem)}.pt"
    if recomputed_path.resolve() == sidecar_path:
        raise RuntimeError("Recomputed output path collides with saved sidecar path.")
    torch.save(recomputed, recomputed_path)
    sidecar_hash, recomputed_hash = _sha256(sidecar_path), _sha256(recomputed_path)
    if sidecar_hash == recomputed_hash:
        raise RuntimeError("Saved and independently recomputed sidecars have identical hashes.")
    saved_for_compare = dict(saved)
    saved_metadata = dict(_metadata(saved))
    saved_metadata["frame_indices"] = saved_indices
    saved_for_compare["metadata"] = saved_metadata
    report = _compare(saved_for_compare, recomputed, min_mean_cosine=args.min_mean_cosine, min_frame_cosine=args.min_frame_cosine)
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    report.update({
        "mode": "real_recomputation",
        "dataset_source": record.get("_dataset_source"),
        "sample_id": record.get("id"),
        "video_path": str(video_path),
        "video_frame_count": len(vr),
        "sampling_policy": {"function": "llava.utils.process_video_with_decord", "video_fps": args.video_fps, "frames_upbound": args.frames_upbound, "force_sample": True},
        "selected_source_frame_indices": selected_indices,
        "sidecar_frame_index_metadata": saved_indices,
        "sidecar_frame_index_resolution": index_source,
        "sidecar_file": str(sidecar_path),
        "recomputed_file": str(recomputed_path),
        "sidecar_file_hash": sidecar_hash,
        "recomputed_file_hash": recomputed_hash,
        "cut3r_checkpoint_identity": str(Path(args.cut3r_weights_path).resolve()),
        "cut3r_preprocessing_configuration": processor_config,
    })
    return report


def _choose_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_id:
        chosen = [item for item in records if str(item.get("id")) == args.sample_id]
        if not chosen:
            raise RuntimeError(f"No training record has sample id {args.sample_id!r}.")
        return chosen[:1]
    if args.dataset_index is not None:
        return [records[args.dataset_index]]
    chosen, seen, seen_sources = [], set(), set()
    unique_records = []
    for item in records:
        key = (item.get("_dataset_source"), item.get("video"))
        if key not in seen:
            seen.add(key)
            unique_records.append(item)
    for item in unique_records:
        source = item.get("_dataset_source")
        if source not in seen_sources:
            chosen.append(item)
            seen_sources.add(source)
        if len(chosen) >= args.num_samples:
            return chosen
    for item in unique_records:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= args.num_samples:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar")
    parser.add_argument("--recomputed")
    parser.add_argument("--data-yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--spatial-features-root")
    parser.add_argument("--spatial-features-subdir", default="spatial_features")
    parser.add_argument("--sidecar-manifest")
    parser.add_argument("--sample-id")
    parser.add_argument("--dataset-index", type=int)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--cut3r-weights-path")
    parser.add_argument("--processor-config-path")
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="diagnostics/cut3r_token_sidecar_parity")
    parser.add_argument("--min-mean-cosine", type=float, default=0.995)
    parser.add_argument("--min-frame-cosine", type=float, default=0.99)
    args = parser.parse_args()
    args._sidecar_manifest = load_cut3r_token_sidecar_manifest(args.sidecar_manifest)
    two_file = args.sidecar is not None or args.recomputed is not None
    if two_file:
        if not args.sidecar or not args.recomputed:
            parser.error("--sidecar and --recomputed must be supplied together.")
        report = _legacy_mode(args)
        print("[CUT3R_TOKEN_PARITY] " + json.dumps(report, sort_keys=True))
        return 0
    required = ("data_yaml", "data_root", "spatial_features_root", "cut3r_weights_path", "processor_config_path")
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        parser.error("real recomputation mode requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("Real CUT3R parity recomputation requires a GPU allocation; do not run it on a login node.")
    records = _choose_records(_load_records(Path(args.data_yaml).resolve()), args)
    alignment_preflight = _preflight_saved_alignments(records, args)
    device = torch.device(args.device)
    cut3r, dtype = _load_cut3r(Path(args.cut3r_weights_path), device, args.precision)
    processor, processor_config = _build_processor(Path(args.processor_config_path))
    reports = [_recompute(record, args, cut3r, dtype, processor, processor_config) for record in records]
    payload = {"passed": True, "frame_alignment_preflight": alignment_preflight, "reports": reports}
    output = Path(args.output_dir).resolve() / "parity_report.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[CUT3R_TOKEN_PARITY] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[CUT3R_TOKEN_PARITY][FAIL] {exc}", file=sys.stderr)
        raise
