#!/usr/bin/env python
"""Prepare ScanNet-only semantic labels for VLM-3R probing."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import torch

from depth_probe_common import DEFAULT_OUTPUT_ROOT, read_json, write_json
from semantic_probe_common import (
    IGNORE_INDEX,
    build_label_mapping,
    downsample_train_labels_majority,
    find_semantic_label_path,
    infer_label_value_space,
    load_label_tensor,
    map_label_tensor_to_train_labels,
    read_scannet_raw_to_nyu40,
    tensor_label_stats,
    write_csv,
)
from train_depth_probes import available_feature_levels, feature_tensor_path


def frame_records_by_video(payload: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for video in payload.get("videos", []):
        cur = dict(video)
        cur["frames"] = [dict(frame) for frame in video.get("frames", [])]
        videos.append(cur)
    return videos


def source_value_counts(videos: list[dict[str, Any]], source_field: str) -> dict[str, int]:
    counts = Counter(str(video.get(source_field, "<missing>")) for video in videos)
    return dict(sorted(counts.items()))


def summarize_payload(payload: dict[str, Any], videos: list[dict[str, Any]]) -> dict[str, Any]:
    train_videos = sum(1 for video in videos if video.get("split") == "train")
    val_videos = sum(1 for video in videos if video.get("split") == "val")
    train_frames = sum(len(video.get("frames", [])) for video in videos if video.get("split") == "train")
    val_frames = sum(len(video.get("frames", [])) for video in videos if video.get("split") == "val")
    out = dict(payload)
    out["videos"] = videos
    out["train_videos"] = train_videos
    out["val_videos"] = val_videos
    out["train_frames"] = train_frames
    out["val_frames"] = val_frames
    return out


def filter_videos_with_frames(videos: list[dict[str, Any]], kept_frame_ids: set[str]) -> list[dict[str, Any]]:
    filtered = []
    for video in videos:
        frames = [dict(frame) for frame in video.get("frames", []) if str(frame.get("frame_sample_id")) in kept_frame_ids]
        if not frames:
            continue
        cur = dict(video)
        cur["frames"] = frames
        filtered.append(cur)
    return filtered


def first_successful_label_paths(
    videos: list[dict[str, Any]],
    label_root: Path,
    limit: int,
) -> list[Path]:
    paths = []
    for video in videos:
        for frame in video.get("frames", []):
            frame_index = int(frame.get("raw_frame_index", frame.get("frame_index")))
            path = find_semantic_label_path(label_root, str(video["scene_id"]), frame_index)
            if path is None:
                continue
            paths.append(path)
            if len(paths) >= limit:
                return paths
    return paths


def select_default_feature_levels(model_label: str, feature_levels: list[str] | None) -> list[str]:
    return feature_levels if feature_levels is not None else available_feature_levels(model_label)


def any_feature_cache_exists(output_root: Path, model_labels: list[str], feature_levels: list[str] | None, fsid: str) -> str | None:
    for model_label in model_labels:
        for level in select_default_feature_levels(model_label, feature_levels):
            path = feature_tensor_path(output_root, model_label, level, fsid)
            if path.exists():
                return str(path)
    return None


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sample-indices", default=str(DEFAULT_OUTPUT_ROOT / "sample_indices.json"))
    parser.add_argument("--scannet-label-root", required=True, help="Root containing ScanNet scene label-filt/label folders.")
    parser.add_argument("--scannet-label-tsv", required=True, help="Official scannet-labels.combined.tsv.")
    parser.add_argument("--scannet20-class-file", required=True, help="Official ScanNet20 class file in NYU40 id order.")
    parser.add_argument(
        "--label-value-space",
        choices=("auto", "raw_id", "nyu40id"),
        default="auto",
        help="Use conservative inference by default, or explicitly override the loaded label value space.",
    )
    parser.add_argument("--source-field", default="source_dataset")
    parser.add_argument("--source-value", default="scannet")
    parser.add_argument("--inspect-samples", type=int, default=8)
    parser.add_argument("--model-labels", default="zero_spatial,vlm3r_baseline")
    parser.add_argument("--feature-levels", default=None, help="Optional comma-separated levels used only for traceability path examples.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Prepare only first N filtered frames for smoke tests.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    sample_indices_path = Path(args.sample_indices)
    label_root = Path(args.scannet_label_root)
    payload = read_json(sample_indices_path)
    videos = frame_records_by_video(payload)
    source_counts = source_value_counts(videos, args.source_field)
    write_json(output_root / "semantic_probe_scannet_source_values.json", {
        "sample_indices": str(sample_indices_path),
        "source_field": args.source_field,
        "unique_source_values": source_counts,
    })
    if args.source_value not in source_counts:
        raise ValueError(
            f"Requested {args.source_field} == {args.source_value!r}, but available values are {source_counts}"
        )

    scannet_videos = [video for video in videos if str(video.get(args.source_field)) == args.source_value]
    if args.limit_frames is not None:
        remaining = int(args.limit_frames)
        limited = []
        for video in scannet_videos:
            if remaining <= 0:
                break
            frames = [dict(frame) for frame in video.get("frames", [])[:remaining]]
            remaining -= len(frames)
            if frames:
                cur = dict(video)
                cur["frames"] = frames
                limited.append(cur)
        scannet_videos = limited

    filtered_payload = summarize_payload(payload, scannet_videos)
    filtered_payload["source_field"] = args.source_field
    filtered_payload["source_value"] = args.source_value
    filtered_payload["unique_source_values_before_filtering"] = source_counts
    write_json(output_root / "semantic_probe_scannet_sample_indices.json", filtered_payload)

    inspect_paths = first_successful_label_paths(scannet_videos, label_root, int(args.inspect_samples))
    if not inspect_paths:
        raise FileNotFoundError(f"No semantic label maps found under {label_root} for filtered ScanNet frames")
    raw_to_nyu40 = read_scannet_raw_to_nyu40(Path(args.scannet_label_tsv))
    inspection_rows = []
    all_uniques: set[int] = set()
    for path in inspect_paths:
        tensor = load_label_tensor(path)
        stats = tensor_label_stats(path, tensor)
        inspection_rows.append(stats)
        all_uniques.update(int(v) for v in stats["unique_label_values"])
    inferred_label_value_space = "not_run"
    if args.label_value_space == "auto":
        try:
            inferred_label_value_space = infer_label_value_space(all_uniques, raw_to_nyu40=raw_to_nyu40)
            label_value_space = inferred_label_value_space
        except ValueError as exc:
            for row in inspection_rows:
                row["label_value_space_mode"] = args.label_value_space
                row["inferred_label_value_space"] = "ambiguous"
                row["applied_label_value_space"] = ""
                row["mapping_path_actually_applied"] = ""
                row["error"] = str(exc)
            write_json(output_root / "semantic_label_space_inspection.json", inspection_rows)
            write_csv(output_root / "semantic_label_space_inspection.csv", inspection_rows)
            raise
    else:
        label_value_space = args.label_value_space
        inferred_label_value_space = "manual_override_not_inferred"
    mapping = build_label_mapping(
        label_tsv=Path(args.scannet_label_tsv),
        scannet20_class_file=Path(args.scannet20_class_file),
        label_value_space=label_value_space,
        ignore_index=IGNORE_INDEX,
    )
    mapping["label_value_space_mode"] = args.label_value_space
    mapping["inferred_label_value_space"] = inferred_label_value_space
    mapping["applied_label_value_space"] = label_value_space
    mapping["label_value_space_overridden"] = args.label_value_space != "auto"
    write_json(output_root / "semantic_label_mapping_scannet.json", mapping)
    mapping_path = str(output_root / "semantic_label_mapping_scannet.json")
    for row in inspection_rows:
        row["label_value_space_mode"] = args.label_value_space
        row["inferred_label_value_space"] = inferred_label_value_space
        row["applied_label_value_space"] = label_value_space
        row["label_value_space_overridden"] = args.label_value_space != "auto"
        row["mapping_path_actually_applied"] = mapping_path
    write_json(output_root / "semantic_label_space_inspection.json", inspection_rows)
    write_csv(output_root / "semantic_label_space_inspection.csv", inspection_rows)

    model_labels = [part.strip() for part in args.model_labels.split(",") if part.strip()]
    feature_levels = parse_csv(args.feature_levels)
    gt_dir = output_root / "semantic_gt_scannet"
    meta_dir = output_root / "semantic_gt_scannet_metadata"
    alignment_rows: list[dict[str, Any]] = []
    kept_frame_ids: set[str] = set()

    for video in scannet_videos:
        for frame in video.get("frames", []):
            fsid = str(frame["frame_sample_id"])
            frame_index = int(frame.get("raw_frame_index", frame.get("frame_index")))
            label_path = find_semantic_label_path(label_root, str(video["scene_id"]), frame_index)
            feature_path_example = any_feature_cache_exists(output_root, model_labels, feature_levels, fsid)
            row = {
                "sample_id": fsid,
                "frame_sample_id": fsid,
                "feature_cache_filename": f"frame_{fsid}.pt",
                "feature_cache_path_example": feature_path_example or "",
                "scene_id": str(video["scene_id"]),
                "frame_index": frame_index,
                "semantic_label_path": str(label_path) if label_path else "",
                "matched_successfully": False,
                "reason_if_failed": "",
            }
            if label_path is None:
                row["reason_if_failed"] = "missing_semantic_label_path"
                alignment_rows.append(row)
                continue
            metadata_path = output_root / "metadata" / f"frame_{fsid}.pt"
            if not metadata_path.exists():
                row["reason_if_failed"] = f"missing_metadata:{metadata_path}"
                alignment_rows.append(row)
                continue
            try:
                metadata = torch.load(metadata_path, map_location="cpu")
                grid_shape = tuple(int(x) for x in metadata["visual_grid_shape"])
                label_tensor = load_label_tensor(label_path)
                train_labels = map_label_tensor_to_train_labels(label_tensor, mapping)
                grid = downsample_train_labels_majority(
                    train_labels,
                    grid_shape,  # type: ignore[arg-type]
                    num_classes=int(mapping["num_classes"]),
                    ignore_index=int(mapping["ignore_index"]),
                )
                gt_path = gt_dir / f"frame_{fsid}.pt"
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(grid, gt_path)
                trace = {
                    "sample_id": fsid,
                    "frame_sample_id": fsid,
                    "feature_cache_filename": f"frame_{fsid}.pt",
                    "feature_cache_path_example": feature_path_example,
                    "scene_id": str(video["scene_id"]),
                    "frame_index": frame_index,
                    "semantic_label_path": str(label_path),
                    "semantic_gt_path": str(gt_path),
                    "visual_grid_shape": list(grid_shape),
                    "label_mapping_path": mapping_path,
                }
                write_json(meta_dir / f"frame_{fsid}.json", trace)
                row["matched_successfully"] = True
                row["semantic_gt_path"] = str(gt_path)
                row["visual_grid_shape"] = list(grid_shape)
                kept_frame_ids.add(fsid)
            except Exception as exc:
                row["reason_if_failed"] = f"{type(exc).__name__}:{exc}"
            alignment_rows.append(row)

    write_json(output_root / "semantic_alignment_report.json", alignment_rows)
    write_csv(output_root / "semantic_alignment_report.csv", alignment_rows)
    semantic_aligned_payload = summarize_payload(payload, filter_videos_with_frames(scannet_videos, kept_frame_ids))
    semantic_aligned_payload["source_field"] = args.source_field
    semantic_aligned_payload["source_value"] = args.source_value
    semantic_aligned_payload["semantic_alignment_passed_frames"] = len(kept_frame_ids)
    write_json(output_root / "semantic_probe_scannet_semantic_aligned_sample_indices.json", semantic_aligned_payload)
    print(
        f"[INFO] Wrote {len(kept_frame_ids)} semantic GT tensors under {gt_dir}; "
        f"semantic-aligned index: {output_root / 'semantic_probe_scannet_semantic_aligned_sample_indices.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
