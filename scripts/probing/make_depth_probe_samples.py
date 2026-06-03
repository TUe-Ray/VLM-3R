#!/usr/bin/env python
"""Create the fixed frame-level sample split for VLM-3R depth probing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from depth_probe_common import (
    DEFAULT_DATA_YAML,
    DEFAULT_FAST_FEATURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POINT_MAPS_SUBDIR,
    frame_sample_id,
    frame_sample_indices,
    load_point_map_sidecar,
    load_yaml_dataset_records,
    raw_sample_ids,
    resolve_sidecar_path,
    scene_id_from_item,
    sidecar_num_frames,
    stable_sample,
    write_json,
)


def build_sidecar_index(feature_root: Path, point_maps_subdir: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dataset_dir in feature_root.iterdir():
        sidecar_dir = dataset_dir / point_maps_subdir
        if not sidecar_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        for path in sidecar_dir.glob("*.pt"):
            index[f"{dataset_name}/videos/{path.stem}.mp4"] = path
    return index


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_yaml_dataset_records(Path(args.data_yaml))
    sidecar_index = build_sidecar_index(Path(args.feature_root), args.point_maps_subdir)
    print(f"[INFO] Indexed {len(sidecar_index)} existing point-map sidecars from {args.feature_root}")
    candidates: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    skip_no_video = 0
    skip_duplicate_video = 0
    skip_no_sidecar = 0
    skip_short = 0
    for dataset_index, item in enumerate(records):
        video_path = item.get("video")
        if not video_path:
            skip_no_video += 1
            continue
        video_path = str(video_path)
        if video_path in seen_videos:
            skip_duplicate_video += 1
            continue
        seen_videos.add(video_path)
        point_path = sidecar_index.get(video_path)
        if point_path is None:
            point_path = resolve_sidecar_path(
                video_path,
                Path(args.feature_root),
                args.point_maps_subdir,
            )
        if point_path is None:
            skip_no_sidecar += 1
            continue
        candidates.append(
            {
                "dataset_index": dataset_index,
                "raw_sample_ids": raw_sample_ids(item, dataset_index),
                "video_path": video_path,
                "scene_id": scene_id_from_item(item),
                "source_dataset": str(item.get("data_source", "")),
                "annotation_path": str(item.get("_annotation_path", "")),
                "point_maps_path": str(point_path),
            }
        )
    print(
        "[INFO] Candidate scan: "
        f"eligible={len(candidates)} skip_no_video={skip_no_video} "
        f"skip_duplicate_video={skip_duplicate_video} "
        f"skip_no_sidecar={skip_no_sidecar} skip_short={skip_short}"
    )
    return candidates


def assign_split_and_frames(args: argparse.Namespace, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_count = int(args.train_videos)
    videos: list[dict[str, Any]] = []
    for selected_idx, item in enumerate(selected):
        num_frames = int(args.num_frames_per_video)
        if num_frames < int(args.frames_per_video):
            raise ValueError(f"--num-frames-per-video must be >= --frames-per-video, got {num_frames}")
        split = "train" if selected_idx < train_count else "val"
        video_sample_id = str(item["raw_sample_ids"][0])
        frames = []
        for frame_index in frame_sample_indices(
            int(num_frames),
            int(args.frames_per_video),
            int(args.seed),
            f"{item['dataset_index']}:{item['video_path']}",
        ):
            fsid = frame_sample_id(video_sample_id, frame_index)
            frames.append(
                {
                    "frame_sample_id": fsid,
                    "frame_index": int(frame_index),
                    "raw_frame_index": int(frame_index),
                    "split": split,
                }
            )
        video = dict(item)
        video.update(
            {
                "num_frames": int(num_frames),
                "video_sample_id": video_sample_id,
                "selected_order": selected_idx,
                "split": split,
                "frames": frames,
            }
        )
        videos.append(video)
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--point-maps-subdir", default=DEFAULT_POINT_MAPS_SUBDIR)
    parser.add_argument("--train-videos", type=int, default=2000)
    parser.add_argument("--val-videos", type=int, default=400)
    parser.add_argument("--frames-per-video", type=int, default=2)
    parser.add_argument("--num-frames-per-video", type=int, default=32, help="Frame-axis length of existing CUT3R sidecars/model input.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    candidates = build_candidates(args)
    total = int(args.train_videos) + int(args.val_videos)
    selected = stable_sample(candidates, total, int(args.seed))
    videos = assign_split_and_frames(args, selected)

    output_path = Path(args.output_json) if args.output_json else Path(args.output_root) / "sample_indices.json"
    payload = {
        "seed": int(args.seed),
        "train_videos": int(args.train_videos),
        "val_videos": int(args.val_videos),
        "frames_per_video": int(args.frames_per_video),
        "num_frames_per_video": int(args.num_frames_per_video),
        "train_frames": int(args.train_videos) * int(args.frames_per_video),
        "val_frames": int(args.val_videos) * int(args.frames_per_video),
        "data_yaml": str(args.data_yaml),
        "feature_root": str(args.feature_root),
        "point_maps_subdir": str(args.point_maps_subdir),
        "videos": videos,
    }
    write_json(output_path, payload)
    print(f"[INFO] Wrote {output_path}")
    print(
        f"[INFO] Selected {len(videos)} videos and "
        f"{sum(len(video['frames']) for video in videos)} frame samples."
    )


if __name__ == "__main__":
    main()
