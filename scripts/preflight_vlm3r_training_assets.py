#!/usr/bin/env python3
"""Fail-fast asset checks before launching distributed VLM-3R training."""

import argparse
import json
import os
import pathlib
import sys


def parse_json_paths(data_yaml):
    paths = []
    with open(data_yaml, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            if stripped.startswith("json_path:"):
                paths.append(stripped.split("json_path:", 1)[1].strip().strip("'\""))
    if not paths:
        raise ValueError(f"No json_path entries found in {data_yaml}")
    return paths


def load_items(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "annotations", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported dataset JSON shape in {json_path}: {type(data).__name__}")


def resolve_video_feature_path(video_rel_path, features_root, features_subdir, video_folder=None):
    video_pt_path = os.path.splitext(video_rel_path)[0] + ".pt"
    normalized_video_pt_path = video_pt_path.lstrip("/\\")
    path_parts = list(pathlib.PurePosixPath(normalized_video_pt_path).parts)

    path_parts_with_subdir = list(path_parts)
    if features_subdir in (None, ""):
        path_parts_with_subdir = list(path_parts)
    elif "videos" in path_parts_with_subdir:
        path_parts_with_subdir[path_parts_with_subdir.index("videos")] = features_subdir
    elif path_parts_with_subdir:
        path_parts_with_subdir = [features_subdir] + path_parts_with_subdir

    candidate_relative_paths = []
    for parts in (path_parts_with_subdir, path_parts):
        if not parts:
            continue
        rel_path = os.path.join(*parts)
        if rel_path not in candidate_relative_paths:
            candidate_relative_paths.append(rel_path)

    candidate_roots = []
    for root in (features_root, video_folder):
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    candidate_paths = []
    for root in candidate_roots:
        for rel_path in candidate_relative_paths:
            candidate_paths.append(os.path.join(root, rel_path))

    if os.path.isabs(video_pt_path):
        if features_subdir in (None, ""):
            abs_candidates = (video_pt_path,)
        else:
            abs_candidates = (video_pt_path.replace("/videos/", f"/{features_subdir}/", 1), video_pt_path)
        for abs_path in abs_candidates:
            if abs_path not in candidate_paths:
                candidate_paths.append(abs_path)

    return next((p for p in candidate_paths if os.path.exists(p)), None), candidate_paths


def add_missing(report, key, value, max_report):
    bucket = report.setdefault(key, [])
    if len(bucket) < max_report:
        bucket.append(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--spatial-features-root", default=None)
    parser.add_argument("--spatial-features-subdir", default="spatial_features")
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--require-spatial-features", action="store_true")
    parser.add_argument("--max-report", type=int, default=25)
    args = parser.parse_args()

    json_paths = parse_json_paths(args.data_yaml)
    report = {}
    counts = {
        "json_files": len(json_paths),
        "samples": 0,
        "video_samples": 0,
        "unique_videos": 0,
        "missing_videos": 0,
        "missing_spatial_features": 0,
    }

    seen_videos = set()
    for json_path in json_paths:
        items = load_items(json_path)
        counts["samples"] += len(items)
        for idx, item in enumerate(items):
            video_rel = item.get("video")
            if not video_rel:
                continue
            counts["video_samples"] += 1
            if video_rel in seen_videos:
                continue
            seen_videos.add(video_rel)

            video_path = video_rel if os.path.isabs(video_rel) else os.path.join(args.video_root, video_rel)
            if args.require_videos and not os.path.exists(video_path):
                counts["missing_videos"] += 1
                add_missing(
                    report,
                    "missing_videos",
                    {"json": json_path, "index": idx, "id": item.get("id"), "path": video_path},
                    args.max_report,
                )

            if args.require_spatial_features:
                spatial_path, candidates = resolve_video_feature_path(
                    video_rel,
                    args.spatial_features_root or args.video_root,
                    args.spatial_features_subdir,
                    video_folder=args.video_root,
                )
                if spatial_path is None:
                    counts["missing_spatial_features"] += 1
                    add_missing(
                        report,
                        "missing_spatial_features",
                        {
                            "json": json_path,
                            "index": idx,
                            "id": item.get("id"),
                            "video": video_rel,
                            "first_candidates": candidates[:3],
                        },
                        args.max_report,
                    )

    counts["unique_videos"] = len(seen_videos)
    print("[PREFLIGHT] " + json.dumps(counts, sort_keys=True))
    if report:
        print("[PREFLIGHT_ERROR] " + json.dumps(report, indent=2, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
