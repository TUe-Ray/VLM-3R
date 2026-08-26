#!/usr/bin/env python3
"""Validate the locally migrated VSiBench MP4 set and required token sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def spatial_feature_subdirs(specification: str) -> list[str]:
    """Return physical subdirectories from the evaluator's layered sidecar syntax."""
    values: list[str] = []
    for item in str(specification).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            _layer, item = item.split(":", 1)
            item = item.strip()
        if not item:
            raise ValueError(f"Invalid spatial_features_subdir entry: {specification!r}")
        values.append(item)
    if not values:
        raise ValueError("spatial_features_subdir must name at least one subdirectory")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-dataset", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--spatial-features-root", type=Path, required=True)
    parser.add_argument("--spatial-features-subdir", required=True)
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Validate MP4 coverage only; intended for decoder smoke tests, not model evaluation.",
    )
    args = parser.parse_args()

    dataset = Dataset.from_file(str(args.arrow_dataset))
    required = {(str(row["dataset"]), str(row["scene_name"])) for row in dataset}
    missing_videos: list[Path] = []
    feature_subdirs = spatial_feature_subdirs(args.spatial_features_subdir)
    missing_features: list[Path] = []
    for source, scene in sorted(required):
        video_file = args.video_root / source / f"{scene}.mp4"
        if not video_file.is_file():
            missing_videos.append(video_file)
        if not args.video_only:
            for feature_subdir in feature_subdirs:
                feature_file = args.spatial_features_root / source / feature_subdir / f"{scene}.pt"
                if not feature_file.is_file():
                    missing_features.append(feature_file)

    print(f"[PREFLIGHT] prompts={len(dataset)} unique_videos={len(required)}")
    print(f"[PREFLIGHT] video_root={args.video_root}")
    print(f"[PREFLIGHT] missing_mp4_videos={len(missing_videos)}")
    if not args.video_only:
        print(f"[PREFLIGHT] cut3r_token_subdirs={feature_subdirs}")
        print(f"[PREFLIGHT] missing_cut3r_token_sidecars={len(missing_features)}")
    for label, values in (("MP4", missing_videos), ("CUT3R token sidecar", missing_features)):
        for value in values[:20]:
            print(f"[PREFLIGHT] missing {label}: {value}")
    if missing_videos or missing_features:
        raise SystemExit("Local VSiBench input preflight failed.")


if __name__ == "__main__":
    main()
