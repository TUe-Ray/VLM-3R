#!/usr/bin/env python3
"""Validate the locally migrated VSiBench MP4 set and required token sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


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
    missing_features: list[Path] = []
    for source, scene in sorted(required):
        video_file = args.video_root / source / f"{scene}.mp4"
        if not video_file.is_file():
            missing_videos.append(video_file)
        if not args.video_only:
            feature_file = args.spatial_features_root / source / args.spatial_features_subdir / f"{scene}.pt"
            if not feature_file.is_file():
                missing_features.append(feature_file)

    print(f"[PREFLIGHT] prompts={len(dataset)} unique_videos={len(required)}")
    print(f"[PREFLIGHT] video_root={args.video_root}")
    print(f"[PREFLIGHT] missing_mp4_videos={len(missing_videos)}")
    if not args.video_only:
        print(f"[PREFLIGHT] missing_cut3r_token_sidecars={len(missing_features)}")
    for label, values in (("MP4", missing_videos), ("CUT3R token sidecar", missing_features)):
        for value in values[:20]:
            print(f"[PREFLIGHT] missing {label}: {value}")
    if missing_videos or missing_features:
        raise SystemExit("Local VSiBench input preflight failed.")


if __name__ == "__main__":
    main()
