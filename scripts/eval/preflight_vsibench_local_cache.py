#!/usr/bin/env python3
"""Validate local VSiBench prompt metadata, 32-frame caches, and sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-dataset", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--spatial-features-root", type=Path, required=True)
    parser.add_argument("--spatial-features-subdir", required=True)
    args = parser.parse_args()

    dataset = Dataset.from_file(str(args.arrow_dataset))
    required = {(str(row["dataset"]), str(row["scene_name"])) for row in dataset}
    missing_frames: list[Path] = []
    missing_features: list[Path] = []
    for source, scene in sorted(required):
        frame_file = args.forward_frames_root / "frames" / source / f"{scene}.pt"
        feature_file = args.spatial_features_root / source / args.spatial_features_subdir / f"{scene}.pt"
        if not frame_file.is_file():
            missing_frames.append(frame_file)
        if not feature_file.is_file():
            missing_features.append(feature_file)

    print(f"[PREFLIGHT] prompts={len(dataset)} unique_videos={len(required)}")
    print(f"[PREFLIGHT] missing_32_frame_caches={len(missing_frames)}")
    print(f"[PREFLIGHT] missing_cut3r_token_sidecars={len(missing_features)}")
    if missing_frames or missing_features:
        for label, values in (("32-frame cache", missing_frames), ("CUT3R token sidecar", missing_features)):
            for value in values[:20]:
                print(f"[PREFLIGHT] missing {label}: {value}")
        raise SystemExit("Local VSiBench input preflight failed.")


if __name__ == "__main__":
    main()
