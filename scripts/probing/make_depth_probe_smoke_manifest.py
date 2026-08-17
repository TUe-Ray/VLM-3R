#!/usr/bin/env python
"""Create a tiny train/validation-preserving manifest for local probe smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_videos(videos: list[dict[str, Any]], split: str, count: int) -> list[dict[str, Any]]:
    selected = [video for video in videos if video.get("split") == split][:count]
    if len(selected) != count:
        raise ValueError(f"Needed {count} {split} videos, found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-videos", type=int, default=1)
    parser.add_argument("--val-videos", type=int, default=1)
    args = parser.parse_args()

    source = Path(args.sample_indices)
    payload = json.loads(source.read_text(encoding="utf-8"))
    videos = list(payload.get("videos", []))
    selected = select_videos(videos, "train", args.train_videos) + select_videos(videos, "val", args.val_videos)
    output = dict(payload)
    output["videos"] = selected
    output["train_videos"] = args.train_videos
    output["val_videos"] = args.val_videos
    output["train_frames"] = sum(len(video["frames"]) for video in selected if video["split"] == "train")
    output["val_frames"] = sum(len(video["frames"]) for video in selected if video["split"] == "val")
    output["smoke_source_manifest"] = str(source)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {destination}: {args.train_videos} train + {args.val_videos} val videos")


if __name__ == "__main__":
    main()
