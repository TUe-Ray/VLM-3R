#!/usr/bin/env python
"""Freeze the unlabeled, prompt-preserving sample list used by C1 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sample-indices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", default="32", help="Positive count or 'all'.")
    parser.add_argument("--splits", default="train", help="Comma-separated probe splits; use 'all' for transductive calibration.")
    args = parser.parse_args()
    if str(args.num_samples).lower() != "all":
        try:
            args.num_samples = int(args.num_samples)
        except ValueError:
            parser.error("--num-samples must be a positive integer or 'all'")
        if args.num_samples <= 0:
            parser.error("--num-samples must be positive")

    source = Path(args.source_sample_indices).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    requested_splits = {value.strip().lower() for value in str(args.splits).split(",") if value.strip()}
    if not requested_splits:
        parser.error("--splits must not be empty")
    candidates = [
        dict(item)
        for item in payload.get("videos", [])
        if "all" in requested_splits or str(item.get("split", "")).strip().lower() in requested_splits
    ]
    candidates.sort(key=lambda item: (int(item.get("selected_order", 10**12)), str(item.get("video_path", ""))))
    selected = candidates if args.num_samples == "all" else candidates[: args.num_samples]
    if args.num_samples != "all" and len(selected) != args.num_samples:
        raise RuntimeError(
            f"Source manifest has only {len(selected)} requested-split samples, requested {args.num_samples}."
        )
    required = ("video_path",)
    for index, item in enumerate(selected):
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise RuntimeError(f"Calibration candidate {index} is missing {missing}: {item}")
    # Do not copy labels, targets, selected-frame depth data, or answers. The
    # dataset is reopened later solely to obtain this video's real user prompt.
    videos = [
        {
            key: item[key]
            for key in ("video_path", "source_dataset", "scene_name", "video_sample_id", "selected_order", "split")
            if key in item
        }
        for item in selected
    ]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "c1_calibration_manifest_v1",
        "selection": f"first_{','.join(sorted(requested_splits))}_selected_order_v1",
        "requested_splits": sorted(requested_splits),
        "source_sample_indices": str(source),
        "source_sample_indices_sha256": sha256_file(source),
        "num_samples": len(videos),
        "videos": videos,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[C1] wrote {len(videos)} unlabeled calibration samples: {output}")


if __name__ == "__main__":
    main()
