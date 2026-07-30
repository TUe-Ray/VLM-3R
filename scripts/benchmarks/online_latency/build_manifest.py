#!/usr/bin/env python3
"""Build the small, deterministic raw-video latency manifest (never an eval task)."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from decord import VideoReader, cpu

from scripts.benchmarks.online_latency.common import GENERATION_ARGS, MANIFEST_VERSION, json_dump

DEFAULT_PARQUETS = [
    "/leonardo_scratch/fast/EUHPC_D32_006/vsibench/test_pruned.parquet",
    "/leonardo_scratch/fast/EUHPC_D32_006/vsibench/test_debiased.parquet",
]
DEFAULT_MEDIA_ROOT = "/leonardo_scratch/fast/EUHPC_D32_006/hf_cache/vsibench"


def prompt_for(row: dict) -> str:
    question = str(row["question"])
    question_type = str(row["question_type"])
    if question_type in {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}:
        return "These are frames of a video.\n" + question + "\nPlease answer the question using a single word or phrase."
    options = row.get("options")
    options = [] if options is None else list(options)
    return "\n".join(["These are frames of a video.", question, "Options:\n" + "\n".join(options),
                      "Answer with the option's letter from the given choices directly."])


def sample_frame_ids(video_path: Path, frames: int) -> list[int]:
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    count = len(vr)
    if count < frames:
        raise ValueError(f"{video_path} has {count} frames; expected >= {frames}")
    # Exact policy used by llava.utils.process_video_with_decord with force_sample.
    import numpy as np
    ids = np.linspace(0, count - 1, frames, dtype=int).tolist()
    if len(ids) != frames or len(set(ids)) != frames:
        raise ValueError(f"Invalid fixed-{frames} selection for {video_path}: {ids}")
    return ids


def build(args: argparse.Namespace) -> dict:
    frames = []
    for source_index, parquet in enumerate(args.parquet):
        frame = pd.read_parquet(parquet).copy()
        frame["_source_index"] = source_index
        frame["_source_row"] = range(len(frame))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    candidates = []
    for row in data.to_dict("records"):
        video = Path(args.media_root) / str(row["dataset"]) / f"{row['scene_name']}.mp4"
        if not video.is_file():
            continue
        prompt = prompt_for(row)
        key = f"{row['_source_index']}:{row['_source_row']}:{row['dataset']}:{row['scene_name']}:{row['question_type']}"
        rank = hashlib.sha256(f"{args.seed}:{key}".encode()).hexdigest()
        candidates.append((str(row["question_type"]), len(prompt), rank, key, row, video, prompt))
    # Light stratification only: round-robin categories after stable within-category ordering.
    by_category: dict[str, list] = {}
    for item in candidates:
        by_category.setdefault(item[0], []).append(item)
    for values in by_category.values():
        values.sort(key=lambda x: (x[1] % 97, x[2]))
    selected = []
    while len(selected) < args.samples and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(selected) < args.samples:
                selected.append(by_category[category].pop(0))
    if len(selected) != args.samples:
        raise RuntimeError(f"Only found {len(selected)} valid raw-video samples")
    entries = []
    for ordinal, (_, _, _, key, row, video, prompt) in enumerate(selected):
        try:
            frame_ids = sample_frame_ids(video, args.frames)
        except Exception:
            # Selection has already been deterministic; fail rather than silently replacing an entry.
            raise
        entries.append({
            "ordinal": ordinal, "split": "warmup" if ordinal < args.warmups else "measured",
            "canonical_key": key, "dataset": str(row["dataset"]), "scene_name": str(row["scene_name"]),
            "question_type": str(row["question_type"]), "raw_video_path": str(video.resolve()),
            "prompt": prompt, "prompt_characters": len(prompt), "frame_ids": frame_ids, "frame_order": frame_ids,
            "frame_count": args.frames, "source": {"parquet": str(args.parquet[int(row["_source_index"])]),
                       "row": int(row["_source_row"])},
        })
    return {"schema_version": MANIFEST_VERSION, "seed": args.seed, "raw_video_only": True,
            "frames_per_sample": args.frames, "warmup_count": args.warmups, "measured_count": args.samples - args.warmups,
            "generation_args": GENERATION_ARGS, "samples": entries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--parquet", action="append", default=None)
    parser.add_argument("--media-root", default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    args.parquet = args.parquet or DEFAULT_PARQUETS
    if args.samples != 20 or args.warmups != 4 or args.frames != 16:
        raise ValueError("This strict benchmark uses exactly 20 samples, 4 warm-ups and 16 frames.")
    json_dump(build(args), args.output)


if __name__ == "__main__":
    main()
