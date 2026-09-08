#!/usr/bin/env python
"""Freeze video-disjoint development and confirmation manifests for depth-subspace analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import stable_int_seed, stable_sample, write_json  # noqa: E402


DEFAULT_SOURCE = (
    "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
    "scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("videos", payload.get("samples", [])):
            if isinstance(item, dict) and item.get("video_path"):
                excluded.add(str(item["video_path"]))
    return excluded


def copied_video(video: dict[str, Any], split: str, order: int) -> dict[str, Any]:
    result = json.loads(json.dumps(video))
    result["source_split"] = str(video.get("split", ""))
    result["split"] = split
    result["selected_order"] = int(order)
    for frame in result.get("frames", []):
        frame["split"] = split
    return result


def manifest_payload(
    *,
    source: Path,
    source_sha256: str,
    seed: int,
    videos: list[dict[str, Any]],
    name: str,
    excluded_paths: list[Path],
) -> dict[str, Any]:
    counts = {split: sum(video.get("split") == split for video in videos) for split in ("train", "val", "dev_eval", "confirmation")}
    return {
        "schema_version": "depth_subspace_manifest_v1",
        "name": name,
        "seed": int(seed),
        "source_manifest": str(source),
        "source_manifest_sha256": source_sha256,
        "excluded_video_manifests": [str(path) for path in excluded_paths],
        "train_videos": counts["train"],
        "val_videos": counts["val"],
        "dev_eval_videos": counts["dev_eval"],
        "confirmation_videos": counts["confirmation"],
        "frames_per_video": len(videos[0].get("frames", [])) if videos else 0,
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-videos", type=int, default=6)
    parser.add_argument("--val-videos", type=int, default=2)
    parser.add_argument("--dev-eval-videos", type=int, default=4)
    parser.add_argument("--confirmation-videos", type=int, default=12)
    parser.add_argument(
        "--exclude-video-manifest",
        action="append",
        default=[],
        help="Manifest containing video_path records that must not enter any output split; repeatable.",
    )
    args = parser.parse_args()
    source = Path(args.source_manifest).resolve()
    excluded_paths = [Path(value).resolve() for value in args.exclude_video_manifest]
    payload = json.loads(source.read_text(encoding="utf-8"))
    excluded = load_excluded(excluded_paths)
    all_videos = [video for video in payload.get("videos", []) if str(video.get("video_path", "")) not in excluded]
    train_candidates = sorted((video for video in all_videos if video.get("split") == "train"), key=lambda item: str(item["video_path"]))
    heldout_candidates = sorted((video for video in all_videos if video.get("split") == "val"), key=lambda item: str(item["video_path"]))
    heldout_count = int(args.val_videos) + int(args.dev_eval_videos) + int(args.confirmation_videos)
    if len(train_candidates) < int(args.train_videos) or len(heldout_candidates) < heldout_count:
        raise ValueError("Source manifest lacks enough non-excluded train or held-out videos")
    train = stable_sample(train_candidates, int(args.train_videos), stable_int_seed(args.seed, "depth_subspace", "train"))
    heldout = stable_sample(heldout_candidates, heldout_count, stable_int_seed(args.seed, "depth_subspace", "heldout"))
    selected: list[dict[str, Any]] = []
    selected.extend(copied_video(video, "train", len(selected)) for video in train)
    selected.extend(copied_video(video, "val", len(selected)) for video in heldout[: int(args.val_videos)])
    dev_start = int(args.val_videos)
    dev_end = dev_start + int(args.dev_eval_videos)
    selected.extend(copied_video(video, "dev_eval", len(selected)) for video in heldout[dev_start:dev_end])
    selected.extend(copied_video(video, "confirmation", len(selected)) for video in heldout[dev_end:])
    paths = {str(video["video_path"]) for video in selected}
    if len(paths) != len(selected):
        raise RuntimeError("Manifest construction produced duplicate videos")
    source_sha256 = sha256_file(source)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot = manifest_payload(
        source=source,
        source_sha256=source_sha256,
        seed=args.seed,
        videos=[video for video in selected if video["split"] != "confirmation"],
        name="depth_subspace_pilot_v1",
        excluded_paths=excluded_paths,
    )
    confirmation = manifest_payload(
        source=source,
        source_sha256=source_sha256,
        seed=args.seed,
        videos=[video for video in selected if video["split"] == "confirmation"],
        name="depth_subspace_confirmation_v1",
        excluded_paths=excluded_paths,
    )
    smoke_videos = (
        [video for video in selected if video["split"] == "train"][:4]
        + [video for video in selected if video["split"] == "val"][:2]
        + [video for video in selected if video["split"] == "dev_eval"][:2]
    )
    smoke = manifest_payload(
        source=source,
        source_sha256=source_sha256,
        seed=args.seed,
        videos=smoke_videos,
        name="depth_subspace_smoke_v1",
        excluded_paths=excluded_paths,
    )
    for filename, content in (
        ("depth_subspace_pilot_v1.json", pilot),
        ("depth_subspace_smoke_v1.json", smoke),
        ("depth_subspace_confirmation_v1.json", confirmation),
    ):
        write_json(output_dir / filename, content)
        print(f"[INFO] wrote {output_dir / filename}")


if __name__ == "__main__":
    main()
