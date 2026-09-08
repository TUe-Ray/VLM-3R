#!/usr/bin/env python
"""Build the fixed post-SFT VF manifest from already frozen video sets.

The existing pre-SFT development manifest supplies the six training and two
validation videos.  The separately frozen pre-SFT confirmation manifest
supplies the twelve held-out evaluation videos, but is renamed to ``dev_eval``
solely because the reusable ridge/VF implementation uses that split name.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def videos_by_split(payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [video for video in payload.get("videos", []) if str(video.get("split")) == split]


def clone_with_split(video: dict[str, Any], split: str) -> dict[str, Any]:
    clone = copy.deepcopy(video)
    clone["split"] = split
    for frame in clone.get("frames", []):
        frame["split"] = split
    return clone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Write the smallest isolated 1 train / 1 validation / 1 held-out extraction manifest.",
    )
    args = parser.parse_args()
    development_path = Path(args.development_manifest).resolve()
    evaluation_path = Path(args.evaluation_manifest).resolve()
    output = Path(args.output).resolve()
    development = json.loads(development_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    train = [clone_with_split(video, "train") for video in videos_by_split(development, "train")]
    val = [clone_with_split(video, "val") for video in videos_by_split(development, "val")]
    held_out = [clone_with_split(video, "dev_eval") for video in videos_by_split(evaluation, "confirmation")]
    if (len(train), len(val), len(held_out)) != (6, 2, 12):
        raise RuntimeError(
            "Expected frozen 6/2/12 inputs, got "
            f"train={len(train)}, val={len(val)}, held_out={len(held_out)}"
        )
    train_val_paths = {str(video["video_path"]) for video in train + val}
    if train_val_paths.intersection(str(video["video_path"]) for video in held_out):
        raise RuntimeError("Post-SFT held-out videos overlap ridge train/validation videos")
    if args.smoke:
        train, val, held_out = train[:1], val[:1], held_out[:1]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "post_sft_depth_subspace_manifest_v1",
        "name": "post_sft_depth_subspace_frozen_vf_v1_smoke" if args.smoke else "post_sft_depth_subspace_frozen_vf_v1",
        "source_development_manifest": str(development_path),
        "source_development_manifest_sha256": sha256(development_path),
        "source_evaluation_manifest": str(evaluation_path),
        "source_evaluation_manifest_sha256": sha256(evaluation_path),
        "selection_rule": (
            "Reuse the pre-SFT frozen train/validation and independently frozen evaluation video identities. "
            "Evaluation is renamed dev_eval only for the shared analysis API."
        ),
        "smoke": bool(args.smoke),
        "train_videos": len(train),
        "val_videos": len(val),
        "dev_eval_videos": len(held_out),
        "frames_per_video": 2,
        "videos": train + val + held_out,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "train": len(train), "val": len(val), "dev_eval": len(held_out)}, indent=2))


if __name__ == "__main__":
    main()
