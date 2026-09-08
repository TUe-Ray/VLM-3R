#!/usr/bin/env python
"""Print one authoritative ScanNet video with the strongest thing-query score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.sample_indices.read_text(encoding="utf-8"))
    best: tuple[float, str] | None = None
    for video in payload.get("videos", []):
        scene = str(video.get("scene_id") or Path(str(video["video_path"])).stem)
        path = args.cache_root / "class_logits" / "scannet" / f"{scene}.pt"
        try:
            cached = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            cached = torch.load(path, map_location="cpu")
        logits = cached.get("class_logits") if isinstance(cached, dict) else None
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (32, 200, 134):
            continue
        foreground = torch.softmax(logits.float(), dim=-1)[..., :-1]
        scores, classes = foreground.max(dim=-1)
        thing_scores = scores[classes < 80]
        score = float(thing_scores.max()) if thing_scores.numel() else float("-inf")
        candidate = (score, str(video["video_path"]))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise SystemExit("No authoritative video has a readable EoMT class-logit payload")
    # The score does not override any checkpoint selection rule.  It only
    # makes the one-video forward smoke deterministic and informative; an
    # empty object block remains a valid historical outcome.
    print(best[1])


if __name__ == "__main__":
    main()
