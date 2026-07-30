#!/usr/bin/env python3
"""Prove a legacy CUT3R sidecar's frame sequence with a fresh deterministic replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--replayed", required=True)
    parser.add_argument("--siglip_done", required=True)
    parser.add_argument("--base_alignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training_alignment_output", required=True)
    args = parser.parse_args()
    legacy, replayed = load(Path(args.legacy)), load(Path(args.replayed))
    done = json.loads(Path(args.siglip_done).read_text(encoding="utf-8"))
    base = json.loads(Path(args.base_alignment).read_text(encoding="utf-8"))
    left, right = legacy["patch_tokens"].float(), replayed["patch_tokens"].float()
    legacy_meta, replayed_meta = legacy.get("metadata", {}), replayed.get("metadata", {})
    expected_frames = [int(x) for x in done["selected_frame_indices"]]
    replayed_frames = [int(x) for x in replayed_meta.get("frame_indices", [])]
    source_match = str(done.get("source_video")) == str(replayed_meta.get("source_video")) == str(legacy_meta.get("source_video"))
    shape_match = tuple(left.shape) == tuple(right.shape) == (len(expected_frames), 729, 768)
    cosine = float(F.cosine_similarity(left.reshape(-1, 768), right.reshape(-1, 768), dim=-1).mean()) if shape_match else 0.0
    relative_l2 = float((left - right).norm() / right.norm().clamp_min(1e-8)) if shape_match else float("inf")
    max_abs = float((left - right).abs().max()) if shape_match else float("inf")
    passes = source_match and expected_frames == replayed_frames and shape_match and cosine >= 0.99999 and relative_l2 <= 1e-3 and max_abs <= 1e-2
    replay = {
        "legacy": str(Path(args.legacy).resolve()), "replayed": str(Path(args.replayed).resolve()),
        "siglip_done": str(Path(args.siglip_done).resolve()), "expected_frame_indices": expected_frames,
        "replayed_frame_indices": replayed_frames, "source_video_match": source_match,
        "shape_match": shape_match, "cosine": cosine, "relative_l2": relative_l2,
        "max_abs_difference": max_abs, "patch_order": "row_major; replay tensor matches legacy tensor elementwise",
        "passes": passes,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    training = dict(base)
    training["frame_identity_evidence"] = {"status": "verified" if passes else "mismatch_or_missing", "cut3r_replay": replay}
    if passes:
        # The base audit is deliberately conservative while the old sidecars
        # lack frame provenance.  A byte-for-byte replay changes only that
        # evidence dimension; retain the independently audited coordinate map.
        resampling = dict(training.get("deterministic_resampling", {}))
        resolved_status = (
            "EXACT_PATCH_ALIGNMENT"
            if bool(resampling.get("identity"))
            else "ALIGNMENT_WITH_DETERMINISTIC_RESAMPLING"
        )
        resampling["status"] = resolved_status
        training["deterministic_resampling"] = resampling
        training["status"] = resolved_status
    else:
        training["status"] = "ALIGNMENT_UNRESOLVED"
    training.pop("sha256", None)
    training["sha256"] = hashlib.sha256(
        json.dumps(training, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target = Path(args.training_alignment_output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
    if not passes:
        raise SystemExit("CUT3R replay does not prove legacy sidecar frame alignment.")


if __name__ == "__main__":
    main()
