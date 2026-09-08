#!/usr/bin/env python
"""Record fixed depth-target statistics for a cached probing run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from depth_probe_common import load_frame_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    sample_indices = Path(args.sample_indices)
    report: dict[str, object] = {
        "sample_indices": str(sample_indices),
        "sample_indices_sha256": sha256(sample_indices),
        "splits": {},
    }
    for split in ("train", "val"):
        records = load_frame_records(sample_indices, split=split)
        valid_tokens = 0
        total_tokens = 0
        depth_sum = 0.0
        depth_min = float("inf")
        depth_max = float("-inf")
        missing: list[str] = []
        for record in records:
            frame_id = str(record["frame_sample_id"])
            gt_path = output_root / "gt_depth" / f"frame_{frame_id}.pt"
            meta_path = output_root / "metadata" / f"frame_{frame_id}.pt"
            if not gt_path.is_file() or not meta_path.is_file():
                missing.append(frame_id)
                continue
            gt = torch.load(gt_path, map_location="cpu").float()
            metadata = torch.load(meta_path, map_location="cpu")
            valid = metadata.get("gt_valid_mask", torch.isfinite(gt) & (gt > 0)).bool()
            values = gt[valid & torch.isfinite(gt) & (gt > 0)]
            total_tokens += int(gt.numel())
            valid_tokens += int(values.numel())
            if values.numel():
                depth_sum += float(values.sum().item())
                depth_min = min(depth_min, float(values.min().item()))
                depth_max = max(depth_max, float(values.max().item()))
        report["splits"][split] = {
            "frames": len(records),
            "missing_frames": missing,
            "total_grid_tokens": total_tokens,
            "valid_depth_tokens": valid_tokens,
            "mean_valid_depth": depth_sum / valid_tokens if valid_tokens else None,
            "min_valid_depth": depth_min if valid_tokens else None,
            "max_valid_depth": depth_max if valid_tokens else None,
        }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote target statistics to {destination}")


if __name__ == "__main__":
    main()
