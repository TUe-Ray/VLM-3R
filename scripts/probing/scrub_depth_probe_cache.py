#!/usr/bin/env python3
"""Quarantine only unreadable probe-cache tensors before a --resume restart."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    roots = [
        args.output_root / "features" / args.model_label,
        args.output_root / "gt_depth",
        args.output_root / "metadata",
    ]
    files = sorted(path for root in roots if root.exists() for path in root.rglob("frame_*.pt"))
    quarantine = args.output_root / "quarantine_corrupt" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corrupt: list[dict[str, str]] = []
    for path in files:
        try:
            torch.load(path, map_location="cpu")
        except Exception as exc:  # preserve the original for forensic recovery
            quarantine.mkdir(parents=True, exist_ok=True)
            destination = quarantine / path.relative_to(args.output_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, destination)
            corrupt.append({"path": str(path), "quarantined_to": str(destination), "error": repr(exc)})

    report = {
        "schema_version": "depth_probe_cache_scrub_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root),
        "model_label": args.model_label,
        "checked_files": len(files),
        "corrupt_files": corrupt,
        "assessment": "PASS" if not corrupt else "PASS_WITH_QUARANTINED_CORRUPT_FILES",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
