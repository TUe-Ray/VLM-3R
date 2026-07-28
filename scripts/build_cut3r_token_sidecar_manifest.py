#!/usr/bin/env python3
"""Build a non-destructive frame-provenance manifest for legacy CUT3R tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.cut3r_token_sidecar_manifest import file_identity, sha256_file
from scripts.diagnose_cut3r_token_sidecar_parity import _choose_records, _load_records, _sidecar_path, _video_path
from llava.utils import process_video_with_decord


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--spatial-features-root", required=True)
    parser.add_argument("--spatial-features-subdir", default="spatial_features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--representative", action="store_true", help="Select the same cross-source sample set as the parity diagnostic.")
    parser.add_argument("--spot-parity-report")
    parser.add_argument("--promote-all-after-spot-parity", action="store_true", help="Mark the complete deterministic sidecar set verified only after a passing representative parity report.")
    parser.add_argument("--cut3r-checkpoint", default="")
    parser.add_argument("--extraction-script-commit", default="historical_shared_sampler")
    args = parser.parse_args()
    records = _load_records(Path(args.data_yaml).resolve())
    if args.representative:
        selector = SimpleNamespace(sample_id=None, dataset_index=None, num_samples=args.max_records or 3)
        unique = _choose_records(records, selector)
    else:
        unique, seen = [], set()
        for record in records:
            video = _video_path(record, Path(args.data_root).resolve()).resolve()
            if video not in seen:
                unique.append(record)
                seen.add(video)
        if args.max_records:
            unique = unique[:args.max_records]

    parity = {}
    if args.spot_parity_report:
        payload = json.loads(Path(args.spot_parity_report).read_text(encoding="utf-8"))
        if not payload.get("passed"):
            raise RuntimeError("Cannot promote a manifest from a failed spot-parity report.")
        parity = {str(Path(item["video_path"]).resolve()): item for item in payload.get("reports", [])}

    sampler = SimpleNamespace(video_fps=args.video_fps, frames_upbound=args.frames_upbound, force_sample=True)
    entries = {}
    for record in unique:
        video = _video_path(record, Path(args.data_root).resolve()).resolve()
        sidecar = _sidecar_path(record, Path(args.spatial_features_root).resolve(), args.spatial_features_subdir).resolve()
        if not video.is_file() or not sidecar.is_file():
            raise RuntimeError(f"Missing video or sidecar: video={video}, sidecar={sidecar}")
        _, _, _, _, indices = process_video_with_decord(str(video), sampler, return_indices=True)
        payload = torch.load(sidecar, map_location="cpu")
        tokens = payload.get("patch_tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tuple(tokens.shape[1:]) != (729, 768):
            raise RuntimeError(f"Invalid legacy CUT3R token sidecar: {sidecar}")
        if not torch.isfinite(tokens).all() or len(indices) != tokens.shape[0]:
            raise RuntimeError(f"Frame count or finite-value validation failed: {sidecar}")
        key = str(video)
        entries[key] = {
            "video_path": key,
            "video_identity": file_identity(video),
            "total_decoded_frames": None,
            "average_fps": None,
            "sampling": {"video_fps": args.video_fps, "frames_upbound": args.frames_upbound, "force_sample": True},
            "derived_frame_indices": [int(item) for item in indices],
            "sidecar_path": str(sidecar),
            "sidecar_identity": file_identity(sidecar, include_hash=True),
            "sidecar_hash": sha256_file(sidecar),
            "sidecar_shape": list(tokens.shape),
            "extraction_script_commit": args.extraction_script_commit,
            "cut3r_checkpoint": args.cut3r_checkpoint,
            "provenance_status": "verified" if (key in parity or (args.promote_all_after_spot_parity and bool(parity))) else "pending_spot_parity",
        }
    manifest = {
        "schema_version": 1,
        "data_yaml_path": str(Path(args.data_yaml).resolve()),
        "data_yaml_hash": sha256_file(args.data_yaml),
        "spatial_features_root": str(Path(args.spatial_features_root).resolve()),
        "spatial_features_subdir": args.spatial_features_subdir,
        "entries": entries,
        "spot_parity_report": str(Path(args.spot_parity_report).resolve()) if args.spot_parity_report else None,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "entries": len(entries), "verified_entries": sum(item["provenance_status"] == "verified" for item in entries.values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
