#!/usr/bin/env python3
"""Create a deterministic real-record dataset restricted to verified sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-yaml", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    verified = {
        str(Path(path).resolve())
        for path, entry in manifest.get("entries", {}).items()
        if entry.get("provenance_status") == "verified"
    }
    if not verified:
        raise RuntimeError("Manifest has no provenance_status=verified entries.")
    source = yaml.safe_load(Path(args.source_yaml).read_text())
    selected = []
    selected = []
    for dataset in source.get("datasets", []):
        annotation = Path(dataset["json_path"])
        for record in json.loads(annotation.read_text()):
            video = record.get("video")
            if not video:
                continue
            candidate = Path(video)
            if not candidate.is_absolute():
                root = Path(manifest["spatial_features_root"])
                candidate = root / candidate
                if not candidate.exists():
                    candidate = root / record.get("data_source", "") / "videos" / Path(video).name
            if str(candidate.resolve()) in verified:
                selected.append(record)
    unique = []
    seen = set()
    for record in selected:
        key = str(record.get("video"))
        if key not in seen:
            unique.append(record)
            seen.add(key)
    if not unique:
        raise RuntimeError("No source records matched verified manifest video paths.")
        raise RuntimeError("No source records matched verified manifest video paths.")
    repeated = [dict(unique[index % len(unique)]) for index in range(args.records)]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "verified_preflight_records.json"
    yaml_path = out / "verified_preflight.yaml"
    json_path.write_text(json.dumps(repeated, indent=2) + "\n")
    yaml_path.write_text(yaml.safe_dump({"datasets": [{"json_path": str(json_path), "sampling_strategy": "all"}]}, sort_keys=False))
    summary = {"manifest": str(Path(args.manifest).resolve()), "records": len(repeated), "unique_videos": [item["video"] for item in unique], "json": str(json_path), "yaml": str(yaml_path)}
    (out / "verified_preflight_dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
