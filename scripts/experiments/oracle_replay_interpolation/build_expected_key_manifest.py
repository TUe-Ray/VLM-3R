#!/usr/bin/env python3
"""Freeze the canonical 5,130-sample VSI-Bench key manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-total", type=int, default=5130)
    parser.add_argument("--task-name", default="vsibench")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    import yaml
    import pyarrow.parquet as pq

    task_path = Path(args.task_yaml).resolve()
    cfg = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    paths = [Path(path).resolve() for path in cfg["dataset_kwargs"]["data_files"][args.split]]
    rows = []
    for source_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        for row_index, row in enumerate(pq.read_table(path).to_pylist()):
            rows.append((source_index, str(path), row_index, row))
    if len(rows) != args.expected_total:
        raise RuntimeError(f"Manifest row count is {len(rows)}, expected {args.expected_total}.")

    entries, seen = [], set()
    for doc_id, (source_index, source_path, source_row, row) in enumerate(rows):
        row_json = canonical_json(row)
        row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        identity = {"task": args.task_name, "split": args.split, "doc_id": doc_id, "row_sha256": row_hash}
        key = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        if key in seen:
            raise RuntimeError(f"Canonical key collision at doc_id={doc_id}: {key}")
        seen.add(key)
        entries.append({**identity, "canonical_key": key, "source_index": source_index, "source_row": source_row, "source_path": source_path})
    if len(seen) != args.expected_total:
        raise RuntimeError(f"Manifest unique-key count is {len(seen)}, expected {args.expected_total}.")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "expected_keys.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for item in sorted(entries, key=lambda item: item["canonical_key"]):
            handle.write(canonical_json(item) + "\n")
    manifest_hash = sha256_path(manifest)
    provenance = {
        "algorithm": "sha256(task,split,doc_id,canonical-source-row-json):v1",
        "task_yaml": str(task_path), "task_yaml_sha256": sha256_path(task_path),
        "parquet_files": [{"path": str(path), "sha256": sha256_path(path)} for path in paths],
        "expected_total": args.expected_total, "unique_keys": len(seen), "collisions": 0,
        "manifest": str(manifest), "manifest_sha256": manifest_hash,
    }
    (output / "expected_keys.provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "expected_keys.sha256").write_text(manifest_hash + "  expected_keys.jsonl\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
