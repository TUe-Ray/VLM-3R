#!/usr/bin/env python3
"""Render opt-in dual-path CUDA-memory JSONL telemetry as a rank table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GIB = 1024**3


def _gib(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value) / GIB:.2f}"


def _records(audit_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(audit_dir.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{number}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    if not args.audit_dir.is_dir():
        raise SystemExit(f"Audit directory does not exist: {args.audit_dir}")

    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for record in _records(args.audit_dir):
        key = (str(record.get("stage", "unknown")), int(record.get("rank", -1)))
        latest[key] = record
    if not latest:
        raise SystemExit(f"No rank_*.jsonl records in: {args.audit_dir}")

    stages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (stage, _rank), record in latest.items():
        stages[stage].append(record)

    print("stage\trank\tallocated_GiB\treserved_GiB\tpeak_alloc_GiB\tpeak_reserved_GiB\tfree_GiB\ttotal_GiB")
    for stage, records in stages.items():
        for record in sorted(records, key=lambda item: int(item["rank"])):
            print(
                f"{stage}\t{record['rank']}\t{_gib(record.get('allocated_bytes'))}\t"
                f"{_gib(record.get('reserved_bytes'))}\t{_gib(record.get('max_allocated_bytes'))}\t"
                f"{_gib(record.get('max_reserved_bytes'))}\t{_gib(record.get('free_bytes'))}\t"
                f"{_gib(record.get('total_bytes'))}"
            )


if __name__ == "__main__":
    main()
