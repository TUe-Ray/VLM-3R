#!/usr/bin/env python3
"""Validate experiment output before allowing a Slurm dependency to promote."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.output_path)
    results_paths = sorted(root.glob("**/results.json"))
    if len(results_paths) != 1:
        raise SystemExit(f"Expected exactly one results.json below {root}, found {len(results_paths)}.")
    results = json.loads(results_paths[0].read_text(encoding="utf-8"))
    if not isinstance(results.get("results"), dict):
        raise SystemExit(f"Invalid scored results file: {results_paths[0]}.")

    items = []
    rank_paths = sorted((root / "telemetry").glob("rank_*/samples.jsonl"))
    for path in rank_paths:
        items.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if len(items) != args.expected_samples:
        raise SystemExit(f"Telemetry count mismatch: expected {args.expected_samples}, got {len(items)}.")
    keys = [item.get("canonical_key") for item in items]
    if any(key in (None, "") for key in keys) or len(set(keys)) != len(keys):
        raise SystemExit("Telemetry canonical keys are missing or duplicated.")
    rank_counts = {rank: sum(int(item.get("rank", -1)) == rank for item in items) for rank in range(args.world_size)}
    missing_ranks = [rank for rank, count in rank_counts.items() if count == 0]
    if missing_ranks:
        raise SystemExit(f"Ranks received no evaluation work: {missing_ranks}.")
    print(json.dumps({
        "results_json": str(results_paths[0]),
        "expected_samples": args.expected_samples,
        "telemetry_samples": len(items),
        "unique_canonical_keys": len(set(keys)),
        "rank_counts": rank_counts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
