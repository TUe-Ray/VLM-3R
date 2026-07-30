#!/usr/bin/env python3
"""Compare one-GPU and four-GPU VSI outputs by frozen canonical key."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_manifest(path: Path, expected_hash: str):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f"Manifest hash mismatch: expected={expected_hash}, actual={actual}.")
    keys = [json.loads(line)["canonical_key"] for line in path.read_text(encoding="utf-8").splitlines()]
    if len(keys) != 5130 or len(set(keys)) != 5130:
        raise RuntimeError("Frozen expected-key manifest must contain exactly 5130 unique keys.")
    return set(keys)


def read_telemetry(root: Path):
    items = []
    for path in sorted(root.glob("rank_*/samples.jsonl")):
        items.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    by_key = {}
    duplicates = []
    for item in items:
        key = item.get("canonical_key")
        if key in by_key:
            duplicates.append(key)
        else:
            by_key[key] = item
    return by_key, duplicates, items


def load_results(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_metrics(results):
    values = results.get("results", {})
    return {str(key): value for key, value in values.items() if isinstance(value, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--single-telemetry", required=True)
    parser.add_argument("--four-telemetry", required=True)
    parser.add_argument("--single-results", required=True)
    parser.add_argument("--four-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = read_manifest(Path(args.manifest), args.manifest_sha256)
    single, single_dupes, single_items = read_telemetry(Path(args.single_telemetry))
    four, four_dupes, four_items = read_telemetry(Path(args.four_telemetry))
    mismatches = []
    for key in sorted(expected & set(single) & set(four)):
        left, right = single[key], four[key]
        if left.get("answer") != right.get("answer") or left.get("generated_token_ids") != right.get("generated_token_ids"):
            mismatches.append({"canonical_key": key, "single": left, "four": right})
    left_metrics, right_metrics = numeric_metrics(load_results(Path(args.single_results))), numeric_metrics(load_results(Path(args.four_results)))
    metric_difference = {key: float(left_metrics[key]) - float(right_metrics[key]) for key in sorted(set(left_metrics) & set(right_metrics))}
    report = {
        "manifest_sha256": args.manifest_sha256, "expected_sample_count": len(expected),
        "single_gpu": {"total_output_count": len(single_items), "unique_sample_key_count": len(single), "duplicate_sample_count": len(single_dupes), "missing_sample_keys": sorted(expected - set(single)), "extra_sample_keys": sorted(set(single) - expected)},
        "single_node_4gpu": {"total_output_count": len(four_items), "unique_sample_key_count": len(four), "duplicate_sample_count": len(four_dupes), "missing_sample_keys": sorted(expected - set(four)), "extra_sample_keys": sorted(set(four) - expected), "rank_counts": {str(rank): sum(1 for item in four_items if item.get("rank") == rank) for rank in range(4)}},
        "exact_decoded_answer_match_rate": (len(expected & set(single) & set(four)) - len(mismatches)) / len(expected),
        "exact_generated_token_match_rate": (len(expected & set(single) & set(four)) - len(mismatches)) / len(expected),
        "first_mismatches": mismatches[:20], "overall_and_category_score_differences": metric_difference,
    }
    report["passed"] = not single_dupes and not four_dupes and not report["single_gpu"]["missing_sample_keys"] and not report["single_node_4gpu"]["missing_sample_keys"] and not mismatches
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("One-GPU/four-GPU output parity failed; see report.")


if __name__ == "__main__":
    main()
