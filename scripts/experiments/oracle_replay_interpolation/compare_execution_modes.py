#!/usr/bin/env python3
"""Compare one-GPU and four-GPU VSI outputs by frozen canonical key."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_manifest(path: Path, expected_hash: str) -> set[str]:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f"Manifest hash mismatch: expected={expected_hash}, actual={actual}.")
    keys = [json.loads(line)["canonical_key"] for line in path.read_text(encoding="utf-8").splitlines()]
    if len(keys) != 5130 or len(set(keys)) != 5130:
        raise RuntimeError("Frozen expected-key manifest must contain exactly 5130 unique keys.")
    return set(keys)


def read_telemetry(root: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("rank_*/samples.jsonl")):
        items.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    by_key: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for item in items:
        key = item.get("canonical_key")
        if not isinstance(key, str) or not key:
            duplicates.append("<missing canonical_key>")
        elif key in by_key:
            duplicates.append(key)
        else:
            by_key[key] = item
    return by_key, duplicates, items


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        output: dict[str, float] = {}
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            output.update(flatten_numeric(item, child))
        return output
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def locate_sample_log(results_path: Path) -> Path:
    candidates = sorted(results_path.parent.glob("vsibench.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one vsibench.json beside {results_path}; found {len(candidates)}.")
    return candidates[0]


def read_scored_samples(results_path: Path) -> tuple[dict[int, dict[str, Any]], list[int]]:
    logs = load_json(locate_sample_log(results_path)).get("logs", [])
    by_doc_id: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for item in logs:
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, int) or doc_id in by_doc_id:
            duplicates.append(doc_id if isinstance(doc_id, int) else -1)
        else:
            by_doc_id[doc_id] = item
    return by_doc_id, duplicates


def score_payload(scored: dict[str, Any] | None) -> Any:
    if not scored:
        return None
    return scored.get("vsibench_score")


def category_scores(scored: dict[int, dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in scored.values():
        payload = score_payload(item)
        if not isinstance(payload, dict):
            continue
        score = next((float(v) for key, v in payload.items() if key.startswith("MRA:") and isinstance(v, (int, float))), None)
        if score is None:
            continue
        dataset = payload.get("dataset", "unknown")
        question_type = payload.get("question_type", "unknown")
        buckets[f"dataset={dataset}"] .append(score)
        buckets[f"question_type={question_type}"] .append(score)
    return {key: sum(values) / len(values) for key, values in sorted(buckets.items())}


def provenance(path: Path, mode: str, manifest_hash: str) -> dict[str, Any]:
    value = load_json(path)
    if value.get("execution_mode") != mode:
        raise RuntimeError(f"Unexpected execution_mode in {path}: {value.get('execution_mode')!r}.")
    if value.get("expected_key_manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Manifest provenance mismatch in {path}: {value.get('expected_key_manifest_sha256')!r}.")
    if not value.get("results_json_path") or value.get("wall_clock_seconds") is None:
        raise RuntimeError(f"Incomplete completion provenance in {path}.")
    return value


def compact_mismatch(key: str, single: dict[str, Any], four: dict[str, Any], single_score: dict[str, Any] | None, four_score: dict[str, Any] | None) -> dict[str, Any]:
    def one(item: dict[str, Any], scored: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "owning_rank": item.get("rank"),
            "decoded_answer": item.get("answer"),
            "generated_token_ids": item.get("generated_token_ids"),
            "generation_arguments": scored.get("arguments") if scored else None,
            "scoring_output": score_payload(scored),
        }
    return {"canonical_key": key, "single_gpu": one(single, single_score), "single_node_4gpu": one(four, four_score)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--single-telemetry", required=True)
    parser.add_argument("--four-telemetry", required=True)
    parser.add_argument("--single-results", required=True)
    parser.add_argument("--four-results", required=True)
    parser.add_argument("--single-provenance", required=True)
    parser.add_argument("--four-provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = read_manifest(Path(args.manifest), args.manifest_sha256)
    single, single_dupes, single_items = read_telemetry(Path(args.single_telemetry))
    four, four_dupes, four_items = read_telemetry(Path(args.four_telemetry))
    single_results_path, four_results_path = Path(args.single_results), Path(args.four_results)
    single_scored, single_scored_dupes = read_scored_samples(single_results_path)
    four_scored, four_scored_dupes = read_scored_samples(four_results_path)
    single_provenance = provenance(Path(args.single_provenance), "single_gpu", args.manifest_sha256)
    four_provenance = provenance(Path(args.four_provenance), "single_node_4gpu", args.manifest_sha256)

    common = sorted(expected & set(single) & set(four))
    answer_mismatches, token_mismatches, score_mismatches, mismatches = [], [], [], []
    for key in common:
        left, right = single[key], four[key]
        left_score = single_scored.get(left.get("doc_id"))
        right_score = four_scored.get(right.get("doc_id"))
        answer_bad = left.get("answer") != right.get("answer")
        tokens_available = left.get("generated_token_ids") is not None and right.get("generated_token_ids") is not None
        token_bad = tokens_available and left.get("generated_token_ids") != right.get("generated_token_ids")
        score_bad = score_payload(left_score) != score_payload(right_score)
        if answer_bad:
            answer_mismatches.append(key)
        if token_bad:
            token_mismatches.append(key)
        if score_bad:
            score_mismatches.append(key)
        if answer_bad or token_bad or score_bad:
            mismatches.append(compact_mismatch(key, left, right, left_score, right_score))

    single_metrics = flatten_numeric(load_json(single_results_path).get("results", {}))
    four_metrics = flatten_numeric(load_json(four_results_path).get("results", {}))
    metric_difference = {key: single_metrics[key] - four_metrics[key] for key in sorted(set(single_metrics) & set(four_metrics))}
    single_categories, four_categories = category_scores(single_scored), category_scores(four_scored)
    category_difference = {key: single_categories[key] - four_categories[key] for key in sorted(set(single_categories) & set(four_categories))}
    token_comparable = [key for key in common if single[key].get("generated_token_ids") is not None and four[key].get("generated_token_ids") is not None]
    rank_counts = Counter(str(item.get("rank")) for item in four_items)
    single_wall = float(single_provenance["wall_clock_seconds"])
    four_wall = float(four_provenance["wall_clock_seconds"])
    single_keys, four_keys = set(single), set(four)
    scoring_missing = {
        "single_gpu": sorted(key for key, item in single.items() if item.get("doc_id") not in single_scored),
        "single_node_4gpu": sorted(key for key, item in four.items() if item.get("doc_id") not in four_scored),
    }
    report = {
        "manifest_sha256": args.manifest_sha256,
        "expected_sample_count": len(expected),
        "single_gpu": {
            "total_output_count": len(single_items), "unique_sample_key_count": len(single),
            "duplicate_sample_count": len(single_dupes), "missing_sample_keys": sorted(expected - single_keys),
            "extra_sample_keys": sorted(single_keys - expected), "scored_log_duplicate_doc_ids": single_scored_dupes,
            "wall_clock_seconds": single_wall, "peak_gpu_memory_bytes_per_rank": {str(rank): max((int(item.get("peak_gpu_memory_allocated_bytes", 0)) for item in single_items if item.get("rank") == rank), default=0) for rank in sorted({item.get("rank") for item in single_items})},
            "effective_throughput_samples_per_second": len(single_items) / single_wall if single_wall else None,
            "provenance": single_provenance,
        },
        "single_node_4gpu": {
            "total_output_count": len(four_items), "unique_sample_key_count": len(four),
            "duplicate_sample_count": len(four_dupes), "missing_sample_keys": sorted(expected - four_keys),
            "extra_sample_keys": sorted(four_keys - expected), "rank_counts": dict(sorted(rank_counts.items())),
            "scored_log_duplicate_doc_ids": four_scored_dupes,
            "wall_clock_seconds": four_wall, "peak_gpu_memory_bytes_per_rank": {str(rank): max((int(item.get("peak_gpu_memory_allocated_bytes", 0)) for item in four_items if item.get("rank") == rank), default=0) for rank in sorted({item.get("rank") for item in four_items})},
            "effective_throughput_samples_per_second": len(four_items) / four_wall if four_wall else None,
            "provenance": four_provenance,
        },
        "speedup_one_gpu_over_four_gpu": single_wall / four_wall if four_wall else None,
        "exact_decoded_answer_match_rate": (len(common) - len(answer_mismatches)) / len(expected),
        "exact_generated_token_match_rate": ((len(token_comparable) - len(token_mismatches)) / len(token_comparable)) if token_comparable else None,
        "per_sample_score_agreement": {"matching": len(common) - len(score_mismatches), "compared": len(common), "match_rate": (len(common) - len(score_mismatches)) / len(expected)},
        "overall_score_differences": metric_difference,
        "per_category_score_differences": category_difference,
        "scoring_log_missing_sample_keys": scoring_missing,
        "first_mismatches": mismatches[:20],
        "possible_causes_if_mismatched": ["distributed aggregation/sharding defect", "rank-local output overwrite", "generation nondeterminism despite deterministic generation arguments"],
    }
    report["passed"] = not any((single_dupes, four_dupes, single_scored_dupes, four_scored_dupes, report["single_gpu"]["missing_sample_keys"], report["single_gpu"]["extra_sample_keys"], report["single_node_4gpu"]["missing_sample_keys"], report["single_node_4gpu"]["extra_sample_keys"], answer_mismatches, token_mismatches, score_mismatches, scoring_missing["single_gpu"], scoring_missing["single_node_4gpu"])) and len(single) == len(four) == len(expected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("One-GPU/four-GPU output parity failed; see report.")


if __name__ == "__main__":
    main()
