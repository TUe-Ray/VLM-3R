#!/usr/bin/env python3
"""Join paired native VSiBench logs for SpatialStack geometry utilization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


# These are a transparent grouping of existing VSiBench question_type values;
# individual native types remain the primary reporting unit.  The comparison
# group deliberately avoids claiming that VSiBench contains a truly
# non-spatial split: counting is weakly spatial and appearance order is
# temporal/non-relational.
DEFAULT_SPATIAL_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "object_abs_distance",
    "route_planning",
)
DEFAULT_COMPARISON_TYPES = ("obj_appearance_order", "object_counting")
SCORE_FIELDS = ("accuracy", "MRA:.5:.95:.05")


def parse_model_args(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in str(value).split(","):
        key, separator, argument = item.partition("=")
        if not separator:
            raise ValueError(f"Malformed model argument {item!r}")
        parsed[key] = argument
    return parsed


def run_label(path: Path, run_root: Path, models: list[str]) -> str:
    relative = path.relative_to(run_root)
    if not relative.parts:
        raise ValueError(f"Could not infer run label from {path}")
    label = relative.parts[0]
    for model in models:
        if label == model or label.startswith(model + "_"):
            return model
    raise ValueError(f"Run directory {label!r} does not begin with a requested model label {models}")


def score_from_doc(doc: dict[str, Any]) -> tuple[float, str]:
    present = [field for field in SCORE_FIELDS if field in doc]
    if len(present) != 1:
        raise ValueError(f"Expected one native VSiBench score in log document, found {present}: {doc}")
    return float(doc[present[0]]), present[0]


def resolved_adapter(run_args: dict[str, str]) -> str:
    runtime = Path(run_args["pretrained"])
    candidates = (runtime / "adapter_model.bin", runtime / "adapter_model.safetensors")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"No adapter weights found beneath logged runtime {runtime}")


def load_runs(run_root: Path, models: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    paths = sorted(run_root.glob("**/vsibench_local_mp4.json"))
    if not paths:
        raise FileNotFoundError(f"No VSiBench per-sample logs found under {run_root}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = parse_model_args(payload.get("args", {}).get("model_args", ""))
        condition = args.get("spatialstack_perturbation_mode", "none")
        if condition not in {"none", "normal", "geometry_off_all"}:
            raise ValueError(f"Unsupported logged perturbation condition {condition!r} in {path}")
        model = run_label(path, run_root, models)
        key = (model, condition)
        if key in found:
            raise RuntimeError(
                f"Found multiple logs for {model}/{condition}: {found[key]['path']} and {path}. "
                "Use a fresh RESULT_ROOT or remove stale run directories before analysis."
            )
        found[key] = {
            "path": path,
            "args": args,
            "adapter": resolved_adapter(args),
            "logs": payload.get("logs", []),
        }
    return found


def indexed_logs(run: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for item in run["logs"]:
        doc = item.get("vsibench_score") or item.get("doc")
        if not isinstance(doc, dict):
            raise ValueError(f"Missing scored VSiBench document in {run['path']}")
        key = (int(item["doc_id"]), str(item["prompt_hash"]))
        if key in indexed:
            raise ValueError(f"Duplicate VSiBench example key {key} in {run['path']}")
        indexed[key] = {"log": item, "doc": doc}
    return indexed


def validate_pair(normal: dict[str, Any], off: dict[str, Any]) -> list[dict[str, Any]]:
    ignored_args = {"pretrained", "spatialstack_perturbation_mode"}
    normal_args = {key: value for key, value in normal["args"].items() if key not in ignored_args}
    off_args = {key: value for key, value in off["args"].items() if key not in ignored_args}
    if normal_args != off_args:
        raise RuntimeError("Normal/off model arguments differ outside the perturbation mode.")
    if normal["adapter"] != off["adapter"]:
        raise RuntimeError(f"Normal/off checkpoints differ: {normal['adapter']} != {off['adapter']}")
    normal_rows = indexed_logs(normal)
    off_rows = indexed_logs(off)
    if set(normal_rows) != set(off_rows):
        raise RuntimeError("Normal/off evaluation manifests differ (doc ID/prompt-hash keys do not match).")
    pairs: list[dict[str, Any]] = []
    for key in sorted(normal_rows):
        first, second = normal_rows[key], off_rows[key]
        first_log, second_log = first["log"], second["log"]
        first_doc, second_doc = first["doc"], second["doc"]
        for field in ("doc_hash", "prompt_hash", "target_hash", "target"):
            if first_log.get(field) != second_log.get(field):
                raise RuntimeError(f"Paired inputs differ at {key}: log field {field!r} changed.")
        for field in ("id", "dataset", "scene_name", "question_type", "question", "ground_truth", "options"):
            if first_doc.get(field) != second_doc.get(field):
                raise RuntimeError(f"Paired inputs differ at {key}: document field {field!r} changed.")
        normal_score, normal_metric = score_from_doc(first_doc)
        off_score, off_metric = score_from_doc(second_doc)
        if normal_metric != off_metric:
            raise RuntimeError(f"Native score field changed for paired example {key}.")
        normal_correct = math.isclose(normal_score, 1.0, abs_tol=1e-12)
        off_correct = math.isclose(off_score, 1.0, abs_tol=1e-12)
        pairs.append(
            {
                "example_id": first_doc["id"],
                "doc_id": key[0],
                "dataset": first_doc["dataset"],
                "scene_name": first_doc["scene_name"],
                "category": first_doc["question_type"],
                "question": first_doc["question"],
                "ground_truth": first_doc["ground_truth"],
                "score_metric": normal_metric,
                "normal_prediction": first_doc.get("prediction", ""),
                "geometry_off_prediction": second_doc.get("prediction", ""),
                "normal_score": normal_score,
                "geometry_off_score": off_score,
                "score_difference": normal_score - off_score,
                "normal_outcome": "correct" if normal_correct else "incorrect",
                "geometry_off_outcome": "correct" if off_correct else "incorrect",
                "transition": (
                    ("correct" if normal_correct else "incorrect")
                    + "_to_"
                    + ("correct" if off_correct else "incorrect")
                ),
                "prompt_hash": key[1],
            }
        )
    return pairs


def mean(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_pairs(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    transitions: dict[tuple[str, str, str], int] = defaultdict(int)
    for pair in pairs:
        by_category[(pair["model"], pair["category"], pair["score_metric"])].append(pair)
        transitions[(pair["model"], pair["category"], pair["transition"])] += 1
    aggregates = []
    for (model, category, metric), values in sorted(by_category.items()):
        normal = mean([value["normal_score"] for value in values])
        off = mean([value["geometry_off_score"] for value in values])
        aggregates.append(
            {
                "model": model,
                "category": category,
                "score_metric": metric,
                "examples": len(values),
                "normal_score_mean": normal,
                "geometry_off_score_mean": off,
                "task_score_drop": normal - off,
            }
        )
    transition_rows = [
        {"model": model, "category": category, "transition": transition, "examples": count}
        for (model, category, transition), count in sorted(transitions.items())
    ]
    return aggregates, transition_rows


def grouped_summary(
    aggregates: list[dict[str, Any]], spatial_types: set[str], comparison_types: set[str]
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_model[row["model"]].append(row)
    rows = []
    for model, values in sorted(by_model.items()):
        spatial = [row["task_score_drop"] for row in values if row["category"] in spatial_types]
        comparison = [row["task_score_drop"] for row in values if row["category"] in comparison_types]
        rows.append(
            {
                "model": model,
                "spatial_category_macro_drop": mean(spatial),
                "spatial_categories_present": ";".join(sorted({row["category"] for row in values if row["category"] in spatial_types})),
                "comparison_category_macro_drop": mean(comparison),
                "comparison_categories_present": ";".join(sorted({row["category"] for row in values if row["category"] in comparison_types})),
                "selective_utilization": mean(spatial) - mean(comparison) if spatial and comparison else float("nan"),
                "comparison_interpretation": "temporal_or_weakly_spatial_not_a_true_nonspatial_control",
            }
        )
    return rows


def verify_baseline(runs: dict[tuple[str, str], dict[str, Any]], models: list[str], baseline: str) -> None:
    for model in models:
        none, normal = runs[(model, baseline)], runs[(model, "normal")]
        if none["adapter"] != normal["adapter"]:
            raise RuntimeError(f"{model}: unperturbed and explicit-normal checkpoints differ.")
        first, second = indexed_logs(none), indexed_logs(normal)
        if set(first) != set(second):
            raise RuntimeError(f"{model}: unperturbed and explicit-normal manifests differ.")
        for key in first:
            a, b = first[key], second[key]
            if a["log"].get("filtered_resps") != b["log"].get("filtered_resps"):
                raise RuntimeError(f"{model}: explicit normal changed prediction for smoke example {key}.")
            a_score, _ = score_from_doc(a["doc"])
            b_score, _ = score_from_doc(b["doc"])
            if not math.isclose(a_score, b_score, abs_tol=1e-12):
                raise RuntimeError(f"{model}: explicit normal changed native score for smoke example {key}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--baseline-condition", choices=("none",), default=None)
    parser.add_argument("--require-normal-baseline-match", action="store_true")
    parser.add_argument("--spatial-question-types", default=",".join(DEFAULT_SPATIAL_TYPES))
    parser.add_argument("--comparison-question-types", default=",".join(DEFAULT_COMPARISON_TYPES))
    args = parser.parse_args()
    if args.require_normal_baseline_match and args.baseline_condition is None:
        parser.error("--require-normal-baseline-match requires --baseline-condition none")
    spatial_types = {value for value in args.spatial_question_types.split(",") if value}
    comparison_types = {value for value in args.comparison_question_types.split(",") if value}
    if spatial_types & comparison_types:
        parser.error("spatial and comparison question-type groups must not overlap")

    runs = load_runs(args.run_root, args.models)
    for model in args.models:
        for condition in ("normal", "geometry_off_all"):
            if (model, condition) not in runs:
                raise FileNotFoundError(f"Missing {condition} log for {model} under {args.run_root}")
    if args.require_normal_baseline_match:
        for model in args.models:
            if (model, args.baseline_condition) not in runs:
                raise FileNotFoundError(f"Missing {args.baseline_condition} log for {model}")
        verify_baseline(runs, args.models, args.baseline_condition)

    pairs = []
    for model in args.models:
        for row in validate_pair(runs[(model, "normal")], runs[(model, "geometry_off_all")]):
            row["model"] = model
            pairs.append(row)
    aggregates, transitions = aggregate_pairs(pairs)
    selective = grouped_summary(aggregates, spatial_types, comparison_types)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_example_paired.csv", pairs)
    (args.output_dir / "per_example_paired.json").write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "aggregate_by_model_category.csv", aggregates)
    write_csv(args.output_dir / "paired_transitions.csv", transitions)
    write_csv(args.output_dir / "selective_utilization.csv", selective)
    summary = [
        "# VSiBench SpatialStack task-conditioned geometry utilization",
        "",
        "Scores are VSiBench's native per-example `accuracy` or `MRA:.5:.95:.05`; no correctness metric was replaced.",
        "`task_score_drop = normal_score - geometry_off_score`.",
        "",
        "The category table is primary. The optional selective summary uses existing `question_type` values:",
        f"- relational-spatial: {', '.join(sorted(spatial_types))}",
        f"- temporal/weak-spatial comparison: {', '.join(sorted(comparison_types))}",
        "",
        "VSiBench does not expose a clearly non-spatial category. The comparison group is therefore not a non-spatial control; size and room-size estimation remain category-only rather than being forced into either group.",
        "",
        "| Model | Spatial macro drop | Comparison macro drop | Selective utilization |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in selective:
        summary.append(
            f"| {row['model']} | {row['spatial_category_macro_drop']:.6f} | "
            f"{row['comparison_category_macro_drop']:.6f} | {row['selective_utilization']:.6f} |"
        )
    (args.output_dir / "interpretation.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[DONE] paired examples: {len(pairs)}")
    print(f"[DONE] outputs: {args.output_dir}")
    if args.require_normal_baseline_match:
        print("[DONE] explicit normal exactly matched the unperturbed smoke outputs and scores.")


if __name__ == "__main__":
    main()
