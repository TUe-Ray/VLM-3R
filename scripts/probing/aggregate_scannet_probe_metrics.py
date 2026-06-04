#!/usr/bin/env python
"""Aggregate per-level ScanNet probe metrics written by Slurm array jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from depth_probe_common import DEFAULT_OUTPUT_ROOT, LLM_LAYERS, PRE_LLM_FEATURES, read_json, write_csv, write_json


FEATURE_ORDER = PRE_LLM_FEATURES + [f"layer_{layer}" for layer in LLM_LAYERS]


def feature_sort_value(feature: str) -> int:
    if feature in FEATURE_ORDER:
        return FEATURE_ORDER.index(feature)
    return 10_000


def load_task_keys(task_file: Path) -> list[tuple[str, str]]:
    keys = []
    for line in task_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Expected '<model_label> <feature_level>' in {task_file}, got {line!r}")
        keys.append((parts[0], parts[1]))
    return keys


def load_rows(output_root: Path, probe_subdir: str, keys: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = []
    missing = []
    for model_label, feature_level in keys:
        path = output_root / probe_subdir / model_label / feature_level / "metrics.json"
        if not path.exists():
            missing.append({"model_label": model_label, "feature_level": feature_level, "metrics_path": str(path)})
            continue
        rows.append(read_json(path))
    rows.sort(key=lambda row: (row["model_label"], feature_sort_value(row["feature_level"])))
    return rows, missing


def write_depth_summary(output_root: Path, stem: str, rows: list[dict[str, Any]], missing: list[dict[str, str]]) -> None:
    lines = [
        f"# {stem} Results",
        "",
        "| Model | Feature | MAE | AbsRel | Delta < 1.25 | Best Epoch | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['feature_level']} | {row['mae']:.6f} | "
            f"{row['absrel']:.6f} | {row['delta125']:.6f} | {row['best_epoch']} | {row['num_tokens']} |"
        )
    if missing:
        lines.extend(["", "## Missing", ""])
        for row in missing:
            lines.append(f"- {row['model_label']}/{row['feature_level']}")
    (output_root / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_semantic_summary(output_root: Path, stem: str, rows: list[dict[str, Any]], missing: list[dict[str, str]]) -> None:
    lines = [
        f"# {stem} Results",
        "",
        "| Model | Feature | Top1 | mIoU GT-Present | GT-Present Classes | Dominant GT Fraction | Low Confidence |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['feature_level']} | "
            f"{row['top1_accuracy']:.4f} | {row['mIoU_gt_present']:.4f} | "
            f"{row['num_gt_present_classes']} | {row['dominant_class_fraction']:.4f} | "
            f"{row['low_confidence']} |"
        )
    if missing:
        lines.extend(["", "## Missing", ""])
        for row in missing:
            lines.append(f"- {row['model_label']}/{row['feature_level']}")
    (output_root / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--probe-subdir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--kind", choices=("depth", "semantic"), required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    keys = load_task_keys(Path(args.task_file))
    rows, missing = load_rows(output_root, args.probe_subdir, keys)
    write_json(output_root / f"{args.stem}_results.json", rows)
    write_csv(output_root / f"{args.stem}_results.csv", rows)
    write_json(output_root / f"{args.stem}_missing.json", missing)
    if args.kind == "depth":
        write_depth_summary(output_root, args.stem, rows, missing)
    else:
        write_semantic_summary(output_root, args.stem, rows, missing)
    print(f"[INFO] Aggregated {len(rows)} rows; missing {len(missing)}")


if __name__ == "__main__":
    main()
