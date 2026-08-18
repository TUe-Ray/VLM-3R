#!/usr/bin/env python
"""Summarize saved pre-SFT fusion probe metrics without loading a model or GPU."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from depth_probe_common import read_json, write_csv, write_json


METRICS = ("mae", "absrel", "delta125")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part) for part in parse_csv(value)]
    if not seeds:
        raise ValueError("At least one fusion seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Fusion seeds must be unique, got {seeds}")
    return seeds


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "probes").glob("*/*/metrics.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected metrics object at {path}, got {type(payload)}")
        payload = dict(payload)
        payload["metrics_path"] = str(path)
        rows.append(payload)
    return rows


def summarize(
    rows: list[dict[str, Any]], *, variants: list[str], seeds: list[int], probe_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in rows if row.get("experiment_variant") in variants]
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    for row in selected:
        variant = str(row["experiment_variant"])
        feature = str(row.get("feature_level"))
        seed = row.get("fusion_init_seed")
        if not isinstance(seed, int):
            issues.append({"kind": "missing_fusion_seed", "metrics_path": row["metrics_path"]})
            continue
        if row.get("probe_seed") != probe_seed:
            issues.append(
                {
                    "kind": "wrong_probe_seed",
                    "metrics_path": row["metrics_path"],
                    "expected": probe_seed,
                    "actual": row.get("probe_seed"),
                }
            )
            continue
        key = (variant, feature)
        existing = grouped.setdefault(key, {}).get(seed)
        if existing is not None:
            issues.append(
                {
                    "kind": "duplicate_seed_result",
                    "variant": variant,
                    "feature_level": feature,
                    "fusion_init_seed": seed,
                    "paths": [existing["metrics_path"], row["metrics_path"]],
                }
            )
            continue
        grouped.setdefault(key, {})[seed] = row

    summaries = []
    for variant in variants:
        features = sorted(feature for current_variant, feature in grouped if current_variant == variant)
        for feature in features:
            by_seed = grouped[(variant, feature)]
            found_seeds = sorted(by_seed)
            missing = [seed for seed in seeds if seed not in by_seed]
            values = {metric: [float(by_seed[seed][metric]) for seed in found_seeds] for metric in METRICS}
            summary: dict[str, Any] = {
                "experiment_variant": variant,
                "feature_level": feature,
                "probe_seed": probe_seed,
                "expected_fusion_seeds": seeds,
                "fusion_seeds": found_seeds,
                "missing_fusion_seeds": missing,
                "n": len(found_seeds),
                "diagnostic_only": feature in {"fusion_output", "projected_features"},
                "seed_scores": [
                    {"fusion_init_seed": seed, **{metric: float(by_seed[seed][metric]) for metric in METRICS}}
                    for seed in found_seeds
                ],
            }
            for metric, metric_values in values.items():
                if metric_values:
                    summary[f"{metric}_mean"] = statistics.fmean(metric_values)
                    summary[f"{metric}_std"] = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
                    summary[f"{metric}_min"] = min(metric_values)
                    summary[f"{metric}_max"] = max(metric_values)
            summaries.append(summary)
    for variant in variants:
        if not any(row["experiment_variant"] == variant for row in summaries):
            issues.append({"kind": "missing_variant", "experiment_variant": variant})
    summaries.sort(key=lambda row: (variants.index(row["experiment_variant"]), row["feature_level"]))
    return summaries, issues


def write_markdown(path: Path, rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> None:
    lines = [
        "# Pre-SFT Fusion Multi-seed Probe Summary",
        "",
        "| Variant | Feature | Seeds | MAE mean ± std | AbsRel mean ± std | δ<1.25 mean ± std |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        suffix = " (diagnostic)" if row["diagnostic_only"] else ""
        lines.append(
            f"| {row['experiment_variant']} | {row['feature_level']}{suffix} | "
            f"{','.join(str(seed) for seed in row['fusion_seeds'])} | "
            f"{row.get('mae_mean', float('nan')):.6f} ± {row.get('mae_std', float('nan')):.6f} | "
            f"{row.get('absrel_mean', float('nan')):.6f} ± {row.get('absrel_std', float('nan')):.6f} | "
            f"{row.get('delta125_mean', float('nan')):.6f} ± {row.get('delta125_std', float('nan')):.6f} |"
        )
    if issues:
        lines.extend(["", "## Missing or invalid inputs", ""])
        lines.extend(f"- `{issue['kind']}`: {issue}" for issue in issues)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Durable experiment root containing probes/.")
    parser.add_argument("--variants", default="ss_identity,vlm3r_native")
    parser.add_argument("--fusion-seeds", default="0,1")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--stem", default="pre_sft_fusion_multiseed")
    args = parser.parse_args()

    root = Path(args.output_root)
    variants = parse_csv(args.variants)
    seeds = parse_seeds(args.fusion_seeds)
    summaries, issues = summarize(load_rows(root), variants=variants, seeds=seeds, probe_seed=args.probe_seed)
    write_json(root / f"{args.stem}_summary.json", summaries)
    write_csv(root / f"{args.stem}_summary.csv", summaries)
    write_json(root / f"{args.stem}_issues.json", issues)
    write_markdown(root / f"{args.stem}_summary.md", summaries, issues)
    print(f"[INFO] Wrote {len(summaries)} summaries; issues={len(issues)}")


if __name__ == "__main__":
    main()
