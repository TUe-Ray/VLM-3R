#!/usr/bin/env python
"""Combine historical ScanNet depth metrics with local parity and missing-layer runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FEATURES = [
    "siglip_output",
    "fusion_output",
    "projected_features",
    "layer_0",
    "layer_1",
    "layer_2",
    "layer_3",
    "layer_6",
    "layer_9",
    "layer_12",
    "layer_15",
    "layer_18",
    "layer_21",
    "layer_24",
    "layer_27",
]
MISSING = {"layer_1", "layer_2", "layer_12", "layer_18", "layer_24"}
NEW_PRE_LLM = {"siglip_output", "projected_features"}
MODELS = ("vlm3r_baseline", "zero_spatial")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_map(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list of metrics at {path}")
    return {(str(row["model_label"]), str(row["feature_level"])): dict(row) for row in payload}


def local_metric(durable_root: Path, model: str, feature: str) -> dict[str, Any] | None:
    path = durable_root / "probes" / model / feature / "metrics.json"
    return dict(read_json(path)) if path.is_file() else None


def row_for(
    historical: dict[tuple[str, str], dict[str, Any]],
    durable_root: Path,
    model: str,
    feature: str,
) -> dict[str, Any] | None:
    old = historical.get((model, feature))
    new = local_metric(durable_root, model, feature)
    if model == "vlm3r_baseline" and feature == "layer_6" and new is not None:
        return {**new, "result_status": "new parity control", "historical_metrics": old}
    if model == "zero_spatial" and feature in NEW_PRE_LLM and new is not None:
        return {**new, "result_status": "new pre-LLM result", "historical_metrics": old}
    if feature in MISSING and new is not None:
        return {**new, "result_status": "new missing-layer result", "historical_metrics": old}
    if old is not None:
        return {**old, "result_status": "historical reused"}
    return None


def metric_text(row: dict[str, Any] | None) -> str:
    if row is None:
        return "not available"
    return f"{row['mae']:.6f} / {row['absrel']:.6f} / {row['delta125']:.6f} ({row['result_status']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-root", required=True)
    parser.add_argument("--historical-baseline", required=True)
    parser.add_argument("--historical-zero", default=None)
    args = parser.parse_args()

    durable_root = Path(args.durable_root)
    historical = metric_map(Path(args.historical_baseline))
    historical.update(metric_map(Path(args.historical_zero) if args.historical_zero else None))
    rows = [
        row
        for model in MODELS
        for feature in FEATURES
        if (row := row_for(historical, durable_root, model, feature)) is not None
    ]
    summary_root = durable_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row if key != "historical_metrics"})
    with (summary_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["model_label"], row["feature_level"]): row for row in rows}
    lines = [
        "# ScanNet Depth-Probe Layer Completion",
        "",
        "Metrics are MAE / AbsRel / δ<1.25.",
        "",
        "| Representation | Baseline | Zero Spatial |",
        "|---|---|---|",
    ]
    for feature in FEATURES:
        label = feature.replace("layer_", "L") if feature.startswith("layer_") else feature
        lines.append(
            f"| {label} | {metric_text(by_key.get(('vlm3r_baseline', feature)))} | "
            f"{metric_text(by_key.get(('zero_spatial', feature)))} |"
        )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Model | " + " | ".join(feature.replace("layer_", "L") for feature in FEATURES) + " |")
    lines.append("|---|" + "|".join("---" for _ in FEATURES) + "|")
    for model in MODELS:
        cells = []
        for feature in FEATURES:
            row = by_key.get((model, feature))
            cells.append(row["result_status"] if row else "not available")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    (summary_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    graph_data = {
        "feature_order": FEATURES,
        "metric_order": ["mae", "absrel", "delta125"],
        "models": list(MODELS),
        "rows": rows,
    }
    (summary_root / "graph_data.json").write_text(json.dumps(graph_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote ScanNet completion summary under {summary_root}")


if __name__ == "__main__":
    main()
