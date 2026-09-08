#!/usr/bin/env python
"""Aggregate paired, residual-masked SpatialStack geometry perturbations.

This is intentionally cache-only: extraction writes one compact JSON per
video, and this program never loads a checkpoint or runs the VLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "hidden_absolute_change_rms",
    "hidden_relative_change",
    "cosine_similarity",
    "source_raw_delta_rms",
    "normalized_propagation",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(values: list[Any]) -> tuple[float | None, float | None, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None, None, None
    array = np.asarray(finite, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0)), float(np.median(array))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Extractor output root containing geometry_perturbation/.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="*", default=None, help="Optional exact model labels to include.")
    args = parser.parse_args()

    root = Path(args.input_root) / "geometry_perturbation"
    requested = set(args.models or [])
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json")):
        model = path.parent.name
        if requested and model not in requested:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise ValueError(f"Invalid perturbation artifact: {path}")
        for row in payload["rows"]:
            if not isinstance(row, dict):
                raise TypeError(f"Invalid result row in {path}")
            rows.append({"model": model, **row, "artifact_path": str(path)})
        detail = payload.get("diagnostics", {})
        if isinstance(detail, dict):
            diagnostics.append({"model": model, "artifact_path": str(path), **detail})
    if not rows:
        raise RuntimeError(f"No geometry perturbation artifacts found under {root}")

    rows.sort(key=lambda row: (
        str(row["model"]), str(row.get("video_id", "")), str(row.get("perturbation", "")),
        -1 if row.get("source_injection_layer") is None else int(row["source_injection_layer"]),
        int(row.get("measured_layer", -1)),
    ))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_video.csv", rows)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("model", "fusion_type", "configured_injection_layers", "perturbation", "source_injection_layer", "measured_layer", "probe_point")
    for row in rows:
        groups[tuple(str(row.get(key)) if key == "configured_injection_layers" else row.get(key) for key in keys)].append(row)
    aggregate: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        aggregate_row = dict(zip(keys, group_key))
        aggregate_row["num_videos"] = len(group_rows)
        for metric in METRICS:
            mean, std, median = numeric_summary([row.get(metric) for row in group_rows])
            aggregate_row[f"{metric}_mean"] = mean
            aggregate_row[f"{metric}_std"] = std
            aggregate_row[f"{metric}_median"] = median
        aggregate.append(aggregate_row)
    write_csv(output_dir / "aggregate.csv", aggregate)
    (output_dir / "diagnostics.json").write_text(
        json.dumps({"schema_version": "spatialstack_geometry_perturbation_aggregate_v1", "runs": diagnostics}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "spatialstack_geometry_perturbation_aggregate_v1",
                "input_root": str(Path(args.input_root).resolve()),
                "models": sorted({str(row["model"]) for row in rows}),
                "num_rows": len(rows),
                "num_artifacts": len(diagnostics),
                "per_video_csv": str((output_dir / "per_video.csv").resolve()),
                "aggregate_csv": str((output_dir / "aggregate.csv").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
