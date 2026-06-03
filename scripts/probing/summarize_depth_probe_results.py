#!/usr/bin/env python
"""Summarize VLM-3R depth probe metrics into matrices and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from depth_probe_common import DEFAULT_OUTPUT_ROOT, LLM_LAYERS, MODEL_PRESETS, PRE_LLM_FEATURES, read_json, write_csv, write_json

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


MODEL_ORDER = ["zero_spatial", "vlm3r_baseline", "geo_rope_fusion"]
FEATURE_ORDER = PRE_LLM_FEATURES + [f"layer_{layer}" for layer in LLM_LAYERS]


def load_metric_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "probes").glob("*/*/metrics.json")):
        rows.append(read_json(path))
    return rows


def metric_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "N/A"
    return f"{row['mae']:.4f} / {row['absrel']:.4f} / {row['delta125']:.4f}"


def write_matrix(output_root: Path, rows: list[dict[str, Any]]) -> None:
    by_key = {(row["model_label"], row["feature_level"]): row for row in rows}
    lines = [
        "| Feature Level | zero_spatial | vlm3r_baseline | geo_rope_fusion |",
        "|---|---|---|---|",
    ]
    labels = {
        "fusion_output": "fusion_output",
        "projected_features": "projected_features",
        **{f"layer_{layer}": f"LLM layer {layer}" for layer in LLM_LAYERS},
    }
    for feature in FEATURE_ORDER:
        cells = [labels.get(feature, feature)]
        for model in MODEL_ORDER:
            cells.append(metric_cell(by_key.get((model, feature))))
        lines.append("| " + " | ".join(cells) + " |")
    matrix_path = output_root / "probes" / "probe_accuracy_matrix.md"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def feature_sort_value(feature: str) -> int:
    if feature == "fusion_output":
        return -2
    if feature == "projected_features":
        return -1
    if feature.startswith("layer_"):
        return int(feature.split("_", 1)[1])
    return 10**9


def save_plot(output_root: Path, rows: list[dict[str, Any]], metric: str, ylabel: str, filename: str) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping plots")
        return
    plot_dir = output_root / "probes" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered_features = [feature for feature in FEATURE_ORDER if any(row["feature_level"] == feature for row in rows)]
    x = list(range(len(ordered_features)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODEL_ORDER:
        ys = []
        for feature in ordered_features:
            row = next((r for r in rows if r["model_label"] == model and r["feature_level"] == feature), None)
            ys.append(float(row[metric]) if row is not None else float("nan"))
        ax.plot(x, ys, marker="o", label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["fusion" if f == "fusion_output" else "projected" if f == "projected_features" else f.replace("layer_", "L") for f in ordered_features],
        rotation=30,
        ha="right",
    )
    ax.set_xlabel("Feature level")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = load_metric_rows(output_root)
    rows.sort(key=lambda row: (MODEL_ORDER.index(row["model_label"]) if row["model_label"] in MODEL_ORDER else 999, feature_sort_value(row["feature_level"])))
    write_json(output_root / "probes" / "results.json", rows)
    write_csv(output_root / "probes" / "results.csv", rows)
    write_matrix(output_root, rows)
    save_plot(output_root, rows, "mae", "MAE (m)", "depth_probe_mae.png")
    save_plot(output_root, rows, "delta125", "delta < 1.25", "depth_probe_delta125.png")
    print(f"[INFO] Wrote summary under {output_root / 'probes'}")


if __name__ == "__main__":
    main()
