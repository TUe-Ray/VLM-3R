#!/usr/bin/env python
"""Aggregate normal, geometry-off, and geometry-delta depth-probe results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("SS123", "SS012_new", "SS036", "SS012_old")
DEFAULT_LAYERS = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)
VARIANTS = (("normal", ""), ("geometry_off", "__geometry_off"), ("geometry_delta", "__geometry_delta"))


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rms_lookup(cache_root: Path, model: str) -> dict[tuple[int, str, str], float]:
    path = cache_root / "geometry_induced_probe" / model / "feature_rms.json"
    rows = load_json(path)
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), str(row["feature_type"]), str(row["split"]))].append(float(row["rms"]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model", "layer", "feature_type", "probe_metric", "probe_metric_value", "mae", "absrel", "delta125",
        "seed", "best_epoch", "num_tokens", "feature_rms_train", "feature_rms_val", "feature_rms_all", "delta_rms_val",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rank_text(rows: list[dict[str, Any]], layer: int) -> str:
    delta_rows = [row for row in rows if row["layer"] == layer and row["feature_type"] == "geometry_delta"]
    ordered = sorted(delta_rows, key=lambda row: float(row["mae"]))
    return " < ".join(f"{row['model']} ({row['mae']:.4f})" for row in ordered)


def write_summary(path: Path, rows: list[dict[str, Any]], prior_summary: str | None) -> None:
    by_key = {(row["model"], row["layer"], row["feature_type"]): row for row in rows}
    models = list(dict.fromkeys(row["model"] for row in rows))
    layers = sorted({int(row["layer"]) for row in rows})
    lines = [
        "# Geometry-Induced Hidden Direction Depth Probe",
        "",
        "The probe metric is MAE (lower is better). RMS is shown beside it rather than divided into MAE, because MAE is an error metric.",
        "",
        "`geometry_delta = normal - all-geometry-off`, paired within the same model forward input, video, selected frame, visual-token order, and requested layer.",
        "",
        "## Layer-wise geometry-delta decodability",
        "",
        "| Layer | " + " | ".join(models) + " |",
        "|---:|" + "|".join("---:" for _ in models) + "|",
    ]
    for layer in layers:
        cells = []
        for model in models:
            row = by_key[(model, layer, "geometry_delta")]
            cells.append(f"{row['mae']:.6f} (RMS {row['feature_rms_val']:.4f})")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Delta-MAE ranking by layer", ""])
    for layer in layers:
        lines.append(f"- L{layer}: {rank_text(rows, layer)}")
    lines.extend(["", "## Baseline comparison", ""])
    lines.extend([
        "The CSV includes normal, all-geometry-off, and geometry-delta probe scores for exactly the same canonical train/validation samples and layers.",
        "Compare separation in the delta-MAE rankings above with the closely grouped perturbation magnitudes from the prior development experiment; this report does not claim a magnitude-normalized score.",
    ])
    if prior_summary:
        lines.append(f"Prior perturbation-magnitude summary: `{prior_summary}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_delta_mae(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["feature_type"] == "geometry_delta":
            grouped[str(row["model"])].append(row)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for model, model_rows in grouped.items():
        model_rows.sort(key=lambda row: int(row["layer"]))
        axis.plot([row["layer"] for row in model_rows], [row["mae"] for row in model_rows], marker="o", label=model)
    axis.set(xlabel="LLM layer L (hidden_states[L + 1])", ylabel="Depth-probe MAE (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument("--prior-magnitude-summary", default=None)
    parser.add_argument("--allow-partial", action="store_true", help="Smoke-only: write available rows instead of requiring all variants.")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    models, layers = parse_csv(args.models), parse_layers(args.layers)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model in models:
        rms = rms_lookup(cache_root, model)
        for layer in layers:
            for feature_type, suffix in VARIANTS:
                label = f"{model}{suffix}"
                metrics_path = cache_root / "probes" / label / f"layer_{layer}" / "metrics.json"
                if not metrics_path.is_file():
                    missing.append(str(metrics_path))
                    continue
                metrics = load_json(metrics_path)
                feature_rms_val = rms.get((layer, feature_type, "val"))
                rows.append({
                    "model": model,
                    "layer": layer,
                    "feature_type": feature_type,
                    "probe_metric": "mae_lower_is_better",
                    "probe_metric_value": metrics["mae"],
                    "mae": metrics["mae"],
                    "absrel": metrics["absrel"],
                    "delta125": metrics["delta125"],
                    "seed": metrics.get("probe_seed"),
                    "best_epoch": metrics.get("best_epoch"),
                    "num_tokens": metrics["num_tokens"],
                    "feature_rms_train": rms.get((layer, feature_type, "train")),
                    "feature_rms_val": feature_rms_val,
                    "feature_rms_all": mean([
                        value for split in ("train", "val") if (value := rms.get((layer, feature_type, split))) is not None
                    ]),
                    "delta_rms_val": feature_rms_val if feature_type == "geometry_delta" else rms.get((layer, "geometry_delta", "val")),
                })
    if missing and not args.allow_partial:
        raise FileNotFoundError("Missing required probe metrics:\n" + "\n".join(missing))
    if not rows:
        raise RuntimeError("No probe metrics found")
    rows.sort(key=lambda row: (str(row["model"]), int(row["layer"]), str(row["feature_type"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "geometry_induced_depth_probe.csv", rows)
    if not missing:
        write_summary(output_dir / "summary.md", rows, args.prior_magnitude_summary)
        plotted = plot_delta_mae(output_dir / "geometry_delta_depth_probe_mae.png", rows)
    else:
        plotted = False
    (output_dir / "diagnostics.json").write_text(json.dumps({"rows": len(rows), "missing": missing, "plot_written": plotted}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output_dir": str(output_dir), "plot_written": plotted}))


if __name__ == "__main__":
    main()
