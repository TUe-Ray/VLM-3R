#!/usr/bin/env python
"""Materialize cached LLM layer dicts into one tensor file per layer.

The depth extraction job stores all probed LLM layers for a frame in
features/<model_label>/llm_layers/frame_<id>.pt. This helper writes split files
such as features/<model_label>/layer_6/frame_<id>.pt so single-layer probe jobs
do not need to load the full layer dict on every sample.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from depth_probe_common import DEFAULT_OUTPUT_ROOT, LLM_LAYERS


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def discover_model_labels(output_root: Path) -> list[str]:
    features_root = output_root / "features"
    if not features_root.exists():
        return []
    labels = []
    for child in sorted(features_root.iterdir()):
        if child.is_dir() and (child / "llm_layers").is_dir():
            labels.append(child.name)
    return labels


def atomic_torch_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(tensor, tmp_path)
    tmp_path.replace(path)


def materialize_model_layers(
    *,
    output_root: Path,
    model_label: str,
    feature_levels: list[str],
    overwrite: bool,
    dry_run: bool,
    limit: int | None,
) -> dict[str, int]:
    source_dir = output_root / "features" / model_label / "llm_layers"
    if not source_dir.is_dir():
        print(f"[WARN] Skipping {model_label}: missing {source_dir}", flush=True)
        return {"frames": 0, "written": 0, "skipped_existing": 0, "missing_layers": 0}

    frame_paths = sorted(source_dir.glob("frame_*.pt"))
    if limit is not None:
        frame_paths = frame_paths[:limit]

    stats = {"frames": 0, "written": 0, "skipped_existing": 0, "missing_layers": 0}
    for frame_path in frame_paths:
        stats["frames"] += 1
        if dry_run:
            for feature_level in feature_levels:
                target = output_root / "features" / model_label / feature_level / frame_path.name
                if target.exists() and not overwrite:
                    stats["skipped_existing"] += 1
                else:
                    stats["written"] += 1
            continue

        payload = torch.load(frame_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected layer dict at {frame_path}, got {type(payload)}")
        for feature_level in feature_levels:
            target = output_root / "features" / model_label / feature_level / frame_path.name
            if target.exists() and not overwrite:
                stats["skipped_existing"] += 1
                continue
            if feature_level not in payload:
                stats["missing_layers"] += 1
                continue
            atomic_torch_save(payload[feature_level], target)
            stats["written"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model-labels",
        default=None,
        help="Comma-separated model labels. Default: all feature dirs with llm_layers.",
    )
    parser.add_argument(
        "--feature-levels",
        default=",".join(f"layer_{layer}" for layer in LLM_LAYERS),
        help="Comma-separated layers to materialize, e.g. layer_6,layer_9.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split layer files.")
    parser.add_argument("--dry-run", action="store_true", help="Count files without loading or writing tensors.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N frame files per model.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    model_labels = parse_csv(args.model_labels) or discover_model_labels(output_root)
    feature_levels = parse_csv(args.feature_levels)
    if not model_labels:
        raise FileNotFoundError(f"No model feature dirs with llm_layers found under {output_root / 'features'}")
    if not feature_levels:
        raise ValueError("No feature levels requested.")

    totals = {"frames": 0, "written": 0, "skipped_existing": 0, "missing_layers": 0}
    print(
        f"[INFO] output_root={output_root} models={model_labels} layers={feature_levels} "
        f"dry_run={args.dry_run} overwrite={args.overwrite}",
        flush=True,
    )
    for model_label in model_labels:
        stats = materialize_model_layers(
            output_root=output_root,
            model_label=model_label,
            feature_levels=feature_levels,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        for key, value in stats.items():
            totals[key] += value
        print(f"[INFO] {model_label}: {stats}", flush=True)
    print(f"[INFO] total: {totals}", flush=True)


if __name__ == "__main__":
    main()
