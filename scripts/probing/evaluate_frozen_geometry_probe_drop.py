#!/usr/bin/env python
"""Evaluate normal-trained depth probes on paired residual-mask geometry-OFF features.

No optimisation is performed here.  The probe checkpoint is hashed before and
after evaluation and the paired feature cache is required to carry the exact
selected frames in the intervention manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from depth_probe_common import load_frame_records, metric_values
from train_depth_probes import DepthProbeMLP


def safe_video_id(record: dict[str, Any]) -> str:
    value = str(record.get("video_sample_id", record.get("video_path", "")))
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"model", "probe_dir", "layer"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Task TSV must contain {sorted(required)}: {path}")
    return rows


def load_probe(task: dict[str, str], device: torch.device) -> tuple[DepthProbeMLP, dict[str, Any], str]:
    directory = Path(task["probe_dir"])
    checkpoint_path = directory / "best.pt"
    metrics_path = directory / "metrics.json"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing frozen probe artifacts in {directory}")
    checkpoint = torch_load(checkpoint_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_level = f"layer_{int(task['layer'])}"
    if checkpoint.get("feature_level") != expected_level or metrics.get("feature_level") != expected_level:
        raise RuntimeError(f"Probe layer mismatch for {directory}: expected {expected_level}")
    d_in = int(checkpoint["d_in"])
    model = DepthProbeMLP(d_in).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, metrics, state_hash(model)


def paired_metrics(
    *,
    task: dict[str, str],
    probe: DepthProbeMLP,
    manifest: Path,
    cache_root: Path,
    feature_root: Path,
    split: str,
    device: torch.device,
    include_delta: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, float] | None, int]:
    level = f"layer_{int(task['layer'])}"
    normal_predictions: list[torch.Tensor] = []
    off_predictions: list[torch.Tensor] = []
    delta_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    records = load_frame_records(manifest, split=split)
    if not records:
        raise RuntimeError(f"No {split} records in {manifest}")
    seen: set[tuple[str, int]] = set()
    with torch.no_grad():
        for record in records:
            video_id = safe_video_id(record)
            feature_path = feature_root / task["model"] / f"video_{video_id}.pt"
            payload = torch_load(feature_path)
            if payload.get("schema_version") != "frozen_probe_geometry_perturbation_features_v1":
                raise RuntimeError(f"Unexpected paired feature schema: {feature_path}")
            if payload.get("model_label") != task["model"] or payload.get("split") != split:
                raise RuntimeError(f"Paired feature provenance mismatch: {feature_path}")
            frame_idx = int(record["frame_index"])
            key = (video_id, frame_idx)
            if key in seen:
                raise RuntimeError(f"Duplicate paired evaluation example: {key}")
            seen.add(key)
            selected = [int(value) for value in payload.get("selected_frames", [])]
            if frame_idx not in selected:
                raise RuntimeError(f"Manifest frame {frame_idx} absent from paired feature payload {feature_path}")
            normal = payload["normal_by_layer"][level][str(frame_idx)].float()
            off = payload["geometry_off_all_by_layer"][level][str(frame_idx)].float()
            fsid = str(record["frame_sample_id"])
            gt = torch_load(cache_root / "gt_depth" / f"frame_{fsid}.pt").float()
            metadata = torch_load(cache_root / "metadata" / f"frame_{fsid}.pt")
            valid = metadata.get("gt_valid_mask", torch.isfinite(gt) & (gt > 0)).bool()
            if normal.shape != off.shape or normal.shape[:-1] != gt.shape or gt.shape != valid.shape:
                raise RuntimeError(
                    f"Paired feature/target shape mismatch {task['model']}/{level}/{fsid}: "
                    f"normal={tuple(normal.shape)} off={tuple(off.shape)} gt={tuple(gt.shape)} valid={tuple(valid.shape)}"
                )
            if int(normal.shape[-1]) != probe.net[0].in_features:
                raise RuntimeError(f"Probe dimension mismatch: {normal.shape[-1]} != {probe.net[0].in_features}")
            normal_predictions.append(probe(normal.reshape(-1, normal.shape[-1]).to(device)).cpu())
            off_predictions.append(probe(off.reshape(-1, off.shape[-1]).to(device)).cpu())
            if include_delta:
                delta_predictions.append(probe((normal - off).reshape(-1, normal.shape[-1]).to(device)).cpu())
            targets.append(gt.reshape(-1))
            masks.append(valid.reshape(-1))
    target = torch.cat(targets)
    mask = torch.cat(masks)
    normal_metrics = metric_values(torch.cat(normal_predictions), target, mask)
    off_metrics = metric_values(torch.cat(off_predictions), target, mask)
    delta_metrics = metric_values(torch.cat(delta_predictions), target, mask) if include_delta else None
    return normal_metrics, off_metrics, delta_metrics, len(seen)


def reference_normal_metrics(
    *,
    task: dict[str, str],
    probe: DepthProbeMLP,
    manifest: Path,
    cache_root: Path,
    split: str,
    device: torch.device,
) -> dict[str, float]:
    """Recompute the saved normal validation metric without touching probe weights."""
    level = f"layer_{int(task['layer'])}"
    feature_model = task.get("reference_model") or task["model"]
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    with torch.no_grad():
        for record in load_frame_records(manifest, split=split):
            fsid = str(record["frame_sample_id"])
            feature_path = cache_root / "features" / feature_model / level / f"frame_{fsid}.pt"
            feature = torch_load(feature_path).float()
            gt = torch_load(cache_root / "gt_depth" / f"frame_{fsid}.pt").float()
            metadata = torch_load(cache_root / "metadata" / f"frame_{fsid}.pt")
            valid = metadata.get("gt_valid_mask", torch.isfinite(gt) & (gt > 0)).bool()
            if feature.shape[:-1] != gt.shape or gt.shape != valid.shape:
                raise RuntimeError(f"Reference feature/target mismatch: {feature_path}")
            if int(feature.shape[-1]) != probe.net[0].in_features:
                raise RuntimeError(f"Reference probe dimension mismatch: {feature_path}")
            predictions.append(probe(feature.reshape(-1, feature.shape[-1]).to(device)).cpu())
            targets.append(gt.reshape(-1))
            masks.append(valid.reshape(-1))
    return metric_values(torch.cat(predictions), torch.cat(targets), torch.cat(masks))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    """Dependency-free layer curve for the primary AbsRel causal drop."""
    width, height, margin = 900, 480, 70
    layers = sorted({int(row["layer"]) for row in rows})
    values = [float(row["absrel_causal_drop"]) for row in rows]
    ymax = max(max(values, default=0.0), 0.0) * 1.1 or 1.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    models = sorted({str(row["model"]) for row in rows})
    x = lambda layer: margin + (width - 2 * margin) * (layers.index(layer) / max(len(layers) - 1, 1))
    y = lambda value: height - margin - (height - 2 * margin) * (value / ymax)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111"/>',
        '<text x="70" y="28" font-family="sans-serif" font-size="18">Frozen-probe causal AbsRel drop (off − normal; positive = harm)</text>',
    ]
    for layer in layers:
        lines.append(f'<text x="{x(layer):.1f}" y="{height-margin+22}" text-anchor="middle" font-family="sans-serif" font-size="12">L{layer}</text>')
    for i in range(5):
        value = ymax * i / 4
        yy = y(value)
        lines.append(f'<line x1="{margin}" y1="{yy:.1f}" x2="{width-margin}" y2="{yy:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{margin-8}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    for index, model in enumerate(models):
        series = sorted((row for row in rows if row["model"] == model), key=lambda row: int(row["layer"]))
        points = " ".join(f"{x(int(row['layer'])):.1f},{y(float(row['absrel_causal_drop'])):.1f}" for row in series)
        color = colors[index % len(colors)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in series:
            lines.append(f'<circle cx="{x(int(row["layer"])):.1f}" cy="{y(float(row["absrel_causal_drop"])):.1f}" r="3" fill="{color}"/>')
        lines.append(f'<text x="{width-margin-5}" y="{margin + 20 * index}" text-anchor="end" fill="{color}" font-family="sans-serif" font-size="13">{html.escape(model)}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="TSV: model, probe_dir, layer")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True, help="Paired extraction root containing gt_depth and metadata.")
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--split", default="dev_eval")
    parser.add_argument("--normal-reference-cache-root", default=None)
    parser.add_argument("--normal-reference-manifest", default=None)
    parser.add_argument("--normal-reference-split", default="val")
    parser.add_argument("--saved-absrel-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-delta-diagnostic", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {output_dir}")
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for task in parse_tasks(Path(args.tasks)):
        probe, saved, before = load_probe(task, device)
        reproduced: dict[str, float] | None = None
        if args.normal_reference_cache_root or args.normal_reference_manifest:
            if not (args.normal_reference_cache_root and args.normal_reference_manifest):
                raise ValueError("Normal-reference cache root and manifest must be supplied together")
            reproduced = reference_normal_metrics(
                task=task, probe=probe, manifest=Path(args.normal_reference_manifest),
                cache_root=Path(args.normal_reference_cache_root), split=args.normal_reference_split, device=device,
            )
            if abs(float(reproduced["absrel"]) - float(saved["absrel"])) > float(args.saved_absrel_tolerance):
                raise RuntimeError(
                    f"Frozen probe failed saved normal reproduction for {task['model']}/L{task['layer']}: "
                    f"saved={saved['absrel']}, reproduced={reproduced['absrel']}"
                )
        normal, off, delta, examples = paired_metrics(
            task=task, probe=probe, manifest=Path(args.manifest), cache_root=Path(args.cache_root),
            feature_root=Path(args.feature_cache_root), split=args.split, device=device,
            include_delta=bool(args.include_delta_diagnostic),
        )
        after = state_hash(probe)
        if before != after:
            raise RuntimeError(f"Frozen probe parameters changed during evaluation: {task['probe_dir']}")
        if not all(math.isfinite(value) for value in (normal["absrel"], off["absrel"])):
            raise RuntimeError(f"Non-finite paired AbsRel for {task['model']}/L{task['layer']}")
        absrel_drop = float(off["absrel"] - normal["absrel"])
        row: dict[str, Any] = {
            "model": task["model"], "layer": int(task["layer"]), "probe_dir": task["probe_dir"],
            "saved_probe_absrel": saved.get("absrel"), "examples": examples,
            "normal_mae": normal["mae"], "geometry_off_mae": off["mae"], "mae_causal_drop": off["mae"] - normal["mae"],
            "normal_absrel": normal["absrel"], "geometry_off_absrel": off["absrel"],
            "absrel_causal_drop": absrel_drop, "absrel_relative_causal_drop": absrel_drop / (abs(normal["absrel"]) + 1e-6),
            "normal_delta125": normal["delta125"], "geometry_off_delta125": off["delta125"],
            "delta125_causal_drop": normal["delta125"] - off["delta125"], "num_tokens": normal["num_tokens"],
            "parameter_hash": before,
        }
        if reproduced is not None:
            row.update({"reproduced_saved_normal_absrel": reproduced["absrel"], "reproduced_saved_normal_mae": reproduced["mae"], "reproduced_saved_normal_delta125": reproduced["delta125"]})
        if delta is not None:
            row.update({"delta_absrel_diagnostic": delta["absrel"], "delta_mae_diagnostic": delta["mae"], "delta_delta125_diagnostic": delta["delta125"]})
        rows.append(row)
    rows.sort(key=lambda row: (str(row["model"]), int(row["layer"])))
    write_csv(output_dir / "causal_probe_drop.csv", rows)
    write_svg(output_dir / "causal_probe_drop_absrel.svg", rows)
    table = ["# Frozen-probe causal geometry drop", "", "Primary metric: AbsRel (lower is better). `absrel_causal_drop = geometry_off − normal`; positive means removing geometry harms the fixed normal readout.", "", "| Model | Layer | Normal AbsRel | Off AbsRel | Drop | Relative drop |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['model']} | {row['layer']} | {row['normal_absrel']:.6f} | {row['geometry_off_absrel']:.6f} | {row['absrel_causal_drop']:.6f} | {row['absrel_relative_causal_drop']:.2%} |")
    (output_dir / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({"tasks": str(Path(args.tasks).resolve()), "manifest": str(Path(args.manifest).resolve()), "cache_root": str(Path(args.cache_root).resolve()), "feature_cache_root": str(Path(args.feature_cache_root).resolve()), "split": args.split, "normal_reference_cache_root": args.normal_reference_cache_root, "normal_reference_manifest": args.normal_reference_manifest, "normal_reference_split": args.normal_reference_split, "delta_diagnostic": bool(args.include_delta_diagnostic), "parameter_hashes_verified": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
