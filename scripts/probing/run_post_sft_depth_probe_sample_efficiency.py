#!/usr/bin/env python
"""Cache-only video-subsampling sweep for the reported post-SFT MLP depth probe.

The implementation deliberately calls :func:`train_one_probe` from the
existing trainer.  Thus the MLP, token preprocessing, fixed validation set,
early stopping, and metric implementation are identical to the reported
post-SFT depth probes.  Only the list of *training videos* changes.

An inventory JSON supplies one fixed comparison representation per model.  A
representation is selected from its already-reported full-data probe before
the sweep begins; it is never re-selected from a smaller sample.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import load_frame_records, read_json, stable_int_seed, write_csv, write_json
from scripts.probing.train_depth_probes import filter_existing_records, train_one_probe


METRICS = ("delta125", "mae", "absrel")
HIGHER_IS_BETTER = {"delta125": True, "mae": False, "absrel": False}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    display_name: str
    source_root: Path
    feature_level: str
    full_metrics_path: Path


def parse_csv_ints(value: str, *, minimum: int) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or any(item < minimum for item in result):
        raise ValueError(f"Expected comma-separated integers >= {minimum}, got {value!r}")
    if len(set(result)) != len(result):
        raise ValueError(f"Sample sizes/seeds must be unique, got {result}")
    return result


def load_inventory(path: Path) -> list[ModelSpec]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise TypeError("Inventory must be a JSON object with a models list.")
    result: list[ModelSpec] = []
    for item in payload["models"]:
        if not isinstance(item, dict):
            raise TypeError(f"Invalid inventory model: {item!r}")
        required = ("label", "source_root", "feature_level", "full_metrics_path")
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise ValueError(f"Inventory item misses {missing}: {item}")
        result.append(
            ModelSpec(
                label=str(item["label"]),
                display_name=str(item.get("display_name", item["label"])),
                source_root=Path(str(item["source_root"])).resolve(),
                feature_level=str(item["feature_level"]),
                full_metrics_path=Path(str(item["full_metrics_path"])).resolve(),
            )
        )
    labels = [item.label for item in result]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Inventory labels must be unique, got {labels}")
    return result


def videos_by_split(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise TypeError("Sample index payload has no videos list.")
    train = [item for item in videos if item.get("split") == "train"]
    val = [item for item in videos if item.get("split") == "val"]
    if not train or not val:
        raise RuntimeError(f"Expected non-empty train and val videos, got train={len(train)}, val={len(val)}")
    return train, val


def frames(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for video in videos:
        for frame in video.get("frames", []):
            record = dict(video)
            record.pop("frames", None)
            record.update(frame)
            result.append(record)
    return result


def select_videos(train: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size > len(train):
        raise ValueError(f"Cannot sample {sample_size} training videos from {len(train)}")
    # Preserve original manifest order after selection.  This makes the full
    # sample with seed 0 exactly match the existing trainer's record order.
    if sample_size == len(train):
        return list(train)
    rng = random.Random(stable_int_seed("post_sft_depth_probe_sample_efficiency_v1", seed, sample_size))
    selected = set(rng.sample(range(len(train)), sample_size))
    return [video for index, video in enumerate(train) if index in selected]


def full_metrics(spec: ModelSpec) -> dict[str, float]:
    payload = read_json(spec.full_metrics_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metrics object: {spec.full_metrics_path}")
    if payload.get("model_label") != spec.label or payload.get("feature_level") != spec.feature_level:
        raise RuntimeError(f"Full metrics identity mismatch: {spec.full_metrics_path}")
    if int(payload.get("num_tokens", -1)) != 75656:
        raise RuntimeError(f"Full metrics do not use the fixed 75,656-token validation protocol: {spec.full_metrics_path}")
    return {metric: float(payload[metric]) for metric in METRICS}


def validate_sources(specs: list[ModelSpec], train_records: list[dict[str, Any]], val_records: list[dict[str, Any]]) -> None:
    failures = []
    for spec in specs:
        if not spec.source_root.is_dir():
            failures.append(f"{spec.label}: missing source_root {spec.source_root}")
            continue
        if not spec.full_metrics_path.is_file():
            failures.append(f"{spec.label}: missing full metrics {spec.full_metrics_path}")
            continue
        full_metrics(spec)
        train_count = len(filter_existing_records(spec.source_root, spec.label, spec.feature_level, train_records))
        val_count = len(filter_existing_records(spec.source_root, spec.label, spec.feature_level, val_records))
        if train_count != len(train_records) or val_count != len(val_records):
            failures.append(
                f"{spec.label}/{spec.feature_level}: cached frames train {train_count}/{len(train_records)}, "
                f"val {val_count}/{len(val_records)}"
            )
    if failures:
        raise RuntimeError("Cache inventory is incomplete:\n- " + "\n- ".join(failures))


def average_ranks(values: list[float], *, higher_is_better: bool) -> np.ndarray:
    ordered = np.asarray(values, dtype=float)
    sort_values = -ordered if higher_is_better else ordered
    order = np.argsort(sort_values, kind="mergesort")
    ranks = np.empty(len(ordered), dtype=float)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(sort_values[order[end]], sort_values[order[position]], rel_tol=1e-12, abs_tol=1e-12):
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return ranks


def spearman(values: list[float], reference: list[float], *, higher_is_better: bool) -> float:
    first = average_ranks(values, higher_is_better=higher_is_better)
    second = average_ranks(reference, higher_is_better=higher_is_better)
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def mean_std_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"mean": float("nan"), "std": float("nan"), "ci95_halfwidth": float("nan"), "n": 0}
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return {"mean": float(np.mean(finite)), "std": std, "ci95_halfwidth": 1.96 * std / math.sqrt(len(finite)), "n": int(len(finite))}


def write_analysis(output_dir: Path, rows: list[dict[str, Any]], refs: dict[str, dict[str, float]], specs: list[ModelSpec]) -> None:
    summary: list[dict[str, Any]] = []
    for spec in specs:
        for sample_size in sorted({int(row["sample_size"]) for row in rows}):
            selected = [row for row in rows if row["model"] == spec.label and int(row["sample_size"]) == sample_size]
            item: dict[str, Any] = {
                "model": spec.label,
                "display_name": spec.display_name,
                "feature_level": spec.feature_level,
                "sample_size": sample_size,
                "reference_delta125": refs[spec.label]["delta125"],
                "reference_mae": refs[spec.label]["mae"],
                "reference_absrel": refs[spec.label]["absrel"],
            }
            for metric in METRICS:
                stats = mean_std_ci([float(row[metric]) for row in selected])
                item.update({f"{metric}_{key}": value for key, value in stats.items()})
                item[f"{metric}_gap_from_full"] = item[f"{metric}_mean"] - refs[spec.label][metric]
            summary.append(item)
    write_json(output_dir / "per_model_summary.json", summary)
    write_csv(output_dir / "per_model_summary.csv", summary)

    ranking: list[dict[str, Any]] = []
    labels = [spec.label for spec in specs]
    for sample_size in sorted({int(row["sample_size"]) for row in rows}):
        seeds = sorted({int(row["seed"]) for row in rows if int(row["sample_size"]) == sample_size})
        for metric in METRICS:
            reference = [refs[label][metric] for label in labels]
            seed_values = []
            for seed in seeds:
                by_model = {str(row["model"]): row for row in rows if int(row["sample_size"]) == sample_size and int(row["seed"]) == seed}
                if set(by_model) != set(labels):
                    continue
                value = spearman([float(by_model[label][metric]) for label in labels], reference, higher_is_better=HIGHER_IS_BETTER[metric])
                seed_values.append(value)
            stats = mean_std_ci(seed_values)
            ranking.append({"sample_size": sample_size, "metric": metric, "reference": "reported_full_data", **{f"spearman_{key}": value for key, value in stats.items()}})
    write_json(output_dir / "ranking_stability.json", ranking)
    write_csv(output_dir / "ranking_stability.csv", ranking)

    lines = ["# Post-SFT MLP depth-probe sample efficiency", "", "All values use the existing MLP probe and fixed 193-video validation split.", ""]
    for spec in specs:
        lines.extend([f"## {spec.display_name} (`{spec.feature_level}`)", "", "| Train videos | δ<1.25 mean ± std | MAE mean ± std | AbsRel mean ± std |", "|---:|---:|---:|---:|"])
        for row in [item for item in summary if item["model"] == spec.label]:
            lines.append(
                f"| {row['sample_size']} | {row['delta125_mean']:.6f} ± {row['delta125_std']:.6f} | "
                f"{row['mae_mean']:.6f} ± {row['mae_std']:.6f} | {row['absrel_mean']:.6f} ± {row['absrel_std']:.6f} |"
            )
        lines.append("")
    lines.extend(["## Ranking stability", "", "| Train videos | Metric | Spearman mean ± std |", "|---:|---|---:|"])
    for row in ranking:
        lines.append(f"| {row['sample_size']} | {row['metric']} | {row['spearman_mean']:.4f} ± {row['spearman_std']:.4f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for metric, ylabel in (("delta125", "δ<1.25"), ("mae", "MAE (m)"), ("absrel", "AbsRel")):
        figure, axis = plt.subplots(figsize=(9, 5))
        for spec in specs:
            items = [item for item in summary if item["model"] == spec.label]
            x = [item["sample_size"] for item in items]
            y = [item[f"{metric}_mean"] for item in items]
            ci = [item[f"{metric}_ci95_halfwidth"] for item in items]
            axis.plot(x, y, marker="o", label=spec.display_name)
            axis.fill_between(x, np.asarray(y) - np.asarray(ci), np.asarray(y) + np.asarray(ci), alpha=0.16)
            axis.axhline(refs[spec.label][metric], linestyle="--", linewidth=0.8, alpha=0.45)
        axis.set_xscale("symlog", linthresh=25)
        axis.set_xlabel("Probe-training videos")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"sample_efficiency_{metric}.png", dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for metric in METRICS:
        items = [item for item in ranking if item["metric"] == metric]
        x = [item["sample_size"] for item in items]
        y = [item["spearman_mean"] for item in items]
        ci = [item["spearman_ci95_halfwidth"] for item in items]
        axis.plot(x, y, marker="o", label=metric)
        axis.fill_between(x, np.asarray(y) - np.asarray(ci), np.asarray(y) + np.asarray(ci), alpha=0.16)
    axis.set_xscale("symlog", linthresh=25)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("Probe-training videos")
    axis.set_ylabel("Spearman vs. reported full-data ranking")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "ranking_stability.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-sizes", default="25,50,100,200,400,600,800,1000,1006")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--model-labels", default=None, help="Optional comma-separated subset of inventory labels.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-artifacts", action="store_true", help="Retain individual probe checkpoints and histories.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inventory_path = Path(args.inventory).resolve()
    split_path = Path(args.sample_indices).resolve()
    output_dir = Path(args.output_dir).resolve()
    sizes = parse_csv_ints(args.sample_sizes, minimum=1)
    seeds = parse_csv_ints(args.seeds, minimum=0)
    specs = load_inventory(inventory_path)
    if args.model_labels:
        requested = [item.strip() for item in args.model_labels.split(",") if item.strip()]
        absent = sorted(set(requested) - {spec.label for spec in specs})
        if absent:
            raise ValueError(f"Requested labels absent from inventory: {absent}")
        specs = [spec for spec in specs if spec.label in requested]
    payload = read_json(split_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected split JSON object: {split_path}")
    train_videos, val_videos = videos_by_split(payload)
    if max(sizes) > len(train_videos):
        raise ValueError(f"Requested max sample size {max(sizes)} exceeds official train split {len(train_videos)}")
    train_records, val_records = frames(train_videos), frames(val_videos)
    if len(train_records) != 2012 or len(val_records) != 386:
        raise RuntimeError(f"Unexpected official frame counts: train={len(train_records)}, val={len(val_records)}")
    validate_sources(specs, train_records, val_records)
    refs = {spec.label: full_metrics(spec) for spec in specs}
    output_dir.mkdir(parents=True, exist_ok=True)
    # Each model is run in its own output directory by the formal launcher.
    # Refuse an accidental duplicate process rather than letting its cleanup
    # race a checkpoint write from the first process.
    lock_handle = (output_dir / ".sample_efficiency.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another sweep process owns {output_dir}") from exc
    write_json(output_dir / "run_config.json", {
        "schema_version": "post_sft_depth_probe_sample_efficiency_v1",
        "inventory": str(inventory_path), "sample_indices": str(split_path),
        "train_videos": len(train_videos), "val_videos": len(val_videos),
        "train_frames": len(train_records), "val_frames": len(val_records),
        "sample_sizes": sizes, "seeds": seeds, "device": args.device,
        "probe_hyperparameters": {"epochs": 50, "batch_size": 32, "lr": 1e-3, "early_stop_patience": 10, "num_workers": 0},
        "models": [{"label": spec.label, "display_name": spec.display_name, "source_root": str(spec.source_root), "feature_level": spec.feature_level, "full_metrics_path": str(spec.full_metrics_path), "full_reference": refs[spec.label]} for spec in specs],
    })
    if args.dry_run:
        print(json.dumps({"status": "PASS", "models": [spec.label for spec in specs], "sample_sizes": sizes, "seeds": seeds}, indent=2))
        return

    raw_path = output_dir / "raw_results.json"
    existing: list[dict[str, Any]] = []
    if raw_path.is_file():
        loaded = read_json(raw_path)
        if not isinstance(loaded, list):
            raise TypeError(f"Existing raw results are not a list: {raw_path}")
        existing = [dict(item) for item in loaded if isinstance(item, dict)]
    completed = {(str(row.get("model")), int(row.get("sample_size")), int(row.get("seed"))) for row in existing}
    rows = list(existing)
    for spec in specs:
        for sample_size in sizes:
            subset_videos = select_videos(train_videos, sample_size, 0)
            # Probe seed controls both the documented training RNG and the
            # independent video-level subset; recompute inside the seed loop.
            for seed in seeds:
                key = (spec.label, sample_size, seed)
                if key in completed:
                    print(f"[SKIP] {spec.label} n={sample_size} seed={seed}", flush=True)
                    continue
                selected_videos = select_videos(train_videos, sample_size, seed)
                selected_records = frames(selected_videos)
                probe_subdir = output_dir / "probe_artifacts" / f"n_{sample_size:04d}" / f"seed_{seed:02d}"
                task_args = SimpleNamespace(
                    probe_seed=seed, device=args.device, batch_size=32, num_workers=0,
                    lr=1e-3, epochs=50, early_stop_patience=10,
                    probe_subdir=str(probe_subdir), experiment_variant="post_sft_sample_efficiency_v1",
                    fusion_init_seed=None, spatialstack_output_init=None, shared_llm_layers=None,
                )
                print(f"[RUN] model={spec.label} feature={spec.feature_level} n={sample_size} seed={seed} train_videos={len(selected_videos)} val_videos={len(val_videos)}", flush=True)
                result = train_one_probe(
                    output_root=spec.source_root, model_label=spec.label, feature_level=spec.feature_level,
                    train_records=selected_records, val_records=val_records, args=task_args,
                )
                row = {
                    "model": spec.label, "display_name": spec.display_name, "feature_level": spec.feature_level,
                    "source_root": str(spec.source_root), "sample_size": sample_size, "seed": seed,
                    "train_videos": len(selected_videos), "val_videos": len(val_videos),
                    "reference_delta125": refs[spec.label]["delta125"], "reference_mae": refs[spec.label]["mae"], "reference_absrel": refs[spec.label]["absrel"],
                    **{metric: float(result[metric]) for metric in METRICS},
                    "best_epoch": int(result["best_epoch"]), "num_tokens": int(result["num_tokens"]),
                }
                rows.append(row)
                completed.add(key)
                write_json(raw_path, rows)
                write_csv(output_dir / "raw_results.csv", rows)
                if not args.keep_artifacts:
                    shutil.rmtree(probe_subdir, ignore_errors=True)
                write_analysis(output_dir, rows, refs, specs)
    write_analysis(output_dir, rows, refs, specs)
    print(f"[DONE] raw={raw_path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
