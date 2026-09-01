#!/usr/bin/env python
"""Cache-only absolute-convergence sweep for the three post-SFT anchor models.

This intentionally reuses ``train_one_probe`` from the reported MLP depth
probe.  It changes only the ordered training-video list passed to that
trainer; validation, token preprocessing, targets, optimizer, architecture,
and early stopping remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import (  # noqa: E402
    load_frame_records,
    read_json,
    stable_int_seed,
    write_csv,
    write_json,
)
from scripts.probing.train_depth_probes import filter_existing_records, train_one_probe  # noqa: E402

CANONICAL_LAYERS = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)
SWEEP_POINTS = ("projected_features", "layer_0", "layer_1", "layer_6", "layer_27")
SIZES = (25, 100, 200, 400)
SEEDS = (0, 1, 2)
METRICS = ("delta125", "absrel", "mae")
OFFICIAL_SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"

MODELS: dict[str, dict[str, str]] = {
    "vlm3r_baseline": {
        "display_name": "Baseline",
        "source_root": "/home/shaoruei/probe_cache/scannet_depth_layers_v1/full",
        "pre_llm": "fusion_output",
        "reference_column": "Baseline",
    },
    "cut3r_spatialstack_44323703": {
        "display_name": "SpatialStack 0/1/2",
        "source_root": "/home/shaoruei/probe_cache/post_sft_anchor_pilot_v1/full",
        "pre_llm": "siglip_output",
        "reference_column": "Spatial Stack to Layer 0/1/2",
    },
    "zero_spatial": {
        "display_name": "0 spatial",
        "source_root": "/home/shaoruei/probe_cache/scannet_depth_layers_v1/full",
        "pre_llm": "siglip_output",
        "reference_column": "Zero Spatial Feature",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_points(label: str) -> tuple[str, ...]:
    return (MODELS[label]["pre_llm"], "projected_features") + tuple(f"layer_{layer}" for layer in CANONICAL_LAYERS)


def flatten(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for video in videos:
        base = dict(video)
        frames = base.pop("frames", [])
        for frame in frames:
            item = dict(base)
            item.update(frame)
            result.append(item)
    return result


def parse_xlsx(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read only the two owner-supplied metric sheets without openpyxl."""
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "p": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def column_number(reference: str) -> int:
        letters = "".join(ch for ch in reference if ch.isalpha())
        out = 0
        for char in letters:
            out = out * 26 + ord(char) - 64
        return out

    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{ns['m']}}}t")))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels.findall("p:Relationship", ns)}
        sheets: dict[str, list[dict[int, str | None]]] = {}
        for sheet in workbook.find("m:sheets", ns):
            name = sheet.attrib["name"].strip().lower()
            target = targets[sheet.attrib[f"{{{ns['r']}}}id"]]
            target = target if target.startswith("xl/") else f"xl/{target}"
            root = ET.fromstring(archive.read(target))
            rows: list[dict[int, str | None]] = []
            for row in root.findall(".//m:sheetData/m:row", ns):
                values: dict[int, str | None] = {}
                for cell in row.findall("m:c", ns):
                    raw = cell.find("m:v", ns)
                    if raw is None:
                        value = None
                    elif cell.attrib.get("t") == "s":
                        value = shared[int(raw.text)]
                    else:
                        value = raw.text
                    values[column_number(cell.attrib["r"])] = value
                if values:
                    rows.append(values)
            sheets[name] = rows
    required = {"delta125", "absrel"}
    if not required.issubset(sheets):
        raise RuntimeError(f"Reference workbook must contain sheets {sorted(required)}, found {sorted(sheets)}")
    reference: dict[str, dict[str, dict[str, float]]] = {label: {} for label in MODELS}
    for metric in sorted(required):
        rows = sheets[metric]
        header = {str(value).strip(): column for column, value in rows[0].items() if value}
        for label, spec in MODELS.items():
            column = header.get(spec["reference_column"])
            if column is None:
                raise RuntimeError(f"Workbook {metric} sheet lacks column {spec['reference_column']!r}")
            for row in rows[1:]:
                feature = str(row.get(1, "")).strip()
                if feature not in SWEEP_POINTS:
                    continue
                value = row.get(column)
                if value is None:
                    raise RuntimeError(f"Workbook missing {metric}/{spec['reference_column']}/{feature}")
                reference[label].setdefault(feature, {})[metric] = float(value)
    for label in MODELS:
        for feature in SWEEP_POINTS:
            missing = sorted({"delta125", "absrel"} - set(reference[label].get(feature, {})))
            if missing:
                raise RuntimeError(f"Missing workbook reference {label}/{feature}: {missing}")
    return reference


def load_official(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if sha256(path) != OFFICIAL_SPLIT_SHA256:
        raise RuntimeError("Official split SHA-256 mismatch")
    videos = payload.get("videos", [])
    train = [item for item in videos if item.get("split") == "train"]
    val = [item for item in videos if item.get("split") == "val"]
    if len(train) != 1006 or len(val) != 193:
        raise RuntimeError(f"Expected 1006/193 videos, found {len(train)}/{len(val)}")
    return train, val, payload


def selection_payload(train: list[dict[str, Any]], val: list[dict[str, Any]], n: int, seed: int) -> dict[str, Any]:
    if n == 400:
        selected = list(train)
        selector = "canonical_400_fixed_across_probe_seeds_v1"
    else:
        rng = random.Random(stable_int_seed("post_sft_anchor_pilot_v1", n, seed, hash_json([x["video_path"] for x in train])))
        chosen = set(rng.sample(range(len(train)), n))
        selected = [video for index, video in enumerate(train) if index in chosen]
        selector = "post_sft_anchor_pilot_v1"
    payload = {
        "schema_version": "post_sft_anchor_pilot_subset_v1",
        "selector": selector,
        "sample_size": n,
        "seed": seed,
        "train_videos": n,
        "val_videos": len(val),
        "videos": selected + list(val),
    }
    payload["subset_sha256"] = hash_json([item["video_path"] for item in selected])
    return payload


def prepare(shared_root: Path, split: Path, workbook: Path) -> dict[str, Any]:
    train, val, source_payload = load_official(split)
    rng = random.Random(stable_int_seed("post_sft_depth_probe_sample_efficiency_v1", 0, 400))
    chosen = set(rng.sample(range(len(train)), 400))
    pool = [video for index, video in enumerate(train) if index in chosen]
    manifests = shared_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    pool_payload = {
        "schema_version": "post_sft_anchor_pilot_canonical_400_v1",
        "source_split": str(split),
        "source_split_sha256": sha256(split),
        "selector": "post_sft_depth_probe_sample_efficiency_v1(seed=0,sample_size=400)",
        "train_videos": 400,
        "val_videos": 193,
        "videos": pool + val,
    }
    pool_payload["canonical_pool_sha256"] = hash_json([item["video_path"] for item in pool])
    write_json(manifests / "canonical_400.json", pool_payload)
    subsets: list[dict[str, Any]] = []
    for size in SIZES:
        for seed in SEEDS:
            payload = selection_payload(pool, val, size, seed)
            out = manifests / f"n_{size:04d}_seed_{seed:02d}.json"
            write_json(out, payload)
            subsets.append({"sample_size": size, "seed": seed, "path": str(out), "subset_sha256": payload["subset_sha256"]})
    refs = parse_xlsx(workbook)
    reference_manifest = {
        "schema_version": "post_sft_anchor_pilot_reference_v1",
        "reference_workbook": str(workbook.resolve()),
        "reference_workbook_sha256": sha256(workbook),
        "official_split": str(split),
        "official_split_sha256": sha256(split),
        "reference_metrics": ["delta125", "absrel"],
        "models": MODELS,
        "sweep_points": list(SWEEP_POINTS),
        "references": refs,
    }
    write_json(shared_root / "reference_manifest.json", reference_manifest)
    write_json(shared_root / "subset_index.json", subsets)
    return {"pool": pool_payload, "subsets": subsets, "reference": reference_manifest, "source": source_payload}


def load_prepared(shared_root: Path, split: Path, workbook: Path) -> dict[str, Any]:
    """Reuse immutable shared manifests when the wrapper prepared them first."""
    reference_path = shared_root / "reference_manifest.json"
    subset_path = shared_root / "subset_index.json"
    if not reference_path.is_file() or not subset_path.is_file():
        return prepare(shared_root, split, workbook)
    reference = read_json(reference_path)
    subsets = read_json(subset_path)
    if (
        not isinstance(reference, dict)
        or not isinstance(subsets, list)
        or reference.get("reference_workbook_sha256") != sha256(workbook)
        or reference.get("official_split_sha256") != sha256(split)
    ):
        return prepare(shared_root, split, workbook)
    return {"reference": reference, "subsets": subsets}


def cache_failures(label: str, split: Path) -> list[str]:
    train_records = load_frame_records(split, split="train")
    val_records = load_frame_records(split, split="val")
    spec = MODELS[label]
    root = Path(spec["source_root"])
    failures: list[str] = []
    for feature in canonical_points(label):
        train_count = len(filter_existing_records(root, label, feature, train_records))
        val_count = len(filter_existing_records(root, label, feature, val_records))
        if train_count != len(train_records) or val_count != len(val_records):
            failures.append(f"{feature}: train={train_count}/{len(train_records)} val={val_count}/{len(val_records)}")
    return failures


def run_sweep(label: str, split: Path, workbook: Path, shared_root: Path, output_dir: Path, device: str, keep_artifacts: bool) -> None:
    if label not in MODELS:
        raise ValueError(f"Unsupported anchor model {label}")
    prepared = load_prepared(shared_root, split, workbook)
    failures = cache_failures(label, split)
    if failures:
        raise RuntimeError(f"Incomplete canonical cache for {label}:\n- " + "\n- ".join(failures))
    refs = prepared["reference"]["references"][label]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.json"
    existing = read_json(raw_path) if raw_path.is_file() else []
    if not isinstance(existing, list):
        raise TypeError(f"Expected raw-result list: {raw_path}")
    rows = [dict(item) for item in existing if isinstance(item, dict)]
    completed = {(str(x.get("model")), str(x.get("feature_level")), int(x.get("sample_size")), int(x.get("seed")), str(x.get("subset_sha256"))) for x in rows}
    val_records = load_frame_records(split, split="val")
    for item in prepared["subsets"]:
        subset = read_json(Path(item["path"]))
        train_videos = [video for video in subset["videos"] if video.get("split") == "train"]
        train_records = flatten(train_videos)
        for feature in SWEEP_POINTS:
            key = (label, feature, int(item["sample_size"]), int(item["seed"]), str(item["subset_sha256"]))
            if key in completed:
                continue
            artifact = output_dir / "probe_artifacts" / feature / f"n_{int(item['sample_size']):04d}_seed_{int(item['seed']):02d}"
            task_args = SimpleNamespace(
                probe_seed=int(item["seed"]), device=device, batch_size=32, num_workers=0, lr=1e-3,
                epochs=50, early_stop_patience=10, probe_subdir=str(artifact),
                experiment_variant="post_sft_anchor_sample_efficiency_v1",
                fusion_init_seed=None, spatialstack_output_init=None, shared_llm_layers=None,
            )
            print(f"[RUN] {label}/{feature} n={item['sample_size']} seed={item['seed']} subset={item['subset_sha256']}", flush=True)
            result = train_one_probe(
                output_root=Path(MODELS[label]["source_root"]), model_label=label, feature_level=feature,
                train_records=train_records, val_records=val_records, args=task_args,
            )
            if int(result["num_tokens"]) != 75656:
                raise RuntimeError(f"Unexpected validation tokens: {result['num_tokens']}")
            row = {
                "model": label, "display_name": MODELS[label]["display_name"], "feature_level": feature,
                "sample_size": int(item["sample_size"]), "seed": int(item["seed"]),
                "subset_sha256": str(item["subset_sha256"]), "train_videos": len(train_videos), "val_videos": 193,
                "reference_delta125": refs[feature]["delta125"], "reference_absrel": refs[feature]["absrel"],
                "delta125": float(result["delta125"]), "absrel": float(result["absrel"]), "mae": float(result["mae"]),
                "best_epoch": int(result["best_epoch"]), "num_tokens": int(result["num_tokens"]),
            }
            rows.append(row)
            completed.add(key)
            write_json(raw_path, rows)
            write_csv(output_dir / "raw_results.csv", rows)
            if not keep_artifacts:
                shutil.rmtree(artifact, ignore_errors=True)
    write_json(raw_path, rows)
    write_csv(output_dir / "raw_results.csv", rows)


def stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def analyze(shared_root: Path, output_root: Path) -> None:
    prepared = prepare(shared_root, Path(shared_root / "official_split_path.txt").read_text().strip() if (shared_root / "official_split_path.txt").is_file() else Path("/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"), Path(shared_root / "reference_workbook_path.txt").read_text().strip() if (shared_root / "reference_workbook_path.txt").is_file() else Path("/home/shaoruei/SpatialFocus/post-sft-result-for-codex.xlsx"))
    all_rows: list[dict[str, Any]] = []
    for label in MODELS:
        path = output_root / "models" / label / "raw_results.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        if not isinstance(payload, list):
            raise TypeError(path)
        all_rows.extend(dict(item) for item in payload if isinstance(item, dict))
    expected = len(MODELS) * len(SWEEP_POINTS) * len(SIZES) * len(SEEDS)
    if len(all_rows) != expected:
        raise RuntimeError(f"Expected {expected} raw rows, found {len(all_rows)}")
    summary: list[dict[str, Any]] = []
    marginal: list[dict[str, Any]] = []
    for label in MODELS:
        for feature in SWEEP_POINTS:
            per_size: dict[int, dict[str, Any]] = {}
            for size in SIZES:
                rows = [row for row in all_rows if row["model"] == label and row["feature_level"] == feature and int(row["sample_size"]) == size]
                if len(rows) != 3:
                    raise RuntimeError(f"Expected three seeds for {label}/{feature}/n={size}, got {len(rows)}")
                item: dict[str, Any] = {"model": label, "display_name": MODELS[label]["display_name"], "feature_level": feature, "sample_size": size, "subset_sha256_seed_0": next(row["subset_sha256"] for row in rows if int(row["seed"]) == 0), "subset_sha256_seed_1": next(row["subset_sha256"] for row in rows if int(row["seed"]) == 1), "subset_sha256_seed_2": next(row["subset_sha256"] for row in rows if int(row["seed"]) == 2)}
                for metric in METRICS:
                    values = [float(row[metric]) for row in sorted(rows, key=lambda x: int(x["seed"]))]
                    mean, std = stats(values)
                    item.update({f"{metric}_seed_{seed}": value for seed, value in zip(SEEDS, values)})
                    item[f"{metric}_mean"] = mean
                    item[f"{metric}_std"] = std
                    if metric in ("delta125", "absrel"):
                        full = float(rows[0][f"reference_{metric}"])
                        signed = mean - full
                        absolute = abs(signed)
                        item[f"full_{metric}"] = full
                        item[f"{metric}_signed_gap"] = signed
                        item[f"{metric}_absolute_gap"] = absolute
                        item[f"{metric}_relative_gap"] = absolute / abs(full) if full else math.nan
                summary.append(item)
                per_size[size] = item
            for metric in ("delta125", "absrel"):
                for before, after in zip(SIZES, SIZES[1:]):
                    before_item, after_item = per_size[before], per_size[after]
                    marginal.append({
                        "model": label, "display_name": MODELS[label]["display_name"], "feature_level": feature,
                        "metric": metric, "from_sample_size": before, "to_sample_size": after,
                        "absolute_gap_before": before_item[f"{metric}_absolute_gap"], "absolute_gap_after": after_item[f"{metric}_absolute_gap"],
                        "absolute_gap_improvement": before_item[f"{metric}_absolute_gap"] - after_item[f"{metric}_absolute_gap"],
                        "combined_seed_std": math.hypot(before_item[f"{metric}_std"], after_item[f"{metric}_std"]),
                    })
    write_json(output_root / "raw_results.json", all_rows)
    write_csv(output_root / "raw_results.csv", all_rows)
    write_json(output_root / "summary.json", summary)
    write_csv(output_root / "summary.csv", summary)
    write_json(output_root / "marginal_improvement.json", marginal)
    write_csv(output_root / "marginal_improvement.csv", marginal)
    lines = ["# Three-anchor post-SFT absolute convergence", "", "Absolute delta125/AbsRel gap is the primary convergence quantity.", ""]
    for feature in SWEEP_POINTS:
        lines += [f"## {feature}", "", "| Model | Videos | δ absolute gap | AbsRel absolute gap |", "|---|---:|---:|---:|"]
        for row in [x for x in summary if x["feature_level"] == feature]:
            lines.append(f"| {row['display_name']} | {row['sample_size']} | {row['delta125_absolute_gap']:.6f} | {row['absrel_absolute_gap']:.6f} |")
        lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for feature in SWEEP_POINTS:
        figure, axes = plt.subplots(len(MODELS), 2, figsize=(10, 8), sharex=True)
        for row_index, label in enumerate(MODELS):
            for col_index, metric in enumerate(("delta125", "absrel")):
                axis = axes[row_index, col_index]
                items = [x for x in summary if x["model"] == label and x["feature_level"] == feature]
                x = [x["sample_size"] for x in items]
                y = [x[f"{metric}_mean"] for x in items]
                err = [x[f"{metric}_std"] for x in items]
                full = items[0][f"full_{metric}"]
                axis.errorbar(x, y, yerr=err, marker="o", capsize=3, label="reduced probe")
                axis.axhline(full, linestyle="--", color="black", linewidth=1, label="full-data reference")
                axis.set_xscale("symlog", linthresh=25)
                axis.set_title(f"{MODELS[label]['display_name']} — {metric}")
                axis.set_xlabel("training videos")
                axis.set_ylabel(metric)
                if row_index == 0 and col_index == 0:
                    axis.legend(fontsize=8)
        figure.suptitle(f"Absolute-convergence pilot — {feature}")
        figure.tight_layout()
        figure.savefig(output_root / f"convergence_{feature}.png", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(len(MODELS), len(SWEEP_POINTS), figsize=(18, 8), sharex=True)
    for row_index, label in enumerate(MODELS):
        for col_index, feature in enumerate(SWEEP_POINTS):
            axis = axes[row_index, col_index]
            items = [x for x in summary if x["model"] == label and x["feature_level"] == feature]
            axis.errorbar([x["sample_size"] for x in items], [x["mae_mean"] for x in items], yerr=[x["mae_std"] for x in items], marker="o", capsize=3)
            axis.set_xscale("symlog", linthresh=25)
            axis.set_title(feature)
            if col_index == 0: axis.set_ylabel(f"{MODELS[label]['display_name']}\nMAE")
            if row_index == len(MODELS) - 1: axis.set_xlabel("videos")
    figure.tight_layout()
    figure.savefig(output_root / "mae_supplementary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "preflight", "sweep", "analyze"), required=True)
    parser.add_argument("--model-label", choices=tuple(MODELS), default=None)
    parser.add_argument("--split", type=Path, default=Path("/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"))
    parser.add_argument("--reference-workbook", type=Path, default=REPO_ROOT / "post-sft-result-for-codex.xlsx")
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    args.shared_root.mkdir(parents=True, exist_ok=True)
    (args.shared_root / "official_split_path.txt").write_text(str(args.split.resolve()) + "\n", encoding="utf-8")
    (args.shared_root / "reference_workbook_path.txt").write_text(str(args.reference_workbook.resolve()) + "\n", encoding="utf-8")
    if args.mode == "prepare":
        payload = prepare(args.shared_root, args.split, args.reference_workbook)
        print(json.dumps({"status": "PASS", "canonical_pool_sha256": payload["pool"]["canonical_pool_sha256"], "sweep_points": SWEEP_POINTS}, indent=2))
    elif args.mode == "preflight":
        prepare(args.shared_root, args.split, args.reference_workbook)
        labels = [args.model_label] if args.model_label else list(MODELS)
        report = {label: cache_failures(label, args.split) for label in labels}
        write_json(args.output_dir / "preflight.json", report)
        print(json.dumps(report, indent=2))
    elif args.mode == "sweep":
        if not args.model_label:
            parser.error("--model-label is required for sweep")
        run_sweep(args.model_label, args.split, args.reference_workbook, args.shared_root, args.output_dir, args.device, args.keep_artifacts)
    else:
        analyze(args.shared_root, args.output_dir)


if __name__ == "__main__":
    main()
