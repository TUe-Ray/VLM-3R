#!/usr/bin/env python3
"""Build scene-disjoint manifests for CUT3R gauge-translation training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DATASETS = ("scannet", "scannetpp", "arkitscenes")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def scene_group(dataset: str, stem: str) -> str:
    if dataset == "scannet":
        match = re.match(r"(scene\d{4})_\d{2}$", stem)
        if match:
            return f"{dataset}:{match.group(1)}"
    return f"{dataset}:{stem}"


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _string_values(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _string_values(child)


def load_exclusions(paths: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"benchmark exclusion source does not exist: {path}")
        if path.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("pandas with parquet support is required for VSI-Bench exclusions") from exc
            rows = pd.read_parquet(path).to_dict(orient="records")
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text())
            rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            for text in _string_values(row):
                candidate = Path(text).stem
                if candidate:
                    excluded.add(candidate)
                match = re.search(r"scene\d{4}_\d{2}", text)
                if match:
                    excluded.add(match.group(0))
    return excluded


def _find_video(video_dir: Path, stem: str) -> Path | None:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    matches = [path for path in video_dir.rglob(f"{stem}.*") if path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(matches)[0] if matches else None


def discover_records(
    datasets: Iterable[str],
    token_root: Path,
    final_token_root: Path,
    pointmap_root: Path,
    video_root: Path,
    context_root: Path,
    excluded: set[str],
    require_context: bool,
) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    reasons: Counter = Counter()
    for dataset in datasets:
        layer6_dir = token_root / dataset / "spatial_features_dec_6"
        layer9_dir = token_root / dataset / "spatial_features_dec_9"
        layer12_dir = final_token_root / dataset / "spatial_features"
        pointmap_dir = pointmap_root / dataset / "spatial_features_points"
        if not layer6_dir.is_dir():
            reasons[f"{dataset}:missing_layer6_dir"] += 1
            continue
        for layer6 in sorted(layer6_dir.glob("*.pt")):
            stem = layer6.stem
            if stem in excluded:
                reasons[f"{dataset}:benchmark_excluded"] += 1
                continue
            paths = {
                "layer6_path": layer6,
                "layer9_path": layer9_dir / layer6.name,
                "layer12_path": layer12_dir / layer6.name,
                "pointmap_path": pointmap_dir / layer6.name,
                "context_path": context_root / dataset / layer6.name,
            }
            missing = [key for key in ("layer9_path", "layer12_path", "pointmap_path") if not paths[key].is_file()]
            if require_context and not paths["context_path"].is_file():
                missing.append("context_path")
            if missing:
                reasons[f"{dataset}:missing_{'+'.join(missing)}"] += 1
                continue
            video = _find_video(video_root / dataset / "videos", stem)
            if video is None:
                reasons[f"{dataset}:missing_video"] += 1
                continue
            records.append(
                {
                    "id": f"{dataset}/{stem}",
                    "dataset": dataset,
                    "stem": stem,
                    "scene_group": scene_group(dataset, stem),
                    "video_path": str(video),
                    **{key: str(value) for key, value in paths.items()},
                }
            )
    return records, reasons


def split_records(records: Sequence[dict], seed: int, validation_fraction: float) -> tuple[list[dict], list[dict]]:
    validation_groups = {
        record["scene_group"]
        for record in records
        if _stable_fraction(record["scene_group"], seed) < validation_fraction
    }
    train = [record for record in records if record["scene_group"] not in validation_groups]
    validation = [record for record in records if record["scene_group"] in validation_groups]
    assert {item["scene_group"] for item in train}.isdisjoint(
        {item["scene_group"] for item in validation}
    )
    return train, validation


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--final-token-root", type=Path, required=True)
    parser.add_argument("--pointmap-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--require-context", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "official", "source"), default="source")
    parser.add_argument("--smoke-train-videos", type=int, default=16)
    parser.add_argument("--smoke-validation-videos", type=int, default=8)
    args = parser.parse_args()

    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    excluded = load_exclusions(args.exclude)
    records, reasons = discover_records(
        datasets,
        args.token_root,
        args.final_token_root,
        args.pointmap_root,
        args.video_root,
        args.context_root,
        excluded,
        args.require_context,
    )
    train, validation = split_records(records, args.seed, args.validation_fraction)
    if args.mode == "smoke":
        requested_train = int(args.smoke_train_videos)
        requested_validation = int(args.smoke_validation_videos)
        if requested_train < 16 or requested_validation < 8:
            raise ValueError("smoke counts cannot be below the required 16 train and 8 validation videos")
        if len(train) < requested_train or len(validation) < requested_validation:
            raise RuntimeError(
                f"smoke requests {requested_train} train/{requested_validation} validation videos, "
                f"found {len(train)}/{len(validation)}"
            )
        train, validation = train[:requested_train], validation[:requested_validation]
    deviation = None
    if args.mode == "official" and (len(train) < 1000 or len(validation) < 200):
        deviation = {
            "target_train": 1000,
            "target_validation": 200,
            "available_train": len(train),
            "available_validation": len(validation),
            "exclusions_preserved": True,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "all.jsonl", [*train, *validation])
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "datasets": datasets,
        "require_context": args.require_context,
        "excluded_stem_count": len(excluded),
        "train_videos": len(train),
        "validation_videos": len(validation),
        "train_scenes": len({item["scene_group"] for item in train}),
        "validation_scenes": len({item["scene_group"] for item in validation}),
        "dataset_train_counts": dict(Counter(item["dataset"] for item in train)),
        "dataset_validation_counts": dict(Counter(item["dataset"] for item in validation)),
        "rejections": dict(reasons),
        "official_target_deviation": deviation,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
